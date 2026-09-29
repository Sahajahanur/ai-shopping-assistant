"""
Tool Call Accuracy Evaluation Suite.

Evaluates whether the agent calls the correct tools with the expected parameters
for a representative test set of user queries.
"""

import json
import sys
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage

load_dotenv()

from shopping_agent import (
    llm,
    ALL_TOOLS,
    build_system_prompt,
    clear_user_preferences,
    get_user_preferences,
    set_user_preference
)
from langchain_core.messages import HumanMessage


def get_agent_tool_calls(user_query: str) -> list[dict]:
    """Invoke the tool-bound model on the query and extract the tool call decisions with rate-limit auto-wait."""
    import re
    import time
    system_msg = SystemMessage(content=build_system_prompt())
    model_with_tools = llm.bind_tools(ALL_TOOLS)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model_with_tools.invoke([
                system_msg,
                HumanMessage(content=user_query)
            ])
            tool_calls = []
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tc in response.tool_calls:
                    tool_calls.append({
                        "name": tc.get("name"),
                        "args": tc.get("args", {})
                    })
            return tool_calls
        except Exception as e:
            err_str = str(e)
            if "429" in err_str and attempt < max_retries - 1:
                match = re.search(r"try again in ([\d\.]+)s", err_str)
                wait_sec = float(match.group(1)) + 2 if match else 25
                print(f"  ⏳ Groq token limit hit. Auto-waiting {wait_sec:.0f}s for quota release (Attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_sec)
            else:
                raise e


def validate_arg(actual_val, expected_spec) -> bool:
    """Validate an actual argument value against an expected spec (value or predicate callable)."""
    if callable(expected_spec):
        return bool(expected_spec(actual_val))
    if isinstance(expected_spec, float):
        try:
            return abs(float(actual_val) - expected_spec) < 1e-4
        except (ValueError, TypeError):
            return False
    if isinstance(expected_spec, int):
        try:
            return int(actual_val) == expected_spec
        except (ValueError, TypeError):
            return False
    if isinstance(expected_spec, bool):
        return bool(actual_val) == expected_spec
    return str(actual_val).strip().lower() == str(expected_spec).strip().lower()


TEST_DATASET = [
    {
        "id": "TC-1",
        "description": "Organic honey under $20 filter",
        "query": "I want organic honey under $20",
        "expected_tool": "search_products",
        "expected_args": {
            "query": lambda q: "honey" in str(q).lower(),
            "is_organic": True,
            "max_price": 20.0
        }
    },
    {
        "id": "TC-2",
        "description": "Basic keyword search for oats",
        "query": "Show me oats",
        "expected_tool": "search_products",
        "expected_args": {
            "query": lambda q: "oats" in str(q).lower()
        }
    },
    {
        "id": "TC-3",
        "description": "Product rating inquiry for Product ID 7",
        "query": "What is the rating for product 7?",
        "expected_tool": "get_rating",
        "expected_args": {
            "product_id": 7
        }
    },
    {
        "id": "TC-4",
        "description": "Order history query",
        "query": "What have I ordered before?",
        "expected_tool": "get_order_history",
        "expected_args": {}
    },
    {
        "id": "TC-5",
        "description": "Save preference: always prefer organic",
        "query": "Please remember that I always prefer organic",
        "expected_tool": "set_saved_preference",
        "expected_args": {
            "key": lambda k: "organic" in str(k).lower(),
            "value": lambda v: str(v).lower() in ("true", "yes", "1", "organic", "always")
        }
    },
    {
        "id": "TC-6",
        "description": "Save preference: never want items over $20",
        "query": "Remember that I never want items over $20",
        "expected_tool": "set_saved_preference",
        "expected_args": {
            "key": lambda k: "price" in str(k).lower() or "budget" in str(k).lower(),
            "value": lambda v: "20" in str(v)
        }
    }
]


def run_tool_call_eval():
    print("=" * 70)
    print("🧪 RUNNING TOOL CALL ACCURACY EVALUATION")
    print("=" * 70)

    # Backup existing preferences so evaluation does not wipe out active user session
    backup_prefs = get_user_preferences()
    clear_user_preferences()

    try:
        _execute_eval(backup_prefs)
    finally:
        clear_user_preferences()
        for k, v in backup_prefs.items():
            set_user_preference(k, str(v))


def _execute_eval(backup_prefs):

    results = []
    passed_count = 0

    import time

    for tc in TEST_DATASET:
        test_id = tc["id"]
        query = tc["query"]
        expected_tool = tc["expected_tool"]
        expected_args = tc["expected_args"]

        time.sleep(2)  # Pause to respect Groq per-minute token quota
        print(f"\n▶ [{test_id}] Query: \"{query}\"")
        try:
            actual_calls = get_agent_tool_calls(query)
        except Exception as e:
            print(f"  ❌ Agent Execution Error: {e}")
            results.append({
                "id": test_id,
                "query": query,
                "passed": False,
                "reason": f"Execution error: {e}"
            })
            continue

        print(f"  Observed tool calls ({len(actual_calls)}):")
        for call in actual_calls:
            print(f"    - {call['name']}({call['args']})")

        # Find matching tool call
        matching_call = None
        for call in actual_calls:
            if call["name"] == expected_tool:
                matching_call = call
                break

        if not matching_call:
            print(f"  ❌ FAIL: Expected tool '{expected_tool}' was NOT called.")
            results.append({
                "id": test_id,
                "query": query,
                "expected_tool": expected_tool,
                "passed": False,
                "reason": f"Expected tool '{expected_tool}' was not called"
            })
            continue

        # Validate arguments
        args_passed = True
        failed_args = []
        actual_args = matching_call["args"]

        for arg_name, expected_val in expected_args.items():
            if arg_name not in actual_args:
                args_passed = False
                failed_args.append(f"Missing argument '{arg_name}'")
            else:
                act_val = actual_args[arg_name]
                if not validate_arg(act_val, expected_val):
                    args_passed = False
                    failed_args.append(
                        f"Arg '{arg_name}': actual '{act_val}' does not match expected spec"
                    )

        if args_passed:
            print(f"  ✅ PASS: Called '{expected_tool}' with correct parameters.")
            passed_count += 1
            results.append({
                "id": test_id,
                "query": query,
                "expected_tool": expected_tool,
                "actual_args": actual_args,
                "passed": True
            })
        else:
            print(f"  ❌ FAIL: Parameter mismatch: {', '.join(failed_args)}")
            results.append({
                "id": test_id,
                "query": query,
                "expected_tool": expected_tool,
                "actual_args": actual_args,
                "passed": False,
                "reason": ", ".join(failed_args)
            })

    total = len(TEST_DATASET)
    accuracy = (passed_count / total) * 100

    print("\n" + "=" * 70)
    print("📊 EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Test Cases: {total}")
    print(f"Passed:           {passed_count}")
    print(f"Failed:           {total - passed_count}")
    print(f"Accuracy:         {accuracy:.1f}%")
    print("=" * 70)

    # Save results to JSON
    with open("eval_tool_calls_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "total": total,
            "passed": passed_count,
            "accuracy_percent": accuracy,
            "results": results
        }, f, indent=2)

    print("📁 Results saved to eval_tool_calls_results.json")

    # Assert test suite passed
    assert passed_count == total, f"Only {passed_count}/{total} tests passed."
    print("🎉 All tool call accuracy tests passed successfully!")


if __name__ == "__main__":
    try:
        run_tool_call_eval()
    except AssertionError as ae:
        print(f"\n❌ Assertion Failed: {ae}")
        sys.exit(1)
