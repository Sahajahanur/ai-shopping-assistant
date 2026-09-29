"""
Response Quality Evaluation Suite using LLM-as-a-Judge.

Scores agent responses on three core dimensions:
1. Relevance: Did it directly address the user's intent?
2. Correctness: Did it show the right products matching the criteria from the database?
3. Format Compliance: Did it use the required numbered list format (1., 2., 3., etc.)?
"""

import json
import re
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

from shopping_agent import run_agent, clear_user_preferences, get_user_preferences, set_user_preference

# LLM Judge instance
judge_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

JUDGE_SYSTEM_PROMPT = """You are an impartial, expert evaluation judge for an AI Shopping Assistant of an online grocery store.
Evaluate the AI Assistant's response to a user query based on the following three dimensions on a 1-5 scale:

1. RELEVANCE (Score 1-5):
- 5: Directly answers the user's request without irrelevant tangents.
- 4: Strong answer addressing the core shopping intent.
- 3: Partially answers the request or includes noticeable tangents.
- 1-2: Fails to address the core user request.

2. CORRECTNESS (Score 1-5):
- 5: Products, prices, ratings, and filters (such as organic status or budget limits) accurately reflect the store database and query constraints. Ratings shown by the assistant come from the store's reviews database. No hallucinated fake products.
- 4: Accurate recommendations with minor phrasing variations.
- 3: Mostly correct but has minor discrepancies in filtering or pricing.
- 1-2: Hallucinates fake products, wrong prices, or completely misses valid items.

3. FORMAT COMPLIANCE (Score 1-5):
- 5: When presenting products or past order lists, STRICTLY uses a clear numbered list format (1., 2., 3., etc.) with required details (Product ID, Name, Price, Rating, Organic status where applicable).
- 4: Uses numbered list with slight styling variations.
- 3: Uses a bulleted list or a table instead of a clear numbered list.
- 1-2: Outputs a plain wall of unstructured text or fails to list items.

GROUND TRUTH STORE CATALOG REFERENCE (with authentic average ratings & timestamps from store/orders database):
- Honey:
  * ID 1: Organic Raw Honey ($14.99, Organic: Yes, Avg Rating: 4.62 ★, 4 reviews)
  * ID 2: Wildflower Honey ($12.99, Organic: No, Avg Rating: 3.83 ★, 3 reviews)
  * ID 3: Organic Manuka Honey ($29.99, Organic: Yes, Avg Rating: 4.83 ★, 3 reviews)
  * ID 4: Clover Honey ($8.99, Organic: No, Avg Rating: 3.50 ★, 3 reviews)
  * ID 5: Organic Buckwheat Honey ($18.99, Organic: Yes, Avg Rating: 4.62 ★, 4 reviews)
  * ID 6: Orange Blossom Honey ($15.99, Organic: No, Avg Rating: 4.17 ★, 3 reviews)
  * ID 7: Organic Acacia Honey ($17.99, Organic: Yes, Avg Rating: 4.75 ★, 4 reviews)
  * ID 8: Creamed Honey ($11.99, Organic: No, Avg Rating: 4.00 ★, 3 reviews)
- Oats:
  * ID 18: Rolled Oats ($5.49, Organic: No, Avg Rating: 4.33 ★, 3 reviews)
  * ID 20: Steel-Cut Oats ($6.99, Organic: No, Avg Rating: 3.83 ★, 3 reviews)
  * ID 30: Oat Milk ($4.49, Organic: No, Avg Rating: 4.33 ★, 3 reviews)
- Milks (Plant-based / Non-dairy alternatives):
  * ID 29: Organic Almond Milk ($4.99, Organic: Yes, Avg Rating: 4.50 ★, 3 reviews)
  * ID 30: Oat Milk ($4.49, Organic: No, Avg Rating: 4.33 ★, 3 reviews)
  * ID 31: Organic Coconut Milk ($3.99, Organic: Yes, Avg Rating: 4.50 ★, 3 reviews)
  * ID 32: Soy Milk ($3.49, Organic: No, Avg Rating: 3.67 ★, 3 reviews)
- Past Orders in Database:
  * Order ID 2: Rolled Oats ($5.49, Product ID 18, Ordered on: 2026-09-29 16:39:07)
  * Order ID 1: Organic Acacia Honey ($17.99, Product ID 7, Ordered on: 2026-09-29 11:33:49)

Return ONLY valid JSON matching this schema:
{
    "relevance": {
        "score": 1-5,
        "explanation": "..."
    },
    "correctness": {
        "score": 1-5,
        "explanation": "..."
    },
    "format_compliance": {
        "score": 1-5,
        "explanation": "..."
    },
    "passed": true/false (true if relevance >= 4, correctness >= 4, format_compliance >= 4),
    "summary": "Concise judge summary"
}
"""

TEST_CASES = [
    {
        "id": "RQ-1",
        "description": "Organic honey under $20 with rating requirement",
        "query": "I want to buy organic honey with 4.5+ rating and under $20.",
        "expectations": "Should list organic honeys under $20 with rating >= 4.5 (IDs 1, 5, 7) as a numbered list."
    },
    {
        "id": "RQ-2",
        "description": "Search for oat products",
        "query": "Show me oat products in the store.",
        "expectations": "Should list oat items from the store (such as Rolled Oats, Steel-Cut Oats, and/or Oat Milk) as a numbered list."
    },
    {
        "id": "RQ-3",
        "description": "Order history query",
        "query": "What have I ordered before?",
        "expectations": "Should summarize previous orders (Order 2: Rolled Oats, Order 1: Organic Acacia Honey with prices and dates) as a numbered list."
    },
    {
        "id": "RQ-4",
        "description": "Milk alternatives under $5",
        "query": "Recommend non-dairy milk options under $5.",
        "expectations": "Should recommend plant-based milk options under $5 (Almond Milk, Oat Milk, Coconut Milk, and/or Soy Milk) as a numbered list."
    }
]


def evaluate_response_quality():
    print("=" * 75)
    print("⚖️  RUNNING RESPONSE QUALITY EVALUATION (LLM-AS-A-JUDGE)")
    print("=" * 75)

    backup_prefs = get_user_preferences()
    clear_user_preferences()

    try:
        _execute_response_eval(backup_prefs)
    finally:
        clear_user_preferences()
        for k, v in backup_prefs.items():
            set_user_preference(k, str(v))


def _execute_response_eval(backup_prefs):
    eval_results = []
    total_relevance = 0
    total_correctness = 0
    total_format = 0
    all_passed = True

    for tc in TEST_CASES:
        test_id = tc["id"]
        query = tc["query"]
        expectations = tc["expectations"]

        print(f"\n▶ [{test_id}] Query: \"{query}\"")
        try:
            agent_response = run_agent([{"role": "user", "content": query}])
        except Exception as e:
            print(f"  ❌ Agent Execution Error: {e}")
            all_passed = False
            continue

        print(f"  Agent Response:\n{'-'*40}\n{agent_response}\n{'-'*40}")

        # Invoke LLM-as-a-judge
        judge_prompt = f"""
USER QUERY:
"{query}"

EXPECTATIONS:
{expectations}

AI ASSISTANT RESPONSE:
\"\"\"
{agent_response}
\"\"\"

Score this response on Relevance, Correctness, and Format Compliance (numbered list).
"""

        try:
            judge_res = judge_llm.invoke([
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=judge_prompt)
            ])
            raw_content = judge_res.content.strip()

            if raw_content.startswith("```"):
                raw_content = re.sub(r"^```(?:json)?", "", raw_content)
                raw_content = re.sub(r"```$", "", raw_content).strip()

            scores = json.loads(raw_content)

        except Exception as e:
            print(f"  ❌ Judge Evaluation Error: {e}")
            all_passed = False
            continue

        rel_score = scores.get("relevance", {}).get("score", 0)
        cor_score = scores.get("correctness", {}).get("score", 0)
        fmt_score = scores.get("format_compliance", {}).get("score", 0)
        is_passed = bool(rel_score >= 4 and cor_score >= 4 and fmt_score >= 4)

        total_relevance += rel_score
        total_correctness += cor_score
        total_format += fmt_score

        if not is_passed:
            all_passed = False

        status_str = "✅ PASS" if is_passed else "❌ FAIL"
        print(f"  Judge Verdict: {status_str}")
        print(f"    - Relevance:         {rel_score}/5 ({scores.get('relevance', {}).get('explanation', '')})")
        print(f"    - Correctness:       {cor_score}/5 ({scores.get('correctness', {}).get('explanation', '')})")
        print(f"    - Format Compliance: {fmt_score}/5 ({scores.get('format_compliance', {}).get('explanation', '')})")
        print(f"    - Summary:           {scores.get('summary', '')}")

        eval_results.append({
            "id": test_id,
            "query": query,
            "agent_response": agent_response,
            "scores": {
                "relevance": rel_score,
                "correctness": cor_score,
                "format_compliance": fmt_score
            },
            "passed": is_passed,
            "details": scores
        })

    num_cases = len(TEST_CASES)
    avg_relevance = total_relevance / num_cases if num_cases else 0
    avg_correctness = total_correctness / num_cases if num_cases else 0
    avg_format = total_format / num_cases if num_cases else 0
    overall_avg = (avg_relevance + avg_correctness + avg_format) / 3

    print("\n" + "=" * 75)
    print("📊 RESPONSE QUALITY EVALUATION SUMMARY")
    print("=" * 75)
    print(f"Total Evaluated Queries:    {num_cases}")
    print(f"Average Relevance:          {avg_relevance:.2f} / 5.0")
    print(f"Average Correctness:        {avg_correctness:.2f} / 5.0")
    print(f"Average Format Compliance:  {avg_format:.2f} / 5.0")
    print(f"Overall Quality Score:      {overall_avg:.2f} / 5.0 ({(overall_avg/5.0)*100:.1f}%)")
    print("=" * 75)

    with open("eval_response_quality_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "num_test_cases": num_cases,
            "average_relevance": avg_relevance,
            "average_correctness": avg_correctness,
            "average_format_compliance": avg_format,
            "overall_average": overall_avg,
            "all_passed": all_passed,
            "results": eval_results
        }, f, indent=2)

    print("📁 Results saved to eval_response_quality_results.json")

    assert all_passed, "Some test cases did not pass the LLM Judge rubric."
    print("🎉 All responses met the quality criteria under LLM-as-a-Judge!")


if __name__ == "__main__":
    try:
        evaluate_response_quality()
    except AssertionError as ae:
        print(f"\n❌ Assertion Failed: {ae}")
        sys.exit(1)
