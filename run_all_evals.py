"""
Master Evaluation & Test Runner for AI Shopping Assistant.

Runs all guardrail and evaluation suites:
1. Input Guardrail Verification (guardrails.py)
2. Tool Call Accuracy Evaluation (eval_tool_calls.py)
3. Response Quality Evaluation with LLM-as-a-Judge (eval_response_quality.py)
"""

import sys
import time

from guardrails import check_input_guardrail
from eval_tool_calls import run_tool_call_eval
from eval_response_quality import evaluate_response_quality


def main():
    print("=" * 80)
    print("🚀 STARTING COMPLETE EVALUATION SUITE FOR AI SHOPPING ASSISTANT")
    print("=" * 80)
    start_time = time.time()

    # 1. Guardrail Verification
    print("\n[PHASE 1/3] Testing Input Guardrail...")
    guardrail_cases = [
        ("write me a poem", False),
        ("what is the weather in New York?", False),
        ("organic honey under $20", True),
        ("what have I ordered before?", True),
        ("remember that I always prefer organic", True),
    ]
    guardrail_passed = True
    for q, expected in guardrail_cases:
        allowed, _ = check_input_guardrail(q)
        if allowed != expected:
            guardrail_passed = False
            print(f"  ❌ Failed for '{q}': allowed={allowed}, expected={expected}")
        else:
            print(f"  ✅ Passed for '{q}'")

    if not guardrail_passed:
        print("❌ Input guardrail evaluation failed.")
        sys.exit(1)
    print("✅ All Input Guardrail tests PASSED!")

    # 2. Tool Call Accuracy
    print("\n[PHASE 2/3] Running Tool Call Accuracy Evaluation...")
    try:
        run_tool_call_eval()
    except Exception as e:
        print(f"❌ Tool call evaluation failed: {e}")
        sys.exit(1)

    # 3. Response Quality (LLM-as-a-Judge)
    print("\n[PHASE 3/3] Running Response Quality Evaluation (LLM-as-a-Judge)...")
    try:
        evaluate_response_quality()
    except Exception as e:
        print(f"❌ Response quality evaluation failed: {e}")
        sys.exit(1)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"🎉 ALL EVALUATION SUITES PASSED SUCCESSFULLY IN {elapsed:.1f}s!")
    print("=" * 80)


if __name__ == "__main__":
    main()
