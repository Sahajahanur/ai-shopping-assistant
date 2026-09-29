"""
Input Guardrail for AI Shopping Assistant.

Verifies that incoming user queries are shopping-related before invoking the agent.
Rejects off-topic queries (e.g., poetry, weather, general trivia, coding) with a polite redirect.
"""

import json
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# Fast lightweight guardrail model
guardrail_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

POLITE_REDIRECT_MESSAGE = (
    "I am your AI Shopping Assistant for this store. "
    "I can only help you with discovering products, checking ratings, managing orders, "
    "and setting your shopping preferences. "
    "How can I help you with your shopping today?"
)

GUARDRAIL_SYSTEM_PROMPT = """You are a guardrail classifier for an AI Shopping Assistant of an online grocery and food store.
Your sole job is to evaluate if the user's message is relevant to shopping at our store or completely off-topic.

CLASSIFY AS SHOPPING-RELATED (is_shopping_related: true):
- Searching for products, groceries, food, drinks, ingredients, or items in store (e.g. "organic honey under $20", "find milk", "show tea")
- Inquiring about prices, discounts, ratings, reviews, stock, or product recommendations
- Placing orders, checkout, shopping confirmations or follow-ups ("yes", "order it", "confirm", "buy this", "no", "first one")
- Asking about past purchases or order history ("what have I ordered before?", "show my previous orders")
- Setting, viewing, or updating shopping preferences ("remember that I prefer organic", "my budget is under $20")
- Image search queries ("I uploaded a product image...")
- Greetings or conversational pleasantries in a shopping context ("hello", "hi", "help me shop")

CLASSIFY AS OFF-TOPIC (is_shopping_related: false):
- Creative writing, poems, stories, jokes, essays, songs ("write me a poem", "tell me a joke")
- Weather forecasts, news, sports scores ("what is the weather today", "who won the game")
- Coding, math puzzles, homework, non-store questions ("write a python function", "solve 2+2", "who was Napoleon")
- General chit-chat or philosophy having nothing to do with shopping or the store

Return JSON ONLY:
{
    "is_shopping_related": true or false,
    "reason": "short explanation"
}
"""


def check_input_guardrail(user_message: str) -> tuple[bool, str]:
    """
    Check if the user's message is shopping-related.

    Returns:
        (is_allowed, response_message)
        If allowed: (True, "")
        If rejected: (False, POLITE_REDIRECT_MESSAGE)
    """
    text = (user_message or "").strip()
    if not text:
        return True, ""

    # Instant passes: image uploads, short confirmations
    lower_text = text.lower()
    if lower_text.startswith("i uploaded a product image"):
        return True, ""
    if lower_text in {"yes", "no", "y", "n", "order it", "confirm", "ok", "cancel"}:
        return True, ""

    # Call LLM Guardrail
    try:
        messages = [
            SystemMessage(content=GUARDRAIL_SYSTEM_PROMPT),
            HumanMessage(content=f"User Message: {text}")
        ]
        response = guardrail_llm.invoke(messages)
        content = response.content.strip()

        # Clean JSON markdown if any
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?", "", content)
            content = re.sub(r"```$", "", content).strip()

        data = json.loads(content)
        is_shopping = bool(data.get("is_shopping_related", True))

        if is_shopping:
            return True, ""
        else:
            return False, POLITE_REDIRECT_MESSAGE

    except Exception:
        # Fail-open if classification fails to avoid breaking legitimate user flows
        return True, ""


if __name__ == "__main__":
    test_queries = [
        ("write me a poem", False),
        ("what is the weather in New York?", False),
        ("write a python script to sort numbers", False),
        ("organic honey under $20", True),
        ("what have I ordered before?", True),
        ("remember that I always prefer organic", True),
        ("order it", True),
        ("yes", True),
        ("do you have oat milk?", True),
    ]

    print("Running Input Guardrail verification tests...")
    all_passed = True
    for query, expected in test_queries:
        allowed, msg = check_input_guardrail(query)
        passed = (allowed == expected)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] Query: '{query}' -> Allowed: {allowed} (Expected: {expected})")

    print("\nGuardrail test status:", "ALL PASSED" if all_passed else "SOME FAILED")
