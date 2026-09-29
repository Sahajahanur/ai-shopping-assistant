import os
import json
import base64
import sqlite3

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

import reviews_api
from guardrails import check_input_guardrail, POLITE_REDIRECT_MESSAGE


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# DATABASE PATHS
# ============================================================

BASE_DIR = os.path.dirname(__file__)

DB_PATH = os.path.join(BASE_DIR, "store.db")
ORDERS_DB_PATH = os.path.join(BASE_DIR, "orders.db")


# ============================================================
# MODELS
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=250
)

vision_llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0
)


# ============================================================
# DATABASE HELPERS
# ============================================================

def get_db_connection():
    """Products (store.db) connection."""
    conn = sqlite3.connect(
        DB_PATH,
        timeout=30
    )
    conn.execute("PRAGMA busy_timeout = 30000")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:
        pass
    return conn


def get_orders_connection():
    """Orders and preferences (orders.db) connection."""
    conn = sqlite3.connect(
        ORDERS_DB_PATH,
        timeout=30
    )
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            product_name TEXT,
            price REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_preferences (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    return conn


def get_user_preferences() -> dict:
    """Read all persistent user preferences from orders.db."""
    conn = get_orders_connection()
    try:
        cursor = conn.cursor()
        rows = cursor.execute("SELECT key, value FROM user_preferences").fetchall()
        prefs = {}
        for k, v in rows:
            clean_val = v
            if str(v).lower() in ("true", "yes", "1"):
                clean_val = True
            elif str(v).lower() in ("false", "no", "0"):
                clean_val = False
            else:
                try:
                    clean_val = float(v)
                except (ValueError, TypeError):
                    clean_val = v

            clean_k = str(k).strip().lower()
            if "organic" in clean_k:
                prefs["prefer_organic"] = clean_val
            elif "price" in clean_k or "budget" in clean_k or "limit" in clean_k:
                prefs["max_price"] = clean_val
            else:
                prefs[clean_k] = clean_val
        return prefs
    finally:
        conn.close()


def set_user_preference(key: str, value: str) -> None:
    """Save or update a single preference in orders.db."""
    clean_k = str(key).strip().lower()
    if "organic" in clean_k:
        clean_k = "prefer_organic"
    elif "price" in clean_k or "budget" in clean_k or "limit" in clean_k:
        clean_k = "max_price"

    clean_val = str(value).strip()
    if clean_k == "max_price":
        clean_val = clean_val.replace("$", "").strip()

    conn = get_orders_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO user_preferences (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP
            """,
            (clean_k, clean_val)
        )
        conn.commit()
    finally:
        conn.close()


def delete_user_preference(key: str) -> None:
    """Delete a single preference from orders.db."""
    conn = get_orders_connection()
    try:
        conn.execute("DELETE FROM user_preferences WHERE key = ?", (key.strip().lower(),))
        conn.commit()
    finally:
        conn.close()


def clear_user_preferences() -> None:
    """Clear all preferences from orders.db."""
    conn = get_orders_connection()
    try:
        conn.execute("DELETE FROM user_preferences")
        conn.commit()
    finally:
        conn.close()


# ============================================================
# TOOL 1 — SEARCH PRODUCTS
# ============================================================

class ProductSearchInput(BaseModel):
    query: str = Field(
        description="Product name or keyword to search for"
    )
    max_price: float | None = Field(
        default=None,
        description="Maximum allowed product price"
    )
    is_organic: bool | None = Field(
        default=None,
        description="Whether the product must be organic"
    )


@tool(args_schema=ProductSearchInput)
def search_products(
    query: str,
    max_price: float | None = None,
    is_organic: bool | None = None
) -> str:
    """
    Search products in the store database.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        sql = """
            SELECT id, name, price, is_organic
            FROM products
            WHERE LOWER(name) LIKE LOWER(?)
        """
        params = [f"%{query}%"]

        if max_price is not None:
            sql += " AND price <= ?"
            params.append(max_price)

        if is_organic is not None:
            sql += " AND is_organic = ?"
            params.append(1 if is_organic else 0)

        sql += " ORDER BY price ASC"
        cursor.execute(sql, params)
        rows = cursor.fetchall()

        products = []
        for row in rows:
            products.append({
                "id": row[0],
                "name": row[1],
                "price": row[2],
                "is_organic": bool(row[3])
            })

        return json.dumps(products)
    finally:
        conn.close()


# ============================================================
# TOOL 2 — GET RATING
# ============================================================

class RatingInput(BaseModel):
    product_id: int = Field(
        description="The integer product ID returned by search_products"
    )


@tool(args_schema=RatingInput)
def get_rating(product_id: int) -> dict:
    """
    Get average rating and review count for a product.
    """
    return reviews_api.get_product_rating(product_id)


# ============================================================
# TOOL 3 — CHECKOUT
# ============================================================

class CheckoutInput(BaseModel):
    product_id: int = Field(
        description="The integer product ID of the product to order"
    )


@tool(args_schema=CheckoutInput)
def checkout(product_id: int) -> dict:
    """
    Place an order after explicit user confirmation.
    """
    store_conn = None
    try:
        store_conn = get_db_connection()
        row = store_conn.execute(
            """
            SELECT id, name, price
            FROM products
            WHERE id = ?
            """,
            (product_id,)
        ).fetchone()
    except sqlite3.Error as e:
        return {
            "success": False,
            "message": f"SQLite Error (read): {e}"
        }
    finally:
        if store_conn is not None:
            store_conn.close()

    if not row:
        return {
            "success": False,
            "message": f"Product ID {product_id} was not found."
        }

    product_id_db, product_name, price = row

    orders_conn = None
    try:
        orders_conn = get_orders_connection()
        cursor = orders_conn.execute(
            """
            INSERT INTO orders
            (product_id, product_name, price)
            VALUES (?, ?, ?)
            """,
            (product_id_db, product_name, price)
        )
        orders_conn.commit()

        return {
            "success": True,
            "order_id": cursor.lastrowid,
            "product_id": product_id_db,
            "product_name": product_name,
            "price": price,
            "message": f"Order placed successfully for {product_name}."
        }
    except sqlite3.Error as e:
        if orders_conn is not None:
            orders_conn.rollback()
        return {
            "success": False,
            "message": f"SQLite Error (write): {e}"
        }
    finally:
        if orders_conn is not None:
            orders_conn.close()


# ============================================================
# TOOL 4 — PRODUCT IMAGE ANALYSIS
# ============================================================

class ImageInput(BaseModel):
    image_path: str = Field(
        description="Local path of the uploaded product image"
    )


@tool(args_schema=ImageInput)
def describe_product_image(image_path: str) -> str:
    """
    Analyze a product image and extract information useful for product search.
    """
    if not os.path.exists(image_path):
        return json.dumps({
            "error": f"Image file not found: {image_path}"
        })

    extension = os.path.splitext(image_path)[1].lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp"
    }
    mime_type = mime_types.get(extension, "image/jpeg")

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_data}"
                }
            },
            {
                "type": "text",
                "text": """
Analyze this product image.

Return ONLY valid JSON:

{
    "product_type": "...",
    "search_query": "...",
    "is_organic": true,
    "description": "..."
}

Rules:

- product_type = type of product visible
- search_query = useful keyword(s) for database search
- is_organic = true only when clearly indicated
- description = short description
"""
            }
        ]
    )

    response = vision_llm.invoke([message])
    content = response.content

    try:
        cleaned = (
            content
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )
        data = json.loads(cleaned)
        return json.dumps(data)
    except Exception:
        return json.dumps({
            "product_type": "unknown",
            "search_query": content,
            "is_organic": None,
            "description": content
        })


# ============================================================
# TOOL 5 — ORDER HISTORY (MEMORY)
# ============================================================

class OrderHistoryInput(BaseModel):
    limit: int = Field(
        default=10,
        description="Number of recent orders to retrieve (default is 10)"
    )


@tool(args_schema=OrderHistoryInput)
def get_order_history(limit: int = 10) -> str:
    """
    Retrieve the user's past order history from the orders database.
    Call this when the user asks 'what have I ordered before?', 'show my previous orders',
    'my past orders', or asks about previous purchases.
    """
    conn = get_orders_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, product_id, product_name, price, created_at
            FROM orders
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        )
        rows = cursor.fetchall()
        if not rows:
            return json.dumps({
                "status": "empty",
                "message": "No previous orders found."
            })

        orders = []
        for row in rows:
            orders.append({
                "order_id": row[0],
                "product_id": row[1],
                "product_name": row[2],
                "price": row[3],
                "ordered_at": row[4]
            })

        return json.dumps(orders)
    finally:
        conn.close()


# ============================================================
# TOOL 6 — GET SAVED PREFERENCES (PERSONALIZATION)
# ============================================================

@tool
def get_saved_preferences() -> str:
    """
    Retrieve the user's saved shopping preferences (e.g., preference for organic products, maximum price budget, etc.).
    """
    prefs = get_user_preferences()
    if not prefs:
        return json.dumps({
            "message": "No saved preferences found. Using default settings."
        })
    return json.dumps(prefs)


# ============================================================
# TOOL 7 — SET SAVED PREFERENCE (PERSONALIZATION)
# ============================================================

class SetPreferenceInput(BaseModel):
    max_price: float | None = Field(
        default=None,
        description="The maximum dollar amount/budget limit if the user mentions price or budget (e.g. 15.0 for '$15')"
    )
    prefer_organic: bool | None = Field(
        default=None,
        description="True if user prefers organic, False if non-organic"
    )
    key: str | None = Field(
        default=None,
        description="Optional preference key, e.g. 'max_price' or 'prefer_organic'"
    )
    value: str | None = Field(
        default=None,
        description="Optional preference value"
    )


@tool(args_schema=SetPreferenceInput)
def set_saved_preference(
    max_price: float | None = None,
    prefer_organic: bool | None = None,
    key: str | None = None,
    value: str | None = None
) -> str:
    """
    Save or update a persistent user preference across sessions (e.g. max budget limit or organic preference).
    Call this when the user says 'remember that I never want items over $15', 'I always prefer organic', etc.
    """
    if max_price is not None:
        set_user_preference("max_price", str(max_price))
    if prefer_organic is not None:
        set_user_preference("prefer_organic", "true" if prefer_organic else "false")
    if key and value is not None:
        clean_key = str(key).strip().lower()
        clean_val = str(value).strip()
        if "organic" in clean_key:
            set_user_preference("prefer_organic", "true" if clean_val.lower() in ("true", "1", "yes", "always", "organic") else "false")
        elif "price" in clean_key or "budget" in clean_key or "over" in clean_key or "under" in clean_key:
            set_user_preference("max_price", clean_val.replace("$", "").strip())
        else:
            set_user_preference(clean_key, clean_val)

    current = get_user_preferences()
    return json.dumps({
        "status": "success",
        "message": f"Saved preference. Active preferences are now: {current}."
    })


# ============================================================
# SYSTEM PROMPT TEMPLATE (TOKEN-OPTIMIZED)
# ============================================================

SYSTEM_PROMPT_TEMPLATE = """You are an AI Shopping Assistant for an online store.
Tools available: search_products, get_rating, checkout, describe_product_image, get_order_history, get_saved_preferences, set_saved_preference.

User Preferences:
{user_preferences}

Rules:
1. Product Search:
   - When searching, use simple core nouns (e.g. 'honey', 'oats', 'milk').
   - Apply user preferences automatically: if prefer_organic is true, set is_organic=True; if max_price is set, set max_price.
   - Always call get_rating for matching items.
   - FORMAT REQUIREMENT: Always display matching products as a numbered list (1., 2., 3., etc.):
     1. Product Name (Product ID: <id>) - Price: $<price> | Rating: <avg_rating> ★ (<count> reviews) | Organic: <Yes/No>
2. Order History:
   - When asked what user ordered before, call get_order_history.
   - Present past orders as a clean numbered list (1., 2., etc.):
     1. Product Name (Product ID: <id>) - Price: $<price> - Ordered on: <date>
3. Preferences:
   - If the user asks to remember or set a budget/price limit (e.g. "I never want items over $15", "set max budget to $20"):
     IMMEDIATELY call set_saved_preference with key='max_price' and value matching the dollar amount mentioned (e.g. '15'). Do not ask for clarification; extract the number and call the tool immediately.
   - If the user asks to remember an organic preference (e.g. "I always prefer organic"):
     IMMEDIATELY call set_saved_preference with key='prefer_organic' and value='true'.
   - When asked what preferences are saved, call get_saved_preferences.
4. Checkout:
   - Always confirm exact product name, ID, and price first. Call checkout only after user confirms ("yes").
"""


def build_system_prompt() -> str:
    """Build system prompt dynamically injecting current active preferences."""
    prefs = get_user_preferences()
    if prefs:
        lines = []
        for k, v in prefs.items():
            if k == "prefer_organic":
                status = "Active (Always prefer organic)" if (v is True or str(v).lower() == "true") else "Disabled"
                lines.append(f"- prefer_organic: {status}")
            elif k == "max_price":
                lines.append(f"- max_price: ${v} (Never want items over ${v})")
            else:
                lines.append(f"- {k}: {v}")
        prefs_text = "\n".join(lines)
    else:
        prefs_text = "No preferences saved yet."

    return SYSTEM_PROMPT_TEMPLATE.format(user_preferences=prefs_text)


# ============================================================
# CREATE AGENT
# ============================================================

ALL_TOOLS = [
    search_products,
    get_rating,
    checkout,
    describe_product_image,
    get_order_history,
    get_saved_preferences,
    set_saved_preference
]

agent = create_agent(
    model=llm,
    tools=ALL_TOOLS,
    system_prompt=build_system_prompt()
)


# ============================================================
# RESPONSE HELPERS
# ============================================================

def _content_to_text(content) -> str:
    """Convert AI message content into plain text."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts).strip()

    return ""


def extract_reply(result: dict) -> str:
    """
    Extract the final useful response from the agent.
    If the final AI message is empty, inspect the tool result.
    """
    messages = result.get("messages", [])

    # 1. Last non-empty AI response
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai":
            text = _content_to_text(message.content)
            if text:
                return text

    # 2. Inspect tool result
    for message in reversed(messages):
        if getattr(message, "type", None) != "tool":
            continue

        content = message.content
        try:
            if isinstance(content, str):
                data = json.loads(content)
            else:
                data = content
        except Exception:
            return _content_to_text(content) or "Done."

        if isinstance(data, dict) and "message" in data:
            if data.get("success"):
                return (
                    f"✅ {data['message']}\n\n"
                    f"Order ID: {data.get('order_id', 'N/A')} | "
                    f"Price: ${data.get('price', 'N/A')}"
                )
            return f"❌ {data['message']}"

        return "Done."

    return "Sorry, I couldn't generate a response. Please try again."


def run_agent(messages: list[dict], enable_guardrail: bool = True) -> str:
    """
    Run the shopping agent using the complete conversation history.
    Executes input guardrail first to reject off-topic queries.
    Injects dynamic user preferences into system prompt.
    """
    if not messages:
        return "Hello! How can I help you with your shopping today?"

    latest_user_text = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            latest_user_text = msg.get("content", "")
            break

    # 1. Input Guardrail Check
    if enable_guardrail and latest_user_text:
        is_allowed, redirect_msg = check_input_guardrail(latest_user_text)
        if not is_allowed:
            return redirect_msg

    # 2. Dynamic system prompt with latest stored user preferences
    active_prompt = build_system_prompt()
    agent_messages = [SystemMessage(content=active_prompt)]
    for m in messages:
        agent_messages.append(m)

    result = agent.invoke({
        "messages": agent_messages
    })

    return extract_reply(result)


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":
    test_queries = [
        "What have I ordered before?",
        "I want to buy organic honey with 4.5+ rating and less than $20."
    ]

    for q in test_queries:
        print(f"\n--- Testing Query: '{q}' ---")
        response = run_agent([{"role": "user", "content": q}])
        print(response)