import os
import tempfile

import streamlit as st

from shopping_agent import (
    run_agent,
    get_user_preferences,
    set_user_preference,
    clear_user_preferences
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Shopping Assistant",
    page_icon="🛒",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🛒 AI Shopping Assistant")

st.caption(
    "Tell me what you want — I'll search, rate, "
    "and order the best match for you. "
    "Features: Order History Memory, Saved Preferences, and Input Guardrails."
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_image" not in st.session_state:
    st.session_state.pending_image = None


# ============================================================
# HELPER
# ============================================================

def get_reply() -> str:
    """
    Run the shopping agent. If an error occurs,
    display a clean message instead of crashing.
    """
    try:
        reply = run_agent(st.session_state.messages)
    except Exception as e:
        return f"❌ Something went wrong: {e}"

    return reply.replace("`", "")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # 1. User Preferences & Personalization
    # --------------------------------------------------------
    st.header("⚙️ User Preferences & Memory")
    st.caption("Preferences are remembered across all sessions.")

    current_prefs = get_user_preferences()

    organic_pref = current_prefs.get("prefer_organic", False)
    max_price_pref = current_prefs.get("max_price", None)

    with st.expander("Saved Preferences", expanded=True):
        has_any_pref = bool(organic_pref or (max_price_pref is not None and str(max_price_pref).strip() != ""))
        if has_any_pref:
            st.markdown(f"**Prefers Organic:** {'✅ Yes' if organic_pref else '❌ No'}")
            budget_str = f"${float(max_price_pref):g}" if (max_price_pref is not None and str(max_price_pref).strip() != "") else "Not set"
            st.markdown(f"**Max Budget:** {budget_str}")
        else:
            st.info("No preferences saved yet. You can set them here or tell the assistant in chat.")

        new_organic = st.checkbox("Always prefer organic", value=bool(organic_pref), key=f"chk_org_{bool(organic_pref)}")
        price_val = float(max_price_pref) if (max_price_pref is not None and str(max_price_pref).strip() != "") else 0.0
        new_max_price = st.number_input("Max price limit ($0 for none)", min_value=0.0, max_value=100.0, value=price_val, step=1.0, key=f"num_price_{price_val}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Preferences", use_container_width=True):
                set_user_preference("prefer_organic", "true" if new_organic else "false")
                if new_max_price > 0:
                    set_user_preference("max_price", str(new_max_price))
                else:
                    from shopping_agent import delete_user_preference
                    delete_user_preference("max_price")
                st.success("Preferences updated!")
                st.rerun()

        with col2:
            if st.button("Clear All", use_container_width=True):
                clear_user_preferences()
                st.info("Preferences cleared.")
                st.rerun()

    # --------------------------------------------------------
    # 2. Quick Actions
    # --------------------------------------------------------
    st.markdown("---")
    st.header("⚡ Quick Actions")

    if st.button("📜 What have I ordered before?", use_container_width=True):
        st.session_state.messages.append({
            "role": "user",
            "content": "What have I ordered before?"
        })
        st.rerun()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # --------------------------------------------------------
    # 3. Shop by Image
    # --------------------------------------------------------
    st.markdown("---")
    st.header("📷 Shop by Image")
    st.caption("Upload a photo of a product and I'll find similar items in our store.")

    uploaded_file = st.file_uploader(
        "Upload product image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:
        st.image(
            uploaded_file,
            caption=uploaded_file.name,
            use_container_width=True
        )

    if (
        uploaded_file
        and st.button("Find similar products", use_container_width=True)
    ):
        suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            image_path = tmp.name

        prompt = (
            "I uploaded a product image. "
            "Please analyze it and find similar products in the store. "
            f"Image path: {image_path}"
        )

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        st.session_state.pending_image = uploaded_file.name
        st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user" and msg["content"].startswith("I uploaded a product image"):
            filename = msg["content"].split("Image path:")[-1].strip()
            st.markdown(f"Searching by image: **{os.path.basename(filename)}**")
        else:
            st.markdown(msg["content"].replace("$", r"\$"))


# ============================================================
# PROCESS IMAGE SEARCH
# ============================================================

if (
    st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
    and st.session_state.pending_image
):
    with st.chat_message("assistant"):
        with st.spinner("Analyzing image and searching..."):
            response = get_reply()
        st.markdown(response.replace("$", r"\$"))

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
    st.session_state.pending_image = None
    st.rerun()


# ============================================================
# NORMAL CHAT
# ============================================================

if prompt := st.chat_input("e.g. I want organic honey under $15 with 4+ rating, or ask 'what have I ordered before?'"):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt.replace("$", r"\$"))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_reply()
        st.markdown(response.replace("$", r"\$"))

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
    st.rerun()