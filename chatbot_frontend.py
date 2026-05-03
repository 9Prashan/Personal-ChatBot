import queue
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from chatbot_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    submit_async_task,
    thread_document_metadata,
)

# =============================================================================
# Utilities
# =============================================================================

def generate_thread_id() -> uuid.UUID:
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    _add_thread(thread_id)
    st.session_state["message_history"] = []


def _add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id) -> list[dict]:
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])
    result = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        result.append({"role": role, "content": msg.content})
    return result


# =============================================================================
# Session State Initialisation
# =============================================================================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

_add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs: dict = st.session_state["ingested_docs"].setdefault(thread_key, {})

# =============================================================================
# Sidebar
# =============================================================================

st.sidebar.title("LangGraph Chatbot")
st.sidebar.markdown(f"**Thread:** `{thread_key}`")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# ---- PDF uploader ----
st.sidebar.subheader("📄 Upload a PDF")

if thread_docs:
    latest = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"`{latest.get('filename')}` indexed "
        f"({latest.get('chunks')} chunks / {latest.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already indexed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed!", state="complete", expanded=False)

# ---- Past conversations ----
st.sidebar.subheader("🗂 Past Conversations")
selected_thread = None
threads = st.session_state["chat_threads"][::-1]
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for tid in threads:
        if st.sidebar.button(str(tid), key=f"side-{tid}"):
            selected_thread = tid

# =============================================================================
# Main chat area
# =============================================================================

st.title("🤖 Multi-Utility Chatbot")
st.caption("Powered by Google Gemini · Tools: web search, stock prices, calculator, RAG, MCP")

# Render history
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me anything…")

if user_input:
    # Show user message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    # Stream assistant response
    with st.chat_message("assistant"):
        status_holder: dict = {"box": None}

        def _ai_stream():
            """Generator that yields assistant token strings, shows tool status boxes."""
            event_queue: queue.Queue = queue.Queue()

            async def _run_stream():
                try:
                    async for chunk, _ in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        event_queue.put(chunk)
                except Exception as exc:
                    event_queue.put(("__error__", exc))
                finally:
                    event_queue.put(None)  # sentinel

            submit_async_task(_run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break

                # Error passthrough
                if isinstance(item, tuple) and item[0] == "__error__":
                    raise item[1]

                # Tool result → update status widget
                if isinstance(item, ToolMessage):
                    tool_name = getattr(item, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}`…", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}`…",
                            state="running",
                            expanded=True,
                        )

                # AI token → stream to UI
                # Gemini returns content as a list of dicts: [{"type": "text", "text": "..."}]
                # Handle both list (Gemini) and plain string formats
                if isinstance(item, AIMessage) and item.content:
                    content = item.content
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text", "")
                                if text:
                                    yield text
                    elif isinstance(content, str):
                        yield content

        ai_message = st.write_stream(_ai_stream())

        # Finalise tool status
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    # Show doc metadata if available
    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"📎 Document: {doc_meta.get('filename')} "
            f"({doc_meta.get('chunks')} chunks, {doc_meta.get('documents')} pages)"
        )

st.divider()

# Handle sidebar thread selection (must be at bottom to avoid re-render glitches)
if selected_thread:
    st.session_state["thread_id"] = selected_thread
    st.session_state["message_history"] = load_conversation(selected_thread)
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()