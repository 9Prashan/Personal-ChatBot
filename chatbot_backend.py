from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
import threading
from typing import Annotated, Any, Dict, Optional, TypedDict

import aiosqlite
import requests
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool, BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# =============================================================================
# API Key — works both locally (.env) and on Streamlit Cloud (st.secrets)
# =============================================================================
def _get_api_key() -> str:
    # Try environment / .env first
    key = os.getenv("GOOGLE_API_KEY", "")
    if key:
        return key
    # Fall back to Streamlit secrets (cloud deployment)
    try:
        import streamlit as st
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        raise RuntimeError(
            "GOOGLE_API_KEY not found. Add it to .env (local) "
            "or Streamlit Cloud secrets (deployment)."
        )

GOOGLE_API_KEY = _get_api_key()

# =============================================================================
# SQLite path — use /tmp on cloud (writable), local path otherwise
# =============================================================================
DB_PATH = os.path.join(tempfile.gettempdir(), "chatbot.db")

# =============================================================================
# Async event loop (needed for async streaming)
# =============================================================================
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop (non-blocking)."""
    return _submit_async(coro)


# =============================================================================
# 1. LLM + Embeddings  (Google Gemini — free tier)
# =============================================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",           # Free: 15 RPM, 1M TPM, 1500 req/day
    google_api_key=GOOGLE_API_KEY,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",  # Free: 1500 req/day
    google_api_key=GOOGLE_API_KEY,
)

# =============================================================================
# 2. PDF retriever store (per thread, in-memory)
# =============================================================================
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """Build a FAISS retriever for the uploaded PDF and store it for the thread."""
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        return _THREAD_METADATA[str(thread_id)]
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})


# =============================================================================
# 3. Tools
# =============================================================================
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage.
    """
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {"error": "No document indexed for this chat. Upload a PDF first.", "query": query}

    result = retriever.invoke(query)
    return {
        "query": query,
        "context": [doc.page_content for doc in result],
        "metadata": [doc.metadata for doc in result],
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


# =============================================================================
# 4. MCP Client (only remote HTTP server — stdio removed for cloud compatibility)
# =============================================================================
def _load_mcp_tools() -> list[BaseTool]:
    try:
        client = MultiServerMCPClient(
            {
                "expense": {
                    "transport": "streamable_http",
                    "url": "https://splendid-gold-dingo.fastmcp.app/mcp",
                },
            }
        )
        return run_async(client.get_tools())
    except Exception:
        return []


mcp_tools: list[BaseTool] = _load_mcp_tools()

# Full tool list
all_tools = [search_tool, get_stock_price, calculator, rag_tool, *mcp_tools]
llm_with_tools = llm.bind_tools(all_tools)

# =============================================================================
# 5. State
# =============================================================================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# =============================================================================
# 6. Nodes
# =============================================================================
async def chat_node(state: ChatState, config=None):
    """LLM node — may answer directly or request tool calls."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            f"the `rag_tool` and include the thread_id `{thread_id}`. "
            "You can also use web search, stock price, calculator, and MCP tools when helpful. "
            "If no document is available and the user asks about one, ask them to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(all_tools)


# =============================================================================
# 7. Checkpointer (async SQLite, stored in /tmp for cloud compatibility)
# =============================================================================
async def _init_checkpointer():
    conn = await aiosqlite.connect(DB_PATH)
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())


# =============================================================================
# 8. Graph
# =============================================================================
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# =============================================================================
# 9. Helpers
# =============================================================================
async def _alist_threads():
    all_threads: set = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads() -> list:
    return run_async(_alist_threads())