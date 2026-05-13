# Standard
import os
import dotenv

# Third-party
import psycopg
from aiosqlite import connect
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.postgres import PostgresStore
from langgraph.prebuilt import tools_condition

# Local
from src.state import ChatBotState
from src.chatbots.nodes import (
    init_SystemMessage,
    chat_node,
    remember_node,
    tool_node,
    summarize_conversation,
    retriever_node,
    retrieve_user_memory_node,
)
from src.chatbots.node_conditions import (
        MemoryCondition
    )
# --------------------------------------------------------------------------------------

dotenv.load_dotenv()

DB_POSTGRES_URL = os.getenv("DB_POSTGRES_URL")


async def base_chatbot():

    # ✅ Async Postgres connection with autocommit
    postgres_conn = psycopg.connect(
        conninfo=DB_POSTGRES_URL,
        autocommit=True
    )

    # ✅ Store setup
    store = PostgresStore(conn=postgres_conn)
    store.setup()

    # ✅ Build graph
    builder_graph = StateGraph(ChatBotState)

    builder_graph.add_node("init_SystemMessage", init_SystemMessage)
    builder_graph.add_node("chat_node", chat_node)
    builder_graph.add_node("tool_node", tool_node)
    builder_graph.add_node("summarize_node", summarize_conversation)
    builder_graph.add_node("remember_node", remember_node)
    builder_graph.add_node("retriever_node", retriever_node)
    builder_graph.add_node("retrieve_user_memory_node", retrieve_user_memory_node)

    # ✅ RAG routing
    builder_graph.add_conditional_edges(START, MemoryCondition, {
        "uploaded_documents": "retriever_node",
        "user_memories":"retrieve_user_memory_node",
        "chat_node": "chat_node"
    })
    
    builder_graph.add_edge("retriever_node", END)

    # ✅ Tool routing
    builder_graph.add_conditional_edges("chat_node", tools_condition, {
        "tools": "tool_node",
        "__end__": "summarize_node"
    })

    builder_graph.add_edge("tool_node", "chat_node")
    builder_graph.add_edge("summarize_node", "remember_node")


    # ✅ Async SQLite checkpoint
    conn = await connect("data/vighnamitraai.db", check_same_thread=False)
    checkpointer = AsyncSqliteSaver(conn=conn)

    # ✅ Compile graph
    return builder_graph.compile(
        checkpointer=checkpointer,
        store=store
    )