from langgraph.graph import StateGraph,START,END
from langchain_core.messages.utils import trim_messages,count_tokens_approximately
from src.state import ChatBotState
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import SystemMessage,AIMessage,RemoveMessage,HumanMessage
from sqlite3 import connect
from  psycopg import connect as postgres_connect
from langgraph.store.postgres import PostgresStore
from src.chatbots.nodes import chat_node,remember_node,summarize_conversation,retriever_node
from src.chatbots.node_condtions import need_rag_condition,conversation_summarize_condition
import os,dotenv

dotenv.load_dotenv()






DB_POSTGRES_URL = os.getenv("DB_POSTGRES_URL")


import streamlit as st






def base_chatbot():

    postgres_conn = postgres_connect(
        host="localhost",
        dbname="postgres",
        user="postgres",
        password="postgres",
        port=5442

    )
    store = PostgresStore(conn=postgres_conn)
    store

    builder_graph= StateGraph(ChatBotState)

    builder_graph.add_node("chat_node",chat_node)
    builder_graph.add_node("summarize_node",summarize_conversation)
    builder_graph.add_node("retriever_node",retriever_node)
    builder_graph.add_node("remember_node",remember_node)




    builder_graph.add_conditional_edges(START,need_rag_condition,{
            True:"retriever_node",
            False:"chat_node"
        })


    builder_graph.add_conditional_edges("chat_node",conversation_summarize_condition,{
            True:"summarize_node",
            False:"remember_node"
        })
    builder_graph.add_edge("retriever_node","chat_node")
    builder_graph.add_edge("summarize_node","remember_node")
    builder_graph.add_edge("remember_node",END)

    conn = connect("data/vighnamitraai.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)


    return builder_graph.compile(checkpointer=checkpointer,store=store)


