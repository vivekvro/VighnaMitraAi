from langgraph.graph import StateGraph,START,END
from langchain_core.messages.utils import trim_messages,count_tokens_approximately
from src.state import ChatBotState
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import SystemMessage,AIMessage,RemoveMessage,HumanMessage
from sqlite3 import connect
from  psycopg import connect as postgres_connect
from langgraph.store.postgres import PostgresStore
from src.chatbots.nodes import (
    chat_node,remember_node,
    summarize_conversation,
    retriever_node,remember_pass_node
    )
from src.chatbots.node_condtions import (
    need_rag_condition,
    conversation_summarize_condition,
    need_remember_condition
    )
import os,dotenv

dotenv.load_dotenv()






DB_POSTGRES_URL = os.getenv("DB_POSTGRES_URL")




def base_chatbot():

    postgres_conn = postgres_connect(
        host="postgres",
        dbname="postgres",
        user="postgres",
        password="postgres",
        port=5432

    )
    store = PostgresStore(conn=postgres_conn)
    store

    builder_graph= StateGraph(ChatBotState)
    
    builder_graph.add_node("chat_node",chat_node)

    builder_graph.add_node("summarize_node",summarize_conversation)

    builder_graph.add_node("remember_pass_node",remember_pass_node)
    builder_graph.add_node("remember_node",remember_node)


    builder_graph.add_node("retriever_node",retriever_node)



    builder_graph.add_conditional_edges(START,need_rag_condition,{
            True:"retriever_node",
            False:"chat_node"
        })
    # builder_graph.add_edge("chat_node","remember_pass_node")
    # builder_graph.add_edge("chat_node","summarizer_pass_node")



    need_remember_condition
    builder_graph.add_conditional_edges("chat_node",conversation_summarize_condition,{
            True:"summarize_node",
            False:"remember_pass_node"
        })
    builder_graph.add_edge("summarize_node","remember_pass_node")

    builder_graph.add_conditional_edges("remember_pass_node",need_remember_condition,{
            "need_to_remember":"remember_node",
            "no_need_to_remember":END
        })




    builder_graph.add_edge("retriever_node","chat_node")
    builder_graph.add_edge("summarize_node",END)

    conn = connect("data/vighnamitraai.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)


    return builder_graph.compile(checkpointer=checkpointer,store=store)


