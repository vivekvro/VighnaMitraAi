from langgraph.graph import StateGraph,START,END
from asyncio import run
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
    need_remember_condition
    )

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode,tools_condition


from src.configs.config_methods import load_config


import os,dotenv

dotenv.load_dotenv()






DB_POSTGRES_URL = os.getenv("DB_POSTGRES_URL")

async def get_tools():

    servers = await load_config()

    client = MultiServerMCPClient(servers)

    tools = await client.get_tools()
    return tools













async def base_chatbot():

    

    postgres_conn = postgres_connect(
        host="localhost",
        dbname="postgres",
        user="postgres",
        password="postgres",
        port=5442

    )
    store = PostgresStore(conn=postgres_conn)
    store


    tools = await get_tools()


    builder_graph= StateGraph(ChatBotState)



    
    builder_graph.add_node("chat_node",chat_node)


    builder_graph.add_node("tools",ToolNode(tools=tools))



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
    builder_graph.add_conditional_edges("chat_node",tools_condition,{
            "tools":"tools",
            "__end__":"summarize_node"
        })
    builder_graph.add_edge("tools","chat_node")
    builder_graph.add_edge("summarize_node","remember_pass_node")

    builder_graph.add_conditional_edges("remember_pass_node",need_remember_condition,{
            "need_to_remember":"remember_node",
            "no_need_to_remember":END
        })




    builder_graph.add_edge("retriever_node","chat_node")

    conn = connect("data/vighnamitraai.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)


    return builder_graph.compile(checkpointer=checkpointer,store=store)


