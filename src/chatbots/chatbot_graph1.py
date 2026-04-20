from langgraph.graph import StateGraph,START,END
from langchain_core.messages.utils import trim_messages,count_tokens_approximately
from src.LLMs.load_llm import llama_3_3_70b_versatile
from src.state import ChatBotState
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import SystemMessage,AIMessage,RemoveMessage,HumanMessage
from sqlite3 import connect





llm = llama_3_3_70b_versatile()




def chat_node(state:ChatBotState):
    messages = []

    if state['summary']:
        messages.append(SystemMessage(content=f"Conversation Summary:\n{state['summary']}"))
    messages.extend(state["messages"])

    response=llm.invoke(messages)
    return {"messages":[response]}


def summarize_conversation(state: ChatBotState):
    existing_summary = state['summary']
    if existing_summary:
        prompt =  (
            f"existing summary:\n{existing_summary}\n\n"
            "extend the summary using the new converstion above."
        )
    else :
        prompt = "Summarize the conversation above."
    messages_for_summary = state["messages"] + [
        HumanMessage(content=prompt)
    ]
    response = llm.invoke(messages_for_summary)
    messages_to_delete = state['messages'][:-2]
    return {
        "summary":response.content,
        "messages":[RemoveMessage(id=m.id) for m in messages_to_delete]
    }


def should_summarize_condition(state: ChatBotState):
    return len(state['messages']) > 6


def base_chatbot():






    builder_graph= StateGraph(ChatBotState)

    builder_graph.add_node("chat_node",chat_node)
    builder_graph.add_node("summarize_node",summarize_conversation)


    builder_graph.add_edge(START,"chat_node")
    builder_graph.add_conditional_edges("chat_node",should_summarize_condition,{
        True:"summarize_node",
        False:END
    })
    builder_graph.add_edge("summarize_node",END)
    conn = connect("data/vighnamitraai.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)

    return builder_graph.compile(checkpointer=checkpointer)



