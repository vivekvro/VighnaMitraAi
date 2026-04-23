from src.state import ChatBotState
from pydantic import BaseModel,Field
from src.LLMs.load_llm import gpt_oss_120b
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
from typing import Literal




llm = gpt_oss_120b()

#-------remember_node------------

def need_remember_condition(state: ChatBotState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]

    ns = ("user", user_id, "details")
    items = store.search(ns)
    existing_memory = "\n".join(it.value.get("data", "") for it in items)

    last_msgs = state['messages'][-6:]
    contents = [
        f"{'human' if isinstance(msg, HumanMessage) else 'ai'} - {msg.content}"
        for msg in last_msgs
        if not isinstance(msg, ToolMessage)
    ]
    last_msgs_content = "\n".join(contents)

    prompt = f"""
        You are a strict classifier.

        Decide if NEW long-term memory should be stored.

        CURRENT USER DETAILS:
        {existing_memory if existing_memory else "(empty)"}

        LAST CHAT:
        {last_msgs_content}

        Rules:
        - Return "yes" ONLY if there is NEW personal, stable, reusable info
        (preferences, goals, identity, habits)
        - Return "no" if:
        - info is temporary
        - generic
        - OR already exists

        Output ONLY one word:
        yes or no
"""

    response = llm.invoke(prompt)
    decision = response.content.strip().lower()

    return "need_to_remember" if decision == "yes" else "no_need_to_remember"







#------ Conversation Summarizer condition -------------------------
def conversation_summarize_condition(state: ChatBotState):
    if len(state['messages']) > 20  or  count_tokens_approximately(state['messages']) > 2800:
        return True
    else:
        return False
 
#------------- RAG condition----------------------------
class RagCondition(BaseModel):
    need_rag:bool = Field(
        description=
        """Set to True if the user's query requires information from stored documents 
        (e.g., specific, factual, or document-based queries). 
        Set to False if the query can be answered using general knowledge or does not depend on stored documents. for example normal conversation"""
        )
def need_rag_condition(state: ChatBotState):
    llm = gpt_oss_120b()

    query = state['messages'][-1].content

    prompt = f"""
You are a classifier that decides whether a user's query requires retrieval from stored documents (RAG).

Rules:
- Answer ONLY "true" or "false"
- Return "true" if the query depends on user documents, uploaded files, memory, or private knowledge
- Return "false" if it is a normal conversation (greetings, casual talk, opinions)
- Return "false" if general knowledge is enough
- Be strict

Examples:
- "What does my document say about AI?" → true
- "Summarize my uploaded file" → true
- "Hi, how are you?" → false
- "Explain machine learning" → false

Query: {query}
"""
    response = llm.invoke(prompt)
    return response.content.strip().lower() == "true"