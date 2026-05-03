# Standard
from typing import Literal

# Third-party
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.store.base import BaseStore


# Local
from src.state import ChatBotState
from src.LLMs.load_llm import gpt_oss_120b




llm = gpt_oss_120b()

#-------remember_node------------


class RememberNodeConditon(BaseModel):
    need_to_remember :bool= Field(description="""Return True only if the conversation includes persistent, reusable information (e.g., preferences, identity, goals, constraints, or important context).
Return False if the content is generic, one-time, or not useful for future conversations.
""")

def need_remember_condition(state: ChatBotState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]

    ns = ("user", user_id, "details")
    items = store.search(ns)
    existing_memory = "\n".join(it.value.get("data", "") for it in items) if items else "(empty)"

    last_msgs = state['messages'][-6:]
    contents = [
        f"{'human'} - {msg.content}"
        for msg in last_msgs
        if isinstance(msg, HumanMessage)
    ]
    last_msgs_content = "\n".join(contents)


    prompt = PromptTemplate()

    prompt = f"""
        You are a strict classifier.

        Decide if NEW long-term memory should be stored.

        CURRENT USER DETAILS:
        {existing_memory}

        LAST CHAT:
        {last_msgs_content}

        Rules:
        - Return True ONLY if there is NEW personal, stable, reusable info
        (preferences, goals, identity, habits)
        - Return False if:
            - info is temporary
            - generic
            - OR already exists

        Output ONLY one word:
        True or False
"""
    chain = prompt | llm | parser
    response = llm.invoke(prompt)
    decision = response.content.strip().lower()

    return "need_to_remember" if decision == "yes" else "no_need_to_remember"







#------------- RAG condition----------------------------
class RagConditionClass(BaseModel):
    need_rag: bool = Field(
        description="""
Return True if the user's query requires retrieving information from a knowledge base,
documents, or vector database (RAG).

Return False if the query can be answered using:
- general knowledge
- reasoning or logic
- latest info (of recent event,if not asking from uploaded documents)
- conversation
- tool usage (calculations, API calls, etc.)
"""
    )


def need_rag_condition(state: ChatBotState):
    llm = gpt_oss_120b()

    query = state['messages'][-1].content

    parser = PydanticOutputParser(pydantic_object=RagConditionClass)

    prompt = PromptTemplate(
        template="""
        You are a routing classifier in an AI system.

        Decide whether the query needs RAG.

        {format_instructions}

        Rules:
        - True → needs external knowledge, documents,  latest info (if asked from uploaded Documents)
        - False → general knowledge, reasoning, conversation, or tool usage

        User Query:
        {user_input}
        """,
        input_variables=['user_input'],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    chain = prompt | llm | parser
    response = chain.invoke({"user_input": query})
    return response.need_rag