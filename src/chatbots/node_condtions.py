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