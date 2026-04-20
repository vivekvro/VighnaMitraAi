from pydantic import Field
from typing import Literal,TypedDict,Annotated,Dict,List,Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages






class BaseChatState(TypedDict):
    messages:Annotated[List[BaseMessage],add_messages]


class SummaryState(TypedDict):
    summary :Annotated[str,Field(description="Summary of the chat history.")]

class RagState(TypedDict):
    retriever:Any
    sources:List






class ChatBotState(BaseChatState,SummaryState,RagState):
    pass