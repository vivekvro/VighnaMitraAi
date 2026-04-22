from pydantic import Field
from typing import Literal,TypedDict,Annotated,Dict,List,Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_community.vectorstores import VectorStore






class BaseChatState(TypedDict):
    messages:Annotated[List[BaseMessage],add_messages]


class SummaryState(TypedDict):
    summary :Annotated[str,Field(description="Summary of the chat history.")]






class ChatBotState(BaseChatState,SummaryState):
    trace:List[str]=Field(default_factory=list,description="in this add used tools and nodes.")