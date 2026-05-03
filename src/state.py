from pydantic import Field
from typing import Literal,TypedDict,Annotated,Dict,List,Any,Tuple
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_community.vectorstores import VectorStore
from operator import add







class BaseChatState(TypedDict):
    messages:Annotated[List[BaseMessage],add_messages]


class SummaryState(TypedDict):
    summary :Annotated[str,"Updated summary combining previous + new conversation chunk"]
    summary_end_index: int = 0






class ChatBotState(BaseChatState,SummaryState):
    user_id:str
    trace:List[str]=Field(default_factory=list,description="in this add used tools and nodes.")