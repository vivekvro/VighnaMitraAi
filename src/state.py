from pydantic import Field
from typing import Literal,TypedDict,Annotated,Dict,List,Any,Tuple
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_community.vectorstores import VectorStore
from operator import add







class BaseChatState(TypedDict):
    messages:Annotated[List[BaseMessage],add_messages]


class SummaryState(TypedDict):
    summary_content :Annotated[str,"Updated summary combining previous + new conversation chunk"]=None
    summary_end_index: int = 0

class UserDetails(TypedDict):
    user_id:str
    user_memroy:str=""






class ChatBotState(BaseChatState):
    summary:SummaryState
    user_details:UserDetails
    trace:List[str]=Field(default_factory=list,description="in this add used tools and nodes.")