from pydantic import Field
from typing import TypedDict,Annotated,List,Optional
from langchain_core.messages import BaseMessage,SystemMessage
from langgraph.graph.message import add_messages
from src.chatbots.node_conditions import FetchUploadedDocsDetails,FetchUserMemoryDetails



class BaseChatState(TypedDict):
    system_message:List[SystemMessage]
    messages:Annotated[List[BaseMessage],add_messages]

class SummaryState(TypedDict):
    summary_content :Annotated[str,"Updated summary combining previous + new conversation chunk"]=None
    summary_end_index: int = Field(default=0)

class UserDetails(TypedDict):
    user_id:str
    user_memory:Optional[str]
class Retrieval_schema(TypedDict):
    user_msg:str
    rag_details: List[FetchUploadedDocsDetails]
    user_memories: List[FetchUserMemoryDetails]

class ChatBotState(BaseChatState):
    summary:SummaryState
    retrieval_details:Optional[Retrieval_schema]
    user_details:UserDetails
    trace:List[str]=Field(
        default_factory=list,
        description="in this add used tools and nodes.")