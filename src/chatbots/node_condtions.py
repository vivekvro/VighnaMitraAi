from src.state import ChatBotState
from pydantic import BaseModel,Field
from src.LLMs.load_llm import gpt_oss_120b
from langchain_core.messages.utils import count_tokens_approximately






#-------remember_node------------








#------ Conversation Summarizer condition -------------------------
def conversation_summarize_condition(state: ChatBotState):
    if len(state['messages']) > 6 or  count_tokens_approximately(state['messages']) > 2800:
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