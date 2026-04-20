from langchain_groq import ChatGroq
from dotenv import load_dotenv
def llama_3_3_70b_versatile(temperature=0.7):
    load_dotenv()
    return ChatGroq(model="llama-3.3-70b-versatile" ,temperature=temperature)
def gpt_oss_120b(temperature=0.7):
    load_dotenv()
    return ChatGroq(model="openai/gpt-oss-120b",temperature=temperature)

def gpt_oss_20b(temperature=0.7):
    load_dotenv()
    return ChatGroq(model="openai/gpt-oss-20b",temperature=temperature)


def qwen3_32b(temperature=0.7):
    load_dotenv()
    return ChatGroq(model="qwen/qwen3-32b",temperature=temperature)
