# 🧠 VighnaMitra AI

VighnaMitra AI is an intelligent, memory-enabled conversational system built using **LangGraph**, **FastAPI**, and **LLMs (Groq + HuggingFace)**.  
It combines **RAG (Retrieval-Augmented Generation)**, **long-term memory**, and **context-aware summarization** to deliver smarter, evolving conversations.

---

## 🚀 Key Features

- 🔍 **RAG-based Retrieval**
  - Dynamically fetches relevant context before generating responses

- 💬 **Context-Aware Chat Engine**
  - Handles natural conversations with memory of past interactions

- 🧠 **Long-Term Memory System**
  - Stores useful user information for future conversations
  - Avoids redundant or irrelevant memory storage

- ✂️ **Conversation Summarization**
  - Maintains context window efficiency by summarizing when needed

- ⚡ **FastAPI Backend + Streamlit Frontend**
  - Clean API layer with interactive UI

---

## 🏗️ Architecture Overview (Conceptual)

Instead of focusing on flowchart complexity, here’s the **core logic**:

1. Input arrives  
2. System decides:  
   - Need external knowledge? → use retriever  
   - Otherwise → direct chat  
3. Response generated  
4. Optional:
   - Summarize conversation (if needed)  
   - Store memory (if useful)  
5. Return final response  

👉 Think of it as:  
**"Chat + Retrieval + Memory + Optimization loop"**

---

## 📦 Tech Stack

- **LangGraph** → orchestration  
- **FastAPI** → backend APIs  
- **Streamlit** → frontend UI  
- **Groq API** → fast LLM inference  
- **HuggingFace** → embeddings / models  
- **FAISS / Vector DB** → retrieval  
- **PostgreSQL** → persistent memory  

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd VighnaMitraAI
```

---

### 2. Create `.env` file

```env
USER_AGENT="VighnaMitraAI/v1.1"

GROQ_API_KEY="api"
HF_TOKEN="api"

MAIN_URL="http://127.0.0.1:8000"

DB_POSTGRES_URL="postgresql://postgres:postgres@localhost:5442/postgres"
```

---

### 3. Install dependencies

(If using `uv`)
```bash
uv sync
```

(or pip)
```bash
pip install 
```

---

### 4. Run Backend

```bash
uvicorn src.routes:app --reload
```

---

### 5. Run Frontend

```bash
streamlit run app.py
```

---

## 🧠 Memory System Insight

Your system uses **selective memory storage**, not brute-force logging.

✔ Stores:
- User preferences  
- Long-term useful facts  

❌ Avoids:
- Temporary context  
- Redundant data  

👉 This makes it scalable and intelligent over time.

---

## 🔮 Future Improvements (Planned)

- Better memory update/delete logic  
- Time-aware contextual memory  
- MCP-based tools (web search, automation)  
- VLM (Vision-Language Models)  
- Advanced agentic workflows  

---

## 📌 Project Goal

To build a **production-ready AI assistant** that:
- Remembers intelligently  
- Adapts over time  
- Combines retrieval + reasoning + personalization  

---

## 👨‍💻 Author

**Vivek Singh**  
BCA | AI/ML Enthusiast | Building real-world AI systems  

---

## 🧠 AI Insight (Latest)

- OpenAI and others are pushing **memory-native agents** → persistent context is becoming the future of AI systems.
