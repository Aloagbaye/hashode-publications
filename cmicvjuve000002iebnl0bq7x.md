---
title: "🧱 Building a Smarter RAG Pipeline on Azure: From FastAPI to GraphRAG (Part 1)"
datePublished: Mon Nov 24 2025 08:18:00 GMT+0000 (Coordinated Universal Time)
cuid: cmicvjuve000002iebnl0bq7x
slug: building-a-smarter-rag-pipeline-on-azure-from-fastapi-to-graphrag-part-1
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1763971985664/e10ed1ba-e8e9-4832-ab3b-e7818dae55e9.png
tags: azure, fastapi, openai, mlops, llm, large-language-models

---

Retrieval-Augmented Generation (RAG) is already transforming how organizations search internal knowledge — but most tutorials only show simple, toy examples.

In this series, I walk through how I built a **real-world RAG pipeline on Azure**, combining:

* FastAPI
    
* Azure OpenAI
    
* Hybrid search
    
* GraphRAG (in the final stage)
    

This first article covers the setup foundations:

> **Module 0 — Environment Setup**  
> **Module 1 — FastAPI + Azure OpenAI**

Before writing any code, I set up a clean development environment to make the project easy to scale, containerize, and deploy.

---

## **1\. Install Tools**

I installed the following tools:

* **Python 3.11+**
    
* **VS Code**
    
* **Docker Desktop**
    
* **Azure CLI**
    
* **Azure Developer CLI (azd)**
    

These will handle local development, containerization, and Azure deployment later on.

---

## **2\. Create the Project**

```bash
mkdir azure-rag
cd azure-rag
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## **3\. Install Dependencies**

```bash
pip install fastapi uvicorn python-dotenv azure-ai-openai
```

`azure-ai-openai` is the updated official SDK for Azure OpenAI.

---

## **4\. Create** `.env`

Inside the project root:

```plaintext
AZURE_OPENAI_ENDPOINT="your-endpoint-here"
AZURE_OPENAI_KEY="your-api-key"
AZURE_OPENAI_MODEL="gpt-4o-mini"
```

This keeps secrets out of GitHub.

---

# ⚙️ Module 1 — FastAPI + Azure OpenAI

With the environment ready, the next step is building a simple REST API to interact with Azure OpenAI.

Later, this API will become the “brain” behind the full RAG pipeline.

---

# 📁 Project Structure

A clean structure makes the project easier to expand later:

```plaintext
azure-rag/
 ├── app/
 │   ├── main.py
 │   ├── config.py
 │   ├── routes/
 │   │     └── chat.py
 │   └── utils/
 ├── .env
 └── requirements.txt
```

---

# ⚙️ Step 1: Configuration Loader

```python
# app/config.py
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")
```

---

# ✉️ Step 2: Build the Chat Route

```python
# app/routes/chat.py
from fastapi import APIRouter
from app.config import OPENAI_ENDPOINT, OPENAI_KEY, OPENAI_MODEL
from azure.ai.openai import AzureOpenAI

router = APIRouter()

client = AzureOpenAI(
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_KEY,
    api_version="2024-08-01-preview"
)

@router.post("/chat")
async def chat(message: str):
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": message}]
    )
    return {"response": response.choices[0].message["content"]}
```

---

# 🚀 Step 3: FastAPI Main Application

```python
# app/main.py
from fastapi import FastAPI
from app.routes.chat import router as chat_router

app = FastAPI(title="Azure RAG API")

app.include_router(chat_router, prefix="/api")
```

---

# ▶️ Step 4: Run the API

```bash
uvicorn app.main:app --reload
```

Visit:

```plaintext
http://localhost:8000/docs
```

You now have a working API that sends requests to Azure OpenAI.

---

# 🔮 What’s Coming in Part 2

In the next article, I’ll cover:

* Creating embeddings
    
* Chunking and preprocessing
    
* Hybrid search (keyword + vector)
    
* Retrieval pipeline design
    
* Clean RAG service architecture
    

To follow along with the entire series, stay tuned for Part 2!

---

# 📝 Final Notes

The full working code will be available on GitHub:

👉 **https://github.com/Aloagbaye/azure-genai**