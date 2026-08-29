from fastapi import FastAPI
from pydantic import BaseModel

import sys
from pathlib import Path

# Allow importing scripts modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(
    0,
    str(PROJECT_ROOT / "scripts")
)

from assistant import MavalAssistant
from retriever import MavalRetriever
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="MAVAL AI Assistant API",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load once when server starts
assistant = MavalAssistant()
retriever = MavalRetriever()


class ChatRequest(BaseModel):
    message: str
    customer_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list
    model: str
    language: str



@app.get("/")
def home():
    return {
        "status": "MAVAL AI Assistant running"
    }



@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    try:
        print("QUESTION:", request.message)

        # Retrieve catalog information
        if is_small_talk(request.message):
            return {
                "answer": "Hello! I am the MAVAL technical assistant. How can I help you with our aluminum products?",
                "sources": [],
                "model": "llama3.1:latest",
                "language": "English"
            }


        chunks = retriever.search(
            request.message
        )

        print("CHUNKS FOUND:", len(chunks))

        result = assistant.answer(
            request.message,
            chunks
        )

        print("ANSWER GENERATED")

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "model": result["model"],
            "language": result["language"]
        }

    except Exception as e:
        print("ERROR:", e)
        raise e
def is_small_talk(message: str):
    greetings = [
        "hello",
        "hi",
        "hey",
        "bonjour",
        "salut",
        "مرحبا"
    ]

    return message.lower().strip() in greetings

@app.get("/health")
def health():
    return {
        "api": "running",
        "assistant": "loaded",
        "retriever": "loaded"
    }