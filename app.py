import os
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

GENAI_API_KEY = os.getenv("GENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "models/gemini-flash-latest")

if not GENAI_API_KEY:
    raise RuntimeError("GENAI_API_KEY not set in environment")

# Configure Gemini API
genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# Initialize FastAPI
app = FastAPI(title="Gemini Chat Backend")

# CORS setup
origins = [
    "http://localhost:3000",  # React dev server
    # Add deployed frontend URL here for production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class Message(BaseModel):
    sender: str  # "user" or "bot"
    text: str

class ChatRequest(BaseModel):
    messages: List[Message]  # full conversation history

# Chat endpoint
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        # Build a prompt from conversation history
        parts = []
        for m in req.messages:
            role = "User" if m.sender.lower() == "user" else "Assistant"
            parts.append(f"{role}: {m.text.strip()}")
        parts.append("Assistant:")
        prompt = "\n".join(parts)

        # Call Gemini
        response = model.generate_content(prompt)
        reply = response.text.strip() if hasattr(response, "text") else str(response)

        return {"response": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/")
async def root():
    return {"status": "ok"}
