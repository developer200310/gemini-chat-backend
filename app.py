import os
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

# Load env
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "models/gemini-flash-latest")

if not GENAI_API_KEY:
    raise RuntimeError("GENAI_API_KEY not set in environment")

genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

app = FastAPI(title="Gemini Chat Backend")

# Update origins to your Vercel domain when deployed for security. For dev use, '*' is fine.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    sender: str  # "user" or "bot"
    text: str

class ChatRequest(BaseModel):
    messages: List[Message]  # full conversation history (client keeps history)

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        # Build a plain text prompt from full conversation history.
        # Format: "User: ...\nAssistant: ...\nUser: ...\nAssistant:"
        parts = []
        for m in req.messages:
            role = "User" if m.sender.lower() == "user" else "Assistant"
            parts.append(f"{role}: {m.text.strip()}")
        # Add the assistant cue to request the next message
        parts.append("Assistant:")
        prompt = "\n".join(parts)

        # Call the model (string input)
        response = model.generate_content(prompt)
        # The wrapper returns .text usually
        reply = response.text.strip() if hasattr(response, "text") else str(response)

        return {"response": reply}
    except Exception as e:
        # Return helpful error
        raise HTTPException(status_code=500, detail=str(e))
