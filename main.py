# backend/app.py
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

genai.configure(api_key="AIzaSyAx-fPpv_vla3K6tW5oIf56P5NDaGS_wA0")
model = genai.GenerativeModel("gemini-2.5-flash")  # update to a working model

app = FastAPI()

# Allow your React app to talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        response = model.generate_content(request.message)
        return {"response": response.text}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:chat", host="0.0.0.0", port=8000, reload=True)
