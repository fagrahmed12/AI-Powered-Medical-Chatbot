"""
FastAPI entry point.

Endpoints:
  POST /chat            {session_id, message}         -> medical answer OR structured action JSON
  POST /voice            multipart audio file + session_id -> {"transcribed_text": "..."}
                          (client is expected to then call /chat with that text)
  POST /reset            {session_id}                  -> clears conversation history
  GET  /health

Run:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.chatbot import HospitalChatbot

app = FastAPI(title="Al Raya Medical Group - AI Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bot = HospitalChatbot(api_key=os.environ.get("GROQ_API_KEY"))


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ResetRequest(BaseModel):
    session_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    result = bot.handle_message(req.session_id, req.message)
    return result


@app.post("/reset")
def reset(req: ResetRequest):
    bot.reset_session(req.session_id)
    return {"status": "reset"}


@app.post("/voice")
async def voice(session_id: str = Form(...), file: UploadFile = File(...)):
    from app.voice import transcribe_audio

    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        text = transcribe_audio(tmp_path)
    finally:
        os.unlink(tmp_path)

    return {"transcribed_text": text}
