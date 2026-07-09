
import os
import uuid

import gradio as gr

from app.chatbot import HospitalChatbot

bot = HospitalChatbot(api_key=os.environ.get("GROQ_API_KEY"))


def respond(message, chat_history, session_id):
    if session_id is None:
        session_id = str(uuid.uuid4())

    chat_history = chat_history or []

    result = bot.handle_message(session_id, message)

    # The chat window always shows natural language - never raw JSON.
    # "answer" covers medical turns, "message" covers action turns (the
    # full structured JSON with action/doctor_name/specialty/branch/etc.
    # is still returned by bot.handle_message() and by the /chat API in
    # main.py for anything that needs to consume exact structured data).
    display_text = result.get("answer") or result.get("message") or "..."

    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": display_text},
    ]
    return chat_history, session_id


def transcribe_and_fill(audio_path):
    if not audio_path:
        return ""
    from app.voice import transcribe_audio
    return transcribe_audio(audio_path)


with gr.Blocks(title="Al Raya Medical Group - AI Chatbot") as demo:
    gr.Markdown("## 🏥 Al Raya Medical Group — AI Medical Assistant (Arabic / English)")
    session_state = gr.State(None)

    # Gradio's Chatbot API differs slightly across versions: older releases need
    # type="messages" to opt into the role/content dict format, while newer
    # releases (6.x+) made that the only format and dropped the "type" kwarg
    # entirely (passing it raises TypeError). Handle both transparently.
    try:
        chatbot_ui = gr.Chatbot(height=480, type="messages")
    except TypeError:
        chatbot_ui = gr.Chatbot(height=480)
    with gr.Row():
        msg = gr.Textbox(placeholder="اكتب رسالتك هنا / type your message...", scale=4)
        send = gr.Button("Send", scale=1)

    with gr.Accordion("🎙️ Voice input (optional)", open=False):
        audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath")
        transcribe_btn = gr.Button("Transcribe to text box")
        transcribe_btn.click(transcribe_and_fill, inputs=audio_in, outputs=msg)

    def submit(message, history, session_id):
        if not message or not message.strip():
            return message, history, session_id
        history, session_id = respond(message, history, session_id)
        return "", history, session_id

    send.click(submit, inputs=[msg, chatbot_ui, session_state], outputs=[msg, chatbot_ui, session_state])
    msg.submit(submit, inputs=[msg, chatbot_ui, session_state], outputs=[msg, chatbot_ui, session_state])

if __name__ == "__main__":
    demo.launch(share=True)
