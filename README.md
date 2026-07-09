# Al Raya Medical Group — AI Medical Chatbot

An AI chatbot for a hospital group that answers patient symptom questions in Arabic or
English and helps them book an appointment with the right doctor and branch.

## Approach & System Design

Each incoming message goes through the same pipeline:

1. A quick TF-IDF search over the hospital's medical knowledge base pulls the 2-3 most
   relevant entries for the message (if any).
2. The message, the conversation history, and that context are sent to the LLM along
   with a set of tools: `list_branches`, `list_specializations`, `list_doctors`, and
   `book_appointment`.
3. The model either replies in plain text (a medical answer) or calls one of the tools
   (an action request). These two paths never mix in the same turn — the model can only
   do one or the other, since a tool call and a text reply are structurally different
   things in the API response.
4. If a tool was called, the actual lookup runs in Python against the hospital's data
   (`hospital_service.py`), using fuzzy name matching so things like "Dr. Sarah" resolve
   to "Dr. Sarah Hassan". The model never invents doctors, branches, or specialties —
   it only extracts what the user asked for, and Python resolves it against real data.
5. The conversation history is stored per session so follow-up messages ("yes, book that
   one") carry over context from earlier turns.

The medical knowledge base is small and curated, so TF-IDF keyword matching is enough —
it's accurate, doesn't need an external embeddings API, and runs offline.

## AI Model(s) Used & Why

| Purpose | Model | Why |
|---|---|---|
| Conversation, medical answers, deciding medical vs. action, extracting booking details | `openai/gpt-oss-120b` via the Groq API | Free tier, no billing required, supports proper tool/function calling, handles Arabic and English well |
| Medical knowledge retrieval | TF-IDF (scikit-learn) | Small dataset, no need for an embeddings API — deterministic and free |
| Doctor / branch / specialty matching | `difflib` (Python standard library) | Deterministic fuzzy matching against real hospital records, no hallucination risk |
| Voice transcription (optional) | `faster-whisper` (small model) | Runs locally, free, handles Arabic and English |

Groq was chosen because it has a genuinely free tier (no credit card) with reliable
access — get a key at console.groq.com/keys.

## Prompt Design Strategy

The system prompt gives the model a fixed identity (hospital name, branches) and two
strict rules:

- **Language mirroring** — always reply in the same language the user used.
- **Two response types, never mixed** — for a medical question, answer in plain text and
  end by suggesting the user book an appointment. For anything the user wants to *do*
  (book, list doctors, list specialties, list branches), call the matching tool instead
  of writing text, using whatever details are available from the whole conversation so
  far, not just the latest message.

The model is explicitly told to treat tool results as the only source of truth for
hospital facts, so it doesn't fill in gaps with guesses. Relevant medical knowledge
snippets are appended to each turn only when the retrieval step actually finds a match,
keeping the prompt short.

## How to Configure and Run

**Requirements:** Python 3.10+, and a free Groq API key from console.groq.com/keys.

```bash
cd hospital_chatbot
pip install -r requirements.txt
export GROQ_API_KEY="gsk_your_key_here"
```

**Run the demo UI:**
```bash
python gradio_app.py
```
This opens a local chat interface and also prints a public link.

**Run the API instead:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"u1","message":"I have a severe headache and fever for two days"}'
```

**Running on Google Colab:**
```python
%cd hospital_chatbot
!pip install -r requirements.txt -q

import os
os.environ["GROQ_API_KEY"] = "gsk_your_key_here"

!python gradio_app.py
```

## Generated Hospital Test Dataset

- **Hospital group:** Al Raya Medical Group (مجموعة الراية الطبية)
- **Branches:** Cairo, Alexandria, Riyadh, Dubai
- **18 doctors** across Cardiology, Neurology, Orthopedics, Dermatology, Pediatrics, ENT,
  Endocrinology, Internal Medicine, and Psychiatry (`data/hospital_data.json`)
- **16 bilingual medical knowledge entries** covering common symptoms — headache/fever,
  chest pain, cough, abdominal pain, back pain, rash, joint pain, dizziness,
  hypertension, diabetes symptoms, migraine, insomnia, anxiety, sprains, ear pain, and
  pediatric fever (`data/medical_knowledge.json`)

## Example Conversations

### Multi-turn medical conversation ending in a booking suggestion

```
User: "I've been having a severe headache and fever for two days, what should I do?"
Bot:  الصداع الشديد المصحوب بالحمى لمدة تزيد عن 24 ساعة قد يشير إلى التهاب أو عدوى تستوجب
      الفحص الطبي. يُنصح بالراحة التامة وشرب كميات كافية من الماء وتجنب التعرض للضوء الساطع.
      إذا استمرت الأعراض أو تفاقمت، فمن المهم مراجعة طبيب متخصص في أقرب وقت. هل تودّ حجز
      موعد مع أحد أطبائنا؟

User: "Yes, I want to book with Dr. Sarah in Neurology at the Cairo branch"
Bot:  {
        "action": "book_appointment",
        "doctor_name": "Dr. Sarah Hassan",
        "specialty": "Neurology",
        "branch": "Cairo",
        "hospital": "Al Raya Medical Group - Cairo Branch"
      }
```

### Asking about available doctors or specializations

```
User: "Who are the cardiologists at the Riyadh branch?"
Bot:  {
        "action": "list_doctors",
        "specialty": "Cardiology",
        "branch": "Riyadh",
        "doctors": [
          {"name": "Dr. Khalid Al-Mansouri", "specialty": "Cardiology"},
          {"name": "Dr. Layla Nasser", "specialty": "Cardiology"}
        ]
      }

User: "What specializations are available at the Alexandria branch?"
Bot:  {
        "action": "list_specializations",
        "branch": "Alexandria",
        "specializations": ["Cardiology", "Neurology", "Orthopedics", "Dermatology"]
      }
```

### Voice input being transcribed and processed

```
POST /voice  (audio file + session_id)
Response: {"transcribed_text": "عندي ألم في الصدر من الصباح وأحس بضيق في التنفس"}

# that text is then sent to /chat as a normal message:
POST /chat {"session_id": "...", "message": "عندي ألم في الصدر من الصباح وأحس بضيق في التنفس"}
Bot: ألم الصدر مع ضيق التنفس قد يكون خطيرًا وقد يرتبط بالقلب أو الرئتين، وهو يستدعي تقييمًا
     عاجلاً. هل تودّ حجز موعد مع أحد أطبائنا؟
```
