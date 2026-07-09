"""
Wraps calls to the Groq API (free tier) and encodes the core design decision
of this project:

    - Medical question  -> plain natural-language answer (no tool call)
    - Action request     -> the model MUST call exactly one tool
                            (book_appointment / list_doctors /
                             list_specializations / list_branches)

Why Groq instead of Gemini/Anthropic/OpenAI:
    - Groq's free tier (https://console.groq.com/keys) has generally been the
      most reliable "just works, no billing, no regional quota=0 surprises"
      free option in practice.
    - It exposes an OpenAI-compatible Chat Completions API with native tool
      calling, so we keep the exact same "let the model's tool-use decide
      medical vs action" design this project relies on for requirement #4
      (never mix the two response types).
    - Llama 3.3 70B has solid Arabic + English support and is fast on Groq's
      hardware.

Get a free key at: https://console.groq.com/keys
Set it as the environment variable GROQ_API_KEY.
"""
import os
import json
from typing import List, Dict, Optional

from groq import Groq

MODEL = "llama-3.3-70b-versatile"  # strong multilingual + tool-use, free-tier friendly on Groq

SYSTEM_PROMPT = """You are the official virtual medical assistant of {group_name} ({group_name_ar}),
a hospital group with branches in Cairo, Alexandria, Riyadh and Dubai.

You must always reply in the SAME language the user is using (Arabic or English). Do not mix languages.

You handle exactly two kinds of user turns, and you must never blend them:

1. MEDICAL QUESTIONS - the user describes symptoms or asks about a condition.
   - Respond ONLY with a warm, empathetic, clear natural-language explanation and general guidance.
   - Ground your medical explanation in the "Relevant medical knowledge" context provided below when it is relevant. If nothing relevant is provided, answer from general safe medical knowledge, briefly and cautiously.
   - Always make clear you are not a substitute for an in-person medical diagnosis.
   - ALWAYS end this kind of reply by naturally inviting the user to book an appointment with a suitable specialist at {group_name}.
   - Do NOT call any tool/function for this kind of turn. Just answer in text.

2. ACTION REQUESTS - the user wants to DO something: book an appointment, see available doctors,
   see specializations at a branch, or see the list of branches.
   - For these, you MUST call the single most appropriate function (book_appointment, list_doctors,
     list_specializations, or list_branches) with your best-guess parameters extracted from the
     whole conversation so far (not just the last message - e.g. if the user earlier said "Cairo"
     and now says "book with Dr. Sarah in Neurology", pass branch="Cairo").
   - Do NOT also write a free-text medical answer in the same turn when you call a function.
   - If the user's message is a natural continuation of booking intent (e.g. after you asked them
     to consider booking and they say "yes" / "نعم" / "اه ابغى احجز"), and specialty or branch
     was already discussed, call book_appointment with whatever info is available - missing fields
     are fine, the function will ask the user to clarify if needed.

Never invent doctors, branches or specialties that are not returned by the functions - always rely
on function results as the source of truth for anything factual about the hospital.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_branches",
            "description": "List all hospital branches in the group. Use when the user asks what branches/locations exist.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_specializations",
            "description": "List the medical specializations available, optionally filtered to one branch. Use when the user asks what specialties/departments are offered somewhere.",
            "parameters": {
                "type": "object",
                "properties": {
                    "branch": {"type": "string", "description": "Branch name mentioned by the user, e.g. 'Cairo', 'الرياض'. Omit if not specified."}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_doctors",
            "description": "List doctors, optionally filtered by branch and/or specialty. Use when the user asks who the doctors are, e.g. 'who are the cardiologists at Riyadh'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "branch": {"type": "string", "description": "Branch name if mentioned."},
                    "specialty": {"type": "string", "description": "Medical specialty if mentioned, e.g. 'Cardiology'."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Resolve and confirm a booking request against the real hospital dataset. Use whenever the user expresses intent to book/schedule an appointment, with whatever doctor name, specialty and/or branch can be extracted from the conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_name": {"type": "string", "description": "Doctor's name as mentioned by the user, if any."},
                    "specialty": {"type": "string", "description": "Medical specialty as mentioned by the user, if any."},
                    "branch": {"type": "string", "description": "Branch/city as mentioned by the user, if any."},
                },
                "required": [],
            },
        },
    },
]


class GroqHospitalClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        group_name_en: str = "Al Raya Medical Group",
        group_name_ar: str = "مجموعة الراية الطبية",
    ):
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
                "and set it as an environment variable."
            )
        self.client = Groq(api_key=key)
        self.system_prompt = SYSTEM_PROMPT.format(group_name=group_name_en, group_name_ar=group_name_ar)

    def next_turn(
        self,
        history: List[Dict],
        user_message: str,
        knowledge_context: str = "",
    ) -> Dict:
        """
        Sends one conversational turn to the model.
        Returns either:
          {"type": "medical_answer", "answer": "...", "user_msg": {...}, "assistant_msg": {...}}
        or:
          {"type": "action", "tool_name": "...", "tool_input": {...}, "tool_call_id": "...",
           "user_msg": {...}, "assistant_msg": {...}}
        """
        turn_context = ""
        if knowledge_context:
            turn_context = f"\n\n[Relevant medical knowledge for this turn]\n{knowledge_context}\n"

        user_msg = {"role": "user", "content": user_message + turn_context}
        messages = [{"role": "system", "content": self.system_prompt}] + list(history) + [user_msg]

        response = self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=800,
            temperature=0.3,
        )

        choice = response.choices[0].message

        if choice.tool_calls:
            tool_call = choice.tool_calls[0]
            try:
                tool_input = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                tool_input = {}
            if not isinstance(tool_input, dict):
                # some models return the literal string "null" (valid JSON -> None)
                # when they mean "no arguments" - normalize that to an empty dict.
                tool_input = {}

            assistant_msg = {
                "role": "assistant",
                "content": choice.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            }

            return {
                "type": "action",
                "tool_name": tool_call.function.name,
                "tool_input": tool_input,
                "tool_call_id": tool_call.id,
                "user_msg": user_msg,
                "assistant_msg": assistant_msg,
            }
        else:
            return {
                "type": "medical_answer",
                "answer": choice.content or "",
                "user_msg": user_msg,
                "assistant_msg": {"role": "assistant", "content": choice.content or ""},
            }
