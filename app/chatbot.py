"""
Conversation orchestrator.

Flow per user turn:
 1. Retrieve relevant medical knowledge snippets (always cheap to do; ignored
    by the model if the turn turns out to be an action request).
 2. Ask the LLM for the next turn (see llm_client.py for the medical-vs-action logic).
 3. If the model answered directly -> that's the final medical response.
 4. If the model called a tool -> execute it against HospitalService, feed the
    tool result back into history, and shape the required action response
    from the REAL data (never from model text), matching the required action
    response shape from the task spec.
 5. Update conversation history (multi-turn context, requirement #1).
"""
from pathlib import Path
from typing import Dict, List
import json

from .retrieval import MedicalKnowledgeRetriever
from .hospital_service import HospitalService
from .llm_client import GroqHospitalClient
from .response_formatting import format_action_as_text

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

ACTION_ID_MAP = {
    "book_appointment": "book_appointment",
    "list_doctors": "list_doctors",
    "list_specializations": "list_specializations",
    "list_branches": "list_branches",
}


class HospitalChatbot:
    def __init__(self, api_key: str = None):
        self.retriever = MedicalKnowledgeRetriever(str(DATA_DIR / "medical_knowledge.json"))
        self.hospital = HospitalService(str(DATA_DIR / "hospital_data.json"))
        self.llm = GroqHospitalClient(api_key=api_key)
        # session_id -> list of plain dict messages (OpenAI/Groq-style chat history)
        self.sessions: Dict[str, List[Dict]] = {}

    def _get_history(self, session_id: str) -> List[Dict]:
        return self.sessions.setdefault(session_id, [])

    def _execute_tool(self, tool_name: str, tool_input: dict) -> dict:
        tool_input = tool_input or {}  # normalize None/falsy to empty dict
        if tool_name == "list_branches":
            return {"branches": self.hospital.list_branches()}
        if tool_name == "list_specializations":
            return self.hospital.list_specializations(branch_query=tool_input.get("branch"))
        if tool_name == "list_doctors":
            return self.hospital.list_doctors(
                branch_query=tool_input.get("branch"),
                specialty_query=tool_input.get("specialty"),
            )
        if tool_name == "book_appointment":
            return self.hospital.resolve_booking(
                doctor_name=tool_input.get("doctor_name"),
                specialty=tool_input.get("specialty"),
                branch=tool_input.get("branch"),
            )
        raise ValueError(f"Unknown tool: {tool_name}")

    def _shape_action_response(self, tool_name: str, tool_result: dict) -> dict:
        """Map raw tool results onto the response shapes shown in the task's samples."""
        action_id = ACTION_ID_MAP[tool_name]

        if tool_name == "book_appointment":
            if tool_result.get("status") == "confirmed":
                return {
                    "action": action_id,
                    "doctor_name": tool_result["doctor_name"],
                    "specialty": tool_result["specialty"],
                    "branch": tool_result["branch"],
                    "hospital": tool_result["hospital"],
                }
            else:
                return {
                    "action": action_id,
                    "status": "needs_clarification",
                    "reason": tool_result.get("reason"),
                    "options": tool_result.get("options", []),
                }

        if tool_name == "list_doctors":
            return {
                "action": action_id,
                "specialty": tool_result.get("specialty"),
                "branch": tool_result.get("branch"),
                "doctors": [
                    {"name": d["name"], "specialty": d["specialty"]}
                    for d in tool_result.get("doctors", [])
                ],
            }

        if tool_name == "list_specializations":
            return {
                "action": action_id,
                "branch": tool_result.get("branch"),
                "specializations": tool_result.get("specializations", []),
            }

        if tool_name == "list_branches":
            return {
                "action": action_id,
                "branches": [b["name_en"] for b in tool_result.get("branches", [])],
            }

        return {"action": action_id, "result": tool_result}

    def handle_message(self, session_id: str, user_message: str) -> dict:
        history = self._get_history(session_id)

        knowledge_hits = self.retriever.search(user_message, top_k=3)
        knowledge_context = "\n".join(
            f"- {k['condition_en']} / {k['condition_ar']}: {k['recommendation_en']}"
            for k in knowledge_hits
        )

        turn = self.llm.next_turn(history, user_message, knowledge_context=knowledge_context)

        if turn["type"] == "medical_answer":
            history.append(turn["user_msg"])
            history.append(turn["assistant_msg"])
            return {"answer": turn["answer"]}

        # action branch: execute tool, then keep history consistent with a
        # tool-result message, while WE format the actual structured JSON
        # returned to the caller from real data (not from model text) to
        # guarantee accuracy.
        tool_result = self._execute_tool(turn["tool_name"], turn["tool_input"])

        history.append(turn["user_msg"])
        history.append(turn["assistant_msg"])
        history.append(
            {
                "role": "tool",
                "tool_call_id": turn["tool_call_id"],
                "content": json.dumps(tool_result, ensure_ascii=False),
            }
        )

        shaped = self._shape_action_response(turn["tool_name"], tool_result)
        # Attach a natural-language rendering so the chat always reads like a
        # normal reply, while the structured fields above stay available for
        # anything that needs to consume exact data (e.g. a real booking system).
        shaped["message"] = format_action_as_text(turn["tool_name"], tool_result, user_message)
        return shaped

    def reset_session(self, session_id: str):
        self.sessions.pop(session_id, None)


if __name__ == "__main__":
    import os
    bot = HospitalChatbot(api_key=os.environ.get("GROQ_API_KEY"))
    sid = "demo"
    print(bot.handle_message(sid, "I've been having a severe headache and fever for two days, what should I do?"))
    print(bot.handle_message(sid, "Yes, book me with Dr. Sarah in Neurology at the Cairo branch"))
