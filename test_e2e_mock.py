"""
End-to-end smoke test that mocks the Groq LLM call so we can verify the ENTIRE
pipeline (retrieval -> tool execution -> hospital data matching -> response
shaping -> multi-turn history) works correctly without needing real network
access or a real API key.

Run: python3 test_e2e_mock.py
"""
import sys
import os
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(__file__))
os.environ["GROQ_API_KEY"] = "dummy-key-for-mock-test"


def make_text_response(text):
    msg = MagicMock()
    msg.content = text
    msg.tool_calls = None
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


def make_tool_call_response(tool_name, tool_args: dict):
    tool_call = MagicMock()
    tool_call.id = "call_123"
    tool_call.function.name = tool_name
    tool_call.function.arguments = json.dumps(tool_args)

    msg = MagicMock()
    msg.content = ""
    msg.tool_calls = [tool_call]

    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


def run():
    from app.chatbot import HospitalChatbot

    bot = HospitalChatbot(api_key="dummy-key-for-mock-test")

    failures = []

    def check(name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name} {detail}")
        if not condition:
            failures.append(name)

    # ---- Test 1: medical question -> plain answer, no tool ----
    fake_answer = ("الصداع الشديد المصحوب بالحمى قد يشير إلى عدوى. "
                   "يُنصح بالراحة. هل تودّ حجز موعد مع أحد أطبائنا؟")
    with patch.object(bot.llm.client.chat.completions, "create",
                       return_value=make_text_response(fake_answer)):
        result = bot.handle_message("test1", "severe headache and fever for two days")

    check("medical answer returns 'answer' key only", "answer" in result and "action" not in result, str(result)[:80])
    check("medical answer text matches model output", result.get("answer") == fake_answer)
    check("history recorded 2 messages", len(bot.sessions["test1"]) == 2, str(len(bot.sessions["test1"])))

    # ---- Test 2: list_doctors action ----
    with patch.object(bot.llm.client.chat.completions, "create",
                       return_value=make_tool_call_response(
                           "list_doctors", {"branch": "Riyadh", "specialty": "Cardiology"})):
        result = bot.handle_message("test2", "Who are the cardiologists at the Riyadh branch?")

    check("list_doctors returns action field", result.get("action") == "list_doctors", str(result))
    check("list_doctors branch correct", result.get("branch") == "Riyadh")
    check("list_doctors specialty correct", result.get("specialty") == "Cardiology")
    doctor_names = sorted(d["name"] for d in result.get("doctors", []))
    check("list_doctors returns exactly 2 real doctors",
          doctor_names == ["Dr. Khalid Al-Mansouri", "Dr. Layla Nasser"], str(doctor_names))
    check("no mixing: no 'answer' key present in action response", "answer" not in result)

    # ---- Test 3: list_specializations action ----
    with patch.object(bot.llm.client.chat.completions, "create",
                       return_value=make_tool_call_response(
                           "list_specializations", {"branch": "Alexandria"})):
        result = bot.handle_message("test3", "What specializations are available at the Alexandria branch?")

    check("list_specializations action correct", result.get("action") == "list_specializations")
    check("list_specializations branch correct", result.get("branch") == "Alexandria")
    check("list_specializations has expected specialties",
          set(result.get("specializations", [])) == {"Cardiology", "Neurology", "Orthopedics", "Dermatology"},
          str(result.get("specializations")))

    # ---- Test 4: multi-turn booking flow (medical -> booking) ----
    fake_answer2 = "قد يشير هذا إلى مشكلة عصبية. هل تودّ حجز موعد؟"
    with patch.object(bot.llm.client.chat.completions, "create",
                       return_value=make_text_response(fake_answer2)):
        r1 = bot.handle_message("test4", "I've been having a severe headache and fever for two days")

    with patch.object(bot.llm.client.chat.completions, "create",
                       return_value=make_tool_call_response(
                           "book_appointment",
                           {"doctor_name": "Dr. Sarah", "specialty": "Neurology", "branch": "Cairo"})):
        r2 = bot.handle_message("test4", "Yes, I want to book with Dr. Sarah in Neurology at the Cairo branch")

    check("booking confirmed with real doctor", r2.get("doctor_name") == "Dr. Sarah Hassan", str(r2))
    check("booking specialty correct", r2.get("specialty") == "Neurology")
    check("booking branch correct", r2.get("branch") == "Cairo")
    check("booking hospital label correct",
          r2.get("hospital") == "Al Raya Medical Group - Cairo Branch", str(r2.get("hospital")))
    check("multi-turn history accumulated correctly (2 + 3 = 5 messages)",
          len(bot.sessions["test4"]) == 5, str(len(bot.sessions["test4"])))

    # ---- Test 5: list_branches action ----
    with patch.object(bot.llm.client.chat.completions, "create",
                       return_value=make_tool_call_response("list_branches", {})):
        result = bot.handle_message("test5", "What branches do you have?")

    check("list_branches action correct", result.get("action") == "list_branches")
    check("list_branches returns 4 branches", len(result.get("branches", [])) == 4, str(result.get("branches")))

    # ---- Test 6: booking with ambiguous/no match -> clarification ----
    with patch.object(bot.llm.client.chat.completions, "create",
                       return_value=make_tool_call_response(
                           "book_appointment", {"specialty": "Cardiology", "branch": "Riyadh"})):
        result = bot.handle_message("test6", "I want to book a cardiologist in Riyadh")

    check("ambiguous booking asks for clarification",
          result.get("status") == "needs_clarification", str(result))
    check("clarification includes real doctor options",
          len(result.get("options", [])) == 2, str(result.get("options")))

    # ---- Test 7: regression - model returns arguments as literal "null" ----
    tool_call_null = MagicMock()
    tool_call_null.id = "call_null"
    tool_call_null.function.name = "list_specializations"
    tool_call_null.function.arguments = "null"  # valid JSON that decodes to Python None
    msg_null = MagicMock()
    msg_null.content = ""
    msg_null.tool_calls = [tool_call_null]
    resp_null = MagicMock()
    resp_null.choices = [MagicMock(message=msg_null)]

    with patch.object(bot.llm.client.chat.completions, "create", return_value=resp_null):
        result = bot.handle_message("test7", "What specializations do you have?")

    check("'null' arguments string doesn't crash, treated as no filters",
          result.get("action") == "list_specializations" and "specializations" in result,
          str(result))

    # ---- Test 8: natural-language 'message' field accompanies action responses ----
    with patch.object(bot.llm.client.chat.completions, "create",
                       return_value=make_tool_call_response("list_doctors", {"specialty": "Cardiology"})):
        result = bot.handle_message("test8en", "Who is the cardiologist?")

    check("EN list_doctors has readable 'message' field",
          "message" in result and result["message"].startswith("Here are the"), result.get("message"))
    check("structured JSON still present alongside message",
          result.get("action") == "list_doctors" and "doctors" in result)

    with patch.object(bot.llm.client.chat.completions, "create",
                       return_value=make_tool_call_response("list_doctors", {"specialty": "القلب"})):
        result_ar = bot.handle_message("test8ar", "مين دكتور القلب المتاح؟")

    check("AR list_doctors message uses Arabic branch names",
          "القاهرة" in result_ar["message"] and "Cairo" not in result_ar["message"],
          result_ar.get("message"))

    print("\n" + "=" * 50)
    if failures:
        print(f"RESULT: {len(failures)} FAILURE(S): {failures}")
        sys.exit(1)
    else:
        print("RESULT: ALL TESTS PASSED ✅")


if __name__ == "__main__":
    run()
