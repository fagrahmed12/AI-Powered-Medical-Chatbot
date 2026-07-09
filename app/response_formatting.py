"""
Converts structured action results (from HospitalService) into natural,
conversational bilingual text - so the chat experience always reads like a
normal reply, even for action turns (list doctors / specializations /
branches / book appointment), while the underlying structured JSON is still
available separately for anything that needs to consume exact data (e.g. a
real booking system, see main.py's /chat response).
"""
import re
from typing import Dict, List

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def detect_lang(text: str) -> str:
    """Very small heuristic: any Arabic script character -> 'ar', else 'en'."""
    return "ar" if _ARABIC_RE.search(text or "") else "en"


def format_list_branches(tool_result: Dict, lang: str) -> str:
    branches = tool_result.get("branches", [])
    names_en = [b["name_en"] for b in branches]
    if lang == "ar":
        names_ar = [b["name_ar"] for b in branches]
        listing = "، ".join(names_ar)
        return f"لدينا فروع في: {listing}. تحب أعرض لك الأطباء أو التخصصات المتاحة في فرع معيّن؟"
    listing = ", ".join(names_en)
    return f"We have branches in: {listing}. Would you like to see the doctors or specializations available at a specific branch?"


def format_list_specializations(tool_result: Dict, lang: str) -> str:
    specs = tool_result.get("specializations", [])
    branch = tool_result.get("branch")
    branch_ar = tool_result.get("branch_ar")
    listing = "، ".join(specs) if lang == "ar" else ", ".join(specs)

    if lang == "ar":
        where = f" في فرع {branch_ar or branch}" if branch else ""
        return f"التخصصات المتاحة{where} هي: {listing}. حابب تعرف الأطباء المتاحين في أي تخصص منهم؟"
    where = f" at the {branch} branch" if branch else ""
    return f"The available specializations{where} are: {listing}. Would you like to see the doctors available in any of these?"


def format_list_doctors(tool_result: Dict, lang: str) -> str:
    doctors: List[Dict] = tool_result.get("doctors", [])
    specialty = tool_result.get("specialty")
    branch = tool_result.get("branch")

    if not doctors:
        if lang == "ar":
            return "للأسف مفيش أطباء متاحين بالمواصفات دي حاليًا. ممكن تجرب تخصص أو فرع تاني؟"
        return "Unfortunately there are no doctors matching that right now. Would you like to try a different specialty or branch?"

    if lang == "ar":
        header_bits = []
        if specialty:
            header_bits.append(f"تخصص {specialty}")
        if branch:
            header_bits.append(f"في فرع {branch}")
        header = " ".join(header_bits) if header_bits else "الأطباء المتاحين"
        lines = "\n".join(
            f"• {d.get('name_ar', d['name'])} — {d.get('branch_ar') or d.get('branch', '')}".rstrip(" —")
            for d in doctors
        )
        return f"إليك أطباء {header}:\n{lines}\nحابب تحجز موعد مع أي منهم؟"

    header_bits = []
    if specialty:
        header_bits.append(specialty)
    if branch:
        header_bits.append(f"at {branch}")
    header = " ".join(header_bits) if header_bits else "the doctors we have"
    lines = "\n".join(f"• {d['name']}" + (f" — {d['branch']}" if d.get("branch") else "") for d in doctors)
    return f"Here are the {header} doctors:\n{lines}\nWould you like to book an appointment with one of them?"


def format_book_appointment(tool_result: Dict, lang: str) -> str:
    if tool_result.get("status") == "confirmed":
        doctor = tool_result["doctor_name"]
        specialty = tool_result["specialty"]
        branch = tool_result["branch"]
        hospital = tool_result["hospital"]
        if lang == "ar":
            return (f"تمام! لقيتلك {doctor}، تخصص {specialty}، في فرع {branch} ({hospital}). "
                     f"هل تؤكد الحجز مع الدكتور ده؟")
        return (f"Great! I found {doctor}, specializing in {specialty}, at the {branch} branch "
                f"({hospital}). Shall I confirm this booking for you?")

    options = tool_result.get("options", [])
    if not options:
        if lang == "ar":
            return "معنديش معلومات كافية لتأكيد الحجز. ممكن تقولي اسم الدكتور أو التخصص أو الفرع اللي تحب تحجز فيه؟"
        return "I don't have enough information to confirm a booking yet. Could you tell me the doctor's name, specialty, or branch you'd like?"

    if lang == "ar":
        lines = "\n".join(f"• {o['doctor_name']} — {o['specialty']} — {o['branch']}" for o in options)
        return f"لقيت أكتر من خيار مطابق، ممكن تحدد أنهي واحد تحب تحجز معاه؟\n{lines}"
    lines = "\n".join(f"• {o['doctor_name']} — {o['specialty']} — {o['branch']}" for o in options)
    return f"I found a few matching options — which one would you like to book?\n{lines}"


_FORMATTERS = {
    "list_branches": format_list_branches,
    "list_specializations": format_list_specializations,
    "list_doctors": format_list_doctors,
    "book_appointment": format_book_appointment,
}


def format_action_as_text(tool_name: str, tool_result: Dict, user_message: str) -> str:
    lang = detect_lang(user_message)
    formatter = _FORMATTERS.get(tool_name)
    if formatter is None:
        return str(tool_result)
    return formatter(tool_result, lang)
