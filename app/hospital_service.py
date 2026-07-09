"""
Hospital directory service: answers structured questions about branches,
specializations and doctors, and resolves booking requests against the
real (generated) hospital dataset using fuzzy matching so that user input
like "Dr. Sarah" or "دكتورة سارة" still matches "Dr. Sarah Hassan".
"""
import json
import difflib
import re
from pathlib import Path
from typing import Optional, List, Dict


def _normalize(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[\u064B-\u0652]", "", text)  # strip Arabic diacritics
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"^(dr\.?|doctor|د\.?|دكتور[ة]?)\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


class HospitalService:
    def __init__(self, data_path: str):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.branches = self.data["branches"]
        self.doctors = self.data["doctors"]
        self.group = self.data["hospital_group"]

    # ---------- lookups ----------
    def list_branches(self) -> List[Dict]:
        return [
            {"id": b["id"], "name_en": b["name_en"], "name_ar": b["name_ar"]}
            for b in self.branches
        ]

    def _find_branch(self, branch_query: Optional[str]) -> Optional[Dict]:
        if not branch_query:
            return None
        nq = _normalize(branch_query)
        for b in self.branches:
            if nq == _normalize(b["name_en"]) or nq == _normalize(b["name_ar"]) or nq == b["id"]:
                return b
        # fuzzy fallback
        candidates = {b["id"]: _normalize(b["name_en"]) for b in self.branches}
        match = difflib.get_close_matches(nq, candidates.values(), n=1, cutoff=0.6)
        if match:
            for b in self.branches:
                if _normalize(b["name_en"]) == match[0]:
                    return b
        return None

    def list_specializations(self, branch_query: Optional[str] = None) -> Dict:
        branch = self._find_branch(branch_query)
        if branch:
            return {
                "branch": branch["name_en"],
                "branch_ar": branch["name_ar"],
                "specializations": branch["specializations"],
            }
        # no branch matched / not specified -> union across all branches
        all_specs = sorted({s for b in self.branches for s in b["specializations"]})
        return {"branch": None, "specializations": all_specs}

    def list_doctors(
        self, branch_query: Optional[str] = None, specialty_query: Optional[str] = None
    ) -> Dict:
        branch = self._find_branch(branch_query)
        results = self.doctors
        if branch:
            results = [d for d in results if d["branch_id"] == branch["id"]]

        if specialty_query:
            nq = _normalize(specialty_query)
            filtered = [
                d for d in results
                if nq in _normalize(d["specialty_en"]) or nq in _normalize(d["specialty_ar"])
            ]
            if not filtered:
                # fuzzy specialty match
                specs = {d["specialty_en"] for d in results}
                match = difflib.get_close_matches(nq, [ _normalize(s) for s in specs], n=1, cutoff=0.5)
                if match:
                    filtered = [d for d in results if _normalize(d["specialty_en"]) == match[0]]
            results = filtered

        return {
            "branch": branch["name_en"] if branch else None,
            "branch_ar": branch["name_ar"] if branch else None,
            "specialty": specialty_query,
            "doctors": [
                {
                    "name": d["name_en"],
                    "name_ar": d["name_ar"],
                    "specialty": d["specialty_en"],
                    "specialty_ar": d["specialty_ar"],
                    "branch": self._branch_name(d["branch_id"]),
                    "branch_ar": self._branch_name_ar(d["branch_id"]),
                }
                for d in results
            ],
        }

    def _branch_name(self, branch_id: str) -> str:
        for b in self.branches:
            if b["id"] == branch_id:
                return b["name_en"]
        return branch_id

    def _branch_name_ar(self, branch_id: str) -> str:
        for b in self.branches:
            if b["id"] == branch_id:
                return b["name_ar"]
        return branch_id

    def _branch_by_id(self, branch_id: str) -> Dict:
        for b in self.branches:
            if b["id"] == branch_id:
                return b
        return {}

    # ---------- booking resolution ----------
    def resolve_booking(
        self,
        doctor_name: Optional[str] = None,
        specialty: Optional[str] = None,
        branch: Optional[str] = None,
    ) -> Dict:
        """
        Try to match the extracted booking intent against real hospital data.
        Returns either a confirmed match or a clarification request with
        candidate options, so the caller always has enough structure to act on.
        """
        branch_obj = self._find_branch(branch)
        candidates = self.doctors
        if branch_obj:
            candidates = [d for d in candidates if d["branch_id"] == branch_obj["id"]]

        if doctor_name:
            nq = _normalize(doctor_name)
            name_scores = []
            for d in candidates:
                score = difflib.SequenceMatcher(None, nq, _normalize(d["name_en"])).ratio()
                name_scores.append((score, d))
            name_scores.sort(key=lambda x: x[0], reverse=True)
            if name_scores and name_scores[0][0] >= 0.45:
                best = name_scores[0][1]
                return {
                    "status": "confirmed",
                    "doctor_name": best["name_en"],
                    "specialty": best["specialty_en"],
                    "branch": self._branch_name(best["branch_id"]),
                    "hospital": self._branch_by_id(best["branch_id"]).get("hospital_label_en"),
                }

        if specialty:
            nq = _normalize(specialty)
            matched = [
                d for d in candidates
                if nq in _normalize(d["specialty_en"]) or _normalize(d["specialty_en"]) in nq
            ]
            if not matched:
                specs = list({_normalize(d["specialty_en"]): d["specialty_en"] for d in candidates}.keys())
                close = difflib.get_close_matches(nq, specs, n=1, cutoff=0.5)
                if close:
                    matched = [d for d in candidates if _normalize(d["specialty_en"]) == close[0]]

            if len(matched) == 1:
                d = matched[0]
                return {
                    "status": "confirmed",
                    "doctor_name": d["name_en"],
                    "specialty": d["specialty_en"],
                    "branch": self._branch_name(d["branch_id"]),
                    "hospital": self._branch_by_id(d["branch_id"]).get("hospital_label_en"),
                }
            elif len(matched) > 1:
                return {
                    "status": "needs_clarification",
                    "reason": "multiple_doctors_match",
                    "options": [
                        {
                            "doctor_name": d["name_en"],
                            "specialty": d["specialty_en"],
                            "branch": self._branch_name(d["branch_id"]),
                        }
                        for d in matched
                    ],
                }

        return {
            "status": "needs_clarification",
            "reason": "insufficient_or_no_match",
            "options": [
                {
                    "doctor_name": d["name_en"],
                    "specialty": d["specialty_en"],
                    "branch": self._branch_name(d["branch_id"]),
                }
                for d in candidates[:5]
            ],
        }


if __name__ == "__main__":
    svc = HospitalService(str(Path(__file__).resolve().parent.parent / "data" / "hospital_data.json"))
    print(svc.list_specializations("Alexandria"))
    print(svc.list_doctors(branch_query="Riyadh", specialty_query="Cardiology"))
    print(svc.resolve_booking(doctor_name="Dr. Sarah", specialty="Neurology", branch="Cairo"))
