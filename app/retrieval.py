"""
Lightweight retrieval layer over the hospital's medical knowledge base.

Design choice:
We use TF-IDF (scikit-learn) instead of a neural embedding API for two reasons:
1. It runs fully offline / locally -> no extra API key, no network dependency,
   works the same on a laptop or in Google Colab.
2. The knowledge base is small (curated symptom/condition entries), so TF-IDF
   keyword-level matching is accurate enough and fully deterministic/explainable,
   which matters for a medical-adjacent use case.

If you later plug in a real, larger medical corpus, swap this class's internals
for a sentence-transformers / vector DB (e.g. FAISS) implementation without
changing its public interface (`search`).
"""
import json
import re
from pathlib import Path
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MedicalKnowledgeRetriever:
    def __init__(self, knowledge_path: str):
        self.knowledge_path = Path(knowledge_path)
        with open(self.knowledge_path, "r", encoding="utf-8") as f:
            self.entries: List[Dict] = json.load(f)

        # Build a combined bilingual text blob per entry for matching,
        # regardless of whether the user typed Arabic or English.
        self._corpus = [self._entry_to_text(e) for e in self.entries]

        # word-level + character n-gram fallback helps a bit with Arabic,
        # which doesn't tokenize as cleanly as English with the default pattern.
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
        )
        self._matrix = self.vectorizer.fit_transform(self._corpus)

    @staticmethod
    def _entry_to_text(entry: Dict) -> str:
        parts = [
            entry.get("keywords_en", ""),
            entry.get("keywords_ar", ""),
            entry.get("condition_en", ""),
            entry.get("condition_ar", ""),
        ]
        return " ".join(parts)

    @staticmethod
    def _normalize_arabic(text: str) -> str:
        # Basic Arabic normalization: unify alef/yeh forms, strip diacritics.
        text = re.sub(r"[\u064B-\u0652]", "", text)  # remove tashkeel
        text = re.sub(r"[إأآا]", "ا", text)
        text = re.sub(r"ى", "ي", text)
        text = re.sub(r"ة", "ه", text)
        return text

    def search(self, query: str, top_k: int = 3, min_score: float = 0.05) -> List[Dict]:
        """Return the top_k most relevant knowledge entries for a free-text query."""
        norm_query = self._normalize_arabic(query)
        q_vec = self.vectorizer.transform([norm_query])
        scores = cosine_similarity(q_vec, self._matrix).flatten()

        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        results = []
        for idx in ranked[:top_k]:
            if scores[idx] >= min_score:
                entry = dict(self.entries[idx])
                entry["_score"] = float(scores[idx])
                results.append(entry)
        return results


if __name__ == "__main__":
    # quick manual smoke test
    r = MedicalKnowledgeRetriever(
        str(Path(__file__).resolve().parent.parent / "data" / "medical_knowledge.json")
    )
    for q in [
        "severe headache and fever for two days",
        "عندي ألم في الصدر وضيق في التنفس",
        "my knee is swollen and stiff",
    ]:
        print("Q:", q)
        for e in r.search(q, top_k=2):
            print("  ->", e["id"], round(e["_score"], 3), e["condition_en"])
        print()
