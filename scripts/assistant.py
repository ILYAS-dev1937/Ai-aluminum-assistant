"""assistant.py — Claude optimized with strict grounding."""

import os
import re
from typing import List, Dict, Any

from openai import OpenAI

from config import OLLAMA_MODEL, OLLAMA_BASE_URL


def detect_language(text: str) -> str:
    """Detect if text is Arabic, French, or English."""

    if any("؀" <= c <= "ۿ" for c in text):
        return "Arabic"

    text_lower = text.lower()

    french_words = [
        "le", "la", "les", "est", "quelle", "quel",
        "comment", "puis-je", "fenêtre", "profil",
        "dormant", "ouvrant", "coulissant",
        "vitres", "vitrage", "galet",
        "serrure", "traverses", "montants", "rail"
    ]

    if any(w in text_lower for w in french_words):
        return "French"

    return "English"


def build_prompt(question: str, lang: str, context: str) -> str:
    """
    Claude system prompt.
    Strict catalog grounding.
    """

    return f"""
You are MAVAL Technical Assistant.

You ONLY answer using the catalog excerpts provided below.

STRICT RULES:
- Never use outside knowledge.
- Never guess.
- Never invent specifications.
- If information is missing, say:
"I don't have that information in the catalog."

LANGUAGE RULE:
The user's language is {lang}.
You MUST answer ONLY in {lang}.
This rule has priority over the catalog language.
Never answer in French unless the user asked in French.

TECHNICAL RULES:
- For compatibility questions, answer YES or NO first.
- Never mention thermal breaks, Low-E, argon, U-values,
  or other technical concepts unless they appear in the catalog.
- Always cite the catalog page number:
  Example: (page 12)

CATALOG EXCERPTS:

{context}

CUSTOMER QUESTION:

{question}
"""


class MavalAssistant:

    def __init__(self):

        self.client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",
            timeout=120
        )


    def _format_context(
        self,
        chunks: List[Dict[str, Any]]
    ) -> str:

        parts = []

        for i, chunk in enumerate(chunks, 1):

            meta = chunk.get(
                "metadata",
                {}
            )

            title = meta.get(
                "title",
                chunk.get("id", "Unknown")
            )

            page = meta.get(
                "source_page",
                "?"
            )

            parts.append(
                f"""
--- EXCERPT {i} ---
Title: {title}
Page: {page}

{chunk['content']}
"""
            )

        return "\n".join(parts)


    def answer(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        temperature: float = 0.0
    ) -> Dict[str, Any]:


        lang = detect_language(question)

        context = self._format_context(
            chunks
        )

        prompt = build_prompt(
            question,
            lang,
            context
        )


        try:
            response = self.client.chat.completions.create(

                model=OLLAMA_MODEL,

                temperature=temperature,

                max_tokens=600,

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
        except Exception as e:
            print("OLLAMA ERROR:", e)
            raise


        answer_text = response.choices[0].message.content


        # Safety check
        if (
            not re.search(
                r"page\s*\d+",
                answer_text,
                re.IGNORECASE
            )
            and chunks
        ):
            answer_text = (
                "[Warning: No catalog citation detected]\n"
                + answer_text
            )


        return {

            "answer": answer_text,

            "sources": [
                {
                    "id": c["id"],
                    "title": c.get(
                        "metadata",
                        {}
                    ).get(
                        "title",
                        ""
                    ),

                    "page": c.get(
                        "metadata",
                        {}
                    ).get(
                        "source_page",
                        ""
                    ),

                    "category": c.get(
                        "metadata",
                        {}
                    ).get(
                        "category",
                        ""
                    ),
                }

                for c in chunks
            ],

            "model": OLLAMA_MODEL,

            "language": lang
        }



if __name__ == "__main__":

    from retriever import MavalRetriever


    assistant = MavalAssistant()

    retriever = MavalRetriever()


    for q in [
        "What is the inertia of profile 998?",
        "Quel est le moment d'inertie du profil 998?"
    ]:

        print(
            f"\nQ: {q}"
        )

        chunks = retriever.search(q)

        result = assistant.answer(
            q,
            chunks
        )

        print(
            f"Lang: {result['language']}"
        )

        print(
            result["answer"][:500]
        )