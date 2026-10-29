"""Pre-retrieval query contextualization and military domain lexicon expansion."""

from __future__ import annotations

import re
from typing import Any

from navy_rag.config import settings

# Maps applicant colloquialisms and domain keywords to authoritative directive identifiers
# so small policy manuals are not crowded out by the 6,200-chunk NEOCS occupational manuals.
_DOMAIN_EXPANSIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(tattoo\w*|neck\s+tattoo|sleeve\w*|ring\s+tattoo|grooming|hair\w*|beard\w*|"
            r"no[\-\s]shave|shave\s+chit|ponytail\w*|braid\w*|bun\w*|sideburn\w*|"
            r"ear\s+gauge\w*|earlobe\w*|piercing\w*|mutilation)\b",
            re.IGNORECASE,
        ),
        "NAVPERS 15665J Ch. 2 Uniform Regulations Grooming Standards",
    ),
    (
        re.compile(
            r"\b(asthma|adhd|adderall|prk|lasik|refractive|depression|antidepressant\w*|"
            r"blood\s+pressure|hypertension|eczema|dermatitis|orthopedic|hardware|"
            r"color\s+vision|color\s+blind\w*|falant|audiometer|hearing\s+loss|"
            r"meps\s+medical|meps\s+physical|accession\s+physical|bumed)\b",
            re.IGNORECASE,
        ),
        "DoDI 6130.03 Vol. 1 MANMED Ch. 15 Medical Standards Physical Examinations",
    ),
    (
        re.compile(
            r"\b(prt|physical\s+readiness|body\s+fat|bca|body\s+composition|abdominal|"
            r"tape\s+test|push[\-\s]?up\w*|plank\w*|alternate\s+cardio|stationary\s+bike|"
            r"postpartum|navmed\s+6110)\b",
            re.IGNORECASE,
        ),
        "OPNAVINST 6110.1K PRT Guide 5A Physical Readiness Program",
    ),
    (
        re.compile(
            r"\b(marijuana|cannabis|thc|urinalysis|drug\s+test|domestic\s+violence|"
            r"lautenberg|probation|open\s+charge\w*|dui|dwi|drunk\s+driving|traffic|"
            r"bankruptcy|expunged|sealed\s+record|juvenile\s+felony|misdemeanor|"
            r"moral\s+waiver|criminal)\b",
            re.IGNORECASE,
        ),
        "COMNAVCRUITCOMINST 1130.8 DoDI 1304.26 Enlistment Qualification Moral Waiver",
    ),
    (
        re.compile(
            r"\b(tier\s+1|tier\s+2|ged|high\s+school\s+diploma|afqt|asvab\s+retest|"
            r"asvab\s+score\s+valid\w*|retest\s+waiting)\b",
            re.IGNORECASE,
        ),
        "DoDI 1145.01 COMNAVCRUITCOMINST 1130.8 ASVAB Enlistment Qualification",
    ),
    (
        re.compile(
            r"\b(re[\-\s]?[1234]\w*|reenlistment\s+code|prior[\-\s]service|break[\-\s]in[\-\s]service|"
            r"adjusted\s+age|other\s+military\s+branch\w*)\b",
            re.IGNORECASE,
        ),
        "MILPERSMAN 1133-020 COMNAVCRUITCOMINST 1130.8 Prior Service Enlistment",
    ),
    (
        re.compile(
            r"\b(ocs|officer\s+candidate|oar|astb\w*|student\s+naval\s+aviator|sna|"
            r"commissioning|baccalaureate)\b",
            re.IGNORECASE,
        ),
        "CNRC ORLMM COMNAVCRUITCOMINST 1131.2 Officer Recruiting Manual",
    ),
    (
        re.compile(
            r"\b(nec|navy\s+enlisted\s+classification)\b",
            re.IGNORECASE,
        ),
        "NAVPERS 18068F Vol. II NEC Manual",
    ),
    (
        re.compile(
            r"\b(occstds|occupational\s+standards)\b",
            re.IGNORECASE,
        ),
        "NAVPERS 18068F Vol. I Occupational Standards",
    ),
)

_FOLLOWUP_PRONOUN_RE = re.compile(
    r"\b(it|that|this|they|them|those|these|there|waived|waiver\s+for\s+it)\b",
    re.IGNORECASE,
)
_ELLIPTICAL_START_RE = re.compile(
    r"^\s*(what\s+about|how\s+about|and\s+what|and\s+for|what\s+if|can\s+it|is\s+it|does\s+that)\b",
    re.IGNORECASE,
)


def _extract_last_user_question(chat_history: list[Any]) -> str:
    """Return the most recent user message content from chat history."""
    for msg in reversed(chat_history):
        if isinstance(msg, dict) and msg.get("role") in ("user", "human"):
            content = str(msg.get("content", "")).strip()
            if content:
                return content
        elif getattr(msg, "type", None) == "human":
            content = str(getattr(msg, "content", "")).strip()
            if content:
                return content
    return ""


def contextualize_followup_query(
    question: str,
    chat_history: list[Any] | None = None,
) -> str:
    """Anchor pronoun-dependent or elliptical follow-up queries with prior turn context."""
    cleaned = question.strip()
    if not chat_history:
        return cleaned

    word_count = len(cleaned.split())
    is_followup = bool(_ELLIPTICAL_START_RE.search(cleaned)) or (
        word_count <= 10 and bool(_FOLLOWUP_PRONOUN_RE.search(cleaned))
    )
    if not is_followup:
        return cleaned

    prev_question = _extract_last_user_question(chat_history)
    if not prev_question or prev_question.lower() == cleaned.lower():
        return cleaned

    return f"{prev_question} — {cleaned}"


def expand_military_query(question: str) -> str:
    """Append authoritative Navy directive identifiers matching domain triggers in the query."""
    if not settings.query_expansion_enabled:
        return question.strip()

    base = question.strip()
    if not base:
        return base

    lower_base = base.lower()
    expansions: list[str] = []

    for pattern, directive_hint in _DOMAIN_EXPANSIONS:
        if pattern.search(base):
            missing_tokens = [
                tok for tok in directive_hint.split() if tok.lower() not in lower_base
            ]
            if missing_tokens:
                expansions.append(" ".join(missing_tokens))

    if not expansions:
        return base

    deduped_tokens: list[str] = []
    seen: set[str] = set()
    for phrase in expansions:
        for tok in phrase.split():
            norm = tok.lower()
            if norm not in seen:
                seen.add(norm)
                deduped_tokens.append(tok)

    return f"{base} ({' '.join(deduped_tokens)})"


def prepare_retrieval_query(
    question: str,
    chat_history: list[Any] | None = None,
) -> str:
    """Contextualize follow-up turns and apply military lexicon expansion for retrieval."""
    contextualized = contextualize_followup_query(question, chat_history)
    return expand_military_query(contextualized)
