from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import json
import math

from src.grok_client import GrokClient

JUDGE_SYSTEM = (
    "You are a strict customer-satisfaction adjudicator for customer service chats. "
    "Rate the user’s satisfaction at the END of the provided conversation. "
    "Use only the conversation turns; do not invent facts. "
    "Output ONLY valid JSON with keys:\n"
    '{\n'
    '  "satisfaction": "satisfied" | "neutral" | "dissatisfied",\n'
    '  "reason": "short explanation (<= 2 sentences)",\n'
    '  "snippets": [\n'
    '     {"turn": <int zero_based_turn_index>, "role": "user|assistant", "quote": "<short excerpt>"}\n'
    '  ]\n'
    '}\n'
    "Guidance:\n"
    "- satisfied: goal resolved or clearly helpful next step acknowledged, positive closing, thanks, or relief.\n"
    "- neutral: partial progress, info provided without resolution, user noncommittal, or escalated appropriately without clear sentiment.\n"
    "- dissatisfied: explicit negative sentiment, unresolved blockers, repeated refusals without helpful alternatives, user frustration, or policy-only responses that leave user stuck.\n"
    "Choose at most 3 snippets that best justify your rating."
)

def score_conversation(
    judge: GrokClient,
    judge_model: str,
    turns: List[Dict[str, str]],
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    turns: list of {"role": "user|assistant", "content": "..."} in chronological order
    Returns: dict with satisfaction label, reason, snippets, plus raw judge info
    """

    out = judge.chat(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": json.dumps({"conversation": turns}, ensure_ascii=False)},
        ],
        temperature=0.0,
        seed=42,
        extra={"metric": "customer_satisfaction", **(meta or {})},
    )

    raw_content = (out.get("content") or "").strip()
    label, reason, snippets = "neutral", "fallback neutral (parse error)", []

    try:
        payload = json.loads(raw_content)
        lab = str(payload.get("satisfaction", "")).lower().strip()
        if lab in {"satisfied", "neutral", "dissatisfied"}:
            label = lab
        reason = str(payload.get("reason", "")).strip()[:300]
        sn = payload.get("snippets") or []
        # sanitize snippets
        clean_sn: List[Dict[str, Any]] = []
        for s in sn[:3]:
            try:
                clean_sn.append({
                    "turn": int(s.get("turn", 0)),
                    "role": "assistant" if str(s.get("role", "")).lower().startswith("assist") else "user",
                    "quote": str(s.get("quote", ""))[:200],
                })
            except Exception:
                continue
        snippets = clean_sn
    except Exception:
        pass

    return {
        "satisfaction": label,
        "reason": reason,
        "snippets": snippets,
        "judge_raw": {
            "content": raw_content,
            "latency_ms": out.get("latency_ms"),
            "usage": out.get("usage"),
            "model": judge_model,
        },
        **(meta or {}),
    }

def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a small dashboard of satisfaction outcomes.
    Also compute a simple Satisfaction Index: +1 for satisfied, 0 for neutral, -1 for dissatisfied.
    """
    n = len(rows) or 1
    counts = {"satisfied": 0, "neutral": 0, "dissatisfied": 0}
    index_sum = 0
    for r in rows:
        s = str(r.get("satisfaction", "neutral")).lower()
        if s not in counts:
            s = "neutral"
        counts[s] += 1
        index_sum += 1 if s == "satisfied" else (-1 if s == "dissatisfied" else 0)

    return {
        "num_conversations": len(rows),
        "rate_satisfied": round(counts["satisfied"] / n, 4),
        "rate_neutral": round(counts["neutral"] / n, 4),
        "rate_dissatisfied": round(counts["dissatisfied"] / n, 4),
        "satisfaction_index": round(index_sum / n, 4),  # in [-1, 1]
        "counts": counts,
    }
