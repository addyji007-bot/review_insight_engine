"""Lightweight, explainable business-aspect detection for customer feedback.

spaCy's NER only finds *named* things (people, organisations, places, dates).
Most customer reviews mention none of those, so the entity column is usually
empty. This module finds the business aspects a customer talks about
(delivery, price, support, app, ...) with a keyword lexicon. It is the first
building block of the Aspect-Based Sentiment Analysis (ABSA) extension.
"""

from __future__ import annotations

import re

ASPECT_KEYWORDS: dict[str, list[str]] = {
    "delivery": [
        "delivery", "deliveries", "delivered", "deliver", "shipping", "shipment", "shipped",
        "arrived", "packaging", "package", "courier", "dispatch", "delayed", "delay",
    ],
    "price": [
        "price", "prices", "pricing", "cost", "costs", "expensive", "cheap", "overpriced",
        "affordable", "value for money", "worth", "discount",
    ],
    "product quality": [
        "quality", "material", "durability", "durable", "sturdy", "defective", "damaged",
        "faulty", "broke", "broken",
    ],
    "battery": ["battery", "battery life", "charging", "charge", "charger"],
    "screen": ["screen", "display", "resolution", "brightness"],
    "sound": ["sound", "speaker", "speakers", "volume", "audio"],
    "customer support": [
        "customer support", "customer service", "customer care", "support", "service",
        "agent", "representative", "helpline", "complaint", "response time", "response",
    ],
    "staff": ["staff", "employee", "employees", "manager"],
    "app / website": [
        "mobile app", "app", "apps", "application", "website", "site", "login", "interface",
        "checkout", "payment", "payments", "navigation",
    ],
    "refund / return": [
        "refund", "refunds", "return", "returned", "returns", "replacement", "exchange",
        "money back", "cancellation", "cancel",
    ],
    "process": ["process", "procedure", "wait time", "waiting time", "waiting", "queue"],
    "ease of use": ["easy to use", "user friendly", "user-friendly", "setup", "set up"],
}

# keyword -> aspect (each keyword belongs to exactly one aspect)
_KEYWORD_TO_ASPECT: dict[str, str] = {}
for _aspect, _words in ASPECT_KEYWORDS.items():
    for _word in _words:
        _KEYWORD_TO_ASPECT.setdefault(_word.lower(), _aspect)

# One alternation, longest keyword first, so "customer support" wins over "support".
_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in sorted(_KEYWORD_TO_ASPECT, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)


def find_aspect_mentions(text: str) -> list[dict]:
    """Return non-overlapping aspect mentions with character offsets.

    Each item: {"text", "label", "start", "end", "aspect"}.
    """
    text = str(text)
    mentions: list[dict] = []
    for match in _PATTERN.finditer(text):
        aspect = _KEYWORD_TO_ASPECT[match.group().lower()]
        mentions.append(
            {
                "text": match.group(),
                "label": aspect,
                "aspect": aspect,
                "start": match.start(),
                "end": match.end(),
            }
        )
    return mentions


def aspect_names(text: str) -> list[str]:
    """Unique aspect names mentioned in a text, in order of first appearance."""
    seen: list[str] = []
    for mention in find_aspect_mentions(text):
        if mention["aspect"] not in seen:
            seen.append(mention["aspect"])
    return seen
