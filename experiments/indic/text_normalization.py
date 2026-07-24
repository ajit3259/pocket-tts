"""Conservative Hindi text normalization with auditable changes."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Literal

NumberMode = Literal["preserve", "cardinal"]

NORMALIZER_VERSION = "hi-v1"
MAX_CARDINAL = 9_999_999_999
DEVANAGARI_TO_ASCII_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Hindi numbers below 100 are irregular enough to require a reviewed lexicon.
HINDI_NUMBERS_0_TO_99 = (
    "शून्य",
    "एक",
    "दो",
    "तीन",
    "चार",
    "पाँच",
    "छह",
    "सात",
    "आठ",
    "नौ",
    "दस",
    "ग्यारह",
    "बारह",
    "तेरह",
    "चौदह",
    "पंद्रह",
    "सोलह",
    "सत्रह",
    "अठारह",
    "उन्नीस",
    "बीस",
    "इक्कीस",
    "बाईस",
    "तेईस",
    "चौबीस",
    "पच्चीस",
    "छब्बीस",
    "सत्ताईस",
    "अट्ठाईस",
    "उनतीस",
    "तीस",
    "इकतीस",
    "बत्तीस",
    "तैंतीस",
    "चौंतीस",
    "पैंतीस",
    "छत्तीस",
    "सैंतीस",
    "अड़तीस",
    "उनतालीस",
    "चालीस",
    "इकतालीस",
    "बयालीस",
    "तैंतालीस",
    "चौवालीस",
    "पैंतालीस",
    "छियालीस",
    "सैंतालीस",
    "अड़तालीस",
    "उनचास",
    "पचास",
    "इक्यावन",
    "बावन",
    "तिरपन",
    "चौवन",
    "पचपन",
    "छप्पन",
    "सत्तावन",
    "अट्ठावन",
    "उनसठ",
    "साठ",
    "इकसठ",
    "बासठ",
    "तिरसठ",
    "चौंसठ",
    "पैंसठ",
    "छियासठ",
    "सड़सठ",
    "अड़सठ",
    "उनहत्तर",
    "सत्तर",
    "इकहत्तर",
    "बहत्तर",
    "तिहत्तर",
    "चौहत्तर",
    "पचहत्तर",
    "छिहत्तर",
    "सतहत्तर",
    "अठहत्तर",
    "उनासी",
    "अस्सी",
    "इक्यासी",
    "बयासी",
    "तिरासी",
    "चौरासी",
    "पचासी",
    "छियासी",
    "सतासी",
    "अट्ठासी",
    "नवासी",
    "नब्बे",
    "इक्यानवे",
    "बानवे",
    "तिरानवे",
    "चौरानवे",
    "पचानवे",
    "छियानवे",
    "सत्तानवे",
    "अट्ठानवे",
    "निन्यानवे",
)

_DIGITS = "0-9०-९"
_INTEGER_RE = re.compile(rf"(?<![\w.+\-₹$€£])([{_DIGITS}](?:[{_DIGITS},]*[{_DIGITS}])?)(?![\w.%])")
_NUMERIC_TOKEN_RE = re.compile(rf"\S*[{_DIGITS}]\S*")
_PUNCTUATION_I_RE = re.compile(r"(?<=[\u0900-\u097f])I(?=\s|[।!?.,]|$)")
_WESTERN_GROUPING_RE = re.compile(r"[0-9]{1,3}(?:,[0-9]{3})+")
_INDIAN_GROUPING_RE = re.compile(r"[0-9]{1,2}(?:,[0-9]{2})*,[0-9]{3}")


@dataclass(frozen=True)
class NormalizationChange:
    kind: str
    original: str
    replacement: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizationReviewItem:
    kind: str
    token: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizationResult:
    text: str
    changes: tuple[NormalizationChange, ...]
    review_items: tuple[NormalizationReviewItem, ...]
    version: str = NORMALIZER_VERSION

    @property
    def needs_review(self) -> bool:
        return bool(self.review_items)


def hindi_integer_to_words(number: int) -> str:
    """Render a non-negative integer using common Hindi cardinal forms."""

    if not isinstance(number, int) or isinstance(number, bool):
        raise TypeError("number must be an integer")
    if not 0 <= number <= MAX_CARDINAL:
        raise ValueError(f"number must be between 0 and {MAX_CARDINAL}")
    if number < 100:
        return HINDI_NUMBERS_0_TO_99[number]

    remaining = number
    parts: list[str] = []
    for scale, name in ((10_000_000, "करोड़"), (100_000, "लाख"), (1_000, "हज़ार"), (100, "सौ")):
        count, remaining = divmod(remaining, scale)
        if count:
            parts.extend((hindi_integer_to_words(count), name))
    if remaining:
        parts.append(hindi_integer_to_words(remaining))
    return " ".join(parts)


def _parse_grouped_integer(token: str) -> int:
    ascii_token = token.translate(DEVANAGARI_TO_ASCII_DIGITS)
    if "," in ascii_token and not (
        _WESTERN_GROUPING_RE.fullmatch(ascii_token) or _INDIAN_GROUPING_RE.fullmatch(ascii_token)
    ):
        raise ValueError("invalid comma grouping")

    digits = ascii_token.replace(",", "")
    if len(digits) > 1 and digits.startswith("0"):
        raise ValueError("leading zeros usually indicate an identifier")
    value = int(digits)
    if value > MAX_CARDINAL:
        raise ValueError(f"value exceeds {MAX_CARDINAL}")
    return value


def _has_adjacent_letter(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start else ""
    after = text[end] if end < len(text) else ""
    return before.isalpha() or after.isalpha()


def normalize_hindi_text(text: str, *, number_mode: NumberMode = "preserve") -> NormalizationResult:
    """Normalize Hindi/Hinglish text while exposing every change and ambiguity."""

    if number_mode not in ("preserve", "cardinal"):
        raise ValueError(f"Unsupported number mode: {number_mode}")

    changes: list[NormalizationChange] = []
    review_items: list[NormalizationReviewItem] = []

    normalized = unicodedata.normalize("NFC", text)
    if normalized != text:
        changes.append(NormalizationChange("unicode-nfc", text, normalized))

    whitespace_normalized = " ".join(normalized.split())
    if whitespace_normalized != normalized:
        changes.append(NormalizationChange("whitespace", normalized, whitespace_normalized))
    normalized = whitespace_normalized

    for match in _PUNCTUATION_I_RE.finditer(normalized):
        review_items.append(NormalizationReviewItem("possible-punctuation-I", match.group()))

    integer_matches = list(_INTEGER_RE.finditer(normalized))
    handled_spans: list[tuple[int, int]] = []
    replacements: list[tuple[int, int, str]] = []
    for match in integer_matches:
        token = match.group()
        handled_spans.append(match.span())
        if number_mode == "preserve":
            review_items.append(NormalizationReviewItem("unexpanded-cardinal", token))
            continue
        try:
            replacement = hindi_integer_to_words(_parse_grouped_integer(token))
        except ValueError:
            review_items.append(NormalizationReviewItem("invalid-cardinal", token))
            continue
        replacements.append((*match.span(), replacement))
        changes.append(NormalizationChange("cardinal", token, replacement))

    for match in _NUMERIC_TOKEN_RE.finditer(normalized):
        if any(start <= match.start() and match.end() <= end for start, end in handled_spans):
            continue
        digit_spans = [
            (index, index + 1)
            for index in range(match.start(), match.end())
            if normalized[index] in "0123456789०१२३४५६७८९"
        ]
        kind = (
            "embedded-digit"
            if any(_has_adjacent_letter(normalized, start, end) for start, end in digit_spans)
            else "unsupported-number-context"
        )
        review_items.append(NormalizationReviewItem(kind, match.group()))

    for start, end, replacement in reversed(replacements):
        normalized = normalized[:start] + replacement + normalized[end:]

    return NormalizationResult(
        text=normalized, changes=tuple(changes), review_items=tuple(review_items)
    )
