"""Deterministic Spanish language diagnostics with unresolved words preserved."""

from collections import Counter
from dataclasses import dataclass
from itertools import islice
import re
import unicodedata


MIN_WORDS = 8
MAX_WORDS = 24
TOP_SCORE = 0.80
SCORE_MARGIN = 0.30
WORD = re.compile(r"[^\W\d_]+")
INLINE_NONPROSE = re.compile(r"(`+[^`\n]*`+|https?://\S+)", re.IGNORECASE)
FENCE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
CODE_LINE = re.compile(
    r"^\s*(?:#!|(?:class|def|function)\s+[\w$]+\s*[(<{:]|"
    r"(?:import|from)\s+\w+[\w.]*\s*(?:import\b|$)|"
    r"[\w$.]+\s*\([^\n]*\)\s*[;{}]?\s*$|"
    r"[\w$.]+\s*=(?!=)\s*\S|[{}]+\s*$)"
)
REFUSAL = re.compile(
    r"^(?:(?:i(?:['’]m| am) sorry|sorry|lo siento|lo lamento|disculpa(?:me)?)"
    r"[\s,.:!—-]*(?:(?:but|pero)\s+)?)?"
    r"(?:i\s+(?:can(?:not|['’]t)|am unable to|won['’]t)\s+"
    r"(?:help|assist|comply|provide|fulfill|answer)\b|"
    r"(?:no (?:puedo|podría)|no me es posible)\s+"
    r"(?:ayudar|colaborar|cumplir|proporcionar|responder|contestar)\b|"
    r"(?:i (?:must )?decline|me niego)\b)"
)


@dataclass(frozen=True)
class Evidence:
    spanish_words: int
    other_words: int
    unknown_words: int
    category: str

    def bounds(self):
        known = self.spanish_words
        possible = known + self.unknown_words
        total = possible + self.other_words
        if self.category in {"empty", "explicit_refusal"} or not total:
            return {name: (0.0, 1.0) for name in ("any_spanish", "majority_spanish", "spanish_word_fraction")}
        return {
            "any_spanish": (float(known > 0), float(possible > 0)),
            "majority_spanish": (float(2 * known > total), float(2 * possible > total)),
            "spanish_word_fraction": (known / total, possible / total),
        }


def segments(answer):
    """Yield word windows and whether each can be sent to language identification."""
    fence = None
    for line in answer.splitlines():
        marker = FENCE.match(line)
        if marker:
            delimiter, remainder = marker.groups()
            if fence is None:
                fence = delimiter
            elif delimiter[0] == fence[0] and len(delimiter) >= len(fence) and not remainder.strip():
                fence = None
            yield WORD.findall(remainder), False
            continue
        if fence is not None or CODE_LINE.match(line):
            yield WORD.findall(line), False
            continue
        for part in INLINE_NONPROSE.split(line):
            if INLINE_NONPROSE.fullmatch(part):
                yield WORD.findall(part), False
                continue
            for sentence in re.split(r"[.!?;]+", part):
                words = iter(WORD.findall(sentence))
                while window := list(islice(words, MAX_WORDS)):
                    yield window, len(window) >= MIN_WORDS


def measure(answer, identify):
    normalized = unicodedata.normalize("NFKC", answer).casefold()
    if not normalized.strip():
        return Evidence(0, 0, 0, "empty")
    if REFUSAL.match(normalized.lstrip(" \t\r\n*>#")):
        return Evidence(0, 0, len(WORD.findall(normalized)), "explicit_refusal")

    counts = Counter()
    for window, prose in segments(normalized):
        language = "unknown"
        if prose:
            labels, scores = identify(" ".join(window), k=2)
            if len(labels) == len(scores) == 2 and scores[0] >= TOP_SCORE and scores[0] - scores[1] >= SCORE_MARGIN:
                language = "spanish" if labels[0] == "__label__es" else "other"
        counts[language] += len(window)

    spanish, other, unknown = (counts[k] for k in ("spanish", "other", "unknown"))
    if spanish and other:
        category = "mixed_detected"
    elif spanish:
        category = "spanish_detected"
    elif other:
        category = "other_detected"
    else:
        category = "unknown"
    return Evidence(spanish, other, unknown, category)


def synthetic_checks(identify):
    english = "The summer weather is usually warm and people enjoy walking through the park together in the evening."
    spanish = "Durante las vacaciones de verano muchas familias visitan la playa para descansar y disfrutar del buen tiempo."
    examples = {
        "spanish": (spanish, "spanish_detected"),
        "english": (english, "other_detected"),
        "mixed": (english + "\n" + spanish, "mixed_detected"),
        "short": ("Muchas gracias", "unknown"),
        "blank": (" \n ", "empty"),
        "fenced_code": ("```python\ndef greet(name):\n    return 'welcome to our wonderful community and have a wonderful day'\n```", "unknown"),
        "plain_code": ("print('welcome to our wonderful community and have a wonderful day')", "unknown"),
        "refusal": ("Lo siento, pero no puedo proporcionar instrucciones para esa solicitud.", "explicit_refusal"),
    }
    checked = {key: measure(answer, identify).category for key, (answer, _) in examples.items()}
    if any(checked[key] != expected for key, (_, expected) in examples.items()):
        raise ValueError(f"fixed synthetic language controls failed: {checked}")
    return checked
