# # from curses import raw
# import json
# import re
# import unicodedata
# from collections import Counter
# from pathlib import Path
# from typing import Union

# # =============================
# # CONFIG — FREEZE
# # =============================
# SHORT_TEXT_MAX_LEN = 120
# HEADER_FOOTER_FREQ_THRESHOLD = 0.6
# SYMBOL_RUN_MIN = 4

# VISION_FILLER_PATTERNS = [
#     r"\bappears to be\b",
#     r"\bgeneric image\b",
#     r"\bimage of\b",
# ]

# CONTROL_CHARS_REGEX = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
# SYMBOL_RUN_REGEX = re.compile(rf"([^A-Za-z0-9\s])\1{{{SYMBOL_RUN_MIN - 1},}}")
# HYPHEN_LINEBREAK_REGEX = re.compile(r"(\w+)-\n(\w+)")
# SENTENCE_PUNCTUATION_REGEX = re.compile(r"[.!?;:]")

# # =============================
# # Internal helpers
# # =============================
# def _drop_vision_filler(text: str) -> str:
#     lower = text.lower()
#     for pat in VISION_FILLER_PATTERNS:
#         if re.search(pat, lower):
#             return ""
#     return text


# def _unicode_normalize(text: str) -> str:
#     return unicodedata.normalize("NFKC", text)


# def _trim_whitespace(text: str) -> str:
#     return text.strip()


# def _normalize_spaces(text: str) -> str:
#     return re.sub(r"[ \t]+", " ", text)


# def _normalize_newlines(text: str) -> str:
#     text = text.replace("\r\n", "\n").replace("\r", "\n")
#     return re.sub(r"\n{3,}", "\n\n", text)


# def _repair_hyphenated_linebreaks(text: str) -> str:
#     return HYPHEN_LINEBREAK_REGEX.sub(r"\1\2", text)


# def _collapse_symbol_runs(text: str) -> str:
#     return SYMBOL_RUN_REGEX.sub(r"\1", text)


# def _strip_control_chars(text: str) -> str:
#     return CONTROL_CHARS_REGEX.sub("", text)


# def _collapse_newlines_in_short_text(text: str) -> str:
#     if (
#         len(text) <= SHORT_TEXT_MAX_LEN
#         and "\n" in text
#         and not SENTENCE_PUNCTUATION_REGEX.search(text)
#     ):
#         return text.replace("\n", " ")
#     return text

# def _coerce_content_to_string(content) -> str:
#     """
#     Ensures content is a string.
#     - list[str] -> joined with newline
#     - None -> empty string
#     - everything else -> str(...)
#     """
#     if content is None:
#         return ""
#     if isinstance(content, list):
#         return "\n".join(str(x) for x in content if x is not None)
#     if isinstance(content, str):
#         return content
#     return str(content)

# # =============================
# # PUBLIC FUNCTION (ONLY ONE)
# # =============================
# def clean_content_units_file(
#     input_path: Union[str, Path],
#     output_suffix: str = "_cleaned",
# ) -> Path:
#     """
#     Cleans only the `content` field of a content_units.jsonl file.
#     Writes a new file in the same directory with a suffix.
#     """
#     input_path = Path(input_path)
#     output_path = input_path.with_name(
#         input_path.stem + output_suffix + input_path.suffix
#     )

#     # ---- Load records
#     with input_path.open("r", encoding="utf-8") as f:
#         records = [json.loads(line) for line in f]

#     # ---- Header/footer detection (statistical)
#     counter = Counter()
#     for r in records:
#         # text = r.get("content", "").strip()
#         raw = _coerce_content_to_string(r.get("content"))
#         text = raw.strip()
#         if 0 < len(text) <= SHORT_TEXT_MAX_LEN:
#             counter[text] += 1

#     header_footer_candidates = {
#         t
#         for t, freq in counter.items()
#         if freq / len(records) >= HEADER_FOOTER_FREQ_THRESHOLD
#     }

#     # ---- Cleaning pipeline
#     for r in records:
#         # text = r.get("content", "")
#         text = _coerce_content_to_string(r.get("content"))

#         # 1. Drop vision filler
#         text = _drop_vision_filler(text)
#         if not text:
#             r["content"] = ""
#             continue

#         # 2. Unicode normalization
#         text = _unicode_normalize(text)

#         # 3. Trim whitespace
#         text = _trim_whitespace(text)

#         # 4. Normalize spaces
#         text = _normalize_spaces(text)

#         # 5. Normalize newlines
#         text = _normalize_newlines(text)

#         # 6. Repair hyphenated line breaks (gated)
#         text = _repair_hyphenated_linebreaks(text)

#         # 7. Collapse repeated symbol runs (gated)
#         text = _collapse_symbol_runs(text)

#         # 8. Remove repeated headers/footers (statistical)
#         if text in header_footer_candidates:
#             text = ""

#         # 9. Strip control characters
#         text = _strip_control_chars(text)

#         # 10. Collapse newlines in short, non-sentence text
#         text = _collapse_newlines_in_short_text(text)

#         r["content"] = text

#     # ---- Write output
#     with output_path.open("w", encoding="utf-8") as f:
#         for r in records:
#             f.write(json.dumps(r, ensure_ascii=False) + "\n")

#     return output_path

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Union, List

# =============================
# CONFIG — FROZEN
# =============================
SHORT_TEXT_MAX_LEN = 120
HEADER_FOOTER_FREQ_THRESHOLD = 0.6
SYMBOL_RUN_MIN = 4
MAX_CONSECUTIVE_DUPLICATES = 1

VISION_FILLER_PATTERNS = [
    r"\bappears to be\b",
    r"\bgeneric image\b",
    r"\bimage of\b",
]

CONTROL_CHARS_REGEX = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
SYMBOL_RUN_REGEX = re.compile(rf"([^A-Za-z0-9\s])\1{{{SYMBOL_RUN_MIN - 1},}}")
HYPHEN_LINEBREAK_REGEX = re.compile(r"(\w+)-\n(\w+)")
SENTENCE_PUNCTUATION_REGEX = re.compile(r"[.!?;:]")

# =============================
# Structural normalization
# =============================
def _coerce_content_to_string(content) -> str:
    if content is None:
        return ""
    if isinstance(content, list):
        return "\n".join(str(x) for x in content if x is not None)
    if isinstance(content, str):
        return content
    return str(content)

# =============================
# Cleaning helpers
# =============================
def _drop_vision_filler(text: str) -> str:
    lower = text.lower()
    for pat in VISION_FILLER_PATTERNS:
        if re.search(pat, lower):
            return ""
    return text


def _unicode_normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _trim_whitespace(text: str) -> str:
    return text.strip()


def _normalize_spaces(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text)


def _normalize_newlines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", text)


def _repair_hyphenated_linebreaks(text: str) -> str:
    return HYPHEN_LINEBREAK_REGEX.sub(r"\1\2", text)


def _collapse_symbol_runs(text: str) -> str:
    return SYMBOL_RUN_REGEX.sub(r"\1", text)


def _strip_control_chars(text: str) -> str:
    return CONTROL_CHARS_REGEX.sub("", text)


def _collapse_newlines_in_short_text(text: str) -> str:
    if (
        len(text) <= SHORT_TEXT_MAX_LEN
        and "\n" in text
        and not SENTENCE_PUNCTUATION_REGEX.search(text)
    ):
        return text.replace("\n", " ")
    return text


def _collapse_consecutive_duplicates(texts: List[str]) -> List[str]:
    """Keep at most N consecutive identical texts."""
    result = []
    last = None
    count = 0

    for t in texts:
        if t == last:
            count += 1
            if count <= MAX_CONSECUTIVE_DUPLICATES:
                result.append(t)
        else:
            last = t
            count = 1
            result.append(t)

    return result

# =============================
# PUBLIC ENTRY POINT
# =============================
def clean_content_units_file(
    input_path: Union[str, Path],
    output_suffix: str = "_cleaned",
) -> Path:
    input_path = Path(input_path)
    output_path = input_path.with_name(
        input_path.stem + output_suffix + input_path.suffix
    )

    # ---- Load
    with input_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    # ---- Header/footer detection
    counter = Counter()
    for r in records:
        raw = _coerce_content_to_string(r.get("content"))
        text = raw.strip()
        if 0 < len(text) <= SHORT_TEXT_MAX_LEN:
            counter[text] += 1

    header_footer_candidates = {
        t
        for t, freq in counter.items()
        if freq / len(records) >= HEADER_FOOTER_FREQ_THRESHOLD
    }

    # ---- Clean content (exact order)
    cleaned_texts = []
    for r in records:
        text = _coerce_content_to_string(r.get("content"))

        text = _drop_vision_filler(text)
        if not text:
            cleaned_texts.append("")
            continue

        text = _unicode_normalize(text)
        text = _trim_whitespace(text)
        text = _normalize_spaces(text)
        text = _normalize_newlines(text)
        text = _repair_hyphenated_linebreaks(text)
        text = _collapse_symbol_runs(text)

        if text in header_footer_candidates:
            text = ""

        text = _strip_control_chars(text)
        text = _collapse_newlines_in_short_text(text)

        assert isinstance(text, str)
        cleaned_texts.append(text)

    # ---- Collapse excessive consecutive duplicates
    cleaned_texts = _collapse_consecutive_duplicates(cleaned_texts)

    # ---- Write output
    with output_path.open("w", encoding="utf-8") as f:
        for r, text in zip(records, cleaned_texts):
            r["content"] = text
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return output_path