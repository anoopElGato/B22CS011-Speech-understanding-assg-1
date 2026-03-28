import re

_DIGIT_WORDS = {
    "0": " ZERO ",
    "1": " ONE ",
    "2": " TWO ",
    "3": " THREE ",
    "4": " FOUR ",
    "5": " FIVE ",
    "6": " SIX ",
    "7": " SEVEN ",
    "8": " EIGHT ",
    "9": " NINE ",
}


def normalize_transcript(text: str) -> str:
    """Match Common Voice text to the base Wav2Vec2 tokenizer vocabulary."""
    if text is None:
        return ""

    text = str(text)
    for digit, word in _DIGIT_WORDS.items():
        text = text.replace(digit, word)

    text = text.upper()
    text = text.replace("-", " ")
    text = re.sub(r"[^A-Z' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_english_transcript(text: str) -> bool:
    """Keep rows that remain representable for the base English tokenizer."""
    if text is None:
        return False

    raw = str(text)
    normalized = normalize_transcript(raw)
    if not normalized:
        return False

    if re.search(r"[A-Za-z]", raw):
        return True

    raw_without_punct = re.sub(r"[\s.,!?;:'\"()\[\]{}\-_/\\]+", "", raw)
    return bool(raw_without_punct) and raw_without_punct.isdigit()
