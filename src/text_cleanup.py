import re
import unicodedata


MOJIBAKE_REPLACEMENTS = {
    "●": "-",
    "•": "-",
    "→": "->",
    "–": "-",
    "—": "-",
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "…": "...",
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€": '"',
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
    "Â": "",
}


def sanitize_text(value):
    text = unicodedata.normalize("NFKC", str(value or ""))

    for damaged, replacement in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(damaged, replacement)

    broken = r"(?:ï¿½|\ufffd)"
    text = re.sub(rf"(?<=\w){broken}(?=\w)", "'", text)
    text = re.sub(rf"\s+{broken}\s+", " - ", text)
    text = re.sub(rf"{broken}(?=\w)", '"', text)
    text = re.sub(rf"(?<=\w){broken}", '"', text)
    text = re.sub(broken, "", text)

    return text
