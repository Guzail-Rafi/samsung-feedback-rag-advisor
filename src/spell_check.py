import os
import re
import sys
import pandas as pd
from spellchecker import SpellChecker

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


INPUT_PATH = "data/processed/clean_comments.csv"
OUTPUT_PATH = "data/processed/comments_with_spellcheck.csv"
TEXT_COLUMN = "clean_comment"


CUSTOM_WORDS = {
    "samsung", "galaxy", "ultra", "spen", "s", "pen",
    "s25", "s24", "s23", "s22", "s27",
    "iphone", "apple", "ios", "android",
    "gemini", "bixby", "oneui", "ui",
    "snapdragon", "exynos",
    "youtube", "ai",
    "idk", "whos", "yeah", "yep", "nah", "tho", "tbh", "imo",
    "lol", "lmao", "bro", "gonna", "wanna", "kinda",
    "dont", "doesnt", "didnt", "isnt", "cant", "wont",
    "ive", "im", "youre"
}

WORD_PATTERN = re.compile(r"^([^A-Za-z0-9]*)([A-Za-z][A-Za-z']*)([^A-Za-z0-9]*)$")
NON_ASCII_PATTERN = re.compile(r"[^\x00-\x7F]")


def match_case(original, corrected):
    if original.isupper():
        return corrected.upper()
    if original.istitle():
        return corrected.capitalize()
    return corrected


def should_skip_word(word):
    lookup = word.lower()

    return (
        lookup in CUSTOM_WORDS
        or lookup == ""
        or any(char.isdigit() for char in lookup)
        or "'" in lookup
        or len(lookup) <= 2
    )


def correct_token(token, spell):
    match = WORD_PATTERN.match(token)

    if not match:
        return token

    prefix, word, suffix = match.groups()
    lookup = word.lower()

    if should_skip_word(lookup):
        return token

    if lookup in spell:
        return token

    corrected = spell.correction(lookup)

    if not corrected or corrected == lookup:
        return token

    return f"{prefix}{match_case(word, corrected)}{suffix}"


def build_token_corrector(spell):
    cache = {}

    def correct_token_cached(token):
        if token not in cache:
            cache[token] = correct_token(token, spell)
        return cache[token]

    return correct_token_cached


def correct_text(text, correct_token_func):
    if pd.isna(text):
        return ""

    text = str(text)

    if NON_ASCII_PATTERN.search(text):
        return text.strip()

    words = text.split()
    corrected_words = [correct_token_func(word) for word in words]

    return " ".join(corrected_words).strip()


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}. Run preprocessing.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Input CSV must contain a '{TEXT_COLUMN}' column.")

    spell = SpellChecker(distance=1)
    spell.word_frequency.load_words(CUSTOM_WORDS)
    correct_token_func = build_token_corrector(spell)

    df["pre_spellcheck_comment"] = df[TEXT_COLUMN]
    df["spellchecked_comment"] = df[TEXT_COLUMN].apply(
        lambda x: correct_text(x, correct_token_func)
    )

    # Downstream scripts already use clean_comment, so keep the corrected text there.
    df[TEXT_COLUMN] = df["spellchecked_comment"]
    df["comment_length"] = df[TEXT_COLUMN].fillna("").astype(str).str.len()
    df["word_count"] = df[TEXT_COLUMN].fillna("").astype(str).str.split().str.len()

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Spell check completed!")
    print("Saved to:", OUTPUT_PATH)

    print("\nPreview:")
    print(df[["pre_spellcheck_comment", "spellchecked_comment"]].head(10))


if __name__ == "__main__":
    main()
