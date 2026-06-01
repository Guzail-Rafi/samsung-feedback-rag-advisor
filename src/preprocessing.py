import os
import re
import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


INPUT_PATH = "data/raw/youtube_comments.csv"
OUTPUT_PATH = "data/processed/clean_comments.csv"
MAX_COMMENTS = int(os.getenv("MAX_COMMENTS", "15000"))
SAMPLE_RANDOM_STATE = int(os.getenv("SAMPLE_RANDOM_STATE", "42"))


# Simple emoji sentiment dictionary
# You can add more emojis later
EMOJI_SENTIMENT = {
    "😍": 1.0,
    "❤": 1.0,
    "❤️": 1.0,
    "🔥": 0.8,
    "👍": 0.7,
    "😊": 0.7,
    "😁": 0.6,
    "😂": 0.3,
    "🤣": 0.3,

    "😐": 0.0,
    "🤔": 0.0,

    "👎": -0.7,
    "😡": -1.0,
    "😠": -0.9,
    "😭": -0.8,
    "😢": -0.8,
    "💀": -0.6,
    "🤮": -1.0,
    "😤": -0.8,
    "😒": -0.6,
    "🙄": -0.5,

    "❌": -0.6,
    "✔️": 0.4,
    "✅": 0.5,
    "💯": 0.8,
    "🤩": 0.9,
    "😎": 0.6,
    "😞": -0.7,
    "😔": -0.7,
    "🤦": -0.6,
    "🤦‍♂️": -0.6,
    "🤦‍♀️": -0.6,
    "😬": -0.4,
    "🥲": -0.4,
    "😅": 0.1,
    "😆": 0.3,
    "👏": 0.7,
    "🙏": 0.3
}


def extract_emojis(text):
    """
    Extracts emojis that exist in our emoji sentiment dictionary.
    """
    text = str(text)
    emojis_found = []

    for emoji in EMOJI_SENTIMENT.keys():
        if emoji in text:
            emojis_found.extend([emoji] * text.count(emoji))

    return emojis_found


def calculate_emoji_sentiment(text):
    """
    Calculates average emoji sentiment score.
    If no emoji is found, score is 0.
    """
    emojis = extract_emojis(text)

    if len(emojis) == 0:
        return 0

    scores = [EMOJI_SENTIMENT[emoji] for emoji in emojis]
    return sum(scores) / len(scores)


def count_emojis(text):
    """
    Counts sentiment-related emojis from our dictionary.
    """
    emojis = extract_emojis(text)
    return len(emojis)


def clean_text_keep_words(text):
    """
    Cleans comment text for normal NLP.
    This version removes URLs, mentions, and symbols,
    but the original comment_text remains saved separately.
    """

    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove hashtag symbol but keep the word
    text = text.replace("#", "")

    # Remove emojis and symbols only from clean_comment
    # Important: original comment_text still keeps emojis
    text = re.sub(r"[^\w\s.,!?']", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Lowercase
    text = text.lower().strip()

    return text


def limit_comments(df):
    if MAX_COMMENTS <= 0 or len(df) <= MAX_COMMENTS:
        return df.reset_index(drop=True)

    return (
        df.sample(n=MAX_COMMENTS, random_state=SAMPLE_RANDOM_STATE)
        .sort_index()
        .reset_index(drop=True)
    )


def main():
    df = pd.read_csv(INPUT_PATH)

    print("Original rows:", len(df))

    # Remove missing comments
    df = df.dropna(subset=["comment_text"])

    # Remove duplicate comments
    df = df.drop_duplicates(subset=["comment_id"])

    # Keep original text as it is
    df["original_comment"] = df["comment_text"]

    # Cleaned text for NLP
    df["clean_comment"] = df["comment_text"].apply(clean_text_keep_words)

    # Emoji features
    df["emoji_count"] = df["comment_text"].apply(count_emojis)
    df["emoji_sentiment_score"] = df["comment_text"].apply(calculate_emoji_sentiment)

    # Basic text features
    df["comment_length"] = df["clean_comment"].str.len()
    df["word_count"] = df["clean_comment"].str.split().str.len()

    # Remove very short comments after cleaning
    df = df[df["word_count"] >= 3]

    # Convert date column
    df["comment_published_at"] = pd.to_datetime(df["comment_published_at"], errors="coerce")

    rows_before_sampling = len(df)
    df = limit_comments(df)

    # Save cleaned file
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Cleaned rows:", len(df))
    if rows_before_sampling != len(df):
        print(f"Sampled rows: {len(df)} of {rows_before_sampling}")
    print("Saved to:", OUTPUT_PATH)

    print("\nPreview:")
    print(df[[
        "original_comment",
        "clean_comment",
        "emoji_count",
        "emoji_sentiment_score",
        "word_count"
    ]].head(10))


if __name__ == "__main__":
    main()
