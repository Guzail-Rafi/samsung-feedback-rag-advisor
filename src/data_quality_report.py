import sys

import pandas as pd

from preprocessing import MAX_COMMENTS, clean_text_keep_words


RAW_PATH = "data/raw/youtube_comments.csv"
CATEGORY_PATH = "data/processed/comments_with_categories.csv"


def display_text(value, limit=58):
    text = " ".join(str(value).split())
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    raw = pd.read_csv(RAW_PATH)
    raw_rows = len(raw)
    missing_rows = int(raw["comment_text"].isna().sum())

    quality = raw.dropna(subset=["comment_text"]).copy()
    duplicate_rows = int(quality.duplicated(subset=["comment_id"]).sum())
    quality = quality.drop_duplicates(subset=["comment_id"]).copy()
    quality["clean_comment"] = quality["comment_text"].map(clean_text_keep_words)
    quality["word_count"] = quality["clean_comment"].str.split().str.len()

    short = quality[quality["word_count"] < 3].copy()
    eligible = quality[quality["word_count"] >= 3].copy()
    sample_size = min(MAX_COMMENTS, len(eligible)) if MAX_COMMENTS > 0 else len(eligible)
    cap_excluded = len(eligible) - sample_size

    classified = pd.read_csv(CATEGORY_PATH)
    english_rows = int((classified["language"] == "en").sum())
    non_english_rows = len(classified) - english_rows

    print("=" * 72)
    print("YOUTUBE COMMENT DATA-QUALITY AND FILTERING SUMMARY")
    print("=" * 72)
    print(f"Raw comments collected:                    {raw_rows:>7,}")
    print(f"Missing comment text removed:              {missing_rows:>7,}")
    print(f"Duplicate comment IDs removed:             {duplicate_rows:>7,}")
    print(f"Comments removed after cleaning (<3 words):{len(short):>7,}")
    print(f"Comments eligible after quality filtering: {len(eligible):>7,}")
    print(f"Comments selected by experiment cap:       {sample_size:>7,}")
    print(f"Eligible comments excluded only by cap:    {cap_excluded:>7,}")
    print(f"Non-English/unknown comments filtered:     {non_english_rows:>7,}")
    print(f"English comments retained for NLP/RAG:     {english_rows:>7,}")

    examples = short.sort_values(
        ["word_count", "comment_text"],
        kind="stable",
    )
    example_rows = []
    used_counts = set()

    for _, row in examples.iterrows():
        count = int(row["word_count"])
        if count in used_counts:
            continue
        example_rows.append(
            {
                "Original noisy comment": display_text(row["comment_text"]),
                "Cleaned text": display_text(row["clean_comment"]),
                "Words": count,
                "Filtering decision": "Removed: fewer than 3 words",
            }
        )
        used_counts.add(count)
        if used_counts == {0, 1, 2}:
            break

    print("\nRepresentative low-information comments:")
    print(pd.DataFrame(example_rows).to_string(index=False))
    print("\nNote: the 15,000-row cap is experimental sampling, not noise removal.")


if __name__ == "__main__":
    main()
