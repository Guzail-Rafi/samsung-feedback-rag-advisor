import os
import re
import pandas as pd
import spacy


INPUT_PATH = "data/processed/comments_with_topics.csv"
OUTPUT_PATH = "data/processed/comments_with_ner.csv"
ENTITY_OUTPUT_PATH = "data/processed/ner_entities.csv"
SPACY_MODEL = "en_core_web_sm"

nlp = None


CUSTOM_PRODUCTS = [
    "samsung", "galaxy", "galaxy s25", "galaxy s25 ultra",
    "galaxy s24", "galaxy s24 ultra", "s25 ultra", "s24 ultra",
    "s pen", "spen", "one ui", "gemini", "bixby",
    "iphone", "iphone pro max", "apple", "google"
]

ENTITY_COLUMNS = [
    "comment_id",
    "video_id",
    "comment_text",
    "entity",
    "entity_type",
    "sentiment_label",
    "issue_category",
    "topic_name"
]


def get_nlp():
    global nlp

    if nlp is None:
        try:
            nlp = spacy.load(SPACY_MODEL)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{SPACY_MODEL}' is not installed. "
                f"Install it with: python -m spacy download {SPACY_MODEL}"
            ) from exc

    return nlp


def contains_phrase(text, phrase):
    pattern = r"(?<!\w)" + re.escape(phrase).replace(r"\ ", r"\s+") + r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def extract_entities(text):
    if pd.isna(text):
        return []

    text = str(text)
    doc = get_nlp()(text)

    entities = []

    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "PERSON", "GPE", "NORP"]:
            entities.append({
                "entity": ent.text,
                "entity_type": ent.label_
            })

    # Custom Samsung/product entity matching
    for product in CUSTOM_PRODUCTS:
        if contains_phrase(text, product):
            entities.append({
                "entity": product,
                "entity_type": "CUSTOM_PRODUCT_OR_BRAND"
            })

    # Remove duplicates
    unique_entities = []
    seen = set()

    for ent in entities:
        key = (ent["entity"].lower(), ent["entity_type"])
        if key not in seen:
            seen.add(key)
            unique_entities.append(ent)

    return unique_entities


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}. Run topic_modeling.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    text_column = "spellchecked_comment"

    if text_column not in df.columns:
        text_column = "clean_comment"

    if text_column not in df.columns:
        raise ValueError("Input CSV must contain 'spellchecked_comment' or 'clean_comment'.")

    all_entity_rows = []
    comment_entities = []
    comment_entity_types = []

    for idx, row in df.iterrows():
        entities = extract_entities(row[text_column])

        entity_names = [ent["entity"] for ent in entities]
        entity_types = [ent["entity_type"] for ent in entities]

        comment_entities.append(", ".join(entity_names))
        comment_entity_types.append(", ".join(entity_types))

        for ent in entities:
            all_entity_rows.append({
                "comment_id": row.get("comment_id", idx),
                "video_id": row.get("video_id", ""),
                "comment_text": row.get(text_column, ""),
                "entity": ent["entity"],
                "entity_type": ent["entity_type"],
                "sentiment_label": row.get("sentiment_label", ""),
                "issue_category": row.get("issue_category", ""),
                "topic_name": row.get("topic_name", "")
            })

    df["named_entities"] = comment_entities
    df["named_entity_types"] = comment_entity_types

    entity_df = pd.DataFrame(all_entity_rows, columns=ENTITY_COLUMNS)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    entity_df.to_csv(ENTITY_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("NER extraction completed!")
    print("Saved comments with NER to:", OUTPUT_PATH)
    print("Saved entity list to:", ENTITY_OUTPUT_PATH)

    print("\nTop entities:")
    if entity_df.empty:
        print("No entities found.")
    else:
        print(entity_df["entity"].value_counts().head(20))


if __name__ == "__main__":
    main()
