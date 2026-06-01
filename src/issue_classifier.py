import os
import sys
import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException
from sentence_transformers import SentenceTransformer, util

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DetectorFactory.seed = 0

INPUT_PATH = "data/processed/comments_with_sentiment.csv"
OUTPUT_PATH = "data/processed/comments_with_categories.csv"


# =========================
# 1. CATEGORY DEFINITIONS
# =========================

CATEGORY_DESCRIPTIONS = {
    "Battery / Charging": (
        "Comments about battery life, battery drain, poor charging, slow charging, "
        "phone not lasting long, needing to charge often, power issues, charger problems."
    ),

    "Camera": (
        "Comments about camera quality, photos, videos, zoom, lens, portrait mode, "
        "night photography, blurry images, camera performance."
    ),

    "AI / Gemini": (
        "Comments about Galaxy AI, Gemini, artificial intelligence features, AI tools, "
        "Circle to Search, assistant features, AI marketing, AI being unnecessary."
    ),

    "S-Pen / Features": (
        "Comments about S-Pen, stylus, Bluetooth removal, missing features, buttons, "
        "feature changes, useful or removed phone features."
    ),

    "Display / Screen": (
        "Comments about screen, display, brightness, crease, green line, foldable screen, "
        "inside screen, display durability, screen quality."
    ),

    "Performance / Processor": (
        "Comments about speed, lag, processor, Snapdragon, performance, overheating, "
        "heat, gaming performance, phone being slow or fast."
    ),

    "Design / Build": (
        "Comments about phone design, body, thinness, size, weight, durability, build quality, "
        "edges, shape, materials."
    ),

    "Price / Value": (
        "Comments about price, cost, expensive phones, value for money, overpriced products, "
        "whether the phone is worth buying."
    ),

    "Software / One UI": (
        "Comments about software, One UI, Android updates, bugs, user interface, update problems, "
        "software experience."
    ),

    "Customer Support / Warranty": (
        "Comments about Samsung support, customer service, warranty, repairs, replacement, "
        "service centers, refund, bad after-sales service."
    ),

    "Positive Feedback": (
        "Comments praising Samsung products, saying the phone is great, amazing, beautiful, "
        "excellent, perfect, or expressing love for Samsung."
    ),

    "Other": (
        "General comments that do not clearly match any specific Samsung product issue."
    )
}


# =========================
# 2. DIRECT KEYWORDS
# =========================

ISSUE_KEYWORDS = {
    "Battery / Charging": [
        "battery", "charging", "charge", "charger", "power", "drain"
    ],
    "Camera": [
        "camera", "photo", "photos", "video", "zoom", "lens", "portrait"
    ],
    "AI / Gemini": [
        "ai", "galaxy ai", "gemini", "circle to search", "assistant"
    ],
    "S-Pen / Features": [
        "s pen", "spen", "stylus", "bluetooth", "mute button"
    ],
    "Display / Screen": [
        "screen", "display", "crease", "green line", "fold"
    ],
    "Performance / Processor": [
        "processor", "snapdragon", "performance", "lag", "slow", "heat", "overheating"
    ],
    "Design / Build": [
        "design", "thin", "body", "edges", "durable", "build", "weight", "size"
    ],
    "Price / Value": [
        "price", "expensive", "cheap", "cost", "worth", "value", "overpriced", "money"
    ],
    "Software / One UI": [
        "software", "one ui", "update", "updates", "android", "bug", "bugs", "ui"
    ],
    "Customer Support / Warranty": [
        "support", "customer service", "warranty", "repair", "service center", "refund", "replacement"
    ],
    "Positive Feedback": [
        "love", "best", "great", "amazing", "perfect", "excellent", "awesome", "good", "nice"
    ]
}


# =========================
# 3. LOAD EMBEDDING MODEL
# =========================

try:
    model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
except OSError:
    model = SentenceTransformer("all-MiniLM-L6-v2")

category_names = list(CATEGORY_DESCRIPTIONS.keys())
category_texts = list(CATEGORY_DESCRIPTIONS.values())

category_embeddings = model.encode(
    category_texts,
    convert_to_tensor=True
)


# =========================
# 4. CLASSIFICATION FUNCTIONS
# =========================

def keyword_classify(text):
    """
    First tries direct keyword matching.
    This is useful for clear comments like 'battery life horrible'.
    """

    text = str(text).lower()

    for category, keywords in ISSUE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                return category

    return None


def semantic_classify(text):
    """
    Uses embeddings to understand indirect meaning.
    """

    text = str(text)

    comment_embedding = model.encode(
        text,
        convert_to_tensor=True
    )

    similarities = util.cos_sim(
        comment_embedding,
        category_embeddings
    )[0]

    best_index = similarities.argmax().item()
    best_score = similarities[best_index].item()
    best_category = category_names[best_index]

    # If confidence is too low, mark as Other
    if best_score < 0.25:
        return "Other", round(best_score, 3)

    return best_category, round(best_score, 3)


def hybrid_classify(text):
    """
    Uses keyword classification first.
    If no keyword is found, then uses semantic classification.
    """

    keyword_result = keyword_classify(text)

    if keyword_result:
        return keyword_result, "keyword", 1.0

    semantic_category, semantic_score = semantic_classify(text)

    return semantic_category, "semantic", semantic_score


def detect_language(text):
    try:
        return detect(str(text))
    except LangDetectException:
        return "unknown"


# =========================
# 5. MAIN FUNCTION
# =========================

def main():
    df = pd.read_csv(INPUT_PATH)
    df["language"] = df["clean_comment"].apply(detect_language)

    categories = []
    methods = []
    confidence_scores = []

    for text, language in zip(df["clean_comment"], df["language"]):
        if language != "en":
            categories.append("Non-English")
            methods.append("language_filter")
            confidence_scores.append(0.0)
            continue

        if len(str(text).split()) < 3:
            categories.append("Other")
            methods.append("too_short")
            confidence_scores.append(0.0)
            continue

        category, method, confidence = hybrid_classify(text)

        categories.append(category)
        methods.append(method)
        confidence_scores.append(confidence)

    df["issue_category"] = categories
    df["classification_method"] = methods
    df["category_confidence"] = confidence_scores

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Hybrid issue classification completed!")
    print("Saved to:", OUTPUT_PATH)

    print("\nIssue category counts:")
    print(df["issue_category"].value_counts())

    print("\nClassification method counts:")
    print(df["classification_method"].value_counts())

    print("\nPreview:")
    print(df[[
        "clean_comment",
        "language",
        "sentiment_label",
        "issue_category",
        "classification_method",
        "category_confidence"
    ]].head(20))


if __name__ == "__main__":
    main()
