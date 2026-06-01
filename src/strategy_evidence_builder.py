import os
import pandas as pd


INPUT_PATH = "data/processed/comments_with_topics.csv"
OUTPUT_PATH = "data/processed/strategy_evidence.csv"


STRATEGY_RULES = {
    "Battery / Charging": {
        "customer_signal": "Users want longer real-world battery life, better charging, and fewer power-related compromises.",
        "customer_recommendation": "Improve battery capacity, charging speed, thermal efficiency, and real-world battery optimization.",
        "profit_recommendation": "Position stronger battery life as a premium productivity and gaming feature to justify flagship pricing.",
        "business_impact": "High customer satisfaction impact and strong premium value signal.",
        "priority": "High"
    },
    "S-Pen / Features": {
        "customer_signal": "Users are concerned about removed or reduced S-Pen functionality, especially Bluetooth support.",
        "customer_recommendation": "Restore advanced S-Pen features such as Bluetooth gestures, remote shutter, and productivity tools.",
        "profit_recommendation": "Use enhanced S-Pen features as an Ultra-exclusive differentiator to protect premium pricing.",
        "business_impact": "High loyalty impact among Ultra and Note-series users.",
        "priority": "High"
    },
    "AI / Gemini": {
        "customer_signal": "Users show mixed reactions to Galaxy AI and Gemini; some value useful AI, while others feel AI is over-marketed.",
        "customer_recommendation": "Make AI features practical, optional, fast, and directly useful for daily tasks.",
        "profit_recommendation": "Bundle AI with premium productivity features, but avoid making AI feel like the only upgrade.",
        "business_impact": "Medium to high value if AI is linked to real use cases.",
        "priority": "Medium"
    },
    "Camera": {
        "customer_signal": "Users discuss camera quality, fake camera comparisons, interface issues, and expectations for flagship camera performance.",
        "customer_recommendation": "Improve camera consistency, low-light quality, video processing, and camera app usability.",
        "profit_recommendation": "Market camera upgrades as premium creator features to support flagship positioning.",
        "business_impact": "High purchase influence, especially for flagship buyers.",
        "priority": "High"
    },
    "Display / Screen": {
        "customer_signal": "Users mention screen quality, durability, green lines, privacy display, accidental touches, and screen protection concerns.",
        "customer_recommendation": "Improve display durability, reduce green-line risk, improve screen protection, and refine privacy display features.",
        "profit_recommendation": "Use stronger display durability and privacy display as premium differentiators.",
        "business_impact": "High satisfaction and trust impact.",
        "priority": "High"
    },
    "Price / Value": {
        "customer_signal": "Users compare price with value, upgrades, innovation, and whether premium pricing is justified.",
        "customer_recommendation": "Improve perceived value by adding meaningful upgrades instead of removing features.",
        "profit_recommendation": "Maintain premium pricing only if supported by visible high-value features.",
        "business_impact": "High impact on purchase decisions and upgrade willingness.",
        "priority": "High"
    },
    "Software / One UI": {
        "customer_signal": "Users discuss One UI, software updates, similarity to Apple, bugs, and software innovation.",
        "customer_recommendation": "Improve software stability, update experience, customization, and useful One UI features.",
        "profit_recommendation": "Use software ecosystem improvements to increase retention and reduce switching to competitors.",
        "business_impact": "Medium to high retention impact.",
        "priority": "Medium"
    },
    "Performance / Processor": {
        "customer_signal": "Users mention speed, processor, heating, performance, and gaming/productivity expectations.",
        "customer_recommendation": "Improve sustained performance, heat control, and gaming efficiency.",
        "profit_recommendation": "Market performance improvements toward gamers, creators, and power users.",
        "business_impact": "Medium to high premium segment impact.",
        "priority": "Medium"
    },
    "Design / Build": {
        "customer_signal": "Users discuss design, build quality, thickness, durability, fold design, and premium feel.",
        "customer_recommendation": "Balance thin design with durability, battery, and feature retention.",
        "profit_recommendation": "Use premium materials and durability as value justification.",
        "business_impact": "Medium customer perception impact.",
        "priority": "Medium"
    },
    "Customer Support / Warranty": {
        "customer_signal": "Users mention repair, warranty, replacement, service support, and after-sales trust.",
        "customer_recommendation": "Improve warranty clarity, repair experience, and service reliability.",
        "profit_recommendation": "Better after-sales support can increase trust, retention, and repeat purchases.",
        "business_impact": "Medium trust and retention impact.",
        "priority": "Medium"
    }
}


def get_goal_relevance(row, category):
    sentiment = str(row.get("sentiment_label", "")).lower()
    issue = str(row.get("issue_category", ""))

    if sentiment == "negative":
        return "customer_satisfaction"

    if issue in ["Price / Value", "Positive Feedback"]:
        return "profit"

    if category in ["Battery / Charging", "S-Pen / Features", "Camera", "Display / Screen", "Price / Value"]:
        return "balanced"

    return "balanced"


def build_strategy_chunk(row):
    category = row["issue_category"]
    rules = STRATEGY_RULES.get(category)

    if not rules:
        return None

    engagement = float(row.get("like_count", 0) or 0) + float(row.get("reply_count", 0) or 0)

    goal_relevance = get_goal_relevance(row, category)

    strategy_text = (
        f"Strategic Evidence for Samsung Product Roadmap\n"
        f"Issue Category: {category}\n"
        f"Sentiment: {row.get('sentiment_label', '')}\n"
        f"Topic: {row.get('topic_name', '')}\n"
        f"User Comment: {row.get('clean_comment', '')}\n"
        f"Customer Signal: {rules['customer_signal']}\n"
        f"Customer Satisfaction Recommendation: {rules['customer_recommendation']}\n"
        f"Profit Recommendation: {rules['profit_recommendation']}\n"
        f"Business Impact: {rules['business_impact']}\n"
        f"Priority: {rules['priority']}\n"
        f"Goal Relevance: {goal_relevance}\n"
        f"Engagement Score: {engagement}\n"
    )

    return {
        "comment_id": row.get("comment_id", ""),
        "video_id": row.get("video_id", ""),
        "video_title": row.get("video_title", ""),
        "issue_category": category,
        "sentiment_label": row.get("sentiment_label", ""),
        "topic_name": row.get("topic_name", ""),
        "clean_comment": row.get("clean_comment", ""),
        "like_count": row.get("like_count", 0),
        "reply_count": row.get("reply_count", 0),
        "engagement_total": engagement,
        "customer_signal": rules["customer_signal"],
        "customer_recommendation": rules["customer_recommendation"],
        "profit_recommendation": rules["profit_recommendation"],
        "business_impact": rules["business_impact"],
        "priority": rules["priority"],
        "goal_relevance": goal_relevance,
        "strategy_text": strategy_text
    }


def main():
    df = pd.read_csv(INPUT_PATH)

    df = df[df["language"] == "en"].copy()
    df = df.dropna(subset=["clean_comment"])
    df = df[df["word_count"] >= 3].copy()

    useful_categories = list(STRATEGY_RULES.keys())
    df = df[df["issue_category"].isin(useful_categories)].copy()

    # Prioritize useful evidence:
    # negative comments, neutral complaints, high engagement, and feature/value discussions
    df["engagement_total"] = (
        df["like_count"].fillna(0).astype(float) +
        df["reply_count"].fillna(0).astype(float)
    )

    df = df.sort_values(
        by=["engagement_total", "category_confidence", "word_count"],
        ascending=False
    )

    rows = []

    for _, row in df.iterrows():
        chunk = build_strategy_chunk(row)
        if chunk:
            rows.append(chunk)

    strategy_df = pd.DataFrame(rows)

    os.makedirs("data/processed", exist_ok=True)
    strategy_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Strategy evidence built!")
    print("Saved to:", OUTPUT_PATH)
    print("Total strategy evidence rows:", len(strategy_df))

    print("\nGoal relevance counts:")
    print(strategy_df["goal_relevance"].value_counts())

    print("\nTop issue categories:")
    print(strategy_df["issue_category"].value_counts().head(10))


if __name__ == "__main__":
    main()