import os
import pandas as pd
from dotenv import load_dotenv
from openai_client import generate_chat_response, get_openai_client


# =========================
# 1. CONFIG
# =========================

load_dotenv()

INPUT_PATH = "data/processed/comments_with_ner.csv"
OUTPUT_PATH = "data/processed/llm_summaries.csv"

client = get_openai_client()


# =========================
# 2. SAMPLE COMMENTS
# =========================

def sample_comments(df, n=40):
    """
    Selects a balanced sample of comments.
    Prioritizes comments with likes/replies and keeps useful text only.
    """

    df = df.dropna(subset=["clean_comment"]).copy()
    df = df[df["word_count"] >= 3]

    if df.empty:
        return ""

    df["engagement_total"] = (
        df["like_count"].fillna(0).astype(float) +
        df["reply_count"].fillna(0).astype(float)
    )

    df = df.sort_values(
        by=["engagement_total", "word_count"],
        ascending=False
    )

    sample_df = df.head(n)

    comments = []

    for _, row in sample_df.iterrows():
        comments.append(
            f"- Comment: {row['clean_comment']}\n"
            f"  Sentiment: {row['sentiment_label']}\n"
            f"  Issue Category: {row['issue_category']}\n"
            f"  Topic: {row['topic_name']}\n"
            f"  Video: {row['video_title']}"
        )

    return "\n".join(comments)


# =========================
# 3. OPENAI SUMMARY
# =========================

def generate_summary(title, comments_text):
    """
    Sends selected comments to the configured LLM and generates an academic summary.
    """

    if not comments_text.strip():
        return "No enough comments available for this summary."

    system_prompt = """
You are an academic NLP assistant for a university project.

You must summarize only the provided YouTube comment evidence.
Do not invent facts.
Do not overgeneralize beyond the comments.
Use careful wording such as "the sampled comments suggest".
Mention mixed opinions when present.
Keep the summary concise, professional, and suitable for a project report.
"""

    user_prompt = f"""
Summary Title:
{title}

YouTube Comment Evidence:
{comments_text}

Write:
1. Short overview.
2. Main user concerns or opinions.
3. Sentiment pattern.
4. Key product issues mentioned.
5. One report-ready insight.
"""

    return generate_chat_response(
        client=client,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ],
        temperature=0.2,
        max_completion_tokens=700,
    )


# =========================
# 4. MAIN
# =========================

def main():
    df = pd.read_csv(INPUT_PATH)

    # Use English comments only for summarization
    df = df[df["language"] == "en"].copy()

    summaries = []

    summary_tasks = []

    # Overall summary
    summary_tasks.append({
        "summary_type": "overall_feedback",
        "title": "Overall Samsung User Feedback Summary",
        "data": df
    })

    # Negative comments summary
    summary_tasks.append({
        "summary_type": "negative_feedback",
        "title": "Negative Feedback Summary",
        "data": df[df["sentiment_label"] == "negative"]
    })

    # Positive comments summary
    summary_tasks.append({
        "summary_type": "positive_feedback",
        "title": "Positive Feedback Summary",
        "data": df[df["sentiment_label"] == "positive"]
    })

    # Issue-wise summaries
    issue_categories = [
        "Battery / Charging",
        "AI / Gemini",
        "S-Pen / Features",
        "Camera",
        "Display / Screen",
        "Price / Value",
        "Software / One UI"
    ]

    for category in issue_categories:
        category_df = df[df["issue_category"] == category]

        summary_tasks.append({
            "summary_type": f"issue_{category}",
            "title": f"{category} Summary",
            "data": category_df
        })

    # Generate summaries
    for task in summary_tasks:
        print("\n====================================")
        print("Generating summary:", task["title"])
        print("====================================")

        comments_text = sample_comments(task["data"], n=40)
        summary = generate_summary(task["title"], comments_text)

        summaries.append({
            "summary_type": task["summary_type"],
            "summary_title": task["title"],
            "comment_count_used": len(task["data"]),
            "summary": summary
        })

        print(summary)

    output_df = pd.DataFrame(summaries)

    os.makedirs("data/processed", exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\nLLM summarization completed!")
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
