import json
import textwrap

import pandas as pd


DETAIL_PATH = "data/processed/deepeval_rag_detailed.csv"
CASES_PATH = "data/processed/deepeval_rag_cases.json"
TARGET_QUERY = "What evidence is there about display durability?"

QUALITATIVE_LABELS = [
    ("Weak", "Discusses future phone form factors, not durability."),
    ("Relevant", "Directly mentions accidental touches and cracking."),
    ("Relevant", "Reports a screen durability test and comparison."),
    ("Partial", "Mentions lifespan and refresh rate, but lacks clear evidence."),
    ("Not relevant", "Discusses lock-screen customisation rather than durability."),
]


def comment_from_context(context):
    first_line = str(context).splitlines()[0]
    return first_line.removeprefix("Comment: ").strip()


def shorten(value, width=70):
    return textwrap.shorten(
        " ".join(str(value).split()),
        width=width,
        placeholder="...",
    )


def main():
    detailed = pd.read_csv(DETAIL_PATH)
    metric_row = detailed[
        (detailed["query"] == TARGET_QUERY)
        & (detailed["metric"] == "ContextualRelevancyMetric")
    ].iloc[0]

    with open(CASES_PATH, encoding="utf-8") as handle:
        cases = json.load(handle)

    case = next(item for item in cases if item["query"] == TARGET_QUERY)

    print("=" * 78)
    print("RETRIEVAL LIMITATION: LOW-RELEVANCE EVIDENCE EXAMPLE")
    print("=" * 78)
    print(f"RAG system:                 {metric_row['rag_system']}")
    print(f"Query:                      {TARGET_QUERY}")
    print(f"Contextual relevancy score: {float(metric_row['score']):.2f}")
    print(f"Passing threshold:          {float(metric_row['threshold']):.2f}")
    print("Evaluation result:          FAIL")
    print()
    print("Top-5 retrieved evidence: qualitative inspection")

    rows = []
    for rank, (context, label_data) in enumerate(
        zip(case["retrieval_context"], QUALITATIVE_LABELS),
        start=1,
    ):
        label, explanation = label_data
        rows.append(
            {
                "Rank": rank,
                "Assessment": label,
                "Retrieved comment": shorten(comment_from_context(context)),
                "Reason": explanation,
            }
        )

    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print("Diagnostic finding:")
    print(
        "The retriever matched the broad Display / Screen category, but some "
        "results did not address durability specifically. This demonstrates "
        "the limitation of broad category and semantic matches for narrowly "
        "worded queries."
    )
    print()
    print("Note: assessments are qualitative inspection, not additional ground truth.")


if __name__ == "__main__":
    main()
