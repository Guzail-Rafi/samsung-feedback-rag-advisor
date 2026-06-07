import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS", "180")
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS", "420")

import pandas as pd
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from web_rag_bridge import (
    run_feedback_rag,
    run_strategy_rag_live,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DETAIL_PATH = OUTPUT_DIR / "deepeval_rag_detailed.csv"
SUMMARY_PATH = OUTPUT_DIR / "deepeval_rag_summary.csv"
REPORT_PATH = OUTPUT_DIR / "deepeval_rag_report.txt"
CASES_PATH = OUTPUT_DIR / "deepeval_rag_cases.json"

DEFAULT_JUDGE_MODEL = "gpt-4.1-mini"
DEFAULT_THRESHOLD = 0.7
DEFAULT_MAX_PER_RAG = 5
DEFAULT_METRIC_RETRIES = 2

FEEDBACK_QUERIES = [
    "Why are users unhappy about the S-Pen?",
    "What are users saying about Samsung battery life?",
    "What complaints mention camera quality?",
    "What do users think about Galaxy AI and Gemini?",
    "What evidence is there about display durability?",
]

STRATEGY_QUERIES = [
    "How should Samsung design the S27 Ultra for maximum customer satisfaction?",
    "How should Samsung design the S27 Ultra for maximum profit?",
    "What features should Samsung prioritize in the S27 Ultra?",
    "What product roadmap should Samsung follow for the next Ultra phone?",
    "How can Samsung reduce customer complaints in the next flagship?",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Feedback RAG and Strategy RAG answer quality with DeepEval."
    )
    parser.add_argument(
        "--max-per-rag",
        type=int,
        default=DEFAULT_MAX_PER_RAG,
        help="Maximum number of queries evaluated for each RAG system.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("DEEPEVAL_MODEL", DEFAULT_JUDGE_MODEL),
        help="OpenAI model used by DeepEval as the LLM judge.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum passing score for each metric.",
    )
    parser.add_argument(
        "--reuse-cases",
        action="store_true",
        help="Reuse previously generated answers and contexts from the case cache.",
    )
    parser.add_argument(
        "--metric-retries",
        type=int,
        default=DEFAULT_METRIC_RETRIES,
        help="Maximum attempts for each DeepEval metric after transient API failures.",
    )
    return parser.parse_args()


def clean_text(value):
    return str(value or "").strip()


def format_feedback_context(evidence):
    return [
        "\n".join(
            [
                f"Comment: {clean_text(row.get('comment'))}",
                f"Sentiment: {clean_text(row.get('sentiment'))}",
                f"Issue category: {clean_text(row.get('issue_category'))}",
                f"Topic: {clean_text(row.get('topic'))}",
                f"Video: {clean_text(row.get('video'))}",
            ]
        )
        for row in evidence
    ]


def format_strategy_context(evidence):
    return [
        "\n".join(
            [
                f"Comment: {clean_text(row.get('comment'))}",
                f"Sentiment: {clean_text(row.get('sentiment'))}",
                f"Issue category: {clean_text(row.get('issue_category'))}",
                f"Customer signal: {clean_text(row.get('customer_signal'))}",
                f"Customer recommendation: {clean_text(row.get('customer_recommendation'))}",
                f"Profit recommendation: {clean_text(row.get('profit_recommendation'))}",
                f"Business impact: {clean_text(row.get('business_impact'))}",
                f"Priority: {clean_text(row.get('priority'))}",
            ]
        )
        for row in evidence
    ]


def generate_case(rag_system, query):
    start = time.perf_counter()

    if rag_system == "feedback_rag":
        result = run_feedback_rag(query, query, "No prior conversation.")
        retrieval_context = format_feedback_context(result["evidence"])
    elif rag_system == "strategy_rag":
        result = run_strategy_rag_live(query, query, "No prior conversation.")
        retrieval_context = format_strategy_context(result["evidence"])
    else:
        raise ValueError(f"Unsupported RAG system: {rag_system}")

    return {
        "rag_system": rag_system,
        "query": query,
        "answer": result["answer"],
        "retrieval_context": retrieval_context,
        "generation_latency_ms": round((time.perf_counter() - start) * 1000, 2),
        "retrieval_score": result.get("retrieval", {}).get("score"),
    }


def generate_cases(max_per_rag):
    cases = []

    for rag_system, queries in [
        ("feedback_rag", FEEDBACK_QUERIES),
        ("strategy_rag", STRATEGY_QUERIES),
    ]:
        for index, query in enumerate(queries[:max_per_rag], start=1):
            print(f"Generating {rag_system} case {index}/{min(max_per_rag, len(queries))}: {query}")
            cases.append(generate_case(rag_system, query))

    CASES_PATH.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    return cases


def load_cases(max_per_rag):
    if not CASES_PATH.exists():
        raise FileNotFoundError(
            f"Case cache not found: {CASES_PATH}. Run without --reuse-cases first."
        )

    cached = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    selected = []

    for rag_system in ["feedback_rag", "strategy_rag"]:
        selected.extend(
            [case for case in cached if case["rag_system"] == rag_system][:max_per_rag]
        )

    return selected


def build_metrics(judge_model, threshold):
    shared = {
        "model": judge_model,
        "threshold": threshold,
        "include_reason": True,
        "async_mode": False,
    }
    return [
        AnswerRelevancyMetric(**shared),
        FaithfulnessMetric(**shared),
        ContextualRelevancyMetric(**shared),
    ]


def evaluate_case(case_data, judge_model, threshold, metric_retries):
    test_case = LLMTestCase(
        input=case_data["query"],
        actual_output=case_data["answer"],
        retrieval_context=case_data["retrieval_context"],
        name=f"{case_data['rag_system']}: {case_data['query']}",
        metadata={"rag_system": case_data["rag_system"]},
    )
    rows = []

    for metric in build_metrics(judge_model, threshold):
        print(f"  Measuring {metric.__class__.__name__}...")
        error = ""
        score = None
        reason = ""
        passed = 0

        for attempt in range(1, metric_retries + 1):
            try:
                metric.measure(test_case)
                score = float(metric.score)
                reason = clean_text(metric.reason)
                passed = int(metric.is_successful())
                error = ""
                break
            except Exception as exc:
                error = f"{exc.__class__.__name__}: {exc}"
                if attempt < metric_retries:
                    print(f"    Attempt {attempt} failed; retrying...")
                    time.sleep(2)

        rows.append(
            {
                "rag_system": case_data["rag_system"],
                "query": case_data["query"],
                "metric": metric.__class__.__name__,
                "score": score,
                "threshold": threshold,
                "passed": passed,
                "judge_model": judge_model,
                "reason": reason,
                "error": error,
                "retrieval_context_count": len(case_data["retrieval_context"]),
                "retrieval_score": case_data.get("retrieval_score"),
                "generation_latency_ms": case_data["generation_latency_ms"],
                "answer": case_data["answer"],
            }
        )

    return rows


def build_summary(detailed):
    valid = detailed[detailed["score"].notna()].copy()

    if valid.empty:
        return pd.DataFrame(
            columns=[
                "rag_system",
                "metric",
                "evaluated_cases",
                "average_score",
                "minimum_score",
                "pass_rate",
            ]
        )

    return (
        valid.groupby(["rag_system", "metric"], as_index=False)
        .agg(
            evaluated_cases=("query", "count"),
            average_score=("score", "mean"),
            minimum_score=("score", "min"),
            pass_rate=("passed", "mean"),
        )
        .round(4)
    )


def build_report(detailed, summary, judge_model, threshold):
    valid = detailed[detailed["score"].notna()].copy()
    failed = valid[valid["passed"] == 0].sort_values("score")
    errors = detailed[detailed["error"].astype(bool)]
    overall_score = float(valid["score"].mean()) if not valid.empty else 0.0
    overall_pass_rate = float(valid["passed"].mean()) if not valid.empty else 0.0
    completion_rate = len(valid) / len(detailed) if len(detailed) else 0.0

    lines = [
        "=" * 78,
        "DEEPEVAL RAG GENERATION-QUALITY EVALUATION",
        "=" * 78,
        "",
        "Methodology:",
        f"- Judge model: {judge_model}",
        f"- Passing threshold: {threshold:.2f}",
        "- AnswerRelevancyMetric measures whether the generated answer addresses the query.",
        "- FaithfulnessMetric measures whether answer claims are supported by retrieved evidence.",
        "- ContextualRelevancyMetric measures whether retrieved evidence is relevant to the query.",
        "- Contextual precision and recall are excluded because no human-labelled ideal contexts are available.",
        "- Scores are LLM-as-a-judge estimates and should be supplemented with manual inspection.",
        "",
        "Summary by RAG system and metric:",
        summary.to_string(index=False),
        "",
        f"Overall average score: {overall_score:.4f}",
        f"Overall metric pass rate: {overall_pass_rate * 100:.1f}%",
        f"Metric completion rate: {completion_rate * 100:.1f}%",
        f"Metric errors: {len(errors)}",
        "",
        "Lowest-scoring cases for qualitative inspection:",
        failed[
            ["rag_system", "query", "metric", "score", "reason"]
        ].head(8).to_string(index=False),
        "",
        "Report-ready interpretation:",
        "- Retrieval ablation results explain which retrieval components improve ranking quality.",
        "- DeepEval results add end-to-end evidence about answer relevance, grounding, and retrieved-context usefulness.",
        "- LangSmith traces can be used to inspect the exact execution stages behind low-scoring examples.",
        "- DeepEval scores should not be described as human ground truth because they are generated by an LLM judge.",
        "",
        "Generated artifacts:",
        f"- {DETAIL_PATH}",
        f"- {SUMMARY_PATH}",
        f"- {REPORT_PATH}",
        f"- {CASES_PATH}",
    ]

    return "\n".join(lines)


def main():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError("OPENAI_API_KEY is required for DeepEval's OpenAI judge.")

    if args.max_per_rag < 1:
        raise ValueError("--max-per-rag must be at least 1.")

    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1.")

    if args.metric_retries < 1:
        raise ValueError("--metric-retries must be at least 1.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.max_per_rag) if args.reuse_cases else generate_cases(args.max_per_rag)
    rows = []

    for index, case_data in enumerate(cases, start=1):
        print(
            f"\nEvaluating case {index}/{len(cases)} "
            f"({case_data['rag_system']}): {case_data['query']}"
        )
        rows.extend(
            evaluate_case(
                case_data,
                args.judge_model,
                args.threshold,
                args.metric_retries,
            )
        )

    detailed = pd.DataFrame(rows)
    summary = build_summary(detailed)
    detailed.to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    report = build_report(detailed, summary, args.judge_model, args.threshold)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
