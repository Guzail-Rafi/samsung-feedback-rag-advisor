import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from agent_router import AVAILABLE_AGENTS, route_intent, route_intent_rules
from routing_evaluation_data import ROUTING_QRELS


OUTPUT_DIR = Path("data/processed")
DETAILED_OUTPUT_PATH = OUTPUT_DIR / "evaluation_router_before_after.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "evaluation_router_before_after_summary.csv"
CONFUSION_OUTPUT_PATH = OUTPUT_DIR / "evaluation_langchain_router_confusion_matrix.csv"
REPORT_OUTPUT_PATH = OUTPUT_DIR / "evaluation_langchain_router_report.txt"
MAX_WORKERS = 5


def route_and_measure(router_name, query, expected_agent):
    start = time.perf_counter()
    decision = route_intent_rules(query) if router_name == "rule_based" else route_intent(query)
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "router": router_name,
        "query": query,
        "expected_agent": expected_agent,
        "selected_agent": decision["selected_agent"],
        "correct": int(decision["selected_agent"] == expected_agent),
        "latency_ms": round(latency_ms, 2),
        "confidence": decision["confidence"],
        "routing_method": decision["routing_method"],
        "router_model": decision["router_model"],
        "normalized_query": decision["normalized_query"],
        "reason": decision["reason"],
    }


def evaluate_rule_router():
    return [
        route_and_measure("rule_based", query, expected_agent)
        for query, expected_agent in ROUTING_QRELS
    ]


def evaluate_langchain_router():
    rows = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(route_and_measure, "langchain_llm", query, expected_agent)
            for query, expected_agent in ROUTING_QRELS
        ]

        for future in as_completed(futures):
            rows.append(future.result())

    query_order = {query: index for index, (query, _) in enumerate(ROUTING_QRELS)}
    return sorted(rows, key=lambda row: query_order[row["query"]])


def build_summary(results):
    return (
        results.groupby("router", as_index=False)
        .agg(
            query_count=("query", "count"),
            correct_routes=("correct", "sum"),
            routing_accuracy=("correct", "mean"),
            avg_latency_ms=("latency_ms", "mean"),
            median_latency_ms=("latency_ms", "median"),
            avg_confidence=("confidence", "mean"),
            fallback_count=(
                "routing_method",
                lambda values: sum("fallback" in value for value in values),
            ),
        )
        .round(4)
    )


def build_report(results, summary):
    rule_summary = summary[summary["router"] == "rule_based"].iloc[0]
    llm_summary = summary[summary["router"] == "langchain_llm"].iloc[0]
    llm_rows = results[results["router"] == "langchain_llm"]
    corrected = results.pivot(
        index=["query", "expected_agent"],
        columns="router",
        values="correct",
    ).reset_index()
    corrected = corrected[
        (corrected["rule_based"] == 0) & (corrected["langchain_llm"] == 1)
    ]
    llm_errors = llm_rows[llm_rows["correct"] == 0]

    lines = [
        "=" * 78,
        "LANGCHAIN LLM ROUTER: BEFORE AND AFTER EVALUATION",
        "=" * 78,
        "",
        summary.to_string(index=False),
        "",
        "Report-ready findings:",
        (
            f"- Rule-based routing accuracy: {rule_summary['routing_accuracy'] * 100:.1f}% "
            f"({int(rule_summary['correct_routes'])}/{int(rule_summary['query_count'])})."
        ),
        (
            f"- LangChain LLM routing accuracy: {llm_summary['routing_accuracy'] * 100:.1f}% "
            f"({int(llm_summary['correct_routes'])}/{int(llm_summary['query_count'])})."
        ),
        (
            "- Accuracy change after LangChain LLM routing: "
            f"{(llm_summary['routing_accuracy'] - rule_summary['routing_accuracy']) * 100:+.1f} percentage points."
        ),
        (
            f"- Average LangChain router latency: {llm_summary['avg_latency_ms']:.2f} ms "
            f"versus {rule_summary['avg_latency_ms']:.2f} ms for deterministic rules."
        ),
        (
            f"- LangChain fallback count: {int(llm_summary['fallback_count'])}. "
            "A fallback indicates that structured LLM routing failed and deterministic rules were used."
        ),
        "",
        "Queries corrected by LangChain:",
        corrected[["query", "expected_agent"]].to_string(index=False),
        "",
        "Remaining LangChain routing errors:",
        llm_errors[["query", "expected_agent", "selected_agent", "reason"]].to_string(index=False),
        "",
        "Interpretation:",
        "- The LLM router improves semantic understanding of paraphrases and ambiguous wording.",
        "- Local Llama preserves LLM routing when OpenAI fails; deterministic rules remain the final fallback.",
        "- The accuracy improvement must be considered against added API latency and cost.",
        "",
        "Generated artifacts:",
        f"- {DETAILED_OUTPUT_PATH}",
        f"- {SUMMARY_OUTPUT_PATH}",
        f"- {CONFUSION_OUTPUT_PATH}",
        f"- {REPORT_OUTPUT_PATH}",
    ]

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Evaluating deterministic rule router...")
    rule_rows = evaluate_rule_router()

    print("Evaluating LangChain structured-output LLM router...")
    llm_rows = evaluate_langchain_router()

    results = pd.DataFrame(rule_rows + llm_rows)
    summary = build_summary(results)
    llm_results = results[results["router"] == "langchain_llm"]
    confusion = pd.crosstab(
        llm_results["expected_agent"],
        llm_results["selected_agent"],
        rownames=["expected_agent"],
        colnames=["selected_agent"],
        dropna=False,
    ).reindex(index=AVAILABLE_AGENTS, columns=AVAILABLE_AGENTS, fill_value=0)

    results.to_csv(DETAILED_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    confusion.to_csv(CONFUSION_OUTPUT_PATH, encoding="utf-8-sig")

    report = build_report(results, summary)
    REPORT_OUTPUT_PATH.write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
