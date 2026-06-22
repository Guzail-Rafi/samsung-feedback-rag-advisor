import argparse
import subprocess
import sys
from pathlib import Path


CORE_STEPS = [
    ("Preprocessing", "preprocessing.py"),
    ("Spell check", "spell_check.py"),
    ("Sentiment analysis", "sentiment_analysis.py"),
    ("Issue classification", "issue_classifier.py"),
    ("Keyword extraction", "keyword_extraction.py"),
    ("Topic modeling", "topic_modeling.py"),
    ("Topic modeling by sentiment", "topic_modeling_by_sentiment.py"),
    ("NER extraction", "ner_extraction.py"),
    ("User segmentation", "user_segmentation.py"),
]

RAG_STEPS = [
    ("LLM summarization", "llm_summarizer.py"),
    ("RAG retrieval", "rag_pipeline.py"),
    ("RAG retrieval evaluation", "rag_evaluation.py"),
    ("RAG answer generation", "rag_answer_generator.py"),
    ("BGE reranker evaluation", "rag_bge_reranker_evaluation.py"),
    ("Strategy evidence builder", "strategy_evidence_builder.py"),
    ("Strategy RAG", "strategy_rag.py"),
    ("Agent router", "agent_router.py"),
    ("MLflow monitoring", "mlflow_monitoring.py"),
]


def run_step(name, script_path, project_root):
    print(f"\n=== {name} ===")
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=project_root,
        check=True
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the Samsung feedback NLP/RAG pipeline."
    )
    parser.add_argument(
        "--include-rag",
        action="store_true",
        help="Also run the RAG, summary, strategy, and monitoring stages."
    )
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    steps = CORE_STEPS + (RAG_STEPS if args.include_rag else [])

    for name, script_name in steps:
        run_step(name, src_dir / script_name, project_root)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
