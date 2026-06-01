# Samsung Feedback RAG Advisor

Samsung Feedback RAG Advisor is a university NLP and RAG project that analyzes YouTube comments about Samsung products. The system cleans raw comments, performs sentiment analysis, classifies customer issues, extracts keywords and named entities, discovers topics, retrieves evidence using RAG, and generates product strategy recommendations.

The project is designed to support a dashboard or report that explains what customers are saying about Samsung phones, which issues appear most often, and what actions could improve customer satisfaction and business value.

## Features

- YouTube comment collection using the YouTube Data API
- Text preprocessing and emoji sentiment features
- Spell correction for cleaned comments
- Sentiment analysis using VADER
- Hybrid issue classification using keywords and sentence embeddings
- Keyword extraction with TF-IDF
- General topic modeling with LDA
- Sentiment-specific topic modeling for positive, negative, and neutral comments
- Named entity recognition for brands, products, and competitors
- RAG retrieval over customer feedback evidence
- OpenAI-based summaries and answer generation
- Strategy recommendation generation
- Agent routing for different analysis questions
- MLflow monitoring and artifact logging

## Project Structure

```text
.
|-- data/
|   |-- raw/
|   |   `-- youtube_comments.csv
|   `-- processed/
|       |-- clean_comments.csv
|       |-- comments_with_spellcheck.csv
|       |-- comments_with_sentiment.csv
|       |-- comments_with_categories.csv
|       |-- comments_with_topics.csv
|       |-- comments_with_sentiment_topics.csv
|       |-- comments_with_ner.csv
|       |-- topic_keywords.csv
|       |-- topic_keywords_by_sentiment.csv
|       |-- rag_retrieval_results.csv
|       |-- rag_answers.csv
|       |-- strategy_evidence.csv
|       `-- strategy_rag_results.csv
|-- src/
|   |-- youtube_comment_scraper.py
|   |-- preprocessing.py
|   |-- spell_check.py
|   |-- sentiment_analysis.py
|   |-- issue_classifier.py
|   |-- keyword_extraction.py
|   |-- topic_modeling.py
|   |-- topic_modeling_by_sentiment.py
|   |-- ner_extraction.py
|   |-- rag_pipeline.py
|   |-- rag_answer_generator.py
|   |-- strategy_evidence_builder.py
|   |-- strategy_rag.py
|   |-- agent_router.py
|   |-- mlflow_monitoring.py
|   `-- run_pipeline.py
|-- .env.example
`-- README.md
```

Note: `run_pipeline.py` expects `src/topic_modeling_by_sentiment.py` to exist. If that file is not present locally, add it before running the full core pipeline.

## Pipeline Order

The core NLP pipeline runs in this order:

```text
Preprocessing
-> Spell check
-> Sentiment analysis
-> Issue classification
-> Keyword extraction
-> Topic modeling
-> Topic modeling by sentiment
-> NER extraction
```

The extended RAG and monitoring stages run after the core NLP pipeline:

```text
LLM summarization
-> RAG retrieval
-> RAG retrieval evaluation
-> RAG answer generation
-> BGE reranker evaluation
-> Strategy evidence builder
-> Strategy RAG
-> Agent router
-> MLflow monitoring
```

## Setup

Create and activate a Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the main dependencies:

```powershell
pip install pandas numpy scikit-learn sentence-transformers vaderSentiment langdetect pyspellchecker spacy python-dotenv openai mlflow google-api-python-client
python -m spacy download en_core_web_sm
```

Create a `.env` file from `.env.example`:

```powershell
copy .env.example .env
```

Then fill in:

```text
YOUTUBE_API_KEY=your_youtube_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=your_openai_model_here
```

Do not commit `.env` because it contains private API keys.

## Running the Project

To run the core NLP pipeline:

```powershell
python src\run_pipeline.py
```

To run the core pipeline plus RAG, strategy, agent routing, and MLflow monitoring:

```powershell
python src\run_pipeline.py --include-rag
```

The preprocessing stage limits the dataset to 15,000 cleaned comments by default to keep the pipeline fast enough for experimentation and demos.

To run with 10,000 comments instead:

```powershell
$env:MAX_COMMENTS="10000"
python src\run_pipeline.py
```

To disable the limit and use all cleaned comments:

```powershell
$env:MAX_COMMENTS="0"
python src\run_pipeline.py
```

## Running Individual Stages

Each stage can also be run manually:

```powershell
python src\preprocessing.py
python src\spell_check.py
python src\sentiment_analysis.py
python src\issue_classifier.py
python src\keyword_extraction.py
python src\topic_modeling.py
python src\ner_extraction.py
```

For RAG and strategy outputs:

```powershell
python src\rag_pipeline.py
python src\rag_answer_generator.py
python src\strategy_evidence_builder.py
python src\strategy_rag.py
python src\agent_router.py
python src\mlflow_monitoring.py
```

## Important Outputs

| File | Purpose |
| --- | --- |
| `data/processed/clean_comments.csv` | Cleaned comment dataset |
| `data/processed/comments_with_spellcheck.csv` | Spell-corrected comments |
| `data/processed/comments_with_sentiment.csv` | Sentiment scores and labels |
| `data/processed/comments_with_categories.csv` | Issue category classification |
| `data/processed/top_keywords_overall.csv` | Overall extracted keywords |
| `data/processed/top_keywords_by_category.csv` | Keywords grouped by issue category |
| `data/processed/comments_with_topics.csv` | General LDA topic modeling output |
| `data/processed/topic_keywords.csv` | General topic keywords |
| `data/processed/comments_with_sentiment_topics.csv` | Topic modeling output split by sentiment |
| `data/processed/topic_keywords_by_sentiment.csv` | Topic keywords for positive, negative, and neutral comments |
| `data/processed/comments_with_ner.csv` | Comments with named entities |
| `data/processed/ner_entities.csv` | Extracted entity table |
| `data/processed/rag_retrieval_results.csv` | Retrieved evidence for RAG queries |
| `data/processed/rag_answers.csv` | Generated RAG answers |
| `data/processed/strategy_evidence.csv` | Evidence used for strategy recommendations |
| `data/processed/strategy_rag_results.csv` | Product strategy RAG results |
| `data/processed/agent_router_results.csv` | Agent routing test outputs |

## Method Summary

The project performs topic modeling at two levels:

1. General topic modeling across useful English comments.
2. Sentiment-specific topic modeling for positive, negative, and neutral comments.

This allows the system to identify what users discuss overall, and also what they discuss positively, negatively, and neutrally.

Topic labels are manually interpreted based on the top words generated by LDA. Because LDA is unsupervised, these labels should be treated as descriptive interpretations rather than fixed ground-truth classes.

## MLflow Monitoring

MLflow logs metrics and output files from the pipeline, including:

- total processed comments
- sentiment distribution
- issue category distribution
- RAG evaluation metrics
- BGE reranker evaluation metrics
- generated summaries
- agent routing outputs
- keyword, topic, sentiment-topic, RAG, and strategy CSV artifacts

Run monitoring with:

```powershell
python src\mlflow_monitoring.py
```

MLflow tracking files are saved in:

```text
mlruns/
```

## Dashboard Direction

The recommended dashboard should be an analytics interface, not a marketing page. It should include:

- dataset overview
- sentiment distribution
- top issue categories
- general topics and sentiment-specific topics
- keyword trends
- named entities
- RAG question-answer panel
- supporting evidence comments
- strategy recommendations
- pipeline run status
- MLflow monitoring summary

## Academic Report Wording

Suggested wording:

```text
The system performs both general topic modeling and sentiment-specific topic modeling. General topic modeling identifies broad themes across the overall comment dataset, while sentiment-specific topic modeling identifies separate themes within positive, negative, and neutral comments. Topic labels were manually interpreted based on the top words generated by LDA.
```

## Notes

- `.env` should not be committed.
- `mlruns/` should not be committed unless specifically required.
- `__pycache__/` and `.pyc` files are generated automatically and should not be committed.
- Large generated files can be regenerated from the pipeline if the raw data and scripts are available.
