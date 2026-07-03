# Galaxy Insight: Samsung Feedback RAG Advisor

![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=111)
![Python](https://img.shields.io/badge/Python-NLP%20%2B%20RAG-3776AB?logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Monitoring-0194E2)
![Status](https://img.shields.io/badge/Status-Final%20Project-success)

Galaxy Insight is a full-stack NLP and Retrieval-Augmented Generation project
that turns Samsung YouTube comments into an interactive product intelligence
dashboard. It combines sentiment analysis, issue classification, topic
modeling, user segmentation, named entity extraction, feedback RAG, strategy
RAG, document RAG, live agent routing, and MLflow monitoring.

The project is built so a reviewer can open the website, move through the
dashboard pages, and inspect the generated insights immediately. Live chat,
document upload, web-augmented strategy, and MLflow tracing are available after
the full Python and environment setup.

## Contents

- [What This Project Does](#what-this-project-does)
- [Quick Start: Open the Website](#quick-start-open-the-website)
- [Full Setup: Enable Live AI Features](#full-setup-enable-live-ai-features)
- [Pages to Explore](#pages-to-explore)
- [Example Questions](#example-questions)
- [Run the NLP and RAG Pipeline](#run-the-nlp-and-rag-pipeline)
- [Project Architecture](#project-architecture)
- [Important Outputs](#important-outputs)
- [Evaluation and Monitoring](#evaluation-and-monitoring)
- [Troubleshooting](#troubleshooting)

## What This Project Does

Galaxy Insight analyzes Samsung-related YouTube comments and converts them into
usable product, customer, and strategy intelligence.

| Area | What it does |
| --- | --- |
| Comment processing | Cleans raw comments, filters language, normalizes text, and prepares analysis-ready data. |
| Sentiment and issues | Labels comments by sentiment and classifies major complaint themes such as battery, camera, S-Pen, AI, display, and price. |
| Keywords and topics | Extracts TF-IDF keywords and discovers broader themes with topic modeling. |
| Entity extraction | Tracks Samsung products, competitors, features, and brands mentioned in the comments. |
| User segmentation | Groups comments into behavioral audience personas using TF-IDF, TruncatedSVD, and KMeans. |
| Feedback RAG | Answers customer-feedback questions using retrieved YouTube comment evidence. |
| Strategy RAG | Produces evidence-backed Samsung product and business recommendations. |
| Document RAG | Lets users upload Samsung-related documents and ask cited questions about them. |
| Agent routing | Sends each user question to the right analytics, RAG, document, or strategy agent. |
| Monitoring | Logs pipeline metrics, live traces, artifacts, and evaluation results in MLflow. |

## Quick Start: Open the Website

Use this path if you only want to launch the dashboard and click through the
project quickly.

### 1. Clone the repository

```powershell
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Install frontend dependencies

This project uses Next.js 16, which requires Node.js `20.9.0` or newer.

```powershell
npm install
```

### 3. Start the dashboard

```powershell
npm run dev
```

Open:

```text
http://localhost:3000
```

The dashboard pages can be explored immediately because the repository includes
prepared files in `data/processed/`. The live AI chat, document upload, and RAG
API routes need the full setup below.

## Full Setup: Enable Live AI Features

Use this path when you want `/advisor`, `/documents`, `/refinement`, live RAG,
and MLflow tracing to work end to end.

If you skipped the quick start, install the frontend dependencies first:

```powershell
npm install
```

### 1. Create a Python virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install Python dependencies

```powershell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The first RAG request may also download local embedding or reranking models used
by `sentence-transformers`.

### 3. Create your environment file

Windows PowerShell:

```powershell
copy .env.example .env
```

macOS or Linux:

```bash
cp .env.example .env
```

Then edit `.env` and add your keys:

```text
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=use_a_model_available_to_your_account
YOUTUBE_API_KEY=your_youtube_api_key_here
LANGSMITH_TRACING=false
MLFLOW_TRACING_ENABLED=true
```

Useful optional settings:

```text
LLM_FALLBACK_ENABLED=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
WEB_AUGMENTED_STRATEGY_ENABLED=true
WEB_SEARCH_REGION=ae-en
```

Do not commit `.env`. It is ignored because it contains private keys.

### 4. Optional: enable local Llama fallback

If you want the app to fall back to a local model when OpenAI is unavailable,
install Ollama and pull the fallback model:

```powershell
ollama pull llama3.2:3b
```

The fallback order is:

```text
OpenAI
-> local Ollama model
-> deterministic routing rules, only if both LLM routers fail
```

### 5. Start the full app

Start Next.js from the same terminal where the Python virtual environment is
active. This lets the Next.js API routes call the correct `python` executable.

```powershell
npm run dev
```

Open:

```text
http://localhost:3000
```

## Pages to Explore

| Route | What to check |
| --- | --- |
| `/` | Project overview, headline KPIs, RAG architecture, strategy summary, and monitoring snapshot. |
| `/advisor` | ChatGPT-style Samsung intelligence chat with agent routing across analytics, RAG, strategy, web research, and documents. |
| `/rag` | Feedback RAG answers, retrieved comments, confidence levels, and Precision@5 evaluation. |
| `/strategy` | Evidence-backed product strategy recommendations and roadmap priorities for Samsung. |
| `/refinement` | Interactive roadmap refinement where suggested changes are accepted, rejected, or converted into alternatives. |
| `/sentiment` | Positive, neutral, and negative feedback distribution. |
| `/issues` | Complaint categories such as battery, S-Pen, camera, AI, display, and price/value. |
| `/topics` | General and sentiment-specific topic modeling outputs. |
| `/keywords` | Top overall keywords and issue-specific keywords. |
| `/entities` | Samsung products, competitors, features, and brand mentions extracted from feedback. |
| `/segmentation` | Behavioral user personas generated from comment clusters. |
| `/documents` | Upload Samsung-related PDF, DOCX, TXT, Markdown, CSV, JSON, or HTML files and ask cited questions. |
| `/pipeline` | Processing stages and their generated artifacts. |
| `/monitoring` | Live MLflow trace summaries, pipeline runs, RAG metrics, and evaluation tables. |

## Example Questions

Try these in `/advisor` after completing the full setup:

```text
Give me an overall summary of Samsung feedback.
Why are users unhappy about the S-Pen?
What are users saying about battery life?
Which issues are most common?
What should Samsung prioritize for the next Ultra?
How should Samsung design the S27 Ultra?
How should Samsung price the next Ultra in the UAE using current offers?
Compare Samsung and iPhone signals in the comments.
```

After uploading a Samsung-related document in `/documents`, try:

```text
Summarize the uploaded document.
What product risks does this document mention?
Use the uploaded document to support a Samsung roadmap recommendation.
```

## Run the NLP and RAG Pipeline

The dashboard already includes generated outputs, so rerunning the pipeline is
not required just to view the website.

To rerun the core NLP pipeline:

```powershell
python src\run_pipeline.py
```

To rerun the extended pipeline with RAG, strategy generation, routing, and
MLflow monitoring:

```powershell
python src\run_pipeline.py --include-rag
```

By default, preprocessing limits the demo dataset to 15,000 cleaned comments so
the project remains practical for local runs.

To change the limit:

```powershell
$env:MAX_COMMENTS="10000"
python src\run_pipeline.py
```

To disable the limit:

```powershell
$env:MAX_COMMENTS="0"
python src\run_pipeline.py
```

Pipeline note: `src/run_pipeline.py` includes a sentiment-specific topic stage
named `src/topic_modeling_by_sentiment.py`. If your checkout does not include
that file, restore it before running the full core pipeline, or run the
available stages individually. This does not affect opening the dashboard with
the prepared outputs.

### Useful Individual Stages

```powershell
python src\preprocessing.py
python src\spell_check.py
python src\sentiment_analysis.py
python src\issue_classifier.py
python src\keyword_extraction.py
python src\topic_modeling.py
python src\ner_extraction.py
python src\user_segmentation.py
python src\rag_pipeline.py
python src\rag_answer_generator.py
python src\strategy_evidence_builder.py
python src\strategy_rag.py
python src\agent_router.py
python src\router_evaluation.py
python src\deepeval_rag_evaluation.py
python src\mlflow_monitoring.py
```

## MLflow UI

Start the MLflow server:

```powershell
npm run mlflow:ui
```

Open:

```text
http://127.0.0.1:5000
```

The `/monitoring` page reads real MLflow data from `data/mlflow.db` when it is
available. Live advisor requests and document RAG calls can create nested MLflow
traces when `MLFLOW_TRACING_ENABLED=true`.

## Project Architecture

```text
YouTube comments
-> Python NLP pipeline
-> processed CSV artifacts
-> embeddings and ChromaDB vector stores
-> Next.js dashboard
-> API routes
-> Python bridge scripts
-> analytics agents, Feedback RAG, Strategy RAG, Document RAG
-> OpenAI or Ollama generation
-> MLflow and optional LangSmith tracing
```

### Main Frontend Pieces

| Path | Purpose |
| --- | --- |
| `app/page.tsx` | Overview dashboard. |
| `app/components/AppShell.tsx` | Shared navigation and layout shell. |
| `app/components/AdvisorChat.tsx` | Main routed Samsung chat interface. |
| `app/components/DocumentChat.tsx` | Upload and chat UI for Samsung documents. |
| `app/api/advisor/route.ts` | Next.js route that calls `src/web_rag_bridge.py`. |
| `app/api/documents/route.ts` | Next.js route that calls `src/document_rag_bridge.py`. |
| `app/api/refinement/route.ts` | Next.js route that calls `src/roadmap_refinement_bridge.py`. |
| `app/api/mlflow/route.ts` | Next.js route that calls `src/mlflow_status_bridge.py`. |

### Main Python Pieces

| Path | Purpose |
| --- | --- |
| `src/run_pipeline.py` | Orchestrates the core and extended pipeline. |
| `src/preprocessing.py` | Cleans and prepares YouTube comments. |
| `src/sentiment_analysis.py` | Applies VADER sentiment scoring. |
| `src/issue_classifier.py` | Classifies customer complaint themes. |
| `src/topic_modeling.py` | Builds LDA topic models. |
| `src/user_segmentation.py` | Builds behavioral personas from comment clusters. |
| `src/ner_extraction.py` | Extracts products, brands, competitors, and features. |
| `src/rag_pipeline.py` | Retrieves feedback evidence. |
| `src/rag_answer_generator.py` | Generates grounded Feedback RAG answers. |
| `src/strategy_evidence_builder.py` | Builds strategy-ready evidence rows. |
| `src/strategy_rag.py` | Retrieves evidence and generates strategy recommendations. |
| `src/agent_router.py` | Routes questions to specialist agents. |
| `src/langchain_router.py` | LLM-based structured router with fallback behavior. |
| `src/document_rag_bridge.py` | Ingests uploaded Samsung documents and answers with citations. |
| `src/web_rag_bridge.py` | Live advisor bridge used by the dashboard. |
| `src/mlflow_monitoring.py` | Logs batch metrics and artifacts to MLflow. |

## Important Outputs

| File or folder | Purpose |
| --- | --- |
| `data/raw/youtube_comments.csv` | Raw collected YouTube comments. |
| `data/processed/clean_comments.csv` | Cleaned analysis-ready comments. |
| `data/processed/comments_with_sentiment.csv` | Sentiment labels and scores. |
| `data/processed/comments_with_categories.csv` | Issue category labels. |
| `data/processed/top_keywords_overall.csv` | Top extracted keywords. |
| `data/processed/top_keywords_by_category.csv` | Keywords grouped by issue category. |
| `data/processed/comments_with_topics.csv` | General topic modeling output. |
| `data/processed/topic_keywords.csv` | Topic keyword summaries. |
| `data/processed/comments_with_ner.csv` | Comments enriched with named entities. |
| `data/processed/ner_entities.csv` | Extracted entity table. |
| `data/processed/user_personas.csv` | Persona summaries and recommendations. |
| `data/processed/user_segmentation_assignments.csv` | Comment-level cluster assignments. |
| `data/processed/rag_retrieval_results.csv` | Retrieved evidence for Feedback RAG queries. |
| `data/processed/rag_answers.csv` | Generated Feedback RAG answers. |
| `data/processed/strategy_evidence.csv` | Strategy evidence generated from feedback. |
| `data/processed/strategy_rag_results.csv` | Strategy RAG recommendations. |
| `data/processed/agent_router_results.csv` | Agent routing test outputs. |
| `data/processed/deepeval_rag_detailed.csv` | Per-query DeepEval scores. |
| `data/processed/deepeval_rag_summary.csv` | Aggregated DeepEval results. |
| `data/vector_db/` | Local ChromaDB vector stores generated at runtime. |
| `data/uploaded_documents/` | Uploaded document storage generated at runtime. |
| `data/mlflow.db` | Local MLflow tracking database generated at runtime. |

Runtime-generated folders such as `data/vector_db/`, `data/uploaded_documents/`,
`mlruns/`, `mlartifacts/`, and `.next/` are ignored by Git.

## Evaluation and Monitoring

The project includes several evaluation layers:

| Evaluation | Purpose |
| --- | --- |
| Manual Precision@5 | Checks whether retrieved comments are relevant for Feedback RAG queries. |
| RAG retrieval metrics | Compares weighted, semantic, lexical, and reranker signals. |
| Strategy ablation | Compares strategy retrieval and recommendation variants. |
| Router evaluation | Compares deterministic routing with the LangChain structured router. |
| DeepEval | Uses an LLM judge for answer relevancy, faithfulness, and contextual relevancy. |
| MLflow monitoring | Tracks pipeline runs, live advisor traces, spans, metrics, parameters, and artifacts. |

The dashboard reports a prepared manual RAG Precision@5 average of `93%` for
the included evaluation set.

Important academic limitation: topic labels, personas, and LLM-judged scores are
interpretations of model outputs, not ground-truth facts about every Samsung
customer. The project reports these signals as decision support evidence, not as
absolute market truth.

## Tech Stack

| Layer | Tools |
| --- | --- |
| Frontend | Next.js, React, TypeScript, Tailwind CSS, Recharts, Framer Motion, Lucide icons |
| NLP | pandas, scikit-learn, spaCy, VADER, pyspellchecker, langdetect |
| RAG | ChromaDB, sentence-transformers, hybrid retrieval, BGE reranker evaluation |
| LLM | OpenAI primary generation with optional local Ollama fallback |
| Agents | LangChain structured routing plus deterministic fallback rules |
| Monitoring | MLflow tracing, MLflow artifacts, optional LangSmith traces |
| Evaluation | Custom retrieval evaluation, router evaluation, DeepEval |

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `npm run dev` says Node is unsupported | Install Node.js `20.9.0` or newer, then run `npm install` again. |
| The dashboard opens but `/advisor` fails | Activate `.venv`, install `requirements.txt`, and start `npm run dev` from that same terminal. |
| `ModuleNotFoundError` from a Python bridge | Run `pip install -r requirements.txt` inside the active virtual environment. |
| spaCy model is missing | Run `python -m spacy download en_core_web_sm`. |
| First RAG answer is slow | The app may be building embeddings, creating ChromaDB collections, or downloading local models. The next request should be faster. |
| OpenAI quota or authentication fails | Check `OPENAI_API_KEY` and `OPENAI_MODEL` in `.env`, or enable Ollama fallback. |
| LangSmith errors appear | Set `LANGSMITH_TRACING=false` unless you have a LangSmith key. |
| `/monitoring` has little or no live data | Run a few advisor or document requests, or run `python src\mlflow_monitoring.py`, then refresh the page. |
| Full pipeline fails at sentiment-specific topics | Restore `src/topic_modeling_by_sentiment.py` or run the available individual stages listed above. |

## Repository Notes

- Keep `.env` private.
- Do not commit generated folders such as `.next/`, `node_modules/`,
  `data/vector_db/`, `data/uploaded_documents/`, `mlruns/`, or `mlartifacts/`.
- The included `data/processed/` files make the dashboard reviewable without
  rerunning the full data pipeline.
- Uploaded documents are intentionally limited to Samsung-related content so
  the document RAG feature stays aligned with the project scope.

## Academic Summary

Galaxy Insight demonstrates how social-media feedback can be transformed into a
decision-support system. It moves from raw comments to structured analytics,
then from analytics to grounded RAG answers, and finally from evidence to
actionable Samsung product strategy. The result is not just a notebook or a
model output, but a working dashboard where reviewers can inspect the pipeline,
ask questions, evaluate evidence, and understand how customer feedback informs
product decisions.
