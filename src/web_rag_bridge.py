import contextlib
import json
import os
import re
import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd
from langsmith import traceable

from mlflow_tracing import (
    flush_mlflow_traces,
    get_active_mlflow_trace_id,
    mlflow_span,
    mlflow_status,
)
from text_cleanup import sanitize_text
from tracing_utils import (
    sanitize_trace_inputs,
    sanitize_trace_outputs,
    tracing_enabled,
    tracing_status,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.chdir(PROJECT_ROOT)


FOLLOW_UP_STARTERS = [
    "what about",
    "and",
    "also",
    "same",
    "then",
    "compare",
    "continue",
    "expand",
    "explain more",
    "why",
    "how about",
]


def clean_value(value):
    if pd.isna(value):
        return ""
    return sanitize_text(value)


def clean_web_strategy_answer(value):
    text = sanitize_text(value)
    cleaned_lines = []

    for line in text.splitlines():
        if re.match(r"^\s*(?:sources?|source links?)\s*:", line, flags=re.IGNORECASE):
            continue

        line = re.sub(
            r"\s+(?:sources?|source links?)\s*:\s*.*$",
            "",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(r"https?://[^\s)\]]+", "", line)
        line = re.sub(r"\s+([,.;:])", r"\1", line).rstrip(" ;,")
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def get_model_name():
    with contextlib.redirect_stdout(sys.stderr):
        from openai_client import get_openai_model

        return get_openai_model()


def get_generation_metadata():
    with contextlib.redirect_stdout(sys.stderr):
        from openai_client import get_last_llm_metadata

        return get_last_llm_metadata()


def normalize_messages(messages):
    normalized = []

    for message in messages or []:
        role = clean_value(message.get("role")).strip()
        content = clean_value(message.get("content")).strip()

        if role in {"assistant", "user"} and content:
            normalized.append({"role": role, "content": content})

    return normalized[-10:]


def conversation_text(messages):
    if not messages:
        return "No prior conversation."

    lines = []
    for message in messages[-8:]:
        role = "User" if message["role"] == "user" else "Assistant"
        content = message["content"].replace("\n", " ").strip()
        lines.append(f"{role}: {content[:700]}")

    return "\n".join(lines)


def is_follow_up(query):
    query_lower = query.lower().strip()
    words = query_lower.split()

    if len(words) <= 5:
        return True

    return any(query_lower.startswith(starter) for starter in FOLLOW_UP_STARTERS)


def build_memory_query(query, messages):
    previous_user_messages = [
        message["content"]
        for message in messages
        if message["role"] == "user" and message["content"].strip().lower() != query.lower()
    ]

    if not previous_user_messages:
        return query

    recent_user_context = " ".join(previous_user_messages[-3:])

    if is_follow_up(query):
        return f"Follow-up question: {query}\nCurrent focus: {query}\nPrevious conversation context: {recent_user_context}"

    return f"{query}\nRecent conversation context: {recent_user_context}"


@mlflow_span("Generate Grounded Feedback Answer", "LLM")
@traceable(
    name="Generate Grounded Feedback Answer",
    run_type="chain",
    tags=["feedback-rag", "generation"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
def generate_feedback_answer(query, retrieval_query, evidence_text, confidence, history_text):
    with contextlib.redirect_stdout(sys.stderr):
        from openai_client import generate_chat_response, get_openai_client

        client = get_openai_client()

    system_prompt = """
You are an academic NLP assistant for a university conversational RAG project.

You must answer using only:
1. The retrieved YouTube comment evidence.
2. The conversation history only for interpreting follow-up questions.

Write in polished, concise UK English with a professional academic and business
tone. Integrate evidence naturally rather than repeating the same attribution
phrases. Avoid over-explaining.

Do not invent facts.
Do not overgeneralize beyond the retrieved evidence.
If the evidence is mixed, clearly say it is mixed.
If the user's question is a follow-up, connect it to the previous turn.
Keep the answer concise, balanced, and professional.
Use plain ASCII punctuation and stay under 350 words.
"""

    user_prompt = f"""
Conversation history:
{history_text}

Original user question:
{query}

Contextual retrieval query used for RAG:
{retrieval_query}

Retrieved YouTube Comment Evidence:
{evidence_text}

RAG Confidence:
{confidence}

Write:
1. A direct answer based only on the retrieved evidence.
2. Key evidence-based points.
3. A short confidence explanation.
4. If this was a follow-up, mention how it relates to the previous topic.
"""

    return sanitize_text(generate_chat_response(
        client=client,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": sanitize_text(user_prompt.strip())},
        ],
        temperature=0.2,
        max_completion_tokens=600,
    ))


@mlflow_span("Generate Grounded Strategy Answer", "LLM")
@traceable(
    name="Generate Grounded Strategy Answer",
    run_type="chain",
    tags=["strategy-rag", "generation"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
def generate_strategy_answer_with_memory(
    query,
    retrieval_query,
    goal,
    evidence_text,
    history_text,
    product_lifecycle,
):
    with contextlib.redirect_stdout(sys.stderr):
        from openai_client import generate_chat_response, get_openai_client

        client = get_openai_client()

    system_prompt = """
You are a senior product and commercial strategist writing for a university and
business audience. Respond as the decision owner, not as a feedback analyst or
research summariser.

Turn the supplied evidence into a focused product strategy. Use the evidence
silently to constrain and prioritise the plan, but do not narrate the retrieval
process and do not repeatedly say "users said", "comments show", or "the
evidence suggests". The interface displays supporting evidence separately.

Write in polished, concise UK English. Use a professional GPT-style business and
academic tone. Prefer precise phrases such as "battery-led, creator-focused
premium flagship" over repetitive constructions such as "battery-first,
camera-first, creator-first". Avoid inflated consultancy language.

For every strategy answer, organise the reasoning around five profit levers:
1. Revenue growth
2. Margin protection
3. Customer retention
4. Ecosystem and services monetisation
5. Risk

For a future Galaxy Ultra profit question, the strategy should normally combine
premium positioning; visible improvements in battery, camera, practical AI, and
S Pen utility; targeted trade-ins, storage upgrades, financing, and bundles
rather than broad price cuts; disciplined campaign and feature costs; and
ecosystem monetisation across Galaxy devices and services, when supported by
the supplied evidence.

Make clear choices. State what Samsung should do, what it should deprioritise,
how the plan should be executed, how success should be measured, and which
trade-offs must be managed.

Use conversation history only to understand follow-up questions or negotiation
context. Ground every recommendation in the supplied evidence. Do not invent
product facts, market facts, numerical targets, budgets, dates, or Samsung
commitments. When exact targets are unavailable, name the KPI to measure rather
than making up a number.

Integrate evidence naturally into the reasoning. Do not create an evidence dump
or awkward source list in the main answer. Add ecosystem and services strategy
where relevant, including devices, subscriptions, software, support, payments,
cloud, or connected services.

Respect the supplied product lifecycle classification:
- For future_product, create a hypothetical future-device strategy. Do not
  present current-model features, prices, offers, or specifications as facts
  about the future device. Use current products only as benchmarks.
- For current_product, diagnose the existing product and recommend concrete
  enhancements to its product experience, positioning, pricing, promotion, or
  lifecycle strategy.
- For previous_product, focus on portfolio role, remaining demand, pricing,
  upgrade paths, and lessons for current/future models.
End with a decisive recommendation. Use plain ASCII punctuation and stay under
350 words.
"""

    user_prompt = f"""
Conversation history:
{history_text}

Original user strategy question:
{query}

Contextual retrieval query used for Strategy RAG:
{retrieval_query}

Detected strategy goal:
{goal}

Product lifecycle classification:
{json.dumps(product_lifecycle, ensure_ascii=False, indent=2)}

Retrieved Strategy Evidence:
{evidence_text}

Write exactly these Markdown headings:
**Strategic Recommendation**
**Revenue Growth**
**Margin Protection**
**Customer Retention and Ecosystem**
**Key Risk**
**Final Recommendation**

Open with a direct recommendation in no more than two sentences. Keep each
section compact and action-oriented. Include practical product, marketing, and
commercial actions rather than generic priorities. Where relevant, cover AI,
camera, battery, S Pen, storage, bundles, services, trade-ins, financing, and
channel execution. If this is a follow-up, incorporate the changed requirement
directly rather than recapping the conversation.

Do not include a section named customer feedback, retrieved evidence, evidence
reasoning, research findings, or what changed from prior context. Keep any
confidence statement to one honest sentence within Key Risk.
"""

    return sanitize_text(generate_chat_response(
        client=client,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": sanitize_text(user_prompt.strip())},
        ],
        temperature=0.2,
        max_completion_tokens=900,
    ))


@mlflow_span("Strategy Synthesizer", "LLM")
@traceable(
    name="Strategy Synthesizer",
    run_type="chain",
    tags=["web-augmented-strategy-rag", "synthesis"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
def synthesize_web_augmented_strategy(
    query,
    internal_answer,
    internal_evidence,
    external_evidence,
    history_text,
    pricing_scenario=None,
    product_lifecycle=None,
    research_focus=None,
):
    with contextlib.redirect_stdout(sys.stderr):
        from openai_client import generate_chat_response, get_openai_client

        client = get_openai_client()

    system_prompt = """
You are a senior product, marketing, and commercial strategist writing for a
university and business audience. Respond as the decision owner, not as a
market-research summariser.

Use only the supplied internal YouTube strategy evidence and external web
evidence. Use the evidence to make a clear strategic decision rather than
producing an evidence-by-evidence summary.

The internal YouTube evidence may describe general Galaxy customer priorities
rather than the named model. Attribute a signal specifically to the target
model only when the supplied comment explicitly names that model. Otherwise,
label it as a cross-model customer signal.

Do not invent current prices, offers, dates, market shares, competitor claims,
numerical targets, budgets, or Samsung commitments. Treat external snippets as
limited evidence and cite them only with compact [Web N] markers. If external
evidence is unavailable or weak, say so and make the recommendation provisional.
Every external factual claim must cite its matching [Web N] evidence ID.
Never state a current price, offer, date, or product availability unless it
appears explicitly in that evidence item's snippet. Do not infer willingness to
pay, market demand, market share, or sales success from promotional offer pages.

Write in polished, concise UK English with a professional GPT-style business and
academic tone. Prefer precise phrases such as "battery-led, creator-focused
premium flagship" over repetitive constructions such as "battery-first,
camera-first, creator-first". Avoid inflated consultancy language.

State what Samsung should do, why it creates customer and commercial value,
what should be deprioritised, how to execute it, and how to measure success.
When no numerical target is supplied, name the relevant KPI instead.
Compare credible strategic options before choosing one. Explain why the selected
option is stronger and why at least one alternative was rejected. Do not merely
list evidence; perform a decision analysis.

For every strategy answer, organise the recommendation around five profit
levers: revenue growth, margin protection, customer retention, ecosystem and
services monetisation, and risk.

For a future Galaxy Ultra profit question, the strategy should normally combine
premium positioning; visible improvements in battery, camera, practical AI, and
S Pen utility; targeted trade-ins, storage upgrades, financing, and bundles
rather than broad price cuts; disciplined campaign and feature costs; and
ecosystem monetisation across Galaxy devices and services, when supported by
the supplied evidence.

When the goal is profit, build a usable marketing plan: target audiences,
campaign promise, demonstrations, channels, launch offers, ecosystem bundles,
revenue levers, cost controls, and KPIs. Do not reduce the answer to product
features.

Write for a general reader, not a strategy consultant. Use short sentences and
common words. The reader should understand the recommendation on the first
read. Avoid jargon such as "price architecture", "list-price discipline",
"effective value", "attach rate", "portfolio strategy", "high-salience",
"conversion funnel", and "margin-accretive". If a business term is necessary,
explain it immediately in ordinary language.

For pricing questions, distinguish three things explicitly:
1. Verified current facts, each cited with [Web N].
2. Strategic recommendations, which are decisions rather than facts.
3. Illustrative profit scenarios, which must show their assumptions and must
   never be described as forecasts or guaranteed profit increases.

Do not claim that a holiday, event, or discount will increase profit by a
specific percentage unless that percentage comes from the supplied scenario
calculation. Prefer targeted trade-ins, bundles, storage upgrades, financing,
and channel incentives over permanent list-price cuts when they protect margin.

Respect the supplied product lifecycle classification. If the target is a
future product, current product pages and offers are benchmarks only; never
describe them as the future product's confirmed price, offer, specification, or
launch plan. If the target is the current product, analyze its verified current
position and recommend how Samsung should enhance it now.
Integrate evidence naturally into the reasoning rather than listing sources
awkwardly. When competitor playbook evidence is supplied, identify the tactic,
the reported business result, and how Samsung could adapt it. Do not claim the
tactic caused the result unless the source explicitly proves causation. Prefer
wording such as "the tactic accompanied growth" or "this provides a tested
pattern."

Add ecosystem and services strategy where relevant, particularly for platform
companies such as Samsung, Apple, Google, and Microsoft. Keep the confidence
statement short and honest. End with a decisive recommendation.
Use plain ASCII punctuation. Stay under 420 words.
Use no more than three bullets in each section.
"""

    user_prompt = f"""
Conversation history:
{history_text}

Strategy question:
{query}

Product lifecycle classification:
{json.dumps(product_lifecycle, ensure_ascii=False, indent=2)}

Automatically inferred research objectives:
{json.dumps(research_focus or [], ensure_ascii=False, indent=2)}

Customer Strategist Agent output based on internal YouTube evidence:
{internal_answer}

Internal YouTube strategy evidence:
{json.dumps(internal_evidence, ensure_ascii=False, indent=2)}

External web evidence:
{json.dumps(external_evidence, ensure_ascii=False, indent=2)}

Illustrative pricing scenario model:
{json.dumps(pricing_scenario, ensure_ascii=False, indent=2)}

If an illustrative pricing scenario is supplied AND the research objectives
contain uae_pricing_offers or uae_retail_events, write exactly:
**Strategic Recommendation**
**Revenue Growth**
**Margin Protection**
**Customer Retention and Ecosystem**
**Risk and Confidence**
**Final Recommendation**

If an illustrative pricing scenario is supplied WITHOUT UAE research objectives,
write exactly:
**Strategic Recommendation**
**Revenue Growth**
**Margin Protection**
**Customer Retention and Ecosystem**
**Risk and Confidence**
**Final Recommendation**

Under Strategic Recommendation, give the complete recommendation in no more
than two polished sentences. A reader should understand the decision without
reading further.

Otherwise, write exactly:
**Strategic Recommendation**
**Revenue Growth**
**Margin Protection**
**Customer Retention and Ecosystem**
**Risk and Confidence**
**Final Recommendation**

Use compact [Web N] citations only where an external fact materially supports
the reasoning. Never print raw URLs, a source register, a bibliography, or a
list of source titles in the main answer. The interface displays clickable
sources in the External web evidence panel. Mention evidence limitations
briefly in Risk and Confidence.
For each event in the promotion calendar, include it only if supported by a
[Web N] source; otherwise label it as a proposed test window rather than a
verified event. Recommend an explicit effective-discount test range for each
window using the supplied scenario thresholds. Label those ranges as proposed
tests, not observed best practices. Prefer 0-5% equivalent value through
bundles, trade-ins, financing, or storage upgrades for lighter windows. Use
5-10% only for major retail windows and only when forecast incremental volume
clears the modeled threshold. Do not recommend more than 10% without supplied
commercial evidence. In Illustrative Profit Scenarios, label all percentages
as assumptions, break-even calculations, or targets - never observed outcomes.
Explain the profit calculation in one simple example only. For example: "If
Samsung gives a 5% discount, it must sell about 17% more phones just to make the
same total gross profit under the assumed margin." Do not show more than two
scenario percentages in the main answer; the full table is already visible in
the research panel.
Briefly compare the selected strategy with the obvious alternative and explain
why the alternative is weaker. Do not create a long options section.
Name the actual product areas involved. Do not say "hero upgrades", "visible
value", or "meaningful improvements" without identifying concrete features such
as camera, battery, charging, S Pen, storage, durability, or practical AI.
Keep citations to the minimum needed. Put only uncertainty, not source links,
in the Risk and Confidence section.

Under Revenue Growth, cover the target audience, campaign promise, concrete
product demonstrations, channel plan, premium phone revenue, higher-storage
mix, and at least two competitor tactics where evidence is available.

Under Margin Protection, cover cost control, discount limits, trade-ins,
financing, bundles, partner co-funding, and the rejection of broad price cuts.

Under Customer Retention and Ecosystem, cover upgrade paths, loyalty, Galaxy
Watch, Buds, Tab, SmartThings, software, services, or subscriptions where
relevant.

Under Risk and Confidence, state the principal strategic risk and give one
brief, honest confidence sentence. Integrate citations into the reasoning and
do not append a long source register.

End with one decisive, submission-ready recommendation.
Do not add an internal-evidence summary or a separate source register.
"""

    return clean_web_strategy_answer(generate_chat_response(
        client=client,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": sanitize_text(user_prompt.strip())},
        ],
        temperature=0.15,
        max_completion_tokens=1000,
    ))


def build_pricing_scenario(query):
    query_lower = query.lower()
    pricing_terms = [
        "price",
        "pricing",
        "discount",
        "offer",
        "promotion",
        "profit",
        "margin",
        "trade-in",
        "trade in",
    ]

    if not any(term in query_lower for term in pricing_terms):
        return None

    assumed_margin = 0.35
    target_profit_growth = 0.05
    scenarios = []

    for discount in [0.05, 0.10, 0.15]:
        discounted_margin = assumed_margin - discount
        break_even_volume_uplift = assumed_margin / discounted_margin - 1
        target_volume_uplift = (
            (1 + target_profit_growth) * assumed_margin / discounted_margin - 1
        )
        scenarios.append(
            {
                "effective_discount": f"{discount:.0%}",
                "assumed_baseline_gross_margin": f"{assumed_margin:.0%}",
                "unit_volume_uplift_to_preserve_gross_profit": (
                    f"{break_even_volume_uplift:.1%}"
                ),
                "unit_volume_uplift_for_5_percent_gross_profit_growth": (
                    f"{target_volume_uplift:.1%}"
                ),
            }
        )

    return {
        "status": "Illustrative decision model - not a forecast",
        "baseline": (
            "Price index 100, unit volume index 100, and assumed gross margin 35%. "
            "Replace these assumptions with Samsung finance data before execution."
        ),
        "formula": (
            "Required volume ratio = target gross profit ratio x baseline margin "
            "/ (baseline margin - effective discount)."
        ),
        "scenarios": scenarios,
        "interpretation": (
            "A promotion should proceed only when expected incremental unit volume "
            "exceeds the calculated threshold after accounting for channel and "
            "promotion costs."
        ),
    }


def feedback_evidence_rows(results):
    rows = []

    for _, row in results.head(5).iterrows():
        rows.append(
            {
                "comment": clean_value(row.get("clean_comment")),
                "sentiment": clean_value(row.get("sentiment_label")),
                "issue_category": clean_value(row.get("issue_category")),
                "topic": clean_value(row.get("topic_name")),
                "video": clean_value(row.get("video_title")),
                "weighted_score": float(row.get("weighted_retrieval_score", 0)),
                "similarity_score": float(row.get("similarity_score", 0)),
            }
        )

    return rows


def strategy_evidence_rows(results):
    rows = []
    selected_indices = []
    seen_categories = set()

    for index, row in results.iterrows():
        category = clean_value(row.get("issue_category")) or "General"
        if category not in seen_categories:
            selected_indices.append(index)
            seen_categories.add(category)
        if len(selected_indices) >= 5:
            break

    if len(selected_indices) < 5:
        for index in results.index:
            if index not in selected_indices:
                selected_indices.append(index)
            if len(selected_indices) >= 5:
                break

    for _, row in results.loc[selected_indices].iterrows():
        rows.append(
            {
                "comment": clean_value(row.get("clean_comment")),
                "sentiment": clean_value(row.get("sentiment_label")),
                "issue_category": clean_value(row.get("issue_category")),
                "customer_signal": clean_value(row.get("customer_signal")),
                "customer_recommendation": clean_value(row.get("customer_recommendation")),
                "profit_recommendation": clean_value(row.get("profit_recommendation")),
                "business_impact": clean_value(row.get("business_impact")),
                "priority": clean_value(row.get("priority")),
                "strategy_score": float(row.get("strategy_retrieval_score", 0)),
                "goal_relevance_score": float(row.get("goal_relevance_score", 0)),
            }
        )

    return rows


def build_customer_strategy_brief(goal, evidence):
    lines = [f"Strategy goal: {goal}", "Top internal customer evidence:"]

    for item in evidence[:3]:
        category = item.get("issue_category") or "General feedback"
        signal = item.get("customer_signal") or item.get("comment") or "No signal available"
        recommendation = (
            item.get("customer_recommendation")
            or item.get("profit_recommendation")
            or "No recommendation available"
        )
        lines.append(f"- {category}: {signal[:220]} Recommendation: {recommendation[:220]}")

    return sanitize_text("\n".join(lines))


@mlflow_span("Run Analytical Agent", "TOOL")
@traceable(
    name="Run Analytical Agent",
    run_type="tool",
    tags=["analytical-agent"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
def run_analytical_tool(agent_name, query, retrieval_query, history_text):
    with contextlib.redirect_stdout(sys.stderr):
        from agent_router import (
            issue_agent,
            keyword_agent,
            sentiment_agent,
            summarization_agent,
            topic_agent,
        )

        tool_handlers = {
            "summarization_agent": {
                "handler": summarization_agent,
                "sources": ["llm_summaries.csv"],
            },
            "sentiment_agent": {
                "handler": sentiment_agent,
                "sources": ["comments_with_ner.csv"],
            },
            "issue_agent": {
                "handler": issue_agent,
                "sources": ["comments_with_ner.csv"],
            },
            "topic_agent": {
                "handler": topic_agent,
                "sources": ["comments_with_ner.csv", "topic_keywords.csv"],
            },
            "keyword_agent": {
                "handler": keyword_agent,
                "sources": ["top_keywords_overall.csv"],
            },
        }

        tool = tool_handlers.get(agent_name)

        if tool is None:
            raise ValueError(f"Unsupported analytical agent: {agent_name}")

        answer = tool["handler"](retrieval_query)

    return {
        "mode": "analytical_tool",
        "selectedAgent": agent_name,
        "answer": answer,
        "contextualQuery": retrieval_query,
        "memoryUsed": history_text != "No prior conversation.",
        "sources": tool["sources"],
    }


@mlflow_span("Feedback RAG Agent", "CHAIN")
@traceable(
    name="Feedback RAG Agent",
    run_type="chain",
    tags=["feedback-rag", "agent"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
def run_feedback_rag(query, retrieval_query, history_text):
    with contextlib.redirect_stdout(sys.stderr):
        from rag_answer_generator import (
            INPUT_PATH,
            build_comment_text,
            calculate_rag_confidence,
            format_evidence,
            load_or_create_embeddings,
            load_or_create_feedback_vector_store,
            retrieve_comments,
        )

        df = pd.read_csv(INPUT_PATH)
        df = df[df["language"] == "en"].copy()
        df = df.dropna(subset=["clean_comment"])
        df = df[df["word_count"] >= 3].copy()
        df = df.reset_index(drop=True)
        df["rag_text"] = df.apply(build_comment_text, axis=1)

        embeddings = load_or_create_embeddings(df["rag_text"].tolist())
        vector_collection = load_or_create_feedback_vector_store(df, embeddings)
        results = retrieve_comments(
            query=retrieval_query,
            df=df,
            comment_embeddings=embeddings,
            top_k=8,
            vector_collection=vector_collection,
            candidate_count=700,
        )
        confidence = calculate_rag_confidence(results)
        evidence_text = format_evidence(results)

    answer = generate_feedback_answer(query, retrieval_query, evidence_text, confidence, history_text)
    llm_metadata = get_generation_metadata()

    return {
        "mode": "feedback_rag",
        "selectedAgent": "feedback_rag_agent",
        "answer": answer,
        "llm": llm_metadata,
        "confidence": confidence,
        "contextualQuery": retrieval_query,
        "memoryUsed": history_text != "No prior conversation.",
        "sources": ["comments_with_ner.csv", "ChromaDB: feedback_comments"],
        "evidence": feedback_evidence_rows(results),
        "retrieval": {
            "vector_store": "ChromaDB",
            "collection": "feedback_comments",
            "embedding_model": "all-MiniLM-L6-v2",
            "top_k": 8,
            "score": round(float(results["weighted_retrieval_score"].mean()), 3),
        },
    }


@mlflow_span("Strategy RAG Agent", "CHAIN")
@traceable(
    name="Strategy RAG Agent",
    run_type="chain",
    tags=["strategy-rag", "agent"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
def run_strategy_rag_live(
    query,
    retrieval_query,
    history_text,
    product_lifecycle,
    generate_answer=True,
):
    with contextlib.redirect_stdout(sys.stderr):
        from strategy_rag import (
            INPUT_PATH,
            detect_strategy_goal,
            format_strategy_evidence,
            load_or_create_embeddings,
            load_or_create_strategy_vector_store,
            retrieve_strategy_evidence,
        )

        df = pd.read_csv(INPUT_PATH)
        df = df.dropna(subset=["strategy_text"]).copy()
        df = df.reset_index(drop=True)

        embeddings = load_or_create_embeddings(df["strategy_text"].tolist())
        vector_collection = load_or_create_strategy_vector_store(df, embeddings)
        goal = detect_strategy_goal(retrieval_query)
        results = retrieve_strategy_evidence(
            query=retrieval_query,
            goal=goal,
            df=df,
            embeddings=embeddings,
            top_k=12,
            vector_collection=vector_collection,
            candidate_count=900,
        )
        if (
            product_lifecycle.get("lifecycle") == "future_product"
            and goal == "profit"
        ):
            feature_query = (
                f"What concrete product features should Samsung prioritize in "
                f"{product_lifecycle.get('requested_model') or 'the next Galaxy Ultra'} "
                "to create clear customer value, differentiation, and upgrade demand?"
            )
            feature_results = retrieve_strategy_evidence(
                query=feature_query,
                goal="balanced",
                df=df,
                embeddings=embeddings,
                top_k=12,
                vector_collection=vector_collection,
                candidate_count=900,
            )
            candidate_pool = (
                pd.concat([results, feature_results], ignore_index=True)
                .drop_duplicates(subset=["clean_comment"], keep="first")
                .sort_values("strategy_retrieval_score", ascending=False)
            )
            coverage_rows = []
            for category in [
                "Camera",
                "Battery / Charging",
                "S-Pen / Features",
                "AI / Gemini",
                "Price / Value",
            ]:
                candidates = candidate_pool[
                    candidate_pool["issue_category"] == category
                ]
                if candidates.empty:
                    continue
                coverage_rows.append(candidates.iloc[[0]])

            results = (
                pd.concat([*coverage_rows, candidate_pool], ignore_index=True)
                .drop_duplicates(subset=["clean_comment"], keep="first")
                .head(15)
            )
        evidence_text = format_strategy_evidence(results)

    evidence = strategy_evidence_rows(results)

    if generate_answer:
        answer = generate_strategy_answer_with_memory(
            query,
            retrieval_query,
            goal,
            evidence_text,
            history_text,
            product_lifecycle,
        )
        llm_metadata = get_generation_metadata()
    else:
        answer = build_customer_strategy_brief(goal, evidence)
        llm_metadata = {}

    return {
        "mode": "strategy_rag",
        "selectedAgent": "strategy_rag_agent",
        "answer": answer,
        "llm": llm_metadata,
        "strategyGoal": goal,
        "productLifecycle": product_lifecycle,
        "contextualQuery": retrieval_query,
        "memoryUsed": history_text != "No prior conversation.",
        "sources": ["strategy_evidence.csv", "ChromaDB: strategy_evidence"],
        "evidence": evidence,
        "retrieval": {
            "vector_store": "ChromaDB",
            "collection": "strategy_evidence",
            "embedding_model": "all-MiniLM-L6-v2",
            "top_k": 12,
            "score": round(float(results["strategy_retrieval_score"].mean()), 3),
            "goal_relevance": round(float(results["goal_relevance_score"].mean()), 3),
        },
    }


@mlflow_span("Customer Strategist Agent", "CHAIN")
@traceable(
    name="Customer Strategist Agent",
    run_type="chain",
    tags=["web-augmented-strategy-rag", "customer-strategist", "internal-evidence"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
def run_customer_strategist_agent(query, retrieval_query, history_text, product_lifecycle):
    return run_strategy_rag_live(
        query,
        retrieval_query,
        history_text,
        product_lifecycle,
        generate_answer=False,
    )


@mlflow_span("Web-Augmented Strategy RAG", "CHAIN")
@traceable(
    name="Web-Augmented Strategy RAG",
    run_type="chain",
    tags=["web-augmented-strategy-rag", "advanced-extension"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
def run_web_augmented_strategy_rag(
    query,
    retrieval_query,
    history_text,
    research_focus,
    product_lifecycle,
):
    with contextlib.redirect_stdout(sys.stderr):
        from web_research import search_market_evidence

    internal_result = run_customer_strategist_agent(
        query,
        retrieval_query,
        history_text,
        product_lifecycle,
    )
    pricing_scenario = build_pricing_scenario(query)
    web_research = search_market_evidence(
        retrieval_query,
        research_focus,
        max_results=8 if pricing_scenario else 5,
    )
    product_lifecycle = web_research.get(
        "product_lifecycle",
        product_lifecycle,
    )
    external_evidence = [
        {"evidence_id": f"Web {index}", **item}
        for index, item in enumerate(web_research["evidence"], start=1)
    ]
    answer = synthesize_web_augmented_strategy(
        query=query,
        internal_answer=internal_result["answer"],
        internal_evidence=internal_result["evidence"][:3],
        external_evidence=external_evidence,
        history_text=history_text,
        pricing_scenario=pricing_scenario,
        product_lifecycle=product_lifecycle,
        research_focus=research_focus,
    )
    llm_metadata = get_generation_metadata()

    if len(external_evidence) >= 3:
        confidence = "High"
    elif external_evidence:
        confidence = "Medium"
    else:
        confidence = "Low"

    if llm_metadata.get("fallback_used") and confidence == "High":
        confidence = "Medium"
    if pricing_scenario and confidence == "High":
        confidence = "Medium"

    return {
        "mode": "web_augmented_strategy_rag",
        "selectedAgent": "web_augmented_strategy_rag",
        "answer": answer,
        "llm": llm_metadata,
        "confidence": confidence,
        "strategyGoal": internal_result.get("strategyGoal"),
        "productLifecycle": product_lifecycle,
        "contextualQuery": retrieval_query,
        "memoryUsed": history_text != "No prior conversation.",
        "sources": internal_result["sources"] + [
            f"DDGS external evidence ({len(external_evidence)} sources)"
        ],
        "evidence": internal_result["evidence"],
        "internalStrategyAnswer": internal_result["answer"],
        "externalEvidence": external_evidence,
        "webResearch": {
            "provider": web_research["provider"],
            "queries": web_research.get("queries", []),
            "result_count": web_research["result_count"],
            "retrieved_at": web_research["retrieved_at"],
            "errors": web_research["errors"],
            "focus": research_focus,
            "pricing_scenario": pricing_scenario,
            "product_lifecycle": product_lifecycle,
            "product_verification": web_research.get("product_verification"),
        },
        "retrieval": internal_result["retrieval"],
        "toolTrace": [
            "web_augmented_strategy_rag",
            "customer_strategist_agent",
            "market_research_agent",
            "strategy_synthesizer",
        ],
    }


@mlflow_span("Samsung Document RAG Agent", "CHAIN")
@traceable(
    name="Samsung Document RAG Agent",
    run_type="chain",
    tags=["document-rag", "agent"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
def run_samsung_document_rag(query, messages):
    with contextlib.redirect_stdout(sys.stderr):
        from document_rag_bridge import answer_document_question

    document_result = answer_document_question(query, messages)
    evidence = document_result["evidence"]

    return {
        "mode": "document_rag",
        "selectedAgent": "samsung_document_rag",
        "answer": document_result["answer"],
        "llm": document_result.get("llm", {}),
        "confidence": document_result["confidence"],
        "contextualQuery": query,
        "memoryUsed": bool(messages),
        "sources": sorted({item["filename"] for item in evidence}),
        "evidence": evidence,
        "retrieval": {
            "vector_store": "ChromaDB",
            "collection": "samsung_documents",
            "embedding_model": "all-MiniLM-L6-v2",
            "top_k": 6,
            "score": round(
                sum(item["similarity"] for item in evidence) / len(evidence),
                3,
            ) if evidence else 0,
        },
        "toolTrace": ["samsung_document_rag", "document_retrieval", "document_answer_generation"],
    }


@mlflow_span("Samsung Advisor Request", "CHAIN")
@traceable(
    name="YouTube Intelligence Advisor Request",
    run_type="chain",
    tags=["advisor", "multi-agent-rag"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
def run_advisor(payload):
    query = clean_value(payload.get("message")).strip()
    messages = normalize_messages(payload.get("messages", []))

    if not query:
        raise ValueError("message is required")

    retrieval_query = build_memory_query(query, messages)
    with contextlib.redirect_stdout(sys.stderr):
        from web_strategy_policy import classify_product_lifecycle, is_feedback_request

    routing_query = (
        retrieval_query
        if is_follow_up(query) and not is_feedback_request(query)
        else query
    )
    history_text = conversation_text(messages)

    with contextlib.redirect_stdout(sys.stderr):
        from agent_router import AVAILABLE_AGENTS, route_intent
        from document_rag_bridge import load_manifest

        document_names = [item.get("filename", "") for item in load_manifest() if item.get("filename")]
        routing = route_intent(routing_query, document_names=document_names)

    selected_agent = routing["selected_agent"]
    routed_query = (
        routing.get("rewritten_query")
        or routing.get("normalized_query")
        or retrieval_query
    )
    product_lifecycle = classify_product_lifecycle(routed_query)

    if selected_agent == "web_augmented_strategy_rag":
        result = run_web_augmented_strategy_rag(
            query,
            routed_query,
            history_text,
            routing.get("external_research_focus", []),
            product_lifecycle,
        )
    elif selected_agent == "strategy_rag_agent":
        result = run_strategy_rag_live(
            query,
            routed_query,
            history_text,
            product_lifecycle,
        )
    elif selected_agent == "feedback_rag_agent":
        result = run_feedback_rag(query, routed_query, history_text)
    elif selected_agent == "samsung_document_rag":
        result = run_samsung_document_rag(query, messages)
    else:
        result = run_analytical_tool(selected_agent, query, routed_query, history_text)

    if selected_agent in {
        "feedback_rag_agent",
        "strategy_rag_agent",
        "web_augmented_strategy_rag",
        "samsung_document_rag",
    }:
        llm_metadata = result.get("llm", {})
        result["model"] = llm_metadata.get("model") or get_model_name()
        result["llmProvider"] = llm_metadata.get("provider")
        result["llmFallbackUsed"] = llm_metadata.get("fallback_used", False)
        result["llmFallbackReason"] = llm_metadata.get("fallback_reason")

    result["routingReason"] = routing["reason"]
    result["matchedTerms"] = routing["matched_terms"]
    result["routingConfidence"] = routing["confidence"]
    result["routingMethod"] = routing["routing_method"]
    result["routerModel"] = routing["router_model"]
    result["routerProvider"] = routing.get("router_provider")
    result["routerFallbackUsed"] = routing.get("router_fallback_used", False)
    result["routerFallbackReason"] = routing.get("router_fallback_reason")
    result["needsExternalResearch"] = routing.get("needs_external_research", False)
    result["externalResearchFocus"] = routing.get("external_research_focus", [])
    result["rewrittenQuery"] = routing.get("rewritten_query", routed_query)
    result["normalizedQuery"] = routed_query
    result["toolTrace"] = [routing["routing_method"]] + result.get(
        "toolTrace",
        [selected_agent],
    )
    result["availableTools"] = AVAILABLE_AGENTS
    result["query"] = query
    result["productLifecycle"] = result.get("productLifecycle", product_lifecycle)
    result["uploadedDocumentCount"] = len(document_names)
    result["langsmithTracing"] = tracing_status()
    result["mlflowTracing"] = mlflow_status()
    result["mlflowTraceId"] = get_active_mlflow_trace_id()

    return result


def main():
    payload = json.loads(sys.stdin.read() or "{}")
    trace_id = uuid4()
    result = run_advisor(
        payload,
        langsmith_extra={
            "run_id": trace_id,
            "metadata": {
                "source": "live_advisor",
                "message_count": len(payload.get("messages", [])),
            },
            "tags": ["live-advisor-request"],
        },
    )

    if tracing_enabled():
        result["langsmithTraceId"] = str(trace_id)

    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            json.dumps(
                {
                    "error": str(error),
                    "type": error.__class__.__name__,
                },
                ensure_ascii=True,
            )
        )
        sys.exit(1)
    finally:
        flush_mlflow_traces()
