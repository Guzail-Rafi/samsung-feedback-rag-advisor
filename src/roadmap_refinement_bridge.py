import contextlib
import json
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from text_cleanup import sanitize_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_EVIDENCE_PATH = PROJECT_ROOT / "data" / "processed" / "strategy_evidence.csv"
STRATEGY_EMBEDDINGS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "strategy_evidence_embeddings.npy"
)
VALID_VERDICTS = {"Accepted", "Rejected", "Alternative suggested"}
VALID_ACTIONS = {"add", "move", "remove", "none"}


def clean_text(value, limit=500):
    return sanitize_text(str(value or "")).strip()[:limit]


def normalize_roadmap(value):
    if not isinstance(value, list) or not value:
        raise ValueError("A non-empty roadmap is required.")

    roadmap = []
    seen_ids = set()

    for raw_phase in value:
        if not isinstance(raw_phase, dict):
            raise ValueError("Each roadmap phase must be an object.")

        phase_id = clean_text(raw_phase.get("id"), 80)
        title = clean_text(raw_phase.get("title"), 160)
        raw_items = raw_phase.get("items")

        if not phase_id or phase_id in seen_ids or not title or not isinstance(raw_items, list):
            raise ValueError("Each roadmap phase needs a unique id, title, and item list.")

        items = []
        for raw_item in raw_items:
            item = clean_text(raw_item, 160)
            if item and item.casefold() not in {existing.casefold() for existing in items}:
                items.append(item)

        roadmap.append({"id": phase_id, "title": title, "items": items})
        seen_ids.add(phase_id)

    return roadmap


def compact_strategy_evidence(request_text):
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        with contextlib.redirect_stdout(sys.stderr):
            from sentence_transformers import SentenceTransformer

            df = pd.read_csv(STRATEGY_EVIDENCE_PATH)
            df = df.dropna(subset=["strategy_text"]).reset_index(drop=True)
            embeddings = np.load(STRATEGY_EMBEDDINGS_PATH)

            if len(df) != len(embeddings):
                raise ValueError("The strategy evidence and embedding cache are out of sync.")

            model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                local_files_only=True,
            )
            query_embedding = model.encode([request_text])[0]

        embedding_norms = np.linalg.norm(embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        similarities = np.dot(embeddings, query_embedding) / np.maximum(
            embedding_norms * query_norm,
            1e-12,
        )
        priority_scores = (
            df["priority"].map({"High": 1.0, "Medium": 0.6, "Low": 0.3}).fillna(0.5)
        )
        df = df.copy()
        df["strategy_retrieval_score"] = 0.85 * similarities + 0.15 * priority_scores
        ranked = df.sort_values("strategy_retrieval_score", ascending=False)

        selected_indices = []
        category_counts = {}
        for index, row in ranked.iterrows():
            category = clean_text(row.get("issue_category"), 100) or "General"
            if category_counts.get(category, 0) >= 2:
                continue
            selected_indices.append(index)
            category_counts[category] = category_counts.get(category, 0) + 1
            if len(selected_indices) >= 8:
                break

        results = ranked.loc[selected_indices]

        evidence = []
        for _, row in results.iterrows():
            evidence.append(
                {
                    "issue_category": clean_text(row.get("issue_category"), 100),
                    "customer_signal": clean_text(row.get("customer_signal"), 260),
                    "customer_recommendation": clean_text(
                        row.get("customer_recommendation"), 260
                    ),
                    "profit_recommendation": clean_text(
                        row.get("profit_recommendation"), 260
                    ),
                    "business_impact": clean_text(row.get("business_impact"), 180),
                    "priority": clean_text(row.get("priority"), 40),
                    "retrieval_score": float(row.get("strategy_retrieval_score", 0)),
                }
            )

        return evidence, None
    except Exception as error:
        return [], error.__class__.__name__


def extract_json_object(raw_text):
    text = clean_text(raw_text, 12000)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.I)
    if fenced:
        parsed = json.loads(fenced.group(1))
        if isinstance(parsed, dict):
            return parsed

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("The refinement model did not return a valid JSON object.")


def generate_decision(request_text, roadmap, evidence, evidence_error=None):
    with contextlib.redirect_stdout(sys.stderr):
        from openai_client import (
            generate_chat_response,
            get_last_llm_metadata,
            get_openai_client,
        )

        client = get_openai_client()

    phase_ids = [phase["id"] for phase in roadmap]
    evidence_note = (
        f"Strategy evidence retrieval was unavailable ({evidence_error}). "
        "Use the roadmap's phase meanings and mark confidence honestly."
        if evidence_error
        else "Strategy evidence was retrieved successfully."
    )

    system_prompt = """
You are the decision engine for a product-roadmap refinement interface.
Interpret the user's meaning semantically. Do not rely on literal keyword
matching, and do not reject or ignore a request merely because its wording is
new.

Choose the phase from the phase title, purpose, and current items:
- Early phases normally handle trust, reliability, foundational hardware, and
  urgent customer pain.
- Middle phases normally handle practical experiences, software, AI, and
  workflow improvements after the foundations are sound.
- Later phases normally handle premium value, commercial proof, launch
  messaging, trade-ins, bundles, and lower-priority differentiation.

Use retrieved evidence as a constraint, not as an exact phrase dictionary.
Accept a sensible request when its concept is supported. If the requested
timing is poor but the idea is useful, choose "Alternative suggested" and put
it in the better phase. Reject only when it directly conflicts with the
strategy or removes a strategically essential capability without a credible
replacement.

Action rules:
- If the user proposes a capability that is not already listed, use "add".
- Use "move" only when source_item exactly names an existing roadmap item.
- Use "remove" only when source_item exactly names an existing roadmap item.
- Use "none" only when rejecting the request or when no useful roadmap change
  can be inferred.

Return JSON only. Do not use Markdown. Follow this schema:
{
  "verdict": "Accepted" | "Rejected" | "Alternative suggested",
  "action": "add" | "move" | "remove" | "none",
  "target_phase_id": string | null,
  "source_phase_id": string | null,
  "item": string | null,
  "source_item": string | null,
  "rationale": string
}

For add, write a concise 2-8 word roadmap item. For move or remove, source_item
must reproduce the closest existing roadmap item. A rejected decision must use
action "none". Keep the rationale under 45 words and explain the strategic
reason, not the mechanics.

Examples of action semantics:
- A request for better cooling when cooling is not listed is an "add".
- A request to bring an existing launch item forward is a "move".
- A request to delete an existing feature is a "remove".
"""

    user_prompt = f"""
Current roadmap:
{json.dumps(roadmap, ensure_ascii=False, indent=2)}

Allowed phase ids:
{json.dumps(phase_ids)}

User negotiation request:
{request_text}

Retrieved strategy evidence:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Evidence status:
{evidence_note}

Decide the roadmap update and return one JSON object.
"""

    with contextlib.redirect_stdout(sys.stderr):
        raw_answer = generate_chat_response(
            client=client,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": sanitize_text(user_prompt.strip())},
            ],
            temperature=0.1,
            max_completion_tokens=500,
        )
        metadata = get_last_llm_metadata()

    decision = validate_decision(extract_json_object(raw_answer), roadmap)
    return make_decision_applicable(decision, roadmap, request_text), metadata


def optional_text(value, limit=160):
    text = clean_text(value, limit)
    return text or None


def validate_decision(value, roadmap):
    phase_ids = {phase["id"] for phase in roadmap}
    verdict = clean_text(value.get("verdict"), 40)
    action = clean_text(value.get("action"), 20).lower()
    target_phase_id = optional_text(value.get("target_phase_id"), 80)
    source_phase_id = optional_text(value.get("source_phase_id"), 80)
    item = optional_text(value.get("item"), 160)
    source_item = optional_text(value.get("source_item"), 160)
    rationale = clean_text(value.get("rationale"), 500)

    if verdict not in VALID_VERDICTS:
        raise ValueError("The refinement model returned an unsupported verdict.")
    if action not in VALID_ACTIONS:
        raise ValueError("The refinement model returned an unsupported action.")
    if target_phase_id is not None and target_phase_id not in phase_ids:
        raise ValueError("The refinement model selected an unknown target phase.")
    if source_phase_id is not None and source_phase_id not in phase_ids:
        source_phase_id = None
    if action in {"add", "move"} and (not target_phase_id or not item):
        raise ValueError("The refinement model omitted the target phase or roadmap item.")
    if action == "remove" and not (source_item or item):
        raise ValueError("The refinement model omitted the item to remove.")
    if verdict == "Rejected":
        action = "none"
        target_phase_id = None
    if not rationale:
        rationale = "The request was assessed against the roadmap and retrieved strategy evidence."

    return {
        "verdict": verdict,
        "action": action,
        "target_phase_id": target_phase_id,
        "source_phase_id": source_phase_id,
        "item": item,
        "source_item": source_item,
        "rationale": rationale,
    }


def normalize_item(value):
    return re.sub(r"[^a-z0-9]+", " ", clean_text(value, 200).lower()).strip()


def find_existing_item(roadmap, requested_item, preferred_phase_id=None):
    target = normalize_item(requested_item)
    if not target:
        return None

    candidates = []
    for phase in roadmap:
        for item in phase["items"]:
            normalized = normalize_item(item)
            score = SequenceMatcher(None, target, normalized).ratio()
            if target == normalized:
                score = 1.0
            elif target in normalized or normalized in target:
                score = max(score, 0.86)
            if phase["id"] == preferred_phase_id:
                score += 0.04
            candidates.append((score, phase["id"], item))

    if not candidates:
        return None

    score, phase_id, item = max(candidates, key=lambda candidate: candidate[0])
    return (phase_id, item) if score >= 0.56 else None


def add_unique(items, item):
    normalized = normalize_item(item)
    if any(normalize_item(existing) == normalized for existing in items):
        return False
    items.append(item)
    return True


def concise_item_label(candidate, request_text):
    candidate_text = clean_text(candidate, 200)
    candidate_words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'&+-]*", candidate_text)

    if 2 <= len(candidate_words) <= 8 and len(candidate_text) <= 80:
        return candidate_text.rstrip(" .!?")

    text = clean_text(request_text, 240)
    text = re.sub(
        r"^(?:please\s+)?(?:can|could|should|would)\s+(?:we|you|samsung)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^(?:please\s+)?(?:add|include|prioriti[sz]e|introduce|build|improve|make|put)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.split(
        r"\b(?:because|so that|so|in order to|which would|as this)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'&+-]*", text)
    while words and words[0].lower() in {"a", "an", "the", "also"}:
        words.pop(0)

    label = " ".join(words[:8]).strip()
    if not label:
        label = "Requested roadmap improvement"
    return label[0].upper() + label[1:]


def make_decision_applicable(decision, roadmap, request_text):
    action = decision["action"]

    if action == "add":
        decision["item"] = concise_item_label(decision["item"], request_text)
        return decision

    if action not in {"move", "remove"}:
        return decision

    requested_item = decision["source_item"] or decision["item"]
    match = find_existing_item(roadmap, requested_item, decision["source_phase_id"])
    if match:
        decision["source_phase_id"], decision["source_item"] = match
        return decision

    if action == "move" and decision["target_phase_id"]:
        decision["action"] = "add"
        decision["source_phase_id"] = None
        decision["source_item"] = None
        decision["item"] = concise_item_label(decision["item"], request_text)
        decision["rationale"] = (
            "This is a new capability rather than an existing item to move. "
            "Its strategic intent fits the selected phase."
        )
        return decision

    decision["action"] = "none"
    decision["verdict"] = "Alternative suggested"
    decision["target_phase_id"] = None
    decision["rationale"] = (
        "The requested removal does not match an existing roadmap item, so the "
        "current roadmap is preserved."
    )
    return decision


def apply_decision(roadmap, decision):
    updated = [
        {"id": phase["id"], "title": phase["title"], "items": list(phase["items"])}
        for phase in roadmap
    ]
    phase_by_id = {phase["id"]: phase for phase in updated}
    action = decision["action"]

    if action == "none":
        return updated, "No roadmap change applied."

    if action == "add":
        target = phase_by_id[decision["target_phase_id"]]
        changed = add_unique(target["items"], decision["item"])
        update = (
            f'{decision["item"]} added to {target["title"]}.'
            if changed
            else f'{decision["item"]} is already covered in {target["title"]}.'
        )
        return updated, update

    requested_item = decision["source_item"] or decision["item"]
    match = find_existing_item(updated, requested_item, decision["source_phase_id"])

    if not match:
        return updated, (
            f'No matching roadmap item was found for "{requested_item}", so no change was applied.'
        )

    source_phase_id, existing_item = match
    source = phase_by_id[source_phase_id]

    if action == "remove":
        source["items"].remove(existing_item)
        return updated, f'{existing_item} removed from {source["title"]}.'

    target = phase_by_id[decision["target_phase_id"]]
    if source_phase_id == target["id"]:
        return updated, f'{existing_item} is already in {target["title"]}.'

    source["items"].remove(existing_item)
    add_unique(target["items"], existing_item)
    return updated, (
        f'{existing_item} moved from {source["title"]} to {target["title"]}.'
    )


def process_request(payload):
    request_text = clean_text(payload.get("request"), 1200)
    if not request_text:
        raise ValueError("A refinement request is required.")

    roadmap = normalize_roadmap(payload.get("roadmap"))
    evidence, evidence_error = compact_strategy_evidence(request_text)
    decision, metadata = generate_decision(
        request_text,
        roadmap,
        evidence,
        evidence_error=evidence_error,
    )
    updated_roadmap, update = apply_decision(roadmap, decision)

    return {
        "roadmap": updated_roadmap,
        "decision": {
            "request": request_text,
            "verdict": decision["verdict"],
            "rationale": decision["rationale"],
            "update": update,
        },
        "model": metadata.get("model"),
        "llmProvider": metadata.get("provider"),
        "llmFallbackUsed": metadata.get("fallback_used", False),
        "evidenceCount": len(evidence),
        "evidenceAvailable": evidence_error is None,
    }


def main():
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        result = process_request(payload)
        print(json.dumps(result, ensure_ascii=True))
    except Exception as error:
        print(
            json.dumps(
                {
                    "error": clean_text(str(error), 700),
                    "errorType": error.__class__.__name__,
                },
                ensure_ascii=True,
            )
        )
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            from mlflow_tracing import flush_mlflow_traces

            flush_mlflow_traces()
        except Exception:
            pass
