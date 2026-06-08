import base64
import contextlib
import csv
import hashlib
import io
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from langsmith import traceable

from mlflow_tracing import flush_mlflow_traces, mlflow_span, mlflow_status
from text_cleanup import sanitize_text
from tracing_utils import (
    sanitize_trace_inputs,
    sanitize_trace_outputs,
    tracing_enabled,
    tracing_status,
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_DB_PATH = PROJECT_ROOT / "data" / "vector_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploaded_documents"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "samsung_documents_manifest.json"
COLLECTION_NAME = "samsung_documents"
MAX_FILE_BYTES = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".html", ".htm", ".pdf", ".docx"}

SAMSUNG_SIGNALS = [
    "samsung",
    "galaxy",
    "one ui",
    "oneui",
    "s pen",
    "s-pen",
    "smartthings",
    "exynos",
    "samsung knox",
    "galaxy ai",
    "galaxy buds",
    "galaxy watch",
    "galaxy fold",
    "galaxy flip",
    "neo qled",
    "samsung display",
    "samsung electronics",
]

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_filename(filename):
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", Path(filename or "document.txt").name)
    return cleaned[:180] or "document.txt"


def load_manifest():
    if not MANIFEST_PATH.exists():
        return []

    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def save_manifest(documents):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(documents, indent=2, ensure_ascii=True), encoding="utf-8")


def get_collection(create=True):
    import chromadb

    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))

    if create:
        return client.get_or_create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL_NAME},
        )

    return client.get_collection(COLLECTION_NAME)


def decode_text(content):
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def extract_pdf(content):
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise ValueError("PDF support requires pypdf. Run: pip install pypdf") from error

    reader = PdfReader(io.BytesIO(content))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def extract_docx(content):
    try:
        from docx import Document
    except ImportError as error:
        raise ValueError("DOCX support requires python-docx. Run: pip install python-docx") from error

    document = Document(io.BytesIO(content))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def extract_document_text(filename, content):
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type {extension or '(none)'}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if extension == ".pdf":
        text = extract_pdf(content)
    elif extension == ".docx":
        text = extract_docx(content)
    elif extension == ".csv":
        decoded = decode_text(content)
        rows = csv.reader(io.StringIO(decoded))
        text = "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)
    elif extension in {".html", ".htm"}:
        decoded = decode_text(content)
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", decoded, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
    else:
        text = decode_text(content)

    text = sanitize_text(text).replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(text) < 120:
        raise ValueError("The document contains too little readable text to index.")

    return text


def samsung_relevance(text):
    lowered = text.lower()
    matched = sorted({signal for signal in SAMSUNG_SIGNALS if signal in lowered})
    model_mentions = re.findall(r"\b(?:s|a|z)\d{2}\s*(?:ultra|plus|\+|fe)?\b", lowered)
    relevant = bool(matched or model_mentions)
    return relevant, matched[:10], len(model_mentions)


def chunk_text(text, chunk_size=1000, overlap=160):
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        remaining = paragraph
        while remaining:
            room = chunk_size - len(current)
            piece = remaining[:room]
            current = f"{current}\n{piece}".strip()
            remaining = remaining[room:]

            if len(current) >= chunk_size:
                chunks.append(current)
                current = current[-overlap:]

    if current.strip():
        chunks.append(current.strip())

    return [chunk for chunk in chunks if len(chunk) >= 80]


@mlflow_span("Samsung Document Ingestion", "CHAIN")
@traceable(
    name="Samsung Document Ingestion",
    run_type="chain",
    tags=["document-rag", "ingestion", "chromadb"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
def ingest_document(filename, content):
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("Document is larger than the 10 MB upload limit.")

    filename = safe_filename(filename)
    text = extract_document_text(filename, content)
    relevant, matched_signals, model_mentions = samsung_relevance(text)

    if not relevant:
        raise ValueError(
            "This document does not appear to be Samsung-related. "
            "Only Samsung, Galaxy, Samsung product, customer-feedback, or strategy documents are accepted."
        )

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No useful document chunks could be created.")

    content_hash = hashlib.sha256(content).hexdigest()
    documents = load_manifest()
    duplicate = next((item for item in documents if item.get("content_hash") == content_hash), None)
    if duplicate:
        return {"status": "already_indexed", "document": duplicate, "documents": documents}

    document_id = uuid4().hex
    uploaded_at = utc_now()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored_path = UPLOAD_DIR / f"{document_id}_{filename}"
    stored_path.write_bytes(content)

    embeddings = get_embedding_model().encode(chunks, show_progress_bar=False).tolist()
    collection = get_collection()
    ids = [f"{document_id}-{index}" for index in range(len(chunks))]
    metadatas = [
        {
            "document_id": document_id,
            "filename": filename,
            "chunk_index": index,
            "uploaded_at": uploaded_at,
            "row_index": index,
        }
        for index in range(len(chunks))
    ]
    collection.upsert(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)

    document = {
        "document_id": document_id,
        "filename": filename,
        "extension": Path(filename).suffix.lower(),
        "size_bytes": len(content),
        "character_count": len(text),
        "chunk_count": len(chunks),
        "uploaded_at": uploaded_at,
        "content_hash": content_hash,
        "matched_samsung_signals": matched_signals,
        "model_mentions": model_mentions,
        "stored_path": str(stored_path.relative_to(PROJECT_ROOT)),
    }
    documents.append(document)
    save_manifest(documents)

    return {"status": "indexed", "document": document, "documents": documents}


@mlflow_span("Samsung Document Retrieval", "RETRIEVER")
@traceable(
    name="Samsung Document Retrieval",
    run_type="retriever",
    tags=["document-rag", "chromadb", "retrieval"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
def retrieve_document_chunks(query, top_k=6):
    documents = load_manifest()
    if not documents:
        return []

    collection = get_collection()
    query_embedding = get_embedding_model().encode([query], show_progress_bar=False)[0].tolist()
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    evidence = []
    for index, (document, metadata, distance) in enumerate(
        zip(result["documents"][0], result["metadatas"][0], result["distances"][0]),
        start=1,
    ):
        similarity = 1.0 - float(distance)
        if similarity < 0.12:
            continue
        evidence.append(
            {
                "evidence_id": f"Doc {index}",
                "filename": metadata.get("filename"),
                "chunk_index": metadata.get("chunk_index"),
                "content": sanitize_text(document),
                "similarity": round(similarity, 3),
            }
        )

    return evidence


@mlflow_span("Samsung Document Answer Generation", "LLM")
@traceable(
    name="Samsung Document Answer Generation",
    run_type="chain",
    tags=["document-rag", "generation"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
def answer_document_question(question, messages):
    from openai_client import generate_chat_response, get_last_llm_metadata, get_openai_client

    evidence = retrieve_document_chunks(question)
    if not evidence:
        return {
            "answer": (
                "I could not find relevant evidence in the uploaded Samsung documents. "
                "Upload a relevant document or ask a question covered by the indexed files."
            ),
            "evidence": [],
            "confidence": "Low",
            "llm": {},
        }

    history = "\n".join(
        f"{message.get('role', 'user').title()}: {sanitize_text(message.get('content'))[:500]}"
        for message in (messages or [])[-6:]
        if message.get("content")
    ) or "No previous conversation."
    evidence_text = "\n\n".join(
        f"[{item['evidence_id']}] Source: {item['filename']}, chunk {item['chunk_index']}\n{item['content']}"
        for item in evidence
    )

    system_prompt = """
You are Samsung Document Assistant, a conversational Samsung-domain RAG assistant.
Answer only from the uploaded Samsung document evidence.
You may explain, summarize, compare, and reason across the supplied evidence.
Never invent product facts or use outside knowledge.
If evidence is incomplete, say what is missing.
Use [Doc N] citations for factual claims.
Stay under 450 words and use clear Markdown headings or bullets when useful.
Use plain ASCII punctuation.
"""
    user_prompt = f"""
Conversation:
{history}

Question:
{sanitize_text(question)}

Uploaded Samsung document evidence:
{evidence_text}
"""
    answer = generate_chat_response(
        get_openai_client(),
        [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.15,
        max_completion_tokens=750,
    )
    average_similarity = sum(item["similarity"] for item in evidence[:3]) / min(len(evidence), 3)
    confidence = "High" if average_similarity >= 0.55 else "Medium" if average_similarity >= 0.3 else "Low"

    return {
        "answer": sanitize_text(answer),
        "evidence": evidence,
        "confidence": confidence,
        "llm": get_last_llm_metadata(),
    }


def delete_document(document_id):
    documents = load_manifest()
    document = next((item for item in documents if item.get("document_id") == document_id), None)
    if not document:
        raise ValueError("Document not found.")

    collection = get_collection()
    collection.delete(where={"document_id": document_id})
    stored_path = PROJECT_ROOT / document.get("stored_path", "")
    if stored_path.is_file() and UPLOAD_DIR in stored_path.parents:
        stored_path.unlink()

    documents = [item for item in documents if item.get("document_id") != document_id]
    save_manifest(documents)
    return {"status": "deleted", "documents": documents}


@mlflow_span("Samsung Document RAG Request", "CHAIN")
@traceable(
    name="Samsung Document RAG Request",
    run_type="chain",
    tags=["samsung-document-rag"],
    process_inputs=sanitize_trace_inputs,
    process_outputs=sanitize_trace_outputs,
)
def handle(payload):
    action = payload.get("action", "list")

    if action == "list":
        return {
            "action": action,
            "documents": load_manifest(),
            "collection": COLLECTION_NAME,
            "tracing": tracing_status(),
            "mlflowTracing": mlflow_status(),
        }

    if action == "ingest":
        encoded = payload.get("content_base64", "")
        if not encoded:
            raise ValueError("Document content is required.")
        result = ingest_document(payload.get("filename", "document.txt"), base64.b64decode(encoded))
        return {"action": action, **result, "collection": COLLECTION_NAME}

    if action == "chat":
        question = sanitize_text(payload.get("message")).strip()
        if not question:
            raise ValueError("Message is required.")
        result = answer_document_question(question, payload.get("messages", []))
        llm = result.pop("llm", {})
        return {
            "action": action,
            "selectedAgent": "samsung_document_rag",
            "collection": COLLECTION_NAME,
            "sources": sorted({item["filename"] for item in result["evidence"]}),
            "model": llm.get("model"),
            "llmProvider": llm.get("provider"),
            "llmFallbackUsed": llm.get("fallback_used", False),
            "llmFallbackReason": llm.get("fallback_reason"),
            **result,
        }

    if action == "delete":
        return {"action": action, **delete_document(payload.get("document_id", ""))}

    raise ValueError(f"Unsupported action: {action}")


def main():
    payload = json.loads(sys.stdin.read() or "{}")
    trace_id = uuid4()
    result = handle(
        payload,
        langsmith_extra={
            "run_id": trace_id,
            "metadata": {"source": "samsung_document_rag", "action": payload.get("action", "list")},
            "tags": ["samsung-document-rag"],
        },
    )
    if tracing_enabled():
        result["langsmithTraceId"] = str(trace_id)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error": str(error), "type": error.__class__.__name__}, ensure_ascii=True))
        sys.exit(1)
    finally:
        flush_mlflow_traces()
