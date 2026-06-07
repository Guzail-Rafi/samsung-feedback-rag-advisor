import hashlib
import math
from pathlib import Path

import chromadb
import numpy as np
from chromadb.errors import NotFoundError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_DB_PATH = PROJECT_ROOT / "data" / "vector_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 5000

FEEDBACK_COLLECTION = "feedback_comments"
STRATEGY_COLLECTION = "strategy_evidence"


def get_content_hash(texts):
    hasher = hashlib.sha256()

    for text in texts:
        hasher.update(str(text).encode("utf-8", errors="ignore"))
        hasher.update(b"\0")

    return hasher.hexdigest()


def clean_metadata(metadata):
    cleaned = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if hasattr(value, "item"):
            value = value.item()

        if isinstance(value, float) and math.isnan(value):
            continue

        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)

    return cleaned


def dataframe_metadata(df, columns):
    records = []

    for row_index, row in df.reset_index(drop=True).iterrows():
        metadata = {"row_index": int(row_index)}
        metadata.update({
            column: row.get(column)
            for column in columns
            if column in df.columns
        })
        records.append(clean_metadata(metadata))

    return records


def load_or_create_collection(collection_name, texts, embeddings, metadatas):
    """
    Creates or refreshes a persistent Chroma collection.

    Chroma stores the document text, metadata, and vectors. The content hash
    prevents stale vectors from being reused after the source dataset changes.
    """

    texts = [str(text) for text in texts]
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if len(texts) != len(embeddings) or len(texts) != len(metadatas):
        raise ValueError("Vector-store texts, embeddings, and metadata must have equal lengths.")

    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    content_hash = get_content_hash(texts)

    try:
        collection = client.get_collection(collection_name)
    except NotFoundError:
        collection = None

    if collection is not None:
        metadata = collection.metadata or {}
        is_current = (
            collection.count() == len(texts)
            and metadata.get("content_hash") == content_hash
            and metadata.get("embedding_model") == EMBEDDING_MODEL_NAME
        )

        if is_current:
            print(f"Loading ChromaDB collection: {collection_name}")
            return collection

        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "content_hash": content_hash,
            "embedding_model": EMBEDDING_MODEL_NAME,
        },
    )

    print(f"Building ChromaDB collection: {collection_name}")

    for start in range(0, len(texts), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(texts))
        collection.upsert(
            ids=[f"{collection_name}-{index}" for index in range(start, end)],
            documents=texts[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end],
        )

    return collection


def query_collection(collection, query_embedding, candidate_count):
    if collection.count() == 0:
        return [], []

    query_vector = np.asarray(query_embedding, dtype=np.float32).reshape(-1).tolist()
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(candidate_count, collection.count()),
        include=["metadatas", "distances"],
    )

    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    row_indices = [int(metadata["row_index"]) for metadata in metadatas]
    similarities = [1.0 - float(distance) for distance in distances]

    return row_indices, similarities
