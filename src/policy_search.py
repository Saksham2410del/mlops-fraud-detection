# Finds the bank policy rules that apply to a transaction. This is the
# retrieval part of the RAG setup: the rules found here are added to the
# Llama 3.2 prompt in report.py.
#
# How it works:
#   1. Every rule in policies/*.md is turned into an embedding (a vector that
#      captures its meaning) using nomic-embed-text through Ollama.
#   2. The transaction is described in a few lines of text and embedded too.
#   3. The rules whose vectors are closest to it (cosine similarity) win.
#
# There are only ~10 rules, so a numpy array is all the vector store we need.

import os
from pathlib import Path

import numpy as np
import ollama

from src.config import EMBED_MODEL, POLICIES_DIR, TOP_K_POLICIES

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Built on the first search and reused after that
_rules = None
_vectors = None


def load_policies(policies_dir=POLICIES_DIR):
    """Split every policy file into rules. Each rule starts with '## FP-xx: title'."""
    rules = []
    for path in sorted(Path(policies_dir).glob("*.md")):
        sections = path.read_text(encoding="utf-8").split("\n## ")[1:]
        for section in sections:
            title, _, body = section.partition("\n")
            rules.append({
                "id": title.split(":")[0].strip(),
                "title": title.strip(),
                "text": body.strip(),
            })
    return rules


def embed(texts):
    client = ollama.Client(host=OLLAMA_HOST, timeout=30)
    vectors = np.array(client.embed(model=EMBED_MODEL, input=texts)["embeddings"])
    # Normalise to length 1, then a plain dot product = cosine similarity
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def search(query, k=TOP_K_POLICIES):
    global _rules, _vectors
    if _rules is None:
        _rules = load_policies()
        # nomic-embed-text expects these prefixes on documents and queries
        _vectors = embed([f"search_document: {r['title']}. {r['text']}" for r in _rules])

    query_vector = embed([f"search_query: {query}"])[0]
    scores = _vectors @ query_vector
    best = np.argsort(scores)[::-1][:k]
    return [{**_rules[i], "score": round(float(scores[i]), 3)} for i in best]
