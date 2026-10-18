import numpy as np

import src.policy_search as ps


def test_policies_are_split_into_rules():
    rules = ps.load_policies()
    ids = [r["id"] for r in rules]

    assert len(ids) == 10
    assert set(ids) == {f"FP-{i:02d}" for i in range(1, 11)}
    assert all(r["text"] for r in rules)


def test_search_returns_closest_rules(monkeypatch):
    # Fake embeddings: one dimension per keyword, so we can predict the ranking
    keywords = ["drain", "routine", "mule"]

    def fake_embed(texts):
        vectors = np.array([[t.lower().count(k) + 0.01 for k in keywords] for t in texts])
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    monkeypatch.setattr(ps, "embed", fake_embed)
    monkeypatch.setattr(ps, "_rules", None)

    results = ps.search("account drain by transfer", k=2)

    assert len(results) == 2
    assert results[0]["id"] == "FP-01"
    assert results[0]["score"] >= results[1]["score"]
