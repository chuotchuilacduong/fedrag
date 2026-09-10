"""Semantic topic-cluster + Dirichlet client partitioning ("topic skew").

Alternative to the default index-based partition in `data_preprocess.py`
(`partition_linearrag_chunks`, plain `chunk.index % num_clients`). Groups the
retrieval corpus into article-level allocation units, embeds each unit with a
frozen all-MiniLM-L6-v2 encoder, clusters into K semantic "topic proxies"
per dataset via K-means, then allocates topics across clients via a
symmetric Dirichlet(alpha) distribution (or a topic-agnostic random
baseline that skips clustering entirely).

K is a proposed experimental parameter, not an established count of natural
topics -- describe the clusters as topic proxies, not ground-truth topics.
Cluster definitions differ between datasets (cluster 5 in one dataset need
not correspond to cluster 5 in another) and are fit ONCE per dataset, then
reused across every alpha/random setting for that dataset -- otherwise
changing alpha would also change what a "topic" means.

Output type (`list[ClientChunks]`) matches `partition_linearrag_chunks`
exactly, so this is a drop-in replacement -- nothing downstream (trigraph
building, Stage B condensation, QA dataset construction, PPR maps) needs to
change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fedcond_grag.dataloader.data_preprocess import ClientChunks, LinearRAGChunk

DEFAULT_NUM_TOPIC_CLUSTERS = 20
DEFAULT_WINDOW_TOKENS = 254  # + 2 special tokens = 256, all-MiniLM-L6-v2's limit


@dataclass
class ArticleGroup:
    """One allocation unit: all distinct passages sharing a title."""

    title: str
    chunks: list[LinearRAGChunk] = field(default_factory=list)
    passage_texts: list[str] = field(default_factory=list)  # "{title}: {text}" per unique passage


def _chunk_title_and_text(chunk: LinearRAGChunk) -> tuple[str, str]:
    """chunk.body is "{title}: {text}" (see scripts/setup_datasets.py's
    _build_chunks: f"{i}:{title}: {text}"). Recover (title, text)."""
    title, sep, text = chunk.body.partition(": ")
    if not sep:
        return chunk.body.strip(), ""
    return title.strip(), text.strip()


def dedupe_and_group_chunks(chunks: list[LinearRAGChunk]) -> list[ArticleGroup]:
    """Step 1 of the protocol: remove duplicate passages, then group
    passages sharing a title into one allocation unit. Distinct paragraphs
    from the same title (e.g. MuSiQue) are kept, not discarded, and all land
    in the same group -- so the whole article is assigned to one client."""
    seen_passages: set[tuple[str, str]] = set()
    by_title: dict[str, ArticleGroup] = {}
    order: list[str] = []
    for chunk in chunks:
        title, text = _chunk_title_and_text(chunk)
        key = (title, text)
        if key in seen_passages:
            continue
        seen_passages.add(key)
        if title not in by_title:
            by_title[title] = ArticleGroup(title=title)
            order.append(title)
        group = by_title[title]
        group.chunks.append(chunk)
        group.passage_texts.append(chunk.body)
    return [by_title[title] for title in order]


def embed_article_groups(
    groups: list[ArticleGroup],
    model,
    *,
    window_tokens: int = DEFAULT_WINDOW_TOKENS,
    batch_size: int = 512,
) -> np.ndarray:
    """Step 2: one L2-normalized embedding per article group.

    Per passage: split into non-overlapping `window_tokens`-token windows,
    embed each window, average -> passage embedding. Then average passage
    embeddings within the group, and L2-normalize the result. All windows
    across all passages are flattened into one batched encode() call for
    speed, then reassembled -- equivalent to encoding one at a time.
    """
    tokenizer = model.tokenizer

    all_windows: list[str] = []
    passage_window_counts: list[int] = []
    group_passage_counts: list[int] = []
    for group in groups:
        group_passage_counts.append(len(group.passage_texts))
        for text in group.passage_texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                windows = [""]
            else:
                windows = [
                    tokenizer.decode(ids[i : i + window_tokens])
                    for i in range(0, len(ids), window_tokens)
                ]
            all_windows.extend(windows)
            passage_window_counts.append(len(windows))

    dim = (
        model.get_embedding_dimension()
        if hasattr(model, "get_embedding_dimension")
        else model.get_sentence_embedding_dimension()
    )
    if not all_windows:
        return np.zeros((0, dim), dtype=np.float32)

    window_embeddings: list[np.ndarray] = []
    for start in range(0, len(all_windows), batch_size):
        batch = all_windows[start : start + batch_size]
        window_embeddings.append(
            model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        )
    window_embeddings_arr = np.concatenate(window_embeddings, axis=0)

    passage_embeddings: list[np.ndarray] = []
    offset = 0
    for count in passage_window_counts:
        passage_embeddings.append(window_embeddings_arr[offset : offset + count].mean(axis=0))
        offset += count

    group_embeddings = np.zeros((len(groups), dim), dtype=np.float32)
    p_offset = 0
    for i, count in enumerate(group_passage_counts):
        stacked = np.stack(passage_embeddings[p_offset : p_offset + count], axis=0)
        group_embeddings[i] = stacked.mean(axis=0)
        p_offset += count

    norms = np.linalg.norm(group_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (group_embeddings / norms).astype(np.float32)


def fit_topic_clusters(
    embeddings: np.ndarray, n_clusters: int = DEFAULT_NUM_TOPIC_CLUSTERS, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Step 3: plain K-means, exact settings per the protocol."""
    from sklearn.cluster import KMeans

    clusterer = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=seed,
        algorithm="lloyd",
    )
    topic_ids = clusterer.fit_predict(embeddings)
    return topic_ids, clusterer.cluster_centers_


def _load_or_fit_clusters(
    groups: list[ArticleGroup], n_clusters: int, seed: int, cache_path: str | Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    titles = [g.title for g in groups]
    if cache_path is not None:
        cache_path = Path(cache_path)
        emb_path = cache_path.with_suffix(".embeddings.npy")
        if cache_path.exists() and emb_path.exists():
            meta = json.loads(cache_path.read_text())
            if meta.get("titles") == titles and meta.get("n_clusters") == n_clusters and meta.get("seed") == seed:
                return (
                    np.array(meta["topic_ids"]),
                    np.array(meta["centers"]),
                    np.load(emb_path),
                )

    from fedcond_grag.client.stage_b_condense.node_text_embedder import load_frozen_encoder

    model = load_frozen_encoder()
    embeddings = embed_article_groups(groups, model)
    topic_ids, centers = fit_topic_clusters(embeddings, n_clusters=n_clusters, seed=seed)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "titles": titles, "n_clusters": n_clusters, "seed": seed,
            "topic_ids": topic_ids.tolist(), "centers": centers.tolist(),
        }))
        np.save(emb_path, embeddings)

    return topic_ids, centers, embeddings


def dirichlet_allocate(
    topic_ids: np.ndarray, num_clients: int, alpha: float, seed: int,
) -> list[int]:
    """Step 4: for each topic, sample pi_k ~ Dirichlet(alpha,...,alpha),
    shuffle its article indices with a fixed seed, round target counts down
    then give remaining articles to the clients with the largest fractional
    remainders. Returns client_id per article (parallel to topic_ids)."""
    rng = np.random.default_rng(seed)
    assign = np.full(len(topic_ids), -1, dtype=int)
    for k in sorted(set(topic_ids.tolist())):
        idx = np.where(topic_ids == k)[0].copy()
        rng.shuffle(idx)
        n_k = len(idx)
        if n_k == 0:
            continue
        pi = rng.dirichlet(np.full(num_clients, alpha))
        raw = n_k * pi
        counts = np.floor(raw).astype(int)
        remainder = n_k - int(counts.sum())
        fracs = raw - counts
        order = np.argsort(-fracs)
        for j in range(remainder):
            counts[order[j % num_clients]] += 1
        pos = 0
        for c in range(num_clients):
            take = int(counts[c])
            assign[idx[pos : pos + take]] = c
            pos += take
    assert (assign >= 0).all(), "every article must be assigned to exactly one client"
    return assign.tolist()


def random_allocate(num_groups: int, num_clients: int, seed: int) -> list[int]:
    """Topic-agnostic reference: shuffle all article groups (fixed seed),
    split into num_clients approximately-equal parts."""
    rng = np.random.default_rng(seed)
    idx = np.arange(num_groups)
    rng.shuffle(idx)
    base = num_groups // num_clients
    counts = np.full(num_clients, base, dtype=int)
    counts[: num_groups % num_clients] += 1
    assign = np.full(num_groups, -1, dtype=int)
    pos = 0
    for c in range(num_clients):
        assign[idx[pos : pos + counts[c]]] = c
        pos += counts[c]
    assert (assign >= 0).all()
    return assign.tolist()


def partition_chunks_by_topic(
    chunks: list[LinearRAGChunk],
    *,
    num_clients: int,
    mode: str,  # "random" | "dirichlet"
    alpha: float | None = None,
    num_topic_clusters: int = DEFAULT_NUM_TOPIC_CLUSTERS,
    seed: int = 42,
    cluster_cache_path: str | Path | None = None,
) -> tuple[list[ClientChunks], dict]:
    """Top-level entry point. Returns (clients, diagnostics) where clients
    has the same type `partition_linearrag_chunks` returns."""
    groups = dedupe_and_group_chunks(chunks)
    diagnostics: dict = {
        "mode": mode,
        "num_articles": len(groups),
        "num_passages_after_dedup": sum(len(g.chunks) for g in groups),
        "num_passages_before_dedup": len(chunks),
    }

    if mode == "random":
        assign = random_allocate(len(groups), num_clients, seed)
    elif mode == "dirichlet":
        if alpha is None:
            raise ValueError("alpha is required for partition mode 'dirichlet'")
        topic_ids, centers, embeddings = _load_or_fit_clusters(
            groups, num_topic_clusters, seed, cluster_cache_path
        )
        assign = dirichlet_allocate(topic_ids, num_clients, alpha, seed)
        diagnostics["alpha"] = alpha
        diagnostics["num_topic_clusters"] = num_topic_clusters
        diagnostics["topic_ids"] = topic_ids.tolist()
        diagnostics["representative_titles"] = representative_titles(groups, embeddings, topic_ids, centers)
    else:
        raise ValueError(f"unknown partition mode {mode!r} (expected 'random' or 'dirichlet')")

    clients = [ClientChunks(client_id=i, num_clients=num_clients, chunks=[]) for i in range(num_clients)]
    for group, cid in zip(groups, assign):
        clients[cid].chunks.extend(group.chunks)

    diagnostics["client_sizes_articles"] = [assign.count(c) for c in range(num_clients)]
    diagnostics["client_sizes_passages"] = [len(cl.chunks) for cl in clients]
    if mode == "dirichlet":
        hist = topic_histogram(assign, np.array(diagnostics["topic_ids"]), num_clients, num_topic_clusters)
        diagnostics["topic_histogram_per_client"] = hist.tolist()
        diagnostics["mean_pairwise_js_divergence"] = pairwise_mean_js(hist)

    # Manifest: article-to-client and passage(chunk-index)-to-client, so this
    # partition can be reconstructed / cross-referenced (e.g. against gold
    # question evidence) without re-running clustering or allocation.
    diagnostics["article_to_client"] = {group.title: int(cid) for group, cid in zip(groups, assign)}
    diagnostics["passage_index_to_client"] = {
        int(chunk.index): int(cid)
        for group, cid in zip(groups, assign)
        for chunk in group.chunks
    }

    return clients, diagnostics


# ---------------------------------------------------------------------------
# Reporting (Appendix C.1)
# ---------------------------------------------------------------------------

def topic_histogram(assign: list[int], topic_ids: np.ndarray, num_clients: int, num_topics: int) -> np.ndarray:
    hist = np.zeros((num_clients, num_topics))
    for a, t in zip(assign, topic_ids.tolist()):
        hist[a, t] += 1
    row_sums = hist.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return hist / row_sums


def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def pairwise_mean_js(hist: np.ndarray) -> float:
    n = hist.shape[0]
    vals = [_jensen_shannon_divergence(hist[i], hist[j]) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(vals)) if vals else 0.0


def representative_titles(
    groups: list[ArticleGroup], embeddings: np.ndarray, topic_ids: np.ndarray, centers: np.ndarray, k: int = 5,
) -> dict[int, list[str]]:
    """Five representative article titles per cluster, by distance to the
    K-means centroid -- for inspecting cluster interpretability rather than
    inventing topic names up front."""
    out: dict[int, list[str]] = {}
    for c_id in range(centers.shape[0]):
        idx = np.where(topic_ids == c_id)[0]
        if len(idx) == 0:
            out[c_id] = []
            continue
        dists = np.linalg.norm(embeddings[idx] - centers[c_id], axis=1)
        nearest = idx[np.argsort(dists)][:k]
        out[c_id] = [groups[i].title for i in nearest]
    return out
