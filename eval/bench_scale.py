"""Scale benchmark: retrieval latency and memory at 1k / 10k / 50k chunks (roadmap 22.5).

  python -m eval.bench_scale                 # sizes 1000 10000 50000
  python -m eval.bench_scale --sizes 1000 5000 --fail-p95-ms 150

Uses random vectors and synthetic text, so it measures the index + fusion + rerank path only
(no PDF extraction, no Ollama). Exit code 1 when the p95 target is missed.
"""

from __future__ import annotations

import argparse
import json
import random
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from app.config import Settings
from app.core.timing import percentile
from app.services.chunker import Chunk
from app.services.fakes import FakeEmbedder
from app.services.reranker import build_reranker
from app.services.retriever import Retriever
from app.services.vector_store import BM25Index, ChromaStore

VOCAB = [
    f"{random.Random(i).choice(['alpha', 'delta', 'omni', 'terra', 'nova', 'vita', 'kilo', 'mega'])}{i}"
    for i in range(4000)
]


def make_chunks(n: int, rng: random.Random, start: int = 0) -> list[Chunk]:
    out = []
    for i in range(start, start + n):
        doc, page = f"DOC{(i // 500) + 1:03d}", (i // 5) % 100 + 1
        text = " ".join(rng.choices(VOCAB, k=100))
        out.append(Chunk(f"{doc}-p{page:04d}-c{i:06d}", doc, f"f{doc}.pdf", page, i % 5, text))
    return out


def run(sizes: list[int], queries: int, ef_search: int) -> list[dict[str, float]]:
    results = []
    tmp = Path(tempfile.mkdtemp(prefix="docqa-bench-"))
    try:
        settings = Settings(_env_file=None, rerank_backend="lexical", hnsw_ef_search=ef_search, min_similarity=0.0)
        emb = FakeEmbedder()
        store = ChromaStore(tmp / "chroma", emb.model_name, ef_search=ef_search)
        bm25 = BM25Index()
        retriever = Retriever(settings, emb, store, bm25, build_reranker(settings))
        rng = random.Random(1)
        nprng = np.random.default_rng(1)
        have = 0
        for size in sizes:
            t0 = time.time()
            while have < size:
                batch = min(2000, size - have)
                chunks = make_chunks(batch, rng, have)
                vecs = nprng.normal(size=(batch, emb.dim)).astype("float32")
                vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
                store.upsert(chunks, vecs.tolist())
                bm25.add_chunks(chunks)
                have += batch
            build_s = time.time() - t0
            lat = []
            for _ in range(queries):
                q = " ".join(rng.choices(VOCAB, k=6))
                t = time.perf_counter()
                retriever.retrieve(q)
                lat.append((time.perf_counter() - t) * 1000)
            results.append(
                {
                    "chunks": size,
                    "build_s": round(build_s, 1),
                    "p50_ms": round(percentile(lat, 50), 1),
                    "p95_ms": round(percentile(lat, 95), 1),
                    "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
                }
            )
            print(json.dumps(results[-1]), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 50000])
    ap.add_argument("--queries", type=int, default=60)
    ap.add_argument("--ef-search", type=int, default=100)
    ap.add_argument("--fail-p95-ms", type=float, default=150.0)
    ap.add_argument("--out", type=Path, default=Path("eval/results/scale.json"))
    args = ap.parse_args()
    res = run(args.sizes, args.queries, args.ef_search)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 1 if any(r["p95_ms"] > args.fail_p95_ms for r in res) else 0


if __name__ == "__main__":
    sys.exit(main())
