import re
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from .config import CHROMA_PERSIST_PATH, CHROMA_COLLECTION_PREFIX
from .embeddings import embed_texts  # uses Gemini
from .embeddings import _split_text

import time
MAX_SUBCHUNKS_PER_REPO = 120   # hard cap
MAX_EMBED_SECONDS      = 12.0  # soft deadline

# We’ll use Chroma’s client in-memory (or persistent if env is set)
def _make_client():
    if CHROMA_PERSIST_PATH:
        return chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
    return chromadb.Client()

_client = _make_client()

def _sanitize_name(name: str) -> str:
    # Chroma collection names must be simple; replace odd chars with '_'
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)

def collection_name_for_repo(repo_key: str) -> str:
    # repo_key is "owner/name@sha"
    return _sanitize_name(f"{CHROMA_COLLECTION_PREFIX}__{repo_key}")

def get_or_create_collection(name: str):
    try:
        return _client.get_collection(name=name)
    except Exception:
        return _client.create_collection(name=name)



def upsert_chunks(repo_key: str, chunks: List[Dict[str, Any]]) -> None:
    coll = get_or_create_collection(collection_name_for_repo(repo_key))

    ids, docs, metas = [], [], []
    used = 0
    t0 = time.time()

    for c in chunks:
        if used >= MAX_SUBCHUNKS_PER_REPO or (time.time() - t0) > MAX_EMBED_SECONDS:
            break

        text = (c.get("text") or "").strip()
        if not text:
            continue

        sub_chunks = _split_text(text)
        for idx, sub in enumerate(sub_chunks):
            if used >= MAX_SUBCHUNKS_PER_REPO or (time.time() - t0) > MAX_EMBED_SECONDS:
                break
            cid = f"{c['path']}#{c['start_line']}-{c['end_line']}::{idx}"
            ids.append(cid)
            docs.append(sub)
            metas.append({
                "path": c["path"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "kind": c.get("kind"),
                "symbol": c.get("symbol"),
            })
            used += 1

    if not docs:
        return

    embeddings = embed_texts(docs)
    packed = [
        (i, d, m, e)
        for i, d, m, e in zip(ids, docs, metas, embeddings)
        if e
    ]
    if not packed:
        return

    coll.upsert(
        ids=[p[0] for p in packed],
        documents=[p[1] for p in packed],
        metadatas=[p[2] for p in packed],
        embeddings=[p[3] for p in packed],
    )


def query_topk(repo_key: str, query_text: str, top_k: int = 25) -> List[Dict[str, Any]]:
    coll = get_or_create_collection(collection_name_for_repo(repo_key))
    if coll.count() == 0:
        return []
    q_vecs = embed_texts([query_text])
    if not q_vecs:
        return []
    q_embed = q_vecs[0]

    res = coll.query(
        query_embeddings=[q_embed],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    out = []
    for i in range(len(ids)):
        meta = metas[i] or {}
        out.append({
            "id": ids[i],
            "path": meta.get("path"),
            "start_line": meta.get("start_line"),
            "end_line": meta.get("end_line"),
            "kind": meta.get("kind"),
            "symbol": (meta.get("symbol") if isinstance(meta.get("symbol"), (str, type(None))) else None),
            "preview": (docs[i] or "")[:500],
            "score": dists[i] if i < len(dists) else None
        })
    return out
