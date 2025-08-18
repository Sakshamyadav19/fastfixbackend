import re
import asyncio
import httpx
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
from .gh_client import gh_graphql, gh_get, gh_raw_file
from .config import REPO_CACHE_MAX
from .embeddings import embed_texts
import numpy as np
from .vectordb import upsert_chunks, query_topk
from typing import List
from .vectordb import query_topk

import time
def _log(msg: str):
    print(f"[RepoPack] {msg}", flush=True)



# In-memory LRU-ish cache (simple dict + eviction)
_REPO_PACKS: Dict[str, dict] = {}

GQL_ISSUE_MIN = """
query IssueOrPR($owner:String!, $name:String!, $number:Int!) {
  repository(owner:$owner, name:$name) {
    nameWithOwner
    defaultBranchRef {
      name
      target { ... on Commit { oid } }
    }
    issueOrPullRequest(number:$number) {
      __typename
      ... on Issue {
        id number title bodyText url
      }
      ... on PullRequest {
        id number title bodyText url
      }
    }
  }
}
"""


# Flask-relevant file patterns
FLASK_CODE = re.compile(r"""
    (^|/)(app\.py|wsgi\.py|manage\.py)$|
    (^|/)(__init__\.py|routes\.py|views\.py|errors\.py|models\.py|forms\.py)$|
    (^|/)(api|blueprints|routes|controllers)(/|$)
""", re.VERBOSE | re.IGNORECASE)

DOC_FILES = re.compile(r"(^|/)(README\.md|README|CONTRIBUTING\.md|CONTRIBUTING|Makefile|Dockerfile)$", re.IGNORECASE)
TEST_FILES = re.compile(r"(^|/)(tests?/)", re.IGNORECASE)
PY_FILE = re.compile(r"\.py$", re.IGNORECASE)
MD_FILE = re.compile(r"\.md$", re.IGNORECASE)

@dataclass
class Chunk:
    path: str
    start_line: int
    end_line: int
    kind: str  # "code" | "doc" | "config" | "test"
    text: str
    symbol: str | None = None
    tags: List[str] = None

def _evict_if_needed():
    if len(_REPO_PACKS) > REPO_CACHE_MAX:
        # pop an arbitrary (oldest would be nicer; simple for now)
        _REPO_PACKS.pop(next(iter(_REPO_PACKS)))

async def _get_repo_sha_and_issue(client, owner: str, name: str, number: int) -> Tuple[str, dict]:
    data = await gh_graphql(client, GQL_ISSUE_MIN, {"owner": owner, "name": name, "number": number})
    repo = (data.get("data") or {}).get("repository") or {}
    dbr = repo.get("defaultBranchRef") or {}
    sha = ((dbr.get("target") or {}).get("oid"))  # may be None
    item = repo.get("issueOrPullRequest")

    if item is None:
        raise RuntimeError(f"Issue/PR #{number} not found in {owner}/{name}")

    # Normalize to our expected shape
    issue = {
        "id": item.get("id"),
        "number": item.get("number"),
        "title": item.get("title"),
        "bodyText": item.get("bodyText") or "",
        "url": item.get("url"),
        "type": item.get("__typename"),
    }

    # Fallback if defaultBranchRef was missing/null
    if not sha:
        # 1) ask REST for default branch name
        repo_json = await gh_get(client, f"/repos/{owner}/{name}")
        default_branch = repo_json.get("default_branch") or "main"
        # 2) resolve the head ref to a commit SHA
        ref = await gh_get(client, f"/repos/{owner}/{name}/git/ref/heads/{default_branch}")
        obj = ref.get("object") or {}
        sha = obj.get("sha")
        if not sha:
            raise RuntimeError(f"Could not resolve default branch SHA for {owner}/{name}")

    return sha, issue


async def _list_tree(client, owner: str, name: str, sha: str) -> List[dict]:
    # Git tree listing with ?recursive=1
    tree = await gh_get(client, f"/repos/{owner}/{name}/git/trees/{sha}", params={"recursive": "1"})
    return tree.get("tree", [])

def _want_file(path: str) -> bool:
    return bool(
        FLASK_CODE.search(path) or
        DOC_FILES.search(path) or
        TEST_FILES.search(path) or
        PY_FILE.search(path)
    )

def _chunk_python(text: str) -> List[Chunk]:
    lines = text.splitlines()
    chunks: List[Chunk] = []
    # naive function/class splitter
    starts = []
    for i, line in enumerate(lines):
        if re.match(r"^\s*def\s+\w+\s*\(", line) or re.match(r"^\s*class\s+\w+\s*[:\(]", line):
            starts.append(i)
    starts.append(len(lines))  # sentinel

    for i in range(len(starts)-1):
        s = starts[i]
        e = starts[i+1]
        header = lines[s].strip() if s < len(lines) else ""
        symbol = None
        m1 = re.match(r"^\s*def\s+(\w+)\s*\(", header)
        m2 = re.match(r"^\s*class\s+(\w+)\s*[:\(]", header)
        if m1: symbol = m1.group(1)
        if m2: symbol = m2.group(1)
        chunk_text = "\n".join(lines[s:e]).strip()
        if chunk_text:
            chunks.append(Chunk(path="", start_line=s+1, end_line=e, kind="code", text=chunk_text, symbol=symbol, tags=[]))
    # fallback: if no defs/classes, return single file chunk
    if not chunks and text.strip():
        chunks.append(Chunk(path="", start_line=1, end_line=len(lines), kind="code", text=text, symbol=None, tags=[]))
    return chunks

def _chunk_markdown(text: str) -> List[Chunk]:
    lines = text.splitlines()
    chunks: List[Chunk] = []
    # split by headings
    starts = [i for i, line in enumerate(lines) if re.match(r"^#{1,6}\s", line)]
    starts = [0] + starts + [len(lines)]
    seen = set()
    for i in range(len(starts)-1):
        s, e = starts[i], starts[i+1]
        if (s, e) in seen or s == e:
            continue
        seen.add((s, e))
        chunk = "\n".join(lines[s:e]).strip()
        if chunk:
            chunks.append(Chunk(path="", start_line=s+1, end_line=e, kind="doc", text=chunk, symbol=None, tags=[]))
    return chunks or [Chunk(path="", start_line=1, end_line=len(lines), kind="doc", text=text, symbol=None, tags=[])]

def _chunk_plain(text: str, kind: str) -> List[Chunk]:
    # whole-file chunk
    lines = text.splitlines()
    return [Chunk(path="", start_line=1, end_line=len(lines), kind=kind, text=text, symbol=None, tags=[])]

def _keyword_score(query: str, text: str) -> int:
    # very simple overlap score
    terms = [t for t in re.split(r"[^a-zA-Z0-9_]+", query.lower()) if t]
    score = 0
    tlow = text.lower()
    for t in terms:
        if t in tlow:
            score += 1
    return score

async def build_repo_pack(owner: str, name: str, issue_number: int) -> dict:
    t0 = time.time()
    _log(f"start {owner}/{name} issue#{issue_number}")

    async with httpx.AsyncClient(timeout=20) as client:
        _log("resolve sha + issue via GraphQL")
        sha, issue = await _get_repo_sha_and_issue(client, owner, name, issue_number)
        key = f"{owner}/{name}@{sha}"

        if key in _REPO_PACKS:
            _log(f"cache HIT {key}")
            pack = _REPO_PACKS[key].copy()
            pack["issue"] = issue
            _log(f"done (cache) in {time.time()-t0:.2f}s")
            return pack

        _log(f"cache MISS {key} → list tree")
        tree = await _list_tree(client, owner, name, sha)
        if not tree:
            raise RuntimeError(f"No tree returned for {owner}/{name}@{sha}. The repo may be empty or API-limited.")

        # ---- strict budgets so first run is snappy ----
        MAX_FILES        = 60         # was 120
        MAX_TOTAL_CHARS  = 120_000    # was 250k
        MAX_FILE_CHARS   = 30_000     # was 50k
        # ----------------------------------------------

        candidates = [t for t in tree if t.get("type") == "blob" and _want_file(t.get("path",""))][:MAX_FILES]
        _log(f"{len(candidates)} candidate files (budget: {MAX_FILES})")

        chunks: List[dict] = []
        total_chars = 0
        for node in candidates:
            path = node["path"]
            try:
                content = await gh_raw_file(client, owner, name, sha, path)
            except httpx.HTTPError:
                continue

            if len(content) > MAX_FILE_CHARS:
                continue
            total_chars += len(content)
            if total_chars > MAX_TOTAL_CHARS:
                _log("character budget reached; stopping fetch")
                break

            if MD_FILE.search(path) or path.lower() in ("readme", "contributing"):
                in_chunks = _chunk_markdown(content)
            elif PY_FILE.search(path):
                in_chunks = _chunk_python(content)
            elif path.lower() in ("makefile", "dockerfile"):
                in_chunks = _chunk_plain(content, "config")
            elif TEST_FILES.search(path):
                in_chunks = _chunk_plain(content, "test")
            else:
                in_chunks = _chunk_plain(content, "doc")

            for c in in_chunks:
                c.path = path
                chunks.append(asdict(c))

        _log(f"fetched & chunked {len(chunks)} chunks (chars≈{total_chars})")

        # Upsert to vector DB (embeddings happen here)
        _log("upsert to Chroma (will embed sub-chunks)")
        repo_key = key
        upsert_chunks(repo_key, chunks)
        _log("upsert complete")

        pack = {"repo_key": key, "sha": sha, "issue": issue, "chunks": chunks}
        _REPO_PACKS[key] = pack
        _evict_if_needed()
        _log(f"done cold build in {time.time()-t0:.2f}s")
        return pack


def retrieve_issue_context_embed_chroma(pack: dict, top_k: int = 25) -> list[dict]:
    issue_text = f"{pack['issue']['title']}\n{pack['issue'].get('bodyText','')}"
    return query_topk(pack["repo_key"], issue_text, top_k=top_k)



def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def retrieve_issue_context_embed(pack: dict, top_k: int = 25) -> list[dict]:
    """
    Embedding-based retrieval: embed the issue text and score against chunks.
    """
    issue_text = f"{pack['issue']['title']}\n{pack['issue'].get('bodyText','')}"
    query_vec = embed_texts([issue_text])[0]

    scored = []
    for ch in pack["chunks"]:
        if "embedding" not in ch:
            continue
        sim = cosine_sim(query_vec, ch["embedding"])
        scored.append((sim, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for s, c in scored[:top_k]]



def retrieve_chunks_for_hints(pack: dict, k: int = 6, min_sim: float = 0.35) -> List[dict]:
    """
    Use vector search to get top chunks relevant to the issue, dedupe by path, cap to k.
    Converts Chroma distance -> crude similarity = 1 - distance, filters by min_sim.
    """
    issue_text = f"{pack['issue']['title']}\n{pack['issue'].get('bodyText','')}"
    # ask for more, then downselect to diverse set
    raw = query_topk(pack["repo_key"], issue_text, top_k=24) or []
    by_path = {}
    for c in raw:
        # distance lower is better; derive similarity
        sim = 0.0
        if c.get("score") is not None:
            try:
                sim = max(0.0, 1.0 - float(c["score"]))
            except Exception:
                sim = 0.0
        # keep first/best per path meeting threshold
        p = c.get("path")
        if not p or sim < min_sim:
            continue
        if p not in by_path:
            c2 = dict(c)
            c2["similarity"] = round(sim, 3)
            by_path[p] = c2
        if len(by_path) >= k:
            break
    return list(by_path.values())
