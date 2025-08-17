import time
from typing import List
import google.generativeai as genai
from .config import GEMINI_API_KEY, GEMINI_MODEL

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

# Force a valid embeddings model
EMBED_MODEL = GEMINI_MODEL or "models/text-embedding-004"

genai.configure(api_key=GEMINI_API_KEY)

MAX_EMBED_CHARS = 2000  # safe per-call size

def _split_text(text, max_len=MAX_EMBED_CHARS):
    text = (text or "")
    chunks = []
    while len(text) > max_len:
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        part = text[:split_at].strip()
        if part:
            chunks.append(part)
        text = text[split_at:]
    tail = text.strip()
    if tail:
        chunks.append(tail)
    return chunks

def _embed_one(text: str) -> List[float]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Attempted to embed empty text")
    if len(text) > MAX_EMBED_CHARS:
        text = text[:MAX_EMBED_CHARS]  # double-guard
    resp = genai.embed_content(model=EMBED_MODEL, content=text)
    # Handle response shapes for different SDK versions
    if isinstance(resp, dict):
        emb = resp.get("embedding")
        if isinstance(emb, dict) and "values" in emb:
            return list(emb["values"])
        if isinstance(emb, list):
            return emb
    if hasattr(resp, "embedding"):
        emb = resp.embedding
        if hasattr(emb, "values"):
            return list(emb.values)
        return list(emb)
    raise RuntimeError("Unexpected embedding response shape from Gemini")

def embed_texts(texts):
    out = []
    for t in texts:
        t = (t or "").strip()
        if not t:
            continue
        for chunk in _split_text(t):
            out.append(_embed_one(chunk))
    return out
