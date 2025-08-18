from flask import Blueprint, request, jsonify
import asyncio, re, time
import json


from .github import github_graphql_search
from .repopack import build_repo_pack, retrieve_chunks_for_hints
from .prompt import SYSTEM_PROMPT, build_user_message
from .llm import generate_hints_text
from .config import (
    STARTER_KIT_ROUTE_TIMEOUT_SECS,
    RETRIEVAL_TIMEOUT_SECS,
    HINTS_CACHE_TTL,
)

api_bp = Blueprint("api", __name__)

# ---------------------------
# Helpers
# ---------------------------

_HINTS_CACHE = {}

def _summarize_issue_text(md: str) -> str:
    """LLM-free summary: prefer 'Objective:' line or first bullet, else first paragraph; strip markdown."""
    if not md:
        return ""
    text = md.strip()

    m = re.search(r'(?im)^\s*(?:\*\*?)?\s*objective[:\-\s]+(.+)$', text)
    if m and m.group(1).strip():
        candidate = m.group(1).strip()
    else:
        m = re.search(r'(?m)^\s*(?:-|\*|\d+\.)\s+(.+)$', text)
        candidate = m.group(1).strip() if m else ""

    if not candidate:
        candidate = text.split("\n\n", 1)[0].strip()

    # Strip markdown/html
    candidate = re.sub(r'`([^`]*)`', r'\1', candidate)
    candidate = re.sub(r'\*\*([^*]+)\*\*', r'\1', candidate)
    candidate = re.sub(r'\*([^*]+)\*', r'\1', candidate)
    candidate = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', candidate)
    candidate = re.sub(r'<[^>]+>', '', candidate)
    candidate = re.sub(r'\s+', ' ', candidate).strip()

    if len(candidate) > 240:
        candidate = candidate[:237].rstrip() + "…"
    return candidate


def _extract_readme_deps_tests(pack: dict) -> tuple[str, str, str]:
    """Return small excerpts for README, deps, tests."""
    readme, deps, tests = [], [], []
    for ch in pack.get("chunks", []):
        p = (ch.get("path") or "").lower()
        txt = ch.get("text") or ""
        if "readme" in p and len(readme) < 1:
            readme.append(txt[:1600])
        if p.endswith("requirements.txt") or p.endswith("pyproject.toml") or p.endswith("package.json"):
            if len(deps) < 2:
                deps.append(txt[:800])
        if "/test" in p or "tests/" in p:
            if len(tests) < 6:
                tests.append(p)
    return (readme[0] if readme else ""), ("\n\n".join(deps) if deps else ""), ("\n".join(tests) if tests else "")


def _fallback_hints(issue: dict) -> dict:
    """Minimal deterministic hints when LLM times out/errors."""
    title = issue.get("title", "")
    return {
        "high_level_goal": f"Address the request in: {title}"[:200],
        "where_to_work": ["Skim the README for run steps and scan tests related to the feature."],
        "what_to_change": ["Locate the function or route mentioned in the issue and adjust the minimal logic."],
        "how_to_verify": ["Run the repository's test or dev scripts and validate the behavior manually."],
        "gotchas": ["Avoid large refactors; keep the change scoped to the issue."],
    }


# ---- Graceful parser for LLM output ----

_HEADING_PAT = re.compile(
    r"""(?imx)
    ^\s*
    (?:\#{1,6}\s*)?            # optional markdown heading marks
    (?:\*\*)?                  # optional opening bold
    (?:\d+ [\.\)] \s*)?        # optional numbering like '1.' or '1)'
    (High[- ]?level\ goal|Where\ to\ work|What\ to\ change|How\ to\ verify|Gotchas)
    (?:\*\*)?                  # optional closing bold
    \s* :? \s*$                # optional trailing colon
    """
)

def _split_sections(text: str) -> dict:
    """
    Try to split by headings. If headings are missing, return {} and let classifier handle it.
    """
    out = {"High-level goal": "", "Where to work": "", "What to change": "", "How to verify": "", "Gotchas": ""}
    text = (text or "")
    # light markdown cleanup so headings match more reliably
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # strip bold
    text = re.sub(r'`([^`]*)`', r'\1', text)  
    blocks = {}
    # Find headings and their positions
    matches = list(_HEADING_PAT.finditer(text))
    if not matches:
        return {}
    # Append end sentinel
    spans = [(m.group(1), m.start(), m.end()) for m in matches]
    spans.append(("__END__", len(text), len(text)))
    # Capture content between headings
    for i in range(len(spans) - 1):
        name, _, end = spans[i]
        next_start = spans[i+1][1]
        content = text[end:next_start].strip()
        blocks[name] = content
    # Normalize to output keys
    out["High-level goal"] = blocks.get("High-level goal", "").strip()
    out["Where to work"]   = blocks.get("Where to work", "").strip()
    out["What to change"]  = blocks.get("What to change", "").strip()
    out["How to verify"]   = blocks.get("How to verify", "").strip()
    out["Gotchas"]         = blocks.get("Gotchas", "").strip()
    return out


_CITATION_PAT = re.compile(r"\(([^\s()]+):\s*(\d+)[–-](\d+)\)")  # (path: start–end) en-dash or hyphen
_CMD_HINT_PAT = re.compile(r"\b(pytest|npm|pnpm|yarn|python\s+manage\.py|docker|make)\b", re.I)

def _bulletize(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    # normalize bullets: strip common bullet markers and numeric prefixes
    norm = []
    for l in lines:
        l = l.lstrip("•*- \t")
        l = re.sub(r'^\s*\d+[\.\)]\s*', '', l)   # drop leading "1." or "1)"
        if l:
            norm.append(l)

    items = norm if len(norm) >= 2 else [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if s.strip()]

    # cap & dedupe
    items = [i[:300] for i in items if i]
    seen, uniq = set(), []
    for i in items:
        key = i.lower()
        if key in seen: 
            continue
        seen.add(key)
        uniq.append(i)
    return uniq[:8]



def _classify_into_schema(text: str) -> dict:
    """
    If headings are missing, distribute content into our 5 buckets using heuristics.
    """
    items = _bulletize(text)
    if not items:
        return {
            "high_level_goal": "",
            "where_to_work": [],
            "what_to_change": [],
            "how_to_verify": [],
            "gotchas": [],
        }

    high_level_goal = ""
    where_to_work, what_to_change, how_to_verify, gotchas = [], [], [], []

    # First short sentence as goal if it reads like a summary
    for i in items:
        if len(i.split()) <= 22 and re.search(r"\b(fix|implement|add|handle|return|update|support)\b", i, re.I):
            high_level_goal = i
            break

    for i in items:
        # Skip if it's exactly our high-level goal
        if high_level_goal and i == high_level_goal:
            continue

        has_citation = bool(_CITATION_PAT.search(i))
        has_cmd = bool(_CMD_HINT_PAT.search(i))
        looks_gotcha = bool(re.match(r"(?i)^(beware|note|gotcha|caution|edge case|watch out)", i))

        if has_cmd:
            how_to_verify.append(i)
        elif looks_gotcha:
            gotchas.append(i)
        elif has_citation:
            # If it references code, prefer "where to work"
            where_to_work.append(i)
        else:
            what_to_change.append(i)

    # Balance & cap
    where_to_work = where_to_work[:3]
    what_to_change = what_to_change[:3]
    how_to_verify = how_to_verify[:2]
    gotchas = gotchas[:2]

    return {
        "high_level_goal": high_level_goal,
        "where_to_work": where_to_work,
        "what_to_change": what_to_change,
        "how_to_verify": how_to_verify,
        "gotchas": gotchas,
    }

_JSON_BLOCK = re.compile(r'\{[\s\S]*\}')

_EXPECTED_KEYS = ["high_level_goal","where_to_work","what_to_change","how_to_verify","gotchas"]

def _clean_item(s: str) -> str:
    s = (s or "").strip()
    # drop leading bullets/numbering/headings
    s = s.lstrip("•*- \t")
    s = re.sub(r'^\d+[\.\)]\s*', '', s)          # 1.  2) etc
    s = re.sub(r'^\s*(High[- ]?level goal|Where to work|What to change|How to verify|Gotchas)\s*:?\s*$', '', s, flags=re.I)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _norm_list(xs, limit):
    out = []
    for x in (xs or []):
        x = _clean_item(str(x))
        if not x: 
            continue
        if len(x) <= 2:   # drop "1." or other stubs
            continue
        out.append(x[:300])
        if len(out) >= limit:
            break
    # dedupe preserve order
    seen, uniq = set(), []
    for i in out:
        k = i.lower()
        if k in seen: 
            continue
        seen.add(k)
        uniq.append(i)
    return uniq

def _normalize_schema(d: dict) -> dict:
    # ensure all keys present with correct types
    out = {k: d.get(k) for k in _EXPECTED_KEYS}
    out["high_level_goal"] = _clean_item(out.get("high_level_goal") or "")
    out["where_to_work"]   = _norm_list(out.get("where_to_work") or [], 3)
    out["what_to_change"]  = _norm_list(out.get("what_to_change") or [], 3)
    out["how_to_verify"]   = _norm_list(out.get("how_to_verify") or [], 2)
    out["gotchas"]         = _norm_list(out.get("gotchas") or [], 2)

    # rebalance: if all lists empty but we have a goal, clone a gentle nudge
    if not any([out["where_to_work"], out["what_to_change"], out["how_to_verify"]]):
        if out["high_level_goal"]:
            out["what_to_change"] = [out["high_level_goal"]]

    # if goal missing but what_to_change has content, promote first item
    if not out["high_level_goal"] and out["what_to_change"]:
        out["high_level_goal"] = out["what_to_change"][0]

    return out

def _parse_llm_json(raw_text: str) -> dict | None:
    """
    Try to parse strict JSON from the model; return normalized dict or None.
    """
    if not raw_text:
        return None
    # grab the first JSON-looking block
    m = _JSON_BLOCK.search(raw_text)
    block = m.group(0) if m else raw_text
    try:
        obj = json.loads(block)
        # allow top-level nesting under "hints"
        if isinstance(obj, dict) and "hints" in obj and isinstance(obj["hints"], dict):
            obj = obj["hints"]
        if not isinstance(obj, dict):
            return None
        # accept both snake_case and space headings
        mapped = {}
        for k in _EXPECTED_KEYS:
            if k in obj:
                mapped[k] = obj[k]
            else:
                # allow alternate keys e.g. "High-level goal"
                alt = {
                    "high_level_goal": ["High-level goal","High level goal","Goal"],
                    "where_to_work": ["Where to work"],
                    "what_to_change": ["What to change"],
                    "how_to_verify": ["How to verify","Verification"],
                    "gotchas": ["Gotchas","Caveats","Pitfalls"],
                }[k]
                for a in alt:
                    if a in obj:
                        mapped[k] = obj[a]
                        break
        return _normalize_schema(mapped)
    except Exception:
        return None

def _postprocess_llm_text(text: str) -> dict:
    """
    Convert raw LLM text to our JSON schema (gracefully handles missing headings).
    """
    text = (text or "").strip()
    # enforce soft cap ~250 words
    words = text.split()
    if len(words) > 260:
        text = " ".join(words[:260])

    # Try structured split first
    sections = _split_sections(text)
    if sections:
        def _to_list(txt):
            return [l.strip("•- ").strip() for l in txt.splitlines() if l.strip()][:6]
        return {
            "high_level_goal": (sections.get("High-level goal") or "").strip(),
            "where_to_work": _to_list(sections.get("Where to work", "")),
            "what_to_change": _to_list(sections.get("What to change", "")),
            "how_to_verify": _to_list(sections.get("How to verify", "")),
            "gotchas": _to_list(sections.get("Gotchas", "")),
        }

    # Headings missing: classify heuristically
    return _classify_into_schema(text)


# ---------------------------
# Routes
# ---------------------------

@api_bp.get("/search")
def search():
    """
    Example: /api/search?skills=python,flask
    Returns a list of issue cards with repo fingerprints.
    """
    raw = request.args.get("skills", "") or ""
    skills = [s.strip() for s in raw.split(",") if s.strip()]
    if not skills:
        skills = ["python"]

    try:
        data = asyncio.run(github_graphql_search(skills, first=20))
        return jsonify(data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.get("/starter_kit")
def starter_kit():
    """
    Usage: /api/starter_kit?owner=<org>&repo=<name>&number=<issue_number>
    Returns issue summary, run_hints, and LLM hints (no files/snippets section).
    """
    owner = request.args.get("owner")
    repo  = request.args.get("repo")
    number = request.args.get("number")

    if not (owner and repo and number and number.isdigit()):
        return jsonify({"error": "owner, repo, number are required"}), 400

    async def _work():
        pack = await build_repo_pack(owner, repo, int(number))

        # Add summary (LLM-free)
        issue = pack.get("issue", {})
        issue["summary"] = _summarize_issue_text(issue.get("bodyText", ""))

        # Heuristic run hints
        run_hints = []
        for ch in pack.get("chunks", []):
            p = (ch.get("path") or "").lower()
            txt = ch.get("text") or ""
            if p.endswith("package.json"):
                if '"dev"' in txt:
                    run_hints.append("npm run dev  # from package.json")
                elif '"start"' in txt:
                    run_hints.append("npm start  # from package.json")
                else:
                    run_hints.append("npm install && npm run <script>  # check package.json")
            if p.endswith("requirements.txt") or p.endswith("pyproject.toml"):
                run_hints.append("python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt")
            if p.endswith("manage.py"):
                run_hints.append("python manage.py runserver")
            if p.endswith("dockerfile"):
                run_hints.append("docker build -t app . && docker run -p 3000:3000 app")
        run_hints = list(dict.fromkeys(run_hints))[:5]

        # Retrieval (K=6) with its own timeout
        try:
            chunks = await asyncio.wait_for(
                asyncio.to_thread(retrieve_chunks_for_hints, pack, 6, 0.35),
                timeout=RETRIEVAL_TIMEOUT_SECS,
            )
        except asyncio.TimeoutError:
            chunks = []

        # Repo metadata excerpts
        readme_excerpt, deps_excerpt, tests_excerpt = _extract_readme_deps_tests(pack)

        # Cache key
        cache_key = f"hints::{pack['repo_key']}::#{issue.get('number')}"
        cached = _HINTS_CACHE.get(cache_key)
        hints, fallback = None, None
        if cached:
            expires_at, cached_data = cached
            if time.time() < expires_at and not cached_data.get("hints_fallback", False):
                hints = cached_data["hints"]
                fallback = False
            else:
                # expired or was a fallback — treat as miss so we try again
                _HINTS_CACHE.pop(cache_key, None)


        # If not cached, call LLM (with graceful parsing)
        # If not cached, call LLM (with graceful parsing)
        if hints is None:
            user_msg = build_user_message(issue, readme_excerpt, deps_excerpt, tests_excerpt, chunks)
            try:
                raw_text = await generate_hints_text(SYSTEM_PROMPT, user_msg)
                # NEW: try JSON first
                parsed = _parse_llm_json(raw_text)
                if parsed:
                    hints = parsed
                else:
                    # fallback to the existing text post-processor
                    hints = _postprocess_llm_text(raw_text)
                fallback = False
            except Exception as e:
                print("LLM_ERROR:", repr(e))
                hints = _fallback_hints(issue)
                fallback = True



        return {
            "owner": owner,          # added for clickable citations on the client
            "repo": repo,            # added for clickable citations on the client
            "repo_key": pack["repo_key"],
            "sha": pack["sha"],
            "issue": issue,
            "run_hints": run_hints,
            "hints": hints,
            "hints_fallback": fallback,
        }

    try:
        result = asyncio.run(asyncio.wait_for(_work(), timeout=STARTER_KIT_ROUTE_TIMEOUT_SECS))
        return jsonify(result), 200
    except asyncio.TimeoutError:
        # Total route timeout — return minimal fallback (still include owner/repo for links)
        return jsonify({
            "owner": owner,
            "repo": repo,
            "error": "starter_kit_timeout",
            "hints": _fallback_hints({"title": f"{owner}/{repo} #{number}"}),
            "hints_fallback": True,
        }), 200
    except Exception as e:
        import traceback, sys
        print("STARTER_KIT_ERROR:", repr(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
