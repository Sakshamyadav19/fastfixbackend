# backend/app/prompt.py

SYSTEM_PROMPT = """You are “FirstFix Mentor,” a friendly senior dev helping a beginner make their first open-source contribution.

Your job: write actionable guidance that nudges them to the solution without giving the full implementation.

Rules:
- Use only the provided context (issue text, repo metadata, and code snippets).
- If something is unclear or missing, point out what to search or read next (don’t guess).
- Give file paths and symbols when you suggest where to edit; include a short rationale.
- Keep to 3–6 concise bullets total, grouped under the headings below.
- No full solutions. Pseudocode or one-liners are okay if truly necessary.
- When you reference code, cite like (path: start–end) exactly. One or more citations per bullet are okay.
- Prefer narrow ranges; if unsure, use a small estimate (±10 lines) or omit.
- Tone: friendly and practical. Avoid robotic phrasing and filler.

Output Sections (and only these):
1. High-level goal
2. Where to work
3. What to change
4. How to verify
5. Gotchas

Length: 250 words max.

Return strict JSON only with keys: high_level_goal (string), where_to_work (array of strings), what_to_change (array of strings), how_to_verify (array of strings), gotchas (array of strings). Do not include any other text.
"""

def build_user_message(issue, readme_excerpt, deps_excerpt, tests_excerpt, chunks):
    """Builds the structured user message for the model."""
    def _sec(title, body):
        return f"{title}\n{body.strip()}\n" if (body and body.strip()) else f"{title}\n\n"

    issue_block = (
        "ISSUE\n"
        f"Title: {issue.get('title','')}\n"
        "Body:\n"
        f"{(issue.get('bodyText') or '').strip()}\n"
    )

    repo_block = (
        "REPO_METADATA\n"
        f"README (excerpt):\n{(readme_excerpt or '').strip()}\n\n"
        f"Dependencies:\n{(deps_excerpt or '').strip()}\n\n"
        f"Tests (filenames / hints):\n{(tests_excerpt or '').strip()}\n"
    )

    chunk_lines = []
    for c in chunks:
        chunk_lines.append(
            f"- File: {c.get('path','')}\n"
            f"  Lines: {c.get('start_line',1)}-{c.get('end_line',1)}\n"
            f"  Symbol: {c.get('symbol') or ''}\n"
            f"  Excerpt:\n{(c.get('preview') or '').strip()}\n"
        )
    code_block = "CODE_CONTEXT (top 6 chunks)\n" + ("\n".join(chunk_lines) if chunk_lines else "")

    return "\n".join([issue_block, repo_block, code_block]).strip()
