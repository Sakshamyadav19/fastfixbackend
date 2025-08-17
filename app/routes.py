from flask import Blueprint, request, jsonify
import asyncio
from .github import github_graphql_search
from .repopack import build_repo_pack, retrieve_issue_context_embed_chroma

api_bp = Blueprint("api", __name__)

@api_bp.get("/search")
def search():
    """
    Example: /api/search?skills=python,flask
    Returns a list of issue cards with repo fingerprints.
    """
    raw = request.args.get("skills", "") or ""
    skills = [s.strip() for s in raw.split(",") if s.strip()]

    # default to a gentle value if empty
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
    Returns a context bundle: repo@sha + top chunks relevant to the issue.
    """
    owner = request.args.get("owner")
    repo  = request.args.get("repo")
    number = request.args.get("number")

    # validate
    if not (owner and repo and number and number.isdigit()):
        return jsonify({"error": "owner, repo, number are required"}), 400

    try:
        print("Getting Pack for:", owner, repo, number)
        pack = asyncio.run(build_repo_pack(owner, repo, int(number)))
        print("Pack built:", pack["repo_key"], pack["sha"])
        top_chunks = retrieve_issue_context_embed_chroma(pack, top_k=25)
        print("Top chunks retrieved:", len(top_chunks))


        # Simple "how to run" heuristic from README/pyproject chunks
        run_hints = []
        for ch in pack["chunks"]:
            if ch["kind"] == "doc" and ch["path"].lower().startswith("readme"):
                if "flask" in ch["text"].lower():
                    if "flask run" in ch["text"].lower():
                        run_hints.append("flask run")
                if "pip install -e ." in ch["text"]:
                    run_hints.append("pip install -e .")
            if ch["path"].endswith("pyproject.toml") or "requirements.txt" in ch["path"].lower():
                run_hints.append("Create venv and install deps (pyproject/requirements found)")

        resp = {
            "repo_key": pack["repo_key"],
            "sha": pack["sha"],
            "issue": pack["issue"],
            "run_hints": list(dict.fromkeys(run_hints))[:5],  # unique, max 5
            "top_chunks": [
                {
                    "path": c["path"],
                    "start_line": c["start_line"],
                    "end_line": c["end_line"],
                    "kind": c.get("kind"),
                    "symbol": c.get("symbol"),
                    # From Chroma we already receive a preview; fall back to empty string
                    "preview": c.get("preview", "")[:500],
                }
                for c in top_chunks
            ],

        }
        return jsonify(resp), 200
    except Exception as e:
        # Log a clearer server-side error and return message to client
        import traceback, sys
        print("STARTER_KIT_ERROR:", repr(e), file=sys.stderr)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
