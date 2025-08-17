import os
import httpx

GQL_ENDPOINT = "https://api.github.com/graphql"

# GraphQL query: search open issues with filters, plus repo fingerprint fields
GQL_SEARCH = """
query SearchIssues($query: String!, $first: Int!) {
  search(type: ISSUE, query: $query, first: $first) {
    issueCount
    nodes {
      ... on Issue {
        id
        number
        title
        url
        createdAt
        updatedAt
        labels(first: 10) { nodes { name } }
        repository {
          nameWithOwner
          url
          isPrivate
          isArchived
          stargazerCount
          pushedAt
          primaryLanguage { name }
          languages(first: 5, orderBy: {field: SIZE, direction: DESC}) {
            edges { size node { name } }
          }
          repositoryTopics(first: 6) { nodes { topic { name } } }
          defaultBranchRef {
            target { ... on Commit { oid } } # commit SHA of default branch HEAD
          }
        }
      }
    }
  }
}
"""

def build_github_query(skills: list[str]) -> str:
    """
    Build a GitHub *issue* search query.

    Notes:
    - Use is:issue and is:open (NOT state:open).
    - `language:` does NOT work for issue search, so we don't use it.
    - Keep user tokens (e.g., "flask") as free-text to match title/body.
    - Add archived:false to avoid archived repos.
    """
    tokens = [t.strip() for t in skills if t.strip()]

    # Common beginner-friendly labels vary by repo; we'll OR them via multiple calls,
    # but keep a "default" string here too.
    base = ["is:issue", "is:open", "archived:false"]

    # Put user tokens as free text (e.g., flask, python). They'll match title/body.
    parts = base + tokens
    return " ".join(parts)

def with_label(q: str, label: str) -> str:
    # Append a label qualifier safely
    # Example: q="is:issue is:open flask" -> "is:issue is:open flask label:\"good first issue\""
    if '"' in label:
        # avoid messy escaping; we only call with our constants below
        label_part = f"label:{label}"
    else:
        label_part = f'label:"{label}"'
    return f"{q} {label_part}"


async def github_graphql_search(skills: list[str], first: int = 20):
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN not set")

    base_q = build_github_query(skills)

    # Try common variants. We'll fetch fewer per call and merge de-duped by issue id.
    label_variants = [
        "good first issue",
        "good-first-issue",
        "help wanted",
    ]

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=20) as client:
        all_nodes = []
        seen_ids = set()

        # Pull ~first/len(labels) from each to keep total near `first`
        per_call = max(5, first // len(label_variants))

        for lbl in label_variants:
            q = with_label(base_q, lbl)
            r = await client.post(
                GQL_ENDPOINT,
                headers=headers,
                json={"query": GQL_SEARCH, "variables": {"query": q, "first": per_call}},
            )
            r.raise_for_status()
            data = r.json()
            nodes = (data.get("data", {})
                        .get("search", {})
                        .get("nodes", []))
            for n in nodes:
                nid = n.get("id")
                if nid and nid not in seen_ids:
                    seen_ids.add(nid)
                    all_nodes.append(n)

    # If still empty (e.g., user typed something rare), try one fallback without labels
    if not all_nodes:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                GQL_ENDPOINT,
                headers=headers,
                json={"query": GQL_SEARCH, "variables": {"query": base_q, "first": first}},
            )
            r.raise_for_status()
            data = r.json()
            all_nodes = (data.get("data", {})
                           .get("search", {})
                           .get("nodes", []))

    # Normalize to cards
    cards = []
    for n in all_nodes:
        repo = n.get("repository", {}) or {}
        lang_edges = (repo.get("languages", {}) or {}).get("edges", []) or []
        languages = [{"name": e["node"]["name"], "share": e["size"]} for e in lang_edges]

        cards.append({
            "issue": {
                "id": n.get("id"),
                "number": n.get("number"),
                "title": n.get("title"),
                "url": n.get("url"),
                "createdAt": n.get("createdAt"),
                "updatedAt": n.get("updatedAt"),
                "labels": [l["name"] for l in (n.get("labels", {}) or {}).get("nodes", [])]
            },
            "repo": {
                "nameWithOwner": repo.get("nameWithOwner"),
                "url": repo.get("url"),
                "isPrivate": repo.get("isPrivate"),
                "isArchived": repo.get("isArchived"),
                "stargazerCount": repo.get("stargazerCount"),
                "pushedAt": repo.get("pushedAt"),
                "primaryLanguage": (repo.get("primaryLanguage") or {}).get("name"),
                "languages": languages,
                "topics": [t["topic"]["name"] for t in (repo.get("repositoryTopics", {}) or {}).get("nodes", [])],
                "defaultBranchSha": ((repo.get("defaultBranchRef") or {}).get("target") or {}).get("oid"),
            }
        })

    return {"items": cards}
