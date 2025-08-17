import httpx
from .config import GITHUB_TOKEN, GITHUB_GRAPHQL_URL, GITHUB_API_URL, GITHUB_RAW_URL

def gh_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

async def gh_graphql(client: httpx.AsyncClient, query: str, variables: dict):
    r = await client.post(
        GITHUB_GRAPHQL_URL,
        headers=gh_headers(),
        json={"query": query, "variables": variables},
        timeout=30
    )
    r.raise_for_status()
    return r.json()

async def gh_get(client: httpx.AsyncClient, path: str, params: dict | None = None):
    url = f"{GITHUB_API_URL}{path}"
    r = await client.get(url, headers=gh_headers(), params=params, timeout=30)
    r.raise_for_status()
    return r.json()

async def gh_raw_file(client: httpx.AsyncClient, owner: str, repo: str, sha: str, path: str) -> str:
    # Raw file content via raw.githubusercontent.com/{owner}/{repo}/{sha}/{path}
    url = f"{GITHUB_RAW_URL}/{owner}/{repo}/{sha}/{path}"
    r = await client.get(url, timeout=30)
    r.raise_for_status()
    return r.text
