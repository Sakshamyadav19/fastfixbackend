import os
from dotenv import load_dotenv

load_dotenv()

# External URLs (env-overridable)
GITHUB_GRAPHQL_URL = os.getenv("GITHUB_GRAPHQL_URL", "https://api.github.com/graphql")
GITHUB_API_URL     = os.getenv("GITHUB_API_URL", "https://api.github.com")
GITHUB_RAW_URL     = os.getenv("GITHUB_RAW_URL", "https://raw.githubusercontent.com")

# Auth
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# CORS
CORS_ORIGIN = os.getenv("CORS_ORIGIN")

# Repo Pack cache sizing
REPO_CACHE_MAX = int(os.getenv("REPO_CACHE_MAX", "20"))



# Gemini embeddings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")


# Chroma
CHROMA_PERSIST_PATH = os.getenv("CHROMA_PERSIST_PATH")  # if unset -> in-memory
CHROMA_COLLECTION_PREFIX = os.getenv("CHROMA_COLLECTION_PREFIX", "firstfix")


STARTER_KIT_ROUTE_TIMEOUT_SECS = int(os.getenv("STARTER_KIT_ROUTE_TIMEOUT_SECS", "10"))
RETRIEVAL_TIMEOUT_SECS         = int(os.getenv("RETRIEVAL_TIMEOUT_SECS", "3"))
LLM_TIMEOUT_SECS               = int(os.getenv("LLM_TIMEOUT_SECS", "6"))
HINTS_CACHE_TTL                = int(os.getenv("HINTS_CACHE_TTL", "3600"))  # seconds



