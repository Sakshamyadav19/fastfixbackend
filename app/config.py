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
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "http://localhost:5173")

# Repo Pack cache sizing
REPO_CACHE_MAX = int(os.getenv("REPO_CACHE_MAX", "20"))



# Gemini embeddings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")


# Chroma
CHROMA_PERSIST_PATH = os.getenv("CHROMA_PERSIST_PATH")  # if unset -> in-memory
CHROMA_COLLECTION_PREFIX = os.getenv("CHROMA_COLLECTION_PREFIX", "firstfix")


