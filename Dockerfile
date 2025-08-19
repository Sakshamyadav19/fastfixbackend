FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# system deps (git for shallow clones if needed)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/ /app/

# install
RUN pip install --no-cache-dir -r requirements.txt

# default port for many platforms
ENV PORT=8080
# You can tune these via env on the platform
ENV STARTER_KIT_ROUTE_TIMEOUT_SECS=10 \
    RETRIEVAL_TIMEOUT_SECS=3 \
    LLM_TIMEOUT_SECS=6

# Run with gunicorn
CMD exec gunicorn -w 2 -k gthread -t 90 \
  -b 0.0.0.0:${PORT} \
  --threads 8 \
  wsgi:app
