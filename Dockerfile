FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (better cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the backend
COPY . /app/

# Gunicorn settings
ENV PORT=8080
CMD exec gunicorn -w 2 -k gthread -t 90 \
  -b 0.0.0.0:${PORT} \
  --threads 8 \
  wsgi:app
