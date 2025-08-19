# backend/wsgi.py
from app import create_app

app = create_app()   # Gunicorn will look for "app" here
