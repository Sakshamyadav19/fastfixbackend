from flask import Flask
from flask_cors import CORS
from .config import CORS_ORIGIN, GITHUB_TOKEN

def create_app():
    app = Flask(__name__)
    CORS(app)

    @app.get("/health")
    def health():
        return {"ok": True, "has_token": bool(GITHUB_TOKEN)}, 200

    from .routes import api_bp
    app.register_blueprint(api_bp, url_prefix="/api")

    if not GITHUB_TOKEN:
        app.logger.warning("GITHUB_TOKEN is not set. /api/* calls to GitHub will fail.")

    return app
