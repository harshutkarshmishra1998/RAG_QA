from pathlib import Path
from dotenv import load_dotenv
import os

# --------------------------------------------------
# Project setup
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[0] # as you move inside as many folders increase the number (e.g. parents[1] for one level up, etc.)
load_dotenv(PROJECT_ROOT / ".env")

def require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing environment variable: {key}")
    return value

ENV_KEYS = {
    "LANGCHAIN_API_KEY": "LANGCHAIN_API",
    "LANGCHAIN_TRACING_V2": "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_PROJECT": "LANGCHAIN_PROJECT",
    "OPENAI_API_KEY": "OPENAI_API",
    "GROQ_API_KEY": "GROQ_API",
}

# ---- LOAD + EXPORT ----
_loaded = {}

for env_name, source_key in ENV_KEYS.items():
    value = require_env(source_key)
    os.environ[env_name] = value
    _loaded[env_name] = value

# ---- OPTIONAL: SAFE DEBUG (NO LEAKS) ----
if __name__ == "__main__":
    for k in _loaded:
        print(f"{k}: loaded")