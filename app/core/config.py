import os
import yaml
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# === CONFIG ===
DATA_DIR = os.getenv("DATA_DIR", None)
DB_DIR = os.getenv("DB_DIR", None)
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", None)
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", None)
CONFIG_FILE = os.getenv("CONFIG_FILE", "config.yaml")
ATTEMPT_LIMIT = int(os.getenv("ATTEMPT_LIMIT", 2))


def load_config():
    try:
        with open(CONFIG_FILE, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load config.yaml: {e}")
        return {
            "support_agent": {
                "name": "Support Agent",
                "email": "support@example.com"
            }
        } 