"""Configuração global: carrega variáveis de ambiente e caminhos."""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"

load_dotenv(BASE_DIR / ".env")

ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")

CATEGORIES_FILE = CONFIG_DIR / "categories.yaml"
BANK_PROFILES_FILE = CONFIG_DIR / "bank_profiles.yaml"
CATEGORIZATION_CACHE_FILE = DATA_DIR / "categorization_cache.json"
USER_RULES_FILE = DATA_DIR / "user_rules.json"

DATA_DIR.mkdir(exist_ok=True)
