"""Categorização de transações: regras do usuário → regex → Claude API."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import yaml

from leitor.models.transaction import Transaction
from leitor.utils.config import (
    ANTHROPIC_API_KEY,
    CATEGORIES_FILE,
    CATEGORIZATION_CACHE_FILE,
)


def _normalize_text(text: str) -> str:
    """Remove acentos e coloca em minúsculas para comparação."""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    return ascii_str.lower()


def _load_categories() -> dict[str, dict]:
    with open(CATEGORIES_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_cache() -> dict[str, dict]:
    if CATEGORIZATION_CACHE_FILE.exists():
        try:
            return json.loads(CATEGORIZATION_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache(cache: dict) -> None:
    CATEGORIZATION_CACHE_FILE.parent.mkdir(exist_ok=True)
    CATEGORIZATION_CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _cache_key(description: str) -> str:
    return _normalize_text(description)[:80]


class Categorizer:
    def __init__(self):
        self._categories = _load_categories()
        self._cache = _load_cache()
        self._user_rules: list[dict] = []  # injetado pelo Learner

    def set_user_rules(self, rules: list[dict]) -> None:
        self._user_rules = rules

    def categorize_one(self, description: str) -> tuple[str, str, str]:
        """Retorna (category, subcategory, source)."""
        norm = _normalize_text(description)

        # 1. Regras do usuário (máxima prioridade)
        for rule in self._user_rules:
            if rule.get("match_type") == "contains":
                if _normalize_text(rule["pattern"]) in norm:
                    return rule["category"], rule.get("subcategory", "geral"), "user_rule"
            elif rule.get("match_type") == "starts_with":
                if norm.startswith(_normalize_text(rule["pattern"])):
                    return rule["category"], rule.get("subcategory", "geral"), "user_rule"

        # 2. Cache Claude
        key = _cache_key(description)
        if key in self._cache:
            cached = self._cache[key]
            return cached["category"], cached.get("subcategory", "geral"), "claude"

        # 3. Regex / keywords
        for category, data in self._categories.items():
            for keyword in data.get("keywords", []):
                if _normalize_text(keyword) in norm:
                    return category, data.get("subcategory", "geral"), "regex"

        return "outros", "geral", "regex"

    def categorize_batch_claude(self, transactions: list[Transaction]) -> None:
        """Envia transações não categorizadas para Claude API em batch."""
        if not ANTHROPIC_API_KEY:
            return

        uncategorized = [
            t for t in transactions
            if t.category == "outros" and t.categorization_source == "regex"
        ]
        if not uncategorized:
            return

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

            batch_size = 50
            for i in range(0, len(uncategorized), batch_size):
                batch = uncategorized[i:i + batch_size]
                descriptions = [t.description for t in batch]

                categories_list = list(self._categories.keys())
                prompt = (
                    f"Categorize cada descrição de transação bancária brasileira abaixo.\n"
                    f"Categorias disponíveis: {', '.join(categories_list)}\n\n"
                    f"Responda APENAS com um JSON array, um objeto por linha, no formato:\n"
                    f'[{{"category": "categoria", "subcategory": "subcategoria_especifica"}}]\n\n'
                    f"Descrições (em ordem):\n"
                    + "\n".join(f"{j+1}. {d}" for j, d in enumerate(descriptions))
                )

                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}],
                )

                content = response.content[0].text.strip()
                # Extrai o JSON do response
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if not json_match:
                    continue

                results = json.loads(json_match.group())
                for txn, result in zip(batch, results):
                    cat = result.get("category", "outros")
                    sub = result.get("subcategory", "geral")
                    txn.category = cat
                    txn.subcategory = sub
                    txn.categorization_source = "claude"
                    # Salva no cache
                    key = _cache_key(txn.description)
                    self._cache[key] = {"category": cat, "subcategory": sub}

            _save_cache(self._cache)

        except Exception as e:
            # Falha silenciosa: mantém "outros" para não travar o fluxo
            pass

    def categorize_all(self, transactions: list[Transaction]) -> list[Transaction]:
        """Categoriza todas as transações: user_rules → regex → Claude API."""
        for txn in transactions:
            cat, sub, source = self.categorize_one(txn.description)
            txn.category = cat
            txn.subcategory = sub
            txn.categorization_source = source

        # Envia os "outros" para o Claude
        self.categorize_batch_claude(transactions)

        return transactions
