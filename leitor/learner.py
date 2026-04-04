"""Sistema de aprendizado manual: gerencia regras de categorização do usuário."""
from __future__ import annotations

import json

from leitor.models.transaction import Transaction
from leitor.utils.config import USER_RULES_FILE


class Learner:
    """Persiste e aplica regras de categorização aprendidas pelo usuário."""

    def __init__(self):
        self._rules: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if USER_RULES_FILE.exists():
            try:
                data = json.loads(USER_RULES_FILE.read_text(encoding="utf-8"))
                return data.get("rules", [])
            except Exception:
                pass
        return []

    def _save(self) -> None:
        USER_RULES_FILE.parent.mkdir(exist_ok=True)
        USER_RULES_FILE.write_text(
            json.dumps({"rules": self._rules}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @property
    def rules(self) -> list[dict]:
        return list(self._rules)

    def add_rule_contains(self, pattern: str, category: str, subcategory: str = "geral") -> dict:
        """Adiciona regra: 'se descrição contiver pattern → categoria'."""
        rule = {
            "match_type": "contains",
            "pattern": pattern.lower().strip(),
            "category": category,
            "subcategory": subcategory,
        }
        # Evita duplicata
        for existing in self._rules:
            if existing.get("match_type") == "contains" and existing.get("pattern") == rule["pattern"]:
                existing["category"] = category
                existing["subcategory"] = subcategory
                self._save()
                return existing
        self._rules.insert(0, rule)
        self._save()
        return rule

    def add_rule_exact_id(self, transaction_id: str, category: str, subcategory: str = "geral") -> dict:
        """Adiciona exceção pontual para um transaction.id específico."""
        rule = {
            "match_type": "exact_id",
            "transaction_id": transaction_id,
            "category": category,
            "subcategory": subcategory,
        }
        for existing in self._rules:
            if existing.get("match_type") == "exact_id" and existing.get("transaction_id") == transaction_id:
                existing["category"] = category
                existing["subcategory"] = subcategory
                self._save()
                return existing
        self._rules.insert(0, rule)
        self._save()
        return rule

    def remove_rule(self, index: int) -> None:
        if 0 <= index < len(self._rules):
            self._rules.pop(index)
            self._save()

    def apply_to_transaction(self, txn: Transaction) -> Transaction:
        """Aplica regras do usuário a uma transação. Modifica in-place."""
        import unicodedata

        def norm(s: str) -> str:
            nfkd = unicodedata.normalize("NFKD", s)
            return nfkd.encode("ascii", "ignore").decode("ascii").lower()

        for rule in self._rules:
            mtype = rule.get("match_type")
            if mtype == "exact_id" and rule.get("transaction_id") == txn.id:
                txn.category = rule["category"]
                txn.subcategory = rule.get("subcategory", "geral")
                txn.categorization_source = "manual"
                return txn
            if mtype == "contains" and norm(rule["pattern"]) in norm(txn.description):
                txn.category = rule["category"]
                txn.subcategory = rule.get("subcategory", "geral")
                txn.categorization_source = "user_rule"
                return txn

        return txn

    def apply_to_all(self, transactions: list[Transaction]) -> list[Transaction]:
        """Aplica todas as regras a uma lista de transações."""
        return [self.apply_to_transaction(t) for t in transactions]

    def recategorize_after_rule(
        self, transactions: list[Transaction], new_rule: dict
    ) -> list[Transaction]:
        """Recategoriza todas as transações impactadas por uma nova regra."""
        import unicodedata

        def norm(s: str) -> str:
            nfkd = unicodedata.normalize("NFKD", s)
            return nfkd.encode("ascii", "ignore").decode("ascii").lower()

        affected = []
        for txn in transactions:
            mtype = new_rule.get("match_type")
            if mtype == "contains" and norm(new_rule["pattern"]) in norm(txn.description):
                txn.category = new_rule["category"]
                txn.subcategory = new_rule.get("subcategory", "geral")
                txn.categorization_source = "user_rule"
                affected.append(txn)
            elif mtype == "exact_id" and new_rule.get("transaction_id") == txn.id:
                txn.category = new_rule["category"]
                txn.subcategory = new_rule.get("subcategory", "geral")
                txn.categorization_source = "manual"
                affected.append(txn)

        return affected
