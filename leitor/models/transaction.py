"""Modelo canônico de transação bancária."""
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal


@dataclass
class Transaction:
    id: str                          # SHA256(bank+date+amount+description[:40])
    bank: str
    date: date
    description: str
    amount: Decimal                  # negativo = saída, positivo = entrada
    category: str = "outros"
    subcategory: str = "geral"
    transaction_type: str = "debit"  # "debit" | "credit"
    categorization_source: str = "regex"  # "regex" | "claude" | "manual" | "user_rule"
    source_file: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def is_income(self) -> bool:
        return self.amount > 0

    @property
    def is_expense(self) -> bool:
        return self.amount < 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bank": self.bank,
            "date": self.date.isoformat(),
            "description": self.description,
            "amount": float(self.amount),
            "category": self.category,
            "subcategory": self.subcategory,
            "transaction_type": self.transaction_type,
            "categorization_source": self.categorization_source,
            "source_file": self.source_file,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Transaction":
        from datetime import date as date_type
        return cls(
            id=data["id"],
            bank=data["bank"],
            date=date_type.fromisoformat(data["date"]),
            description=data["description"],
            amount=Decimal(str(data["amount"])),
            category=data.get("category", "outros"),
            subcategory=data.get("subcategory", "geral"),
            transaction_type=data.get("transaction_type", "debit"),
            categorization_source=data.get("categorization_source", "regex"),
            source_file=data.get("source_file", ""),
            tags=data.get("tags", []),
        )
