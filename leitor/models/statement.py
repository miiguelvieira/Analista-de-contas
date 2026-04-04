"""Modelo de extrato bancário (uma conta, um período)."""
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from leitor.models.transaction import Transaction


@dataclass
class Statement:
    bank: str
    account_holder: str | None = None
    period_start: date | None = None
    period_end: date | None = None
    opening_balance: Decimal | None = None
    closing_balance: Decimal | None = None
    transactions: list[Transaction] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)

    @property
    def total_income(self) -> Decimal:
        return sum((t.amount for t in self.transactions if t.amount > 0), Decimal("0"))

    @property
    def total_expenses(self) -> Decimal:
        return sum((t.amount for t in self.transactions if t.amount < 0), Decimal("0"))

    @property
    def net(self) -> Decimal:
        return self.total_income + self.total_expenses
