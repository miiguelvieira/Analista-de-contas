"""Perfil financeiro consolidado de múltiplos bancos."""
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from leitor.models.statement import Statement
from leitor.models.transaction import Transaction


@dataclass
class MonthlySummary:
    year: int
    month: int
    income: Decimal = Decimal("0")
    expenses: Decimal = Decimal("0")
    balance: Decimal = Decimal("0")  # saldo acumulado ao fim do mês

    @property
    def net(self) -> Decimal:
        return self.income + self.expenses  # expenses é negativo

    @property
    def savings_rate(self) -> float:
        if self.income == 0:
            return 0.0
        return float(self.net / self.income)


@dataclass
class FinancialProfile:
    statements: list[Statement] = field(default_factory=list)
    all_transactions: list[Transaction] = field(default_factory=list)
    period_start: date | None = None
    period_end: date | None = None
    credit_score: int | None = None
    score_pillars: dict[str, float] = field(default_factory=dict)
    max_loan: Decimal | None = None
    loan_details: dict = field(default_factory=dict)
    monthly_summaries: dict[str, MonthlySummary] = field(default_factory=dict)

    @property
    def banks(self) -> list[str]:
        return list({s.bank for s in self.statements})

    @property
    def total_income(self) -> Decimal:
        return sum((t.amount for t in self.all_transactions if t.amount > 0), Decimal("0"))

    @property
    def total_expenses(self) -> Decimal:
        return sum((t.amount for t in self.all_transactions if t.amount < 0), Decimal("0"))

    @property
    def net(self) -> Decimal:
        return self.total_income + self.total_expenses
