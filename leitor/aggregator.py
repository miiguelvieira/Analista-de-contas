"""Agrega múltiplos extratos em um FinancialProfile unificado."""
from __future__ import annotations

from datetime import date
from decimal import Decimal

from leitor.models.profile import FinancialProfile, MonthlySummary
from leitor.models.statement import Statement
from leitor.models.transaction import Transaction


def build_profile(statements: list[Statement]) -> FinancialProfile:
    """Combina múltiplos extratos em um FinancialProfile consolidado."""
    # Deduplica transações por ID
    seen_ids: set[str] = set()
    all_transactions: list[Transaction] = []

    for stmt in statements:
        for txn in stmt.transactions:
            if txn.id not in seen_ids:
                seen_ids.add(txn.id)
                all_transactions.append(txn)

    all_transactions.sort(key=lambda t: t.date)

    if not all_transactions:
        return FinancialProfile(statements=statements)

    period_start = all_transactions[0].date
    period_end = all_transactions[-1].date

    monthly = _compute_monthly_summaries(all_transactions)

    return FinancialProfile(
        statements=statements,
        all_transactions=all_transactions,
        period_start=period_start,
        period_end=period_end,
        monthly_summaries=monthly,
    )


def _compute_monthly_summaries(transactions: list[Transaction]) -> dict[str, MonthlySummary]:
    summaries: dict[str, MonthlySummary] = {}

    for txn in transactions:
        key = f"{txn.date.year}-{txn.date.month:02d}"
        if key not in summaries:
            summaries[key] = MonthlySummary(year=txn.date.year, month=txn.date.month)

        if txn.amount > 0:
            summaries[key].income += txn.amount
        else:
            summaries[key].expenses += txn.amount

    # Calcula saldo acumulado cronologicamente
    running_balance = Decimal("0")
    for key in sorted(summaries):
        ms = summaries[key]
        running_balance += ms.income + ms.expenses
        ms.balance = running_balance

    return summaries


def merge_profiles(profiles: list[FinancialProfile]) -> FinancialProfile:
    """Une múltiplos FinancialProfiles (caso de processamento parcial)."""
    all_statements = []
    for p in profiles:
        all_statements.extend(p.statements)
    return build_profile(all_statements)
