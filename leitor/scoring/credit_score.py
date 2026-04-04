"""Engine de score de crédito (0–1000) baseado em 6 pilares."""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

import numpy as np

from leitor.models.profile import FinancialProfile, MonthlySummary


SCORE_WEIGHTS = {
    "income_regularity": 0.25,
    "expense_control": 0.20,
    "savings_rate": 0.20,
    "debt_ratio": 0.15,
    "payment_consistency": 0.10,
    "balance_stability": 0.10,
}

SCORE_BANDS = [
    (851, 1000, "Muito Baixo Risco", "#27ae60"),
    (701, 850, "Baixo Risco", "#2ecc71"),
    (501, 700, "Risco Médio", "#f39c12"),
    (301, 500, "Alto Risco", "#e67e22"),
    (0, 300, "Muito Alto Risco", "#e74c3c"),
]


def get_band(score: int) -> tuple[str, str]:
    """Retorna (label, cor_hex) para o score."""
    for low, high, label, color in SCORE_BANDS:
        if low <= score <= high:
            return label, color
    return "Sem dados", "#95a5a6"


def _pillar_income_regularity(monthly: dict[str, MonthlySummary]) -> float:
    """P1: Regularidade de renda (0–100). CV baixo = score alto."""
    incomes = [float(ms.income) for ms in monthly.values() if ms.income > 0]
    if len(incomes) < 2:
        return 50.0  # dados insuficientes: score neutro
    mean = np.mean(incomes)
    if mean == 0:
        return 0.0
    cv = np.std(incomes) / mean
    return float(max(0.0, 100.0 - cv * 200.0))


def _pillar_expense_control(monthly: dict[str, MonthlySummary]) -> float:
    """P2: Controle de gastos (0–100). Gasto/renda baixo = score alto."""
    ratios = []
    for ms in monthly.values():
        if ms.income > 0:
            ratio = float(abs(ms.expenses) / ms.income)
            ratios.append(ratio)
    if not ratios:
        return 50.0
    avg_ratio = np.mean(ratios)
    return float(max(0.0, 100.0 - avg_ratio * 100.0))


def _pillar_savings_rate(profile: FinancialProfile) -> float:
    """P3: Taxa de poupança global (0–100). 25% poupança = 100pts."""
    if profile.total_income == 0:
        return 0.0
    rate = float(profile.net / profile.total_income)
    return float(min(100.0, max(0.0, rate * 400.0)))


def _pillar_debt_ratio(profile: FinancialProfile) -> float:
    """P4: Índice de dívida (0–100). Detecta pagamentos fixos recorrentes."""
    if not profile.all_transactions:
        return 50.0

    # Identifica débitos recorrentes: mesmo nome base em ≥2 meses
    monthly_by_desc: dict[str, set[str]] = defaultdict(set)
    monthly_amount: dict[str, list[Decimal]] = defaultdict(list)

    for txn in profile.all_transactions:
        if txn.amount < 0:
            key = f"{txn.date.year}-{txn.date.month:02d}"
            desc_key = txn.description[:20].strip().lower()
            monthly_by_desc[desc_key].add(key)
            monthly_amount[desc_key].append(abs(txn.amount))

    # Considera "dívida" se aparece em ≥2 meses diferentes
    total_monthly_debt = Decimal("0")
    num_months = max(len(profile.monthly_summaries), 1)
    for desc_key, months in monthly_by_desc.items():
        if len(months) >= 2:
            avg_amount = sum(monthly_amount[desc_key]) / len(monthly_amount[desc_key])
            total_monthly_debt += avg_amount

    avg_monthly_income = profile.total_income / num_months
    if avg_monthly_income == 0:
        return 0.0

    debt_ratio = float(total_monthly_debt / avg_monthly_income)
    return float(max(0.0, 100.0 - debt_ratio * 200.0))


def _pillar_payment_consistency(profile: FinancialProfile) -> float:
    """P5: Consistência de pagamentos recorrentes (0–100)."""
    if not profile.monthly_summaries:
        return 50.0

    num_months = len(profile.monthly_summaries)
    if num_months < 2:
        return 70.0

    # Conta descritores que aparecem em múltiplos meses
    monthly_by_desc: dict[str, set[str]] = defaultdict(set)
    for txn in profile.all_transactions:
        if txn.amount < 0:
            key = f"{txn.date.year}-{txn.date.month:02d}"
            desc_key = txn.description[:20].strip().lower()
            monthly_by_desc[desc_key].add(key)

    recurring = {d: months for d, months in monthly_by_desc.items() if len(months) >= 2}
    if not recurring:
        return 70.0

    consistencies = []
    for months in recurring.values():
        consistency = len(months) / num_months
        consistencies.append(min(1.0, consistency))

    return float(np.mean(consistencies) * 100.0)


def _pillar_balance_stability(profile: FinancialProfile) -> float:
    """P6: Estabilidade de saldo (0–100). Nunca negativo = 100pts."""
    if not profile.monthly_summaries:
        return 50.0

    balances = [float(ms.balance) for ms in profile.monthly_summaries.values()]
    if not balances:
        return 50.0

    min_bal = min(balances)
    avg_bal = sum(balances) / len(balances)

    if avg_bal <= 0:
        return max(0.0, 50.0 + min_bal / abs(avg_bal) * 50.0) if avg_bal != 0 else 0.0

    ratio = min_bal / avg_bal
    return float(min(100.0, max(0.0, ratio * 100.0)))


def compute_score(profile: FinancialProfile) -> tuple[int, dict[str, float]]:
    """Calcula o score de crédito e retorna (score, {pilar: valor_0_100})."""
    monthly = profile.monthly_summaries

    pillars = {
        "income_regularity": _pillar_income_regularity(monthly),
        "expense_control": _pillar_expense_control(monthly),
        "savings_rate": _pillar_savings_rate(profile),
        "debt_ratio": _pillar_debt_ratio(profile),
        "payment_consistency": _pillar_payment_consistency(profile),
        "balance_stability": _pillar_balance_stability(profile),
    }

    weighted_sum = sum(
        pillars[p] * SCORE_WEIGHTS[p] for p in pillars
    )
    score = int(round(weighted_sum * 10.0))
    score = max(0, min(1000, score))

    return score, pillars


PILLAR_LABELS = {
    "income_regularity": "Regularidade de Renda",
    "expense_control": "Controle de Gastos",
    "savings_rate": "Taxa de Poupança",
    "debt_ratio": "Índice de Dívida",
    "payment_consistency": "Consistência de Pagamentos",
    "balance_stability": "Estabilidade de Saldo",
}
