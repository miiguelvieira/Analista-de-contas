"""Calculadora de limite de empréstimo baseado em score e fluxo de caixa."""
from __future__ import annotations

from decimal import Decimal

import numpy as np

from leitor.models.profile import FinancialProfile
from leitor.utils.currency import format_brl

COMMITMENT_RATIO = 0.30  # máx 30% da renda livre para parcelas
DEFAULT_MAX_TERM_MONTHS = 60
DEFAULT_ANNUAL_RATE = 0.0199  # 1.99% a.m. (padrão mercado PF 2026)


def _pmt(principal: float, monthly_rate: float, n_months: int) -> float:
    """Calcula parcela (PMT) de financiamento."""
    if monthly_rate == 0:
        return principal / n_months
    r = monthly_rate
    return principal * (r * (1 + r) ** n_months) / ((1 + r) ** n_months - 1)


def compute_loan(
    profile: FinancialProfile,
    score: int,
    annual_rate: float = DEFAULT_ANNUAL_RATE * 12,
    max_term_months: int = DEFAULT_MAX_TERM_MONTHS,
) -> dict:
    """
    Calcula o empréstimo máximo recomendado.

    Retorna dict com:
        max_loan: Decimal
        suggested_term_months: int
        monthly_payment: float
        avg_monthly_income: Decimal
        avg_monthly_expenses: Decimal
        monthly_free_cash: Decimal
        score_multiplier: float
        sensitivity: list[dict]  # [{term, rate, payment, max_loan}]
    """
    num_months = max(len(profile.monthly_summaries), 1)
    avg_income = profile.total_income / num_months
    avg_expenses = abs(profile.total_expenses) / num_months
    free_cash = avg_income - avg_expenses

    monthly_rate = annual_rate / 12

    # Passo 1: capacidade de pagamento
    max_monthly_payment = free_cash * Decimal(str(COMMITMENT_RATIO))
    if monthly_rate > 0 and max_monthly_payment > 0:
        # Resolve PMT para principal: PV = PMT * [(1-(1+r)^-n)/r]
        r = monthly_rate
        n = max_term_months
        factor = (1 - (1 + r) ** (-n)) / r
        capacity_loan = float(max_monthly_payment) * factor
    else:
        capacity_loan = float(max_monthly_payment) * max_term_months

    # Passo 2: score-based multiplier
    score_multiplier = (score / 1000) * 8
    score_loan = float(avg_income) * score_multiplier

    # Passo 3: conservador + desconto de risco
    raw_max = min(capacity_loan, score_loan)
    risk_discount = 1 - (1 - score / 1000) * 0.5
    recommended = raw_max * risk_discount

    # Prazo sugerido baseado no score
    if score >= 800:
        suggested_term = 60
    elif score >= 600:
        suggested_term = 48
    elif score >= 400:
        suggested_term = 36
    else:
        suggested_term = 24

    monthly_payment = _pmt(recommended, monthly_rate, suggested_term) if recommended > 0 else 0

    # Tabela de sensibilidade: taxa × prazo
    rates_annual = [0.12, 0.18, 0.24, 0.30, 0.36]
    terms = [12, 24, 36, 48, 60]
    sensitivity = []
    for r_annual in rates_annual:
        row = {"taxa_anual": f"{r_annual*100:.0f}%", "parcelas": {}}
        for t in terms:
            r_m = r_annual / 12
            payment = _pmt(recommended, r_m, t) if recommended > 0 else 0
            row["parcelas"][str(t)] = round(payment, 2)
        sensitivity.append(row)

    return {
        "max_loan": Decimal(str(round(recommended, 2))),
        "max_loan_fmt": format_brl(Decimal(str(round(recommended, 2)))),
        "suggested_term_months": suggested_term,
        "monthly_payment": round(monthly_payment, 2),
        "monthly_payment_fmt": format_brl(Decimal(str(round(monthly_payment, 2)))),
        "avg_monthly_income": avg_income,
        "avg_monthly_expenses": avg_expenses,
        "monthly_free_cash": free_cash,
        "score_multiplier": score_multiplier,
        "annual_rate_used": annual_rate,
        "sensitivity": sensitivity,
        "terms": terms,
    }
