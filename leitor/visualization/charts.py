"""Funções de criação de gráficos Plotly para o dashboard."""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from leitor.models.profile import FinancialProfile
from leitor.models.transaction import Transaction
from leitor.scoring.credit_score import SCORE_BANDS, PILLAR_LABELS

CATEGORY_COLORS = {
    "alimentacao": "#e74c3c",
    "transporte": "#3498db",
    "saude": "#2ecc71",
    "lazer": "#9b59b6",
    "moradia": "#f39c12",
    "educacao": "#1abc9c",
    "salario": "#27ae60",
    "investimentos": "#2980b9",
    "impostos": "#7f8c8d",
    "transferencias": "#bdc3c7",
    "outros": "#95a5a6",
}

BANK_COLORS = {
    "nubank": "#8A05BE",
    "bradesco": "#CC0000",
    "itau": "#EC7000",
    "bb": "#FFD700",
    "santander": "#EC0000",
    "caixa": "#0066CC",
    "inter": "#FF7A00",
    "generic": "#95a5a6",
}


def _txns_to_df(transactions: list[Transaction]) -> pd.DataFrame:
    rows = [
        {
            "date": t.date,
            "description": t.description,
            "amount": float(t.amount),
            "category": t.category,
            "subcategory": t.subcategory,
            "bank": t.bank,
            "type": t.transaction_type,
        }
        for t in transactions
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df


# ─── GRÁFICO 1: Fluxo de caixa (barras + linha de saldo) ─────────────────────

def cashflow_chart(
    transactions: list[Transaction],
    granularity: str = "month",  # "day" | "week" | "month" | "year"
) -> go.Figure:
    df = _txns_to_df(transactions)
    if df.empty:
        return go.Figure().update_layout(title="Sem dados")

    freq_map = {"day": "D", "week": "W", "month": "ME", "year": "YE"}
    label_map = {"day": "Dia", "week": "Semana", "month": "Mês", "year": "Ano"}
    freq = freq_map.get(granularity, "ME")

    income = df[df["amount"] > 0].set_index("date")["amount"].resample(freq).sum().fillna(0)
    expense = df[df["amount"] < 0].set_index("date")["amount"].resample(freq).sum().fillna(0)
    balance = (income + expense).cumsum()

    all_dates = income.index.union(expense.index).union(balance.index)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=income.index, y=income.values,
        name="Entradas", marker_color="#27ae60", opacity=0.8,
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        x=expense.index, y=abs(expense.values),
        name="Saídas", marker_color="#e74c3c", opacity=0.8,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=balance.index, y=balance.values,
        name="Saldo acumulado", line=dict(color="#3498db", width=2),
        mode="lines+markers",
    ), secondary_y=True)

    fig.update_layout(
        title=f"Fluxo de Caixa por {label_map[granularity]}",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(title_text="Valor (R$)", secondary_y=False)
    fig.update_yaxes(title_text="Saldo acumulado (R$)", secondary_y=True)
    return fig


# ─── GRÁFICO 2: Renda vs Gastos (barras mensais) ─────────────────────────────

def income_vs_expense_chart(transactions: list[Transaction]) -> go.Figure:
    df = _txns_to_df(transactions)
    if df.empty:
        return go.Figure().update_layout(title="Sem dados")

    monthly_income = df[df["amount"] > 0].set_index("date")["amount"].resample("ME").sum().fillna(0)
    monthly_expense = df[df["amount"] < 0].set_index("date")["amount"].resample("ME").sum().fillna(0)
    labels = [d.strftime("%b/%Y") for d in monthly_income.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=monthly_income.values, name="Renda", marker_color="#27ae60"))
    fig.add_trace(go.Bar(x=labels, y=abs(monthly_expense.values), name="Gastos", marker_color="#e74c3c"))

    fig.update_layout(
        title="Renda vs Gastos Mensais",
        barmode="group",
        xaxis_title="Mês",
        yaxis_title="Valor (R$)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─── GRÁFICO 3: Donut por categoria ──────────────────────────────────────────

def expense_donut_chart(transactions: list[Transaction]) -> go.Figure:
    df = _txns_to_df(transactions)
    expenses = df[df["amount"] < 0].copy()
    if expenses.empty:
        return go.Figure().update_layout(title="Sem gastos")

    by_cat = expenses.groupby("category")["amount"].sum().abs().sort_values(ascending=False)

    colors = [CATEGORY_COLORS.get(c, "#95a5a6") for c in by_cat.index]
    fig = go.Figure(go.Pie(
        labels=by_cat.index,
        values=by_cat.values,
        marker=dict(colors=colors),
        hole=0.45,
        textinfo="label+percent",
    ))
    fig.update_layout(
        title="Gastos por Categoria",
        showlegend=True,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─── GRÁFICO 4: Sunburst categoria → subcategoria → descrição ────────────────

def expense_sunburst_chart(transactions: list[Transaction]) -> go.Figure:
    df = _txns_to_df(transactions)
    expenses = df[df["amount"] < 0].copy()
    if expenses.empty:
        return go.Figure().update_layout(title="Sem gastos")

    expenses["abs_amount"] = expenses["amount"].abs()
    # Trunca descrição para legibilidade
    expenses["short_desc"] = expenses["description"].str[:30]

    fig = px.sunburst(
        expenses,
        path=["category", "subcategory", "short_desc"],
        values="abs_amount",
        color="category",
        color_discrete_map=CATEGORY_COLORS,
        title="Detalhamento de Gastos",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─── GRÁFICO 5: Gauge de score de crédito ────────────────────────────────────

def credit_score_gauge(score: int) -> go.Figure:
    band_label, band_color = _get_band(score)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title={"text": f"Score de Crédito<br><span style='font-size:0.8em;color:{band_color}'>{band_label}</span>"},
        gauge={
            "axis": {"range": [0, 1000], "tickwidth": 1},
            "bar": {"color": band_color},
            "steps": [
                {"range": [0, 300], "color": "#fadbd8"},
                {"range": [300, 500], "color": "#fdebd0"},
                {"range": [500, 700], "color": "#fef9e7"},
                {"range": [700, 850], "color": "#eafaf1"},
                {"range": [850, 1000], "color": "#d5f5e3"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": score,
            },
        },
        number={"suffix": "/1000"},
    ))
    fig.update_layout(
        height=300,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _get_band(score: int) -> tuple[str, str]:
    for low, high, label, color in SCORE_BANDS:
        if low <= score <= high:
            return label, color
    return "N/A", "#95a5a6"


# ─── GRÁFICO 6: Radar dos 6 pilares ──────────────────────────────────────────

def score_radar_chart(pillars: dict[str, float]) -> go.Figure:
    labels = [PILLAR_LABELS.get(k, k) for k in pillars]
    values = list(pillars.values())
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(52, 152, 219, 0.3)",
        line=dict(color="#3498db"),
        name="Score",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Perfil de Crédito — 6 Pilares",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─── GRÁFICO 7: Evolução patrimonial + projeção ───────────────────────────────

def patrimony_evolution_chart(
    profile: FinancialProfile,
    projection_months: int = 6,
) -> go.Figure:
    if not profile.monthly_summaries:
        return go.Figure().update_layout(title="Sem dados")

    sorted_keys = sorted(profile.monthly_summaries)
    dates = [
        date(int(k[:4]), int(k[5:7]), 1)
        for k in sorted_keys
    ]
    balances = [float(profile.monthly_summaries[k].balance) for k in sorted_keys]

    if len(dates) >= 2:
        x_num = np.arange(len(dates))
        coeffs = np.polyfit(x_num, balances, 1)
        poly = np.poly1d(coeffs)

        # Projeção
        proj_x = np.arange(len(dates), len(dates) + projection_months)
        proj_y = poly(proj_x)
        last_date = dates[-1]
        proj_dates = [
            date(last_date.year + (last_date.month + i - 1) // 12,
                 (last_date.month + i - 1) % 12 + 1, 1)
            for i in range(1, projection_months + 1)
        ]

        # Intervalo de confiança simples (±1 std dos resíduos)
        residuals = np.array(balances) - poly(x_num)
        std = np.std(residuals)
        ci_upper = proj_y + 1.645 * std
        ci_lower = proj_y - 1.645 * std
    else:
        proj_dates = []
        proj_y = ci_upper = ci_lower = []

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=balances,
        name="Saldo Real",
        fill="tozeroy",
        fillcolor="rgba(39, 174, 96, 0.2)",
        line=dict(color="#27ae60", width=2),
    ))

    if len(proj_dates) > 0:
        fig.add_trace(go.Scatter(
            x=proj_dates + proj_dates[::-1],
            y=list(ci_upper) + list(ci_lower[::-1]),
            fill="toself",
            fillcolor="rgba(52, 152, 219, 0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="IC 90%",
            showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=proj_dates, y=proj_y,
            name=f"Projeção ({projection_months}m)",
            line=dict(color="#3498db", width=2, dash="dash"),
        ))

    fig.update_layout(
        title="Evolução Patrimonial e Projeção",
        xaxis_title="Mês",
        yaxis_title="Saldo Acumulado (R$)",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─── GRÁFICO 8: Heatmap de taxa de poupança ───────────────────────────────────

def savings_heatmap(profile: FinancialProfile) -> go.Figure:
    if not profile.monthly_summaries:
        return go.Figure().update_layout(title="Sem dados")

    rows = []
    for key, ms in profile.monthly_summaries.items():
        rows.append({
            "year": ms.year,
            "month": ms.month,
            "savings_rate": ms.savings_rate * 100,
        })

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="year", columns="month", values="savings_rate")

    month_labels = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                    "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    col_labels = [month_labels[m - 1] for m in pivot.columns]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=col_labels,
        y=[str(y) for y in pivot.index],
        colorscale="RdYlGn",
        zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        colorbar=dict(title="Taxa %"),
    ))
    fig.update_layout(
        title="Taxa de Poupança Mensal (%)",
        xaxis_title="Mês",
        yaxis_title="Ano",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─── GRÁFICO 9: Contribuição por banco ────────────────────────────────────────

def bank_contribution_chart(transactions: list[Transaction]) -> go.Figure:
    df = _txns_to_df(transactions)
    if df.empty:
        return go.Figure().update_layout(title="Sem dados")

    df["month"] = df["date"].dt.to_period("M").astype(str)
    by_bank_month = df[df["amount"] < 0].groupby(["month", "bank"])["amount"].sum().abs().reset_index()

    banks = by_bank_month["bank"].unique()
    fig = go.Figure()
    for bank in banks:
        bank_data = by_bank_month[by_bank_month["bank"] == bank]
        fig.add_trace(go.Bar(
            x=bank_data["month"],
            y=bank_data["amount"],
            name=bank.capitalize(),
            marker_color=BANK_COLORS.get(bank, "#95a5a6"),
        ))

    fig.update_layout(
        title="Gastos por Banco",
        barmode="stack",
        xaxis_title="Mês",
        yaxis_title="Valor (R$)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─── GRÁFICO 10: Gastos mensais por categoria (barras empilhadas) ─────────────

def monthly_category_chart(transactions: list[Transaction]) -> go.Figure:
    df = _txns_to_df(transactions)
    if df.empty:
        return go.Figure().update_layout(title="Sem dados")

    expenses = df[df["amount"] < 0].copy()
    expenses["month"] = expenses["date"].dt.to_period("M").astype(str)
    by_cat = expenses.groupby(["month", "category"])["amount"].sum().abs().reset_index()

    categories = by_cat["category"].unique()
    fig = go.Figure()
    for cat in categories:
        cat_data = by_cat[by_cat["category"] == cat]
        fig.add_trace(go.Bar(
            x=cat_data["month"],
            y=cat_data["amount"],
            name=cat.capitalize(),
            marker_color=CATEGORY_COLORS.get(cat, "#95a5a6"),
        ))

    fig.update_layout(
        title="Gastos Mensais por Categoria",
        barmode="stack",
        xaxis_title="Mês",
        yaxis_title="Valor (R$)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
