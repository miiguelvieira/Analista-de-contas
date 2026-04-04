"""Composição das abas do dashboard Streamlit."""
from __future__ import annotations

from decimal import Decimal

import pandas as pd
import streamlit as st

from leitor.models.profile import FinancialProfile
from leitor.models.transaction import Transaction
from leitor.scoring.credit_score import PILLAR_LABELS, get_band
from leitor.utils.currency import format_brl
from leitor.visualization import charts


def _kpi(label: str, value: str, delta: str | None = None, color: str = "normal"):
    st.metric(label=label, value=value, delta=delta)


# ─── ABA 1: VISÃO GERAL ──────────────────────────────────────────────────────

def tab_overview(profile: FinancialProfile):
    st.subheader("Visão Geral")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Renda Total", format_brl(profile.total_income))
    c2.metric("Gastos Totais", format_brl(abs(profile.total_expenses)))
    c3.metric("Saldo Líquido", format_brl(profile.net),
              delta_color="normal" if profile.net >= 0 else "inverse")
    savings_pct = (
        float(profile.net / profile.total_income * 100) if profile.total_income else 0
    )
    c4.metric("Taxa de Poupança", f"{savings_pct:.1f}%")

    st.divider()

    granularity = st.radio(
        "Granularidade do gráfico",
        options=["day", "week", "month", "year"],
        format_func=lambda x: {"day": "Dia", "week": "Semana", "month": "Mês", "year": "Ano"}[x],
        horizontal=True,
        key="overview_granularity",
    )

    st.plotly_chart(
        charts.cashflow_chart(profile.all_transactions, granularity),
        use_container_width=True,
    )

    st.plotly_chart(
        charts.income_vs_expense_chart(profile.all_transactions),
        use_container_width=True,
    )


# ─── ABA 2: GASTOS ───────────────────────────────────────────────────────────

def tab_expenses(profile: FinancialProfile, learner=None):
    st.subheader("Categorias de Gastos")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(charts.expense_donut_chart(profile.all_transactions), use_container_width=True)
    with col2:
        st.plotly_chart(charts.expense_sunburst_chart(profile.all_transactions), use_container_width=True)

    st.plotly_chart(charts.monthly_category_chart(profile.all_transactions), use_container_width=True)

    st.divider()
    st.subheader("Transações")

    _transactions_table(profile, learner)

    if learner:
        st.divider()
        _user_rules_editor(learner, profile)


def _transactions_table(profile: FinancialProfile, learner=None):
    txns = profile.all_transactions
    if not txns:
        st.info("Nenhuma transação encontrada.")
        return

    all_categories = [
        "alimentacao", "transporte", "saude", "lazer", "moradia",
        "educacao", "salario", "investimentos", "impostos",
        "transferencias", "outros",
    ]

    search = st.text_input("Buscar transação", placeholder="Nome, banco, categoria...")

    rows = [
        {
            "ID": t.id,
            "Data": t.date.strftime("%d/%m/%Y"),
            "Banco": t.bank.capitalize(),
            "Descrição": t.description,
            "Valor (R$)": float(t.amount),
            "Categoria": t.category,
            "Fonte": t.categorization_source,
        }
        for t in txns
    ]
    df = pd.DataFrame(rows)

    if search:
        mask = df.apply(lambda row: search.lower() in str(row).lower(), axis=1)
        df = df[mask]

    st.dataframe(
        df.drop(columns=["ID"]),
        use_container_width=True,
        height=350,
    )

    if learner:
        with st.expander("✏️ Corrigir categorização de uma transação"):
            selected_desc = st.selectbox(
                "Escolha a transação",
                options=[t.description[:60] for t in txns],
                key="correct_desc",
            )
            selected_txn = next((t for t in txns if t.description[:60] == selected_desc), None)
            if selected_txn:
                new_cat = st.selectbox(
                    "Nova categoria",
                    options=all_categories,
                    index=all_categories.index(selected_txn.category)
                    if selected_txn.category in all_categories else 0,
                    key="new_cat",
                )
                new_sub = st.text_input("Subcategoria (opcional)", value=selected_txn.subcategory, key="new_sub")

                rule_type = st.radio(
                    "Aplicar regra a:",
                    options=["so_essa", "contiver", "exact_id"],
                    format_func=lambda x: {
                        "so_essa": "Só esta transação",
                        "contiver": f'Sempre que descrição contiver "{selected_txn.description.split()[0]}"',
                        "exact_id": "Só este ID específico",
                    }[x],
                    key="rule_type",
                )

                if st.button("Salvar regra e recategorizar", key="save_rule"):
                    if rule_type == "contiver":
                        pattern = selected_txn.description.split()[0].lower()
                        rule = learner.add_rule_contains(pattern, new_cat, new_sub)
                        affected = learner.recategorize_after_rule(profile.all_transactions, rule)
                        st.success(f"Regra criada! {len(affected)} transação(ões) recategorizadas.")
                    else:
                        learner.add_rule_exact_id(selected_txn.id, new_cat, new_sub)
                        selected_txn.category = new_cat
                        selected_txn.subcategory = new_sub
                        selected_txn.categorization_source = "manual"
                        st.success("Categorização salva para esta transação.")
                    st.rerun()


def _user_rules_editor(learner, profile: FinancialProfile):
    st.subheader("Minhas Regras Aprendidas")
    rules = learner.rules
    if not rules:
        st.info("Nenhuma regra personalizada ainda. Corrija uma categorização para criar regras.")
        return

    for i, rule in enumerate(rules):
        col1, col2, col3 = st.columns([4, 2, 1])
        mtype = rule.get("match_type", "")
        if mtype == "contains":
            col1.markdown(f'**Contém** `{rule["pattern"]}` → **{rule["category"]}** / {rule.get("subcategory", "geral")}')
        elif mtype == "exact_id":
            col1.markdown(f'**ID** `{rule["transaction_id"][:12]}...` → **{rule["category"]}**')
        col2.markdown(f'<small>Fonte: {mtype}</small>', unsafe_allow_html=True)
        if col3.button("🗑️", key=f"del_rule_{i}"):
            learner.remove_rule(i)
            # Recategoriza as transações afetadas removendo a regra
            st.success("Regra removida.")
            st.rerun()


# ─── ABA 3: SCORE & EMPRÉSTIMO ───────────────────────────────────────────────

def tab_score(profile: FinancialProfile):
    score = profile.credit_score or 0
    pillars = profile.score_pillars
    loan = profile.loan_details

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(charts.credit_score_gauge(score), use_container_width=True)
        band_label, band_color = get_band(score)
        st.markdown(
            f"<h3 style='color:{band_color};text-align:center'>{band_label}</h3>",
            unsafe_allow_html=True,
        )

    with col2:
        if pillars:
            st.plotly_chart(charts.score_radar_chart(pillars), use_container_width=True)

    if pillars:
        st.subheader("Detalhamento dos Pilares")
        for key, value in pillars.items():
            label = PILLAR_LABELS.get(key, key)
            color = "#27ae60" if value >= 70 else "#f39c12" if value >= 40 else "#e74c3c"
            st.markdown(
                f"**{label}**: "
                f"<span style='color:{color}'>{value:.1f}/100</span>",
                unsafe_allow_html=True,
            )
            st.progress(int(value) / 100)

    st.divider()
    st.subheader("Calculadora de Empréstimo")

    if not loan:
        st.info("Calcule o score primeiro para ver a estimativa de empréstimo.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Empréstimo Máximo", loan.get("max_loan_fmt", "—"))
    c2.metric("Parcela Estimada", loan.get("monthly_payment_fmt", "—"),
              help=f'Prazo de {loan.get("suggested_term_months", "—")} meses')
    c3.metric("Renda Média Mensal", format_brl(loan.get("avg_monthly_income", Decimal("0"))))

    st.subheader("Ajuste sua simulação")
    col_rate, col_term = st.columns(2)
    custom_rate = col_rate.slider(
        "Taxa de juros anual (%)", min_value=6.0, max_value=60.0,
        value=float(loan.get("annual_rate_used", 0.2388) * 100),
        step=0.5, key="loan_rate",
    )
    custom_term = col_term.select_slider(
        "Prazo (meses)", options=[12, 24, 36, 48, 60],
        value=loan.get("suggested_term_months", 36), key="loan_term",
    )

    from leitor.scoring.loan_calculator import _pmt
    custom_monthly_rate = (custom_rate / 100) / 12
    max_loan_val = float(loan.get("max_loan", Decimal("0")))
    if max_loan_val > 0:
        custom_pmt = _pmt(max_loan_val, custom_monthly_rate, custom_term)
        st.info(
            f"Para empréstimo de **{format_brl(loan['max_loan'])}** "
            f"em **{custom_term} meses** a **{custom_rate:.1f}% a.a.**: "
            f"parcela estimada de **{format_brl(Decimal(str(round(custom_pmt, 2))))}**"
        )

    # Tabela de sensibilidade
    if loan.get("sensitivity"):
        st.subheader("Tabela de Sensibilidade — Parcela (R$)")
        terms = loan["terms"]
        rows_data = {}
        for row in loan["sensitivity"]:
            rows_data[row["taxa_anual"]] = [row["parcelas"].get(str(t), 0) for t in terms]
        sens_df = pd.DataFrame(rows_data, index=terms).T
        sens_df.index.name = "Taxa anual"
        sens_df.columns = [f"{t}m" for t in terms]
        st.dataframe(sens_df.style.format("R$ {:.2f}"), use_container_width=True)


# ─── ABA 4: TENDÊNCIAS ───────────────────────────────────────────────────────

def tab_trends(profile: FinancialProfile):
    st.subheader("Evolução Patrimonial")

    projection_months = st.slider(
        "Meses de projeção", min_value=1, max_value=24, value=6, key="proj_months"
    )
    st.plotly_chart(
        charts.patrimony_evolution_chart(profile, projection_months),
        use_container_width=True,
    )

    st.subheader("Taxa de Poupança por Mês")
    st.plotly_chart(charts.savings_heatmap(profile), use_container_width=True)


# ─── ABA 5: BANCOS ───────────────────────────────────────────────────────────

def tab_banks(profile: FinancialProfile):
    st.subheader("Contribuição por Banco")
    st.plotly_chart(
        charts.bank_contribution_chart(profile.all_transactions),
        use_container_width=True,
    )

    st.subheader("Resumo por Banco")
    bank_summary = {}
    for txn in profile.all_transactions:
        b = txn.bank
        if b not in bank_summary:
            bank_summary[b] = {"transações": 0, "entradas": 0.0, "saídas": 0.0}
        bank_summary[b]["transações"] += 1
        if txn.amount > 0:
            bank_summary[b]["entradas"] += float(txn.amount)
        else:
            bank_summary[b]["saídas"] += abs(float(txn.amount))

    rows = [
        {
            "Banco": b.capitalize(),
            "Transações": v["transações"],
            "Entradas": format_brl(Decimal(str(round(v["entradas"], 2)))),
            "Saídas": format_brl(Decimal(str(round(v["saídas"], 2)))),
        }
        for b, v in bank_summary.items()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Extrato bruto por banco
    st.subheader("Extrato Bruto por Banco")
    selected_bank = st.selectbox("Selecione o banco", options=list(bank_summary.keys()),
                                  format_func=str.capitalize, key="raw_bank")
    bank_txns = [t for t in profile.all_transactions if t.bank == selected_bank]
    rows_raw = [
        {
            "Data": t.date.strftime("%d/%m/%Y"),
            "Descrição": t.description,
            "Valor": format_brl(t.amount),
            "Categoria": t.category,
        }
        for t in bank_txns
    ]
    st.dataframe(pd.DataFrame(rows_raw), use_container_width=True, height=400)
