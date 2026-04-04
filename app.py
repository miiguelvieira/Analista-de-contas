"""Leitor de Conta — Streamlit App Entry Point."""
import tempfile
from pathlib import Path

import streamlit as st

from leitor.aggregator import build_profile
from leitor.categorizer import Categorizer
from leitor.learner import Learner
from leitor.models.statement import Statement
from leitor.normalizer import normalize_all
from leitor.parsers.pdf_parser import parse_file, detect_bank
from leitor.scoring.credit_score import compute_score
from leitor.scoring.loan_calculator import compute_loan
from leitor.visualization import dashboard

st.set_page_config(
    page_title="Leitor de Conta",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS custom ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 0.5rem; }
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 10px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏦 Leitor de Conta")
st.caption("Análise inteligente de extratos bancários · Score de Crédito · Categorização com IA")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Carregar Extratos")

    uploaded_files = st.file_uploader(
        "Selecione os arquivos de extrato",
        type=["csv", "pdf", "ofx", "qif", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Aceita CSV (Nubank, Itaú, BB...), PDF (Bradesco, Santander...) e OFX",
    )

    st.subheader("⚙️ Configurações")
    use_claude = st.toggle(
        "Usar Claude IA para categorização",
        value=True,
        help="Requer ANTHROPIC_API_KEY no arquivo .env",
    )

    debug_mode = st.toggle("🔍 Modo debug (mostra texto extraído)", value=False)

    if not uploaded_files:
        st.info("Faça upload de pelo menos um extrato bancário para começar.")

# ─── Estado da sessão ─────────────────────────────────────────────────────────
if "profile" not in st.session_state:
    st.session_state.profile = None
if "learner" not in st.session_state:
    st.session_state.learner = Learner()

# ─── Processamento dos arquivos ───────────────────────────────────────────────
if uploaded_files:
    process_key = str(sorted([f.name + str(f.size) for f in uploaded_files]))
    if st.session_state.get("process_key") != process_key:
        with st.spinner("Processando extratos..."):
            statements: list[Statement] = []
            errors = []

            for uploaded in uploaded_files:
                try:
                    suffix = Path(uploaded.name).suffix
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = Path(tmp.name)

                    bank = detect_bank(tmp_path)

                    # Debug: mostra texto extraído pelo pdfplumber antes de parsear
                    if debug_mode and tmp_path.suffix.lower() == ".pdf":
                        try:
                            import pdfplumber
                            with pdfplumber.open(tmp_path) as pdf:
                                sample_text = ""
                                for pg in pdf.pages[:2]:
                                    sample_text += pg.extract_text() or ""
                                    sample_text += "\n---página---\n"
                            st.info(f"**{uploaded.name}** — texto extraído (primeiros 1500 chars):\n```\n{sample_text[:1500]}\n```")
                        except Exception as e:
                            st.warning(f"Não foi possível ler o PDF para debug: {e}")

                    raws = parse_file(tmp_path, bank_hint=bank)

                    if not raws:
                        # Tenta mostrar prévia do arquivo para diagnóstico
                        try:
                            preview = tmp_path.read_text(encoding="utf-8", errors="ignore")[:300]
                        except Exception:
                            preview = "(não foi possível ler o arquivo)"
                        errors.append(
                            f"⚠️ **{uploaded.name}** — banco detectado: `{bank}` — "
                            f"nenhuma transação extraída.\n\n"
                            f"Primeiras linhas do arquivo:\n```\n{preview}\n```"
                        )
                        continue

                    transactions = normalize_all(raws)
                    if not transactions:
                        errors.append(f"⚠️ {uploaded.name}: transações encontradas ({len(raws)}) mas datas/valores inválidos.")
                        continue

                    # Categorização
                    categorizer = Categorizer()
                    learner: Learner = st.session_state.learner
                    categorizer.set_user_rules(learner.rules)

                    if use_claude:
                        transactions = categorizer.categorize_all(transactions)
                    else:
                        for txn in transactions:
                            cat, sub, src = categorizer.categorize_one(txn.description)
                            txn.category = cat
                            txn.subcategory = sub
                            txn.categorization_source = src

                    # Aplica regras do usuário por cima
                    transactions = learner.apply_to_all(transactions)

                    stmt = Statement(
                        bank=bank,
                        transactions=transactions,
                        source_files=[uploaded.name],
                        period_start=transactions[0].date if transactions else None,
                        period_end=transactions[-1].date if transactions else None,
                    )
                    statements.append(stmt)

                except Exception as e:
                    errors.append(f"❌ {uploaded.name}: {e}")

            if errors:
                for err in errors:
                    st.warning(err, icon="⚠️")

            if statements:
                profile = build_profile(statements)
                score, pillars = compute_score(profile)
                profile.credit_score = score
                profile.score_pillars = pillars
                profile.loan_details = compute_loan(profile, score)
                st.session_state.profile = profile
                st.session_state.process_key = process_key
            elif not errors:
                st.error("Nenhum extrato pôde ser processado.")

# ─── Dashboard ────────────────────────────────────────────────────────────────
profile = st.session_state.profile
learner = st.session_state.learner

if profile and profile.all_transactions:
    # Filtro de datas na sidebar
    with st.sidebar:
        st.subheader("📅 Filtro de Período")
        min_date = profile.period_start
        max_date = profile.period_end
        if min_date and max_date:
            date_range = st.date_input(
                "Período",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="date_filter",
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_filter, end_filter = date_range
                filtered_txns = [
                    t for t in profile.all_transactions
                    if start_filter <= t.date <= end_filter
                ]
                if len(filtered_txns) != len(profile.all_transactions):
                    from leitor.models.statement import Statement as Stmt
                    from leitor.aggregator import build_profile as bp
                    filtered_stmt = Stmt(
                        bank="filtered",
                        transactions=filtered_txns,
                    )
                    profile = bp([filtered_stmt])
                    score, pillars = compute_score(profile)
                    profile.credit_score = score
                    profile.score_pillars = pillars
                    profile.loan_details = compute_loan(profile, score)

        st.divider()
        st.subheader("📊 Resumo")
        st.markdown(f"**Bancos:** {', '.join(s.bank.capitalize() for s in profile.statements)}")
        st.markdown(f"**Transações:** {len(profile.all_transactions)}")
        if profile.period_start and profile.period_end:
            st.markdown(
                f"**Período:** {profile.period_start.strftime('%d/%m/%Y')} "
                f"→ {profile.period_end.strftime('%d/%m/%Y')}"
            )

    # Abas do dashboard
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Visão Geral",
        "💸 Gastos",
        "🏅 Score & Empréstimo",
        "📉 Tendências",
        "🏦 Bancos",
    ])

    with tab1:
        dashboard.tab_overview(profile)

    with tab2:
        dashboard.tab_expenses(profile, learner)

    with tab3:
        dashboard.tab_score(profile)

    with tab4:
        dashboard.tab_trends(profile)

    with tab5:
        dashboard.tab_banks(profile)

else:
    # Tela de boas-vindas
    st.markdown("""
    ## Como usar

    1. **Faça upload** dos seus extratos bancários na barra lateral
       - CSV do Nubank, Itaú, BB, Bradesco, Santander, Caixa...
       - PDF de qualquer banco brasileiro
       - Arquivos OFX/QIF

    2. **Múltiplos bancos** são suportados simultaneamente — o sistema consolida tudo em um perfil único

    3. **Analise** seus dados:
       - 📈 Evolução do seu patrimônio ao longo do tempo
       - 💸 Categorização inteligente dos seus gastos
       - 🏅 Score de crédito baseado no seu comportamento financeiro
       - 💰 Estimativa de valor de empréstimo que você conseguiria
       - 📉 Projeção de tendências futuras

    4. **Ensine o agente**: corrija categorizações incorretas e o sistema aprende para o futuro

    ---
    > 💡 **Dica**: para melhores resultados, carregue extratos de pelo menos 3 meses.
    """)

    col1, col2, col3 = st.columns(3)
    col1.info("🔒 Seus dados ficam 100% locais — nenhum dado é enviado a servidores externos (exceto descrições anônimas de transações para categorização via Claude API, se ativado)")
    col2.info("🤖 A categorização com Claude API usa apenas as descrições das transações, sem valores ou dados pessoais")
    col3.info("📁 Formatos aceitos: CSV, PDF, OFX, QIF, XLSX")
