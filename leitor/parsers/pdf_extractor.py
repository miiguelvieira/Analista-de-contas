"""Extração robusta de transações de PDFs bancários brasileiros.

Estratégias em ordem de confiabilidade:
1. Tabelas formais (pdfplumber extract_tables)
2. Palavras com coordenadas X/Y → reconstrução de linhas
3. Texto puro com regex
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from leitor.parsers.base import TransactionRaw
from leitor.utils.currency import parse_brl
from leitor.utils.date_utils import parse_date

# Padrão de data DD/MM/YYYY ou DD/MM/YY
DATE_RE = re.compile(r"\b(\d{2}/\d{2}/\d{2,4})\b")
# Valor BRL com vírgula decimal obrigatória: 1.234,56 ou 1234,56 (com sinal e R$ opcionais)
AMOUNT_RE = re.compile(
    r"(?<![,\d])"
    r"([-+]?\s*R?\$?\s*\d{1,3}(?:\.\d{3})*,\d{2})"
    r"(?!\d)"
)
# Valor simples sem separador de milhar: 123,45 ou 1234,56
AMOUNT_SIMPLE_RE = re.compile(r"(?<![,\d])(\d{1,6},\d{2})(?!\d)")


def _norm(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(text))
    return nfkd.encode("ascii", "ignore").decode("ascii").lower().strip()


# ─── ESTRATÉGIA 1: Tabelas formais ───────────────────────────────────────────

def extract_from_tables(pages, bank: str, source: str) -> list[TransactionRaw]:
    results = []
    for page in pages:
        try:
            # Tenta diferentes configurações de extração de tabela
            for settings in [
                {},  # padrão
                {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
                {"vertical_strategy": "text", "horizontal_strategy": "text"},
                {"vertical_strategy": "explicit", "horizontal_strategy": "lines",
                 "explicit_vertical_lines": page.curves + page.edges},
            ]:
                try:
                    tables = page.extract_tables(settings) if settings else page.extract_tables()
                    for table in (tables or []):
                        rows = _parse_table(table, bank, source)
                        if rows:
                            results.extend(rows)
                    if results:
                        break
                except Exception:
                    continue
        except Exception:
            continue
    return results


def _parse_table(table: list[list], bank: str, source: str) -> list[TransactionRaw]:
    if not table or len(table) < 2:
        return []

    # Detecta índices das colunas por nome (primeira linha) ou por conteúdo
    date_idx = _find_col_idx_by_name(table[0], ["data", "dt", "dia"])
    desc_idx = _find_col_idx_by_name(table[0], ["hist", "descr", "lanc", "memo", "operacao", "benefic"])
    amount_idx = _find_col_idx_by_name(table[0], ["valor", "debito", "credito", "moviment"])
    credit_idx = _find_col_idx_by_name(table[0], ["credito", "entrada", "c"])
    debit_idx = _find_col_idx_by_name(table[0], ["debito", "saida", "d"])

    # Se não achou por nome, tenta por conteúdo das primeiras linhas de dados
    if date_idx is None:
        date_idx = _find_col_idx_by_data(table, "date")
    if amount_idx is None:
        amount_idx = _find_col_idx_by_data(table, "amount")
    if desc_idx is None:
        num_cols = max(len(r) for r in table if r)
        candidates = [i for i in range(num_cols) if i not in (date_idx, amount_idx, credit_idx, debit_idx)]
        desc_idx = candidates[0] if candidates else None

    if date_idx is None:
        return []

    results = []
    for row in table[1:]:
        if not row or all(not str(c or "").strip() for c in row):
            continue
        try:
            date_str = str(row[date_idx] or "").strip() if date_idx < len(row) else ""
            if not date_str or parse_date(date_str) is None:
                continue

            desc = str(row[desc_idx] or "").strip() if desc_idx is not None and desc_idx < len(row) else ""

            # Tenta valor único ou crédito/débito separados
            amount_str = ""
            amount_val = None

            if amount_idx is not None and amount_idx < len(row):
                amount_str = str(row[amount_idx] or "").strip()
                amount_val = parse_brl(amount_str) if amount_str else None

            if (amount_val is None or amount_val == 0) and credit_idx is not None and debit_idx is not None:
                c_str = str(row[credit_idx] or "").strip() if credit_idx < len(row) else ""
                d_str = str(row[debit_idx] or "").strip() if debit_idx < len(row) else ""
                c_val = parse_brl(c_str) if c_str else None
                d_val = parse_brl(d_str) if d_str else None
                if c_val and c_val != 0:
                    amount_val = abs(c_val)
                    amount_str = c_str
                elif d_val and d_val != 0:
                    amount_val = -abs(d_val)
                    amount_str = d_str

            if amount_val is None or amount_str == "":
                continue

            ttype = "credit" if amount_val > 0 else "debit"
            results.append(TransactionRaw(
                bank=bank, date_str=date_str, description=desc or "Sem descrição",
                amount_str=amount_str, transaction_type=ttype, source_file=source,
            ))
        except Exception:
            continue

    return results


def _find_col_idx_by_name(header_row: list, hints: list[str]) -> int | None:
    for i, cell in enumerate(header_row):
        n = _norm(str(cell or ""))
        if any(h in n for h in hints):
            return i
    return None


def _find_col_idx_by_data(table: list[list], col_type: str) -> int | None:
    num_cols = max((len(r) for r in table if r), default=0)
    scores = [0] * num_cols
    for row in table[1:6]:
        for i, cell in enumerate(row):
            if i >= num_cols:
                continue
            val = str(cell or "").strip()
            if col_type == "date" and DATE_RE.search(val):
                scores[i] += 1
            elif col_type == "amount" and (AMOUNT_RE.search(val) or AMOUNT_SIMPLE_RE.search(val)):
                scores[i] += 1
    best = max(range(num_cols), key=lambda i: scores[i]) if num_cols else None
    return best if best is not None and scores[best] > 0 else None


# ─── ESTRATÉGIA 2: Palavras com coordenadas X/Y ──────────────────────────────

@dataclass
class _Word:
    x0: float
    y: float
    text: str


def extract_from_words(pages, bank: str, source: str) -> list[TransactionRaw]:
    """Reconstrói linhas a partir das coordenadas de palavras no PDF."""
    results = []
    for page in pages:
        try:
            words = page.extract_words(x_tolerance=5, y_tolerance=3)
            if not words:
                continue
            w_objs = [_Word(w["x0"], round(w["top"], 1), w["text"]) for w in words]
            lines = _group_by_y(w_objs)
            results.extend(_parse_word_lines(lines, bank, source))
        except Exception:
            continue
    return results


def _group_by_y(words: list[_Word], tolerance: float = 4.0) -> list[str]:
    """Agrupa palavras na mesma linha (Y ± tolerance) e ordena por X."""
    if not words:
        return []
    words.sort(key=lambda w: (w.y, w.x0))
    lines: list[list[_Word]] = []
    current: list[_Word] = [words[0]]
    for w in words[1:]:
        if abs(w.y - current[0].y) <= tolerance:
            current.append(w)
        else:
            lines.append(current)
            current = [w]
    if current:
        lines.append(current)
    return [" ".join(w.text for w in sorted(line, key=lambda w: w.x0)) for line in lines]


def _parse_word_lines(lines: list[str], bank: str, source: str) -> list[TransactionRaw]:
    results = []
    for line in lines:
        txn = _parse_line(line, bank, source)
        if txn:
            results.append(txn)
    return results


# ─── ESTRATÉGIA 3: Regex sobre texto puro ────────────────────────────────────

def extract_from_text(pages, bank: str, source: str) -> list[TransactionRaw]:
    results = []
    for page in pages:
        try:
            text = page.extract_text() or ""
            for line in text.splitlines():
                txn = _parse_line(line.strip(), bank, source)
                if txn:
                    results.append(txn)
        except Exception:
            continue
    return results


# ─── Parser de linha individual ───────────────────────────────────────────────

def _parse_line(line: str, bank: str, source: str) -> TransactionRaw | None:
    """Tenta extrair (data, descrição, valor) de uma linha de texto."""
    if len(line) < 10:
        return None

    date_match = DATE_RE.search(line)
    if not date_match:
        return None

    date_str = date_match.group(1)
    if parse_date(date_str) is None:
        return None

    # Busca valores APENAS na parte da linha após a data
    date_end = date_match.end()
    rest = line[date_end:]

    amount_matches = list(AMOUNT_RE.finditer(rest))
    if not amount_matches:
        return None

    # O PRIMEIRO valor após a data é a transação; os seguintes são saldo acumulado
    first_match = amount_matches[0]
    amount_str = first_match.group(1).strip()
    amount_val = parse_brl(amount_str)

    # Se o primeiro é zero e há mais, tenta o próximo
    if amount_val == 0 and len(amount_matches) > 1:
        first_match = amount_matches[1]
        amount_str = first_match.group(1).strip()
        amount_val = parse_brl(amount_str)

    # Descrição = texto entre o fim da data e o início do primeiro valor
    desc = rest[:first_match.start()].strip(" -–|:")

    desc = re.sub(r"\s{2,}", " ", desc).strip()
    if not desc:
        desc = "Sem descrição"

    ttype = "credit" if amount_val > 0 else "debit"
    return TransactionRaw(
        bank=bank, date_str=date_str, description=desc,
        amount_str=amount_str, transaction_type=ttype, source_file=source,
    )


# ─── Função principal: tenta 4 estratégias em cascata ───────────────────────

def extract_pdf(path: Path, bank: str) -> list[TransactionRaw]:
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("Instale 'pdfplumber': pip install pdfplumber")

    source = str(path)
    results: list[TransactionRaw] = []
    try:
        with pdfplumber.open(path) as pdf:
            pages = pdf.pages

            # Estratégia 1: tabelas formais
            results = extract_from_tables(pages, bank, source)
            if len(results) >= 3:
                return results

            # Estratégia 2: palavras com coordenadas (bom para HTML→PDF como Nubank)
            results2 = extract_from_words(pages, bank, source)
            if len(results2) > len(results):
                results = results2
            if len(results) >= 3:
                return results

            # Estratégia 3: texto puro com regex
            results3 = extract_from_text(pages, bank, source)
            if len(results3) > len(results):
                results = results3
            if len(results) >= 3:
                return results

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Erro ao abrir PDF: {e}")

    # Estratégia 4: Agente Claude (fallback inteligente — funciona com qualquer layout)
    # Ativado quando as estratégias anteriores encontram < 3 transações
    try:
        from leitor.parsers.claude_pdf_parser import extract_with_claude
        claude_results = extract_with_claude(path, bank)
        if len(claude_results) > len(results):
            return claude_results
    except Exception:
        pass

    return results
