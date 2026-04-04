"""Agente Claude para extração inteligente de transações de PDFs bancários.

Estratégia:
1. Extrai texto bruto do PDF com pdfplumber (página por página)
2. Envia o texto para o Claude API com prompt especializado
3. Claude retorna JSON estruturado com as transações identificadas
4. Converte em TransactionRaw

Vantagem: funciona com qualquer layout de PDF bancário brasileiro,
incluindo os gerados por openhtmltopdf (Nubank), tabelas complexas,
multi-coluna, etc.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from leitor.parsers.base import TransactionRaw
from leitor.utils.config import ANTHROPIC_API_KEY
from leitor.utils.currency import parse_brl
from leitor.utils.date_utils import parse_date

SYSTEM_PROMPT = """Você é um especialista em extratos bancários brasileiros.
Sua tarefa é extrair TODAS as transações financeiras de um texto de extrato bancário.

Para cada transação, retorne um objeto JSON com:
- date: string no formato "DD/MM/YYYY"
- description: descrição da transação (texto limpo, sem valores)
- amount: valor numérico (negativo para débitos/saídas, positivo para créditos/entradas)
- type: "debit" ou "credit"

Regras importantes:
- Inclua TODAS as transações visíveis, sem exceção
- Para débitos (saídas, pagamentos, compras): amount NEGATIVO
- Para créditos (salário, depósitos, PIX recebido, estorno): amount POSITIVO
- Ignore saldo, totais e linhas de cabeçalho — capture apenas movimentações
- Se houver coluna separada de débito e crédito, use o sinal correto
- Valores em BRL: use ponto como decimal (ex: -45.90, não -45,90)

Retorne APENAS um array JSON válido, sem texto adicional, sem markdown:
[{"date":"DD/MM/YYYY","description":"...","amount":-45.90,"type":"debit"},...]"""


def _extract_pdf_text(path: Path) -> list[str]:
    """Extrai texto de cada página do PDF. Retorna lista de textos por página."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("Instale pdfplumber: pip install pdfplumber")

    pages_text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            # Tenta extrair texto; se vazio, tenta extrair palavras
            text = page.extract_text() or ""
            if not text.strip():
                words = page.extract_words()
                if words:
                    # Reconstrói texto agrupando palavras por linha (Y)
                    from collections import defaultdict
                    by_y: dict[float, list] = defaultdict(list)
                    for w in words:
                        y_key = round(w["top"] / 5) * 5  # agrupa por faixa de 5pt
                        by_y[y_key].append((w["x0"], w["text"]))
                    lines = []
                    for y in sorted(by_y.keys()):
                        line_words = sorted(by_y[y], key=lambda x: x[0])
                        lines.append("  ".join(w[1] for w in line_words))
                    text = "\n".join(lines)
            if text.strip():
                pages_text.append(text)
    return pages_text


def _chunk_pages(pages: list[str], max_chars: int = 6000) -> list[str]:
    """Agrupa páginas em chunks que caibam no contexto do Claude."""
    chunks = []
    current = ""
    for page in pages:
        if len(current) + len(page) > max_chars and current:
            chunks.append(current)
            current = page
        else:
            current = current + "\n\n---PÁGINA---\n\n" + page if current else page
    if current:
        chunks.append(current)
    return chunks


def _call_claude(text_chunk: str, bank: str) -> list[dict]:
    """Chama o Claude API e retorna lista de dicts de transações."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("Instale anthropic: pip install anthropic")

    if not ANTHROPIC_API_KEY:
        return []

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    user_message = (
        f"Banco: {bank.upper()}\n\n"
        f"Texto do extrato:\n\n{text_chunk}"
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()

    # Remove possível markdown ```json ... ```
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Tenta extrair o array JSON mesmo com lixo ao redor
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return []


def _dict_to_raw(item: dict, bank: str, source: str) -> TransactionRaw | None:
    """Converte um dict retornado pelo Claude em TransactionRaw."""
    date_str = str(item.get("date", "")).strip()
    if not date_str or parse_date(date_str) is None:
        return None

    description = str(item.get("description", "")).strip()
    if not description:
        description = "Sem descrição"

    amount_raw = item.get("amount", 0)
    try:
        amount_val = float(amount_raw)
    except (ValueError, TypeError):
        amount_val = 0.0

    if amount_val == 0:
        return None

    # Claude já devolve com sinal correto; converte para string BRL
    amount_str = f"{amount_val:.2f}".replace(".", ",")
    ttype = item.get("type", "credit" if amount_val > 0 else "debit")

    return TransactionRaw(
        bank=bank,
        date_str=date_str,
        description=description,
        amount_str=amount_str,
        transaction_type=ttype,
        source_file=source,
    )


def extract_with_claude(path: Path, bank: str) -> list[TransactionRaw]:
    """Extrai transações de um PDF usando o Claude API como agente de leitura.

    Retorna lista vazia se a API key não estiver configurada ou se falhar.
    """
    if not ANTHROPIC_API_KEY:
        return []

    pages = _extract_pdf_text(path)
    if not pages:
        return []

    chunks = _chunk_pages(pages)
    all_results: list[TransactionRaw] = []
    seen_keys: set[str] = set()

    for chunk in chunks:
        try:
            items = _call_claude(chunk, bank)
            for item in items:
                raw = _dict_to_raw(item, bank, str(path))
                if raw is None:
                    continue
                # Deduplicação leve por data+descrição+valor
                key = f"{raw.date_str}|{raw.description[:20]}|{raw.amount_str}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_results.append(raw)
        except Exception:
            continue

    return all_results
