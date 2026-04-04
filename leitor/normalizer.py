"""Converte TransactionRaw → Transaction (canônica)."""
from __future__ import annotations

import hashlib
from decimal import Decimal

from leitor.models.transaction import Transaction
from leitor.parsers.base import TransactionRaw
from leitor.utils.currency import parse_brl
from leitor.utils.date_utils import parse_date


def _make_id(bank: str, date_str: str, amount_str: str, description: str) -> str:
    raw = f"{bank}|{date_str}|{amount_str}|{description[:40]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def normalize(raw: TransactionRaw) -> Transaction | None:
    """Converte um TransactionRaw para Transaction. Retorna None se inválido."""
    parsed_date = parse_date(raw.date_str)
    if parsed_date is None:
        return None

    amount = parse_brl(raw.amount_str)
    if amount == Decimal("0") and not raw.amount_str.strip().startswith("0"):
        return None

    ttype = raw.transaction_type
    if ttype == "unknown":
        ttype = "credit" if amount > 0 else "debit"

    txn_id = _make_id(raw.bank, raw.date_str, str(amount), raw.description)

    return Transaction(
        id=txn_id,
        bank=raw.bank,
        date=parsed_date,
        description=raw.description,
        amount=amount,
        transaction_type=ttype,
        source_file=raw.source_file,
    )


def normalize_all(raws: list[TransactionRaw]) -> list[Transaction]:
    """Normaliza lista de TransactionRaw, descartando inválidos."""
    result = []
    seen_ids: set[str] = set()
    for raw in raws:
        txn = normalize(raw)
        if txn is None:
            continue
        if txn.id in seen_ids:
            continue
        seen_ids.add(txn.id)
        result.append(txn)

    return sorted(result, key=lambda t: t.date)
