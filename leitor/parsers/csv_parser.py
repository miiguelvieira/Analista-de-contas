"""Parser genérico para CSV/Excel, configurado por bank_profiles.yaml."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from leitor.parsers.base import AbstractParser, TransactionRaw
from leitor.utils.config import BANK_PROFILES_FILE


def _load_profiles() -> dict:
    with open(BANK_PROFILES_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


class CSVParser(AbstractParser):
    """Parseia CSV/Excel usando o perfil do banco em bank_profiles.yaml."""

    bank_name = "generic"

    def __init__(self, bank: str = "generic"):
        self.bank_name = bank
        self._profiles = _load_profiles()

    @classmethod
    def can_handle(cls, path: Path) -> bool:
        return path.suffix.lower() in (".csv", ".xlsx", ".xls", ".txt")

    def _parse(self, path: Path) -> list[TransactionRaw]:
        profile = self._profiles.get(self.bank_name, self._profiles.get("generic", {}))
        csv_cfg = profile.get("csv", {})

        if csv_cfg.get("auto_detect", False):
            return self._parse_auto(path)

        encoding = csv_cfg.get("encoding", "utf-8")
        separator = csv_cfg.get("separator", ",")
        date_col = csv_cfg.get("date_col", "Data")
        desc_col = csv_cfg.get("description_col", "Descrição")
        amount_col = csv_cfg.get("amount_col", "Valor")
        date_fmt = csv_cfg.get("date_format", "%d/%m/%Y")

        if path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            for enc in [encoding, "utf-8", "latin-1", "cp1252"]:
                try:
                    for sep in [separator, ";", ",", "\t"]:
                        try:
                            df = pd.read_csv(path, encoding=enc, sep=sep, on_bad_lines="skip")
                            if len(df.columns) > 1:
                                break
                        except Exception:
                            continue
                    break
                except Exception:
                    continue

        df.columns = [str(c).strip() for c in df.columns]
        df = df.dropna(how="all")

        # Tenta encontrar as colunas por nome ou posição
        date_col = _find_col(df, date_col)
        desc_col = _find_col(df, desc_col)
        amount_col = _find_col(df, amount_col)

        if not all([date_col, desc_col, amount_col]):
            return self._parse_auto(path)

        results = []
        for _, row in df.iterrows():
            amount_str = str(row[amount_col]).strip()
            if not amount_str or amount_str.lower() in ("nan", "none", ""):
                continue

            from leitor.utils.currency import parse_brl
            amount = parse_brl(amount_str)
            ttype = "credit" if amount > 0 else "debit"

            results.append(TransactionRaw(
                bank=self.bank_name,
                date_str=str(row[date_col]).strip(),
                description=str(row[desc_col]).strip(),
                amount_str=amount_str,
                transaction_type=ttype,
                source_file=str(path),
            ))

        return results

    def _parse_auto(self, path: Path) -> list[TransactionRaw]:
        """Detecção heurística de colunas para bancos desconhecidos."""
        profiles = self._profiles.get("generic", {}).get("csv", {})
        encodings = profiles.get("encoding_candidates", ["utf-8", "latin-1", "cp1252"])

        df = None
        for enc in encodings:
            for sep in [";", ",", "\t"]:
                try:
                    df = pd.read_csv(path, encoding=enc, sep=sep, on_bad_lines="skip")
                    if len(df.columns) >= 3:
                        break
                except Exception:
                    continue
            if df is not None and len(df.columns) >= 3:
                break

        if df is None or df.empty:
            return []

        df.columns = [str(c).strip() for c in df.columns]
        df = df.dropna(how="all")

        date_col = _guess_date_col(df)
        amount_col = _guess_amount_col(df)
        desc_col = _guess_desc_col(df, exclude=[date_col, amount_col])

        if not all([date_col, amount_col, desc_col]):
            return []

        results = []
        for _, row in df.iterrows():
            amount_str = str(row[amount_col]).strip()
            if not amount_str or amount_str.lower() in ("nan", "none", ""):
                continue

            from leitor.utils.currency import parse_brl
            amount = parse_brl(amount_str)
            ttype = "credit" if amount > 0 else "debit"

            results.append(TransactionRaw(
                bank=self.bank_name,
                date_str=str(row[date_col]).strip(),
                description=str(row[desc_col]).strip(),
                amount_str=amount_str,
                transaction_type=ttype,
                source_file=str(path),
            ))

        return results


def _find_col(df: pd.DataFrame, name: str) -> str | None:
    """Busca coluna por nome exato ou parcial (case-insensitive)."""
    for col in df.columns:
        if col.strip().lower() == name.strip().lower():
            return col
    for col in df.columns:
        if name.strip().lower() in col.strip().lower():
            return col
    return None


def _guess_date_col(df: pd.DataFrame) -> str | None:
    date_hints = ["data", "date", "dt", "dia"]
    for col in df.columns:
        if any(h in col.lower() for h in date_hints):
            return col
    return None


def _guess_amount_col(df: pd.DataFrame) -> str | None:
    amount_hints = ["valor", "value", "amount", "quantia", "debito", "credito", "r$"]
    for col in df.columns:
        if any(h in col.lower() for h in amount_hints):
            return col
    return None


def _guess_desc_col(df: pd.DataFrame, exclude: list) -> str | None:
    desc_hints = ["descr", "histor", "lancam", "memo", "detalhe", "operacao"]
    for col in df.columns:
        if col in exclude:
            continue
        if any(h in col.lower() for h in desc_hints):
            return col
    # Fallback: coluna de texto mais longa
    for col in df.columns:
        if col in exclude:
            continue
        sample = df[col].dropna().astype(str)
        if sample.str.len().mean() > 5:
            return col
    return None
