"""Parser genérico para CSV/Excel, configurado por bank_profiles.yaml."""
from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd
import yaml

from leitor.parsers.base import AbstractParser, TransactionRaw
from leitor.utils.config import BANK_PROFILES_FILE

ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
SEPARATORS = [";", ",", "\t", "|"]
MAX_SKIP_ROWS = 15  # máximo de linhas de cabeçalho extra para pular


def _load_profiles() -> dict:
    with open(BANK_PROFILES_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm(text: str) -> str:
    """Remove acentos e coloca em minúsculas."""
    nfkd = unicodedata.normalize("NFKD", str(text))
    return nfkd.encode("ascii", "ignore").decode("ascii").lower().strip()


class CSVParser(AbstractParser):
    bank_name = "generic"

    def __init__(self, bank: str = "generic"):
        self.bank_name = bank
        self._profiles = _load_profiles()

    @classmethod
    def can_handle(cls, path: Path) -> bool:
        return path.suffix.lower() in (".csv", ".xlsx", ".xls", ".txt")

    def _parse(self, path: Path) -> list[TransactionRaw]:
        if path.suffix.lower() in (".xlsx", ".xls"):
            return self._parse_excel(path)
        return self._parse_auto(path)

    def _parse_excel(self, path: Path) -> list[TransactionRaw]:
        for skip in range(MAX_SKIP_ROWS):
            try:
                df = pd.read_excel(path, skiprows=skip)
                df.columns = [str(c).strip() for c in df.columns]
                df = df.dropna(how="all")
                date_col, amount_col, desc_col = _find_columns(df)
                if date_col and amount_col and desc_col:
                    return _extract_rows(df, date_col, amount_col, desc_col, self.bank_name, str(path))
            except Exception:
                continue
        return []

    def _parse_auto(self, path: Path) -> list[TransactionRaw]:
        """Detecção heurística robusta: testa encoding × separador × linhas puladas."""
        best: list[TransactionRaw] = []

        for enc in ENCODINGS:
            for sep in SEPARATORS:
                for skip in range(MAX_SKIP_ROWS):
                    try:
                        df = pd.read_csv(
                            path,
                            encoding=enc,
                            sep=sep,
                            skiprows=skip,
                            on_bad_lines="skip",
                            dtype=str,
                        )
                        df.columns = [str(c).strip() for c in df.columns]
                        df = df.dropna(how="all")

                        if len(df.columns) < 3 or df.empty:
                            continue

                        date_col, amount_col, desc_col = _find_columns(df)
                        if not (date_col and amount_col and desc_col):
                            continue

                        results = _extract_rows(df, date_col, amount_col, desc_col, self.bank_name, str(path))
                        if len(results) > len(best):
                            best = results
                            if len(best) > 5:
                                return best  # encontrou resultado bom, retorna logo
                    except Exception:
                        continue

        return best


def _find_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    """Encontra as colunas de data, valor e descrição em um DataFrame."""
    date_col = _guess_date_col(df)
    amount_col = _guess_amount_col(df)
    desc_col = _guess_desc_col(df, exclude=[c for c in [date_col, amount_col] if c])
    return date_col, amount_col, desc_col


def _extract_rows(
    df: pd.DataFrame,
    date_col: str,
    amount_col: str,
    desc_col: str,
    bank: str,
    source: str,
) -> list[TransactionRaw]:
    from leitor.utils.currency import parse_brl
    from leitor.utils.date_utils import parse_date

    results = []
    for _, row in df.iterrows():
        amount_str = str(row[amount_col]).strip()
        if not amount_str or amount_str.lower() in ("nan", "none", "", "-"):
            continue

        date_str = str(row[date_col]).strip()
        if not date_str or date_str.lower() in ("nan", "none", ""):
            continue

        # Valida que a data é parseable
        if parse_date(date_str) is None:
            continue

        amount = parse_brl(amount_str)
        # Ignora linha se valor é 0 e o campo não era literalmente "0"
        if amount == 0 and amount_str not in ("0", "0,00", "0.00", "R$ 0,00"):
            continue

        ttype = "credit" if amount > 0 else "debit"
        desc = str(row[desc_col]).strip()
        if not desc or desc.lower() in ("nan", "none"):
            desc = "Sem descrição"

        results.append(TransactionRaw(
            bank=bank,
            date_str=date_str,
            description=desc,
            amount_str=amount_str,
            transaction_type=ttype,
            source_file=source,
        ))

    return results


def _guess_date_col(df: pd.DataFrame) -> str | None:
    date_hints = ["data", "date", "dt", "dia", "vencimento", "competencia"]
    for col in df.columns:
        n = _norm(col)
        if any(h == n or n.startswith(h) for h in date_hints):
            return col
    # Segunda passagem: verifica se os valores da coluna parecem datas
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(5)
        if _looks_like_dates(sample):
            return col
    return None


def _guess_amount_col(df: pd.DataFrame) -> str | None:
    amount_hints = ["valor", "value", "amount", "quantia", "r$", "debito", "credito",
                    "saida", "entrada", "lancamento", "movimentacao"]
    for col in df.columns:
        n = _norm(col)
        if any(h in n for h in amount_hints):
            return col
    # Segunda passagem: verifica se os valores parecem monetários
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(10)
        if _looks_like_amounts(sample):
            return col
    return None


def _guess_desc_col(df: pd.DataFrame, exclude: list[str]) -> str | None:
    desc_hints = ["descr", "histor", "lancam", "memo", "detalhe", "operacao",
                  "estabelecimento", "beneficiario", "pagamento", "complemento", "titulo"]
    for col in df.columns:
        if col in exclude:
            continue
        n = _norm(col)
        if any(h in n for h in desc_hints):
            return col
    # Fallback: coluna de texto mais longa (excluindo data e valor)
    best_col, best_len = None, 0
    for col in df.columns:
        if col in exclude:
            continue
        sample = df[col].dropna().astype(str)
        if sample.empty:
            continue
        avg_len = sample.str.len().mean()
        if avg_len > best_len:
            best_len = avg_len
            best_col = col
    return best_col if best_len > 3 else None


def _looks_like_dates(series: pd.Series) -> bool:
    """Heurística: verifica se a série parece ter datas."""
    import re
    date_pattern = re.compile(r"\d{2}[/\-\.]\d{2}[/\-\.]\d{2,4}|\d{4}[/\-]\d{2}[/\-]\d{2}")
    count = sum(1 for v in series if date_pattern.search(str(v)))
    return count >= min(2, len(series))


def _looks_like_amounts(series: pd.Series) -> bool:
    """Heurística: verifica se a série parece ter valores monetários."""
    import re
    amount_pattern = re.compile(r"^-?\s*R?\$?\s*[\d.,]+$")
    count = sum(1 for v in series if amount_pattern.match(str(v).strip()))
    return count >= min(3, len(series))
