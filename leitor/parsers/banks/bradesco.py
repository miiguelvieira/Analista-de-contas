"""Parser para extrato do Bradesco (PDF e CSV)."""
from pathlib import Path

from leitor.parsers.base import AbstractParser, TransactionRaw
from leitor.utils.currency import parse_brl


class BradescoParser(AbstractParser):
    bank_name = "bradesco"

    @classmethod
    def can_handle(cls, path: Path) -> bool:
        if path.suffix.lower() not in (".pdf", ".csv"):
            return False
        try:
            content = path.read_text(encoding="latin-1", errors="ignore").lower()
            return "bradesco" in content
        except Exception:
            return False

    def _parse(self, path: Path) -> list[TransactionRaw]:
        if path.suffix.lower() == ".pdf":
            return self._parse_pdf(path)
        return self._parse_csv(path)

    def _parse_csv(self, path: Path) -> list[TransactionRaw]:
        import pandas as pd
        for enc in ["latin-1", "cp1252", "utf-8"]:
            for sep in [";", ","]:
                try:
                    df = pd.read_csv(path, encoding=enc, sep=sep, on_bad_lines="skip")
                    if len(df.columns) >= 3:
                        break
                except Exception:
                    continue
            else:
                continue
            break
        else:
            return []

        df.columns = [c.strip() for c in df.columns]
        df = df.dropna(how="all")

        date_col = next((c for c in df.columns if "data" in c.lower()), None)
        desc_col = next((c for c in df.columns if any(k in c.lower() for k in ["hist", "descr", "lanc"])), None)
        amount_col = next((c for c in df.columns if "valor" in c.lower()), None)

        if not all([date_col, desc_col, amount_col]):
            return []

        results = []
        for _, row in df.iterrows():
            amount_str = str(row[amount_col]).strip()
            amount = parse_brl(amount_str)
            ttype = "credit" if amount > 0 else "debit"
            results.append(TransactionRaw(
                bank="bradesco",
                date_str=str(row[date_col]).strip(),
                description=str(row[desc_col]).strip(),
                amount_str=amount_str,
                transaction_type=ttype,
                source_file=str(path),
            ))
        return results

    def _parse_pdf(self, path: Path) -> list[TransactionRaw]:
        from leitor.parsers.pdf_extractor import extract_pdf
        return extract_pdf(path, "bradesco")


def _extract_from_table(table: list[list], bank: str, source: str) -> list[TransactionRaw]:
    """Extrai transações de tabela do pdfplumber."""
    results = []
    if not table or len(table) < 2:
        return results

    header = [str(c or "").strip().lower() for c in table[0]]
    date_idx = next((i for i, h in enumerate(header) if "data" in h), None)
    desc_idx = next((i for i, h in enumerate(header) if any(k in h for k in ["hist", "descr", "lanc"])), None)
    amount_idx = next((i for i, h in enumerate(header) if "valor" in h), None)

    if date_idx is None or desc_idx is None or amount_idx is None:
        return results

    for row in table[1:]:
        try:
            date_str = str(row[date_idx] or "").strip()
            desc = str(row[desc_idx] or "").strip()
            amount_str = str(row[amount_idx] or "").strip()
            if not date_str or not amount_str:
                continue
            amount = parse_brl(amount_str)
            ttype = "credit" if amount > 0 else "debit"
            results.append(TransactionRaw(
                bank=bank,
                date_str=date_str,
                description=desc,
                amount_str=amount_str,
                transaction_type=ttype,
                source_file=source,
            ))
        except Exception:
            continue

    return results


def _extract_from_text_lines(text: str, bank: str, source: str) -> list[TransactionRaw]:
    """Fallback: tenta parsear linhas de texto do PDF (heurístico)."""
    import re
    results = []
    # Padrão: DD/MM/YYYY ... valor (com ou sem sinal)
    pattern = re.compile(
        r"(\d{2}/\d{2}/\d{4})\s+(.+?)\s+([-+]?\s*[\d.,]+)\s*$",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        date_str, desc, amount_str = match.group(1), match.group(2).strip(), match.group(3).strip()
        amount = parse_brl(amount_str)
        ttype = "credit" if amount > 0 else "debit"
        results.append(TransactionRaw(
            bank=bank,
            date_str=date_str,
            description=desc,
            amount_str=amount_str,
            transaction_type=ttype,
            source_file=source,
        ))
    return results
