"""Parser para extrato do Itaú (CSV e PDF)."""
from pathlib import Path

from leitor.parsers.base import AbstractParser, TransactionRaw
from leitor.utils.currency import parse_brl


class ItauParser(AbstractParser):
    bank_name = "itau"

    @classmethod
    def can_handle(cls, path: Path) -> bool:
        try:
            content = path.read_text(encoding="latin-1", errors="ignore").lower()
            return "itau" in content or "itaú" in content
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
        desc_col = next((c for c in df.columns if any(k in c.lower() for k in ["lança", "lanca", "hist", "descr"])), None)
        amount_col = next((c for c in df.columns if "valor" in c.lower()), None)

        if not all([date_col, desc_col, amount_col]):
            return []

        results = []
        for _, row in df.iterrows():
            amount_str = str(row[amount_col]).strip()
            amount = parse_brl(amount_str)
            ttype = "credit" if amount > 0 else "debit"
            results.append(TransactionRaw(
                bank="itau",
                date_str=str(row[date_col]).strip(),
                description=str(row[desc_col]).strip(),
                amount_str=amount_str,
                transaction_type=ttype,
                source_file=str(path),
            ))
        return results

    def _parse_pdf(self, path: Path) -> list[TransactionRaw]:
        from leitor.parsers.banks.bradesco import _extract_from_table, _extract_from_text_lines
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("Instale 'pdfplumber': pip install pdfplumber")

        results = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    results.extend(_extract_from_table(table, "itau", str(path)))
                if not tables:
                    text = page.extract_text() or ""
                    results.extend(_extract_from_text_lines(text, "itau", str(path)))
        return results
