"""Parser para extrato do Banco do Brasil (CSV e PDF)."""
from pathlib import Path

from leitor.parsers.base import AbstractParser, TransactionRaw
from leitor.utils.currency import parse_brl


class BBParser(AbstractParser):
    bank_name = "bb"

    @classmethod
    def can_handle(cls, path: Path) -> bool:
        try:
            content = path.read_text(encoding="latin-1", errors="ignore").lower()
            return "banco do brasil" in content or "banco brasil" in content
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
                bank="bb",
                date_str=str(row[date_col]).strip(),
                description=str(row[desc_col]).strip(),
                amount_str=amount_str,
                transaction_type=ttype,
                source_file=str(path),
            ))
        return results

    def _parse_pdf(self, path: Path) -> list[TransactionRaw]:
        from leitor.parsers.pdf_extractor import extract_pdf
        return extract_pdf(path, "bb")
