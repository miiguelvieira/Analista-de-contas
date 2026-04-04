"""Parser para extrato do Nubank (CSV)."""
from pathlib import Path

from leitor.parsers.base import AbstractParser, TransactionRaw
from leitor.utils.currency import parse_brl


class NubankParser(AbstractParser):
    """Nubank exporta CSV com colunas: Data, Descrição, Valor (negativo = gasto)."""
    bank_name = "nubank"

    @classmethod
    def can_handle(cls, path: Path) -> bool:
        if path.suffix.lower() != ".csv":
            return False
        try:
            content = path.read_text(encoding="utf-8", errors="ignore").lower()
            return "nubank" in content or ("data" in content and "descrição" in content and "valor" in content)
        except Exception:
            return False

    def _parse(self, path: Path) -> list[TransactionRaw]:
        import pandas as pd
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except Exception:
                continue
        else:
            return []

        df.columns = [c.strip() for c in df.columns]
        df = df.dropna(how="all")

        # Nubank: Data, Descrição, Valor
        date_col = next((c for c in df.columns if "data" in c.lower()), None)
        desc_col = next((c for c in df.columns if "descri" in c.lower()), None)
        amount_col = next((c for c in df.columns if "valor" in c.lower()), None)

        if not all([date_col, desc_col, amount_col]):
            return []

        results = []
        for _, row in df.iterrows():
            amount_str = str(row[amount_col]).strip()
            amount = parse_brl(amount_str)
            # Nubank: negativo = gasto (debit), positivo = pagamento/estorno (credit)
            ttype = "credit" if amount > 0 else "debit"

            results.append(TransactionRaw(
                bank="nubank",
                date_str=str(row[date_col]).strip(),
                description=str(row[desc_col]).strip(),
                amount_str=amount_str,
                transaction_type=ttype,
                source_file=str(path),
            ))

        return results
