"""Parser para arquivos OFX e QIF (padrão aberto, múltiplos bancos)."""
from pathlib import Path

from leitor.parsers.base import AbstractParser, TransactionRaw


class OFXParser(AbstractParser):
    bank_name = "ofx"

    @classmethod
    def can_handle(cls, path: Path) -> bool:
        return path.suffix.lower() in (".ofx", ".qif", ".ofc")

    def _parse(self, path: Path) -> list[TransactionRaw]:
        try:
            from ofxparse import OfxParser as _OfxParser
        except ImportError:
            raise ImportError("Instale 'ofxparse': pip install ofxparse")

        with open(path, "rb") as f:
            ofx = _OfxParser.parse(f)

        results: list[TransactionRaw] = []
        bank_name = "ofx"

        for account in (ofx.accounts if hasattr(ofx, "accounts") else [ofx.account]):
            if account is None:
                continue
            bank_id = getattr(account, "institution", None)
            if bank_id:
                bank_name = str(bank_id.organization or "ofx").lower()

            for txn in (account.statement.transactions if account.statement else []):
                ttype = "credit" if float(txn.amount) > 0 else "debit"
                results.append(TransactionRaw(
                    bank=bank_name,
                    date_str=txn.date.strftime("%Y-%m-%d") if txn.date else "",
                    description=str(txn.memo or txn.payee or ""),
                    amount_str=str(txn.amount),
                    transaction_type=ttype,
                    source_file=str(path),
                    extra={"ofx_id": str(txn.id)},
                ))

        return results
