"""Parser genérico/heurístico — fallback para bancos não identificados."""
from pathlib import Path

from leitor.parsers.base import AbstractParser, TransactionRaw
from leitor.utils.currency import parse_brl


class GenericPDFParser(AbstractParser):
    """Extrai transações de qualquer PDF usando heurísticas."""
    bank_name = "generic"

    @classmethod
    def can_handle(cls, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def _parse(self, path: Path) -> list[TransactionRaw]:
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
                    results.extend(_extract_from_table(table, "generic", str(path)))
                if not tables:
                    text = page.extract_text() or ""
                    results.extend(_extract_from_text_lines(text, "generic", str(path)))

        return results
