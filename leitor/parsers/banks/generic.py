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
        from leitor.parsers.pdf_extractor import extract_pdf
        return extract_pdf(path, "generic")
