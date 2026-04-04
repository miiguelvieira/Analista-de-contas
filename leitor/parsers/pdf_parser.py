"""Detector de banco + despachante para parsers de PDF/CSV/OFX."""
from __future__ import annotations

from pathlib import Path

from leitor.parsers.base import AbstractParser, TransactionRaw
from leitor.parsers.banks.nubank import NubankParser
from leitor.parsers.banks.bradesco import BradescoParser
from leitor.parsers.banks.itau import ItauParser
from leitor.parsers.banks.bb import BBParser
from leitor.parsers.banks.santander import SantanderParser
from leitor.parsers.banks.caixa import CaixaParser
from leitor.parsers.banks.generic import GenericPDFParser
from leitor.parsers.ofx_parser import OFXParser
from leitor.parsers.csv_parser import CSVParser

# Ordem importa: mais específicos primeiro
BANK_PARSERS: list[type[AbstractParser]] = [
    NubankParser,
    BradescoParser,
    ItauParser,
    BBParser,
    SantanderParser,
    CaixaParser,
]

BANK_NAMES = {
    "nubank": NubankParser,
    "bradesco": BradescoParser,
    "itau": ItauParser,
    "bb": BBParser,
    "banco do brasil": BBParser,
    "santander": SantanderParser,
    "caixa": CaixaParser,
}


def detect_bank(path: Path) -> str:
    """Tenta identificar o banco pelo conteúdo do arquivo."""
    try:
        if path.suffix.lower() == ".pdf":
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                first_page_text = (pdf.pages[0].extract_text() or "").lower() if pdf.pages else ""
        else:
            first_page_text = path.read_text(encoding="latin-1", errors="ignore")[:2000].lower()
    except Exception:
        return "generic"

    bank_signatures = {
        "nubank": ["nubank", "nu pagamentos"],
        "bradesco": ["bradesco"],
        "itau": ["itaú", "itau unibanco"],
        "bb": ["banco do brasil", "bancoobrasil"],
        "santander": ["santander"],
        "caixa": ["caixa econômica", "caixa economica", "cef"],
        "inter": ["banco inter", "inter s.a"],
        "c6bank": ["c6 bank", "c6bank"],
        "picpay": ["picpay"],
    }

    for bank, signatures in bank_signatures.items():
        if any(sig in first_page_text for sig in signatures):
            return bank

    return "generic"


def parse_file(path: str | Path, bank_hint: str | None = None) -> list[TransactionRaw]:
    """Ponto de entrada principal. Detecta banco e parseia o arquivo.

    Args:
        path: Caminho do arquivo.
        bank_hint: Nome do banco se já conhecido (pula detecção automática).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    # OFX/QIF: usa parser dedicado independente do banco
    if suffix in (".ofx", ".qif", ".ofc"):
        return OFXParser().parse(path)

    # Detecção de banco
    bank = bank_hint or detect_bank(path)

    # Tenta parser específico do banco
    parser_cls = BANK_NAMES.get(bank)
    if parser_cls:
        parser = parser_cls()
        try:
            results = parser.parse(path)
            if results:
                return results
        except Exception:
            pass

    # Fallback: CSV genérico ou PDF genérico
    if suffix in (".csv", ".xlsx", ".xls", ".txt"):
        csv_parser = CSVParser(bank_hint or "generic")
        results = csv_parser.parse(path)
        # Garante bank name correto
        for r in results:
            if r.bank == "generic" and bank != "generic":
                r.bank = bank
        return results

    if suffix == ".pdf":
        results = GenericPDFParser().parse(path)
        for r in results:
            r.bank = bank
        return results

    raise ValueError(f"Formato não suportado: {suffix}")
