"""Parsing robusto de datas em formatos comuns nos bancos brasileiros."""
from datetime import date
from dateutil import parser as dateutil_parser

COMMON_FORMATS = [
    "%d/%m/%Y",
    "%d/%m/%y",
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d-%m-%y",
    "%d.%m.%Y",
    "%d.%m.%y",
    "%Y/%m/%d",
]


def parse_date(value: str | date | None) -> date | None:
    """Tenta converter string para date usando múltiplos formatos.

    Retorna None se não conseguir parsear.
    """
    if value is None:
        return None
    if isinstance(value, date):
        return value

    text = str(value).strip()
    if not text:
        return None

    from datetime import datetime
    for fmt in COMMON_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    # Fallback: dateutil (mais tolerante)
    try:
        return dateutil_parser.parse(text, dayfirst=True).date()
    except Exception:
        return None


def parse_date_strict(value: str, fmt: str) -> date | None:
    """Parse com formato específico."""
    from datetime import datetime
    try:
        return datetime.strptime(str(value).strip(), fmt).date()
    except ValueError:
        return None
