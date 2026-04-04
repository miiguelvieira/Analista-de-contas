"""Conversão de strings monetárias brasileiras para Decimal."""
import re
from decimal import Decimal, InvalidOperation


def parse_brl(value: str | float | int | None) -> Decimal:
    """Converte 'R$ -1.234,56', '-1234.56', '1.234,56' etc. para Decimal.

    Retorna Decimal('0') se o valor for vazio ou inválido.
    """
    if value is None:
        return Decimal("0")

    if isinstance(value, (int, float)):
        return Decimal(str(value))

    text = str(value).strip()

    # Remove símbolo de moeda e espaços
    text = re.sub(r"R\$\s*", "", text).strip()

    if not text or text in ("-", "+"):
        return Decimal("0")

    # Detecta formato: se tem vírgula após ponto → BR (1.234,56)
    # Se tem ponto após vírgula → US (1,234.56)
    # Se só tem vírgula sem ponto → BR decimal (1234,56)
    if "," in text and "." in text:
        # Qual vem por último?
        last_comma = text.rfind(",")
        last_dot = text.rfind(".")
        if last_comma > last_dot:
            # Formato BR: 1.234,56
            text = text.replace(".", "").replace(",", ".")
        else:
            # Formato US: 1,234.56
            text = text.replace(",", "")
    elif "," in text:
        # Só vírgula: BR decimal → 1234,56
        text = text.replace(",", ".")

    try:
        return Decimal(text)
    except InvalidOperation:
        return Decimal("0")


def format_brl(value: Decimal | float | int) -> str:
    """Formata Decimal como string BRL: R$ 1.234,56"""
    d = Decimal(str(value))
    negative = d < 0
    abs_val = abs(d)

    integer_part, _, decimal_part = f"{abs_val:.2f}".partition(".")

    # Adiciona separadores de milhar
    chunks = []
    integer_str = str(integer_part)
    while len(integer_str) > 3:
        chunks.append(integer_str[-3:])
        integer_str = integer_str[:-3]
    chunks.append(integer_str)
    formatted_int = ".".join(reversed(chunks))

    sign = "-" if negative else ""
    return f"{sign}R$ {formatted_int},{decimal_part}"
