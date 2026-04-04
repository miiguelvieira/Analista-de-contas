"""Contrato base para todos os parsers de extrato."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TransactionRaw:
    """Transação bruta, diretamente extraída do arquivo fonte."""
    bank: str
    date_str: str
    description: str
    amount_str: str
    transaction_type: str = "unknown"  # "debit" | "credit" | "unknown"
    source_file: str = ""
    extra: dict = field(default_factory=dict)


class AbstractParser(ABC):
    """Interface que todos os parsers devem implementar."""

    bank_name: str = "generic"

    def parse(self, file_path: str | Path) -> list[TransactionRaw]:
        """Lê o arquivo e retorna lista de TransactionRaw."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
        return self._parse(path)

    @abstractmethod
    def _parse(self, path: Path) -> list[TransactionRaw]:
        ...

    @classmethod
    def can_handle(cls, path: Path) -> bool:
        """Retorna True se este parser consegue processar o arquivo."""
        return False
