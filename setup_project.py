"""Cria todas as pastas necessárias para o projeto Leitor de Conta / ML Pipeline.

Lê os caminhos diretamente do config.yaml para garantir consistência.

Uso:
    python setup_project.py
    python setup_project.py --config outro_config.yaml
    python setup_project.py --dry-run   # mostra o que seria criado sem criar nada
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Carrega config.yaml ───────────────────────────────────────────────────────

ROOT = Path(__file__).parent.resolve()


def load_config(config_path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        print(
            "Erro: PyYAML não está instalado.\n  pip install pyyaml",
            file=sys.stderr,
        )
        sys.exit(1)

    if not config_path.exists():
        print(f"Erro: config não encontrado: {config_path}", file=sys.stderr)
        sys.exit(1)

    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Resolve pastas a partir do config ────────────────────────────────────────

def collect_dirs(cfg: dict, root: Path) -> list[Path]:
    """Extrai todos os caminhos de pasta do config e converte para absolutos."""
    dirs: list[Path] = []

    # Caminhos declarados em paths:
    for key, rel_path in cfg.get("paths", {}).items():
        dirs.append(root / rel_path)

    # Pastas extras que não vêm do config mas são necessárias
    extras = [
        root / "data",                    # raiz de dados (pai de tudo)
        root / "data" / "splits",         # pai dos splits train/val/test
    ]
    dirs.extend(extras)

    # Remove duplicatas mantendo a ordem
    seen: set[Path] = set()
    unique: list[Path] = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            unique.append(d)

    return sorted(unique)


# ── Gitkeep para pastas vazias ────────────────────────────────────────────────

GITKEEP_DIRS = {
    "data/pdfs",
    "data/images",
    "data/annotations",
    "data/splits/train",
    "data/splits/val",
    "data/splits/test",
    "data/checkpoints",
    "data/logs",
    "data/predictions",
}


def _should_gitkeep(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        return False
    return rel in GITKEEP_DIRS


# ── Criação das pastas ────────────────────────────────────────────────────────

def setup_dirs(dirs: list[Path], root: Path, dry_run: bool) -> None:
    created = 0
    skipped = 0

    for d in dirs:
        existed = d.exists()
        if not dry_run:
            d.mkdir(parents=True, exist_ok=True)

        if existed:
            print(f"  [existe]  {d.relative_to(root)}")
            skipped += 1
        else:
            print(f"  [criado]  {d.relative_to(root)}")
            created += 1

        # Adiciona .gitkeep em pastas de dados para preservar no git
        if _should_gitkeep(d, root):
            gitkeep = d / ".gitkeep"
            if not gitkeep.exists():
                if not dry_run:
                    gitkeep.touch()
                print(f"            + .gitkeep adicionado")

    tag = "[DRY-RUN] " if dry_run else ""
    print(f"\n{tag}{created} pasta(s) criada(s), {skipped} já existia(m).")


# ── Atualiza .gitignore ───────────────────────────────────────────────────────

GITIGNORE_ADDITIONS = """\
# Modelos e checkpoints grandes (não versionar)
data/checkpoints/
# Imagens geradas (reproduzíveis a partir dos PDFs)
data/images/
# Predições de inferência
data/predictions/
"""


def update_gitignore(root: Path, dry_run: bool) -> None:
    gitignore = root / ".gitignore"
    marker = "# Modelos e checkpoints grandes"

    if gitignore.exists():
        content = gitignore.read_text(encoding="utf-8")
        if marker in content:
            return  # já atualizado

    if dry_run:
        print("\n[DRY-RUN] Adicionaria entradas ao .gitignore.")
        return

    with gitignore.open("a", encoding="utf-8") as f:
        f.write("\n" + GITIGNORE_ADDITIONS)
    print("\n.gitignore atualizado com entradas para checkpoints e imagens.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="setup_project",
        description="Cria a estrutura de pastas do projeto a partir do config.yaml.",
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        metavar="FILE",
        help="Caminho para o arquivo de configuração (padrão: config.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra o que seria criado sem modificar o disco.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = ROOT / args.config

    print(f"Projeto: {ROOT}")
    print(f"Config:  {config_path.relative_to(ROOT)}")
    if args.dry_run:
        print("Modo:    DRY-RUN (nenhuma alteração será feita)\n")
    else:
        print()

    cfg = load_config(config_path)
    dirs = collect_dirs(cfg, ROOT)

    print("Estrutura de pastas:")
    setup_dirs(dirs, ROOT, dry_run=args.dry_run)
    update_gitignore(ROOT, dry_run=args.dry_run)

    if not args.dry_run:
        print("\nSetup concluído. Próximos passos:")
        print("  1. Coloque seus PDFs em:  data/pdfs/")
        print("  2. Converta para imagens: python pdf_to_png.py data/pdfs/ -o data/images/")
        print("  3. Anote as imagens em:   data/annotations/")


if __name__ == "__main__":
    main()
