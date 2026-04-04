"""Converte páginas de PDFs em imagens PNG (ou JPEG) para anotação manual.

Uso:
    python pdf_to_png.py relatorio.pdf
    python pdf_to_png.py ./pdfs/ --output ./imgs --dpi 300 --workers 4

Dependências:
    pip install pdf2image tqdm

Poppler (requerido pelo pdf2image):
    Windows: https://github.com/oschwartz10612/poppler-windows/releases
             Extraia e adicione a pasta bin/ ao PATH
    Linux:   sudo apt install poppler-utils
    macOS:   brew install poppler
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _find_poppler_path() -> str | None:
    """Detecta Poppler local em data/poppler/ (Windows sem PATH configurado)."""
    local = Path(__file__).parent / "data" / "poppler"
    if local.exists():
        bins = list(local.rglob("pdftoppm.exe"))
        if bins:
            return str(bins[0].parent)
    return None


POPPLER_PATH = _find_poppler_path()

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from pdf2image import convert_from_path
    from pdf2image.exceptions import (
        PDFInfoNotInstalledError,
        PDFPageCountError,
        PDFSyntaxError,
    )
except ImportError:
    print(
        "Erro: pdf2image não está instalado.\n"
        "  pip install pdf2image\n\n"
        "Também é necessário ter o Poppler instalado no sistema.\n"
        "  Windows: https://github.com/oschwartz10612/poppler-windows/releases\n"
        "  Linux:   sudo apt install poppler-utils\n"
        "  macOS:   brew install poppler",
        file=sys.stderr,
    )
    sys.exit(1)


# ── Conversão de um único PDF ────────────────────────────────────────────────

def convert_pdf(
    pdf_path: Path,
    output_dir: Path,
    dpi: int,
    fmt: str,
) -> tuple[int, float]:
    """Converte todas as páginas de *pdf_path* em imagens e salva em *output_dir*.

    Returns:
        (n_pages, elapsed_seconds)

    Raises:
        RuntimeError: se o PDF não puder ser aberto ou convertido.
    """
    dest = output_dir / pdf_path.stem
    dest.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    try:
        pages = convert_from_path(str(pdf_path), dpi=dpi, poppler_path=POPPLER_PATH)
    except PDFInfoNotInstalledError:
        raise RuntimeError(
            "Poppler não encontrado no PATH. "
            "Instale-o e certifique-se de que 'pdftoppm' está acessível."
        )
    except PDFPageCountError as exc:
        raise RuntimeError(f"Não foi possível contar páginas: {exc}")
    except PDFSyntaxError as exc:
        raise RuntimeError(f"PDF corrompido ou inválido: {exc}")
    except Exception as exc:
        raise RuntimeError(f"Erro inesperado ao converter: {exc}")

    for i, page in enumerate(pages, start=1):
        out_file = dest / f"page_{i:03d}.{fmt}"
        save_fmt = "JPEG" if fmt == "jpeg" else "PNG"
        page.save(str(out_file), save_fmt)

    elapsed = time.perf_counter() - t0
    return len(pages), elapsed


# ── Processamento de pasta ───────────────────────────────────────────────────

def process_folder(
    folder: Path,
    output_dir: Path,
    dpi: int,
    fmt: str,
    workers: int,
) -> dict:
    """Converte todos os PDFs encontrados recursivamente em *folder*.

    Returns:
        dict com chaves: total_pdfs, ok, errors, total_pages, elapsed
    """
    pdf_files = sorted(folder.rglob("*.pdf"))
    if not pdf_files:
        print(f"Nenhum PDF encontrado em: {folder}", file=sys.stderr)
        sys.exit(1)

    print(f"Encontrados {len(pdf_files)} PDF(s) em '{folder}'. Iniciando conversão...")

    results = {"total_pdfs": len(pdf_files), "ok": 0, "errors": 0, "total_pages": 0}
    t0 = time.perf_counter()

    iterator = _tqdm(total=len(pdf_files), unit="pdf") if HAS_TQDM else None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_pdf = {
            executor.submit(convert_pdf, p, output_dir, dpi, fmt): p
            for p in pdf_files
        }
        for future in as_completed(future_to_pdf):
            pdf = future_to_pdf[future]
            try:
                n_pages, elapsed = future.result()
                results["ok"] += 1
                results["total_pages"] += n_pages
                msg = f"[OK] {pdf.name} — {n_pages} página(s) ({elapsed:.1f}s)"
            except RuntimeError as exc:
                results["errors"] += 1
                msg = f"[ERRO] {pdf.name} — {exc}"
            except PermissionError:
                results["errors"] += 1
                msg = f"[ERRO] {pdf.name} — permissão negada"

            if iterator is not None:
                iterator.set_postfix_str(pdf.name[:40])
                iterator.update(1)
            else:
                print(msg)

    if iterator is not None:
        iterator.close()

    results["elapsed"] = time.perf_counter() - t0
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf_to_png",
        description="Converte páginas de PDFs em imagens PNG/JPEG (300 DPI por padrão).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemplos:\n"
            "  python pdf_to_png.py relatorio.pdf\n"
            "  python pdf_to_png.py ./pdfs/ --output ./imgs --dpi 150 --workers 8\n"
            "  python pdf_to_png.py doc.pdf --fmt jpeg --dpi 200 -o /tmp/out"
        ),
    )
    parser.add_argument(
        "input",
        metavar="INPUT",
        help="Arquivo PDF ou pasta contendo PDFs.",
    )
    parser.add_argument(
        "--output", "-o",
        default="output_images",
        metavar="DIR",
        help="Pasta raiz de saída (padrão: ./output_images).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        metavar="N",
        help="Resolução em DPI (padrão: 300).",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        metavar="N",
        help="Threads paralelas para processar múltiplos PDFs (padrão: 4).",
    )
    parser.add_argument(
        "--fmt",
        choices=["png", "jpeg"],
        default="png",
        help="Formato de saída: png ou jpeg (padrão: png).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        print(f"Erro: caminho não encontrado: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            print(f"Erro: o arquivo não é um PDF: {input_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Convertendo '{input_path.name}' @ {args.dpi} DPI → {output_dir}/")
        try:
            n_pages, elapsed = convert_pdf(input_path, output_dir, args.dpi, args.fmt)
            print(
                f"Concluído: {n_pages} página(s) salvas em "
                f"'{output_dir / input_path.stem}' ({elapsed:.1f}s)"
            )
        except RuntimeError as exc:
            print(f"Erro: {exc}", file=sys.stderr)
            sys.exit(1)

    elif input_path.is_dir():
        stats = process_folder(input_path, output_dir, args.dpi, args.fmt, args.workers)
        print(
            f"\nSumário: {stats['ok']}/{stats['total_pdfs']} PDFs convertidos | "
            f"{stats['total_pages']} páginas | "
            f"{stats['errors']} erro(s) | "
            f"{stats['elapsed']:.1f}s total"
        )
        if stats["errors"] > 0:
            sys.exit(1)

    else:
        print(f"Erro: '{input_path}' não é um arquivo nem uma pasta.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
