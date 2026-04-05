"""Pipeline de inferência: PDF → detecções LayoutLMv3 → OCR → regras de negócio.

Fluxo por página
----------------
PDF → imagem (300 DPI) → LayoutLMv3Detector → boxes por classe → NMS
  → para cada 'transaction_row': recorte em alta resolução → Tesseract OCR
  → regras de negócio: transferência? intra-usuário?
  → JSON estruturado + impressão no terminal

Pré-requisitos
--------------
    pip install pdf2image pytesseract pillow torch torchvision transformers \
                pyyaml tqdm
    # Tesseract instalado no sistema:
    #   Windows: https://github.com/UB-Mannheim/tesseract/wiki
    #   Linux:   sudo apt install tesseract-ocr tesseract-ocr-por
    #   macOS:   brew install tesseract tesseract-lang

Uso
---
    python predict.py extrato.pdf --checkpoint data/checkpoints/best
    python predict.py extrato.pdf --checkpoint data/checkpoints/best \\
        --holder "João da Silva" --score-thresh 0.4 --out resultados.json
    python predict.py extrato.pdf --checkpoint data/checkpoints/best --visualize
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import LayoutLMv3Processor

# Importa componentes do script de treino (mesma raiz do projeto)
try:
    from train_layoutlm import (
        LayoutLMv3Detector,
        NUM_CLASSES,
        NUM_PATCHES,
        CATEGORY_NAMES,
        cxcywh_to_xyxy,
    )
except ImportError as exc:
    sys.exit(
        f"Erro: não foi possível importar train_layoutlm.py.\n"
        f"Certifique-se de que o arquivo está na mesma pasta que predict.py.\n{exc}"
    )

log = logging.getLogger(__name__)

# ── Constantes de negócio ─────────────────────────────────────────────────────

TRANSFER_KEYWORDS: list[str] = [
    "ted", "doc", "pix", "transferencia", "transf", "trf",
    "remessa", "envio", "credito em conta", "debito em conta",
]

# Mapeamento palavra-chave → tipo de transferência (para o campo transfer_type)
TRANSFER_TYPE_MAP: dict[str, str] = {
    "pix":          "PIX",
    "ted":          "TED",
    "doc":          "DOC",
    "transferencia":"TRANSFERÊNCIA",
    "transf":       "TRANSFERÊNCIA",
    "trf":          "TRANSFERÊNCIA",
    "remessa":      "TRANSFERÊNCIA",
}

# Cores de visualização por classe (RGB)
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "header":             (52,  152, 219),   # azul
    "transaction_row":    (39,  174,  96),   # verde
    "transfer_indicator": (231,  76,  60),   # vermelho
    "saldo":              (155,  89, 182),   # roxo
    "background":         (189, 195, 199),   # cinza
}

# ── Índice classe → nome ──────────────────────────────────────────────────────

IDX_TO_CLASS: dict[int, str] = {i: name for i, name in enumerate(CATEGORY_NAMES)}
CLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(CATEGORY_NAMES)}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Carregamento do modelo
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[LayoutLMv3Detector, LayoutLMv3Processor]:
    """Carrega backbone (HF) + cabeças (.pt) de um checkpoint salvo por train_layoutlm.py."""
    backbone_dir = checkpoint_dir / "backbone"
    heads_path   = checkpoint_dir / "heads.pt"

    if not backbone_dir.exists():
        raise FileNotFoundError(f"Backbone não encontrado em: {backbone_dir}")
    if not heads_path.exists():
        raise FileNotFoundError(f"Arquivo de cabeças não encontrado: {heads_path}")

    log.info("Carregando processor de %s", backbone_dir)
    processor = LayoutLMv3Processor.from_pretrained(str(backbone_dir), apply_ocr=False)

    log.info("Carregando backbone de %s", backbone_dir)
    model = LayoutLMv3Detector(str(backbone_dir), num_classes=NUM_CLASSES)

    log.info("Carregando cabeças de detecção de %s", heads_path)
    heads = torch.load(heads_path, map_location=device)
    model.class_head.load_state_dict(heads["class_head"])
    model.bbox_head.load_state_dict(heads["bbox_head"])
    epoch = heads.get("epoch", "?")
    log.info("Checkpoint da época %s carregado com sucesso.", epoch)

    model.to(device).eval()
    return model, processor


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PDF → imagens
# ═══════════════════════════════════════════════════════════════════════════════

def _find_poppler_path() -> str | None:
    """Localiza o Poppler em data/poppler/ (bundled) ou retorna None."""
    root = Path(__file__).parent
    for candidate in (root / "data" / "poppler").rglob("pdftoppm.exe"):
        return str(candidate.parent)
    return None


def pdf_to_images(pdf_path: Path, dpi: int = 300) -> list[Image.Image]:
    """Converte cada página do PDF em uma imagem PIL no DPI especificado."""
    try:
        from pdf2image import convert_from_path
    except ImportError:
        sys.exit("Erro: pdf2image não instalado. pip install pdf2image")

    poppler_path = _find_poppler_path()
    log.info("Convertendo '%s' (%d DPI)...", pdf_path.name, dpi)
    try:
        pages = convert_from_path(str(pdf_path), dpi=dpi, poppler_path=poppler_path)
    except Exception as exc:
        raise RuntimeError(f"Falha ao converter PDF: {exc}") from exc

    log.info("%d página(s) convertida(s).", len(pages))
    return pages


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Inferência do modelo
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_page(
    model:       LayoutLMv3Detector,
    processor:   LayoutLMv3Processor,
    image:       Image.Image,           # imagem original (alta resolução)
    device:      torch.device,
    score_thresh: float = 0.35,
    nms_iou_thresh: float = 0.4,
) -> list[dict[str, Any]]:
    """Executa o modelo em uma página e retorna detecções após NMS.

    Returns:
        Lista de dicts com keys: class_name, class_idx, score, bbox_norm,
        bbox_px (x1,y1,x2,y2 em pixels da imagem original).
    """
    w_orig, h_orig = image.size

    # ── Pré-processamento (224×224 para o modelo) ─────────────────────────
    encoding = processor(
        images=image,
        text=["dummy"],
        boxes=[[0, 0, 0, 0]],
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    pixel_values = encoding["pixel_values"].to(device)

    # ── Forward ────────────────────────────────────────────────────────────
    logits, boxes = model(pixel_values)          # (1, 196, C), (1, 196, 4)
    logits = logits.squeeze(0)                   # (196, C)
    boxes  = boxes.squeeze(0)                    # (196, 4) cx,cy,w,h norm

    probs  = logits.softmax(-1)                  # (196, C)
    scores, labels = probs.max(-1)               # (196,), (196,)

    # ── Filtra background e score baixo ───────────────────────────────────
    keep = (labels > 0) & (scores >= score_thresh)
    if not keep.any():
        return []

    scores_k = scores[keep]
    labels_k = labels[keep]
    boxes_k  = boxes[keep]                       # (K, 4) norm cx,cy,w,h

    # ── Converte para xyxy normalizado para NMS ────────────────────────────
    boxes_xyxy_norm = cxcywh_to_xyxy(boxes_k).clamp(0.0, 1.0)  # (K, 4)

    # ── NMS por classe ─────────────────────────────────────────────────────
    try:
        from torchvision.ops import nms
        final_indices: list[int] = []
        for cls_idx in labels_k.unique():
            cls_mask   = labels_k == cls_idx
            cls_boxes  = boxes_xyxy_norm[cls_mask]
            cls_scores = scores_k[cls_mask]
            kept_local = nms(cls_boxes.float(), cls_scores.float(), nms_iou_thresh)
            global_idx = cls_mask.nonzero(as_tuple=True)[0][kept_local]
            final_indices.extend(global_idx.tolist())
    except ImportError:
        # torchvision não disponível: mantém todos (sem NMS)
        final_indices = list(range(len(scores_k)))

    # ── Constrói resultados em coordenadas de pixel da imagem original ────
    detections: list[dict[str, Any]] = []
    for i in final_indices:
        x1n, y1n, x2n, y2n = boxes_xyxy_norm[i].tolist()
        # Escala para pixels da imagem original (alta resolução)
        x1 = int(x1n * w_orig)
        y1 = int(y1n * h_orig)
        x2 = int(x2n * w_orig)
        y2 = int(y2n * h_orig)

        cls_idx  = labels_k[i].item()
        cls_name = IDX_TO_CLASS.get(cls_idx, "unknown")

        detections.append({
            "id":         str(uuid.uuid4()),
            "class_name": cls_name,
            "class_idx":  cls_idx,
            "score":      round(scores_k[i].item(), 4),
            "bbox_norm":  [round(v, 4) for v in [x1n, y1n, x2n, y2n]],
            "bbox_px":    [x1, y1, x2, y2],
        })

    # Ordena de cima para baixo (ordem natural do extrato)
    detections.sort(key=lambda d: d["bbox_px"][1])
    return detections


# ═══════════════════════════════════════════════════════════════════════════════
# 4. OCR das regiões
# ═══════════════════════════════════════════════════════════════════════════════

def _has_tesseract() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def ocr_region(
    image: Image.Image,
    bbox_px: list[int],
    lang: str = "por",
    padding: int = 6,
) -> str:
    """Extrai texto de uma região recortada da imagem usando Tesseract.

    Args:
        image:   Imagem original em alta resolução.
        bbox_px: [x1, y1, x2, y2] em pixels.
        lang:    Idioma do Tesseract ('por' para português, 'eng' para inglês).
        padding: Pixels extras ao redor do recorte para melhorar OCR de bordas.

    Returns:
        Texto extraído (string limpa), ou "" se Tesseract não estiver disponível.
    """
    try:
        import pytesseract
    except ImportError:
        return ""

    x1, y1, x2, y2 = bbox_px
    w, h = image.size

    # Aplica padding com clamp nos limites da imagem
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    if x2 <= x1 or y2 <= y1:
        return ""

    crop = image.crop((x1, y1, x2, y2))

    # PSM 6: bloco de texto uniforme — adequado para linhas de extrato
    config = f"--psm 6 --oem 3 -l {lang}"
    try:
        raw = pytesseract.image_to_string(crop, config=config)
        return " ".join(raw.split())  # normaliza espaços e quebras de linha
    except Exception as exc:
        log.debug("OCR falhou na região %s: %s", bbox_px, exc)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Regras de negócio
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize(text: str) -> str:
    """Remove acentos, converte para minúsculas e mantém apenas alnum + espaço."""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_only = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9 ]", " ", ascii_only.lower())


def classify_transaction(
    ocr_text: str,
    holder_name: str | None,
) -> dict[str, Any]:
    """Aplica regras de negócio ao texto OCR de uma transaction_row.

    Regras (em ordem de prioridade):
    1. Contém keyword de transferência → is_transfer = True
    2. Nome do titular aparece no texto → is_intra_user = True
       (transferência entre contas do mesmo usuário)
    3. Caso contrário → transação comum

    Returns:
        dict com: is_transfer, transfer_type, is_intra_user, matched_keywords,
                  holder_name_found.
    """
    norm = _normalize(ocr_text)
    words = set(norm.split())

    # ── Detecta keywords de transferência ────────────────────────────────
    matched: list[str] = []
    transfer_type = None
    for kw in TRANSFER_KEYWORDS:
        # Verifica como palavra exata ou substring significativa
        if kw in norm:
            matched.append(kw)
            if transfer_type is None:
                transfer_type = TRANSFER_TYPE_MAP.get(kw, "TRANSFERÊNCIA")

    is_transfer = len(matched) > 0

    # ── Detecta nome do titular no texto ─────────────────────────────────
    holder_found = False
    if holder_name and is_transfer:
        norm_holder = _normalize(holder_name)
        # Cada palavra do nome deve aparecer no texto (ordem não importa)
        holder_words = set(norm_holder.split())
        # Aceita se pelo menos metade das palavras do nome aparecerem
        if holder_words:
            overlap = holder_words & set(norm.split())
            holder_found = len(overlap) / len(holder_words) >= 0.5

    # ── Classificação final ───────────────────────────────────────────────
    return {
        "is_transfer":       is_transfer,
        "transfer_type":     transfer_type,
        "is_intra_user":     is_transfer and holder_found,
        "matched_keywords":  matched,
        "holder_name_found": holder_found,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Visualização
# ═══════════════════════════════════════════════════════════════════════════════

def draw_detections(
    image: Image.Image,
    detections: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Salva imagem anotada com bounding boxes coloridas por classe."""
    vis = image.copy()
    draw = ImageDraw.Draw(vis, "RGBA")

    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for det in detections:
        cls   = det["class_name"]
        score = det["score"]
        x1, y1, x2, y2 = det["bbox_px"]
        color = CLASS_COLORS.get(cls, (200, 200, 200))

        # Retângulo semi-transparente
        draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=3)
        draw.rectangle([x1, y1, x2, y1 + 24], fill=color + (200,))

        label = f"{cls} {score:.2f}"
        if det.get("is_intra_user"):
            label += " [intra]"
        elif det.get("is_transfer"):
            label += f" [{det.get('transfer_type', 'TRF')}]"

        draw.text((x1 + 4, y1 + 2), label, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(str(out_path))
    log.info("Visualização salva: %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Pipeline principal por página
# ═══════════════════════════════════════════════════════════════════════════════

def process_page(
    page_num:    int,
    image:       Image.Image,
    model:       LayoutLMv3Detector,
    processor:   LayoutLMv3Processor,
    device:      torch.device,
    holder_name: str | None,
    ocr_lang:    str,
    score_thresh: float,
    nms_iou_thresh: float,
    visualize_dir: Path | None,
) -> dict[str, Any]:
    """Processa uma única página e retorna o dict de resultado."""

    # ── 1. Detecção ───────────────────────────────────────────────────────
    detections = predict_page(
        model, processor, image, device, score_thresh, nms_iou_thresh
    )
    log.info("Página %d: %d detecção(ões).", page_num, len(detections))

    # ── 2. OCR + regras de negócio para transaction_row ──────────────────
    has_tesseract = _has_tesseract()
    if not has_tesseract:
        log.warning("Tesseract não encontrado — campo ocr_text estará vazio.")

    for det in detections:
        det["ocr_text"] = ""
        det["is_transfer"]       = False
        det["transfer_type"]     = None
        det["is_intra_user"]     = False
        det["matched_keywords"]  = []
        det["holder_name_found"] = False

        if det["class_name"] == "transaction_row" and has_tesseract:
            text = ocr_region(image, det["bbox_px"], lang=ocr_lang)
            det["ocr_text"] = text
            if text:
                business = classify_transaction(text, holder_name)
                det.update(business)

    # ── 3. Visualização opcional ──────────────────────────────────────────
    if visualize_dir is not None:
        out_img = visualize_dir / f"page_{page_num:03d}.png"
        draw_detections(image, detections, out_img)

    # ── 4. Sumário da página ──────────────────────────────────────────────
    txn_rows    = [d for d in detections if d["class_name"] == "transaction_row"]
    transfers   = [d for d in txn_rows   if d["is_transfer"]]
    intra_user  = [d for d in transfers  if d["is_intra_user"]]

    return {
        "page_number":            page_num,
        "total_detections":       len(detections),
        "transaction_rows":       len(txn_rows),
        "transfers":              len(transfers),
        "intra_user_transfers":   len(intra_user),
        "detections":             detections,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Impressão no terminal
# ═══════════════════════════════════════════════════════════════════════════════

def _separator(char: str = "-", width: int = 72) -> str:
    return char * width


def print_results(result: dict[str, Any]) -> None:
    """Imprime resultados formatados no terminal."""
    print(_separator("="))
    print(f"  Arquivo : {result['pdf']}")
    print(f"  Páginas : {result['total_pages']}")
    s = result["summary"]
    print(f"  Total transações : {s['total_transaction_rows']}")
    print(f"  Transferências   : {s['total_transfers']}")
    print(f"  Intra-usuário    : {s['total_intra_user']}")
    print(_separator("="))

    for page in result["pages"]:
        pn = page["page_number"]
        txn_rows = [d for d in page["detections"] if d["class_name"] == "transaction_row"]
        if not txn_rows:
            continue

        print(f"\n  PÁGINA {pn}  ({len(txn_rows)} transação(ões))")
        print(_separator())

        for det in txn_rows:
            tag = ""
            if det["is_intra_user"]:
                tag = f"[INTRA-USUÁRIO · {det['transfer_type']}]"
            elif det["is_transfer"]:
                tag = f"[TRANSFERÊNCIA · {det['transfer_type']}]"

            score_str = f"{det['score']:.2f}"
            text_str  = (det["ocr_text"] or "(sem OCR)")[:80]

            print(f"  score={score_str}  {tag}")
            print(f"    +- {text_str}")

    print(_separator("="))


# ═══════════════════════════════════════════════════════════════════════════════
# 9. CLI
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="predict",
        description="Detecta regiões de extrato bancário com LayoutLMv3 + OCR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("pdf",           help="Caminho do arquivo PDF.")
    p.add_argument(
        "--checkpoint", "-c",
        default="data/checkpoints/best",
        help="Pasta do checkpoint (deve conter backbone/ e heads.pt).",
    )
    p.add_argument(
        "--config",
        default="config.yaml",
        help="Arquivo de configuração YAML.",
    )
    p.add_argument(
        "--out", "-o",
        default=None,
        help="Caminho do JSON de saída (padrão: <pdf_stem>_predictions.json).",
    )
    p.add_argument(
        "--holder",
        default=None,
        metavar="NOME",
        help="Nome do titular da conta para detecção de transferências intra-usuário.",
    )
    p.add_argument(
        "--score-thresh",
        type=float, default=0.35,
        help="Score mínimo para aceitar uma detecção.",
    )
    p.add_argument(
        "--nms-thresh",
        type=float, default=0.40,
        help="Limiar IoU para Non-Maximum Suppression.",
    )
    p.add_argument(
        "--dpi",
        type=int, default=300,
        help="DPI para conversão do PDF.",
    )
    p.add_argument(
        "--ocr-lang",
        default="por",
        help="Idioma do Tesseract ('por' para português, 'eng' para inglês).",
    )
    p.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Salva imagens anotadas com as detecções.",
    )
    p.add_argument(
        "--visualize-dir",
        default="data/predictions/visualizations",
        help="Pasta para salvar as imagens anotadas (usado com --visualize).",
    )
    p.add_argument(
        "--pages",
        type=int, nargs="+",
        default=None,
        metavar="N",
        help="Processa apenas as páginas informadas (ex: --pages 1 3 5). Padrão: todas.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Nível de log.",
    )
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )

    pdf_path    = Path(args.pdf)
    ckpt_dir    = Path(args.checkpoint)
    out_path    = Path(args.out) if args.out else pdf_path.with_name(pdf_path.stem + "_predictions.json")
    vis_dir     = Path(args.visualize_dir) if args.visualize else None

    if not pdf_path.exists():
        sys.exit(f"Erro: PDF não encontrado: {pdf_path}")

    # ── Carrega configuração ──────────────────────────────────────────────
    config_path = Path(args.config)
    cfg: dict = {}
    if config_path.exists():
        with config_path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    score_thresh    = args.score_thresh
    nms_thresh      = args.nms_thresh
    dpi             = args.dpi or cfg.get("pdf_conversion", {}).get("dpi", 300)

    # ── Dispositivo ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Dispositivo: %s", device)

    # ── Carrega modelo ─────────────────────────────────────────────────────
    model, processor = load_model(ckpt_dir, device)

    # ── Converte PDF → imagens ─────────────────────────────────────────────
    pages_imgs = pdf_to_images(pdf_path, dpi=dpi)
    total_pages = len(pages_imgs)

    # Filtra páginas se --pages foi informado
    page_indices: list[int]
    if args.pages:
        page_indices = [p - 1 for p in args.pages if 1 <= p <= total_pages]
        if not page_indices:
            sys.exit(f"Erro: nenhuma das páginas informadas existe (total: {total_pages}).")
    else:
        page_indices = list(range(total_pages))

    # ── Processa cada página ──────────────────────────────────────────────
    pages_results: list[dict[str, Any]] = []

    for idx in tqdm(page_indices, desc="Processando páginas", unit="pág"):
        page_result = process_page(
            page_num       = idx + 1,
            image          = pages_imgs[idx],
            model          = model,
            processor      = processor,
            device         = device,
            holder_name    = args.holder,
            ocr_lang       = args.ocr_lang,
            score_thresh   = score_thresh,
            nms_iou_thresh = nms_thresh,
            visualize_dir  = vis_dir,
        )
        pages_results.append(page_result)

    # ── Sumário global ────────────────────────────────────────────────────
    summary = {
        "total_transaction_rows": sum(p["transaction_rows"]     for p in pages_results),
        "total_transfers":        sum(p["transfers"]            for p in pages_results),
        "total_intra_user":       sum(p["intra_user_transfers"] for p in pages_results),
        "total_detections":       sum(p["total_detections"]     for p in pages_results),
    }

    result: dict[str, Any] = {
        "pdf":            str(pdf_path),
        "processed_at":   datetime.now(timezone.utc).isoformat(),
        "total_pages":    total_pages,
        "pages_processed": len(page_indices),
        "holder_name":    args.holder,
        "score_threshold": score_thresh,
        "summary":        summary,
        "pages":          pages_results,
    }

    # ── Impressão no terminal ─────────────────────────────────────────────
    print_results(result)

    # ── Salva JSON ────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log.info("Resultados salvos em: %s", out_path)


if __name__ == "__main__":
    main()
