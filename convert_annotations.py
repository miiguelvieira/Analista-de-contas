"""Converte exportações do Label Studio (bounding boxes) para o formato COCO.

Label Studio exporta bounding boxes em percentual da imagem (x, y, width, height
todos entre 0–100). O COCO usa pixels absolutos com origem no canto superior esquerdo
no formato [x_min, y_min, width, height].

Categorias suportadas: header | transaction_row | transfer_indicator

Uso:
    python convert_annotations.py input.json output.json
    python convert_annotations.py input.json output.json --split 0.8 0.1 0.1
    python convert_annotations.py input.json output.json --images-dir data/images --strict
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ── Categorias do projeto ─────────────────────────────────────────────────────

CATEGORIES: list[dict] = [
    {"id": 1, "name": "header",              "supercategory": "layout"},
    {"id": 2, "name": "transaction_row",     "supercategory": "layout"},
    {"id": 3, "name": "transfer_indicator",  "supercategory": "layout"},
]
CATEGORY_NAME_TO_ID: dict[str, int] = {c["name"]: c["id"] for c in CATEGORIES}

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


# ── Tipos de erro de validação ────────────────────────────────────────────────

class ValidationError(ValueError):
    """Campo obrigatório ausente ou inválido em um item do Label Studio."""


# ── Dataclasses intermediários ────────────────────────────────────────────────

@dataclass
class BBox:
    x: float       # pixels absolutos, canto superior esquerdo
    y: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def as_list(self) -> list[float]:
        return [round(self.x, 2), round(self.y, 2),
                round(self.width, 2), round(self.height, 2)]


@dataclass
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int


@dataclass
class CocoAnnotation:
    id: int
    image_id: int
    category_id: int
    bbox: BBox
    iscrowd: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "image_id": self.image_id,
            "category_id": self.category_id,
            "bbox": self.bbox.as_list,
            "area": round(self.bbox.area, 2),
            "iscrowd": self.iscrowd,
        }


# ── Validação ─────────────────────────────────────────────────────────────────

def _require(obj: dict, *keys: str, context: str = "") -> None:
    """Levanta ValidationError se alguma chave obrigatória estiver ausente."""
    for key in keys:
        if key not in obj or obj[key] is None:
            raise ValidationError(
                f"Campo obrigatório ausente: '{key}'"
                + (f" [{context}]" if context else "")
            )


def _require_positive(value: float, name: str, context: str = "") -> None:
    if value <= 0:
        raise ValidationError(
            f"'{name}' deve ser > 0, recebido {value}"
            + (f" [{context}]" if context else "")
        )


def _validate_label_studio_item(item: dict, idx: int) -> None:
    """Valida a estrutura de um item de nível superior do Label Studio."""
    ctx = f"item #{idx}"
    _require(item, "id", "data", context=ctx)
    _require(item["data"], "image", context=f"{ctx}.data")

    annotations = item.get("annotations") or []
    if not annotations:
        raise ValidationError(f"Nenhuma annotation encontrada [{ctx}]")

    for ann_idx, ann in enumerate(annotations):
        ann_ctx = f"{ctx}.annotations[{ann_idx}]"
        _require(ann, "result", context=ann_ctx)

        for res_idx, result in enumerate(ann.get("result") or []):
            res_ctx = f"{ann_ctx}.result[{res_idx}]"
            _require(result, "type", "value", "original_width", "original_height",
                     context=res_ctx)

            if result["type"] != "rectanglelabels":
                continue  # outros tipos são ignorados, não é erro

            value = result["value"]
            val_ctx = f"{res_ctx}.value"
            _require(value, "x", "y", "width", "height", "rectanglelabels",
                     context=val_ctx)

            for coord in ("x", "y", "width", "height"):
                try:
                    float(value[coord])
                except (TypeError, ValueError):
                    raise ValidationError(
                        f"'{coord}' não é numérico: {value[coord]!r} [{val_ctx}]"
                    )

            _require_positive(result["original_width"],  "original_width",  res_ctx)
            _require_positive(result["original_height"], "original_height", res_ctx)

            labels = value["rectanglelabels"]
            if not labels:
                raise ValidationError(f"'rectanglelabels' está vazio [{val_ctx}]")
            for label in labels:
                if label not in CATEGORY_NAME_TO_ID:
                    raise ValidationError(
                        f"Categoria desconhecida: '{label}' [{val_ctx}]. "
                        f"Categorias válidas: {list(CATEGORY_NAME_TO_ID)}"
                    )


# ── Conversão ─────────────────────────────────────────────────────────────────

def _pct_to_abs(pct: float, dimension: float) -> float:
    """Converte coordenada percentual (0–100) para pixel absoluto."""
    return pct / 100.0 * dimension


def _parse_file_name(image_url: str) -> str:
    """Extrai apenas o nome do arquivo de uma URL ou caminho do Label Studio."""
    # Suporta: /data/upload/1/image.png  |  http://...  |  C:\...\image.png
    return Path(image_url.split("?")[0]).name  # ignora query strings


def convert(
    ls_data: list[dict],
    strict: bool = False,
    images_dir: Path | None = None,
) -> dict:
    """Converte lista de itens do Label Studio para um dicionário COCO.

    Args:
        ls_data:    Lista de tarefas exportadas pelo Label Studio.
        strict:     Se True, abandona toda a conversão ao primeiro erro de validação.
                    Se False, pula itens inválidos e continua.
        images_dir: Se informado, verifica se cada imagem existe nesta pasta.

    Returns:
        Dicionário COCO completo com 'images', 'annotations', 'categories', 'info'.
    """
    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    ann_id = 1
    skipped_items = 0
    skipped_boxes = 0

    # Rastreia file_names já vistos para evitar imagens duplicadas
    seen_file_names: dict[str, int] = {}  # file_name → image_id

    for idx, item in enumerate(ls_data):
        # ── Validação do item ──────────────────────────────────────────────
        try:
            _validate_label_studio_item(item, idx)
        except ValidationError as exc:
            if strict:
                raise
            log.warning("Item ignorado — %s", exc)
            skipped_items += 1
            continue

        # ── Imagem ────────────────────────────────────────────────────────
        image_url: str = item["data"]["image"]
        file_name = _parse_file_name(image_url)

        if file_name in seen_file_names:
            image_id = seen_file_names[file_name]
        else:
            image_id = item["id"]
            # Dimensões vêm do primeiro result com original_width/height
            orig_w, orig_h = _resolve_image_size(item, file_name, images_dir)

            coco_images.append({
                "id": image_id,
                "file_name": file_name,
                "width": orig_w,
                "height": orig_h,
            })
            seen_file_names[file_name] = image_id

            if images_dir is not None:
                img_path = images_dir / file_name
                if not img_path.exists():
                    log.warning("Imagem não encontrada em disco: %s", img_path)

        # ── Bounding boxes ─────────────────────────────────────────────────
        for ann in item.get("annotations") or []:
            # Usa apenas a primeira annotation não descartada (skipped=False)
            if ann.get("was_cancelled") or ann.get("skipped"):
                continue

            for result in ann.get("result") or []:
                if result.get("type") != "rectanglelabels":
                    continue

                value = result["value"]
                orig_w = result["original_width"]
                orig_h = result["original_height"]

                try:
                    bbox = _convert_bbox(value, orig_w, orig_h)
                except ValidationError as exc:
                    if strict:
                        raise
                    log.warning("Bounding box ignorado — %s", exc)
                    skipped_boxes += 1
                    continue

                for label in value["rectanglelabels"]:
                    cat_id = CATEGORY_NAME_TO_ID.get(label)
                    if cat_id is None:
                        skipped_boxes += 1
                        continue

                    ann_obj = CocoAnnotation(
                        id=ann_id,
                        image_id=image_id,
                        category_id=cat_id,
                        bbox=bbox,
                    )
                    coco_annotations.append(ann_obj.to_dict())
                    ann_id += 1

            break  # uma annotation por imagem é suficiente para treino

    if skipped_items:
        log.warning("%d item(ns) ignorado(s) por erros de validação.", skipped_items)
    if skipped_boxes:
        log.warning("%d bounding box(es) ignorado(s).", skipped_boxes)

    return {
        "info": _build_info(),
        "licenses": [],
        "categories": CATEGORIES,
        "images": coco_images,
        "annotations": coco_annotations,
    }


def _convert_bbox(value: dict, orig_w: float, orig_h: float) -> BBox:
    """Converte bbox de percentual LS para pixels absolutos COCO."""
    x_abs = _pct_to_abs(value["x"], orig_w)
    y_abs = _pct_to_abs(value["y"], orig_h)
    w_abs = _pct_to_abs(value["width"],  orig_w)
    h_abs = _pct_to_abs(value["height"], orig_h)

    # Garante que bbox não ultrapassa os limites da imagem
    x_abs = max(0.0, x_abs)
    y_abs = max(0.0, y_abs)
    w_abs = min(w_abs, orig_w - x_abs)
    h_abs = min(h_abs, orig_h - y_abs)

    if w_abs <= 0 or h_abs <= 0:
        raise ValidationError(
            f"Bounding box com dimensão zero após clamp: "
            f"w={w_abs:.2f}, h={h_abs:.2f}"
        )

    return BBox(x_abs, y_abs, w_abs, h_abs)


def _resolve_image_size(item: dict, file_name: str, images_dir: Path | None) -> tuple[int, int]:
    """Resolve as dimensões da imagem a partir do Label Studio ou do disco."""
    for ann in item.get("annotations") or []:
        for result in ann.get("result") or []:
            w = result.get("original_width")
            h = result.get("original_height")
            if w and h:
                return int(w), int(h)

    # Fallback: lê do disco se o arquivo existir
    if images_dir is not None:
        img_path = images_dir / file_name
        if img_path.exists():
            try:
                from PIL import Image as PILImage
                with PILImage.open(img_path) as img:
                    return img.size  # (width, height)
            except Exception:
                pass

    raise ValidationError(
        f"Não foi possível determinar as dimensões de '{file_name}'. "
        "Use --images-dir para ler do disco."
    )


def _build_info() -> dict:
    return {
        "description": "Leitor de Conta — Anotações de Layout Bancário",
        "version": "1.0",
        "year": datetime.now(timezone.utc).year,
        "date_created": datetime.now(timezone.utc).strftime("%Y/%m/%d"),
        "contributor": "",
        "url": "",
    }


# ── Split train/val/test ──────────────────────────────────────────────────────

def split_coco(
    coco: dict,
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, dict]:
    """Divide um COCO em train/val/test sem misturar imagens entre splits.

    Returns:
        Dict {"train": coco_dict, "val": coco_dict, "test": coco_dict}
    """
    train_r, val_r, test_r = ratios
    if abs(train_r + val_r + test_r - 1.0) > 1e-6:
        raise ValueError(f"A soma dos ratios deve ser 1.0, recebido {train_r + val_r + test_r}")

    images = coco["images"][:]
    random.seed(seed)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)

    splits_images = {
        "train": images[:n_train],
        "val":   images[n_train : n_train + n_val],
        "test":  images[n_train + n_val :],
    }

    # Índice inverso: image_id → annotation(s)
    from collections import defaultdict
    ann_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    result: dict[str, dict] = {}
    for split_name, split_imgs in splits_images.items():
        img_ids = {img["id"] for img in split_imgs}
        result[split_name] = {
            "info": coco["info"],
            "licenses": coco["licenses"],
            "categories": coco["categories"],
            "images": split_imgs,
            "annotations": [
                ann for img_id in img_ids for ann in ann_by_image[img_id]
            ],
        }
        log.info(
            "Split %-5s → %3d imagem(ns), %4d anotação(ões)",
            split_name, len(split_imgs), len(result[split_name]["annotations"]),
        )

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="convert_annotations",
        description=(
            "Converte exportação do Label Studio (bounding boxes) para COCO JSON.\n"
            "Categorias: header | transaction_row | transfer_indicator"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemplos:\n"
            "  python convert_annotations.py ls_export.json coco.json\n"
            "  python convert_annotations.py ls_export.json coco.json --split 0.8 0.1 0.1\n"
            "  python convert_annotations.py ls_export.json coco.json --strict --images-dir data/images"
        ),
    )
    parser.add_argument("input",  metavar="INPUT",  help="JSON exportado pelo Label Studio.")
    parser.add_argument("output", metavar="OUTPUT", help="Caminho do COCO JSON de saída.")
    parser.add_argument(
        "--split", nargs=3, type=float, metavar=("TRAIN", "VAL", "TEST"),
        help="Divide em train/val/test com os ratios informados (ex: 0.8 0.1 0.1). "
             "Gera arquivos _train.json, _val.json, _test.json ao lado de OUTPUT.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semente para o shuffle do split (padrão: 42).",
    )
    parser.add_argument(
        "--images-dir", metavar="DIR",
        help="Pasta com as imagens — usada para verificar existência e ler dimensões como fallback.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Abandona a conversão ao primeiro erro de validação (padrão: pula itens inválidos).",
    )
    parser.add_argument(
        "--indent", type=int, default=2,
        help="Indentação do JSON de saída (padrão: 2; use 0 para arquivo compacto).",
    )
    return parser


def _write_json(path: Path, data: dict, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent or None)
    log.info("Salvo: %s", path)


def main() -> None:
    args = build_parser().parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    images_dir  = Path(args.images_dir) if args.images_dir else None

    # ── Leitura do arquivo de entrada ─────────────────────────────────────
    if not input_path.exists():
        log.error("Arquivo não encontrado: %s", input_path)
        sys.exit(1)

    try:
        with input_path.open(encoding="utf-8") as f:
            ls_data = json.load(f)
    except json.JSONDecodeError as exc:
        log.error("JSON inválido em '%s': %s", input_path, exc)
        sys.exit(1)

    if not isinstance(ls_data, list):
        log.error("O JSON do Label Studio deve ser uma lista de tarefas.")
        sys.exit(1)

    log.info("Carregados %d item(ns) do Label Studio.", len(ls_data))

    # ── Conversão ──────────────────────────────────────────────────────────
    try:
        coco = convert(ls_data, strict=args.strict, images_dir=images_dir)
    except ValidationError as exc:
        log.error("Erro de validação (modo --strict): %s", exc)
        sys.exit(1)

    log.info(
        "Conversão concluída: %d imagem(ns), %d anotação(ões).",
        len(coco["images"]), len(coco["annotations"]),
    )

    # ── Distribuição por categoria ─────────────────────────────────────────
    from collections import Counter
    cat_counts = Counter(a["category_id"] for a in coco["annotations"])
    id_to_name = {c["id"]: c["name"] for c in CATEGORIES}
    for cat_id, count in sorted(cat_counts.items()):
        log.info("  %-22s %d", id_to_name[cat_id] + ":", count)

    # ── Saída ─────────────────────────────────────────────────────────────
    if args.split:
        ratios = tuple(args.split)
        try:
            splits = split_coco(coco, ratios, seed=args.seed)
        except ValueError as exc:
            log.error("%s", exc)
            sys.exit(1)

        stem   = output_path.stem
        parent = output_path.parent
        for split_name, split_data in splits.items():
            _write_json(parent / f"{stem}_{split_name}.json", split_data, args.indent)
    else:
        _write_json(output_path, coco, args.indent)


if __name__ == "__main__":
    main()
