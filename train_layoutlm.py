"""Fine-tuning de LayoutLMv3 para detecção de layout em extratos bancários.

Arquitetura
-----------
Backbone : LayoutLMv3 (modo visual — tokens de texto são dummies)
Head     : Classificação por patch (3 classes + background) + regressão de bbox
           Cada um dos 196 patches (14×14) prevê classe e box (cx,cy,w,h) normalizado
Loss     : Focal loss + L1 + GIoU com Hungarian matching (estilo DETR simplificado)
Métricas : IoU por par e mAP (via torchmetrics.detection.MeanAveragePrecision)

Pré-requisitos
--------------
    pip install transformers torch torchvision torchmetrics albumentations \
                scipy pyyaml pillow tqdm

Dataset esperado (COCO JSON gerado por convert_annotations.py):
    data/splits/train/coco_train.json  +  imagens em  data/images/

Uso
---
    python train_layoutlm.py
    python train_layoutlm.py --config config.yaml --train coco_train.json \\
           --val coco_val.json --images data/images --epochs 20
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import LayoutLMv3Model, LayoutLMv3Processor, get_linear_schedule_with_warmup

log = logging.getLogger(__name__)

# ── Categorias (espelha convert_annotations.py) ───────────────────────────────

CATEGORY_NAMES = ["background", "header", "transaction_row", "transfer_indicator", "saldo"]
# índice 0 = background (não existe no COCO, inserido para o modelo ter classe "nada")
# índices 1–3 mapeiam direto ao category_id do COCO (1, 2, 3)
NUM_CLASSES = len(CATEGORY_NAMES)  # 4 (inclui background)

# LayoutLMv3-base: 224×224, patch 16×16 → 14×14 = 196 patches visuais
NUM_PATCHES = 196
PATCH_GRID  = 14


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class BankStatementDataset(Dataset):
    """Carrega imagens + anotações COCO e aplica augmentation."""

    def __init__(
        self,
        coco_json: str | Path,
        images_dir: str | Path,
        processor: LayoutLMv3Processor,
        augment: bool = False,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.processor  = processor
        self.transform  = _build_augmentation(augment)

        with open(coco_json, encoding="utf-8") as f:
            coco = json.load(f)

        # Índice rápido: image_id → metadados
        self.images: list[dict] = coco["images"]
        self._id_to_meta: dict[int, dict] = {img["id"]: img for img in self.images}

        # Agrupamento de anotações por imagem (filtra imagens sem anotação)
        self._anns: dict[int, list[dict]] = defaultdict(list)
        for ann in coco["annotations"]:
            self._anns[ann["image_id"]].append(ann)

        self.image_ids: list[int] = [
            img["id"] for img in self.images if self._anns[img["id"]]
        ]
        log.info("Dataset: %d imagens com anotações (de %d totais).",
                 len(self.image_ids), len(self.images))

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_id  = self.image_ids[idx]
        meta    = self._id_to_meta[img_id]
        anns    = self._anns[img_id]

        # ── Carrega imagem ────────────────────────────────────────────────
        img_path = self.images_dir / meta["file_name"]
        image    = np.array(Image.open(img_path).convert("RGB"))
        h_orig, w_orig = image.shape[:2]

        # ── Boxes no formato COCO [x,y,w,h] em pixels ────────────────────
        boxes_coco  = np.array([a["bbox"] for a in anns], dtype=np.float32)
        labels_orig = np.array([a["category_id"] for a in anns], dtype=np.int64)

        # ── Augmentation (albumentations lida com coords de bbox) ─────────
        if self.transform is not None:
            result = self.transform(
                image=image,
                bboxes=boxes_coco.tolist(),
                labels=labels_orig.tolist(),
            )
            image       = result["image"]
            boxes_coco  = np.array(result["bboxes"], dtype=np.float32) if result["bboxes"] else np.zeros((0, 4), dtype=np.float32)
            labels_orig = np.array(result["labels"],  dtype=np.int64)  if result["labels"]  else np.zeros(0, dtype=np.int64)
            h_orig, w_orig = image.shape[:2]

        # ── Processa com LayoutLMv3Processor (sem OCR) ────────────────────
        # transformers>=5.x exige boxes explícito quando apply_ocr=False
        pil_image = Image.fromarray(image)
        encoding  = self.processor(
            images=pil_image,
            text=["dummy"],              # uma palavra dummy
            boxes=[[0, 0, 0, 0]],        # uma box dummy (coords em pixels 0-1000)
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        pixel_values = encoding["pixel_values"].squeeze(0)  # (3, 224, 224)

        # ── Converte boxes COCO → normalizado cx,cy,w,h ──────────────────
        # COCO: [x_min, y_min, w, h] em pixels
        # Alvo:  [cx, cy, w, h] normalizados para [0,1]
        if len(boxes_coco) > 0:
            cx = (boxes_coco[:, 0] + boxes_coco[:, 2] / 2) / w_orig
            cy = (boxes_coco[:, 1] + boxes_coco[:, 3] / 2) / h_orig
            bw =  boxes_coco[:, 2] / w_orig
            bh =  boxes_coco[:, 3] / h_orig
            target_boxes = np.stack([cx, cy, bw, bh], axis=1).clip(0.0, 1.0)
        else:
            target_boxes = np.zeros((0, 4), dtype=np.float32)

        return {
            "pixel_values": pixel_values,
            "target_boxes":  torch.as_tensor(target_boxes,  dtype=torch.float32),
            "target_labels": torch.as_tensor(labels_orig,   dtype=torch.int64),
            "image_id":      img_id,
        }


def _build_augmentation(active: bool):
    """Pipeline albumentations — rotação pequena + brilho/contraste."""
    try:
        import albumentations as A
    except ImportError:
        log.warning("albumentations não instalado; augmentation desativada. "
                    "pip install albumentations")
        return None

    if not active:
        return None

    return A.Compose(
        [
            # Rotação leve (documentos não podem estar muito inclinados)
            A.Rotate(limit=3, border_mode=0, p=0.5),
            # Variações fotométricas comuns em scans e fotos de tela
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, p=0.3),
            A.GaussNoise(var_limit=(10.0, 40.0), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            # Degradações de digitalização
            A.ImageCompression(quality_lower=80, quality_upper=100, p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["labels"],
            min_visibility=0.3,   # descarta boxes que ficaram <30% visíveis após crop
        ),
    )


def collate_fn(batch: list[dict]) -> dict[str, Any]:
    """Agrupa amostras em batch — boxes/labels ficam como lista (tamanhos variáveis)."""
    return {
        "pixel_values":  torch.stack([b["pixel_values"]  for b in batch]),
        "target_boxes":  [b["target_boxes"]  for b in batch],
        "target_labels": [b["target_labels"] for b in batch],
        "image_ids":     [b["image_id"]      for b in batch],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Modelo
# ═══════════════════════════════════════════════════════════════════════════════

class LayoutLMv3Detector(nn.Module):
    """LayoutLMv3 backbone + cabeças de classificação e regressão por patch."""

    def __init__(self, model_name: str, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.backbone   = LayoutLMv3Model.from_pretrained(model_name)
        hidden          = self.backbone.config.hidden_size  # 768 (base) / 1024 (large)
        self.num_classes = num_classes

        self.class_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, num_classes),
        )
        self.bbox_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 4),
            nn.Sigmoid(),  # cx,cy,w,h em [0,1]
        )

        # Inicializa cabeças com ganho menor para estabilidade inicial
        for head in (self.class_head, self.bbox_head):
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        pixel_values: torch.Tensor,         # (B, 3, 224, 224)
        input_ids:       torch.Tensor | None = None,
        attention_mask:  torch.Tensor | None = None,
        bbox:            torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B      = pixel_values.shape[0]
        device = pixel_values.device

        # Tokens de texto dummy — apenas [CLS] — para modo visual-only
        if input_ids is None:
            input_ids      = torch.zeros(B, 1, dtype=torch.long,  device=device)
            attention_mask = torch.ones( B, 1, dtype=torch.long,  device=device)
            bbox           = torch.zeros(B, 1, 4, dtype=torch.long, device=device)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
        )

        # LayoutLMv3 coloca os tokens visuais nos últimos NUM_PATCHES posições
        # last_hidden_state: (B, text_len + NUM_PATCHES, hidden)
        visual = outputs.last_hidden_state[:, -NUM_PATCHES:, :]  # (B, 196, H)

        logits = self.class_head(visual)  # (B, 196, num_classes)
        boxes  = self.bbox_head(visual)   # (B, 196, 4)
        return logits, boxes


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Funções de loss e matching
# ═══════════════════════════════════════════════════════════════════════════════

def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """[cx,cy,w,h] → [x1,y1,x2,y2] (todos normalizados em [0,1])."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """IoU par-a-par entre dois conjuntos de boxes em formato xyxy.

    Args:
        boxes_a: (N, 4)
        boxes_b: (M, 4)
    Returns:
        (N, M) tensor de IoUs
    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(0)

    inter_x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


def generalized_box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """GIoU diagonal (N,) para N pares correspondentes (N, 4) xyxy."""
    assert boxes_a.shape == boxes_b.shape

    iou = box_iou(boxes_a, boxes_b).diagonal()

    enc_x1 = torch.min(boxes_a[:, 0], boxes_b[:, 0])
    enc_y1 = torch.min(boxes_a[:, 1], boxes_b[:, 1])
    enc_x2 = torch.max(boxes_a[:, 2], boxes_b[:, 2])
    enc_y2 = torch.max(boxes_a[:, 3], boxes_b[:, 3])
    enc_area = (enc_x2 - enc_x1).clamp(0) * (enc_y2 - enc_y1).clamp(0)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(0)
    union  = area_a + area_b - iou * (area_a + area_b - area_a * 0)  # simplified
    union  = (area_a + area_b - box_iou(boxes_a, boxes_b).diagonal() * (area_a + area_b)).clamp(min=1e-6)

    # GIoU = IoU - (enc - union) / enc
    return iou - (enc_area - union) / enc_area.clamp(min=1e-6)


def sigmoid_focal_loss(
    logits: torch.Tensor,   # (N, num_classes)
    targets: torch.Tensor,  # (N,) int64  — índices de classe
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss multiclasse via one-hot encoding."""
    num_classes = logits.shape[-1]
    one_hot = F.one_hot(targets, num_classes=num_classes).float()
    p       = torch.sigmoid(logits)
    ce      = F.binary_cross_entropy_with_logits(logits, one_hot, reduction="none")
    p_t     = p * one_hot + (1 - p) * (1 - one_hot)
    alpha_t = alpha * one_hot + (1 - alpha) * (1 - one_hot)
    loss    = alpha_t * (1 - p_t) ** gamma * ce
    return loss.sum(-1)  # (N,)


def hungarian_matching(
    pred_logits: torch.Tensor,  # (num_preds, num_classes)
    pred_boxes:  torch.Tensor,  # (num_preds, 4) cxcywh
    gt_labels:   torch.Tensor,  # (num_gt,) int64
    gt_boxes:    torch.Tensor,  # (num_gt, 4) cxcywh
    cost_class: float = 1.0,
    cost_l1:    float = 5.0,
    cost_giou:  float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Atribuição ótima predições → ground truth (scipy linear_sum_assignment).

    Returns:
        pred_indices, gt_indices — arrays 1-D de mesmo tamanho.
    """
    from scipy.optimize import linear_sum_assignment

    with torch.no_grad():
        num_gt = gt_labels.shape[0]
        if num_gt == 0:
            empty = torch.zeros(0, dtype=torch.long)
            return empty, empty

        probs = pred_logits.softmax(-1)                # (P, C)
        # Custo de classificação: probabilidade negativa da classe correta
        cost_cls = -probs[:, gt_labels]                # (P, G)

        pred_xy = cxcywh_to_xyxy(pred_boxes)
        gt_xy   = cxcywh_to_xyxy(gt_boxes)

        # Custo L1
        cost_box = torch.cdist(pred_boxes, gt_boxes, p=1)  # (P, G)

        # Custo GIoU: 1 - GIoU de cada par (P, G)
        P, G = pred_boxes.shape[0], gt_boxes.shape[0]
        pred_xy_rep = pred_xy.unsqueeze(1).expand(P, G, 4).reshape(-1, 4)
        gt_xy_rep   = gt_xy.unsqueeze(0).expand(P, G, 4).reshape(-1, 4)
        giou_flat   = generalized_box_iou(pred_xy_rep, gt_xy_rep)  # (P*G,)
        cost_g      = -giou_flat.reshape(P, G)

        C = cost_class * cost_cls + cost_l1 * cost_box + cost_giou * cost_g
        pred_idx, gt_idx = linear_sum_assignment(C.cpu().numpy())

    return (
        torch.as_tensor(pred_idx, dtype=torch.long, device=pred_logits.device),
        torch.as_tensor(gt_idx,   dtype=torch.long, device=pred_logits.device),
    )


def compute_detection_loss(
    logits_batch: torch.Tensor,  # (B, 196, num_classes)
    boxes_batch:  torch.Tensor,  # (B, 196, 4)
    target_labels: list[torch.Tensor],
    target_boxes:  list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Loss por imagem com Hungarian matching; retorna dict com componentes."""
    loss_cls_total  = torch.tensor(0.0, device=logits_batch.device)
    loss_l1_total   = torch.tensor(0.0, device=logits_batch.device)
    loss_giou_total = torch.tensor(0.0, device=logits_batch.device)
    B = logits_batch.shape[0]

    for i in range(B):
        logits   = logits_batch[i]   # (196, C)
        boxes    = boxes_batch[i]    # (196, 4)
        gt_lbls  = target_labels[i].to(logits.device)
        gt_boxes = target_boxes[i].to(logits.device)

        pred_idx, gt_idx = hungarian_matching(logits, boxes, gt_lbls, gt_boxes)

        # ── Classificação ──────────────────────────────────────────────────
        # Patches não atribuídos → background (índice 0)
        all_labels = torch.zeros(NUM_PATCHES, dtype=torch.long, device=logits.device)
        if len(pred_idx):
            all_labels[pred_idx] = gt_lbls[gt_idx]
        loss_cls_total += sigmoid_focal_loss(logits, all_labels).mean()

        # ── Regressão (apenas patches atribuídos) ──────────────────────────
        if len(pred_idx) == 0:
            continue

        matched_pred = boxes[pred_idx]     # (K, 4)
        matched_gt   = gt_boxes[gt_idx]    # (K, 4)

        loss_l1_total   += F.l1_loss(matched_pred, matched_gt)
        loss_giou_total += (1 - generalized_box_iou(
            cxcywh_to_xyxy(matched_pred),
            cxcywh_to_xyxy(matched_gt),
        )).mean()

    return {
        "loss_cls":  loss_cls_total  / B,
        "loss_l1":   loss_l1_total   / B,
        "loss_giou": loss_giou_total / B,
        "loss":      loss_cls_total / B + 5.0 * loss_l1_total / B + 2.0 * loss_giou_total / B,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Métricas
# ═══════════════════════════════════════════════════════════════════════════════

def compute_iou_single(box_pred: torch.Tensor, box_gt: torch.Tensor) -> float:
    """IoU entre dois boxes xyxy individuais (tensores 1-D de tamanho 4)."""
    iou = box_iou(box_pred.unsqueeze(0), box_gt.unsqueeze(0))
    return iou.item()


@torch.no_grad()
def collect_predictions(
    logits_batch: torch.Tensor,
    boxes_batch:  torch.Tensor,
    score_thresh: float = 0.3,
) -> list[dict[str, torch.Tensor]]:
    """Converte saída do modelo em lista de dicts no formato torchmetrics.

    Cada dict: {"boxes": (N,4) xyxy float, "scores": (N,), "labels": (N,) int}
    """
    out = []
    B   = logits_batch.shape[0]
    for i in range(B):
        scores_all, labels = logits_batch[i].softmax(-1).max(-1)  # (196,), (196,)

        # Remove background (label 0) e filtra por confiança
        keep = (labels > 0) & (scores_all >= score_thresh)
        boxes_xyxy = cxcywh_to_xyxy(boxes_batch[i])

        out.append({
            "boxes":  boxes_xyxy[keep],
            "scores": scores_all[keep],
            "labels": labels[keep],
        })
    return out


@torch.no_grad()
def collect_targets(
    target_labels: list[torch.Tensor],
    target_boxes:  list[torch.Tensor],
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """Converte ground truth para o formato torchmetrics."""
    out = []
    for lbls, boxes in zip(target_labels, target_boxes):
        boxes_xyxy = cxcywh_to_xyxy(boxes.to(device))
        out.append({"boxes": boxes_xyxy, "labels": lbls.to(device)})
    return out


def build_map_metric(device: torch.device):
    """Instancia torchmetrics MeanAveragePrecision."""
    try:
        from torchmetrics.detection import MeanAveragePrecision
        metric = MeanAveragePrecision(
            iou_type="bbox",
            iou_thresholds=[0.50, 0.75],   # mAP@50 e mAP@75
            class_metrics=True,
        )
        return metric.to(device)
    except ImportError:
        log.warning("torchmetrics não instalado. pip install torchmetrics[detection]")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Loop de treino / avaliação
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model:      LayoutLMv3Detector,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    scheduler,
    scaler:     torch.cuda.amp.GradScaler | None,
    device:     torch.device,
    grad_clip:  float,
    epoch:      int,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = defaultdict(float)
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)

    for batch in pbar:
        pv     = batch["pixel_values"].to(device)
        t_lbls = batch["target_labels"]
        t_boxes= batch["target_boxes"]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logits, boxes = model(pv)
            losses = compute_detection_loss(logits, boxes, t_lbls, t_boxes)

        loss = losses["loss"]
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()

        for k, v in losses.items():
            totals[k] += v.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate(
    model:       LayoutLMv3Detector,
    loader:      DataLoader,
    device:      torch.device,
    score_thresh: float = 0.3,
    epoch:        int   = 0,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    map_metric = build_map_metric(device)
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False)

    for batch in pbar:
        pv     = batch["pixel_values"].to(device)
        t_lbls = batch["target_labels"]
        t_boxes= batch["target_boxes"]

        logits, boxes = model(pv)
        losses = compute_detection_loss(logits, boxes, t_lbls, t_boxes)
        for k, v in losses.items():
            totals[k] += v.item()

        if map_metric is not None:
            preds   = collect_predictions(logits, boxes, score_thresh)
            targets = collect_targets(t_lbls, t_boxes, device)
            map_metric.update(preds, targets)

        pbar.set_postfix(loss=f"{losses['loss'].item():.4f}")

    n = len(loader)
    metrics = {k: v / n for k, v in totals.items()}

    if map_metric is not None:
        result = map_metric.compute()
        metrics["mAP@50"]  = result.get("map_50",  torch.tensor(0.0)).item()
        metrics["mAP@75"]  = result.get("map_75",  torch.tensor(0.0)).item()
        metrics["mAP"]     = result.get("map",     torch.tensor(0.0)).item()

        # mAP por classe (índices 1-3 = header, transaction_row, transfer_indicator)
        per_class = result.get("map_per_class", [])
        for j, name in enumerate(CATEGORY_NAMES[1:], start=0):
            val = per_class[j].item() if j < len(per_class) else 0.0
            metrics[f"AP_{name}"] = val

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Checkpoints
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model:      LayoutLMv3Detector,
    processor:  LayoutLMv3Processor,
    optimizer:  torch.optim.Optimizer,
    epoch:      int,
    metrics:    dict[str, float],
    ckpt_dir:   Path,
    is_best:    bool = False,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = f"epoch_{epoch:03d}"

    # Backbone via HuggingFace save_pretrained (pesos + config)
    backbone_dir = ckpt_dir / tag / "backbone"
    model.backbone.save_pretrained(str(backbone_dir))
    processor.save_pretrained(str(backbone_dir))

    # Cabeças de detecção + estado do optimizer como .pt
    torch.save(
        {
            "epoch":          epoch,
            "class_head":     model.class_head.state_dict(),
            "bbox_head":      model.bbox_head.state_dict(),
            "optimizer":      optimizer.state_dict(),
            "metrics":        metrics,
        },
        ckpt_dir / tag / "heads.pt",
    )

    if is_best:
        best_dir = ckpt_dir / "best"
        import shutil
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(ckpt_dir / tag, best_dir)
        log.info(">>> Novo melhor modelo salvo em %s (mAP@50=%.4f)",
                 best_dir, metrics.get("mAP@50", 0))

    log.info("Checkpoint salvo: %s", ckpt_dir / tag)


def load_checkpoint(model: LayoutLMv3Detector, ckpt_path: Path, device: torch.device) -> int:
    """Carrega cabeças de detecção de um checkpoint .pt. Retorna época salva."""
    heads = torch.load(ckpt_path / "heads.pt", map_location=device)
    model.class_head.load_state_dict(heads["class_head"])
    model.bbox_head.load_state_dict(heads["bbox_head"])
    log.info("Checkpoint carregado: epoch %d", heads["epoch"])
    return heads["epoch"]


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Configuração e utilidades
# ═══════════════════════════════════════════════════════════════════════════════

def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _log_metrics(tag: str, metrics: dict[str, float], epoch: int) -> None:
    vals = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    log.info("[%s] epoch %03d  %s", tag, epoch, vals)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Main
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_layoutlm",
        description="Fine-tuning LayoutLMv3 para detecção de layout bancário.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",     default="config.yaml",            help="Arquivo YAML de configuração.")
    p.add_argument("--train",      default=None,                     help="COCO JSON de treino (sobrescreve config).")
    p.add_argument("--val",        default=None,                     help="COCO JSON de validação.")
    p.add_argument("--images",     default=None,                     help="Pasta com as imagens PNG.")
    p.add_argument("--model",      default=None,                     help="HF Hub ID ou path do backbone.")
    p.add_argument("--ckpt-dir",   default=None,                     help="Pasta de checkpoints.")
    p.add_argument("--resume",     default=None,                     help="Retoma de checkpoint (pasta epoch_NNN).")
    p.add_argument("--epochs",     type=int,    default=None,        help="Número de épocas.")
    p.add_argument("--batch-size", type=int,    default=None,        help="Batch size.")
    p.add_argument("--lr",         type=float,  default=None,        help="Learning rate.")
    p.add_argument("--no-aug",     action="store_true",              help="Desativa data augmentation.")
    p.add_argument("--fp16",       action="store_true",              help="Ativa mixed precision FP16.")
    p.add_argument("--score-thresh", type=float, default=0.3,        help="Limiar de score para métricas.")
    p.add_argument("--log-level",  default="INFO",                   help="Nível de log (DEBUG/INFO/WARNING).")
    return p


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )

    # ── Configuração ──────────────────────────────────────────────────────────
    root = Path(__file__).parent
    cfg  = load_yaml(root / args.config)

    model_name  = args.model      or cfg["model"]["name"]
    ckpt_dir    = Path(args.ckpt_dir  or root / cfg["paths"]["checkpoints"])
    images_dir  = Path(args.images    or root / cfg["paths"]["images"])
    num_epochs  = args.epochs     or cfg["training"]["num_epochs"]
    batch_size  = args.batch_size or cfg["training"]["batch_size"]
    lr          = args.lr         or cfg["training"]["learning_rate"]
    weight_decay= cfg["training"]["weight_decay"]
    warmup_ratio= cfg["training"]["warmup_ratio"]
    grad_clip   = cfg["training"]["gradient_clip"]
    seed        = cfg["training"]["seed"]
    use_fp16    = args.fp16 or cfg["training"].get("fp16", False)
    score_thresh= args.score_thresh

    ann_dir = root / cfg["paths"]["annotations"]
    train_json = Path(args.train) if args.train else ann_dir / "coco_train.json"
    val_json   = Path(args.val)   if args.val   else ann_dir / "coco_val.json"

    for p, label in [(train_json, "--train"), (val_json, "--val"), (images_dir, "--images")]:
        if not p.exists():
            log.error("%s não encontrado: %s", label, p)
            raise SystemExit(1)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Dispositivo: %s  |  FP16: %s", device, use_fp16)

    # ── Processor e Datasets ──────────────────────────────────────────────────
    log.info("Carregando processor: %s", model_name)
    processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)

    train_ds = BankStatementDataset(train_json, images_dir, processor, augment=not args.no_aug)
    val_ds   = BankStatementDataset(val_json,   images_dir, processor, augment=False)

    # num_workers=0 evita problemas de pickle no Windows com Python 3.14+
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["eval_batch_size"], shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=False,
    )

    # ── Modelo ────────────────────────────────────────────────────────────────
    log.info("Carregando backbone: %s", model_name)
    model = LayoutLMv3Detector(model_name, num_classes=NUM_CLASSES).to(device)

    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(model, Path(args.resume), device) + 1

    # ── Optimizer e Scheduler ─────────────────────────────────────────────────
    # Backbone com LR reduzido; cabeças com LR completo
    backbone_params = list(model.backbone.parameters())
    head_params     = list(model.class_head.parameters()) + list(model.bbox_head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params,     "lr": lr},
    ], weight_decay=weight_decay)

    total_steps   = len(train_loader) * num_epochs
    warmup_steps  = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler() if use_fp16 and device.type == "cuda" else None

    # ── Loop principal ────────────────────────────────────────────────────────
    best_map50 = 0.0
    log_path   = root / cfg["paths"]["logs"] / "train_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, grad_clip, epoch
        )
        val_metrics = evaluate(model, val_loader, device, score_thresh, epoch)

        elapsed = time.time() - t0
        _log_metrics("TRAIN", train_metrics, epoch)
        _log_metrics("VAL  ", val_metrics,   epoch)
        log.info("Época %03d concluída em %.1fs", epoch, elapsed)

        # Salva log em JSONL
        with log_path.open("a", encoding="utf-8") as lf:
            json.dump({"epoch": epoch, "train": train_metrics, "val": val_metrics,
                       "elapsed": elapsed}, lf)
            lf.write("\n")

        is_best = val_metrics.get("mAP@50", 0.0) > best_map50
        if is_best:
            best_map50 = val_metrics.get("mAP@50", 0.0)

        save_checkpoint(model, processor, optimizer, epoch, val_metrics, ckpt_dir, is_best)

    log.info("Treinamento concluído. Melhor mAP@50 = %.4f", best_map50)
    log.info("Melhor modelo em: %s", ckpt_dir / "best")


if __name__ == "__main__":
    main()
