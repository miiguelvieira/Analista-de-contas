# SKILLS.md — Referência de Comandos do Projeto

## Scripts disponíveis

---

### `pdf_to_png.py` — Converter PDFs em imagens

```bash
# Pasta inteira
python pdf_to_png.py data/pdfs/ -o data/images/

# PDF único com DPI personalizado
python pdf_to_png.py extrato.pdf -o data/images/ --dpi 150

# Com múltiplos workers (paralelo)
python pdf_to_png.py data/pdfs/ -o data/images/ --dpi 300 --workers 8

# Gerar JPEG em vez de PNG
python pdf_to_png.py data/pdfs/ -o data/images/ --fmt jpeg
```

---

### `setup_project.py` — Criar estrutura de pastas

```bash
# Cria todas as pastas do projeto
python setup_project.py

# Só mostra o que seria criado (sem alterar disco)
python setup_project.py --dry-run

# Usar config alternativo
python setup_project.py --config outro_config.yaml
```

---

### `convert_annotations.py` — Label Studio → COCO

```bash
# Conversão básica
python convert_annotations.py project-1.json data/annotations/coco.json

# Com split automático train/val/test (80/10/10)
python convert_annotations.py project-1.json data/annotations/coco.json \
    --split 0.8 0.1 0.1

# Com verificação de imagens + modo estrito (falha no primeiro erro)
python convert_annotations.py project-1.json data/annotations/coco.json \
    --split 0.8 0.1 0.1 \
    --images-dir data/images \
    --strict

# Gera: coco_train.json, coco_val.json, coco_test.json
```

---

### `train_layoutlm.py` — Treinar o modelo

```bash
# Treino com defaults do config.yaml
python train_layoutlm.py

# Especificando arquivos explicitamente
python train_layoutlm.py \
    --train data/annotations/coco_train.json \
    --val   data/annotations/coco_val.json \
    --images data/images \
    --epochs 20

# Com mixed precision (GPU NVIDIA)
python train_layoutlm.py --fp16 --epochs 30

# Retomar de checkpoint
python train_layoutlm.py --resume data/checkpoints/epoch_010

# Sem data augmentation
python train_layoutlm.py --no-aug

# Modelo large (mais preciso, mais lento)
python train_layoutlm.py --model microsoft/layoutlmv3-large
```

---

### `predict.py` — Inferência em PDF novo

```bash
# Básico
python predict.py extrato.pdf --checkpoint data/checkpoints/best

# Com nome do titular (detecta transferências intra-usuário)
python predict.py extrato.pdf \
    --checkpoint data/checkpoints/best \
    --holder "João da Silva"

# Com visualização das detecções salvas como imagem
python predict.py extrato.pdf \
    --checkpoint data/checkpoints/best \
    --visualize

# Só processar páginas específicas
python predict.py extrato.pdf --checkpoint data/checkpoints/best --pages 1 3

# Resultado salvo em arquivo específico
python predict.py extrato.pdf \
    --checkpoint data/checkpoints/best \
    --out data/predictions/resultado.json

# Limiar de confiança personalizado
python predict.py extrato.pdf \
    --checkpoint data/checkpoints/best \
    --score-thresh 0.5

# Completo
python predict.py extrato.pdf \
    --checkpoint data/checkpoints/best \
    --holder "João da Silva" \
    --score-thresh 0.4 \
    --visualize \
    --out data/predictions/resultado.json
```

---

### `app.py` — Iniciar o app Streamlit

```bash
# Iniciar o app web
streamlit run app.py

# Porta personalizada
streamlit run app.py --server.port 8502
```

---

### Label Studio — Ferramenta de anotação

```bash
# Iniciar
.venv\Scripts\label-studio start
# Acessa em: http://localhost:8080

# Porta alternativa
.venv\Scripts\label-studio start --port 8090
```

**Template XML para o projeto:**
```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="header"             background="#3498db"/>
    <Label value="transaction_row"    background="#27ae60"/>
    <Label value="transfer_indicator" background="#e74c3c"/>
  </RectangleLabels>
</View>
```

---

## Fluxo completo do pipeline

```
1. python setup_project.py
2. python pdf_to_png.py C:/caminho/pdfs/ -o data/images/
3. .venv\Scripts\label-studio start        ← anota em http://localhost:8080
4. python convert_annotations.py project-1.json data/annotations/coco.json --split 0.8 0.1 0.1
5. python train_layoutlm.py
6. python predict.py novo_extrato.pdf --checkpoint data/checkpoints/best --holder "Seu Nome" --visualize
```

---

## Configuração (`config.yaml`)

Parâmetros que você pode querer ajustar:

| Parâmetro | Padrão | Quando mudar |
|---|---|---|
| `training.batch_size` | `4` | Reduzir se der erro de memória (OOM) |
| `training.learning_rate` | `5e-5` | Reduzir se o treino ficar instável |
| `training.num_epochs` | `10` | Aumentar para ~20–30 com mais dados |
| `training.fp16` | `false` | Mudar para `true` com GPU NVIDIA |
| `model.name` | `layoutlmv3-base` | Trocar por `layoutlmv3-large` para mais precisão |
| `pdf_conversion.dpi` | `300` | 150 é mais rápido; 300 melhor para OCR |
| `inference.confidence_threshold` | `0.80` | Reduzir para ver mais detecções |

---

## Estrutura de pastas

```
Leitor de Conta/
├── app.py                    # App Streamlit
├── config.yaml               # Configurações centralizadas
├── CONTEXT.md                # Estado atual do projeto
├── SKILLS.md                 # Este arquivo
├── CLAUDE.md                 # Arquitetura para Claude Code
├── pdf_to_png.py             # PDF → PNG
├── convert_annotations.py   # Label Studio → COCO
├── train_layoutlm.py         # Treinamento
├── predict.py                # Inferência
├── setup_project.py          # Setup de pastas
├── requirements.txt
├── data/
│   ├── images/               # PNGs gerados
│   ├── annotations/          # COCO JSONs
│   ├── splits/train|val|test/
│   ├── checkpoints/          # Modelos treinados
│   ├── logs/                 # train_log.jsonl
│   ├── predictions/          # JSONs de inferência
│   └── poppler/              # Poppler (Windows, auto-detectado)
└── leitor/                   # Módulo do app Streamlit
    ├── parsers/
    ├── models/
    ├── scoring/
    ├── visualization/
    └── utils/
```
