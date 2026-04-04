# CONTEXT.md — Estado do Projeto (pausa em 04/04/2026)

## O que é este projeto

Pipeline completo de ML para leitura de extratos bancários brasileiros:
1. Converte PDFs em imagens PNG (300 DPI)
2. Anota regiões com Label Studio (bounding boxes)
3. Treina LayoutLMv3 para detectar 3 classes de layout
4. Executa inferência com OCR e regras de negócio

---

## Onde parei

**Passo 5 de 8 — Anotação no Label Studio**

- [x] Passos 1–4 concluídos
- [ ] **Anotar as 8 imagens no Label Studio** ← PRÓXIMO PASSO
- [ ] Exportar JSON do Label Studio
- [ ] Converter anotações para COCO
- [ ] Treinar o modelo
- [ ] Inferência no PDF

---

## Arquivos do projeto

| Arquivo | Função |
|---|---|
| `app.py` | App Streamlit para análise de extratos (produto final) |
| `pdf_to_png.py` | Converte PDFs → PNG (300 DPI) |
| `convert_annotations.py` | Label Studio JSON → COCO JSON |
| `train_layoutlm.py` | Fine-tuning LayoutLMv3 (detecção de layout) |
| `predict.py` | Inferência: PDF → detecções + OCR + regras de negócio |
| `setup_project.py` | Cria estrutura de pastas do projeto |
| `config.yaml` | Parâmetros centralizados (caminhos, hiperparâmetros) |
| `CLAUDE.md` | Guia de arquitetura para Claude Code |

---

## Dados disponíveis

```
data/images/
  Comprovante_2026-04-04_114801/       page_001.png        (1 pág)
  eb2aae41-.../                        page_001-002.png    (2 págs)
  extrato_de_04-01-2026.../            page_001.png        (1 pág)
  extrato-da-sua-conta-.../            page_001-002.png    (2 págs)
  NU_817988955_01MAR2026_31MAR2026/    page_001-002.png    (2 págs)
```

**Total: 5 PDFs → 8 imagens PNG a anotar**

PDFs originais: `C:\Users\Miguel\Desktop\contracheque\`

---

## Classes de anotação

| Classe | Cor no Label Studio | Descrição |
|---|---|---|
| `header` | Azul `#3498db` | Cabeçalho do extrato (banco, período, conta) |
| `transaction_row` | Verde `#27ae60` | Cada linha de movimentação financeira |
| `transfer_indicator` | Vermelho `#e74c3c` | Indicador de TED/DOC/PIX/transferência |

---

## Comandos para retomar

```bash
# 1. Ativar ambiente
cd "C:\Users\Miguel\Desktop\Leitor de Conta"
.venv\Scripts\activate

# 2. Abrir Label Studio (anotar as 8 imagens)
.venv\Scripts\label-studio start
# → http://localhost:8080

# 3. Após exportar o JSON do Label Studio:
.venv\Scripts\python convert_annotations.py project-1.json data/annotations/coco.json --split 0.8 0.1 0.1 --images-dir data/images

# 4. Treinar
.venv\Scripts\python train_layoutlm.py

# 5. Inferência
.venv\Scripts\python predict.py "C:\Users\Miguel\Desktop\contracheque\NU_817988955_01MAR2026_31MAR2026.pdf" --checkpoint data/checkpoints/best --holder "Seu Nome" --visualize
```

---

## Dependências de sistema instaladas

- **Poppler** (pdf2image): em `data/poppler/poppler-24.08.0/Library/bin/` — detectado automaticamente pelo `pdf_to_png.py`
- **Tesseract**: ainda não instalado — necessário para OCR no `predict.py`
  - Download: https://github.com/UB-Mannheim/tesseract/wiki
  - Marcar "Portuguese" durante instalação

---

## Configuração do Label Studio (XML do template)

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

## Arquitetura do modelo

- **Backbone**: `microsoft/layoutlmv3-base` (baixado automaticamente do HuggingFace ~900 MB)
- **Modo**: visual-only (sem OCR para o modelo — usa tokens dummy)
- **Head**: 196 patches 14×14 → classificação (4 classes) + regressão de bbox
- **Loss**: Focal loss + L1 + GIoU com Hungarian matching
- **Métricas**: mAP@50, mAP@75, AP por classe

---

## Notas técnicas

- Python 3.14.3 (`.venv`)
- `albumentations` **não instalado** (requer Visual C++ Build Tools no Windows) — o treino funciona sem ele (sem data augmentation)
- `label-studio` instalado via pip na sessão de 04/04/2026
- `stringzilla` incompatível com Python 3.14 sem compilador C++ → albumentations comentado no requirements.txt
