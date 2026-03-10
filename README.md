# labp103 — Decoder Transformer

Implementação dos blocos fundamentais de um Decoder Transformer usando apenas NumPy:

1. **Máscara Causal** (`src/causal_mask.py`) — Look-Ahead Mask que impede o modelo de olhar para tokens futuros
2. **Cross-Attention** (`src/cross_attention.py`) — Atenção cruzada entre Encoder e Decoder
3. **Loop Auto-Regressivo** (`src/autoregressive.py`) — Geração de tokens um a um com parada em `<EOS>`

## Requisitos

- Python 3.x

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Como rodar

```bash
source .venv/bin/activate

python3 src/causal_mask.py
python3 src/cross_attention.py
python3 src/autoregressive.py
```
