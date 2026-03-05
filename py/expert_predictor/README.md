# Expert Predictor

Trains small MLP networks to predict **which experts a MoE layer will route to** at token `t+1`, given the post-attention-norm embedding at token `t` (optionally plus a history window of previous embeddings).

Supported models out of the box:

| Model | `--model` preset | Experts | Active | Layers | Embed dim |
|---|---|---|---|---|---|
| Mixtral 8x7B | `mixtral_8x7b` | 8 | 2 | 32 | 4096 |
| Mixtral 8x22B | `mixtral_8x22b` | 8 | 2 | 56 | 4096 |
| Qwen3 30B-A3B | `qwen3_30b` | 128 | 8 | 48 | 2048 |
| Qwen3 480B | `qwen3_480b` | 128 | 8 | 94 | 7168 |

---

## Directory layout

```
py/expert_predictor/
├── mlp_predictor.py   — ExpertPredictor model class (MLP backbone)
├── train.py           — per-layer training script
├── sweep.py           — hidden-dim hyperparameter sweep
├── plot.py            — comprehensive results plotter
└── run_train.sh       — convenience launcher (wraps all three scripts)

trainingData/          (two levels up — shared with data collection)
└── <model>/
    ├── *.pt                     ← collected training samples
    └── predictor_models/
        └── layer_<L>/
            └── hidden_<H>/
                ├── embedding_predictor_best.pt
                └── training_metrics.json
```

---

## Prerequisites

The ROCm/PyTorch environment must be activated before running any script:

```bash
source ../../utils/setup.sh   # from py/expert_predictor/
```

`run_train.sh` does this automatically. If calling `train.py`, `sweep.py`, or `plot.py` directly, source it first.

---

## Step 0 — Collect training data

Data collection lives in `../unified_llm_w4a16/`. Run it first if you haven't already:

```bash
cd ../unified_llm_w4a16
bash run_collect_training_data.sh mixtral_8x7b   # produces trainingData/mixtral_8x7b/*.pt
bash run_collect_training_data.sh qwen3_30b      # produces trainingData/qwen3_30b/*.pt
```

---

## Step 1 — Train

### All layers (recommended)

```bash
bash run_train.sh mixtral_8x7b      # trains all 32 layers
bash run_train.sh qwen3_30b         # trains all 48 layers
```

### Single layer (quick test / comparison)

```bash
bash run_train.sh mixtral_8x7b 0    # layer 0 only
bash run_train.sh qwen3_30b 15      # layer 15 only
```

### Direct Python (full control)

```bash
source ../../utils/setup.sh

python train.py \
    --data_dir   ../../trainingData/mixtral_8x7b \
    --output_dir ../../trainingData/mixtral_8x7b/predictor_models \
    --model      mixtral_8x7b \
    --layer_idx  0 \
    --epochs     20 \
    --batch_size 64 \
    --lr         3e-4
```

Key flags:

| Flag | Default | Notes |
|---|---|---|
| `--model` | — | Applies preset for `embedding_dim`, `num_experts`, `top_k`, `num_layers` |
| `--layer_idx` | all layers | Omit to train every layer in sequence |
| `--history` | 1 | Embedding history window: 1 = current token only, 2 = current + previous, … |
| `--hidden_dim` | auto | Override MLP hidden dim (default: auto-sized from `embedding_dim`) |
| `--epochs` | 10 | |
| `--batch_size` | 32 | |
| `--lr` | 1e-3 | |

---

## Step 2 — (Optional) Hyperparameter sweep

Sweeps `hidden_dim` across all (or a subset of) layers:

```bash
bash run_train.sh sweep mixtral_8x7b

# Or directly:
python sweep.py \
    --data_dir   ../../trainingData/mixtral_8x7b \
    --output_dir ../../trainingData/mixtral_8x7b/sweep_results \
    --model      mixtral_8x7b \
    --hidden_dims 256 512 1024 2048 \
    --layers 0 1 2 3    # omit to sweep all layers
    --epochs 15
```

Results are written to `<output_dir>/layer_<L>/hidden_<H>/training_metrics.json`.

---

## Step 3 — Plot

```bash
bash run_train.sh plot mixtral_8x7b

# Or directly (point at any dir containing training_metrics.json files):
python plot.py --sweep_dir ../../trainingData/mixtral_8x7b/predictor_models
python plot.py --sweep_dir ../../trainingData/mixtral_8x7b/sweep_results --phase both
```

### Output figures

| File | Description |
|---|---|
| `plots/per_layer/layer_XX.png` | One subplot per metric (train + val curves vs epoch) |
| `plots/cross_layer/<metric>.png` | Best val accuracy vs layer index, per metric |
| `plots/heatmap_all_metrics.png` | 2-D heatmap: all metrics × all layers |
| `plots/summary.png` | Key metrics overlaid on a single cross-layer figure |

`--phase val` (default) plots validation only. `--phase both` overlays train and val.  
`--no_per_layer` or `--no_cross_layer` skips those plot groups for speed.

---

## Recorded metrics

All metrics are tracked every epoch (for both train and val) and stored in `training_metrics.json`:

| Metric key | Description |
|---|---|
| `acc` / `full_match` | All `top_k` experts predicted correctly (exact set match) |
| `any_correct` | At least 1 of the predicted experts is correct (partial accuracy) |
| `top1_exact` | The single highest-confidence prediction == the actual #1 expert |
| `topN_in_pred` (N=1..top_k) | The actual top-N experts are all found in the predicted top-k set |
| `exact_set_k` (k=1..top_k) | Exact set match at each prefix size k |
| `mean_overlap` | Average number of correct experts in the predicted set (0..top_k) |
| `mean_prefix` | Average length of the longest ordered prefix that matches exactly |
| `loss` | FocalLoss value |

For **Mixtral** (`top_k=2`), the most useful are: `top1_exact`, `any_correct`, `acc`.  
For **Qwen** (`top_k=8`), `mean_overlap` and `any_correct` give the clearest picture of partial credit.

---

## Saved model format

Best-val checkpoints are saved as:

```
layer_<L>/hidden_<H>/embedding_predictor_best.pt
```

Loading a checkpoint:

```python
from mlp_predictor import ExpertPredictor
model = ExpertPredictor.load("path/to/embedding_predictor_best.pt")
model.eval()

# Inference: feed a [1, embedding_dim] post-attn-norm embedding
logits   = model(post_attn_embedding=embedding)          # [1, num_experts]
pred_top = torch.topk(logits, model.top_k, dim=-1).indices
```
