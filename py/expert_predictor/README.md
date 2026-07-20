# Expert Predictor

Trains small MLP networks to predict **which experts a MoE layer will route to** at token `t+1`, given the post-attention-norm embedding at token `t` (optionally plus a history window of previous embeddings or outputs of previous layers).

Supported models out of the box:
- `mixtral_8x7b` (Mixtral 8x7B)
- `mixtral_8x22b` (Mixtral 8x22B)
- `qwen3_30b` (Qwen3 30B-A3B)
- `qwen3_480b` (Qwen3 480B)

---

## Directory Layout

```text
py/expert_predictor/
├── expert_predictor_cross_token.py  — Core script containing MLP model, training loop, hyperparameter sweeps, and pareto generation.
├── run_train.sh                     — Convenience wrapper for training, sweeping, and plotting.
├── run_ablation.sh                  — Automates training over multiple ablation variants (history, architecture, hidden dims) for paper plots.
└── dump_table_data.py               — Maps experimental results to plot data.
```

---

## Prerequisites

The ROCm/PyTorch environment must be activated before running any script:

```bash
source ../../utils/setup.sh
```

*(Note: `run_train.sh` and `run_ablation.sh` do this automatically.)*

---

## Step 1 — Collect Training Data

Data collection lives in `py/utils/run_collect_training_data.sh`. It runs the model on raw text (e.g., WikiText, FineWeb, Orca) and saves the embeddings and target router logits as `.pt` files.

1. Navigate to the utility directory:
   ```bash
   cd ../utils
   ```
2. Run data collection for your desired model (e.g., `qwen3_30b`):
   ```bash
   bash run_collect_training_data.sh qwen3_30b
   ```
   This produces training tensors at `../../trainingDataExtended/qwen3_30b/*.pt` (or equivalent directory).

---

## Step 2 — Train Predictors

Use the consolidated `run_train.sh` wrapper or `expert_predictor_cross_token.py` to train your expert predictors.

### Train all layers

```bash
bash run_train.sh qwen3_30b
```

### Train a single layer (quick testing)

```bash
bash run_train.sh qwen3_30b 15   # train layer 15 only
```

### Direct Execution (Advanced / Custom)

If you need finer control over hyperparameters like learning rate, epochs, embedding history window, or whether to include previous layer features:

```bash
source ../../utils/setup.sh

python expert_predictor_cross_token.py train \
    --data_dir   ../../trainingDataExtended/qwen3_30b \
    --output_dir ../../trainingDataExtended/qwen3_30b/predictor_models \
    --model      qwen3_30b \
    --layer_idx  15 \
    --epochs     10 \
    --batch_size 64 \
    --lr         3e-4 \
    --history    4       # embedding history window
```

---

## Step 3 — Ablation Study (Optional)

To automatically train a grid of models across different architectures, history sizes, and feature sets for the paper's ablation studies:

```bash
bash run_ablation.sh
```
This script loops over different configurations (e.g., Markov vs. State features, history sizes, gating networks) and outputs models into heavily nested directories inside `trainingDataExtended/qwen3_30b_sharded/transformer_ablation_.../`.

---

## Step 4 — Analysis & Pareto Fronts

To evaluate the best speedup / accuracy trade-offs of the trained models and generate data for pareto plots:

```bash
python expert_predictor_cross_token.py pareto \
    --data_dir   ../../trainingDataExtended/qwen3_30b \
    --model      qwen3_30b \
    --layers     15 16 17
```
