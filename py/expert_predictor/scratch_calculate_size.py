import sys
sys.path.append("/mnt/storage/Michael/michaelg/heteroPredict/py/expert_predictor")
from expert_predictor_cross_layer import MultiStepExpertPredictor, MODEL_DEFAULTS

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

emb_dim, num_experts, _, _, _, _ = MODEL_DEFAULTS["mixtral_8x7b"] # 4096, 8
hidden_dim = 128
history = 1
layer_idx = 10

kwargs_full = dict(
    emb_dim=emb_dim, num_experts=num_experts, hidden_dim=hidden_dim, history=history, layer_idx=layer_idx,
    use_embedding=True, use_prefill=True, use_prev=True, use_markov=True, use_prev_layers=True
)

kwargs_emb = dict(
    emb_dim=emb_dim, num_experts=num_experts, hidden_dim=hidden_dim, history=history, layer_idx=layer_idx,
    use_embedding=True, use_prefill=False, use_prev=False, use_markov=False, use_prev_layers=False
)

model_full = MultiStepExpertPredictor(**kwargs_full)
model_emb = MultiStepExpertPredictor(**kwargs_emb)

params_full = count_parameters(model_full)
params_emb = count_parameters(model_emb)

print(f"Full Features Parameters: {params_full:,}")
print(f"Embedding Only Parameters: {params_emb:,}")
print(f"Difference: {params_full - params_emb:,} parameters saved")
print(f"Percentage reduction: {((params_full - params_emb)/params_full)*100:.2f}%")
