import torch
from pprint import pprint

fp = "/mnt/storage/Michael/michaelg/heteroPredict/trainingData/qwen3_30b/wikitext_00000.pt"
payload = torch.load(fp, map_location='cpu', weights_only=False)
print("Keys:", payload.keys())
print("Number of layers:", len(payload.get('layers', [])))

if 'layers' in payload and len(payload['layers']) > 0:
    l0 = payload['layers'][0]
    print("Layer 0 keys:", l0.keys())
    print("Router logits shape:", l0['router_logits'].shape)
    if 'prev_expert_ids' in l0 and l0['prev_expert_ids'] is not None:
        print("Prev expert ids shape:", l0['prev_expert_ids'].shape)
