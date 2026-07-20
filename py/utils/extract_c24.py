import pandas as pd
df = pd.read_csv("trainingData/qwen_traces/final_empirical/sweep_20260623_184257_final.csv")
df = df[(df["cache_size"] == 24) & (df["lookahead"].isin([1,2,3,4,5]))]
for idx, row in df.iterrows():
    if "LRU" in row["label"] or "Oracle" in row["label"]:
        print(f"{row['label']:25s} LA={row['lookahead']} B={row['prefetch_budget']} TPS={row['tokens_per_second']:5.2f} Misses={row['cache_misses']} Pref={row['prefetch_loads']}")
