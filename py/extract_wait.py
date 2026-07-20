import pandas as pd
df = pd.read_csv("trainingData/qwen_traces/final_empirical/sweep_20260623_184257_final.csv")
df = df[(df["cache_size"] == 24) & (df["lookahead"] == 1)]
for idx, row in df.iterrows():
    if "Oracle" in row["label"]:
        print(f"{row['label']:25s}")
        print(f"  TPS: {row['tokens_per_second']:.2f}")
        print(f"  Misses: {row['cache_misses']}")
        print(f"  Prefetches: {row['prefetch_loads']}")
        print(f"  Prefetch Hits Ready: {row['prefetch_hits_ready']}")
        print(f"  Prefetch Hits Wait: {row['prefetch_hits_wait']}")
        print(f"  Prefetch Ticks Skipped: {row['prefetch_ticks_skipped']}")
        print(f"  Avg MS Per Expert: {row['avg_ms_per_expert_load']:.3f}\n")
