import pandas as pd
import matplotlib.pyplot as plt
import os

csv_path = "predictor_latency.csv"
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit(1)

df = pd.read_csv(csv_path)

plt.figure(figsize=(10, 6))
plt.hist(df['latency_ms'], bins=50, color='skyblue', edgecolor='black')
plt.title('TorchScriptPredictor Inference Latency Distribution', fontsize=14)
plt.xlabel('Latency (ms)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

avg_latency = df['latency_ms'].mean()
p99_latency = df['latency_ms'].quantile(0.99)
plt.axvline(avg_latency, color='red', linestyle='dashed', linewidth=2, label=f'Avg: {avg_latency:.3f} ms')
plt.axvline(p99_latency, color='orange', linestyle='dashed', linewidth=2, label=f'P99: {p99_latency:.3f} ms')

plt.legend()
plt.tight_layout()

# Save the plot to the conversation artifacts directory
output_path = "/home/michael/.gemini/antigravity-ide/brain/4aa42593-27da-4ff2-b822-35555864b253/artifacts/predictor_latency_hist.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300)
print(f"Plot saved to {output_path}")

print(f"Predicted total overhead for 60 layers (avg): {avg_latency * 60:.2f} ms")
print(f"Predicted total overhead for 60 layers (p99): {p99_latency * 60:.2f} ms")
