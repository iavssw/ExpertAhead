import numpy as np
import matplotlib.pyplot as plt

def model_mixtral_tps(
    predictor_accuracy: float,
    cache_size: int = 4,
    prefetch_count: int = 1,
    predict_layers: int = 32,
    t_compute_ms: float = 50.0,
    t_load_ms: float = 18.2,
    base_hit_rate: float = 0.65,
    lambda_factor: float = 1.0
):
    """
    Models the TPS of Mixtral 8x7B given hardware and predictor constraints.
    
    Args:
        predictor_accuracy: Probability (0.0 to 1.0) that the prefetched expert is one of the 2 used.
        cache_size: Number of experts held in RAM per layer.
        prefetch_count: Number of experts the predictor fetches in the background per layer.
        predict_layers: How many layers to run the predictor on (0 to 32).
        t_compute_ms: Base GPU math compute time per token (in milliseconds).
        t_load_ms: Time to load 1 expert from SSD to RAM (in milliseconds).
        base_hit_rate: Organic LRU hit rate without any predictor interference.
        lambda_factor: How strongly the router biases to cached experts (mitigates double misses).
    """
    layers = 32
    experts_per_token = 2
    
    # 1. Baseline LRU (No predictor)
    organic_misses_base = layers * experts_per_token * (1.0 - base_hit_rate)
    t_disk_base = organic_misses_base * t_load_ms
    # Baseline is fully synchronous, so time is compute + disk stall
    t_token_base = t_compute_ms + t_disk_base
    tps_base = 1000.0 / t_token_base
    
    # 2. Predictor Model
    # Speculative fetches only touch the PCIe bus if the expert is NOT already in the cache!
    # Roughly (1 - base_hit_rate) of the requested experts actually result in a physical read.
    physical_miss_prob = 1.0 - base_hit_rate
    background_loads = predict_layers * prefetch_count * physical_miss_prob
    
    # How many of the physical prefetch reads were WRONG?
    wrong_guesses = predict_layers * prefetch_count * physical_miss_prob * (1.0 - predictor_accuracy)
    
    # A wrong guess evicts a good expert from the cache. 
    # If the cache is small (e.g., 3), a wrong guess is devastating.
    # The vulnerability scales inversely with cache size (cache=8 means 0 vulnerability).
    cache_vulnerability = max(0.0, (8.0 - cache_size) / 8.0) 
    
    # Induced organic misses due to cache poisoning (mitigated by lambda adapting router)
    lambda_mitigation = max(0.0, 1.0 - (lambda_factor * 0.5)) 
    induced_misses = wrong_guesses * cache_vulnerability * lambda_mitigation
    
    # Organic misses we still have to suffer synchronounsly
    # (Base misses minus the ones the predictor correctly guessed, plus induced misses)
    correct_guesses = predict_layers * prefetch_count * predictor_accuracy
    organic_misses_pred = max(0.0, organic_misses_base - correct_guesses + induced_misses)
    
    # Total Disk Time = Background Loads + Synchronous Organic Misses
    t_disk_pred = (background_loads + organic_misses_pred) * t_load_ms
    
    # If disk operations run asynchronously alongside compute, the token time is the MAX of compute or disk
    # But synchronous organic misses always add to the total time regardless.
    t_background_disk = background_loads * t_load_ms
    t_synchronous_disk = organic_misses_pred * t_load_ms
    
    # The background disk overlaps with compute. 
    # The synchronous disk adds directly to latency.
    t_overlapped = max(t_compute_ms, t_background_disk)
    t_token_pred = t_overlapped + t_synchronous_disk
    tps_pred = 1000.0 / t_token_pred
    
    speedup = tps_pred / tps_base
    return tps_pred, tps_base, speedup

def plot_accuracy_vs_speedup():
    accuracies = np.linspace(0.0, 1.0, 100)
    
    plt.figure(figsize=(12, 8))
    
    # Plot for different numbers of predicted layers
    for predict_layers in [4, 16, 32]:
        speedups_cache5 = [model_mixtral_tps(acc, cache_size=5, predict_layers=predict_layers)[2] for acc in accuracies]
        plt.plot(accuracies * 100, speedups_cache5, linewidth=2, label=f'Predict {predict_layers} Layers (Cache=5)')
        
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Baseline (LRU Only - No Speedup)')
    
    plt.title('Mixtral 8x7B: Speedup vs Predictor Accuracy', fontsize=16)
    plt.xlabel('Predictor Accuracy (%)', fontsize=14)
    plt.ylabel('Relative Speedup (TPS_pred / TPS_base)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Find break-even points
    for predict_layers in [16, 32]:
        speedups = [model_mixtral_tps(acc, cache_size=5, predict_layers=predict_layers)[2] for acc in accuracies]
        break_even_idx = np.where(np.array(speedups) > 1.0)[0]
        if len(break_even_idx) > 0:
            break_even_acc = accuracies[break_even_idx[0]] * 100
            plt.scatter([break_even_acc], [1.0], color='black', zorder=5)
            plt.annotate(f'  {break_even_acc:.1f}% limit', (break_even_acc, 1.02), fontsize=10)

    plt.tight_layout()
    plt.savefig('predictor_speedup_model.png', dpi=300)
    print("Saved plot to 'predictor_speedup_model.png'")

if __name__ == "__main__":
    plot_accuracy_vs_speedup()
    
    # Print an analysis table
    print("\n--- Model Analysis: 32 Predicted Layers ---")
    print(f"{'Accuracy':<10} | {'TPS (Predict)':<15} | {'TPS (Base)':<15} | {'Speedup':<10}")
    print("-" * 55)
    for acc in [0.2, 0.4, 0.6, 0.75, 0.85, 0.95]:
        tps_p, tps_b, speedup = model_mixtral_tps(acc, predict_layers=32)
        print(f"{acc*100:<9.1f}% | {tps_p:<15.2f} | {tps_b:<15.2f} | {speedup:<9.2f}x")
        
    print("\n--- Model Analysis: 8 Predicted Layers ---")
    print(f"{'Accuracy':<10} | {'TPS (Predict)':<15} | {'TPS (Base)':<15} | {'Speedup':<10}")
    print("-" * 55)
    for acc in [0.2, 0.4, 0.6, 0.75, 0.85, 0.95]:
        tps_p, tps_b, speedup = model_mixtral_tps(acc, predict_layers=8)
        print(f"{acc*100:<9.1f}% | {tps_p:<15.2f} | {tps_b:<15.2f} | {speedup:<9.2f}x")
