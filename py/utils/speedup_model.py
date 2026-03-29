import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Dict, Any

def calculate_theoretical_speedup(
    predictor_accuracy: Union[float, np.ndarray],
    num_experts: int,
    active_experts: int,
    cache_size: int,
    prefetch_count: int,
    num_layers: int,
    t_load_ms: float,
    t_compute_ms: float = None,
    compute_bound_tps: float = None,
    total_model_size_gb: float = None,
    expert_size_mb: float = None,
) -> Dict[str, Any]:
    """
    Calculates the theoretical maximum speedup given system and model parameters,
    comparing a baseline (no predictor, on-demand loads) against a system with an expert predictor.
    
    Args:
        predictor_accuracy: Fraction of the truly active experts that are correctly part of the `prefetch_count` predictions.
        num_experts: Total number of experts per layer.
        active_experts: Number of active experts required per token per layer.
        cache_size: Number of experts that can be cached per layer.
        prefetch_count: Number of experts predicted and preloaded per layer by the predictor.
        num_layers: Number of prediction layers in the model.
        t_load_ms: Time to load a single expert in milliseconds.
        t_compute_ms: Compute time per token in milliseconds. Provide either this or compute_bound_tps.
        compute_bound_tps: Compute bound tokens per second. Provide either this or t_compute_ms.
        total_model_size_gb: Total model size in GB. Used to compute memory footprint.
        expert_size_mb: Size of a single expert in MB. Used with total_model_size_gb to compute footprint.
        
    Returns:
        Dictionary containing theoretical times and speedup.
    """
    if t_compute_ms is None and compute_bound_tps is None:
        raise ValueError("Must provide either t_compute_ms or compute_bound_tps")
    
    if t_compute_ms is None:
        t_compute_ms = 1000.0 / compute_bound_tps
        
    # Convert accuracy to numpy array to handle both scalar and arrays
    accuracies = np.atleast_1d(predictor_accuracy)
    
    # 1. Natural Cache Hit Rate & Miss Rate
    # Assuming caching policy yields a natural hit rate roughly proportional to cache capacity
    natural_hit_rate = min(1.0, cache_size / num_experts)
    miss_rate = max(0.0, 1.0 - natural_hit_rate)

    miss_rate_streaming = 1.0
    
    # --- STREAMING BASELINE (No Cache, No Predictor) ---
    streaming_ondemand_loads = num_layers * active_experts * miss_rate_streaming
    t_streaming_ondemand = streaming_ondemand_loads * t_load_ms
    t_streaming_total = t_compute_ms + t_streaming_ondemand
    
    # --- BASELINE WITH CACHE (No Predictor) ---
    # We must load on-demand all active experts that miss the cache
    baseline_ondemand_loads = num_layers * active_experts * miss_rate
    t_baseline_ondemand = baseline_ondemand_loads * t_load_ms
    print(f"Cache baseline: {cache_size}")
    print(f"Baseline ondemand loads: {baseline_ondemand_loads}")
    print(f"Baseline ondemand time: {t_baseline_ondemand}")
    
    # Baseline total time: compute + entirely blocking on-demand loads
    t_baseline_total = t_compute_ms + t_baseline_ondemand
    print(f"Baseline total time: {t_baseline_total}``")

    
    # --- WITH PREDICTOR ---``
    # We confidently preload `prefetch_count` experts per layer.
    # The ones that miss the cache will consume network bandwidth.
    preload_loads = num_layers * prefetch_count * miss_rate
    t_preload = preload_loads * t_load_ms
    print(f"Prefteching loads: {preload_loads} with miss rate: {miss_rate}")
    print(f"Prefteching time: {t_preload}")
    
    # Based on accuracy, how many of the true active experts were NOT in the predicted set?
    unpredicted_active_experts = prefetch_count * (1.0 - accuracies)
    print(f"Made mistake on {unpredicted_active_experts} top experts per layer")
    
    # The correctly predicted experts
    correctly_predicted = prefetch_count * accuracies
    print(f"Correctly predicted {correctly_predicted} top experts per layer")
    
    # The number of incorrectly predicted experts that were preloaded
    # (Assuming we always preload exactly `prefetch_count` experts)
    incorrect_preloads = np.maximum(0, prefetch_count - correctly_predicted)
    
    # Incorrect preloads that miss the cache cause evictions
    incorrect_preloads_miss = incorrect_preloads * miss_rate
    
    # How many of the unpredicted active experts were naturally in the cache?
    unpredicted_cached = unpredicted_active_experts * natural_hit_rate
    
    # What's the probability that a random eviction removes one of the unpredicted active experts?
    # It is the fraction of the cache they occupy.
    eviction_prob = (unpredicted_cached / cache_size) if cache_size > 0 else 0.0
    
    # Number of unpredicted active experts that were in the cache but get evicted due to bad prefetches
    evicted_active = np.minimum(incorrect_preloads_miss * eviction_prob, unpredicted_cached)
    
    # The unpredicted active experts miss the cache naturally (miss_rate), PLUS the ones that got evicted
    effective_misses_per_layer = unpredicted_active_experts * miss_rate + evicted_active
    
    ondemand_loads = num_layers * effective_misses_per_layer
    t_ondemand = ondemand_loads * t_load_ms
    
    # Total time with predictor steady state:
    # Compute and Preloading happen essentially in parallel.
    # On-demand loads stall BOTH compute and preloading since they are strictly necessary immediately.
    # Assuming the network is shared, the total network time is T_preload + T_ondemand.
    # The compute-path time is T_compute + T_ondemand.
    # The bottleneck is the max of the two paths.
    
    t_new_total = np.maximum(t_compute_ms, t_preload) + t_ondemand
    
    speedup = t_baseline_total / t_new_total
    
    # Memory footprint calculations
    memory_footprint_gb = None
    footprint_percent = None
    
    if total_model_size_gb is not None and expert_size_mb is not None:
        total_experts_size_gb = (expert_size_mb * num_experts * num_layers) / 1024.0
        base_size_gb = total_model_size_gb - total_experts_size_gb
        
        cache_size_gb = (cache_size * num_layers * expert_size_mb) / 1024.0
        
        memory_footprint_gb = base_size_gb + cache_size_gb
        footprint_percent = memory_footprint_gb / total_model_size_gb

    results = {
        "t_baseline_total_ms": t_baseline_total,
        "t_baseline_ondemand_ms": t_baseline_ondemand,
        "t_streaming_total_ms": t_streaming_total,
        "t_streaming_ondemand_ms": t_streaming_ondemand,
        "t_new_total_ms": t_new_total,
        "t_preload_ms": t_preload,
        "t_ondemand_ms": t_ondemand,
        "speedup": speedup,
        "speedup_vs_streaming": t_streaming_total / t_new_total,
        "t_compute_ms": t_compute_ms,
        "accuracies": accuracies,
    }
    
    if memory_footprint_gb is not None:
        results["memory_footprint_gb"] = memory_footprint_gb
        results["footprint_percent"] = footprint_percent

    # Add Roofline metrics (Operational Intensity and Attained Performance)
    if expert_size_mb is not None:
        expert_size_gb = expert_size_mb / 1024.0
        total_loads = preload_loads + ondemand_loads
        loaded_gb_per_token = total_loads * expert_size_gb
        
        # Avoid division by zero naturally
        intensity = 1.0 / np.maximum(loaded_gb_per_token, 1e-9) 
        tps = 1000.0 / t_new_total
        
        results["loaded_gb_per_token"] = loaded_gb_per_token
        results["intensity"] = intensity
        results["tps"] = tps
    
    # If scalar was passed, return scalars instead of arrays of shape (1,)
    if np.isscalar(predictor_accuracy) or accuracies.size == 1:
        for k, v in results.items():
            if isinstance(v, np.ndarray) and v.size == 1:
                results[k] = v.item()
                
    return results

def model_mixtral_tps(
    predictor_accuracy: Union[float, np.ndarray],
    cache_size: int = 4,
    prefetch_count: int = 2,
    predict_layers: int = 32,
    t_compute_ms: float = 40.0,
    t_load_ms: float = 16.0,
) -> Dict[str, Any]:
    """Convenience function applying the theoretical speedup model specifically to Mixtral."""
    return calculate_theoretical_speedup(
        predictor_accuracy=predictor_accuracy,
        num_experts=8,
        active_experts=2,
        cache_size=cache_size,
        prefetch_count=prefetch_count,
        num_layers=predict_layers,
        t_load_ms=t_load_ms,
        t_compute_ms=t_compute_ms,
        total_model_size_gb=24.6,
        expert_size_mb=96.0,
    )

    
def model_qwen_30b_tps(
    predictor_accuracy: Union[float, np.ndarray],
    cache_size: int = 8,
    prefetch_count: int = 8,
    predict_layers: int = 48,
    t_compute_ms: float = 28.0,
    t_load_ms: float = 1.5,
) -> Dict[str, Any]:
    """Convenience function applying the theoretical speedup model specifically to Mixtral."""
    return calculate_theoretical_speedup(
        predictor_accuracy=predictor_accuracy,
        num_experts=128,
        active_experts=8,
        cache_size=cache_size,
        prefetch_count=prefetch_count,
        num_layers=predict_layers,
        t_load_ms=t_load_ms,
        t_compute_ms=t_compute_ms,
        total_model_size_gb=16.8,
        expert_size_mb=2.5,
    )

def plot_roofline_for_model(results_dict_list, labels, t_compute_ms, t_load_ms_per_expert, expert_size_mb, filename="roofline.png"):
    """
    Plots a standard Roofline Model showing the relationship between operational intensity and performance.
    """
    plt.figure(figsize=(10, 6))
    
    # 1. Calculate Hardware Boundaries
    peak_tps = 1000.0 / t_compute_ms
    # Bandwidth (GB/s) = (Expert Size GB) / (Load Time Seconds)
    expert_size_gb = expert_size_mb / 1024.0
    load_time_seconds = t_load_ms_per_expert / 1000.0
    bandwidth_gbps = expert_size_gb / load_time_seconds
    
    # Extract limits for x-axis
    max_intensity = max([np.max(res["intensity"]) for res in results_dict_list]) * 1.5
    min_intensity = min([np.min(res["intensity"]) for res in results_dict_list]) * 0.7
    
    x_vals = np.logspace(np.log10(min_intensity), np.log10(max_intensity), 100)
    
    # Roofline boundary: min(Peak Compute, Bandwidth * Intensity)
    y_roof = np.minimum(peak_tps, bandwidth_gbps * x_vals)
    
    plt.plot(x_vals, y_roof, color='black', linewidth=2, label='Theoretical Hardware Limit', zorder=1)
    
    # Plot paths for each configuration
    colors = ['purple', 'blue', 'orange', 'green', 'red', 'cyan']
    for idx, (res, label) in enumerate(zip(results_dict_list, labels)):
        color = colors[idx % len(colors)]
        # Connect accuracies: intensity increases as accuracy increases
        plt.plot(res["intensity"], res["tps"], color=color, linewidth=2, zorder=2, label=label)
        
        # Mark Lowest Accuracy (0%) and Highest Accuracy (100%)
        # accuracies array linearly spaces from 0 to 1
        plt.scatter(res["intensity"][0], res["tps"][0], color=color, marker='o', s=50, zorder=3)
        plt.scatter(res["intensity"][-1], res["tps"][-1], color=color, marker='*', s=150, zorder=3)
        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Operational Intensity [Tokens Computed / GB Loaded]')
    plt.ylabel('Performance [Tokens / Second]')
    plt.title('Roofline Model: Attained Performance vs Intensity')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved roofline plot to '{filename}'")


if __name__ == "__main__":
    # Example usage for Mixtral
    accuracies = np.linspace(0.0, 1.0, 100)
    mixtral_expert_load_time = 8.1
    prefetch_count = 1


    res_cache6 = model_mixtral_tps(
        predictor_accuracy=accuracies,
        cache_size=6,
        prefetch_count=prefetch_count,
        predict_layers=32,
        t_compute_ms=40.0,
        t_load_ms=mixtral_expert_load_time
    )
    
    res_cache4 = model_mixtral_tps(
        predictor_accuracy=accuracies,
        cache_size=4,
        prefetch_count=prefetch_count,
        predict_layers=32,
        t_compute_ms=40.0,
        t_load_ms=mixtral_expert_load_time
    )
    
    res_cache2 = model_mixtral_tps(
        predictor_accuracy=accuracies,
        cache_size=2,
        prefetch_count=prefetch_count,
        predict_layers=32,
        t_compute_ms=40.0,
        t_load_ms=mixtral_expert_load_time
    )
    
    
    print("Done")
    
    plt.figure(figsize=(10, 6))
    
    def label_str(res, hit_rate):
        footprint_gb = res['memory_footprint_gb']
        percent = res['footprint_percent'] * 100
        return f"Cache {res['accuracies'].size if 'cache_size' not in res else res.get('cache_size', 'N/A')} ({hit_rate} hit, {footprint_gb:.1f}GB / {percent:.1f}%)"
    
    # plt.plot(accuracies * 100, res_cache7['speedup'], label=f"Cache Size 7 (87.5% hit, {res_cache7['memory_footprint_gb']:.1f}GB [{res_cache7['footprint_percent']*100:.1f}%])", color='green')
    plt.plot(accuracies * 100, res_cache6['speedup'], label=f"Cache Size 6 (75% hit, {res_cache6['memory_footprint_gb']:.1f}GB [{res_cache6['footprint_percent']*100:.1f}%])", color='purple')
    plt.plot(accuracies * 100, res_cache4['speedup'], label=f"Cache Size 4 (50% hit, {res_cache4['memory_footprint_gb']:.1f}GB [{res_cache4['footprint_percent']*100:.1f}%])", color='blue')
    plt.plot(accuracies * 100, res_cache2['speedup'], label=f"Cache Size 2 (25% hit, {res_cache2['memory_footprint_gb']:.1f}GB [{res_cache2['footprint_percent']*100:.1f}%])", color='orange')
     
    plt.axhline(1.0, color='red', linestyle='--', label='Baseline (Speedup 1.0x)')
    
    plt.xlabel('Predictor Accuracy (%)')
    plt.ylabel('Theoretical Speedup (x)')
    plt.title('Theoretical Maximum Speedup vs. Predictor Accuracy with Simple Cache')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'theor_mixtral_speedup_{mixtral_expert_load_time}_{prefetch_count}.png')
    print(f"Saved plot to 'theor_mixtral_speedup_{mixtral_expert_load_time}_{prefetch_count}.png'")
    
    # --- Plot vs Streaming Baseline ---
    plt.figure(figsize=(10, 6))
    
    plt.plot(accuracies * 100, res_cache6['speedup_vs_streaming'], label="Cache Size 6", color='purple')
    plt.plot(accuracies * 100, res_cache4['speedup_vs_streaming'], label="Cache Size 4", color='blue')
    plt.plot(accuracies * 100, res_cache2['speedup_vs_streaming'], label="Cache Size 2", color='orange')
     
    plt.axhline(1.0, color='red', linestyle='--', label='Streaming Baseline (Speedup 1.0x)')
    
    plt.xlabel('Predictor Accuracy (%)')
    plt.ylabel('Theoretical Speedup (x)')
    plt.title('Theoretical Maximum Speedup vs. Predictor Accuracy with Streaming')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'theor_mixtral_speedup_vs_streaming_{mixtral_expert_load_time}_{prefetch_count}.png')
    print(f"Saved plot to 'theor_mixtral_speedup_vs_streaming_{mixtral_expert_load_time}_{prefetch_count}.png'")

    # --- Roofline Plot ---
    plot_roofline_for_model(
        results_dict_list=[res_cache6, res_cache4, res_cache2],
        labels=["Cache 6", "Cache 4", "Cache 2"],
        t_compute_ms=40.0,
        t_load_ms_per_expert=mixtral_expert_load_time,
        expert_size_mb=96.0,
        filename=f"roofline_mixtral_{mixtral_expert_load_time}_{prefetch_count}.png"
    )

# =================================================================================================

    # Example usage for Qwen 30B
    accuracies = np.linspace(0.0, 1.0, 100)
    qwen_prefetch_count = 4
    qwen_load_ms = 1.5

    res_cache8 = model_qwen_30b_tps(
    predictor_accuracy=accuracies,
    cache_size=8,
    prefetch_count=qwen_prefetch_count,
    predict_layers=48,
    t_compute_ms=28.0,
    t_load_ms=qwen_load_ms)

    res_cache12 = model_qwen_30b_tps(
    predictor_accuracy=accuracies,
    cache_size=12,
    prefetch_count=qwen_prefetch_count,
    predict_layers=48,
    t_compute_ms=28.0,
    t_load_ms=qwen_load_ms)

    res_cache16 = model_qwen_30b_tps(
    predictor_accuracy=accuracies,
    cache_size=16,
    prefetch_count=qwen_prefetch_count,
    predict_layers=48,
    t_compute_ms=28.0,
    t_load_ms=qwen_load_ms)

    res_cache20 = model_qwen_30b_tps(
    predictor_accuracy=accuracies,
    cache_size=20,
    prefetch_count=qwen_prefetch_count,
    predict_layers=48,
    t_compute_ms=28.0,
    t_load_ms=qwen_load_ms)

    res_cache24 = model_qwen_30b_tps(
    predictor_accuracy=accuracies,
    cache_size=24,
    prefetch_count=qwen_prefetch_count,
    predict_layers=48,
    t_compute_ms=28.0,
    t_load_ms=qwen_load_ms)

    res_cache64 = model_qwen_30b_tps(
    predictor_accuracy=accuracies,
    cache_size=64,
    prefetch_count=qwen_prefetch_count,
    predict_layers=48,
    t_compute_ms=28.0,
    t_load_ms=qwen_load_ms)
   
    print("Done")
    
    plt.figure(figsize=(10, 6))
    
    def label_str(res, hit_rate):
        footprint_gb = res['memory_footprint_gb']
        percent = res['footprint_percent'] * 100
        return f"Cache {res['accuracies'].size if 'cache_size' not in res else res.get('cache_size', 'N/A')} ({hit_rate} hit, {footprint_gb:.1f}GB / {percent:.1f}%)"
    
    plt.plot(accuracies * 100, res_cache8['speedup'], label=f"Cache Size 8 (6.25% hit, {res_cache8['memory_footprint_gb']:.1f}GB [{res_cache8['footprint_percent']*100:.1f}%])", color='green')
    plt.plot(accuracies * 100, res_cache12['speedup'], label=f"Cache Size 12 (9.38% hit, {res_cache12['memory_footprint_gb']:.1f}GB [{res_cache12['footprint_percent']*100:.1f}%])", color='purple')
    plt.plot(accuracies * 100, res_cache16['speedup'], label=f"Cache Size 16 (12.5% hit, {res_cache16['memory_footprint_gb']:.1f}GB [{res_cache16['footprint_percent']*100:.1f}%])", color='blue')
    plt.plot(accuracies * 100, res_cache20['speedup'], label=f"Cache Size 20 (15.63% hit, {res_cache20['memory_footprint_gb']:.1f}GB [{res_cache20['footprint_percent']*100:.1f}%])", color='red')
    plt.plot(accuracies * 100, res_cache24['speedup'], label=f"Cache Size 24 (18.75% hit, {res_cache24['memory_footprint_gb']:.1f}GB [{res_cache24['footprint_percent']*100:.1f}%])", color='orange')
    plt.plot(accuracies * 100, res_cache64['speedup'], label=f"Cache Size 64 (50% hit, {res_cache64['memory_footprint_gb']:.1f}GB [{res_cache64['footprint_percent']*100:.1f}%])", color='cyan')
    plt.axhline(1.0, color='red', linestyle='--', label='Baseline (Speedup 1.0x)')
    
    plt.xlabel('Predictor Accuracy (%)')
    plt.ylabel('Theoretical Speedup (x)')
    plt.title('Theoretical Maximum Speedup vs. Predictor Accuracy (Qwen 30B)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'theor_qwen_speedup_{qwen_load_ms}_{qwen_prefetch_count}.png')
    print(f"Saved plot to 'theor_qwen_speedup_{qwen_load_ms}_{qwen_prefetch_count}.png'")

    # --- Roofline Plot ---
    plot_roofline_for_model(
        results_dict_list=[res_cache8, res_cache12, res_cache16, res_cache20, res_cache24, res_cache64],
        labels=["Cache 8", "Cache 12", "Cache 16", "Cache 20", "Cache 24", "Cache 64"],
        t_compute_ms=28.0,
        t_load_ms_per_expert=qwen_load_ms,
        expert_size_mb=2.5,
        filename=f"roofline_qwen_{qwen_load_ms}_{qwen_prefetch_count}.png"
    )
