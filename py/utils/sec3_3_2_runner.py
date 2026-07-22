import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_predict_cached_cache_metrics as sweep

def main():
    parser = sweep.build_arg_parser()
    args = parser.parse_args()

    # Build the 24 configs exactly: 8 cache sizes x (LRU, RANDOM, CacheCond J=6)
    configs = []
    question = "SEC3_3_2"
    for cache_size in args.cache_sizes:
        # LRU
        c_lru = sweep.cfg(question, "Neither (LRU)", "cached", cache_size, 0.0, 0, lookahead=1, prefetch_budget=None, prefetch_threshold=0.0, mode_perplexity=False, mode_generate=True)
        c_lru["cache_policy"] = "LRU"
        configs.append(c_lru)
        
        # RANDOM
        c_rand = sweep.cfg(question, "Neither (RANDOM)", "cached", cache_size, 0.0, 0, lookahead=1, prefetch_budget=None, prefetch_threshold=0.0, mode_perplexity=False, mode_generate=True)
        c_rand["cache_policy"] = "RANDOM"
        configs.append(c_rand)

        # Cache-Cond Only lambda=1.0 FN=6
        c_cc = sweep.cfg(question, "Cache-Cond Only \u03bb=1.0", "cached", cache_size, 1.0, 6, lookahead=1, prefetch_budget=None, prefetch_threshold=0.0, mode_perplexity=False, mode_generate=True)
        configs.append(c_cc)

    print(f"Generated {len(configs)} configurations for sec3_3_2")

    prompts = sweep.load_prompts(args)
    
    sweep.run_all_configs(configs, prompts, args)

if __name__ == "__main__":
    main()
