import numpy as np

# Data
experts = [0, 1, 2, 4, 8, 16, 24, 32, 48, 64]
seq_time_per_expert = [3.7450, 3.7450, 3.4427, 3.3038, 3.2311, 3.2011, 3.1910, 3.1853, 3.1886, 3.1848]
par_time_per_expert = [3.6554, 3.6554, 2.5259, 2.0135, 1.8436, 1.7923, 1.7395, 1.7178, 1.6951, 1.6799]

seq_stall = np.array(experts) * np.array(seq_time_per_expert)
par_stall = np.array(experts) * np.array(par_time_per_expert)

def calc_tps(hit_rate, stall_curve):
    miss_rate = 1 - hit_rate
    m = 8 * miss_rate
    stall_per_layer = np.interp(m, experts, stall_curve)
    stall_per_token = 48 * stall_per_layer
    return 1000 / (49 + stall_per_token)

# Cache 24 empirical hit rates
hr_rand = 0.6735
hr_pred = 0.8849 # empirical Predictor B=12 hit rate with precision loss
hr_oracle = 1.0 # Oracle Full Union

print("=== Sequential IO ===")
tps_rand_seq = calc_tps(hr_rand, seq_stall)
tps_pred_seq = calc_tps(hr_pred, seq_stall)
tps_oracle_seq = calc_tps(hr_oracle, seq_stall)
print(f"RANDOM TPS: {tps_rand_seq:.2f}")
print(f"Predictor B=12 TPS: {tps_pred_seq:.2f}")
print(f"Oracle Upper Bound TPS: {tps_oracle_seq:.2f}")
print(f"Speedup: {tps_pred_seq / tps_rand_seq:.2f}x")

print("\n=== Parallel IO ===")
tps_rand_par = calc_tps(hr_rand, par_stall)
tps_pred_par = calc_tps(hr_pred, par_stall)
tps_oracle_par = calc_tps(hr_oracle, par_stall)
print(f"RANDOM TPS: {tps_rand_par:.2f}")
print(f"Predictor B=12 TPS: {tps_pred_par:.2f}")
print(f"Oracle Upper Bound TPS: {tps_oracle_par:.2f}")
print(f"Speedup: {tps_pred_par / tps_rand_par:.2f}x")
