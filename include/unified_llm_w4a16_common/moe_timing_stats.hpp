#pragma once

#include <cstdint>
#include <iomanip>
#include <iostream>

namespace unified_llm_w4a16_common {

inline void print_moe_stall_bandwidth(std::ostream& os, int64_t stall_loads, int64_t prefetch_loads, double avg_load_time_ms) {
    const int64_t total_loads = stall_loads + prefetch_loads;
    if (total_loads <= 0) {
        return;
    }
    os << "    Bandwidth: StallLoads=" << stall_loads << ", PrefetchLoads=" << prefetch_loads
       << ", AvgLoadTime=" << std::fixed << std::setprecision(2) << avg_load_time_ms << "ms\n";
}

inline void print_moe_miss_bandwidth(std::ostream& os, int64_t miss_loads, double avg_load_time_ms) {
    if (miss_loads <= 0) {
        return;
    }
    os << "    Bandwidth: MissLoads=" << miss_loads << ", AvgLoadTime=" << std::fixed << std::setprecision(2)
       << avg_load_time_ms << "ms\n";
}

inline void print_moe_compute_only(std::ostream& os, int64_t expert_invocations, double total_compute_ms) {
    if (expert_invocations <= 0) {
        return;
    }
    const double avg = total_compute_ms / static_cast<double>(expert_invocations);
    os << "    MoE Compute: ExpertInvocations=" << expert_invocations << ", AvgComputeTime=" << std::fixed
       << std::setprecision(3) << avg << "ms\n";
}

}  // namespace unified_llm_w4a16_common
