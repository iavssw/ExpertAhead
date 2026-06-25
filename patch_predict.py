import os

with open('src/unified_llm_w4a16_predict/unified_llm_w4a16.cpp', 'r') as f:
    content = f.read()

replacements = {
    """size_t MixtureOfExpertsImpl::pick_lru_ready() {
    size_t victim = 0;
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    bool found = false;
    for (size_t i = 0; i < slot_meta_.size(); ++i) {
        if (!expert_slot_ready_[i] || slot_meta_[i].expert_id == -1) {
            continue;
        }
        if (cache_policy_ == CachePolicy::PREFILL) {
            if (std::find(locked_experts_.begin(), locked_experts_.end(), slot_meta_[i].expert_id) !=
                locked_experts_.end()) {
                continue;
            }
        }
        if (slot_meta_[i].last_access < oldest) {
            oldest = slot_meta_[i].last_access;
            victim = i;
            found = true;
        }
    }
    return found ? victim : kNoVictimSlot;
}""": """size_t MixtureOfExpertsImpl::pick_lru_ready() {
    size_t victim = 0;
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    bool found = false;
    for (size_t i = 0; i < slot_meta_.size(); ++i) {
        if (!expert_slot_ready_[i] || slot_meta_[i].expert_id == -1) {
            continue;
        }
        if (std::find(currently_selected_experts_.begin(), currently_selected_experts_.end(), slot_meta_[i].expert_id) != currently_selected_experts_.end()) {
            continue;
        }
        if (cache_policy_ == CachePolicy::PREFILL) {
            if (std::find(locked_experts_.begin(), locked_experts_.end(), slot_meta_[i].expert_id) !=
                locked_experts_.end()) {
                continue;
            }
        }
        if (slot_meta_[i].last_access < oldest) {
            oldest = slot_meta_[i].last_access;
            victim = i;
            found = true;
        }
    }
    if (!found) { // Fallback
        oldest = std::numeric_limits<uint64_t>::max();
        for (size_t i = 0; i < slot_meta_.size(); ++i) {
            if (!expert_slot_ready_[i] || slot_meta_[i].expert_id == -1) continue;
            if (slot_meta_[i].last_access < oldest) {
                oldest = slot_meta_[i].last_access;
                victim = i;
                found = true;
            }
        }
    }
    return found ? victim : kNoVictimSlot;
}""",
    """size_t MixtureOfExpertsImpl::pick_mru_ready() {
    size_t victim = kNoVictimSlot;
    uint64_t newest = 0;
    bool found = false;
    for (size_t i = 0; i < slot_meta_.size(); ++i) {
        if (!expert_slot_ready_[i] || slot_meta_[i].expert_id == -1) {
            continue;
        }
        if (slot_meta_[i].last_access >= newest) {
            newest = slot_meta_[i].last_access;
            victim = i;
            found = true;
        }
    }
    return found ? victim : kNoVictimSlot;
}""": """size_t MixtureOfExpertsImpl::pick_mru_ready() {
    size_t victim = kNoVictimSlot;
    uint64_t newest = 0;
    bool found = false;
    for (size_t i = 0; i < slot_meta_.size(); ++i) {
        if (!expert_slot_ready_[i] || slot_meta_[i].expert_id == -1) {
            continue;
        }
        if (std::find(currently_selected_experts_.begin(), currently_selected_experts_.end(), slot_meta_[i].expert_id) != currently_selected_experts_.end()) {
            continue;
        }
        if (slot_meta_[i].last_access >= newest) {
            newest = slot_meta_[i].last_access;
            victim = i;
            found = true;
        }
    }
    return found ? victim : kNoVictimSlot;
}""",
    """size_t MixtureOfExpertsImpl::pick_lfu_ready() {
    size_t victim = kNoVictimSlot;
    uint64_t least = std::numeric_limits<uint64_t>::max();
    bool found = false;
    for (size_t i = 0; i < slot_meta_.size(); ++i) {
        if (!expert_slot_ready_[i] || slot_meta_[i].expert_id == -1) {
            continue;
        }
        if (slot_meta_[i].access_count < least) {
            least = slot_meta_[i].access_count;
            victim = i;
            found = true;
        }
    }
    return found ? victim : kNoVictimSlot;
}""": """size_t MixtureOfExpertsImpl::pick_lfu_ready() {
    size_t victim = kNoVictimSlot;
    uint64_t least = std::numeric_limits<uint64_t>::max();
    bool found = false;
    for (size_t i = 0; i < slot_meta_.size(); ++i) {
        if (!expert_slot_ready_[i] || slot_meta_[i].expert_id == -1) {
            continue;
        }
        if (std::find(currently_selected_experts_.begin(), currently_selected_experts_.end(), slot_meta_[i].expert_id) != currently_selected_experts_.end()) {
            continue;
        }
        if (slot_meta_[i].access_count < least) {
            least = slot_meta_[i].access_count;
            victim = i;
            found = true;
        }
    }
    return found ? victim : kNoVictimSlot;
}""",
    """size_t MixtureOfExpertsImpl::pick_mfu_ready() {
    size_t victim = kNoVictimSlot;
    uint64_t most = 0;
    bool found = false;
    for (size_t i = 0; i < slot_meta_.size(); ++i) {
        if (!expert_slot_ready_[i] || slot_meta_[i].expert_id == -1) {
            continue;
        }
        if (slot_meta_[i].access_count >= most) {
            most = slot_meta_[i].access_count;
            victim = i;
            found = true;
        }
    }
    return found ? victim : kNoVictimSlot;
}""": """size_t MixtureOfExpertsImpl::pick_mfu_ready() {
    size_t victim = kNoVictimSlot;
    uint64_t most = 0;
    bool found = false;
    for (size_t i = 0; i < slot_meta_.size(); ++i) {
        if (!expert_slot_ready_[i] || slot_meta_[i].expert_id == -1) {
            continue;
        }
        if (std::find(currently_selected_experts_.begin(), currently_selected_experts_.end(), slot_meta_[i].expert_id) != currently_selected_experts_.end()) {
            continue;
        }
        if (slot_meta_[i].access_count >= most) {
            most = slot_meta_[i].access_count;
            victim = i;
            found = true;
        }
    }
    return found ? victim : kNoVictimSlot;
}""",
    """size_t MixtureOfExpertsImpl::pick_clock_ready() {
    if (slot_meta_.empty()) {
        return kNoVictimSlot;
    }
    for (size_t tries = 0; tries < slot_meta_.size() * 2; ++tries) {
        size_t s = clock_hand_;
        clock_hand_ = (clock_hand_ + 1) % slot_meta_.size();
        if (!expert_slot_ready_[s]) {
            continue;
        }
        if (slot_meta_[s].expert_id == -1) {
            return s;
        }
        auto& m = slot_meta_[s];
        if (m.clock_bit == 0) {
            return s;
        }
        m.clock_bit = 0;
    }
    return kNoVictimSlot;
}""": """size_t MixtureOfExpertsImpl::pick_clock_ready() {
    if (slot_meta_.empty()) {
        return kNoVictimSlot;
    }
    for (size_t tries = 0; tries < slot_meta_.size() * 2; ++tries) {
        size_t s = clock_hand_;
        clock_hand_ = (clock_hand_ + 1) % slot_meta_.size();
        if (!expert_slot_ready_[s]) {
            continue;
        }
        if (slot_meta_[s].expert_id == -1) {
            return s;
        }
        auto& m = slot_meta_[s];
        if (std::find(currently_selected_experts_.begin(), currently_selected_experts_.end(), m.expert_id) != currently_selected_experts_.end()) {
            continue;
        }
        if (m.clock_bit == 0) {
            return s;
        }
        m.clock_bit = 0;
    }
    return kNoVictimSlot;
}""",
    """size_t MixtureOfExpertsImpl::pick_lfru_ready() {
    size_t victim = kNoVictimSlot;
    double min_score = std::numeric_limits<double>::max();
    bool found = false;
    for (size_t i = 0; i < slot_meta_.size(); ++i) {
        if (!expert_slot_ready_[i] || slot_meta_[i].expert_id == -1) {
            continue;
        }
        const auto& a = slot_meta_[i];
        double score = static_cast<double>(a.access_count) / (access_clock_ - a.last_access + 1);
        if (score < min_score) {
            min_score = score;
            victim = i;
            found = true;
        }
    }
    return found ? victim : kNoVictimSlot;
}""": """size_t MixtureOfExpertsImpl::pick_lfru_ready() {
    size_t victim = kNoVictimSlot;
    double min_score = std::numeric_limits<double>::max();
    bool found = false;
    for (size_t i = 0; i < slot_meta_.size(); ++i) {
        if (!expert_slot_ready_[i] || slot_meta_[i].expert_id == -1) {
            continue;
        }
        const auto& a = slot_meta_[i];
        if (std::find(currently_selected_experts_.begin(), currently_selected_experts_.end(), a.expert_id) != currently_selected_experts_.end()) {
            continue;
        }
        double score = static_cast<double>(a.access_count) / (access_clock_ - a.last_access + 1);
        if (score < min_score) {
            min_score = score;
            victim = i;
            found = true;
        }
    }
    return found ? victim : kNoVictimSlot;
}"""
}

for old, new_impl in replacements.items():
    if old in content:
        content = content.replace(old, new_impl)
    else:
        print(f"Warning: Could not find block starting with {old[:40]}")

with open('src/unified_llm_w4a16_predict/unified_llm_w4a16.cpp', 'w') as f:
    f.write(content)

print("Patch complete for predict.")
