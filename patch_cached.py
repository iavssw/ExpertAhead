import os

with open('src/unified_llm_w4a16_cached/unified_llm_w4a16.cpp', 'r') as f:
    content = f.read()

replacements = {
    """size_t MixtureOfExpertsImpl::pick_lru() {
    size_t victim = 0;
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i < slot_meta_.size(); ++i) {
        if (cache_policy_ == CachePolicy::PREFILL) {
            if (std::find(locked_experts_.begin(), locked_experts_.end(), slot_meta_[i].expert_id) != locked_experts_.end()) continue;
        }
        if (slot_meta_[i].last_access < oldest) {
            oldest = slot_meta_[i].last_access;
            victim = i;
        }
    }
    return victim;
}""": """size_t MixtureOfExpertsImpl::pick_lru() {
    size_t victim = 0;
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    bool found = false;
    for (size_t i = 0; i < slot_meta_.size(); ++i) {
        if (std::find(currently_selected_experts_.begin(), currently_selected_experts_.end(), slot_meta_[i].expert_id) != currently_selected_experts_.end()) {
            continue;
        }
        if (cache_policy_ == CachePolicy::PREFILL) {
            if (std::find(locked_experts_.begin(), locked_experts_.end(), slot_meta_[i].expert_id) != locked_experts_.end()) continue;
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
            if (slot_meta_[i].last_access < oldest) {
                oldest = slot_meta_[i].last_access;
                victim = i;
            }
        }
    }
    return victim;
}""",
    """size_t MixtureOfExpertsImpl::pick_mru() {
    return std::max_element(slot_meta_.begin(), slot_meta_.end(),
        [](const auto& a, const auto& b){ return a.last_access < b.last_access; })
        - slot_meta_.begin();
}""": """size_t MixtureOfExpertsImpl::pick_mru() {
    size_t victim = 0;
    uint64_t newest = 0;
    bool found = false;
    for (size_t i = 0; i < slot_meta_.size(); ++i) {
        if (std::find(currently_selected_experts_.begin(), currently_selected_experts_.end(), slot_meta_[i].expert_id) != currently_selected_experts_.end()) {
            continue;
        }
        if (slot_meta_[i].last_access >= newest) {
            newest = slot_meta_[i].last_access;
            victim = i;
            found = true;
        }
    }
    if (!found) return 0;
    return victim;
}""",
    """size_t MixtureOfExpertsImpl::pick_lfu() {
    return std::min_element(slot_meta_.begin(), slot_meta_.end(),
        [](const auto& a, const auto& b){ return a.access_count < b.access_count; })
        - slot_meta_.begin();
}""": """size_t MixtureOfExpertsImpl::pick_lfu() {
    size_t victim = 0;
    uint64_t least = std::numeric_limits<uint64_t>::max();
    bool found = false;
    for (size_t i = 0; i < slot_meta_.size(); ++i) {
        if (std::find(currently_selected_experts_.begin(), currently_selected_experts_.end(), slot_meta_[i].expert_id) != currently_selected_experts_.end()) {
            continue;
        }
        if (slot_meta_[i].access_count < least) {
            least = slot_meta_[i].access_count;
            victim = i;
            found = true;
        }
    }
    if (!found) return 0;
    return victim;
}""",
    """size_t MixtureOfExpertsImpl::pick_mfu() {
    return std::max_element(slot_meta_.begin(), slot_meta_.end(),
        [](const auto& a, const auto& b){ return a.access_count < b.access_count; })
        - slot_meta_.begin();
}""": """size_t MixtureOfExpertsImpl::pick_mfu() {
    size_t victim = 0;
    uint64_t most = 0;
    bool found = false;
    for (size_t i = 0; i < slot_meta_.size(); ++i) {
        if (std::find(currently_selected_experts_.begin(), currently_selected_experts_.end(), slot_meta_[i].expert_id) != currently_selected_experts_.end()) {
            continue;
        }
        if (slot_meta_[i].access_count >= most) {
            most = slot_meta_[i].access_count;
            victim = i;
            found = true;
        }
    }
    if (!found) return 0;
    return victim;
}""",
    """size_t MixtureOfExpertsImpl::pick_clock() {
    while (true) {
        auto& m = slot_meta_[clock_hand_];
        if (m.clock_bit == 0) {
            size_t victim = clock_hand_;
            clock_hand_ = (clock_hand_ + 1) % slot_meta_.size();
            return victim;
        }
        m.clock_bit = 0;  // give a second chance
        clock_hand_ = (clock_hand_ + 1) % slot_meta_.size();
    }
}""": """size_t MixtureOfExpertsImpl::pick_clock() {
    for (size_t tries = 0; tries < slot_meta_.size() * 2; ++tries) {
        size_t victim = clock_hand_;
        auto& m = slot_meta_[clock_hand_];
        clock_hand_ = (clock_hand_ + 1) % slot_meta_.size();
        if (std::find(currently_selected_experts_.begin(), currently_selected_experts_.end(), m.expert_id) != currently_selected_experts_.end()) {
            continue;
        }
        if (m.clock_bit == 0) {
            return victim;
        }
        m.clock_bit = 0;  // give a second chance
    }
    return 0; // Fallback
}""",
    """size_t MixtureOfExpertsImpl::pick_lfru() {
    return std::min_element(slot_meta_.begin(), slot_meta_.end(),
        [&](const auto& a, const auto& b) {
            double score_a = (double)a.access_count / (access_clock_ - a.last_access + 1);
            double score_b = (double)b.access_count / (access_clock_ - b.last_access + 1);
            return score_a < score_b;
        }) - slot_meta_.begin();
}""": """size_t MixtureOfExpertsImpl::pick_lfru() {
    size_t victim = 0;
    double min_score = std::numeric_limits<double>::max();
    bool found = false;
    for (size_t i = 0; i < slot_meta_.size(); ++i) {
        if (std::find(currently_selected_experts_.begin(), currently_selected_experts_.end(), slot_meta_[i].expert_id) != currently_selected_experts_.end()) {
            continue;
        }
        double score = (double)slot_meta_[i].access_count / (access_clock_ - slot_meta_[i].last_access + 1);
        if (score < min_score) {
            min_score = score;
            victim = i;
            found = true;
        }
    }
    if (!found) return 0;
    return victim;
}"""
}

for old, new_impl in replacements.items():
    if old in content:
        content = content.replace(old, new_impl)
    else:
        print(f"Warning: Could not find block starting with {old[:40]}")

with open('src/unified_llm_w4a16_cached/unified_llm_w4a16.cpp', 'w') as f:
    f.write(content)

print("Patch complete for cached.")
