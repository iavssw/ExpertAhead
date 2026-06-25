import re

def patch_cached_file():
    with open('src/unified_llm_w4a16_cached/unified_llm_w4a16.cpp', 'r') as f:
        content = f.read()

    # We will replace the whole function bodies.
    # To be safe, we'll replace the block from "size_t MixtureOfExpertsImpl::pick_XYZ() {" 
    # to the closing brace "}" before the next function.

    replacements = {
        'pick_lru': """size_t MixtureOfExpertsImpl::pick_lru() {
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
        'pick_mru': """size_t MixtureOfExpertsImpl::pick_mru() {
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
        'pick_lfu': """size_t MixtureOfExpertsImpl::pick_lfu() {
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
        'pick_mfu': """size_t MixtureOfExpertsImpl::pick_mfu() {
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
        'pick_lfru': """size_t MixtureOfExpertsImpl::pick_lfru() {
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
}""",
        'pick_clock': """size_t MixtureOfExpertsImpl::pick_clock() {
    for (size_t tries = 0; tries < slot_meta_.size() * 2; ++tries) {
        size_t s = clock_hand_;
        auto& m = slot_meta_[s];
        clock_hand_ = (clock_hand_ + 1) % slot_meta_.size();
        
        if (std::find(currently_selected_experts_.begin(), currently_selected_experts_.end(), m.expert_id) != currently_selected_experts_.end()) {
            continue;
        }
        
        if (m.clock_bit == 0) {
            return s;
        }
        m.clock_bit = 0;  // give a second chance
    }
    // Fallback
    return 0;
}"""
    }

    for func_name, new_impl in replacements.items():
        pattern = r"size_t MixtureOfExpertsImpl::" + func_name + r"\(\) \{.*?(?=\nsize_t MixtureOfExpertsImpl::pick_|\nsize_t MixtureOfExpertsImpl::pick_random|\ntorch::Tensor)"
        content = re.sub(pattern, new_impl + "\n", content, flags=re.DOTALL)
    
    with open('src/unified_llm_w4a16_cached/unified_llm_w4a16.cpp', 'w') as f:
        f.write(content)

patch_cached_file()
print("Cached file patched successfully.")
