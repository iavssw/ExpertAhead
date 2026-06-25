import os
import glob
import time

history_dirs = [
    '/home/michael/.antigravity-server/data/User/History',
    '/home/michael/.antigravity-ide-server/data/User/History'
]

def find_latest_file(signature, original_path_hint):
    candidates = []
    for hdir in history_dirs:
        if not os.path.exists(hdir): continue
        for root, _, files in os.walk(hdir):
            for file in files:
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if signature in content:
                            mtime = os.path.getmtime(path)
                            candidates.append((mtime, path, content))
                except Exception:
                    pass
    if not candidates:
        return None
    # Sort by modification time descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # We want to find the latest version BEFORE my python script corrupted it,
    # or before I checked it out.
    # The user might have manually edited it after my script failed.
    # Let's just print the top 5 candidates.
    return candidates[:5]

# For cached backend, let's look for calculate_generation_perplexity
cached_candidates = find_latest_file("calculate_generation_perplexity", "cached")
print("Cached Candidates:")
if cached_candidates:
    for i, (mtime, p, content) in enumerate(cached_candidates):
        print(f"[{i}] {time.ctime(mtime)} : {p}")
        # write them to a temp file to inspect
        with open(f"cached_cand_{i}.cpp", "w") as f:
            f.write(content)
else:
    print("None found")

# For predict backend, let's look for expert_lru_order_ (which is what HEAD had, but wait, the error says expert_lru_order_ was NOT declared).
# This means the header REMOVED expert_lru_order_, so the latest cpp file should NOT have expert_lru_order_, or the newest one with `MixtureOfExpertsImpl::predict_sync`?
# Let's look for a string that the user recently added to predict backend.
predict_candidates = find_latest_file("MixtureOfExpertsImpl::pick_lru_ready()", "predict")
print("\nPredict Candidates:")
if predict_candidates:
    for i, (mtime, p, content) in enumerate(predict_candidates):
        print(f"[{i}] {time.ctime(mtime)} : {p}")
        with open(f"predict_cand_{i}.cpp", "w") as f:
            f.write(content)
else:
    print("None found")
