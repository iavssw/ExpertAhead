// Shared O_DIRECT pread for unpacked expert .bin tensors.
// Include from unified_llm_w4a16_{cached,predict}/helper.cpp inside the translation unit.

#pragma once

#include <cstdint>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace unified_llm_w4a16_common {

inline size_t align_up(size_t n, size_t align = 512) {
    return (n + align - 1) / align * align;
}

// HETEROPREDICT_STRICT_SSD_IO=1: refuse buffered fallbacks; every expert byte comes from O_DIRECT.
inline bool strict_ssd_io() {
    const char* v = std::getenv("HETEROPREDICT_STRICT_SSD_IO");
    return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y');
}

inline void log_strict_ssd_io_once() {
    static bool logged = false;
    if (!logged && strict_ssd_io()) {
        logged = true;
        std::cout << "[MoE I/O] HETEROPREDICT_STRICT_SSD_IO=1: O_DIRECT required (no buffered expert reads)."
                  << std::endl;
    }
}

inline void fadvise_dontneed_fd(int fd) {
    if (fd >= 0) {
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    }
}

inline void fadvise_dontneed_path(const std::string& path) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
        fadvise_dontneed_fd(fd);
        close(fd);
    }
}

// Set HETEROPREDICT_SEQUENTIAL_EXPERT_IO=1 to disable parallel pread dispatches inside
// each expert load (packed + unpacked). For A/B testing LRU TPS vs parallel I/O.
inline bool sequential_expert_io_loads() {
    const char* v = std::getenv("HETEROPREDICT_SEQUENTIAL_EXPERT_IO");
    return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y');
}

inline void log_sequential_expert_io_once() {
    static bool logged = false;
    if (!logged && sequential_expert_io_loads()) {
        logged = true;
        std::cout << "[MoE I/O] HETEROPREDICT_SEQUENTIAL_EXPERT_IO=1: sequential pread per tensor "
                     "(no std::async dispatches within expert load)."
                  << std::endl;
    }
}

// Set HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO=1 to load predicted / on-demand experts
// one-at-a-time instead of dispatching parallel load_expert_weights() calls.
// Default (unset): parallel inter-expert loads (SSD queue depth).
inline bool parallel_inter_expert_loads() {
    const char* v = std::getenv("HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO");
    return !(v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y'));
}

inline void log_parallel_inter_expert_io_once() {
    static bool logged = false;
    if (!logged) {
        logged = true;
        if (parallel_inter_expert_loads()) {
            std::cout << "[MoE I/O] parallel inter-expert loads enabled (default). "
                         "Set HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO=1 for sequential."
                      << std::endl;
        } else {
            std::cout << "[MoE I/O] HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO=1: sequential "
                         "load_expert_weights per batch (no parallel across experts)."
                      << std::endl;
        }
    }
}

inline bool env_truthy(const char* name) {
    const char* v = std::getenv(name);
    return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y');
}

// Umbrella: Jun-17 pre-batch behavior (per-expert decode loads + sequential prefetch thread).
// Pair with HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO=1 for the full historical smoke setup.
inline bool legacy_expert_io() {
    return env_truthy("HETEROPREDICT_LEGACY_EXPERT_IO");
}

// Per-expert ensure_expert_cached in decode forward (no ensure_experts_cached_batch).
inline bool legacy_per_expert_cache() {
    return legacy_expert_io() || env_truthy("HETEROPREDICT_LEGACY_PER_EXPERT_CACHE");
}

// Sequential one-at-a-time load_predicted_experts (predict backend only).
inline bool legacy_prefetch_loads() {
    return legacy_expert_io() || env_truthy("HETEROPREDICT_LEGACY_PREFETCH_LOADS");
}

inline void log_legacy_expert_io_once() {
    static bool logged = false;
    if (!logged && (legacy_per_expert_cache() || legacy_prefetch_loads())) {
        logged = true;
        std::cout << "[MoE I/O] Legacy expert I/O:";
        if (legacy_per_expert_cache()) {
            std::cout << " per-expert cache (no batch)";
        }
        if (legacy_prefetch_loads()) {
            std::cout << " sequential prefetch loads";
        }
        if (legacy_expert_io()) {
            std::cout << " [HETEROPREDICT_LEGACY_EXPERT_IO=1]";
        }
        std::cout << std::endl;
    }
}

inline void read_bin_tensor_pread(const std::string& path, void* dest_ptr, size_t copy_size) {
    log_strict_ssd_io_once();
    static constexpr size_t kAlign = 512;

    struct stat sb {};
    const int fd_stat = open(path.c_str(), O_RDONLY);
    if (fd_stat == -1) {
        throw std::runtime_error("Could not open file: " + path + " (" + strerror(errno) + ")");
    }
    if (fstat(fd_stat, &sb) == -1) {
        close(fd_stat);
        throw std::runtime_error("fstat failed for: " + path);
    }
    close(fd_stat);

    const size_t file_size = static_cast<size_t>(sb.st_size);
    if (file_size != copy_size) {
        throw std::runtime_error(
            "File size mismatch for " + path + " (expected " + std::to_string(copy_size) + ", got " +
            std::to_string(file_size) + ")");
    }

    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    const bool used_odirect = (fd != -1);
    if (!used_odirect) {
        if (strict_ssd_io()) {
            throw std::runtime_error("O_DIRECT open failed for: " + path + " (" + strerror(errno) + ")");
        }
        fd = open(path.c_str(), O_RDONLY);
    }
    if (fd == -1) {
        throw std::runtime_error("Could not open file: " + path + " (" + strerror(errno) + ")");
    }

    auto read_exact = [&](char* dst, size_t nbytes, off_t offset) {
        size_t done = 0;
        while (done < nbytes) {
            const ssize_t ret = pread(fd, dst + done, nbytes - done, offset + static_cast<off_t>(done));
            if (ret < 0) {
                throw std::runtime_error("pread failed for: " + path + " (" + strerror(errno) + ")");
            }
            if (ret == 0) {
                throw std::runtime_error("pread EOF for: " + path);
            }
            done += static_cast<size_t>(ret);
        }
    };

    char* dest = static_cast<char*>(dest_ptr);
    if (used_odirect) {
        const size_t odirect_bytes = (copy_size / kAlign) * kAlign;
        const size_t read_size = align_up(copy_size, kAlign);
        void* staging_raw = nullptr;
        if (posix_memalign(&staging_raw, kAlign, read_size) != 0) {
            close(fd);
            throw std::runtime_error("posix_memalign failed for: " + path);
        }
        char* staging = static_cast<char*>(staging_raw);

        read_exact(staging, odirect_bytes, 0);
        if (copy_size > odirect_bytes) {
            read_exact(staging + odirect_bytes, kAlign, static_cast<off_t>(odirect_bytes));
        }
        std::memcpy(dest, staging, copy_size);
        free(staging_raw);
        fadvise_dontneed_fd(fd);
    } else {
        read_exact(dest, copy_size, 0);
    }
    close(fd);
}

}  // namespace unified_llm_w4a16_common
