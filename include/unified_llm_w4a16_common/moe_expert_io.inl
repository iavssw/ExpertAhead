// Shared O_DIRECT pread for unpacked expert .bin tensors.
// Include from unified_llm_w4a16_{cached,predict}/helper.cpp inside the translation unit.
//
// Expert loads always use O_DIRECT (no buffered fallback). Misaligned tensor sizes
// are read via an aligned staging buffer; the syscall still bypasses page cache.

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

inline constexpr size_t kOdirectAlign = 512;

inline size_t align_up(size_t n, size_t align = kOdirectAlign) {
    return (n + align - 1) / align * align;
}

inline void log_odirect_expert_io_once() {
    static bool logged = false;
    if (!logged) {
        logged = true;
        std::cout << "[MoE I/O] Expert weight reads require O_DIRECT (no buffered fallback)."
                  << std::endl;
    }
}

inline void fadvise_dontneed_fd(int fd) {
    if (fd >= 0) {
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
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

inline void require_odirect_region(const std::string& ctx, uint64_t offset, uint64_t size) {
    if (offset % kOdirectAlign != 0 || size % kOdirectAlign != 0) {
        throw std::runtime_error(
            "O_DIRECT alignment required for " + ctx + " (offset=" + std::to_string(offset) + ", size=" +
            std::to_string(size) + "). Repack experts with 512-byte-aligned EXPK offsets.");
    }
}

inline int open_odirect_or_throw(const std::string& path) {
    const int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd == -1) {
        throw std::runtime_error("O_DIRECT open failed for: " + path + " (" + strerror(errno) + ")");
    }
    if ((fcntl(fd, F_GETFL) & O_DIRECT) == 0) {
        close(fd);
        throw std::runtime_error("O_DIRECT not active after open for: " + path);
    }
    return fd;
}

inline void pread_exact_fd(int fd, char* dst, size_t nbytes, off_t offset, const std::string& path) {
    size_t done = 0;
    while (done < nbytes) {
        const ssize_t ret = pread(fd, dst + done, nbytes - done, offset + static_cast<off_t>(done));
        if (ret < 0) {
            throw std::runtime_error("pread failed for: " + path + " (" + strerror(errno) + ")");
        }
        if (ret == 0) {
            throw std::runtime_error("pread EOF for: " + path + " (wanted " + std::to_string(nbytes) +
                                     " bytes at offset " + std::to_string(offset) + ", got " + std::to_string(done) +
                                     ")");
        }
        done += static_cast<size_t>(ret);
    }
}

// O_DIRECT pread into dest_ptr. Never falls back to buffered I/O.
inline void pread_odirect_region(int fd, void* dest_ptr, size_t copy_size, off_t offset, const std::string& path) {
    char* dest = static_cast<char*>(dest_ptr);
    const bool dest_aligned = (reinterpret_cast<uintptr_t>(dest) % kOdirectAlign) == 0;
    const bool size_aligned = (copy_size % kOdirectAlign) == 0;
    const bool offset_aligned = (static_cast<uint64_t>(offset) % kOdirectAlign) == 0;

    if (dest_aligned && size_aligned && offset_aligned) {
        pread_exact_fd(fd, dest, copy_size, offset, path);
        return;
    }

    if (!offset_aligned || copy_size % kOdirectAlign != 0) {
        throw std::runtime_error(
            "O_DIRECT alignment required for read of " + path + " (offset=" + std::to_string(offset) + ", size=" +
            std::to_string(copy_size) + "). Repack experts with 512-byte-aligned EXPK offsets.");
    }

    const size_t read_size = align_up(copy_size, kOdirectAlign);
    void* staging_raw = nullptr;
    if (posix_memalign(&staging_raw, kOdirectAlign, read_size) != 0) {
        throw std::runtime_error("posix_memalign failed for O_DIRECT staging: " + path);
    }
    char* staging = static_cast<char*>(staging_raw);

    pread_exact_fd(fd, staging, read_size, offset, path);
    std::memcpy(dest, staging, copy_size);
    free(staging_raw);
}

inline void read_bin_tensor_pread(const std::string& path, void* dest_ptr, size_t copy_size) {
    log_odirect_expert_io_once();

    struct stat sb {};
    const int fd_stat = open(path.c_str(), O_RDONLY);
    if (fd_stat == -1) {
        throw std::runtime_error("Could not stat-open file: " + path + " (" + strerror(errno) + ")");
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

    const int fd = open_odirect_or_throw(path);
    pread_odirect_region(fd, dest_ptr, copy_size, 0, path);
    fadvise_dontneed_fd(fd);
    close(fd);
}

}  // namespace unified_llm_w4a16_common
