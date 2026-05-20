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
#include <unistd.h>

namespace unified_llm_w4a16_common {

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

inline void read_bin_tensor_pread(const std::string& path, void* dest_ptr, size_t copy_size) {
    int flags = O_RDONLY | O_DIRECT;
    int fd = open(path.c_str(), flags);
    if (fd == -1 && errno == EINVAL) {
        fd = open(path.c_str(), O_RDONLY);
    }
    if (fd == -1) {
        throw std::runtime_error("Could not open file: " + path + " (" + strerror(errno) + ")");
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw std::runtime_error("fstat failed for: " + path);
    }
    const size_t file_size = static_cast<size_t>(sb.st_size);
    if (file_size != copy_size) {
        close(fd);
        throw std::runtime_error(
            "File size mismatch for " + path + " (expected " + std::to_string(copy_size) + ", got " +
            std::to_string(file_size) + ")");
    }

    const bool is_direct = (fcntl(fd, F_GETFL) & O_DIRECT) != 0;
    char* ptr = static_cast<char*>(dest_ptr);

    if (is_direct && (((uintptr_t)ptr % 512) != 0 || (copy_size % 512) != 0)) {
        int current_flags = fcntl(fd, F_GETFL);
        fcntl(fd, F_SETFL, current_flags & ~O_DIRECT);
    }

    size_t bytes_read = 0;
    while (bytes_read < copy_size) {
        const ssize_t ret = pread(fd, ptr + bytes_read, copy_size - bytes_read, static_cast<off_t>(bytes_read));
        if (ret <= 0) {
            break;
        }
        bytes_read += static_cast<size_t>(ret);
    }
    close(fd);
}

}  // namespace unified_llm_w4a16_common
