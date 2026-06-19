#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <iostream>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>
#include <numeric>
#include <functional>
#include <cstring>
#include <cmath>
#include <future>
#include <algorithm>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static void drop_page_cache() {
    sync();
    int fd = open("/proc/sys/vm/drop_caches", O_WRONLY);
    if (fd == -1) {
        static bool warned = false;
        if (!warned) {
            std::cerr << "[warn] Cannot open /proc/sys/vm/drop_caches.\n"
                         "       Re-run with sudo for true SSD numbers.\n"
                         "       Results below may be page-cache hits.\n\n";
            warned = true;
        }
        return;
    }
    if (write(fd, "3\n", 2) != 2)
        std::cerr << "[warn] Failed to write to drop_caches\n";
    close(fd);
}

static size_t file_size_bytes(const std::string& path) {
    struct stat sb;
    if (stat(path.c_str(), &sb) != 0)
        throw std::runtime_error("stat failed: " + path);
    return static_cast<size_t>(sb.st_size);
}

static size_t align_up(size_t v, size_t align) {
    return (v + align - 1) & ~(align - 1);
}

struct Stats {
    double mean_us, stddev_us, min_us, max_us, median_us;
};

static Stats compute_stats(std::vector<double>& s) {
    std::sort(s.begin(), s.end());
    double sum = std::accumulate(s.begin(), s.end(), 0.0);
    double mean = sum / s.size();
    double sq = 0;
    for (auto v : s) sq += (v - mean) * (v - mean);
    return {mean, std::sqrt(sq / s.size()), s.front(), s.back(),
            s[s.size() / 2]};
}

static void print_stats(const std::string& label, Stats s, size_t bytes) {
    double bw = (bytes / 1e9) / (s.mean_us * 1e-6);
    std::printf("%-40s  mean=%7.0f us  median=%7.0f us  "
                "min=%7.0f us  max=%7.0f us  BW=%5.2f GB/s\n",
                label.c_str(),
                s.mean_us, s.median_us, s.min_us, s.max_us, bw);
}

static void pread_exact(int fd, char* dest, size_t size, off_t offset, const std::string& ctx) {
    size_t read_sz = align_up(size, 512);
    size_t done = 0;
    while (done < read_sz) {
        ssize_t ret = pread(fd, dest + done, read_sz - done, offset + static_cast<off_t>(done));
        if (ret < 0) {
            throw std::runtime_error("pread failed (" + ctx + "): " + std::string(strerror(errno)));
        }
        if (ret == 0) break; // EOF
        done += static_cast<size_t>(ret);
    }
}

struct AlignedBuffer {
    void* ptr = nullptr;
    size_t capacity = 0;

    void ensure(size_t size) {
        size_t required = align_up(size, 512) + 512;
        if (capacity < required) {
            if (ptr) free(ptr);
            if (posix_memalign(&ptr, 4096, required) != 0) {
                throw std::runtime_error("posix_memalign failed");
            }
            capacity = required;
        }
    }

    ~AlignedBuffer() {
        if (ptr) free(ptr);
    }

    AlignedBuffer() = default;
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    
    AlignedBuffer(AlignedBuffer&& other) noexcept : ptr(other.ptr), capacity(other.capacity) {
        other.ptr = nullptr;
        other.capacity = 0;
    }
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr) free(ptr);
            ptr = other.ptr;
            capacity = other.capacity;
            other.ptr = nullptr;
            other.capacity = 0;
        }
        return *this;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Packed Loader
// ─────────────────────────────────────────────────────────────────────────────

struct ExpkDesc {
    char name[32];
    uint64_t offset;
    uint64_t size;
};

static constexpr uint32_t kExpkMagicLe = 0x4B505845u;

size_t load_expert_packed(const std::string& path, bool parallel, std::vector<AlignedBuffer>& bufs) {
    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd == -1) fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) throw std::runtime_error("Cannot open packed expert: " + path);

    alignas(512) char header_buf[512];
    if (pread(fd, header_buf, 512, 0) < 512) {
        close(fd);
        throw std::runtime_error("Short header read: " + path);
    }

    if (*reinterpret_cast<const uint32_t*>(header_buf) != kExpkMagicLe) {
        close(fd);
        throw std::runtime_error("Bad EXPK magic: " + path);
    }

    const uint32_t num_tensors = *reinterpret_cast<const uint32_t*>(header_buf + 4);
    const ExpkDesc* descs = reinterpret_cast<const ExpkDesc*>(header_buf + 8);
    
    if (bufs.size() < num_tensors) bufs.resize(num_tensors);

    size_t total_bytes = 0;
    
    if (parallel) {
        std::vector<std::future<void>> futures;
        for (uint32_t i = 0; i < num_tensors; ++i) {
            size_t size = descs[i].size;
            off_t offset = descs[i].offset;
            total_bytes += size;
            
            bufs[i].ensure(size);
            char* ptr = static_cast<char*>(bufs[i].ptr);
            futures.push_back(std::async(std::launch::async, pread_exact, fd, ptr, size, offset, path));
        }
        for (auto& f : futures) f.get();
    } else {
        for (uint32_t i = 0; i < num_tensors; ++i) {
            size_t size = descs[i].size;
            off_t offset = descs[i].offset;
            total_bytes += size;
            
            bufs[i].ensure(size);
            char* ptr = static_cast<char*>(bufs[i].ptr);
            pread_exact(fd, ptr, size, offset, path);
        }
    }
    
    close(fd);
    return total_bytes;
}

// ─────────────────────────────────────────────────────────────────────────────
// Unpacked Loader
// ─────────────────────────────────────────────────────────────────────────────

size_t load_expert_unpacked(const std::string& prefix_path, bool parallel, std::vector<AlignedBuffer>& bufs) {
    const std::vector<std::string> suffixes = {
        "_gate.qweight.bin", "_gate.scales.bin", "_gate.zeros.bin",
        "_up.qweight.bin", "_up.scales.bin", "_up.zeros.bin",
        "_down.qweight.bin", "_down.scales.bin", "_down.zeros.bin"
    };

    if (bufs.size() < suffixes.size()) bufs.resize(suffixes.size());

    auto load_one = [&](int i) -> size_t {
        std::string path = prefix_path + suffixes[i];
        size_t size = file_size_bytes(path);
        
        int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
        if (fd == -1) fd = open(path.c_str(), O_RDONLY);
        if (fd == -1) throw std::runtime_error("Cannot open: " + path);
        
        bufs[i].ensure(size);
        char* ptr = static_cast<char*>(bufs[i].ptr);
        pread_exact(fd, ptr, size, 0, path);
        close(fd);
        return size;
    };

    size_t total_bytes = 0;
    
    if (parallel) {
        std::vector<std::future<size_t>> futures;
        for (size_t i = 0; i < suffixes.size(); ++i) {
            futures.push_back(std::async(std::launch::async, load_one, i));
        }
        for (auto& f : futures) {
            total_bytes += f.get();
        }
    } else {
        for (size_t i = 0; i < suffixes.size(); ++i) {
            total_bytes += load_one(i);
        }
    }
    
    return total_bytes;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <packed_dir> <unpacked_dir> <iters=10> [num_experts=32]\n";
        return 1;
    }
    
    std::string packed_dir = argv[1];
    std::string unpacked_dir = argv[2];
    int ITERS = std::atoi(argv[3]);
    int num_experts = (argc >= 5) ? std::atoi(argv[4]) : 32;
    std::vector<std::string> packed_paths;
    std::vector<std::string> unpacked_prefixes;
    
    for (int i = 0; i < num_experts; ++i) {
        packed_paths.push_back(packed_dir + "/layer_0_expert_" + std::to_string(i) + ".bin");
        unpacked_prefixes.push_back(unpacked_dir + "/layer_0_expert_" + std::to_string(i));
    }
    
    for (const auto& p : packed_paths) {
        if (access(p.c_str(), F_OK) != 0) {
            std::cerr << "File not found: " << p << "\n";
            return 1;
        }
    }

    std::printf("Testing with %d experts. Iterations: %d.\n", num_experts, ITERS);
    std::printf("Page cache will be dropped before each iteration (if run with sudo).\n\n");

    std::vector<std::vector<AlignedBuffer>> bufs(num_experts);

    auto run_test = [&](const std::string& label, size_t& bytes_read, std::function<void()> fn) -> Stats {
        std::vector<double> samples;
        samples.reserve(ITERS);
        for (int i = 0; i < ITERS; ++i) {
            drop_page_cache();
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        auto s = compute_stats(samples);
        print_stats(label, s, bytes_read);
        return s;
    };

    size_t total_bytes;

    std::printf("--- 1 Expert (Intra-Expert Parallelization) ---\n");
    
    run_test("[Unpacked] 1 Expert Sequential", total_bytes, [&]() {
        total_bytes = load_expert_unpacked(unpacked_prefixes[0], false, bufs[0]);
    });
    
    run_test("[Unpacked] 1 Expert Parallel  ", total_bytes, [&]() {
        total_bytes = load_expert_unpacked(unpacked_prefixes[0], true, bufs[0]);
    });

    run_test("[Packed]   1 Expert Sequential", total_bytes, [&]() {
        total_bytes = load_expert_packed(packed_paths[0], false, bufs[0]);
    });
    
    run_test("[Packed]   1 Expert Parallel  ", total_bytes, [&]() {
        total_bytes = load_expert_packed(packed_paths[0], true, bufs[0]);
    });

    std::printf("\n--- %d Experts (Inter-Expert Parallelization) ---\n", num_experts);
    
    run_test("[Unpacked] K Experts (Seq Loop, Par Tensors)", total_bytes, [&]() {
        total_bytes = 0;
        for (int i = 0; i < num_experts; ++i) {
            total_bytes += load_expert_unpacked(unpacked_prefixes[i], true, bufs[i]);
        }
    });

    run_test("[Unpacked] K Experts (Par Loop, Par Tensors)", total_bytes, [&]() {
        total_bytes = 0;
        std::vector<std::future<size_t>> futures;
        for (int i = 0; i < num_experts; ++i) {
            futures.push_back(std::async(std::launch::async, [&, i]() {
                return load_expert_unpacked(unpacked_prefixes[i], true, bufs[i]);
            }));
        }
        for (auto& f : futures) total_bytes += f.get();
    });

    run_test("[Packed]   K Experts (Seq Loop, Par Tensors)", total_bytes, [&]() {
        total_bytes = 0;
        for (int i = 0; i < num_experts; ++i) {
            total_bytes += load_expert_packed(packed_paths[i], true, bufs[i]);
        }
    });

    run_test("[Packed]   K Experts (Par Loop, Par Tensors)", total_bytes, [&]() {
        total_bytes = 0;
        std::vector<std::future<size_t>> futures;
        for (int i = 0; i < num_experts; ++i) {
            futures.push_back(std::async(std::launch::async, [&, i]() {
                return load_expert_packed(packed_paths[i], true, bufs[i]);
            }));
        }
        for (auto& f : futures) total_bytes += f.get();
    });

    return 0;
}
