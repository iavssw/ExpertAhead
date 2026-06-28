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
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/uio.h>

// ─────────────────────────────────────────────────────────────────────────────
// Thread pool — mirrors model's IOThreadPool (io_thread_pool.hpp) exactly:
//   pre-warmed, persistent, 16 threads by default.
// ─────────────────────────────────────────────────────────────────────────────

class IOThreadPool {
public:
    IOThreadPool(size_t threads) : stop_(false) {
        for (size_t i = 0; i < threads; ++i)
            workers_.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        using R = std::invoke_result_t<F, Args...>;
        auto pkg = std::make_shared<std::packaged_task<R()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<R> res = pkg->get_future();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            tasks_.emplace([pkg]() { (*pkg)(); });
        }
        cv_.notify_one();
        return res;
    }

    ~IOThreadPool() {
        { std::unique_lock<std::mutex> lock(mutex_); stop_ = true; }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_;
};

static IOThreadPool& get_io_pool() {
    // 16 threads matches the model's default HETEROPREDICT_IO_THREADS=16
    static IOThreadPool pool(16);
    return pool;
}

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
    return {mean, std::sqrt(sq / s.size()), s.front(), s.back(), s[s.size() / 2]};
}

static void print_stats(const std::string& label, Stats s, size_t bytes) {
    double bw = (bytes / 1e9) / (s.mean_us * 1e-6);
    std::printf("%-42s  mean=%7.0f us  median=%7.0f us  "
                "min=%7.0f us  max=%7.0f us  BW=%5.2f GB/s\n",
                label.c_str(), s.mean_us, s.median_us, s.min_us, s.max_us, bw);
}

// ─────────────────────────────────────────────────────────────────────────────
// Pinned aligned buffer (mirrors torch::empty(..., pinned_memory(true)))
// ─────────────────────────────────────────────────────────────────────────────

struct AlignedBuffer {
    void* ptr = nullptr;
    size_t capacity = 0;

    void ensure(size_t size) {
        size_t required = align_up(size, 512) + 512;
        if (capacity < required) {
            if (ptr) { munlock(ptr, capacity); free(ptr); }
            if (posix_memalign(&ptr, 4096, required) != 0)
                throw std::runtime_error("posix_memalign failed");
            mlock(ptr, required); // keep pages in RAM — mirrors pinned_memory(true)
            capacity = required;
        }
    }

    ~AlignedBuffer() { if (ptr) { munlock(ptr, capacity); free(ptr); } }

    AlignedBuffer() = default;
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& o) noexcept : ptr(o.ptr), capacity(o.capacity) {
        o.ptr = nullptr; o.capacity = 0;
    }
    AlignedBuffer& operator=(AlignedBuffer&& o) noexcept {
        if (this != &o) {
            if (ptr) free(ptr);
            ptr = o.ptr; capacity = o.capacity;
            o.ptr = nullptr; o.capacity = 0;
        }
        return *this;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Expert loader — exact replica of the model's load_expert_worker
//   (moe_expert_load_packed.inl):
//     O_DIRECT open
//     → 512-byte aligned header read
//     → preadv scatter into 9 sorted iovecs
//     → posix_fadvise(DONTNEED) + close via Finalizer RAII
// ─────────────────────────────────────────────────────────────────────────────

struct ExpkDesc {
    char     name[32];
    uint64_t offset;
    uint64_t size;
};

static constexpr uint32_t kExpkMagicLe = 0x4B505845u; // "EXPK"

size_t load_expert(const std::string& path, std::vector<AlignedBuffer>& bufs) {
    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd == -1)
        throw std::runtime_error("O_DIRECT open failed: " + path +
                                 " (" + strerror(errno) + ")");

    // Mirrors model's Finalizer: fadvise DONTNEED then close on any exit path
    struct Finalizer {
        int fd_;
        ~Finalizer() {
            if (fd_ != -1) { posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED); close(fd_); }
        }
    } finalizer{fd};

    alignas(512) char header_buf[512];
    if (pread(fd, header_buf, 512, 0) < 512)
        throw std::runtime_error("Short header read: " + path);
    if (*reinterpret_cast<const uint32_t*>(header_buf) != kExpkMagicLe)
        throw std::runtime_error("Bad EXPK magic: " + path);

    const uint32_t num_tensors = *reinterpret_cast<const uint32_t*>(header_buf + 4);
    const ExpkDesc* descs_ptr  = reinterpret_cast<const ExpkDesc*>(header_buf + 8);
    std::vector<ExpkDesc> descs(descs_ptr, descs_ptr + num_tensors);

    if (bufs.size() < num_tensors) bufs.resize(num_tensors);

    // Sort by on-disk offset — matches model's IovMapping sort
    std::sort(descs.begin(), descs.end(),
              [](const ExpkDesc& a, const ExpkDesc& b) { return a.offset < b.offset; });

    size_t total_bytes = 0;
    std::vector<struct iovec> iov(num_tensors);
    uint64_t current_offset = 512; // tensors start immediately after 512-byte header
    for (uint32_t i = 0; i < num_tensors; ++i) {
        if (descs[i].offset != current_offset)
            throw std::runtime_error("Non-contiguous tensors in: " + path);
        size_t size = descs[i].size;
        total_bytes += size;
        bufs[i].ensure(size);
        iov[i].iov_base = bufs[i].ptr;
        iov[i].iov_len  = size;
        current_offset += size;
    }

    if (preadv(fd, iov.data(), static_cast<int>(num_tensors), 512) < 0)
        throw std::runtime_error("preadv failed for " + path + ": " + strerror(errno));

    return total_bytes;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <packed_dir> <iters> [num_experts=8]\n"
                  << "  packed_dir   — directory containing layer_L_expert_N.bin files\n"
                  << "  iters        — number of timed repetitions\n"
                  << "  num_experts  — number of experts to load per layer per iteration (default 8 = K)\n";
        return 1;
    }

    const std::string packed_dir = argv[1];
    const int ITERS       = std::atoi(argv[2]);
    const int num_experts = (argc >= 4) ? std::atoi(argv[3]) : 8;

    if (num_experts < 1 || num_experts > 8) {
        std::cerr << "num_experts must be between 1 and 8\n";
        return 1;
    }

    std::vector<std::vector<std::string>> packed_paths(48);
    for (int L = 0; L < 48; ++L) {
        packed_paths[L].reserve(num_experts);
        for (int i = 0; i < num_experts; ++i) {
            packed_paths[L].push_back(packed_dir + "/layer_" + std::to_string(L) + "_expert_" + std::to_string(i) + ".bin");
        }
    }

    for (int L = 0; L < 48; ++L) {
        for (const auto& p : packed_paths[L]) {
            if (access(p.c_str(), F_OK) != 0) {
                std::cerr << "File not found: " << p << "\n";
                return 1;
            }
        }
    }

    std::printf("Benchmark: %d experts per layer across 48 layers, %d iterations.\n", num_experts, ITERS);
    std::printf("Page cache dropped before each iteration (requires sudo for true SSD numbers).\n\n");

    // Pre-warm the pool *before* timing starts
    (void)get_io_pool();

    std::vector<std::vector<AlignedBuffer>> bufs(num_experts);

    auto run_test = [&](const std::string& label, size_t& bytes_read,
                        std::function<void()> fn) -> Stats {
        std::vector<double> samples;
        samples.reserve(ITERS);
        for (int i = 0; i < ITERS; ++i) {
            drop_page_cache();
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            samples.push_back(
                std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        auto s = compute_stats(samples);
        print_stats(label, s, bytes_read);
        return s;
    };

    size_t total_bytes = 0;

    // ── Sequential: one expert at a time (Whole Network)
    run_test("Sequential (K experts, serial loop)", total_bytes, [&]() {
        total_bytes = 0;
        for (int L = 0; L < 48; ++L) {
            for (int i = 0; i < num_experts; ++i) {
                total_bytes += load_expert(packed_paths[L][i], bufs[i]);
            }
        }
    });

    // ── Parallel: pre-warmed IOThreadPool (Whole Network)
    run_test("Parallel   (K experts, IOThreadPool)", total_bytes, [&]() {
        total_bytes = 0;
        auto& pool = get_io_pool();
        for (int L = 0; L < 48; ++L) {
            std::vector<std::future<size_t>> futures;
            futures.reserve(num_experts);
            for (int i = 0; i < num_experts; ++i) {
                futures.push_back(pool.enqueue([&, L, i]() {
                    return load_expert(packed_paths[L][i], bufs[i]);
                }));
            }
            for (auto& f : futures) {
                total_bytes += f.get();
            }
        }
    });

    return 0;
}
