/*
 * benchmark_loader.cpp
 *
 * Benchmarks disk→RAM→GPU transfer strategies for MoE expert prefetching.
 * Focuses on:
 *   1. True disk throughput (O_DIRECT, bypasses page cache)
 *   2. Async prefetch overlap with GPU compute via background pthread
 *   3. No dependencies beyond libc, libpthread, and LibTorch
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
 *   make -j$(nproc)
 *
 * Usage:
 *   sudo ./benchmark_loader <weights_dir> [iters=20]
 *   (sudo needed for drop_caches — without it results reflect page-cache hits)
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <torch/torch.h>

#include <iostream>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>
#include <numeric>
#include <functional>
#include <cstring>
#include <cmath>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>

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
    std::printf("%-44s  mean=%7.0f us  median=%7.0f us  "
                "p0=%7.0f us  p100=%7.0f us  BW=%5.2f GB/s\n",
                label.c_str(),
                s.mean_us, s.median_us, s.min_us, s.max_us, bw);
}

// ─────────────────────────────────────────────────────────────────────────────
// Load methods
// ─────────────────────────────────────────────────────────────────────────────

// 1. mmap + dynamic pinned alloc (original baseline, fd-leak fixed)
static torch::Tensor load_mmap_dynamic_pin(const std::string& path,
                                            torch::ScalarType dtype,
                                            const std::vector<int64_t>& shape) {
    size_t fsz = file_size_bytes(path);
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) throw std::runtime_error("open: " + path);

    auto tensor = torch::empty(shape, torch::TensorOptions()
                                          .dtype(dtype)
                                          .device(torch::kCPU)
                                          .pinned_memory(true));
    size_t copy_sz = std::min(fsz,
        static_cast<size_t>(tensor.numel() * tensor.element_size()));

    void* map = mmap(nullptr, fsz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);  // safe to close immediately after mmap
    if (map == MAP_FAILED) throw std::runtime_error("mmap failed");
    madvise(map, fsz, MADV_SEQUENTIAL);
    std::memcpy(tensor.data_ptr(), map, copy_sz);
    munmap(map, fsz);
    return tensor;
}

// 2. mmap into pre-allocated pinned buffer
static void load_mmap_prealloc(const std::string& path, torch::Tensor& buf) {
    size_t fsz = file_size_bytes(path);
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) throw std::runtime_error("open: " + path);

    size_t copy_sz = std::min(fsz,
        static_cast<size_t>(buf.numel() * buf.element_size()));
    void* map = mmap(nullptr, fsz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) throw std::runtime_error("mmap failed");
    madvise(map, fsz, MADV_SEQUENTIAL);
    std::memcpy(buf.data_ptr(), map, copy_sz);
    munmap(map, fsz);
}

// 3. O_DIRECT + pread — bypasses page cache, true SSD throughput.
//    Pinned memory from cudaMallocHost is page-aligned, satisfying O_DIRECT's
//    512-byte alignment requirement. Read size is rounded up to 512.
static void load_odirect_pread(const std::string& path, void* buf_ptr,
                                size_t tensor_bytes) {
    size_t fsz    = file_size_bytes(path);
    size_t read_sz = align_up(std::min(fsz, tensor_bytes), 512);

    if (reinterpret_cast<uintptr_t>(buf_ptr) % 512 != 0)
        throw std::runtime_error("Buffer not 512-byte aligned for O_DIRECT");

    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd == -1) throw std::runtime_error("open O_DIRECT: " + path);

    char* ptr = static_cast<char*>(buf_ptr);
    size_t done = 0;
    while (done < read_sz) {
        ssize_t ret = pread(fd, ptr + done, read_sz - done,
                            static_cast<off_t>(done));
        if (ret < 0) { close(fd); throw std::runtime_error("pread failed"); }
        if (ret == 0) break;
        done += static_cast<size_t>(ret);
    }
    close(fd);
}

// ─────────────────────────────────────────────────────────────────────────────
// Async prefetch via background pthread
//
// This is the no-external-library equivalent of io_uring submit/wait.
// The thread does an O_DIRECT pread into the pinned staging buffer while the
// caller runs GPU compute. submit() is non-blocking; wait() joins the thread.
// ─────────────────────────────────────────────────────────────────────────────

struct PrefetchArgs {
    const char* path;
    void*       buf_ptr;
    size_t      tensor_bytes;
    std::string error;
};

static void* prefetch_thread_fn(void* arg) {
    auto* a = static_cast<PrefetchArgs*>(arg);
    try {
        load_odirect_pread(a->path, a->buf_ptr, a->tensor_bytes);
    } catch (const std::exception& e) {
        a->error = e.what();
    }
    return nullptr;
}

struct PrefetchHandle {
    pthread_t    thread {};
    PrefetchArgs args   {};
    bool         active {false};

    void submit(const std::string& path, void* buf_ptr, size_t tensor_bytes) {
        args = {path.c_str(), buf_ptr, tensor_bytes, ""};
        active = true;
        if (pthread_create(&thread, nullptr, prefetch_thread_fn, &args) != 0)
            throw std::runtime_error("pthread_create failed");
    }

    void wait() {
        if (!active) return;
        pthread_join(thread, nullptr);
        active = false;
        if (!args.error.empty())
            throw std::runtime_error("Prefetch error: " + args.error);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <weights_dir> [iters=20]\n";
        return 1;
    }
    std::string weights_dir = argv[1];
    int ITERS = (argc >= 3) ? std::atoi(argv[2]) : 20;

    // ── Tensor dimensions (Mixtral-8x7B AWQ) ────────────────────────────────
    const int64_t hidden_size       = 4096;
    const int64_t intermediate_size = 14336;
    const int64_t packed_in         = hidden_size / 2;  // 4-bit packing
    const int64_t out_feat          = intermediate_size;

    std::string gate_path = weights_dir + "/layer_0_expert_0_gate.qweight.bin";
    if (access(gate_path.c_str(), F_OK) != 0) {
        std::cerr << "File not found: " << gate_path << "\n";
        return 1;
    }

    const size_t tensor_bytes = static_cast<size_t>(out_feat * packed_in);
    // Over-allocate by 512 so O_DIRECT can round the read size up without
    // writing past the end of the allocation.
    const int64_t alloc_elems = static_cast<int64_t>(
        align_up(tensor_bytes, 512) + 512);

    std::printf("Tensor : %ldx%ld = %.2f MB\n",
                out_feat, packed_in, tensor_bytes / 1e6);
    std::printf("Iters  : %d  (page cache dropped before each)\n\n", ITERS);

    auto device = torch::kCUDA;

    // ── Buffers ──────────────────────────────────────────────────────────────
    // cudaMallocHost returns page-aligned memory (≥4096 B), which satisfies
    // O_DIRECT's 512-byte alignment requirement with no extra work.
    auto pinned_buf = torch::empty(
        {alloc_elems},
        torch::TensorOptions().dtype(torch::kUInt8)
                              .device(torch::kCPU)
                              .pinned_memory(true));
    // Correctly-shaped view into the same allocation.
    auto pinned_view = pinned_buf.narrow(0, 0, out_feat * packed_in)
                                 .view({out_feat, packed_in});

    auto gpu_tensor = torch::empty({out_feat, packed_in},
        torch::TensorOptions().dtype(torch::kUInt8).device(device));

    // Tensors for the fake "expert compute" — matmul with comparable FLOP count.
    auto gpu_a = torch::randn({512, hidden_size}, device);
    auto gpu_b = torch::randn({hidden_size, 512}, device);

    // ── Warmup ───────────────────────────────────────────────────────────────
    {
        auto t = load_mmap_dynamic_pin(gate_path, torch::kUInt8,
                                       {out_feat, packed_in});
        gpu_tensor.copy_(t, true);
        torch::cuda::synchronize();
    }
    std::printf("=== Warmup done ===\n\n");

    // ── Runner ───────────────────────────────────────────────────────────────
    auto run_test = [&](const std::string& label,
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
        print_stats(label, s, tensor_bytes);
        return s;
    };

    // ─────────────────────────────────────────────────────────────────────────
    // Test 1: mmap + dynamic pinned alloc  (original baseline, fd-leak fixed)
    // ─────────────────────────────────────────────────────────────────────────
    run_test("1. mmap + dynamic pin + H2D", [&]() {
        auto t = load_mmap_dynamic_pin(gate_path, torch::kUInt8,
                                       {out_feat, packed_in});
        gpu_tensor.copy_(t, true);
        torch::cuda::synchronize();
    });

    // ─────────────────────────────────────────────────────────────────────────
    // Test 2: mmap into pre-allocated pinned buffer
    // ─────────────────────────────────────────────────────────────────────────
    run_test("2. mmap + prealloc pin + H2D", [&]() {
        load_mmap_prealloc(gate_path, pinned_view);
        gpu_tensor.copy_(pinned_view, true);
        torch::cuda::synchronize();
    });

    // ─────────────────────────────────────────────────────────────────────────
    // Test 3: O_DIRECT + pread — true SSD speed, no page cache involvement.
    // This is the relevant number for memory-constrained inference where the
    // OS will have evicted page cache entries anyway.
    // ─────────────────────────────────────────────────────────────────────────
    Stats odirect_stats;
    odirect_stats = run_test("3. O_DIRECT + pread + H2D", [&]() {
        load_odirect_pread(gate_path, pinned_buf.data_ptr(), tensor_bytes);
        gpu_tensor.copy_(pinned_view, true);
        torch::cuda::synchronize();
    });

    // ─────────────────────────────────────────────────────────────────────────
    // Test 4: Sequential baseline — disk + H2D + GPU compute, fully serialized.
    // This is the naive per-expert cost without any prefetching.
    // ─────────────────────────────────────────────────────────────────────────
    auto seq_stats = run_test("4. Sequential (disk → H2D → compute)", [&]() {
        load_odirect_pread(gate_path, pinned_buf.data_ptr(), tensor_bytes);
        gpu_tensor.copy_(pinned_view, true);
        torch::cuda::synchronize();
        auto out = torch::mm(gpu_a, gpu_b);
        torch::cuda::synchronize();
        (void)out;
    });

    // ─────────────────────────────────────────────────────────────────────────
    // Test 5: OVERLAPPED — background thread prefetches expert[i+1] from disk
    //         while GPU executes expert[i].
    //
    //   timeline:  [submit thread] ──────────────────────── [join]
    //                              [GPU matmul expert[i]]
    //                                                        [H2D] [sync]
    //
    // If disk_time ≤ compute_time the disk read is fully hidden and wall time
    // collapses to ≈ compute_time + H2D.
    // ─────────────────────────────────────────────────────────────────────────
    PrefetchHandle prefetch;
    auto ovlp_stats = run_test("5. Overlapped (thread ∥ GPU compute)", [&]() {
        // 1. Kick off async disk read into pinned staging buffer.
        prefetch.submit(gate_path, pinned_buf.data_ptr(), tensor_bytes);

        // 2. GPU compute runs on current expert while disk read is in flight.
        auto out = torch::mm(gpu_a, gpu_b);

        // 3. Block until disk read is done.
        prefetch.wait();

        // 4. Non-blocking H2D copy + sync.
        gpu_tensor.copy_(pinned_view, /*non_blocking=*/true);
        torch::cuda::synchronize();
        (void)out;
    });

    // ─────────────────────────────────────────────────────────────────────────
    // Summary
    // ─────────────────────────────────────────────────────────────────────────
    double saved = seq_stats.mean_us - ovlp_stats.mean_us;
    double pct   = 100.0 * saved / seq_stats.mean_us;

    std::printf("\n=== Overlap Analysis ===\n");
    std::printf("O_DIRECT load (test 3) : %8.0f us  — time you need to hide\n",
                odirect_stats.mean_us);
    std::printf("Sequential (test 4)    : %8.0f us\n", seq_stats.mean_us);
    std::printf("Overlapped (test 5)    : %8.0f us\n", ovlp_stats.mean_us);
    std::printf("Latency saved          : %8.0f us  (%.1f%% of sequential)\n",
                saved, pct);

    if (pct > 40)
        std::printf("\n→ Strong overlap. Prefetch is highly effective on this system.\n");
    else if (pct > 10)
        std::printf("\n→ Partial overlap. Compute partially hides disk latency.\n"
                    "  Consider larger batch size or more expert FLOPS to widen the window.\n");
    else
        std::printf("\n→ Minimal overlap gain. Either GPU compute is the bottleneck "
                    "or disk is faster than the kernel.\n");

    std::printf("\nNote: for full hiding you need GPU compute time ≥ %.0f us per expert.\n",
                odirect_stats.mean_us);

    return 0;
}