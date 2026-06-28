#pragma once
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <cstdlib>

namespace unified_llm_w4a16_common {

class BaseThreadPool {
public:
    BaseThreadPool(size_t threads) : stop(false) {
        for(size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                for(;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result_t<F, Args...>> {
        using return_type = typename std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if(stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task](){ (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ~BaseThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker: workers) {
            worker.join();
        }
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

class IOThreadPool : public BaseThreadPool {
public:
    static IOThreadPool& get_instance() {
        static int num_threads = []() {
            const char* env_threads = std::getenv("HETEROPREDICT_IO_THREADS");
            if (env_threads) {
                int t = std::atoi(env_threads);
                if (t > 0) return t;
            }
            return 16; // default 16
        }();
        static IOThreadPool instance(num_threads);
        return instance;
    }
private:
    IOThreadPool(size_t threads) : BaseThreadPool(threads) {}
};

class SpeculativeLoadThreadPool : public BaseThreadPool {
public:
    static SpeculativeLoadThreadPool& get_instance() {
        static int num_threads = []() {
            const char* env_threads = std::getenv("HETEROPREDICT_SPEC_THREADS");
            if (env_threads) {
                int t = std::atoi(env_threads);
                if (t > 0) return t;
            }
            return 8; // default 8
        }();
        static SpeculativeLoadThreadPool instance(num_threads);
        return instance;
    }
private:
    SpeculativeLoadThreadPool(size_t threads) : BaseThreadPool(threads) {}
};

} // namespace unified_llm_w4a16_common
