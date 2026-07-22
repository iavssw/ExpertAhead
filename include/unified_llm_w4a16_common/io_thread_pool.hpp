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
#include <algorithm>

namespace unified_llm_w4a16_common {

class BaseThreadPool {
public:
    BaseThreadPool(size_t threads, size_t reserved_high_priority = 0) : stop(false), active_low_priority(0) {
        max_low_priority = threads > reserved_high_priority ? threads - reserved_high_priority : threads;
        if (max_low_priority == 0) max_low_priority = 1;

        for(size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                for(;;) {
                    std::function<void()> task;
                    bool is_high = false;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] {
                            return this->stop || !this->high_priority_tasks.empty() || 
                                   (!this->low_priority_tasks.empty() && this->active_low_priority < this->max_low_priority);
                        });
                        if(this->stop && this->high_priority_tasks.empty() && this->low_priority_tasks.empty())
                            return;
                        
                        if (!this->high_priority_tasks.empty()) {
                            task = std::move(this->high_priority_tasks.front());
                            this->high_priority_tasks.pop_front();
                            is_high = true;
                        } else if (!this->low_priority_tasks.empty() && this->active_low_priority < this->max_low_priority) {
                            task = std::move(this->low_priority_tasks.front());
                            this->low_priority_tasks.pop_front();
                            this->active_low_priority++;
                            is_high = false;
                        } else {
                            continue;
                        }
                    }
                    task();
                    if (!is_high) {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->active_low_priority--;
                        this->condition.notify_one();
                    }
                }
            });
        }
    }

    template<class F, class... Args>
    auto enqueue(bool high_priority, F&& f, Args&&... args) -> std::future<typename std::invoke_result_t<F, Args...>> {
        using return_type = typename std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if(stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            if (high_priority) {
                high_priority_tasks.emplace_back([task](){ (*task)(); });
            } else {
                low_priority_tasks.emplace_back([task](){ (*task)(); });
            }
        }
        condition.notify_one();
        return res;
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result_t<F, Args...>> {
        return enqueue(false, std::forward<F>(f), std::forward<Args>(args)...);
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
    std::deque<std::function<void()>> high_priority_tasks;
    std::deque<std::function<void()>> low_priority_tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
    size_t active_low_priority;
    size_t max_low_priority;
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
    // Reserve roughly 1/4 of threads (or at least 2 if possible) for high priority tasks
    IOThreadPool(size_t threads) : BaseThreadPool(threads, std::max((size_t)2, threads / 4)) {}
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
