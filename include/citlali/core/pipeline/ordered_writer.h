#pragma once

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <utility>

#include <Eigen/Core>

namespace citlali::pipeline {

struct OrderedWriter {
    std::mutex mutex;
    std::condition_variable cv;
    Eigen::Index next = 0;
    std::exception_ptr failure;

    void wait_turn(Eigen::Index index) {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return failure != nullptr || index == next; });
        if (failure != nullptr) {
            std::rethrow_exception(failure);
        }
    }

    void advance() {
        std::lock_guard<std::mutex> lock(mutex);
        ++next;
        cv.notify_all();
    }

    void cancel(std::exception_ptr error) noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        if (failure == nullptr) {
            failure = std::move(error);
        }
        cv.notify_all();
    }

    template <class Write>
    void write_when_ready(Eigen::Index index, Write &&write) {
        wait_turn(index);
        try {
            std::invoke(std::forward<Write>(write));
        } catch (...) {
            cancel(std::current_exception());
            throw;
        }
        advance();
    }
};

}  // namespace citlali::pipeline
