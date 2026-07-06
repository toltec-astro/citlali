#pragma once

#include <condition_variable>
#include <mutex>

#include <Eigen/Core>

namespace citlali::pipeline {

struct OrderedWriter {
    std::mutex mutex;
    std::condition_variable cv;
    Eigen::Index next = 0;

    void wait_turn(Eigen::Index index) {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return index == next; });
    }

    void advance() {
        std::lock_guard<std::mutex> lock(mutex);
        ++next;
        cv.notify_all();
    }
};

}  // namespace citlali::pipeline
