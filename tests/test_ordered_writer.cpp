#include <citlali/core/pipeline/timestream_output_context.h>

#include <gtest/gtest.h>

#include <atomic>
#include <future>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {

TEST(ordered_writer, serializes_out_of_order_workers) {
    citlali::pipeline::OrderedWriter writer;
    std::mutex values_mutex;
    std::vector<int> values;

    auto second = std::async(std::launch::async, [&] {
        writer.write_when_ready(1, [&] {
            std::lock_guard<std::mutex> lock(values_mutex);
            values.push_back(1);
        });
    });
    auto first = std::async(std::launch::async, [&] {
        writer.write_when_ready(0, [&] {
            std::lock_guard<std::mutex> lock(values_mutex);
            values.push_back(0);
        });
    });

    first.get();
    second.get();
    EXPECT_EQ(values, (std::vector<int>{0, 1}));
}

TEST(ordered_writer, failure_cancels_and_wakes_waiters) {
    citlali::pipeline::OrderedWriter writer;
    std::atomic<bool> later_write_ran{false};
    std::promise<void> waiter_started;
    auto waiter_ready = waiter_started.get_future();

    auto later = std::async(std::launch::async, [&] {
        waiter_started.set_value();
        writer.write_when_ready(1, [&] { later_write_ran = true; });
    });
    waiter_ready.get();

    EXPECT_THROW(
        writer.write_when_ready(
            0, [] { throw std::runtime_error("injected write failure"); }),
        std::runtime_error);
    EXPECT_THROW(later.get(), std::runtime_error);
    EXPECT_FALSE(later_write_ran.load());

    citlali::pipeline::OrderedWriter next_run_writer;
    bool next_run_write_ran = false;
    EXPECT_NO_THROW(next_run_writer.write_when_ready(
        0, [&] { next_run_write_ran = true; }));
    EXPECT_TRUE(next_run_write_ran);
}

TEST(ordered_writer, required_output_failure_cancels_other_streams) {
    citlali::pipeline::TimestreamOutputFlags flags;
    flags.write_rtc = true;
    flags.write_ptc = true;
    const auto writers =
        citlali::pipeline::make_timestream_output_writers(flags);
    std::atomic<bool> ptc_write_ran{false};
    std::promise<void> waiter_started;
    auto waiter_ready = waiter_started.get_future();

    auto waiting_ptc = std::async(std::launch::async, [&] {
        waiter_started.set_value();
        writers.write_when_ready(
            writers.ptc, 1, [&] { ptc_write_ran = true; });
    });
    waiter_ready.get();

    EXPECT_NO_THROW(
        writers.write_when_ready(
            writers.rtc, 0,
            [] { throw std::runtime_error("injected RTC write failure"); }));
    EXPECT_NO_THROW(waiting_ptc.get());
    EXPECT_FALSE(ptc_write_ran.load());
    EXPECT_THROW(writers.rethrow_if_failed(), std::runtime_error);
}

}  // namespace
