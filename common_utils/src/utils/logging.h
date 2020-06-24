#pragma once
#ifndef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#endif
#ifndef SPDLOG_FMT_EXTERNAL
#define SPDLOG_FMT_EXTERNAL
#endif
#include "formatter/container.h"
#include "meta.h"
#include <fmt/ostream.h>
// #include <iostream>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <sstream>

/// Macros to install scope-local logger and log
#define LOGGER_INIT2(level_, name)                                             \
    inline static const auto logger = []() {                                   \
        auto color_sink =                                                      \
            std::make_shared<spdlog::sinks::stdout_color_sink_mt>();           \
        auto logger =                                                          \
            std::make_shared<spdlog::logger>(name, std::move(color_sink));     \
        logger->set_level(level_);                                             \
        return logger;                                                         \
    }()
#define LOGGER_INIT1(level_) LOGGER_INIT2(level_, "")
#define LOGGER_INIT0()                                                         \
    inline static const auto logger = spdlog::default_logger();
#define LOGGER_INIT(...) GET_MACRO(LOGGER_INIT, __VA_ARGS__)
#define LOGGER_TRACE(...) SPDLOG_LOGGER_TRACE(logger, __VA_ARGS__)
#define LOGGER_DEBUG(...) SPDLOG_LOGGER_DEBUG(logger, __VA_ARGS__)
#define LOGGER_INFO(...) SPDLOG_LOGGER_INFO(logger, __VA_ARGS__)
#define LOGGER_WARN(...) SPDLOG_LOGGER_WARN(logger, __VA_ARGS__)
#define LOGGER_ERROR(...) SPDLOG_LOGGER_ERROR(logger, __VA_ARGS__)
#define LOGGER_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(logger, __VA_ARGS__)

namespace logging {

namespace internal {

template <typename Prefunc, typename Postfunc, typename... Pargs>
struct decorated_invoke {
    decorated_invoke(std::tuple<Pargs...> &&, Prefunc &&pre_, Postfunc &&post_)
        : pre(FWD(pre_)), post(FWD(post_)) {}
    Prefunc &&pre;
    Postfunc &&post;
    template <typename F, typename... Args>
    auto operator()(Pargs &&... pargs, F &&func, Args &&... args) const
        -> decltype(auto) {
        decltype(auto) d = pre(pargs...);
        if constexpr (std::is_void_v<std::invoke_result_t<F, Args...>>) {
            FWD(func)(FWD(args)...);
            post(FWD(d));
        } else {
            decltype(auto) ret = FWD(func)(FWD(args)...);
            post(FWD(d));
            return ret;
        }
    }
};

}  // namespace

inline auto init() {
    // std::cout << fmt::format("log level at compile time: {}\n",
    //                         SPDLOG_ACTIVE_LEVEL);
    spdlog::set_level(
        static_cast<spdlog::level::level_enum>(SPDLOG_ACTIVE_LEVEL));
}

inline const auto quiet = internal::decorated_invoke(
    std::tuple<>{},
    []() {
        auto level = spdlog::default_logger()->level();
        spdlog::set_level(spdlog::level::off);
        return level;
    },
    [](auto &&level) { spdlog::set_level(FWD(level)); });

inline const auto timeit = internal::decorated_invoke(
    std::tuple<std::string_view>{},
    [](auto msg) {
        SPDLOG_INFO("**timeit** {}", msg);
        // get time before function invocation
        auto start = std::chrono::high_resolution_clock::now();
        return std::make_tuple(msg, start);
    },
    [](auto &&p) {
        const auto &[msg, start] = FWD(p);
        // get time after function invocation
        const auto &stop = std::chrono::high_resolution_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::duration<double>>(stop -
                                                                      start);
        SPDLOG_INFO("**timeit** {} finished in {}ms", msg,
                    elapsed.count() * 1e3);
    });

inline auto now() { return std::chrono::high_resolution_clock::now(); }
inline auto elapsed_since(
    const std::chrono::time_point<std::chrono::high_resolution_clock> &since) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(
               std::chrono::high_resolution_clock::now() - since)
               .count() *
           1e3;
}

struct scoped_timeit {
    scoped_timeit(std::string_view msg_, double *t = nullptr)
        : msg(msg_), elapsed(t) {
        SPDLOG_INFO("**timeit** {}", msg);
    }
    ~scoped_timeit() {
        auto t = elapsed_since(t0);
        if (elapsed != nullptr) {
            *elapsed = t;
        }
        SPDLOG_INFO("**timeit** {} finished in {}ms", msg, t);
    }
    std::chrono::time_point<std::chrono::high_resolution_clock> t0{now()};
    std::string_view msg{""};
    double *elapsed{nullptr};
};

template <auto level_> struct scoped_loglevel {
    scoped_loglevel() {
        level = spdlog::default_logger()->level();
        spdlog::set_level(level_);
    }
    ~scoped_loglevel() { spdlog::set_level(level); }
    spdlog::level::level_enum level;
};
using scoped_quiet = scoped_loglevel<spdlog::level::off>;

struct Timer {

    explicit Timer(std::size_t maxlap) : m_timer(maxlap) { reset(); }
    void reset() {
        std::fill(m_timer.begin(), m_timer.end(), 0);
        SPDLOG_DEBUG("timer({}) reset to {}", m_timer.size(), m_timer);
    }
    void start(std::size_t i) {
        const auto &now = std::chrono::high_resolution_clock::now();
        m_timer[i] = std::chrono::duration_cast<std::chrono::duration<double>>(
                         now - m_zero)
                         .count() *
                     1e3; // ms
        SPDLOG_DEBUG("timer({}) start at {}", i, m_timer[i]);
    }
    void stop(std::size_t i) {
        const auto &now = std::chrono::high_resolution_clock::now();
        m_timer[i] = std::chrono::duration_cast<std::chrono::duration<double>>(
                         now - m_zero)
                             .count() *
                         1e3 -
                     m_timer[i]; // ms
        SPDLOG_DEBUG("timer({}) stop at {}", i, m_timer[i]);
    }
    void switch_(std::size_t i, std::size_t j) {
        stop(i);
        start(j);
        SPDLOG_DEBUG("timer switch from {} to {}", i, j);
    }
    const double &operator[](std::size_t i) const { return m_timer[i]; }

private:
    std::vector<double> m_timer{};
    std::chrono::time_point<std::chrono::high_resolution_clock> m_zero{
        std::chrono::high_resolution_clock::now()};
};

template <typename Func> class progressbar {
    static const auto overhead = sizeof " [100%]";

    Func func;
    const std::size_t width;
    const double scale{100};
    std::string message;
    const std::string bar;
    std::atomic<int> counter{0};

    auto barstr(double perc) {
        // clamp prog to valid range [0,1]
        if (perc < 0) {
            perc = 0;
        } else if (perc > 1) {
            perc = 1;
        }
        std::stringstream ss;
        auto barwidth = width - message.size();
        auto offset = width - static_cast<unsigned>(barwidth * perc);
        ss << message;
        ss.write(bar.data() + offset, barwidth);
        ss << fmt::format("[{:3.0f}%]", scale * perc);

        return ss.str();
    }

public:
    progressbar(Func func_, std::size_t linewidth, std::string message_,
                const char symbol = '.')
        : func{std::move(func_)}, width{linewidth - overhead},
          message{std::move(message_)}, bar{std::string(width, symbol) +
                                            std::string(width, ' ')} {
        // write(0.0);
    }

    // not copyable or movable
    progressbar(const progressbar &) = delete;
    progressbar &operator=(const progressbar &) = delete;
    progressbar(progressbar &&) = delete;
    progressbar &operator=(progressbar &&) = delete;

    ~progressbar() { func(fmt::format("{}\n", barstr(1.0))); }

    auto write(double perc) { func(barstr(perc)); }
    template <typename N1, typename N2> auto count(N1 total, N2 stride) {
        if (stride < 1) {
            stride = 1;
        }
        ++counter;
        if (counter % stride == 0) {
            write(double(counter) / total);
        }
    }
};

} // namespace logging
