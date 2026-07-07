#pragma once

#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace citlali::pipeline {

struct StageProfileRecord {
    std::int64_t index = 0;
    std::string stage;
    std::string context;
    double elapsed_s = 0.0;
};

class StageProfileCollector {
public:
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        output_path_.clear();
        records_.clear();
        next_index_ = 0;
    }

    void configure_output_dir(const std::string &reduction_dir) {
        if (reduction_dir.empty()) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (!output_path_.empty()) {
            return;
        }

        output_path_ = reduction_dir;
        if (output_path_.back() != '/') {
            output_path_ += "/";
        }
        output_path_ += "citlali_profile.ecsv";

        write_header_unlocked();
        for (const auto &record : records_) {
            append_row_unlocked(record);
        }
    }

    template <class Logger>
    void record(const std::string &stage, const std::string &context,
                double elapsed_s, const Logger &logger) noexcept {
        try {
            StageProfileRecord record;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                record.index = next_index_++;
                record.stage = stage;
                record.context = context;
                record.elapsed_s = elapsed_s;
                records_.push_back(record);

                if (!output_path_.empty()) {
                    append_row_unlocked(record);
                }
            }

            if (logger) {
                logger->info(
                    "profile stage={} context={} elapsed_s={:.6f}",
                    record.stage, record.context, record.elapsed_s);
            }
        } catch (...) {
            // Profiling must never affect reduction behavior.
        }
    }

    std::string output_path() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return output_path_;
    }

private:
    static std::string ecsv_quote(const std::string &value) {
        std::ostringstream os;
        os << '"';
        for (const char ch : value) {
            if (ch == '"' || ch == '\\') {
                os << '\\';
            }
            if (ch == '\n' || ch == '\r' || ch == '\t') {
                os << ' ';
            }
            else {
                os << ch;
            }
        }
        os << '"';
        return os.str();
    }

    void write_header_unlocked() const {
        std::ofstream out(output_path_, std::ios::trunc);
        out << "# %ECSV 1.0\n";
        out << "# ---\n";
        out << "# datatype:\n";
        out << "# - {name: index, datatype: int64}\n";
        out << "# - {name: stage, datatype: string}\n";
        out << "# - {name: context, datatype: string}\n";
        out << "# - {name: elapsed_s, datatype: float64}\n";
        out << "# schema: astropy-2.0\n";
        out << "index stage context elapsed_s\n";
    }

    void append_row_unlocked(const StageProfileRecord &record) const {
        std::ofstream out(output_path_, std::ios::app);
        out << record.index << ' '
            << ecsv_quote(record.stage) << ' '
            << ecsv_quote(record.context) << ' '
            << std::fixed << std::setprecision(6) << record.elapsed_s << '\n';
    }

    mutable std::mutex mutex_;
    std::string output_path_;
    std::vector<StageProfileRecord> records_;
    std::int64_t next_index_ = 0;
};

inline StageProfileCollector &stage_profile_collector() {
    static StageProfileCollector collector;
    return collector;
}

inline void reset_stage_profile() {
    stage_profile_collector().reset();
}

template <class Logger>
void configure_stage_profile_output(const std::string &reduction_dir,
                                    const Logger &logger) noexcept {
    try {
        stage_profile_collector().configure_output_dir(reduction_dir);
        const auto output_path = stage_profile_collector().output_path();
        if (logger && !output_path.empty()) {
            logger->info("stage profile sidecar: {}", output_path);
        }
    } catch (const std::exception &e) {
        if (logger) {
            logger->warn("failed to configure stage profile sidecar in {}: {}",
                         reduction_dir, e.what());
        }
    } catch (...) {
        if (logger) {
            logger->warn("failed to configure stage profile sidecar in {}",
                         reduction_dir);
        }
    }
}

template <class Logger>
class StageProfileScope {
public:
    StageProfileScope(std::string stage, std::string context,
                      Logger logger)
        : stage_(std::move(stage)),
          context_(std::move(context)),
          logger_(std::move(logger)),
          start_(Clock::now()) {}

    StageProfileScope(const StageProfileScope &) = delete;
    StageProfileScope &operator=(const StageProfileScope &) = delete;

    StageProfileScope(StageProfileScope &&other) noexcept
        : stage_(std::move(other.stage_)),
          context_(std::move(other.context_)),
          logger_(std::move(other.logger_)),
          start_(other.start_),
          active_(other.active_) {
        other.active_ = false;
    }

    StageProfileScope &operator=(StageProfileScope &&) = delete;

    ~StageProfileScope() noexcept {
        if (!active_) {
            return;
        }

        const auto elapsed =
            std::chrono::duration<double>(Clock::now() - start_).count();
        stage_profile_collector().record(stage_, context_, elapsed, logger_);
    }

private:
    using Clock = std::chrono::steady_clock;

    std::string stage_;
    std::string context_;
    Logger logger_;
    Clock::time_point start_;
    bool active_ = true;
};

template <class Logger>
StageProfileScope<Logger> profile_stage(const char *stage,
                                        const Logger &logger,
                                        std::string context = {}) {
    return StageProfileScope<Logger>(
        std::string(stage), std::move(context), logger);
}

}  // namespace citlali::pipeline
