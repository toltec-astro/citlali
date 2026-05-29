#pragma once

#include <algorithm>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <spdlog/common.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/base_sink.h>
#include <zlib.h>

namespace citlali::logging {

template <typename Mutex>
class gzip_file_sink final : public spdlog::sinks::base_sink<Mutex> {
public:
    explicit gzip_file_sink(std::string filepath, int level = Z_BEST_SPEED)
        : filepath_(std::move(filepath)) {
        file_ = gzopen(filepath_.c_str(), "wb");
        if (file_ == nullptr) {
            throw spdlog::spdlog_ex("failed to open gzip log file: " + filepath_);
        }
        gzsetparams(file_, level, Z_DEFAULT_STRATEGY);
        gzbuffer(file_, 1 << 20);
    }

    ~gzip_file_sink() override {
        if (file_ != nullptr) {
            gzclose(file_);
            file_ = nullptr;
        }
    }

    const std::string &filepath() const {
        return filepath_;
    }

protected:
    void sink_it_(const spdlog::details::log_msg &msg) override {
        spdlog::memory_buf_t formatted;
        this->formatter_->format(msg, formatted);

        const char *data = formatted.data();
        std::size_t remaining = formatted.size();
        while (remaining > 0) {
            const auto chunk = static_cast<unsigned int>(
                std::min<std::size_t>(remaining, std::numeric_limits<unsigned int>::max()));
            const int nw = gzwrite(file_, data, chunk);
            if (nw == 0) {
                int errnum = Z_OK;
                const char *err = gzerror(file_, &errnum);
                throw spdlog::spdlog_ex(
                    "failed to write gzip log file " + filepath_ + ": " +
                    std::string(err != nullptr ? err : "unknown zlib error"));
            }
            data += nw;
            remaining -= static_cast<std::size_t>(nw);
        }
    }

    void flush_() override {
        if (file_ != nullptr) {
            gzflush(file_, Z_SYNC_FLUSH);
        }
    }

private:
    std::string filepath_;
    gzFile file_{nullptr};
};

using gzip_file_sink_mt = gzip_file_sink<std::mutex>;

inline void attach_reduction_gzip_sink(
    const std::shared_ptr<spdlog::logger> &logger,
    const std::string &filepath) {
    if (!logger) {
        return;
    }
    static std::mutex registry_mutex;
    static std::unordered_map<std::string, std::weak_ptr<gzip_file_sink_mt>> registry;

    std::shared_ptr<gzip_file_sink_mt> gzip_sink_shared;
    {
        std::lock_guard<std::mutex> lock(registry_mutex);
        auto &entry = registry[filepath];
        gzip_sink_shared = entry.lock();
        if (gzip_sink_shared == nullptr) {
            gzip_sink_shared = std::make_shared<gzip_file_sink_mt>(filepath);
            gzip_sink_shared->set_level(spdlog::level::trace);
            entry = gzip_sink_shared;
        }
    }

    auto &sinks = logger->sinks();
    for (const auto &existing_sink : sinks) {
        if (existing_sink == gzip_sink_shared) {
            return;
        }
    }
    sinks.erase(
        std::remove_if(
            sinks.begin(), sinks.end(),
            [&gzip_sink_shared](const spdlog::sink_ptr &other) {
                auto gzip_sink = std::dynamic_pointer_cast<gzip_file_sink_mt>(other);
                return gzip_sink != nullptr && other != gzip_sink_shared;
            }),
        sinks.end());
    sinks.push_back(gzip_sink_shared);
}

inline std::string enable_reduction_gzip_logs(const std::string &redu_dir) {
    auto path = (std::filesystem::path(redu_dir) / "citlali.log.gz").string();
    attach_reduction_gzip_sink(spdlog::get("citlali_logger"), path);
    return path;
}

} // namespace citlali::logging
