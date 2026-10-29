#pragma once

#include <cerrno>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#if defined(__APPLE__)
#include <sys/stdio.h>
#elif defined(__linux__)
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace citlali::pipeline {

// One direct native attempt owns one .pending directory. Failed attempts are
// deliberately retained for diagnosis. Publication never replaces any entry;
// this is visibility/ownership protection, not an fsync durability guarantee.
class StagedOutputDirectory {
public:
    explicit StagedOutputDirectory(const std::filesystem::path &output)
        : output_(output), staging_(output.string() + ".pending") {
        if (!output_.is_absolute() || output_.filename().empty()) {
            throw std::invalid_argument("output must name an absolute directory");
        }
        std::error_code error;
        const auto existing = std::filesystem::symlink_status(output_, error);
        if (error && error != std::errc::no_such_file_or_directory) {
            throw std::filesystem::filesystem_error("inspect output", output_, error);
        }
        if (existing.type() != std::filesystem::file_type::not_found) {
            throw std::runtime_error("output already exists: " + output_.string());
        }
        std::filesystem::create_directories(output_.parent_path());
        // The return value, not a preceding exists() check, grants ownership.
        if (!std::filesystem::create_directory(staging_)) {
            throw std::runtime_error("pending output already claimed: " + staging_.string());
        }
    }

    StagedOutputDirectory(const StagedOutputDirectory &) = delete;
    StagedOutputDirectory &operator=(const StagedOutputDirectory &) = delete;
    StagedOutputDirectory(StagedOutputDirectory &&) = delete;
    StagedOutputDirectory &operator=(StagedOutputDirectory &&) = delete;

    const std::filesystem::path &staging() const noexcept { return staging_; }

    void publish() {
        if (published_) throw std::logic_error("output already published");
#if defined(__APPLE__)
        const int result = ::renamex_np(staging_.c_str(), output_.c_str(), RENAME_EXCL);
#elif defined(__linux__) && defined(SYS_renameat2)
        // RENAME_NOREPLACE is Linux ABI flag 1; use the syscall so older glibc
        // declarations do not weaken the operation to an overwrite rename.
        const int result = static_cast<int>(::syscall(
            SYS_renameat2, AT_FDCWD, staging_.c_str(), AT_FDCWD, output_.c_str(), 1U));
#else
        throw std::runtime_error("atomic no-replace directory publication is unavailable");
#endif
#if defined(__APPLE__) || (defined(__linux__) && defined(SYS_renameat2))
        if (result != 0) {
            const auto error = std::error_code(errno, std::generic_category());
            throw std::filesystem::filesystem_error(
                "publish output without replacement", staging_, output_, error);
        }
        published_ = true;
#endif
    }

private:
    std::filesystem::path output_, staging_;
    bool published_ = false;
};

}  // namespace citlali::pipeline
