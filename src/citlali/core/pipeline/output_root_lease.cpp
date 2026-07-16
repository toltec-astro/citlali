#include <citlali/core/pipeline/output_root_lease.h>

#include <citlali/core/error/error.h>

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fmt/core.h>
#include <system_error>
#include <sys/file.h>
#include <unistd.h>

namespace citlali::pipeline {
namespace {

std::string system_error_message(int error_number) {
    return std::error_code(error_number, std::generic_category()).message();
}

}  // namespace

OutputRootLease::OutputRootLease(
    const std::filesystem::path &output_root)
    : output_root_(output_root),
      lock_path_(output_root_ / output_root_lock_filename) {
    if (output_root_.empty()) {
        throw citlali::error::invalid_config(
            "runtime.output_dir must not be empty");
    }

    std::error_code directory_error;
    std::filesystem::create_directories(output_root_, directory_error);
    if (directory_error) {
        throw citlali::error::output(fmt::format(
            "failed to prepare output root '{}': {}",
            output_root_.string(), directory_error.message()));
    }

    descriptor_ = ::open(lock_path_.c_str(), O_CREAT | O_RDWR, 0666);
    if (descriptor_ < 0) {
        const int open_error = errno;
        throw citlali::error::output(fmt::format(
            "failed to open output-root lock '{}': {}",
            lock_path_.string(), system_error_message(open_error)));
    }

    if (::flock(descriptor_, LOCK_EX | LOCK_NB) != 0) {
        const int lock_error = errno;
        ::close(descriptor_);
        descriptor_ = -1;

        if (lock_error == EWOULDBLOCK || lock_error == EAGAIN) {
            throw citlali::error::output(fmt::format(
                "output root '{}' is already in use by another Citlali "
                "reduction; wait for it to finish or use a different "
                "runtime.output_dir",
                output_root_.string()));
        }
        throw citlali::error::output(fmt::format(
            "failed to lock output root '{}': {}", output_root_.string(),
            system_error_message(lock_error)));
    }
}

OutputRootLease::~OutputRootLease() noexcept {
    if (descriptor_ >= 0) {
        ::flock(descriptor_, LOCK_UN);
        ::close(descriptor_);
    }
}

}  // namespace citlali::pipeline
