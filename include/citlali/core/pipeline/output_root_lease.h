#pragma once

#include <filesystem>

namespace citlali::pipeline {

inline constexpr const char *output_root_lock_filename =
    ".citlali-reduction.lock";

class OutputRootLease {
public:
    explicit OutputRootLease(const std::filesystem::path &output_root);
    ~OutputRootLease() noexcept;

    OutputRootLease(const OutputRootLease &) = delete;
    OutputRootLease &operator=(const OutputRootLease &) = delete;
    OutputRootLease(OutputRootLease &&) = delete;
    OutputRootLease &operator=(OutputRootLease &&) = delete;

    const std::filesystem::path &output_root() const noexcept {
        return output_root_;
    }

    const std::filesystem::path &lock_path() const noexcept {
        return lock_path_;
    }

private:
    std::filesystem::path output_root_;
    std::filesystem::path lock_path_;
    int descriptor_ = -1;
};

}  // namespace citlali::pipeline
