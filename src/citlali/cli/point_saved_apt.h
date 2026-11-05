#pragma once

#include <citlali/core/pipeline/canonical_apt_detector_relation_v2.h>
#include <chrono>
#include <filesystem>
#include <optional>
#include <string>

namespace citlali::cli::detail {

struct SavedMapApt {
    explicit SavedMapApt(const std::filesystem::path &manifest)
        : files(manifest),
          relation(pipeline::admit_canonical_apt_detector_relation_v2(files.verified())) {}

    pipeline::canonical_apt_v2::FreshFilesystemBundle files;
    pipeline::CanonicalAptDetectorRelationV2 relation;
};

// Owned by one weighted health or map invocation. A network borrows the
// selection only while loading. A different manifest requires full admission.
class PointStageApt {
public:
    const SavedMapApt &select(const std::filesystem::path &manifest,
                             const std::string &expected_sha256) {
        const auto start = Clock::now();
        if (current_ && current_->files.verified().manifest_path == manifest) {
            current_->files.require_unchanged();
            ++network_reuses_;
        } else {
            verify_current();
            current_.emplace(manifest);
            ++full_admissions_;
        }
        if (current_->files.verified().manifest_digests.transport_sha256 !=
            "sha256:" + expected_sha256) {
            throw pipeline::canonical_apt_v2::ContractError(
                "weighted POINT APT does not match the network manifest binding");
        }
        selection_seconds_ += std::chrono::duration<double>(Clock::now() - start).count();
        return *current_;
    }

    void verify_current() const {
        if (current_) current_->files.require_unchanged();
    }

    unsigned full_admissions() const noexcept { return full_admissions_; }
    unsigned network_reuses() const noexcept { return network_reuses_; }
    double selection_seconds() const noexcept { return selection_seconds_; }

private:
    using Clock = std::chrono::steady_clock;
    std::optional<SavedMapApt> current_;
    unsigned full_admissions_ = 0;
    unsigned network_reuses_ = 0;
    double selection_seconds_ = 0;
};

}  // namespace citlali::cli::detail
