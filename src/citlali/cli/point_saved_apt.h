#pragma once

#include <citlali/core/pipeline/canonical_apt_detector_relation_v2.h>
#include <chrono>
#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

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
        const bool reused = current_ && current_->files.verified().manifest_path == manifest;
        if (reused) {
            require_manifest(current_->files.verified(), expected_sha256);
            current_->files.require_unchanged();
            ++network_reuses_;
        } else {
            if (current_) current_->files.require_unchanged();
            current_.reset();
            auto next = std::make_unique<SavedMapApt>(manifest);
            const auto &verified = next->files.verified();
            require_manifest(verified, expected_sha256);
            const auto [it, inserted] = used_.try_emplace(manifest, identity(verified));
            if (!inserted) require_identity(it->second, verified);
            current_ = std::move(next);
            ++full_admissions_;
        }
        selection_seconds_ += std::chrono::duration<double>(Clock::now() - start).count();
        return *current_;
    }

    void verify_before_publication() {
        if (!current_) throw std::logic_error("weighted POINT has no live APT selection");
        const auto finish_manifest = current_->files.verified().manifest_path;
        current_->files.require_unchanged();
        // Network loading is over. Release the large parsed selection before
        // rereading any evicted locator; its manifest authenticates every member.
        current_.reset();
        for (const auto &[manifest, admitted] : used_) {
            if (manifest == finish_manifest) continue;
            const auto verified = pipeline::canonical_apt_v2::verify_bundle_filesystem(manifest, true);
            require_identity(admitted, verified);
            ++evicted_rechecks_;
        }
    }

    unsigned full_admissions() const noexcept { return full_admissions_; }
    unsigned network_reuses() const noexcept { return network_reuses_; }
    unsigned evicted_rechecks() const noexcept { return evicted_rechecks_; }
    double selection_seconds() const noexcept { return selection_seconds_; }

private:
    static void require_manifest(const pipeline::canonical_apt_v2::VerifiedBundle &verified,
                                 const std::string &expected_sha256) {
        if (verified.manifest_digests.transport_sha256 != "sha256:" + expected_sha256) {
            throw pipeline::canonical_apt_v2::ContractError(
                "weighted POINT APT does not match the network manifest binding");
        }
    }
    struct Identity {
        std::string manifest_digest;
        std::string receipt_bytes;
        bool operator==(const Identity &) const = default;
    };
    static Identity identity(const pipeline::canonical_apt_v2::VerifiedBundle &verified) {
        return {verified.manifest_digests.transport_sha256, verified.payload.root_receipt_bytes};
    }
    static void require_identity(const Identity &admitted,
                                 const pipeline::canonical_apt_v2::VerifiedBundle &verified) {
        if (admitted != identity(verified)) {
            throw pipeline::canonical_apt_v2::ContractError(
                "weighted POINT APT generation changed since network admission");
        }
    }
    using Clock = std::chrono::steady_clock;
    std::unique_ptr<const SavedMapApt> current_;
    std::map<std::filesystem::path, Identity> used_;
    unsigned full_admissions_ = 0;
    unsigned network_reuses_ = 0;
    unsigned evicted_rechecks_ = 0;
    double selection_seconds_ = 0;
};

}  // namespace citlali::cli::detail
