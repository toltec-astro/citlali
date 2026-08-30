#pragma once

#include <citlali/core/pipeline/ast_scan_motion.h>
#include <citlali/core/pipeline/timestream_native_timing.h>

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace citlali::pipeline {

struct AstScanMotionMappedSupport {
    // network_occurrence retains the network-native reconstructed Unix time.
    NativeSampleIdentity network_occurrence;
    AstTelescopeRecordIdentity lower_source_record;
    AstTelescopeRecordIdentity upper_source_record;
    double lower_source_time_unix_sec = 0.0;
    double upper_source_time_unix_sec = 0.0;
    double lower_weight = 0.0;
    double upper_weight = 0.0;
};

class AstScanMotionMappedRecord {
public:
    bool available() const noexcept;
    AstScanMotionCause causes() const noexcept;
    double scalar_speed_arcsec_per_sec() const noexcept;

private:
    friend class AstScanMotionNetworkView;

    bool available_ = false;
    AstScanMotionCause causes_ =
        AstScanMotionCause::network_mapping_support_unavailable;
    std::size_t lower_source_local_index_ = 0;
    std::size_t upper_source_local_index_ = 0;
    double lower_weight_ = 0.0;
    double upper_weight_ = 0.0;
    double scalar_speed_arcsec_per_sec_ = 0.0;
};

struct AstScanMotionMappedMemoryEvidence {
    std::size_t derived_mapping_record_bytes = 0;
    std::size_t referenced_raw_product_count = 0;
    std::size_t referenced_network_time_axis_count = 0;

    std::size_t logical_owned_bytes() const noexcept {
        return derived_mapping_record_bytes;
    }
};

// ALIGN-owned, network-specific mapped view. It references the exact network
// timing carrier and the immutable raw AST product, stores no copied network
// time axis, and has no cross-network analysis-grid dependency.
class AstScanMotionNetworkView {
public:
    static std::shared_ptr<const AstScanMotionNetworkView> admit(
        std::shared_ptr<const AstScanMotionProduct> raw_product,
        std::shared_ptr<const NativeNetworkAlignment> network_timing);

    TimestreamNetworkId network_id() const noexcept;
    TimestreamNativeRow first_native_row() const noexcept;
    TimestreamNativeRow past_last_native_row() const noexcept;
    std::size_t occurrence_count() const noexcept;
    const std::shared_ptr<const AstScanMotionProduct> &raw_product_handle()
        const noexcept;
    const std::shared_ptr<const NativeNetworkAlignment> &
    network_timing_handle() const noexcept;
    NativeSampleIdentity identity(TimestreamNativeRow native_row) const;
    const AstScanMotionMappedRecord &record(
        TimestreamNativeRow native_row) const;
    std::optional<double> scalar_speed_arcsec_per_sec(
        TimestreamNativeRow native_row) const;
    std::optional<AstScanMotionMappedSupport> support(
        TimestreamNativeRow native_row) const;
    AstScanMotionMappedMemoryEvidence memory_evidence() const noexcept;

private:
    AstScanMotionNetworkView(
        std::shared_ptr<const AstScanMotionProduct> raw_product,
        std::shared_ptr<const NativeNetworkAlignment> network_timing,
        std::vector<AstScanMotionMappedRecord> records);

    std::size_t local_index(TimestreamNativeRow native_row) const;

    std::shared_ptr<const AstScanMotionProduct> raw_product_;
    std::shared_ptr<const NativeNetworkAlignment> network_timing_;
    std::vector<AstScanMotionMappedRecord> records_;
};

class AstScanMotionNetworkViews {
public:
    static std::shared_ptr<const AstScanMotionNetworkViews> admit(
        NativeObservationScope expected_scope,
        std::shared_ptr<const AstScanMotionProduct> raw_product,
        std::vector<std::shared_ptr<const NativeNetworkAlignment>>
            network_timings);

    const NativeObservationScope &scope() const noexcept;
    const std::shared_ptr<const AstScanMotionProduct> &raw_product_handle()
        const noexcept;
    std::span<const TimestreamNetworkId> participant_network_ids()
        const noexcept;
    const AstScanMotionNetworkView &network(
        TimestreamNetworkId network_id) const;

private:
    AstScanMotionNetworkViews(
        NativeObservationScope scope,
        std::shared_ptr<const AstScanMotionProduct> raw_product,
        std::vector<std::shared_ptr<const AstScanMotionNetworkView>> networks,
        std::vector<TimestreamNetworkId> participant_network_ids,
        std::map<TimestreamNetworkId, std::size_t> network_index_by_id);

    NativeObservationScope scope_;
    std::shared_ptr<const AstScanMotionProduct> raw_product_;
    std::vector<std::shared_ptr<const AstScanMotionNetworkView>> networks_;
    std::vector<TimestreamNetworkId> participant_network_ids_;
    std::map<TimestreamNetworkId, std::size_t> network_index_by_id_;
};

}  // namespace citlali::pipeline
