#pragma once

#include <citlali/core/mapmaking/jinc_contract.h>
#include <citlali/core/pipeline/timestream_native_science_projection.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace citlali::pipeline {

// One scan/chunk owner carried beside the compatibility TCData. Numerical
// matrices remain owned by their established processors; this object owns
// only the exact native mapping, revision ledger, and immutable adapter
// results needed for transactional publication.
class NativeScanRuntimeState {
public:
    explicit NativeScanRuntimeState(
        std::shared_ptr<const NativeMeasuredDetectorScan> mapping)
        : mapping_{std::move(mapping)}, ledger_{mapping_} {
        if (!mapping_) {
            throw std::invalid_argument(
                "native scan runtime state requires its measured mapping");
        }
    }

    const std::shared_ptr<const NativeMeasuredDetectorScan> &mapping_handle()
        const noexcept {
        return mapping_;
    }
    NativeMeasuredDetectorLedger &ledger() noexcept { return ledger_; }
    const NativeMeasuredDetectorLedger &ledger() const noexcept {
        return ledger_;
    }

    std::optional<NativeRtcDispatchResult> rtc;
    std::optional<NativePtcPreparedOperation> ptc_prepared;
    std::optional<NativeScienceProjection> science_projection;
    std::optional<Eigen::MatrixXd> kernel;
    std::optional<Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>>
        ptc_preclean_flags;
    std::optional<Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>>
        ptc_flags;
    std::optional<mapmaking::JincProcessingScanTrace>
        jinc_processing_trace;
    std::vector<TimestreamDetectorColumn>
        learned_map_zero_weight_detector_columns;
    Eigen::VectorXd fcf;

private:
    std::shared_ptr<const NativeMeasuredDetectorScan> mapping_;
    NativeMeasuredDetectorLedger ledger_;
};

}  // namespace citlali::pipeline
