#pragma once

#include <citlali/core/mapmaking/jinc_contract.h>
#include <citlali/core/pipeline/native_fruit_loop_feedback.h>
#include <citlali/core/pipeline/native_noise_assignment.h>
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
        std::shared_ptr<const NativeMeasuredDetectorScan> mapping,
        std::optional<std::size_t> selected_first_common_slot = {},
        std::optional<std::size_t> selected_past_last_common_slot = {})
        : mapping_{std::move(mapping)}, ledger_{mapping_} {
        if (!mapping_) {
            throw std::invalid_argument(
                "native scan runtime state requires its measured mapping");
        }
        selected_first_common_slot_ = selected_first_common_slot.value_or(
            mapping_->first_common_slot());
        selected_past_last_common_slot_ =
            selected_past_last_common_slot.value_or(
                mapping_->past_last_common_slot());
        if (selected_first_common_slot_ < mapping_->first_common_slot() ||
            selected_past_last_common_slot_ >
                mapping_->past_last_common_slot() ||
            selected_first_common_slot_ >= selected_past_last_common_slot_) {
            throw std::invalid_argument(
                "native scan selected interval is empty or outside loaded support");
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
    std::size_t selected_first_common_slot() const noexcept {
        return selected_first_common_slot_;
    }
    std::size_t selected_past_last_common_slot() const noexcept {
        return selected_past_last_common_slot_;
    }
    std::size_t loaded_row_count() const noexcept {
        return mapping_->past_last_common_slot() -
            mapping_->first_common_slot();
    }
    std::size_t selected_row_count() const noexcept {
        return selected_past_last_common_slot_ -
            selected_first_common_slot_;
    }

    std::optional<NativeRtcDispatchResult> rtc;
    std::optional<NativePtcPreparedOperation> ptc_prepared;
    std::optional<NativeScienceProjection> science_projection;
    std::optional<NativeScienceProjection> map_projection;
    std::optional<Eigen::MatrixXd> kernel;
    std::optional<Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>>
        ptc_preclean_flags;
    std::optional<Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>>
        ptc_flags;
    std::optional<mapmaking::JincProcessingScanTrace>
        jinc_processing_trace;
    std::optional<NativeNoiseAssignmentSummaryV3> noise_assignment;
    std::optional<NativeFruitLoopFeedbackSummaryV3> fruit_loop_feedback;
    std::vector<TimestreamDetectorColumn>
        learned_map_zero_weight_detector_columns;
    Eigen::VectorXd fcf;

private:
    std::shared_ptr<const NativeMeasuredDetectorScan> mapping_;
    NativeMeasuredDetectorLedger ledger_;
    std::size_t selected_first_common_slot_ = 0;
    std::size_t selected_past_last_common_slot_ = 0;
};

}  // namespace citlali::pipeline
