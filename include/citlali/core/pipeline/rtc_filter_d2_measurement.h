#pragma once

#include <citlali/core/pipeline/paired_readout.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace citlali::pipeline {

inline constexpr const char *rtc_filter_d2_line_strategy_id =
    "citlali-rtc-line-audit-v1";

enum class RtcFilterD2MeasurementStage : std::uint8_t {
    native_prefilter,
    native_post_cleaning_residual,
};

enum class RtcFilterD2SourceMaskDisposition : std::uint8_t {
    applied,
    approved_not_applicable,
};

enum class RtcFilterD2LineMaskDisposition : std::uint8_t {
    applied,
    complete_no_lines,
    pending,
};

struct RtcFilterD2LineProtectionInterval {
    std::string interval_id;
    double low_hz = 0.0;
    double high_hz = 0.0;
    bool effective_before_decimation = false;
    std::string operator_evidence_id;
};

// This is a compact declaration of established line-strategy facts. It is not
// a detector/sample mask and it does not claim that a pre-decimation operator
// exists unless a bound interval explicitly says so.
class RtcFilterD2LineMask {
public:
    static std::shared_ptr<const RtcFilterD2LineMask> admit(
        std::string policy_id, RtcFilterD2LineMaskDisposition disposition,
        std::vector<RtcFilterD2LineProtectionInterval> intervals) {
        if (policy_id.empty()) {
            throw std::invalid_argument(
                "D2 line-mask policy identity is empty");
        }
        if ((disposition ==
                 RtcFilterD2LineMaskDisposition::complete_no_lines ||
             disposition == RtcFilterD2LineMaskDisposition::pending) &&
            !intervals.empty()) {
            throw std::invalid_argument(
                "D2 empty/pending line-mask disposition has intervals");
        }
        for (const auto &interval : intervals) {
            if (interval.interval_id.empty() ||
                !std::isfinite(interval.low_hz) ||
                !std::isfinite(interval.high_hz) || interval.low_hz < 0.0 ||
                !(interval.high_hz > interval.low_hz) ||
                (interval.effective_before_decimation &&
                 interval.operator_evidence_id.empty())) {
                throw std::invalid_argument(
                    "D2 line-protection interval is incomplete");
            }
        }
        std::sort(intervals.begin(), intervals.end(),
                  [](const auto &lhs, const auto &rhs) {
                      if (lhs.low_hz != rhs.low_hz) {
                          return lhs.low_hz < rhs.low_hz;
                      }
                      return lhs.high_hz < rhs.high_hz;
                  });
        for (std::size_t index = 1; index < intervals.size(); ++index) {
            if (intervals[index].low_hz < intervals[index - 1].high_hz) {
                throw std::invalid_argument(
                    "D2 line-protection intervals overlap");
            }
        }
        return std::shared_ptr<const RtcFilterD2LineMask>(
            new RtcFilterD2LineMask{
                std::move(policy_id), disposition, std::move(intervals)});
    }

    const std::string &policy_id() const noexcept { return policy_id_; }
    const char *strategy_id() const noexcept {
        return rtc_filter_d2_line_strategy_id;
    }
    RtcFilterD2LineMaskDisposition disposition() const noexcept {
        return disposition_;
    }
    std::span<const RtcFilterD2LineProtectionInterval> intervals()
        const noexcept {
        return intervals_;
    }

private:
    RtcFilterD2LineMask(
        std::string policy_id, RtcFilterD2LineMaskDisposition disposition,
        std::vector<RtcFilterD2LineProtectionInterval> intervals)
        : policy_id_{std::move(policy_id)}, disposition_{disposition},
          intervals_{std::move(intervals)} {}

    std::string policy_id_;
    RtcFilterD2LineMaskDisposition disposition_;
    std::vector<RtcFilterD2LineProtectionInterval> intervals_;
};

// Source exclusion is genuinely derived route evidence, so it is owned once
// as one compact byte per detector/sample cell. Immutable timing and detector
// axes remain referenced through the accepted PairedReadout occurrence axis.
class RtcFilterD2SourceMask {
public:
    static std::shared_ptr<const RtcFilterD2SourceMask> admit(
        std::shared_ptr<const PairedReadoutOccurrenceAxis> occurrence_axis,
        Eigen::Index detector_count, std::string policy_id,
        RtcFilterD2SourceMaskDisposition disposition,
        std::vector<std::uint8_t> excluded) {
        if (!occurrence_axis || detector_count <= 0 || policy_id.empty()) {
            throw std::invalid_argument(
                "D2 source-mask identity or shape is incomplete");
        }
        const auto rows = occurrence_axis->occurrence_count();
        const auto columns = static_cast<std::size_t>(detector_count);
        if (columns != 0 &&
            rows > std::numeric_limits<std::size_t>::max() / columns) {
            throw std::length_error("D2 source-mask cardinality overflows");
        }
        if (excluded.size() != rows * columns ||
            std::any_of(excluded.begin(), excluded.end(),
                        [](std::uint8_t value) { return value > 1U; })) {
            throw std::invalid_argument(
                "D2 source-mask payload does not match its native axes");
        }
        if (disposition ==
                RtcFilterD2SourceMaskDisposition::approved_not_applicable &&
            std::any_of(excluded.begin(), excluded.end(),
                        [](std::uint8_t value) { return value != 0U; })) {
            throw std::invalid_argument(
                "not-applicable D2 source mask excludes samples");
        }
        return std::shared_ptr<const RtcFilterD2SourceMask>(
            new RtcFilterD2SourceMask{
                std::move(occurrence_axis), detector_count,
                std::move(policy_id), disposition, std::move(excluded)});
    }

    const std::shared_ptr<const PairedReadoutOccurrenceAxis> &
    occurrence_axis_handle() const noexcept {
        return occurrence_axis_;
    }
    TimestreamNetworkId network_id() const noexcept {
        return occurrence_axis_->network_id();
    }
    Eigen::Index detector_count() const noexcept { return detector_count_; }
    const std::string &policy_id() const noexcept { return policy_id_; }
    RtcFilterD2SourceMaskDisposition disposition() const noexcept {
        return disposition_;
    }
    std::span<const std::uint8_t> contiguous_exclusions() const noexcept {
        return excluded_;
    }
    bool excluded(TimestreamNativeRow native_row,
                  Eigen::Index detector) const {
        return excluded_.at(flat_index(native_row, detector)) != 0U;
    }
    std::size_t excluded_count() const noexcept {
        return static_cast<std::size_t>(
            std::count(excluded_.begin(), excluded_.end(), std::uint8_t{1}));
    }

private:
    RtcFilterD2SourceMask(
        std::shared_ptr<const PairedReadoutOccurrenceAxis> occurrence_axis,
        Eigen::Index detector_count, std::string policy_id,
        RtcFilterD2SourceMaskDisposition disposition,
        std::vector<std::uint8_t> excluded)
        : occurrence_axis_{std::move(occurrence_axis)},
          detector_count_{detector_count}, policy_id_{std::move(policy_id)},
          disposition_{disposition}, excluded_{std::move(excluded)} {}

    std::size_t flat_index(TimestreamNativeRow native_row,
                           Eigen::Index detector) const {
        if (native_row < occurrence_axis_->first_native_row() ||
            native_row >= occurrence_axis_->past_last_native_row() ||
            detector < 0 || detector >= detector_count_) {
            throw std::out_of_range(
                "D2 source-mask cell is outside its native axes");
        }
        const auto row = static_cast<std::size_t>(
            native_row - occurrence_axis_->first_native_row());
        return row * static_cast<std::size_t>(detector_count_) +
               static_cast<std::size_t>(detector);
    }

    std::shared_ptr<const PairedReadoutOccurrenceAxis> occurrence_axis_;
    Eigen::Index detector_count_ = 0;
    std::string policy_id_;
    RtcFilterD2SourceMaskDisposition disposition_;
    std::vector<std::uint8_t> excluded_;
};

struct RtcFilterD2CleaningRealization {
    std::string operator_id;
    std::string effective_config_id;
    std::string grouping;
    bool sampling_changed = false;
    bool common_analysis_grid_requested = false;

    bool complete() const noexcept {
        return !operator_id.empty() && !effective_config_id.empty() &&
               !grouping.empty() && !sampling_changed &&
               !common_analysis_grid_requested;
    }
};

struct RtcFilterD2PlaneMemoryEvidence {
    std::size_t owned_numeric_bytes = 0;
    std::size_t owned_residual_validity_bytes = 0;
    std::size_t shared_source_mask_bytes = 0;
    std::size_t referenced_native_axis_count = 0;
};

// A measurement-only in-memory product. The prefilter stage is a zero-copy
// view of one PairedReadout member. The residual stage owns only the derived
// numerical plane and compact additional validity, while retaining the exact
// source network axis, detector axis, source mask, and line declaration.
class RtcFilterD2NetworkPlane {
public:
    static std::shared_ptr<const RtcFilterD2NetworkPlane> observe_prefilter(
        std::shared_ptr<const PairedReadout> paired,
        TimestreamNetworkId network_id, ReadoutMember member,
        std::shared_ptr<const RtcFilterD2SourceMask> source_mask,
        std::shared_ptr<const RtcFilterD2LineMask> line_mask) {
        if (!paired || !source_mask || !line_mask) {
            throw std::invalid_argument(
                "D2 prefilter observer lacks required handles");
        }
        const auto &network = paired->network(network_id);
        require_same_axes(network, *source_mask);
        return std::shared_ptr<const RtcFilterD2NetworkPlane>(
            new RtcFilterD2NetworkPlane{
                std::move(paired), network_id, member,
                RtcFilterD2MeasurementStage::native_prefilter,
                std::move(source_mask), std::move(line_mask), {}, {}, {}});
    }

    static std::shared_ptr<const RtcFilterD2NetworkPlane>
    observe_post_cleaning_residual(
        std::shared_ptr<const RtcFilterD2NetworkPlane> prefilter,
        PairedReadoutMatrix residual_values,
        std::vector<std::uint8_t> residual_valid,
        RtcFilterD2CleaningRealization realization) {
        if (!prefilter ||
            prefilter->stage() !=
                RtcFilterD2MeasurementStage::native_prefilter ||
            !realization.complete()) {
            throw std::invalid_argument(
                "D2 residual lacks a complete native prefilter binding");
        }
        const auto rows = prefilter->occurrence_count();
        const auto columns = prefilter->detector_count();
        if (residual_values.rows() != rows ||
            residual_values.cols() != columns ||
            residual_valid.size() !=
                static_cast<std::size_t>(rows * columns) ||
            !residual_values.array().isFinite().all() ||
            std::any_of(residual_valid.begin(), residual_valid.end(),
                        [](std::uint8_t value) { return value > 1U; })) {
            throw std::invalid_argument(
                "D2 residual payload differs from its native prefilter axes");
        }
        return std::shared_ptr<const RtcFilterD2NetworkPlane>(
            new RtcFilterD2NetworkPlane{
                prefilter->paired_, prefilter->network_id_,
                prefilter->member_,
                RtcFilterD2MeasurementStage::native_post_cleaning_residual,
                prefilter->source_mask_, prefilter->line_mask_,
                std::move(prefilter), std::move(residual_values),
                std::move(residual_valid), std::move(realization)});
    }

    RtcFilterD2MeasurementStage stage() const noexcept { return stage_; }
    TimestreamNetworkId network_id() const noexcept { return network_id_; }
    ReadoutMember member() const noexcept { return member_; }
    Eigen::Index occurrence_count() const noexcept {
        return network().occurrence_count();
    }
    Eigen::Index detector_count() const noexcept {
        return network().detector_count();
    }
    std::int64_t array_id() const noexcept { return array_id_; }
    const PairedReadoutNetwork &network() const {
        return paired_->network(network_id_);
    }
    const std::shared_ptr<const PairedReadout> &paired_handle()
        const noexcept {
        return paired_;
    }
    const std::shared_ptr<const PairedReadoutOccurrenceAxis> &
    occurrence_axis_handle() const noexcept {
        return source_mask_->occurrence_axis_handle();
    }
    std::span<const PairedReadoutDetectorIdentity> detectors() const noexcept {
        return network().detectors();
    }
    const std::shared_ptr<const RtcFilterD2SourceMask> &source_mask_handle()
        const noexcept {
        return source_mask_;
    }
    const std::shared_ptr<const RtcFilterD2LineMask> &line_mask_handle()
        const noexcept {
        return line_mask_;
    }
    const std::shared_ptr<const RtcFilterD2NetworkPlane> &prefilter_handle()
        const noexcept {
        return prefilter_;
    }
    const RtcFilterD2CleaningRealization &cleaning_realization() const {
        if (stage_ !=
            RtcFilterD2MeasurementStage::native_post_cleaning_residual) {
            throw std::logic_error(
                "D2 prefilter plane has no cleaning realization");
        }
        return cleaning_realization_;
    }
    const PairedReadoutMatrix &values() const noexcept {
        return stage_ == RtcFilterD2MeasurementStage::native_prefilter
                   ? network().values(member_)
                   : residual_values_;
    }
    std::span<const double> contiguous_values() const noexcept {
        const auto &plane = values();
        return {plane.data(), static_cast<std::size_t>(plane.size())};
    }
    bool valid(TimestreamNativeRow native_row, Eigen::Index detector) const {
        if (!network().pair_valid(native_row, detector)) return false;
        if (stage_ == RtcFilterD2MeasurementStage::native_prefilter) {
            return true;
        }
        return residual_valid_.at(flat_index(native_row, detector)) != 0U;
    }
    bool source_excluded(TimestreamNativeRow native_row,
                         Eigen::Index detector) const {
        return source_mask_->excluded(native_row, detector);
    }
    const std::string &signal_unit_id() const noexcept {
        const auto &identity = *network().mapping_identity_handle();
        return member_ == ReadoutMember::x ? identity.x_raw_unit_id
                                           : identity.r_raw_unit_id;
    }
    std::vector<NativeContiguousRun> physical_runs() const {
        const auto &axis = *occurrence_axis_handle();
        return partition_native_contiguous_runs(
            *axis.native_timing_handle(), axis.first_native_row(),
            axis.past_last_native_row());
    }
    RtcFilterD2PlaneMemoryEvidence memory_evidence() const noexcept {
        return {
            stage_ ==
                    RtcFilterD2MeasurementStage::native_post_cleaning_residual
                ? static_cast<std::size_t>(residual_values_.size()) *
                      sizeof(double)
                : 0U,
            residual_valid_.size(),
            source_mask_->contiguous_exclusions().size(),
            1U};
    }

private:
    static void require_same_axes(const PairedReadoutNetwork &network,
                                  const RtcFilterD2SourceMask &mask) {
        if (network.network_id() != mask.network_id() ||
            network.detector_count() != mask.detector_count() ||
            network.occurrence_axis_handle().get() !=
                mask.occurrence_axis_handle().get()) {
            throw std::invalid_argument(
                "D2 source mask is not bound to the exact network axes");
        }
    }

    RtcFilterD2NetworkPlane(
        std::shared_ptr<const PairedReadout> paired,
        TimestreamNetworkId network_id, ReadoutMember member,
        RtcFilterD2MeasurementStage stage,
        std::shared_ptr<const RtcFilterD2SourceMask> source_mask,
        std::shared_ptr<const RtcFilterD2LineMask> line_mask,
        std::shared_ptr<const RtcFilterD2NetworkPlane> prefilter,
        PairedReadoutMatrix residual_values,
        std::vector<std::uint8_t> residual_valid,
        RtcFilterD2CleaningRealization cleaning_realization = {})
        : paired_{std::move(paired)}, network_id_{network_id}, member_{member},
          stage_{stage}, source_mask_{std::move(source_mask)},
          line_mask_{std::move(line_mask)}, prefilter_{std::move(prefilter)},
          residual_values_{std::move(residual_values)},
          residual_valid_{std::move(residual_valid)},
          cleaning_realization_{std::move(cleaning_realization)} {
        const auto detector_axis = network().detectors();
        array_id_ = detector_axis.front().array_id;
        if (std::any_of(detector_axis.begin(), detector_axis.end(),
                        [&](const auto &detector) {
                            return detector.array_id != array_id_;
                        })) {
            throw std::invalid_argument(
                "one D2 network plane spans multiple array identities");
        }
    }

    std::size_t flat_index(TimestreamNativeRow native_row,
                           Eigen::Index detector) const {
        const auto &axis = *occurrence_axis_handle();
        if (native_row < axis.first_native_row() ||
            native_row >= axis.past_last_native_row() || detector < 0 ||
            detector >= detector_count()) {
            throw std::out_of_range(
                "D2 residual-validity cell is outside native axes");
        }
        const auto row = static_cast<std::size_t>(
            native_row - axis.first_native_row());
        return row * static_cast<std::size_t>(detector_count()) +
               static_cast<std::size_t>(detector);
    }

    std::shared_ptr<const PairedReadout> paired_;
    TimestreamNetworkId network_id_ = -1;
    ReadoutMember member_ = ReadoutMember::x;
    RtcFilterD2MeasurementStage stage_ =
        RtcFilterD2MeasurementStage::native_prefilter;
    std::shared_ptr<const RtcFilterD2SourceMask> source_mask_;
    std::shared_ptr<const RtcFilterD2LineMask> line_mask_;
    std::shared_ptr<const RtcFilterD2NetworkPlane> prefilter_;
    PairedReadoutMatrix residual_values_;
    std::vector<std::uint8_t> residual_valid_;
    RtcFilterD2CleaningRealization cleaning_realization_;
    std::int64_t array_id_ = -1;
};

}  // namespace citlali::pipeline
