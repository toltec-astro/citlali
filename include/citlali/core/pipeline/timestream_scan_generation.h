#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/timestream_native_consumer_bridge.h>
#include <citlali/core/pipeline/timestream_native_pointing.h>
#include <citlali/core/timestream/timestream.h>

#include <Eigen/Core>

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace citlali::pipeline {

// B2b is an explicit Science/Pointing-only activation.  The legacy scan
// loader remains available to Beammap, but it cannot silently become the
// compatibility route for a native-required reduction.
template <class Engine>
void require_native_science_pointing_ingress(const Engine &engine) {
    if (engine.ptcproc.weight_validation_is_enabled()) {
        throw std::runtime_error(
            "native Science/Pointing ingress does not permit dense legacy-UID weight validation before sparse typed identity state exists");
    }
    if (engine.ptcproc.weight_corr_penalty.enabled ||
        engine.ptcproc.busy_row_suppression.enabled) {
        throw std::runtime_error(
            "native Science/Pointing ingress does not permit nontransactional processed-weight diagnostic penalties");
    }
    if (engine.learning.is_enabled()) {
        throw std::runtime_error(
            "native Science/Pointing ingress does not permit legacy APT-identity learning consumers");
    }
    const auto &line_audit = engine.rtcproc.line_audit;
    if (line_audit.enabled &&
        line_audit.ptc_model_protected_enabled &&
        (line_audit.ptc_apply_fixed_notches ||
         line_audit.ptc_apply_shared_notches ||
         line_audit.ptc_apply_detector_notches)) {
        throw std::runtime_error(
            "native Science/Pointing ingress does not permit model-protected PTC line-audit mutation before bounded native-run transaction support exists");
    }
    if (tod_output_enabled(engine)) {
        throw std::runtime_error(
            "native Science/Pointing ingress cannot write RTC/PTC TOD products before native output provenance is synchronized");
    }
    if (citlali::config::is_maximum_likelihood_map_method(
            mapmaking_config(engine).method)) {
        throw std::runtime_error(
            "native Science/Pointing ingress does not permit maximum-likelihood mapmaking without a native-pointing consumer");
    }
    if (engine.rtcproc.run_polarization) {
        throw std::runtime_error(
            "native Science/Pointing ingress does not permit polarization before a network-native polarization/HWPR carrier exists");
    }
    (void)engine.calib.require_apt_detector_relation();
    if (engine.alignment.native_consumer_plan == nullptr) {
        throw std::runtime_error(
            "native Science/Pointing ingress requires an immutable native alignment plan");
    }
    if (engine.alignment.native_pointing_plan == nullptr ||
        !engine.alignment.native_pointing_plan->bound_to(
            engine.alignment.native_consumer_plan)) {
        throw std::runtime_error(
            "native Science/Pointing ingress requires an exact network-native pointing plan");
    }
}

// Candidate for one Science/Pointing scan.  The telescope scan indices bound
// relational common-slot membership only.  They never supply detector time or
// telescope coordinates; every admitted cell retains its mapped native row
// and reconstructed timestamp, and pointing comes from NativePointingPlan.
struct NativeScienceScanAdmissionPlan {
    NativeOperationIdentity operation;
    Eigen::Index first_inner_output_row = 0;
    Eigen::Index past_last_inner_output_row = 0;
    Eigen::Matrix<Eigen::Index, 4, 1> output_scan_indices;
    std::vector<NativeCompleteCohortRun> runs;
    std::vector<NativeMeasuredScanSegment> segments;
};

template <class ScanIndices>
NativeScienceScanAdmissionPlan plan_native_science_scan(
    const NativeAlignmentPlan &alignment,
    const ScanIndices &relational_scan_indices, Eigen::Index scan) {
    if (relational_scan_indices.rows() != 4 || scan < 0 ||
        scan >= relational_scan_indices.cols() ||
        static_cast<std::uintmax_t>(scan) >
            static_cast<std::uintmax_t>(
                std::numeric_limits<std::int64_t>::max())) {
        throw std::invalid_argument(
            "native Science/Pointing scan requires one valid four-row relational scan column");
    }

    const Eigen::Index inner_first = relational_scan_indices(0, scan);
    const Eigen::Index inner_last = relational_scan_indices(1, scan);
    const Eigen::Index outer_first = relational_scan_indices(2, scan);
    const Eigen::Index outer_last = relational_scan_indices(3, scan);
    if (outer_first < 0 || inner_first < outer_first ||
        inner_last < inner_first || outer_last < inner_last) {
        throw std::invalid_argument(
            "native Science/Pointing relational scan intervals are invalid");
    }
    const auto slot_count = alignment.slot_count();
    if (static_cast<std::uintmax_t>(outer_last) >=
        static_cast<std::uintmax_t>(slot_count)) {
        throw std::out_of_range(
            "native Science/Pointing relational scan exceeds the alignment slots");
    }

    const auto outer_begin = static_cast<std::size_t>(outer_first);
    const auto outer_past = static_cast<std::size_t>(outer_last) + 1;
    const auto inner_begin = static_cast<std::size_t>(inner_first);
    const auto inner_past = static_cast<std::size_t>(inner_last) + 1;
    NativeScienceScanAdmissionPlan candidate{
        NativeOperationIdentity{
            0, static_cast<std::int64_t>(scan)}};
    candidate.runs = partition_complete_native_cohort_runs(
        alignment, candidate.operation, outer_begin, outer_past, 0);
    if (candidate.runs.empty()) {
        throw std::runtime_error(
            "native Science/Pointing scan has no complete measured native cohort run");
    }

    Eigen::Index next_output_row = 0;
    std::optional<Eigen::Index> first_inner_output;
    std::optional<Eigen::Index> past_last_inner_output;
    candidate.segments.reserve(candidate.runs.size());
    for (std::size_t run_index = 0;
         run_index < candidate.runs.size(); ++run_index) {
        const auto &run = candidate.runs[run_index];
        require_complete_native_cohort(run.selection);
        const auto &common_slots =
            run.selection.relational_common_slots();
        if (run.run_ordinal != run_index || common_slots.empty() ||
            common_slots.front() < outer_begin ||
            common_slots.back() >= outer_past ||
            common_slots.size() > static_cast<std::size_t>(
                                      std::numeric_limits<Eigen::Index>::max()) ||
            next_output_row >
                std::numeric_limits<Eigen::Index>::max() -
                    static_cast<Eigen::Index>(common_slots.size())) {
            throw std::logic_error(
                "native Science/Pointing complete run has invalid relational/output ownership");
        }
        const Eigen::Index first_output_row = next_output_row;
        const Eigen::Index run_rows =
            static_cast<Eigen::Index>(common_slots.size());
        next_output_row += run_rows;

        for (Eigen::Index local = 0; local < run_rows; ++local) {
            const auto common_slot = common_slots.at(
                static_cast<std::size_t>(local));
            if (common_slot >= inner_begin && common_slot < inner_past) {
                const Eigen::Index output_row = first_output_row + local;
                if (!first_inner_output.has_value()) {
                    first_inner_output = output_row;
                }
                past_last_inner_output = output_row + 1;
            }
        }
        candidate.segments.emplace_back(
            run.run_ordinal, first_output_row, next_output_row,
            common_slots, run.selection, run.participant_runs);
    }
    if (!first_inner_output.has_value() ||
        !past_last_inner_output.has_value()) {
        throw std::runtime_error(
            "native Science/Pointing scan has no complete measured inner-science row");
    }

    // Relational slot order is strictly increasing, so after absent/invalid
    // slots create nothing, the retained inner rows still form one contiguous
    // output interval.  Verify that invariant rather than inferring physical
    // time from the common-slot coordinate.
    for (const auto &segment : candidate.segments) {
        for (Eigen::Index local = 0; local < segment.row_count(); ++local) {
            const auto common_slot =
                segment.relational_common_slots().at(
                    static_cast<std::size_t>(local));
            const Eigen::Index output_row =
                segment.first_output_row() + local;
            const bool in_relational_inner =
                common_slot >= inner_begin && common_slot < inner_past;
            const bool in_output_inner =
                output_row >= *first_inner_output &&
                output_row < *past_last_inner_output;
            if (in_relational_inner != in_output_inner) {
                throw std::logic_error(
                    "native Science/Pointing inner output interval is not contiguous");
            }
        }
    }

    candidate.first_inner_output_row = *first_inner_output;
    candidate.past_last_inner_output_row = *past_last_inner_output;
    candidate.output_scan_indices <<
        candidate.first_inner_output_row,
        candidate.past_last_inner_output_row - 1, 0,
        next_output_row - 1;
    return candidate;
}

template <class RtcData, class KidsProc, class RawObs, class ScanIndices,
          class TimestreamType>
RtcData make_native_rtc_scan_samples(
    KidsProc &kidsproc, const RawObs &rawobs, Eigen::Index scan,
    const ScanIndices &relational_scan_indices,
    std::shared_ptr<const NativeAlignmentPlan> alignment_plan,
    std::shared_ptr<const NativePointingPlan> pointing_plan,
    std::shared_ptr<const AptDetectorRelation> detector_relation,
    TimestreamType timestream_type) {
    if (!alignment_plan || !pointing_plan || !detector_relation ||
        !pointing_plan->bound_to(alignment_plan)) {
        throw std::invalid_argument(
            "native Science/Pointing scan requires exact alignment, pointing, and typed detector handles");
    }

    auto plan = plan_native_science_scan(
        *alignment_plan, relational_scan_indices, scan);
    auto measured = kidsproc.solve_native_detector_scan(
        rawobs, *alignment_plan, *detector_relation, plan.runs,
        timestream_type);
    auto state = NativeMeasuredScanState::admit(
        plan.operation, alignment_plan, pointing_plan, detector_relation,
        plan.first_inner_output_row, plan.past_last_inner_output_row,
        std::move(plan.segments), std::move(measured.measured_blocks));
    state->require_compatible(
        measured.measured_values.rows(), measured.measured_values.cols(),
        scan);

    // Publish only after solving, typed joins, cohort/run admission, native
    // pointing checks, and ledger construction all succeed.  The rectangular
    // telescope compatibility fields intentionally remain empty.
    RtcData candidate;
    candidate.scans.data = std::move(measured.measured_values);
    candidate.scan_indices.data = plan.output_scan_indices;
    candidate.index.data = scan;
    candidate.native_scan = std::move(state);
    candidate.native_science_mode =
        timestream::NativeScienceMode::native_required;
    candidate.require_native_science_mode_consistent();
    candidate.require_native_scan().require_compatible(
        candidate.scans.data.rows(), candidate.scans.data.cols(),
        candidate.index.data);
    return candidate;
}

class ScanCursor {
public:
    explicit ScanCursor(Eigen::Index scan_count) : scan_count_(scan_count) {}

    std::optional<Eigen::Index> next() noexcept {
        if (next_scan_ >= scan_count_) {
            return std::nullopt;
        }
        return next_scan_++;
    }

private:
    Eigen::Index scan_count_ = 0;
    Eigen::Index next_scan_ = 0;
};

template <class RtcData, class Telescope>
Eigen::Index initialize_rtc_scan(
    RtcData &rtcdata, const Telescope &telescope, Eigen::Index scan) {
    rtcdata.scan_indices.data = telescope.scan_indices.col(scan);
    rtcdata.index.data = scan;
    return rtcdata.scan_indices.data(3) - rtcdata.scan_indices.data(2) + 1;
}

template <class RtcData, class MapBuffer, class Calib,
          class RandomDistribution, class RandomEngine>
void populate_noise_map_signs(
    RtcData &rtcdata, const MapBuffer &omb, const Calib &calib,
    bool enabled, RandomDistribution &rands, RandomEngine &eng) {
    if (!enabled) {
        return;
    }

    if (omb.randomize_dets) {
        rtcdata.noise.data =
            Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>::Zero(
                omb.n_noise, calib.n_dets)
                .unaryExpr([&](int) { return 2 * rands(eng) - 1; });
    }
    else {
        rtcdata.noise.data =
            Eigen::Matrix<int, Eigen::Dynamic, 1>::Zero(omb.n_noise)
                .unaryExpr([&](int) { return 2 * rands(eng) - 1; });
    }
}

template <class RtcData, class KidsProc, class RawObs, class Telescope,
          class StartIndices, class EndIndices, class TCommon, class NwTimes,
          class Masks, class TimestreamType>
void populate_rtc_scan_samples(
    RtcData &rtcdata, KidsProc &kidsproc, RawObs &rawobs, Eigen::Index scan,
    Telescope &telescope, StartIndices &start_indices, EndIndices &end_indices,
    TCommon &t_common, NwTimes &nw_times, Masks &masks,
    bool interp_over_gaps, int scan_length, int n_dets,
    TimestreamType timestream_type) {
    if (!interp_over_gaps) {
        rtcdata.scans.data = kidsproc.populate_rtc_from_rawobs(
            rawobs, scan, telescope.scan_indices, start_indices, end_indices,
            scan_length, n_dets, timestream_type);
        return;
    }

    const double gap_tolerance = 1 / (2 * telescope.fsmp);
    auto scan_rawobs = kidsproc.load_rawobs_gaps(
        rawobs, scan, telescope.scan_indices, start_indices, t_common,
        nw_times, gap_tolerance);
    rtcdata.scans.data = kidsproc.populate_rtc_gaps(
        scan_rawobs, t_common, nw_times, masks, scan, gap_tolerance,
        telescope.scan_indices, scan_length, n_dets, timestream_type);
    decltype(scan_rawobs)().swap(scan_rawobs);
}

}  // namespace citlali::pipeline
