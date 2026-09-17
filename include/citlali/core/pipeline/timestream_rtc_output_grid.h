#pragma once

#include <citlali/core/pipeline/timestream_identity_route_context.h>
#include <citlali/core/pipeline/timestream_rtc_pipeline.h>

namespace citlali::pipeline {

// RTC-owned preparation for the scientific output boundary, NOT a completed
// terminal product. The available ALIGN context supplies native occurrence
// assignments and AST motion, not an AST detector-direction parent. RTC-only
// completion requires output VAL facts and finalization; detector coordinates
// are additionally required when that role is requested. Motion is not promoted
// to detector pointing, and RTC filtering does not acquire an APT dependency.
class RtcOutputGrid {
public:
    struct DetectorGrid {
        TimestreamNetworkId network;
        std::uint32_t detector;
        TimestreamNativeRow first, past_last;
        std::uint32_t factor;
        std::size_t scheduled_count;
    };

    // A slot is stable within this exact grid/result, including unavailable
    // slots. It is NOT an offset into the compact vector of finite output rows
    // and NOT a native row or VAL native-target identity.
    struct Occurrence {
        std::size_t slot;
        IdentityRouteOccurrenceAssignment representative;
        bool x_available, r_available;
        bool representative_replaced, representative_excluded;
        bool replacement_influence, unrepaired_influence;
        RtcNotchRecoveryCause input_cause, support_cause, realized_cause;
        RtcSpeedRestriction speed_restriction;
        std::optional<RtcEventRange> realized_segment;
        // Filter-local footprint only. Donor dependencies remain in the exact
        // retained plans/results; this interval is never labeled total support.
        std::optional<RtcEventRange> filter_footprint;
    };

    static std::shared_ptr<const RtcOutputGrid> prepare(
        std::shared_ptr<const RtcPipelineResult> applied,
        std::shared_ptr<const IdentityRouteAlignContext> align) {
        if (!applied || !align ||
            applied->plan_handle()->input_handle()->parent_handle().get() !=
                align->paired_handle().get() ||
            applied->plan_handle()->snapshot_handle().get() !=
                align->val_snapshot_handle().get()) {
            throw std::invalid_argument(
                "RTC output grid requires exact Apply, ALIGN and frozen VAL");
        }
        auto out = std::shared_ptr<RtcOutputGrid>(new RtcOutputGrid);
        out->applied_ = std::move(applied);
        out->align_ = std::move(align);
        for (const auto &result : out->applied_->detector_results()) {
            const auto &plan = *result->plan_handle();
            const auto &candidate = *plan.assessment_handle()->candidate_handle();
            const auto &spec = candidate.specification();
            const auto &span = out->applied_->plan_handle()->input_handle()->span(candidate.network());
            const auto &motion = out->align_->ast_views_handle()->network(candidate.network());
            if (result->injection_identity() != "none" ||
                plan.domain().motion->raw_product_handle().get() !=
                    motion.raw_product_handle().get() ||
                plan.domain().motion->network_timing_handle().get() !=
                    motion.network_timing_handle().get() ||
                plan.first_native_row() != span.first_native_row ||
                result->filtered_native_pair().rows() !=
                    span.past_last_native_row - span.first_native_row ||
                !spec.notches.empty() || spec.factor == 0) {
                throw std::invalid_argument(
                    "RTC output grid has foreign motion, incomplete support or diagnostic overlay");
            }
            const auto count = static_cast<std::size_t>(
                span.past_last_native_row - span.first_native_row);
            out->detectors_.push_back({candidate.network(), candidate.detector(),
                span.first_native_row, span.past_last_native_row, spec.factor,
                1 + (count - 1) / spec.factor});
            // The retained schedule is a subset of the frozen phase-zero grid.
            // Availability does not create a new grid or compress native time.
            auto previous = span.first_native_row - 1;
            for (auto row : result->output_native_rows()) {
                if (row <= previous || row < span.first_native_row ||
                    row >= span.past_last_native_row ||
                    (row - span.first_native_row) % spec.factor != 0) {
                    throw std::invalid_argument("RTC realized output differs from its frozen grid");
                }
                previous = row;
                for (auto c : {NativeReadoutCoordinate::x, NativeReadoutCoordinate::r}) {
                    if (result->coordinate_stage_available(c, row, true) &&
                        !std::isfinite(result->filtered_native_pair()(
                            row - span.first_native_row, static_cast<int>(c)))) {
                        throw std::invalid_argument("RTC output availability contradicts payload");
                    }
                }
            }
        }
        return out;
    }

    const auto &applied_handle() const noexcept { return applied_; }
    const auto &align_handle() const noexcept { return align_; }
    const auto &input_val_snapshot_handle() const noexcept {
        return applied_->plan_handle()->snapshot_handle();
    }
    const auto &detectors() const noexcept { return detectors_; }

    static ValRtcOutputTarget val_target(std::shared_ptr<const RtcOutputGrid> grid,
        std::size_t detector_grid, std::size_t slot, NativeReadoutCoordinate coordinate) {
        if (!grid || (coordinate != NativeReadoutCoordinate::x &&
                      coordinate != NativeReadoutCoordinate::r))
            throw std::invalid_argument("VAL RTC target requires exact grid and coordinate");
        const auto &g = grid->detectors_.at(detector_grid);
        const auto fact = grid->occurrence(detector_grid, slot);
        auto snapshot = grid->input_val_snapshot_handle();
        auto address = snapshot->address(g.network,
            fact.representative.network_occurrence.native_row(), g.detector);
        return ValRtcOutputTarget{std::move(grid), std::move(snapshot),
                                  std::move(address), slot, coordinate};
    }

    Occurrence occurrence(std::size_t detector_grid, std::size_t slot) const {
        const auto &g = detectors_.at(detector_grid);
        if (slot >= g.scheduled_count)
            throw std::out_of_range("RTC output slot outside declared grid");
        const auto row = g.first + static_cast<TimestreamNativeRow>(slot * g.factor);
        const auto &result = *applied_->detector_results().at(detector_grid);
        const auto &plan = *result.plan_handle();
        const auto local = static_cast<std::size_t>(row - g.first);
        std::optional<RtcEventRange> segment, footprint;
        const auto &runs = plan.runs();
        auto run = std::upper_bound(runs.begin(), runs.end(), row,
            [](auto value, const auto &r) { return value < r.first; });
        if (run != runs.begin()) {
            --run;
            if (row < run->past_last) segment = *run;
        }
        const auto &realized = result.output_native_rows();
        const bool applied = std::binary_search(realized.begin(), realized.end(), row);
        if (applied) {
            const auto half = static_cast<TimestreamNativeRow>(plan.full_support_half_samples());
            footprint = RtcEventRange{row - half, row + half + 1};
            if (!segment || footprint->first < segment->first ||
                footprint->past_last > segment->past_last)
                throw std::logic_error("RTC filter footprint crosses its physical/support boundary");
        }
        return {slot, align_->occurrence_assignment(g.network, row),
            applied && result.coordinate_stage_available(NativeReadoutCoordinate::x, row, true),
            applied && result.coordinate_stage_available(NativeReadoutCoordinate::r, row, true),
            result.representative_replaced(row), result.requires_representative_exclusion(row),
            result.replacement_influence(row, true), result.unrepaired_influence(row, true),
            plan.input_causes().at(local), plan.support_causes().at(local),
            result.causes().at(local), plan.speed_restrictions().at(local),
            segment, footprint};
    }

    std::optional<double> value(std::size_t detector_grid, std::size_t slot,
                                NativeReadoutCoordinate coordinate) const {
        if (coordinate != NativeReadoutCoordinate::x && coordinate != NativeReadoutCoordinate::r)
            throw std::invalid_argument("RTC output coordinate must be x or r");
        const auto fact = occurrence(detector_grid, slot);
        if (!(coordinate == NativeReadoutCoordinate::x ? fact.x_available : fact.r_available))
            return std::nullopt;
        const auto &g = detectors_.at(detector_grid);
        return applied_->detector_results().at(detector_grid)->filtered_native_pair()(
            fact.representative.network_occurrence.native_row() - g.first,
            static_cast<int>(coordinate));
    }

    // Heavy values, native axes, plans, provenance, donor state and VAL remain
    // referenced. Only one small descriptor is owned per detector.
    std::size_t owned_descriptor_bytes() const noexcept {
        return detectors_.size() * sizeof(DetectorGrid);
    }

private:
    RtcOutputGrid() = default;
    std::shared_ptr<const RtcPipelineResult> applied_;
    std::shared_ptr<const IdentityRouteAlignContext> align_;
    std::vector<DetectorGrid> detectors_;
};

} // namespace citlali::pipeline
