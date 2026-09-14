#pragma once
// Test-only anchor adapters for a missed injected candidate. Numerical bodies
// track their respective source-bound production functions:
// timestream_rtc_jump_transition.h::measure and
// timestream_rtc_event_assessment.h::recover. Only candidate lookup is replaced
// by explicit test-supplied metadata, including the onset-edge selection. The diagnostic driver verifies equality
// against both production functions on every detected replay group/coordinate.
// This header is private to the inert injection executable. It neither creates
// RTC evidence nor changes the immutable producer candidates or original data.
namespace {
using namespace citlali::pipeline;
inline RtcJumpTransition diagnostic_measure_at(const RtcSpikeEvidence &spikes,
                                 const RtcAssessedEvent &event, std::size_t c,
                                 const NativeContiguousRun &run,
                                 const std::vector<RtcEventRange> &neighbors,
                                 const RtcSpikeCandidate &seed,
                                 std::span<const RtcSpikeCandidate> members) {
    using namespace rtc_jump_transition_detail;
    using namespace rtc_event_assessment_detail;
    RtcJumpTransition out;
    const auto &fit = event.background[c];
    if (!fit.available()) { out.cause = RtcJumpTransitionCause::background_unavailable; return out; }
    out.frozen_residual_scale = fit.pre_scale_fit.scale;
    const auto &net = spikes.input_handle()->network(event.network);
    const auto &axis = net.occurrence_axis();
    const double center = std::midpoint(time(axis, seed.earlier_row), time(axis, seed.later_row));
    const double low = center - RtcJumpTransitionPolicy::search_seconds;
    const double high = center + RtcJumpTransitionPolicy::search_seconds;
    if (!std::isfinite(low) || !std::isfinite(high) || !std::isfinite(event.origin) ||
        !std::isfinite(event.time_scale) || event.time_scale <= 0 ||
        !std::isfinite(out.frozen_residual_scale) || out.frozen_residual_scale <= 0 ||
        !std::isfinite(RtcJumpTransitionPolicy::residual_sigma * out.frozen_residual_scale)) {
        out.cause = RtcJumpTransitionCause::nonfinite; return out;
    }
    RtcEventRange onset{seed.earlier_row, seed.later_row + 1};
    for (const auto &candidate : members) {
        if (candidate.earlier_row < onset.past_last && candidate.later_row >= onset.first) {
            onset.first = std::min(onset.first, candidate.earlier_row);
            onset.past_last = std::max(onset.past_last, candidate.later_row + 1);
        }
    }
    const auto first_edge = onset.first, last_edge = onset.past_last - 1;
    // Original member edge cells cannot establish a stable plateau in either
    // coordinate. Membership stays immutable; only their support role changes.
    std::vector<RtcEventRange> member_cells;
    for (const auto &candidate : members) {
        member_cells.push_back({candidate.earlier_row, candidate.later_row + 1});
        out.multiple_candidate_edges |= candidate.earlier_row != seed.earlier_row ||
            candidate.later_row != seed.later_row;
    }
    member_cells = merge(std::move(member_cells));
    const auto run_begin = axis.occurrence(run.first_native_row).integration_support.begin_unix_sec;
    const auto run_end = axis.occurrence(run.past_last_native_row - 1).integration_support.end_unix_sec;
    out.observation_truncated = (run_begin > low && run.first_native_row == axis.first_native_row()) ||
        (run_end < high && run.past_last_native_row == axis.past_last_native_row());
    out.acquisition_truncated = (run_begin > low && run.first_native_row != axis.first_native_row()) ||
        (run_end < high && run.past_last_native_row != axis.past_last_native_row());
    const RtcEventRange range{run.first_native_row, run.past_last_native_row};
    const auto begin = lower(axis, range, low), end = lower(axis, range, high);
    out.examined = {begin, begin};
    RtcJumpConfirmation current;
    bool nonfinite = false, geometry = false;
    double previous_end = -INFINITY;
    auto reset = [&] { current = {}; };
    for (auto row = begin; row < end; ++row) {
        out.examined.past_last = row + 1;
        const auto &cell = axis.occurrence(row).integration_support;
        // Complete integration support must lie inside the physical search.
        if (cell.begin_unix_sec < low || cell.end_unix_sec > high) { reset(); continue; }
        ++out.examined_rows;
        const double tolerance = 8 * std::numeric_limits<double>::epsilon() *
            std::max({1.0, std::abs(cell.begin_unix_sec), std::abs(previous_end)});
        if (std::isfinite(previous_end) && std::abs(cell.begin_unix_sec - previous_end) > tolerance) {
            geometry = true; reset();
        }
        previous_end = cell.end_unix_sec;
        if ((row >= first_edge && row <= last_edge) || contains(member_cells, row)) { reset(); continue; }
        if (contains(neighbors, row)) { ++out.excluded_rows; reset(); continue; }
        if (!net.state(coord(c), row, event.detector).valid()) { ++out.invalid_rows; reset(); continue; }
        const double y = net.value(coord(c), row, event.detector);
        const double pre = polynomial(fit.cubic_with_offset, (time(axis, row) - event.origin) / event.time_scale);
        const double post = pre + fit.cubic_with_offset.offset;
        if (!std::isfinite(y) || !std::isfinite(pre) || !std::isfinite(post) ||
            !std::isfinite(y - pre) || !std::isfinite(y - post)) {
            nonfinite = true; reset(); continue;
        }
        const std::size_t side = row < first_edge ? 0 : 1;
        if (!agrees(y, side == 0 ? pre : post, out.frozen_residual_scale)) { reset(); continue; }
        const bool other = agrees(y, side == 0 ? post : pre, out.frozen_residual_scale);
        if (!current.rows.present()) {
            current.rows = {row, row + 1}; current.physical = cell;
            current.also_matches_other_reference = other;
        } else {
            current.rows.past_last = row + 1;
            current.physical.end_unix_sec = cell.end_unix_sec;
            current.also_matches_other_reference &= other;
        }
        if (current.physical.duration_sec() >= RtcJumpTransitionPolicy::confirmation_seconds) {
            out.confirmations[side] = current;
            if (side == 1) break; // First post confirmation, last pre confirmation.
        }
    }
    const auto &pre = out.confirmations[0], &post = out.confirmations[1];
    if (nonfinite) out.cause = RtcJumpTransitionCause::nonfinite;
    else if (geometry) out.cause = RtcJumpTransitionCause::support_geometry_unavailable;
    else if (!pre.rows.present()) out.cause = RtcJumpTransitionCause::pre_confirmation_missing;
    else if (!post.rows.present()) out.cause = RtcJumpTransitionCause::post_confirmation_missing;
    else if (pre.also_matches_other_reference || post.also_matches_other_reference)
        out.cause = RtcJumpTransitionCause::ambiguous_reference;
    else {
        out.affected = {pre.rows.past_last, post.rows.first};
        out.physical_bound = {pre.physical.end_unix_sec, post.physical.begin_unix_sec};
        out.cause = RtcJumpTransitionCause::measured;
        if (!out.affected.present() || !(out.physical_bound.begin_unix_sec < out.physical_bound.end_unix_sec))
            out.cause = RtcJumpTransitionCause::support_geometry_unavailable;
        for (auto row = out.affected.first; row < out.affected.past_last; ++row) {
            const auto &cell = axis.occurrence(row).integration_support;
            if (!(cell.begin_unix_sec < out.physical_bound.end_unix_sec &&
                  cell.end_unix_sec > out.physical_bound.begin_unix_sec))
                out.cause = RtcJumpTransitionCause::support_geometry_unavailable;
            else if (contains(neighbors, row))
                out.cause = RtcJumpTransitionCause::competing_exclusion;
            else if (!net.state(coord(c), row, event.detector).valid())
                out.cause = RtcJumpTransitionCause::invalid_transition_support;
            else if (!std::isfinite(net.value(coord(c), row, event.detector)))
                out.cause = RtcJumpTransitionCause::nonfinite;
        }
        out.exceeds_fitting_exclusion = out.affected.first < event.trial_exclusion.first ||
            out.affected.past_last > event.trial_exclusion.past_last;
        // The first complete post plateau separates unresolved same-group
        // edges from later neighbors. Preserve every later member's existing
        // guard. If it reaches back into either confirmation or the bracket,
        // withhold this measurement; do not absorb that later disturbance or
        // search for another separator. This is not an extra reassessment.
        if (out.available()) for (const auto &candidate : members) {
            if (candidate.earlier_row < post.rows.past_last) continue;
            const auto guard = trial(axis, range, candidate);
            if (guard.first < post.rows.past_last && guard.past_last > pre.rows.first) {
                out.cause = RtcJumpTransitionCause::competing_exclusion;
                break;
            }
        }
    }
    return out;
}
inline RtcEventRecovery diagnostic_recover_at(const RtcSpikeEvidence &spikes,const RtcAssessedEvent &e,
                                RtcEventRange run,std::size_t c,
                                const RtcSpikeCandidate &seed,
                                std::span<const RtcSpikeCandidate> members) {
    using namespace rtc_event_assessment_detail;
    RtcEventRecovery r; const auto &fit=e.background[c];
    if(!fit.available()) return r;
    const auto &net=spikes.input_handle()->network(e.network);const auto &axis=net.occurrence_axis();
    // Every member in this coordinate must precede its recovery confirmation.
    // Keep the original event deadline fixed even when later edges join it.
    const RtcSpikeCandidate *first_seed=nullptr,*last_seed=nullptr;
    for(const auto &candidate:members) if(candidate.coordinate==coord(c)) {
        if(!first_seed) first_seed=&candidate;
        last_seed=&candidate;
    }
    if(!first_seed) first_seed=last_seed=&seed;
    const double center=std::midpoint(time(axis,seed.earlier_row),time(axis,seed.later_row));
    const double deadline=center+RtcEventAssessmentPolicy::search_seconds;
    const auto first=lower(axis,run,center-2),last=lower(axis,run,deadline);
    r.examined={first,last};
    auto good=[&](auto row) {
        if(!net.state(coord(c),row,e.detector).valid()) return false;
        const double y=net.value(coord(c),row,e.detector);
        // The additive post-side offset is deliberately NOT in this reference.
        const double predicted=polynomial(fit.cubic_with_offset,(time(axis,row)-e.origin)/e.time_scale);
        return std::isfinite(y) && std::isfinite(predicted) && std::abs(y-predicted)<=RtcEventAssessmentPolicy::recovery_sigma*fit.pre_scale_fit.scale;
    };
    TimestreamNativeRow quiet=-1,onset=-1;
    double support_end=-INFINITY;
    auto accumulate=[&](auto row) {
        const auto &s=axis.occurrence(row).integration_support;
        if(quiet<0 || s.begin_unix_sec>support_end+8*std::numeric_limits<double>::epsilon()*std::max(1.0,std::abs(s.begin_unix_sec))) quiet=row;
        support_end=std::max(support_end,s.end_unix_sec);
        return support_end-axis.occurrence(quiet).integration_support.begin_unix_sec>=RtcEventAssessmentPolicy::recovery_seconds;
    };
    for(auto row=first;row<first_seed->later_row;++row) {
        if(!good(row)) {quiet=-1;support_end=-INFINITY;continue;}
        if(accumulate(row)) onset=row+1;
    }
    if(onset<0) {r.cause=RtcEventRecoveryCause::onset_unavailable;return r;}
    quiet=-1;support_end=-INFINITY;bool invalid=false;
    bool departure=onset<first_seed->later_row;
    for(auto row=first_seed->later_row;row<last;++row) {
        if(row==last_seed->later_row) {quiet=-1;support_end=-INFINITY;}
        const auto &s=axis.occurrence(row).integration_support;
        if(s.end_unix_sec>deadline) break;
        if(!net.state(coord(c),row,e.detector).valid()) invalid=true;
        else if(!std::isfinite(net.value(coord(c),row,e.detector))) {r.cause=RtcEventRecoveryCause::nonfinite;return r;}
        if(!good(row)) {departure=true;quiet=-1;support_end=-INFINITY;continue;}
        if(accumulate(row) && row>=last_seed->later_row) {
            r.confirmation={quiet,row+1};
            // A seed can be large in only the other coordinate. No invented
            // affected cells in a coordinate that never left its background.
            r.affected=departure && quiet>onset ? RtcEventRange{onset,quiet} : RtcEventRange{};
            r.cause=RtcEventRecoveryCause::recovered;return r;
        }
    }
    r.affected={onset,last};
    if(invalid) r.cause=RtcEventRecoveryCause::invalid_support;
    else if(axis.occurrence(run.past_last-1).integration_support.end_unix_sec<deadline)
        r.cause=run.past_last==axis.past_last_native_row() ? RtcEventRecoveryCause::observation_end : RtcEventRecoveryCause::acquisition_gap;
    else r.cause=RtcEventRecoveryCause::search_limit;
    return r;
}

} // namespace
