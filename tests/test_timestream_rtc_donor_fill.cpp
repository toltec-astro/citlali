#include <citlali/core/pipeline/timestream_rtc_donor_fill.h>
#include "timestream_rtc_reassessment_test_support.h"
#include <gtest/gtest.h>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace {
using namespace citlali::pipeline;
using citlali::test::rtc_reassessment::Input;

struct Trial {
    std::shared_ptr<const NativePairedReadoutObservation> parent;
    std::shared_ptr<const ValSnapshot> val;
    std::shared_ptr<const RtcSpikeEvidence> spikes;
    std::shared_ptr<const RtcEventAssessmentEvidence> assessment;
    std::shared_ptr<const RtcTransientExclusionPlan> exclusions;
    explicit Trial(const Input &in, std::vector<RtcSpikeProtectionRegion> regions = {}, bool peer2 = true) {
        parent = in.freeze(); val = ValSnapshot::initial(parent);
        spikes = learn_rtc_spike_candidates(NativePairedReadoutView::full(parent), val,
            RtcSpikeSourceProtection::admit(parent, "fixture-source-authority", RtcSpikeProtection::outside_source, std::move(regions)), 1);
        std::vector<RtcEventPeerEligibility> peers;
        for (std::uint32_t d = 0; d < 3; ++d)
            peers.push_back({0, d, parent->network(0).detector(d).detector_occurrence_id, d != 2 || peer2});
        assessment = learn_rtc_event_assessment(spikes, RtcEventPeerPopulation::admit(spikes, "fixture-population", peers), 2);
        auto review = RtcEventAssessmentDecision::consider(assessment, val, 3);
        auto amplitude = RtcJumpAmplitudeDecision::consider(review, val, 4);
        auto consistency = RtcJumpConsistencyDecision::consider(RtcJumpConsistencyEvidence::learn(amplitude, 5), val, 6);
        auto transition = RtcJumpTransitionEvidence::learn(RtcJumpTransitionRequest::consider(consistency, val, 7), 8);
        auto audit = RtcJumpSupportEvidence::learn(transition, 9);
        auto refit = RtcJumpRefitEvidence::learn(RtcJumpRefitRequest::consider(audit, val, 10), 11);
        auto remeasurement = RtcJumpReassessmentEvidence::learn(RtcJumpRemeasureRequest::consider(refit, val, 12), 13);
        auto admission = RtcJumpAdmissionDecision::consider(RtcJumpReassessmentDecision::consider(remeasurement, val, 14), val, 15);
        auto scans = RtcExistingScanBinding::admit(parent, "fixture-scans", "fixture-native-relation", "fixture-support-bound",
            RtcExistingScanSupportState::conservative_native_support_bound, {{7, spikes->input_handle()->span(0)}});
        exclusions = RtcTransientExclusionPlan::consider(review->original_screening_handle(),
            RtcJumpExclusionPlan::consider(admission, scans, val, 16), val, 17);
    }
    auto records() const {
        std::vector<RtcDonorDetectorFacts> records;
        const auto &n = parent->network(0); const auto &a = n.occurrence_axis();
        for (std::uint32_t d = 0; d < 3; ++d)
            records.push_back({0, d, n.detector(d).detector_occurrence_id, "fixture-prior-factor-" + std::to_string(d),
                "fixture-compatible-flxscale", 1.0, {a.first_native_row(), a.past_last_native_row()},
                {{a.first_native_row(), a.past_last_native_row()}}, {}});
        return records;
    }
    auto facts(std::vector<RtcDonorDetectorFacts> r) const {
        return RtcDonorFillFacts::bind(assessment, "fixture-preexisting-static-APT", "fixture-compatible-flxscale",
            "fixture-resolved-segments", "fixture-complete-contamination", std::move(r));
    }
    RtcDonorSelectedEvent selection(RtcEventRange r = {600, 601}) const {
        for (std::size_t i = 0; i < assessment->events().size(); ++i)
            if (assessment->events()[i].detector == 0)
                return {assessment, "fixture-explicit-event-selection", RtcDonorSelectionState::accepted_isolated_event, i, r};
        throw std::runtime_error("fixture lacks target event");
    }
    auto plan(std::vector<RtcDonorDetectorFacts> r, RtcEventRange support = {600, 601}) const {
        return RtcDonorFillPlan::consider(selection(support), facts(std::move(r)), exclusions, val, 18);
    }
    auto apply(std::shared_ptr<const RtcDonorFillPlan> p) const {
        const std::array parts{spikes->input_handle()};
        return RtcDonorFillResult::apply(p, spikes->input_handle(), val, parts);
    }
};

TEST(rtc_donor_fill, single_sample_repair_is_finite_but_never_an_independent_measurement) {
    Input in; in.spike(); Trial t(in); auto p = t.plan(t.records()); auto out = t.apply(p);
    ASSERT_TRUE(out->filled()); ASSERT_EQ(p->medians().size(), 3);
    ASSERT_TRUE(out->value_for_conditioning(NativeReadoutCoordinate::x, 0, 600, 0));
    EXPECT_LT(std::abs(*out->value_for_conditioning(NativeReadoutCoordinate::x, 0, 600, 0) - 10), 0.3);
    EXPECT_TRUE(out->requires_map_exclusion(0, 600, 0));
    EXPECT_FALSE(out->value_if_independent(NativeReadoutCoordinate::x, 0, 600, 0));
    EXPECT_FALSE(out->value_for_conditioning(NativeReadoutCoordinate::r, 0, 600, 0));
    EXPECT_FALSE(out->value_if_independent(NativeReadoutCoordinate::r, 0, 600, 0));
    EXPECT_DOUBLE_EQ(t.parent->network(0).value(NativeReadoutCoordinate::x, 600, 0), in.x(500, 0));
    EXPECT_DOUBLE_EQ(t.parent->network(0).value(NativeReadoutCoordinate::r, 600, 0), in.r(500, 0));
    EXPECT_EQ(out->owned_numeric_bytes(), sizeof(double));
}

TEST(rtc_donor_fill, all_retained_coordinates_are_bitwise_unchanged_including_bright_sources) {
    std::size_t checked = 0;
    for (double brightness : {0., 3., 30., 300.}) {
        Input in; in.spike();
        for (std::size_t i = 0; i < in.times.size(); ++i) {
            const double u = (static_cast<double>(i) - 200) / 30;
            in.x(i, 2) += brightness * std::exp(-u * u / 2);
        }
        Trial t(in, {{0, 2, 100, 450, RtcSpikeProtection::protected_source}});
        auto out = t.apply(t.plan(t.records())); ASSERT_TRUE(out->filled());
        for (auto c : {NativeReadoutCoordinate::x, NativeReadoutCoordinate::r})
            for (std::uint32_t d = 0; d < 3; ++d) for (TimestreamNativeRow r = 100; r < 1200; ++r) {
                if (r == 600 && d == 0) continue;
                const auto v = out->value_for_conditioning(c, 0, r, d); ASSERT_TRUE(v);
                EXPECT_EQ(std::bit_cast<std::uint64_t>(*v), std::bit_cast<std::uint64_t>(t.parent->network(0).value(c, r, d)));
                ++checked;
            }
    }
    EXPECT_EQ(checked, 26392);
}

TEST(rtc_donor_fill, compatible_transfer_direction_and_even_median_are_exact) {
    Input in; in.spike(); Trial t(in); auto r = t.records();
    r[0].prior_flxscale = 2.; r[1].prior_flxscale = 3.; r[2].prior_flxscale = 5.;
    auto p = t.plan(r); ASSERT_EQ(p->cause(), RtcDonorFillCause::ready);
    for (const auto &m : p->medians()) {
        EXPECT_EQ(m.central_count, 2); EXPECT_EQ(m.eligible.size(), 2);
        EXPECT_DOUBLE_EQ(m.value, std::midpoint(1.5 * t.parent->network(0).value(NativeReadoutCoordinate::x, m.row, 1),
            2.5 * t.parent->network(0).value(NativeReadoutCoordinate::x, m.row, 2)));
    }
    EXPECT_EQ(p->facts_handle()->factor_authority(), "fixture-preexisting-static-APT");
}

TEST(rtc_donor_fill, one_eligible_donor_is_sufficient_and_reuse_remains_explicit) {
    Input in; in.spike(); Trial t(in, {}, false); auto p = t.plan(t.records());
    ASSERT_EQ(p->cause(), RtcDonorFillCause::ready);
    for (const auto &m : p->medians()) { EXPECT_EQ(m.eligible, std::vector<std::uint32_t>{1}); EXPECT_EQ(m.central[0], 1); }
    EXPECT_TRUE(t.apply(p)->filled());
}

TEST(rtc_donor_fill, donor_order_does_not_change_the_plan_or_output) {
    Input in; in.spike(); Trial t(in); auto a = t.records(), b = a; std::reverse(b.begin(), b.end());
    auto x = t.apply(t.plan(a)), y = t.apply(t.plan(b)); ASSERT_TRUE(x->filled()); ASSERT_TRUE(y->filled());
    EXPECT_EQ(x->value_for_conditioning(NativeReadoutCoordinate::x, 0, 600, 0), y->value_for_conditioning(NativeReadoutCoordinate::x, 0, 600, 0));
}

TEST(rtc_donor_fill, missing_invalid_or_incompatible_factors_never_use_a_fallback_value) {
    Input in; in.spike(); Trial t(in);
    for (int mode = 0; mode < 5; ++mode) {
        auto r = t.records();
        for (auto d : {1, 2}) {
            if (mode == 0) r[d].prior_flxscale.reset();
            if (mode == 1) r[d].prior_flxscale = NAN;
            if (mode == 2) r[d].factor_convention = "incompatible";
            if (mode == 3) r[d].factor_support = {100, 600};
            if (mode == 4) r[d].prior_flxscale = INFINITY;
        }
        auto out = t.apply(t.plan(r)); EXPECT_EQ(out->cause(), RtcDonorFillCause::no_usable_donor);
        EXPECT_TRUE(out->requires_map_exclusion(0, 600, 0));
        EXPECT_FALSE(out->value_for_conditioning(NativeReadoutCoordinate::x, 0, 600, 0));
    }
    auto r = t.records(); r[0].prior_flxscale = 0.;
    EXPECT_EQ(t.plan(r)->cause(), RtcDonorFillCause::target_transfer_unavailable);
}

TEST(rtc_donor_fill, protected_unknown_screening_failed_and_contaminated_donors_are_excluded) {
    for (auto s : {RtcSpikeProtection::protected_source, RtcSpikeProtection::unavailable}) {
        Input in; in.spike(); Trial t(in, {{0, 1, 590, 610, s}, {0, 2, 590, 610, s}});
        EXPECT_EQ(t.plan(t.records())->cause(), RtcDonorFillCause::no_usable_donor);
    }
    Input in; in.spike(); in.r.col(1).setConstant(-3.); Trial t(in);
    auto r = t.records(); r[2].contaminated = {{600, 601}};
    EXPECT_EQ(t.plan(r)->cause(), RtcDonorFillCause::no_usable_donor);
}

TEST(rtc_donor_fill, candidate_guard_excludes_a_donor_without_promoting_it_to_an_event) {
    Input in; in.spike(); in.x(500, 1) += 30; in.x(500, 2) += 30; Trial t(in);
    EXPECT_EQ(t.plan(t.records())->cause(), RtcDonorFillCause::no_usable_donor);
}

TEST(rtc_donor_fill, nonfinite_fill_arithmetic_leaves_the_entire_gap_unrepaired_and_excluded) {
    Input in; in.spike(501);
    for (std::size_t i = 0; i < in.times.size(); ++i)
        in.x(i, 1) = std::array{0., .9, -.9, .45, -.45}[i % 5];
    Trial t(in); auto r = t.records(); r[1].prior_flxscale = std::numeric_limits<double>::max();
    r[2].prior_flxscale.reset(); auto p = t.plan(r, {601, 602});
    ASSERT_EQ(p->cause(), RtcDonorFillCause::ready);
    auto out = t.apply(p); EXPECT_EQ(out->cause(), RtcDonorFillCause::arithmetic_nonfinite);
    EXPECT_FALSE(out->value_for_conditioning(NativeReadoutCoordinate::x, 0, 601, 0));
    EXPECT_TRUE(out->requires_map_exclusion(0, 601, 0)); EXPECT_EQ(out->owned_numeric_bytes(), 0);
}

TEST(rtc_donor_fill, producer_invalid_donor_sample_is_not_used) {
    Input in; in.spike();
    for (auto d : {1, 2}) {
        in.x(500, d) = NAN;
        in.xs[500 * 3 + d] = NativeReadoutCoordinateState::measured(true, false, true, false);
    }
    Trial t(in); EXPECT_EQ(t.plan(t.records())->cause(), RtcDonorFillCause::no_usable_donor);
}

TEST(rtc_donor_fill, boundary_or_segment_unavailability_does_not_create_a_bridge) {
    Input in; in.spike(); Trial t(in); auto r = t.records();
    r[0].stable_segments = {{100, 600}, {600, 1200}};
    EXPECT_EQ(t.plan(r)->cause(), RtcDonorFillCause::boundary_unavailable);
    r = t.records(); r[0].contaminated = {{599, 600}};
    EXPECT_EQ(t.plan(r)->cause(), RtcDonorFillCause::boundary_unavailable);
    r = t.records(); r[1].stable_segments.clear(); r[2].stable_segments.clear();
    EXPECT_EQ(t.plan(r)->cause(), RtcDonorFillCause::no_usable_donor);
    EXPECT_EQ(t.plan(t.records(), {100, 601})->cause(), RtcDonorFillCause::boundary_unavailable);
}

TEST(rtc_donor_fill, an_existing_jump_exclusion_is_not_repaired_or_unflagged) {
    Input in; in.step(); Trial t(in);
    EXPECT_EQ(t.plan(t.records())->cause(), RtcDonorFillCause::target_excluded);
    auto out = t.apply(t.plan(t.records())); EXPECT_FALSE(out->filled());
    EXPECT_TRUE(out->requires_map_exclusion(0, 800, 0));
}

TEST(rtc_donor_fill, physical_gap_cannot_be_hidden_in_a_resolved_segment) {
    Input in; in.spike(510);
    for (std::size_t i = 500; i < in.times.size(); ++i) { in.times[i] += 2.; in.counters[i] += 20; }
    Trial t(in); EXPECT_THROW(t.facts(t.records()), std::invalid_argument);
}

TEST(rtc_donor_fill, background_is_not_reused_for_support_it_was_fitted_through) {
    Input in; in.spike(); Trial t(in);
    EXPECT_EQ(t.plan(t.records(), {400, 800})->cause(), RtcDonorFillCause::background_unavailable);
}

TEST(rtc_donor_fill, newly_known_contamination_of_a_used_fit_sample_invalidates_background_reuse) {
    Input in; in.spike(); Trial t(in);
    const auto &e = t.assessment->events()[t.selection().event];
    for (const auto &side : e.background[0].support) {
        auto r = t.records(); r[0].contaminated = {{side.first_used, side.first_used + 1}};
        auto out = t.apply(t.plan(r)); EXPECT_EQ(out->cause(), RtcDonorFillCause::background_unavailable);
        EXPECT_FALSE(out->value_for_conditioning(NativeReadoutCoordinate::x, 0, 600, 0));
        EXPECT_TRUE(out->requires_map_exclusion(0, 600, 0));
    }
    auto r = t.records(); r[0].contaminated = {{600, 601}};
    // The selected spike was masked out of this fit originally.
    EXPECT_TRUE(t.apply(t.plan(r))->filled());
}

TEST(rtc_donor_fill, explicit_selection_is_required_and_protected_targets_are_not_admitted) {
    Input in; in.spike(); Trial t(in); auto s = t.selection(); s.state = RtcDonorSelectionState::unavailable;
    EXPECT_THROW(RtcDonorFillPlan::consider(s, t.facts(t.records()), t.exclusions, t.val, 18), std::invalid_argument);
    for (auto p : {RtcSpikeProtection::protected_source, RtcSpikeProtection::unavailable}) {
        Trial guarded(in, {{0, 0, 100, 1200, p}});
        EXPECT_THROW(guarded.plan(guarded.records()), std::invalid_argument);
    }
}

TEST(rtc_donor_fill, stale_generation_and_wrong_occurrence_or_support_are_rejected) {
    Input in; in.spike(); Trial t(in), other(in); auto p = t.plan(t.records()); const std::array parts{t.spikes->input_handle()};
    EXPECT_THROW(RtcDonorFillPlan::consider(t.selection(), other.facts(other.records()), t.exclusions, t.val, 18), std::invalid_argument);
    EXPECT_THROW(RtcDonorFillPlan::consider(t.selection(), t.facts(t.records()), t.exclusions, other.val, 18), StaleRtcValGeneration);
    EXPECT_THROW(RtcDonorFillResult::apply(p, other.spikes->input_handle(), t.val, parts), std::invalid_argument);
    EXPECT_THROW(RtcDonorFillResult::apply(p, t.spikes->input_handle(), other.val, parts), StaleRtcValGeneration);
    auto r = t.records(); r[1].detector_occurrence_id = "different-occurrence"; EXPECT_THROW(t.facts(r), std::invalid_argument);
    r = t.records(); r.push_back(r[1]); EXPECT_THROW(t.facts(r), std::invalid_argument);
    r = t.records(); r[1].stable_segments = {{100, 700}, {600, 1200}}; EXPECT_THROW(t.facts(r), std::invalid_argument);
    EXPECT_THROW(t.plan(t.records(), {700, 701}), std::invalid_argument);
}

TEST(rtc_donor_fill, partition_schedule_cannot_change_or_repeat_the_frozen_fill) {
    Input in; in.spike(); Trial t(in); auto p = t.plan(t.records(), {596, 605}); auto a = t.apply(p);
    std::vector<std::shared_ptr<const NativePairedReadoutView>> parts{
        NativePairedReadoutView::admit(t.parent, {{0, 100, 600}}), NativePairedReadoutView::admit(t.parent, {{0, 600, 1200}})};
    auto b = RtcDonorFillResult::apply(p, t.spikes->input_handle(), t.val, parts);
    ASSERT_TRUE(a->filled()); ASSERT_TRUE(b->filled());
    for (auto r = 590; r < 610; ++r)
        EXPECT_EQ(a->value_for_conditioning(NativeReadoutCoordinate::x, 0, r, 0), b->value_for_conditioning(NativeReadoutCoordinate::x, 0, r, 0));
    std::reverse(parts.begin(), parts.end());
    EXPECT_THROW(RtcDonorFillResult::apply(p, t.spikes->input_handle(), t.val, parts), std::invalid_argument);
}

TEST(rtc_donor_fill, taper_matches_background_value_and_slope_without_length_tuning) {
    EXPECT_DOUBLE_EQ(RtcDonorFillPolicy::taper(0), 0); EXPECT_DOUBLE_EQ(RtcDonorFillPolicy::taper(1), 0);
    EXPECT_DOUBLE_EQ(RtcDonorFillPolicy::taper(.5), 1);
    for (double h : {1e-4, 1e-5, 1e-6}) {
        EXPECT_LT(RtcDonorFillPolicy::taper(h) / h, 17 * h);
        EXPECT_LT(RtcDonorFillPolicy::taper(1 - h) / h, 17 * h);
    }
    EXPECT_THROW(RtcDonorFillPolicy::taper(NAN), std::invalid_argument);
    EXPECT_THROW(RtcDonorFillPolicy::taper(1.1), std::invalid_argument);
}

TEST(rtc_donor_fill, review_fixture_keeps_curve_and_reports_exact_fill_support) {
    Input in; in.spike(); Trial t(in); auto p = t.plan(t.records(), {595, 606}); auto out = t.apply(p);
    ASSERT_TRUE(out->filled());
    const auto &e = p->event(); const auto &axis = t.parent->network(0).occurrence_axis();
    std::ofstream csv;
    if (const char *path = std::getenv("CITLALI_DONOR_FILL_REVIEW_CSV")) {
        csv.open(path); ASSERT_TRUE(csv); csv << std::setprecision(17);
        csv << "time_seconds,original_x,background_x,conditioned_x,map_excluded\n";
    }
    for (TimestreamNativeRow r = 550; r < 651; ++r) {
        const double time = rtc_event_assessment_detail::time(axis, r);
        const double u = (time - e.origin) / e.time_scale;
        const auto c = e.background[0].cubic.coefficients;
        const double background = c[0] + c[1]*u + c[2]*u*u + c[3]*u*u*u;
        const auto value = out->value_for_conditioning(NativeReadoutCoordinate::x, 0, r, 0); ASSERT_TRUE(value);
        const bool selected = r >= 595 && r < 606;
        EXPECT_EQ(out->requires_map_exclusion(0, r, 0), selected);
        if (selected) EXPECT_LT(std::abs(*value - background), .3);
        if (csv.is_open()) csv << time - in.times[500] << ',' << in.x(r-100, 0) << ',' << background << ',' << *value << ',' << selected << '\n';
    }
}

TEST(rtc_donor_fill, large_donor_offset_is_removed_and_filter_fixture_has_no_offset_impulse) {
    Input a; a.spike(); Input b = a; b.x.col(1).array() += 1e6; b.x.col(2).array() += 1e6;
    Trial x(a), y(b); auto p = x.plan(x.records(), {595, 606}), q = y.plan(y.records(), {595, 606});
    auto u = x.apply(p), v = y.apply(q); ASSERT_TRUE(u->filled()); ASSERT_TRUE(v->filled());
    double maximum_filtered_difference = 0;
    // Fixed [1,2,1]/4 convolution is a numerical support/edge fixture, not a
    // newly selected RTC filter or a production ringing qualification.
    for (auto r = 592; r < 609; ++r) {
        double uf = 0, vf = 0;
        for (int k = -1; k <= 1; ++k) {
            const double w = k == 0 ? .5 : .25;
            uf += w * *u->value_for_conditioning(NativeReadoutCoordinate::x, 0, r + k, 0);
            vf += w * *v->value_for_conditioning(NativeReadoutCoordinate::x, 0, r + k, 0);
        }
        maximum_filtered_difference = std::max(maximum_filtered_difference, std::abs(uf - vf));
    }
    EXPECT_LT(maximum_filtered_difference, 1e-8);
    for (auto r = 595; r < 606; ++r) EXPECT_TRUE(v->requires_map_exclusion(0, r, 0));
    EXPECT_EQ(q->donor_support(), (RtcEventRange{594, 607}));
}

TEST(rtc_donor_fill, bounded_plan_and_apply_timing_with_sparse_numeric_ownership) {
    Input in; in.spike(); Trial t(in); auto facts = t.facts(t.records()); const std::array parts{t.spikes->input_handle()};
    const auto start = std::chrono::steady_clock::now(); std::size_t bytes = 0;
    for (std::uint64_t i = 0; i < 1000; ++i) {
        auto p = RtcDonorFillPlan::consider(t.selection({595, 606}), facts, t.exclusions, t.val, i + 100);
        auto out = RtcDonorFillResult::apply(p, t.spikes->input_handle(), t.val, parts);
        ASSERT_TRUE(out->filled()); EXPECT_EQ(out->owned_numeric_bytes(), 11 * sizeof(double)); bytes = p->logical_owned_bytes();
    }
    std::cout << "donor_plan_apply_1000_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
              << " plan_logical_bytes=" << bytes << " output_numeric_bytes=88\n";
}
} // namespace
