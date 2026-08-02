#include <citlali/core/pipeline/sci_align_contract.h>
#include <citlali/core/pipeline/sci_align_field_registry.h>
#include <citlali/core/pipeline/sci_align_scan_contract.h>
#include <citlali/core/pipeline/timestream_alignment_state.h>

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace {

namespace align = citlali::pipeline::sci_align;

Eigen::VectorXd vector(std::initializer_list<double> values) {
    Eigen::VectorXd result(static_cast<Eigen::Index>(values.size()));
    Eigen::Index i = 0;
    for (const double value : values) {
        result[i++] = value;
    }
    return result;
}

align::NativeTimingHeader timing_header(align::NativeRateFactor factor) {
    const double multiplier = align::rate_multiplier(factor);
    return {256000000.0,
            2097152.0 / multiplier,
            align::reference_sample_frequency_hz * multiplier};
}

align::ClockCoordinates reference_time(
    std::initializer_list<double> values) {
    return {vector(values), align::ClockCoordinateStage::reference_clock};
}

align::CellQuality quality(align::Origin origin, align::Validity validity,
                           align::Method method = align::Method::none) {
    return {origin, validity, method, align::Reason::none, {}};
}

TEST(sci_align_checked_arithmetic,
     rejects_count_product_and_rounding_overflow_before_allocation) {
    EXPECT_EQ(citlali::pipeline::checked_alignment_count_add(7, 9,
                                                             "fixture"),
              16u);
    EXPECT_THROW(citlali::pipeline::checked_alignment_count_add(
                     std::numeric_limits<std::uint64_t>::max(), 1,
                     "fixture"),
                 std::overflow_error);
    EXPECT_THROW(
        citlali::pipeline::checked_alignment_interface_slot_capacity(
            std::numeric_limits<std::uint64_t>::max(), 2),
        std::overflow_error);
    EXPECT_THROW(citlali::pipeline::checked_alignment_size_product(
                     std::numeric_limits<std::size_t>::max(), 2,
                     "fixture"),
                 std::overflow_error);

    const double exclusive_index_upper =
        -static_cast<double>(std::numeric_limits<Eigen::Index>::min());
    EXPECT_THROW(align::round_half_up_positive(exclusive_index_upper,
                                               "fixture"),
                 std::runtime_error);
}

TEST(sci_align_rate, validates_native_half_one_two_four_algebra) {
    const align::NativeRateFactor factors[] = {
        align::NativeRateFactor::half, align::NativeRateFactor::one,
        align::NativeRateFactor::two, align::NativeRateFactor::four};
    const std::uint64_t expected_accumulation[] = {
        4194304, 2097152, 1048576, 524288};
    const double expected_cadence[] = {
        0.016384, 0.008192, 0.004096, 0.002048};

    for (std::size_t i = 0; i < 4; ++i) {
        const auto validated =
            align::validate_native_timing_header(timing_header(factors[i]));
        EXPECT_EQ(validated.factor, factors[i]);
        EXPECT_EQ(validated.accumulation_length_ticks,
                  expected_accumulation[i]);
        EXPECT_DOUBLE_EQ(validated.cadence_seconds, expected_cadence[i]);
        EXPECT_DOUBLE_EQ(validated.exclusive_half_cell_seconds,
                         expected_cadence[i] / 2.0);
        EXPECT_DOUBLE_EQ(validated.sample_frequency_hz,
                         align::reference_sample_frequency_hz *
                             align::rate_multiplier(factors[i]));
    }

    EXPECT_NO_THROW(
        align::validate_bounded_production_native_timing_header(
            timing_header(align::NativeRateFactor::one)));
    EXPECT_THROW(
        align::validate_bounded_production_native_timing_header(
            timing_header(align::NativeRateFactor::half)),
        std::invalid_argument);
    EXPECT_THROW(
        align::validate_bounded_production_native_timing_header(
            timing_header(align::NativeRateFactor::two)),
        std::invalid_argument);
    EXPECT_THROW(
        align::validate_bounded_production_native_timing_header(
            timing_header(align::NativeRateFactor::four)),
        std::invalid_argument);
}

TEST(sci_align_rate, rejects_missing_inconsistent_mixed_and_arbitrary_rates) {
    auto inconsistent = timing_header(align::NativeRateFactor::one);
    inconsistent.accumulation_length_ticks += 1.0;
    EXPECT_THROW(align::validate_native_timing_header(inconsistent),
                 std::invalid_argument);

    auto arbitrary = timing_header(align::NativeRateFactor::one);
    arbitrary.fpga_frequency_hz = 256000000.0 * 1.25;
    arbitrary.sample_frequency_hz *= 1.25;
    EXPECT_THROW(align::validate_native_timing_header(arbitrary),
                 std::invalid_argument);

    auto missing = timing_header(align::NativeRateFactor::one);
    missing.sample_frequency_hz =
        std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(align::validate_native_timing_header(missing),
                 std::invalid_argument);

    EXPECT_THROW(
        align::require_common_native_rate(
            {timing_header(align::NativeRateFactor::one),
             timing_header(align::NativeRateFactor::two)}),
        std::invalid_argument);

    align::NativeTimingHeader out_of_range_accumulation{
        std::ldexp(1.0, 64) * align::reference_sample_frequency_hz,
        std::ldexp(1.0, 64), align::reference_sample_frequency_hz};
    EXPECT_THROW(
        align::validate_native_timing_header(out_of_range_accumulation),
        std::invalid_argument);
}

TEST(sci_align_legacy_timestamp,
     preserves_named_arithmetic_and_rejects_malformed_sequences) {
    Eigen::MatrixXd fields = Eigen::MatrixXd::Zero(3, 6);
    fields.col(0).setConstant(1000.0);
    fields.col(5).setConstant(600000000.0);
    fields.col(1) << 10.0, 11.0, 12.0;
    fields.col(2) << 100.0, 200.0, 300.0;
    fields.col(4).setZero();

    const auto result = align::reconstruct_legacy_detector_timestamps(
        fields, {7, 9, 10}, 256000000.0);
    ASSERT_EQ(result.seconds.size(), 3);
    EXPECT_DOUBLE_EQ(result.seconds[0], 1010.0 + 100.0 / 256000000.0);
    EXPECT_DOUBLE_EQ(result.seconds[1], 1011.0 + 200.0 / 256000000.0);
    ASSERT_EQ(result.packet_gaps.size(), 1u);
    EXPECT_EQ(result.packet_gaps[0].row_before, 0);
    EXPECT_EQ(result.packet_gaps[0].first_missing_packet, 8);
    EXPECT_EQ(result.packet_gaps[0].missing_packet_count, 1u);
    EXPECT_FALSE(result.producer_clock_authority_available);
    EXPECT_FALSE(result.integration_event_authority_available);
    EXPECT_FALSE(result.absolute_timing_precision_available);

    Eigen::MatrixXd wrapped = Eigen::MatrixXd::Zero(1, 6);
    wrapped(0, 0) = 1000.0;
    wrapped(0, 5) = 600000000.0;
    wrapped(0, 1) = 10.0;
    wrapped(0, 2) = 5.0;
    wrapped(0, 4) = 10.0;
    const auto rollover = align::reconstruct_legacy_detector_timestamps(
        wrapped, {1}, 256000000.0);
    EXPECT_DOUBLE_EQ(
        rollover.seconds[0],
        1010.0 + (5.0 - 10.0 + align::legacy_counter_adjustment_ticks) /
                     256000000.0);

    EXPECT_THROW(align::reconstruct_legacy_detector_timestamps(
                     fields, {7, 7, 10}, 256000000.0),
                 std::invalid_argument);
    fields(1, 1) = 9.0;
    EXPECT_THROW(align::reconstruct_legacy_detector_timestamps(
                     fields, {7, 8, 9}, 256000000.0),
                 std::invalid_argument);
}

TEST(sci_align_legacy_timestamp,
     packet_gap_distance_is_exact_across_the_full_signed_domain) {
    Eigen::MatrixXd fields = Eigen::MatrixXd::Zero(2, 6);
    fields.col(0).setConstant(1000.0);
    fields.col(5).setConstant(600000000.0);
    fields.col(1) << 10.0, 11.0;
    fields.col(2) << 100.0, 200.0;

    const auto result = align::reconstruct_legacy_detector_timestamps(
        fields,
        {std::numeric_limits<std::int64_t>::min(),
         std::numeric_limits<std::int64_t>::max()},
        256000000.0);
    ASSERT_EQ(result.packet_gaps.size(), 1u);
    EXPECT_EQ(result.packet_gaps[0].row_before, 0);
    EXPECT_EQ(result.packet_gaps[0].first_missing_packet,
              std::numeric_limits<std::int64_t>::min() + 1);
    EXPECT_EQ(result.packet_gaps[0].missing_packet_count,
              std::numeric_limits<std::uint64_t>::max() - 1);
}

// T01: the sign is positive-add, the offset is not sample-rounded, and the
// interpolation sources remain available on request.
TEST(sci_align_T01, applies_subsample_offset_once_with_positive_add_sign) {
    align::ClockCoordinates native{
        vector({0.0, 1.0, 2.0}),
        align::ClockCoordinateStage::native_legacy};
    const align::InterfaceOffset positive{
        0.25, true, "fixture", "detector_reference", "s", "positive_add"};
    const auto corrected =
        align::apply_interface_offset_once(native, positive);
    EXPECT_EQ(corrected.stage, align::ClockCoordinateStage::reference_clock);
    EXPECT_DOUBLE_EQ(corrected.seconds[0], 0.25);
    EXPECT_DOUBLE_EQ(corrected.seconds[1], 1.25);

    const Eigen::VectorXd values = vector({0.0, 1.0, 2.0});
    const align::FieldContract scalar{
        align::FieldTopology::continuous_scalar, 2.0, 0.0, std::nullopt};
    const auto aligned = align::align_field_at(
        corrected.seconds, values, 1.0, scalar, align::DetailLevel::expanded);
    EXPECT_DOUBLE_EQ(aligned.value, 0.75);
    EXPECT_EQ(aligned.quality.origin, align::Origin::synthesized);
    EXPECT_EQ(aligned.quality.method, align::Method::linear);
    ASSERT_EQ(aligned.quality.expanded_sources.size(), 2u);
    EXPECT_EQ(aligned.quality.expanded_sources[0].source_row, 0);
    EXPECT_DOUBLE_EQ(aligned.quality.expanded_sources[0].weight, 0.25);
    EXPECT_EQ(aligned.quality.expanded_sources[1].source_row, 1);
    EXPECT_DOUBLE_EQ(aligned.quality.expanded_sources[1].weight, 0.75);

    EXPECT_THROW(align::apply_interface_offset_once(corrected, positive),
                 std::invalid_argument);
    align::InterfaceOffset unauthoritative = positive;
    unauthoritative.authority_resolved = false;
    EXPECT_THROW(align::apply_interface_offset_once(native, unauthoritative),
                 std::invalid_argument);
}

// T02: exact rows and endpoints are one-hot originals; extrapolation is typed
// unavailable.
TEST(sci_align_T02, exact_coincidence_is_one_hot_and_edges_do_not_extrapolate) {
    const Eigen::VectorXd times = vector({0.0, 1.0, 2.0});
    const Eigen::VectorXd values = vector({4.0, 5.0, 6.0});
    const align::FieldContract scalar{
        align::FieldTopology::continuous_scalar, 2.0, 0.0, std::nullopt};

    for (Eigen::Index row = 0; row < 3; ++row) {
        const auto exact = align::align_field_at(
            times, values, times[row], scalar, align::DetailLevel::expanded);
        EXPECT_DOUBLE_EQ(exact.value, values[row]);
        EXPECT_EQ(exact.quality.origin, align::Origin::original);
        EXPECT_EQ(exact.quality.validity, align::Validity::valid);
        EXPECT_EQ(exact.quality.method, align::Method::exact);
        ASSERT_EQ(exact.quality.expanded_sources.size(), 1u);
        EXPECT_EQ(exact.quality.expanded_sources[0].source_row, row);
        EXPECT_DOUBLE_EQ(exact.quality.expanded_sources[0].weight, 1.0);
    }

    const auto before = align::align_field_at(times, values, -0.1, scalar);
    const auto after = align::align_field_at(times, values, 2.1, scalar);
    EXPECT_EQ(before.quality.origin, align::Origin::unavailable);
    EXPECT_EQ(after.quality.origin, align::Origin::unavailable);
    EXPECT_EQ(before.quality.reason, align::Reason::outside_support);
    EXPECT_EQ(after.quality.reason, align::Reason::outside_support);
    EXPECT_TRUE(std::isnan(before.value));
    EXPECT_TRUE(std::isnan(after.value));
}

// T03: linear interpolation and the gap operator preserve DC and affine data.
TEST(sci_align_T03, continuous_scalar_preserves_constant_and_affine_signals) {
    const Eigen::VectorXd times = vector({0.0, 1.0, 2.0});
    const Eigen::VectorXd constant = vector({7.0, 7.0, 7.0});
    const Eigen::VectorXd ramp = vector({2.0, 5.0, 8.0});
    const align::FieldContract scalar{
        align::FieldTopology::continuous_scalar, 1.0, 0.0, std::nullopt};

    for (const double target : {0.25, 0.5, 1.25, 1.75}) {
        EXPECT_DOUBLE_EQ(
            align::align_field_at(times, constant, target, scalar).value,
            7.0);
        EXPECT_NEAR(align::align_field_at(times, ramp, target, scalar).value,
                    2.0 + 3.0 * target, 1.0e-14);
    }

    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(3, 3);
    const Eigen::VectorXd ones = Eigen::VectorXd::Ones(3);
    EXPECT_TRUE(align::conditional_response_small(identity, ones)
                    .isApprox(ones, 0.0));
}

// T04: the non-DC transfer function is the declared fractional-delay response,
// including near Nyquist; it is not replaced by a unity claim.
TEST(sci_align_T04, sinusoidal_response_matches_fractional_linear_operator) {
    constexpr double cadence = 0.008192;
    for (const double alpha : {0.1, 0.5, 0.9}) {
        for (const double normalized_frequency : {0.0, 0.2, 0.49}) {
            const double omega =
                normalized_frequency * 2.0 * std::acos(-1.0) / cadence;
            const std::complex<double> expected =
                std::exp(std::complex<double>{0.0, -omega * cadence * alpha}) *
                ((1.0 - alpha) +
                 alpha * std::exp(
                             std::complex<double>{0.0, omega * cadence}));
            const auto actual = align::fractional_linear_response(
                alpha, omega, cadence);
            EXPECT_NEAR(actual.real(), expected.real(), 1.0e-14);
            EXPECT_NEAR(actual.imag(), expected.imag(), 1.0e-14);
            if (normalized_frequency == 0.0) {
                EXPECT_DOUBLE_EQ(std::abs(actual), 1.0);
            } else if (alpha != 0.0 && alpha != 1.0) {
                EXPECT_LT(std::abs(actual), 1.0);
            }
        }
    }
}

// T05: interpolation follows the declared circular topology and refuses an
// unresolved antipodal bracket.
TEST(sci_align_T05, circular_interpolation_wraps_and_rejects_antipodes) {
    const Eigen::VectorXd times = vector({0.0, 1.0});
    const align::FieldContract degrees{
        align::FieldTopology::circular, 1.0, 360.0, std::nullopt};
    const Eigen::VectorXd across_zero = vector({359.0, 1.0});
    const auto midpoint =
        align::align_field_at(times, across_zero, 0.5, degrees);
    EXPECT_DOUBLE_EQ(midpoint.value, 0.0);
    EXPECT_EQ(midpoint.quality.method, align::Method::circular);

    const Eigen::VectorXd negative_branch = vector({-2.25, -2.20});
    EXPECT_DOUBLE_EQ(
        align::align_field_at(times, negative_branch, 0.5, degrees).value,
        -2.225);
    const Eigen::VectorXd above_period = vector({361.0, 363.0});
    EXPECT_DOUBLE_EQ(
        align::align_field_at(times, above_period, 0.5, degrees).value,
        362.0);

    const Eigen::VectorXd antipodal = vector({0.0, 180.0});
    const auto ambiguous =
        align::align_field_at(times, antipodal, 0.5, degrees);
    EXPECT_EQ(ambiguous.quality.origin, align::Origin::unavailable);
    EXPECT_EQ(ambiguous.quality.reason,
              align::Reason::antipodal_ambiguous);
    EXPECT_TRUE(std::isnan(ambiguous.value));
}

// T06: categorical data are exact-only, while an explicitly declared step
// process uses producer-bounded half-open hold intervals.
TEST(sci_align_T06, exact_only_rejects_linear_and_step_is_half_open) {
    const Eigen::VectorXd times = vector({0.0, 1.0, 2.0});
    const Eigen::VectorXd states = vector({10.0, 20.0, 30.0});
    const align::FieldContract exact_only{
        align::FieldTopology::exact_only, 0.0, 0.0, std::nullopt};
    const auto rejected =
        align::align_field_at(times, states, 0.5, exact_only);
    EXPECT_EQ(rejected.quality.reason,
              align::Reason::operator_not_permitted);

    const align::FieldContract step{
        align::FieldTopology::declared_half_open_step, 1.0, 0.0, 3.0};
    const auto held = align::align_field_at(
        times, states, 0.999, step, align::DetailLevel::expanded);
    EXPECT_DOUBLE_EQ(held.value, 10.0);
    EXPECT_EQ(held.quality.origin, align::Origin::synthesized);
    EXPECT_EQ(held.quality.method, align::Method::held);
    ASSERT_EQ(held.quality.expanded_sources.size(), 1u);
    EXPECT_EQ(held.quality.expanded_sources[0].source_row, 0);

    const auto boundary = align::align_field_at(times, states, 1.0, step);
    EXPECT_DOUBLE_EQ(boundary.value, 20.0);
    EXPECT_EQ(boundary.quality.origin, align::Origin::original);
    const auto final_stop = align::align_field_at(times, states, 3.0, step);
    EXPECT_EQ(final_stop.quality.origin, align::Origin::unavailable);
}

// T07: a missing run has one compact identity and its values/source weights
// are generated only when requested.
TEST(sci_align_T07, bounded_internal_gaps_fill_affine_data_generative_weights) {
    for (const std::size_t gap_size : {1u, 2u, 3u}) {
        std::vector<align::DetectorSlotCell> cells(gap_size + 2);
        cells.front() = {true, 2.0, true, 40, 1.0};
        cells.back() = {true, 2.0 + 3.0 * (gap_size + 1), true, 41, 1.0};
        const auto plan = align::plan_detector_gaps(
            cells, 1.0, align::GapLimits{3, 3.0});
        ASSERT_EQ(plan.runs.size(), 1u);
        EXPECT_EQ(plan.runs[0].begin, 1u);
        EXPECT_EQ(plan.runs[0].end, gap_size + 1);
        EXPECT_EQ(plan.runs[0].action, align::GapAction::linear_fill);
        EXPECT_EQ(plan.synthesized_slot_count, gap_size);
        EXPECT_EQ(plan.unavailable_slot_count, 0u);

        for (std::size_t slot = 1; slot <= gap_size; ++slot) {
            const auto filled = align::detector_slot_value_at(
                cells, plan, slot, align::DetailLevel::expanded);
            EXPECT_DOUBLE_EQ(filled.value, 2.0 + 3.0 * slot);
            EXPECT_EQ(filled.quality.origin, align::Origin::synthesized);
            ASSERT_EQ(filled.quality.expanded_sources.size(), 2u);
            const double lambda =
                static_cast<double>(slot) /
                static_cast<double>(gap_size + 1);
            EXPECT_DOUBLE_EQ(filled.quality.expanded_sources[0].weight,
                             1.0 - lambda);
            EXPECT_DOUBLE_EQ(filled.quality.expanded_sources[1].weight,
                             lambda);
        }
    }

    std::vector<align::DetectorSlotCell> constant(4);
    constant.front() = {true, 8.0, true, 0, 1.0};
    constant.back() = {true, 8.0, true, 3, 1.0};
    const auto plan = align::plan_detector_gaps(
        constant, 1.0, align::GapLimits{2, 2.0});
    EXPECT_DOUBLE_EQ(align::detector_slot_value_at(constant, plan, 1).value,
                     8.0);
    EXPECT_DOUBLE_EQ(align::detector_slot_value_at(constant, plan, 2).value,
                     8.0);
}

// T08: edge, long, and invalid-endpoint runs stay unavailable, while an
// acquired non-finite row remains an original invalid row rather than a gap.
TEST(sci_align_T08, forbidden_gaps_are_unavailable_and_native_invalid_is_original) {
    std::vector<align::DetectorSlotCell> edge_cells(5);
    edge_cells[1] = {true, 1.0, true, 1, 1.0};
    edge_cells[2] = {true, 2.0, true, 2, 1.0};
    const auto edge_plan = align::plan_detector_gaps(
        edge_cells, 1.0, align::GapLimits{3, 3.0});
    ASSERT_EQ(edge_plan.runs.size(), 2u);
    EXPECT_EQ(edge_plan.runs[0].location, align::GapLocation::leading);
    EXPECT_EQ(edge_plan.runs[0].reason, align::Reason::leading_gap);
    EXPECT_EQ(edge_plan.runs[1].location, align::GapLocation::trailing);
    EXPECT_EQ(edge_plan.runs[1].reason, align::Reason::trailing_gap);
    EXPECT_EQ(align::detector_slot_value_at(edge_cells, edge_plan, 0)
                  .quality.origin,
              align::Origin::unavailable);

    std::vector<align::DetectorSlotCell> too_long(5);
    too_long.front() = {true, 0.0, true, 0, 1.0};
    too_long.back() = {true, 4.0, true, 4, 1.0};
    const auto long_plan = align::plan_detector_gaps(
        too_long, 1.0, align::GapLimits{2, 100.0});
    EXPECT_EQ(long_plan.runs[0].reason,
              align::Reason::gap_limit_exceeded);
    EXPECT_EQ(long_plan.synthesized_slot_count, 0u);
    EXPECT_EQ(long_plan.unavailable_slot_count, 3u);

    std::vector<align::DetectorSlotCell> invalid_endpoint(3);
    invalid_endpoint[0] = {true,
                           std::numeric_limits<double>::quiet_NaN(),
                           false, 0, 1.0};
    invalid_endpoint[2] = {true, 2.0, true, 2, 1.0};
    const auto invalid_plan = align::plan_detector_gaps(
        invalid_endpoint, 1.0, align::GapLimits{1, 1.0});
    EXPECT_EQ(invalid_plan.runs[0].reason,
              align::Reason::invalid_gap_endpoint);
    EXPECT_EQ(align::detector_slot_value_at(invalid_endpoint, invalid_plan, 0)
                  .quality.origin,
              align::Origin::original);
    EXPECT_EQ(align::detector_slot_value_at(invalid_endpoint, invalid_plan, 0)
                  .quality.validity,
              align::Validity::invalid);
    EXPECT_EQ(invalid_plan.missing_slot_count, 1u);
}

// T09: one operator governs placement, residual identity, strict half-cell
// admission, collisions, and ordering.
TEST(sci_align_T09, union_lattice_retains_jitter_and_rejects_ambiguity) {
    const auto one = timing_header(align::NativeRateFactor::one);
    const double dt = 0.008192;
    const align::DetectorInterfaceCoordinates early{
        "toltec0", one, reference_time({-dt, 0.0, dt, 2.0 * dt})};
    const align::DetectorInterfaceCoordinates late{
        "toltec1", one, reference_time({0.0, dt, 2.0 * dt, 3.0 * dt})};
    const auto union_lattice =
        align::build_detector_union_lattice({early, late});
    EXPECT_DOUBLE_EQ(union_lattice.phase_seconds, 0.0);
    EXPECT_EQ(union_lattice.first_global_slot, -1);
    EXPECT_EQ(union_lattice.last_global_slot, 3);
    EXPECT_EQ(union_lattice.slot_count(), 5u);
    ASSERT_TRUE(union_lattice.interfaces[0].trailing_unavailable.has_value());
    EXPECT_EQ(union_lattice.interfaces[0].trailing_unavailable->begin, 3);
    ASSERT_TRUE(union_lattice.interfaces[1].leading_unavailable.has_value());
    EXPECT_EQ(union_lattice.interfaces[1].leading_unavailable->begin, -1);
    EXPECT_EQ(union_lattice.interfaces[1].leading_unavailable->end, 0);

    const align::DetectorInterfaceCoordinates jittered{
        "toltec0", one,
        reference_time({0.0, dt + 1.0e-6, 2.0 * dt - 1.0e-6})};
    const auto jitter_lattice =
        align::build_detector_union_lattice({jittered});
    EXPECT_EQ(jitter_lattice.interfaces[0].assignments[1].global_slot, 1);
    EXPECT_NEAR(jitter_lattice.interfaces[0].assignments[1].residual_seconds,
                1.0e-6, 1.0e-18);
    EXPECT_EQ(jitter_lattice.interfaces[0].assignments[2].global_slot, 2);
    EXPECT_NEAR(jitter_lattice.interfaces[0].assignments[2].residual_seconds,
                -1.0e-6, 2.0e-18);

    const align::DetectorInterfaceCoordinates half_tie{
        "toltec0", one, reference_time({0.0, 1.5 * dt})};
    EXPECT_THROW(align::build_detector_union_lattice({half_tie}),
                 std::invalid_argument);

    const align::DetectorInterfaceCoordinates collision{
        "toltec0", one, reference_time({0.0, 0.1 * dt, dt})};
    EXPECT_THROW(align::build_detector_union_lattice({collision}),
                 std::invalid_argument);

    const align::DetectorInterfaceCoordinates nonmonotonic{
        "toltec0", one, reference_time({0.0, dt, 0.5 * dt})};
    EXPECT_THROW(align::build_detector_union_lattice({nonmonotonic}),
                 std::invalid_argument);
}

TEST(sci_align_T09,
     slot_identity_and_union_span_extremes_fail_closed_without_overflow) {
    const auto int64_min = std::numeric_limits<std::int64_t>::min();
    const auto int64_max = std::numeric_limits<std::int64_t>::max();
    const double inclusive_lower = static_cast<double>(int64_min);
    const double exclusive_upper = -inclusive_lower;

    EXPECT_EQ(align::round_half_up_slot(inclusive_lower), int64_min);
    EXPECT_THROW(align::round_half_up_slot(exclusive_upper),
                 std::overflow_error);
    EXPECT_THROW(
        align::round_half_up_slot(std::nextafter(
            inclusive_lower, -std::numeric_limits<double>::infinity())),
        std::overflow_error);

    const double largest_representable_in_range =
        std::nextafter(exclusive_upper, 0.0);
    EXPECT_EQ(
        align::round_half_up_slot(largest_representable_in_range),
        static_cast<std::int64_t>(largest_representable_in_range));

    align::DetectorLattice lattice;
    lattice.first_global_slot = int64_min;
    lattice.last_global_slot = 0;
    EXPECT_EQ(lattice.slot_count(),
              static_cast<std::uint64_t>(int64_max) + 2U);

    lattice.last_global_slot = int64_max;
    EXPECT_THROW(lattice.slot_count(), std::overflow_error);

    const align::HalfOpenSlotInterval full_signed_span{
        int64_min, int64_max};
    EXPECT_EQ(full_signed_span.size(),
              std::numeric_limits<std::uint64_t>::max());
}

// T12: conditional covariance includes shared-endpoint cross-output terms.
TEST(sci_align_T12, conditional_covariance_keeps_shared_endpoint_correlation) {
    Eigen::MatrixXd mapping(2, 3);
    mapping << 0.5, 0.5, 0.0,
               0.0, 0.5, 0.5;
    Eigen::MatrixXd covariance(3, 3);
    covariance << 4.0, 1.0, 0.5,
                  1.0, 9.0, 2.0,
                  0.5, 2.0, 16.0;
    const Eigen::MatrixXd expected =
        mapping * covariance * mapping.transpose();
    const auto actual =
        align::conditional_covariance_small(mapping, covariance);
    EXPECT_TRUE(actual.isApprox(expected, 0.0));
    EXPECT_GT(actual(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(
        actual(0, 0),
        align::linear_interpolation_variance(0.5, 4.0, 9.0, 1.0));
}

// T13: timing parameters propagate separately from detector noise and the
// deterministic curvature bound remains explicit.
TEST(sci_align_T13, offset_covariance_and_curvature_bound_are_separate) {
    Eigen::MatrixXd jacobian(2, 1);
    jacobian << 3.0, 3.0;
    Eigen::MatrixXd offset_covariance(1, 1);
    offset_covariance << 0.04;
    const auto timing =
        align::timing_covariance_small(jacobian, offset_covariance);
    EXPECT_DOUBLE_EQ(timing(0, 0), 0.36);
    EXPECT_DOUBLE_EQ(timing(0, 1), 0.36);
    EXPECT_DOUBLE_EQ(timing(1, 0), 0.36);
    EXPECT_DOUBLE_EQ(timing(1, 1), 0.36);

    EXPECT_DOUBLE_EQ(align::linear_interpolation_error_bound(2.0, 2.0),
                     1.0);
    const double quadratic_midpoint_error =
        std::abs(1.0 - 0.5 * (0.0 + 4.0));
    EXPECT_LE(quadratic_midpoint_error,
              align::linear_interpolation_error_bound(2.0, 2.0));
}

// T14: nominal support, acquired exposure, and synthesized support are not
// interchangeable, and a half-open scan slice counts each cell once.
TEST(sci_align_T14, exposure_counts_only_valid_original_intersection) {
    std::vector<align::ExposureCell> cells;
    cells.push_back({{0.0, 1.0}, {{0.1, 0.9}},
                     quality(align::Origin::original,
                             align::Validity::valid, align::Method::exact)});
    cells.push_back({{1.0, 2.0}, std::nullopt,
                     quality(align::Origin::synthesized,
                             align::Validity::valid, align::Method::linear)});
    cells.push_back({{2.0, 3.0}, std::nullopt,
                     quality(align::Origin::unavailable,
                             align::Validity::invalid)});
    cells.push_back({{3.0, 4.0}, {{2.5, 3.75}},
                     quality(align::Origin::original,
                             align::Validity::valid, align::Method::exact)});

    const auto full = align::summarize_exposure(cells, 0, cells.size());
    EXPECT_DOUBLE_EQ(full.nominal_span_seconds, 4.0);
    EXPECT_DOUBLE_EQ(full.acquired_exposure_seconds, 1.55);
    EXPECT_EQ(full.original_valid_count, 2u);
    EXPECT_EQ(full.synthesized_count, 1u);
    EXPECT_EQ(full.unavailable_count, 1u);

    const auto first_scan = align::summarize_exposure(cells, 0, 2);
    const auto second_scan = align::summarize_exposure(cells, 2, 4);
    EXPECT_DOUBLE_EQ(first_scan.acquired_exposure_seconds +
                         second_scan.acquired_exposure_seconds,
                     full.acquired_exposure_seconds);
    EXPECT_DOUBLE_EQ(first_scan.nominal_span_seconds +
                         second_scan.nominal_span_seconds,
                     full.nominal_span_seconds);
}

// T15: optional streams never determine the detector union; unavailable
// telescope values and absent-HWPR polarization eligibility are explicit.
TEST(sci_align_T15, missing_auxiliary_streams_do_not_trim_detector_lattice) {
    const auto one = timing_header(align::NativeRateFactor::one);
    const align::DetectorInterfaceCoordinates detector{
        "toltec0", one, reference_time({0.0, 0.008192, 0.016384})};
    const auto lattice = align::build_detector_union_lattice({detector});
    EXPECT_EQ(lattice.slot_count(), 3u);

    const Eigen::VectorXd empty_time;
    const Eigen::VectorXd empty_value;
    const Eigen::VectorXd targets = vector({0.0, 0.008192, 0.016384});
    const align::FieldContract scalar{
        align::FieldTopology::continuous_scalar, 0.1, 0.0, std::nullopt};
    const auto telescope = align::align_field_series(
        empty_time, empty_value, targets, scalar);
    ASSERT_EQ(telescope.size(), lattice.slot_count());
    for (const auto &cell : telescope) {
        EXPECT_EQ(cell.quality.origin, align::Origin::unavailable);
        EXPECT_EQ(cell.quality.reason, align::Reason::missing_stream);
    }

    const auto missing_telescope =
        align::missing_stream_disposition(align::StreamRole::telescope);
    EXPECT_TRUE(missing_telescope.detector_lattice_preserved);
    EXPECT_FALSE(missing_telescope.intensity_eligible);
    const auto missing_hwpr =
        align::missing_stream_disposition(align::StreamRole::hwpr);
    EXPECT_TRUE(missing_hwpr.detector_lattice_preserved);
    EXPECT_TRUE(missing_hwpr.intensity_eligible);
    EXPECT_FALSE(missing_hwpr.polarization_eligible);
}

TEST(sci_align_registry, binds_twenty_fields_and_two_exact_aliases) {
    EXPECT_EQ(align::active_field_registry.size(), 20U);
    EXPECT_EQ(align::active_field_aliases.size(), 2U);
    EXPECT_EQ(align::active_field_registry_version,
              "sci-align-active-field-registry-v2");
    EXPECT_EQ(align::active_field_registry_authority,
              "ALIGN-P0-D004-plus-SCI-ALIGN-001-HOLD-PRODUCER-AUTHORITY-2026-08-02");
    EXPECT_EQ(align::active_hold_native_semantics_authority,
              "SCI-ALIGN-001-HOLD-PRODUCER-AUTHORITY-2026-08-02;sha256=d6edb175c3aa62ccf92d9644675ece9c8db572a90146370a9c201c296f211c7e");

    std::set<std::string_view> field_ids;
    for (const auto &entry : align::active_field_registry) {
        EXPECT_TRUE(field_ids.insert(entry.field_id).second);
        EXPECT_EQ(align::active_field_source_dtype(entry), "float64");
        EXPECT_EQ(align::active_field_source_shape(entry), "time");
        EXPECT_FALSE(align::active_field_validity_policy(entry).empty());
        EXPECT_FALSE(align::active_field_support_rule(entry).empty());
        EXPECT_FALSE(
            align::active_field_runtime_maximum_support_span_sec(entry)
                .has_value());
    }

    const auto *hold = align::active_field_entry("Hold");
    ASSERT_NE(hold, nullptr);
    EXPECT_EQ(hold->unit, "1");
    EXPECT_EQ(align::active_field_raw_unit(*hold),
              "boolean_raw_attribute_conflicts_with_observed_multi_bit_word");
    EXPECT_EQ(align::field_operator_name(hold->permitted_operator),
              "legacy_4x_linear_any_nonzero");
    EXPECT_NE(hold->scientific_identity.find("producer-defined"),
              std::string_view::npos);
    EXPECT_NE(hold->frame.find("zero only science-valid"),
              std::string_view::npos);
    EXPECT_NE(align::active_field_validity_policy(*hold).find(
                  "unknown_bits_fail_closed"),
              std::string_view::npos);
    EXPECT_NE(align::active_field_validity_policy(*hold).find(
                  "transition_side_unresolved"),
              std::string_view::npos);
    EXPECT_EQ(align::active_field_source_authority(*hold),
              align::active_hold_native_semantics_authority);
    EXPECT_EQ(
        hold->output_identity,
        "Hold: post-nonzero 0/1 compatibility alias; exact raw word "
        "retained internally; no routine exporter");

    EXPECT_EQ(align::active_field_aliases[0].canonical_field_id,
              "lmt.source_ra");
    EXPECT_EQ(align::active_field_aliases[1].canonical_field_id,
              "lmt.source_dec");
}

}  // namespace
