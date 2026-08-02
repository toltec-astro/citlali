#include <tula/logging.h>

#include <citlali/core/pipeline/timestream_scan_context.h>
#include <citlali/core/pipeline/timestream_scan_generation.h>
#include <citlali/core/pipeline/timestream_output_provenance.h>
#include <citlali/core/engine/detail/kidsproc_gap_cardinality.h>
#include <citlali/core/utils/utils.h>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <tuple>

namespace {

TEST(sci_align_gap_guard,
     gap_loader_cardinality_ignores_legacy_direct_path_offsets) {
    EXPECT_NO_THROW(
        citlali::engine_detail::require_gap_stream_cardinality(11, 11));
    EXPECT_THROW(
        citlali::engine_detail::require_gap_stream_cardinality(11, 10),
        std::runtime_error);
}

struct GapGuardLogger {
    template <class... Args>
    void error(const char *, Args &&...) {}

    template <class... Args>
    void debug(const char *, Args &&...) {}
};

struct GapGuardCalibration {
    std::map<int, std::tuple<Eigen::Index, Eigen::Index>> nw_limits;
};

struct GapGuardFlags {
    Eigen::MatrixXi data;
};

struct GapGuardRtcData {
    GapGuardFlags flags;
};

struct PlannedGapScans {
    Eigen::MatrixXd data;
};

struct PlannedGapRtcData {
    PlannedGapScans scans;
};

struct PlannedGapRawObs {};

struct PlannedGapTelescope {
    double fsmp = 1.0;
    Eigen::MatrixXI scan_indices;
};

struct PlannedGapKidsProc {
    Eigen::VectorXd native_time;
    Eigen::MatrixXd native_data;

    template <class... Args>
    Eigen::MatrixXd populate_rtc_from_rawobs(Args &&...) {
        throw std::logic_error(
            "planned-gap fixture entered the direct path");
    }

    template <class RawObs, class ScanIndices, class StartIndices,
              class CommonTime, class NetworkTimes>
    std::vector<int> load_rawobs_gaps(
        RawObs &, Eigen::Index, ScanIndices &, StartIndices &,
        CommonTime &, NetworkTimes &, double) {
        return {0};
    }

    template <class Loaded, class CommonTime, class NetworkTimes,
              class Masks, class Permissions, class ScanIndices,
              class TimestreamType>
    Eigen::MatrixXd populate_rtc_gaps(
        Loaded &, CommonTime &common_time, NetworkTimes &, Masks &masks,
        const Permissions &permissions, int scan, double cadence,
        double half_cell, ScanIndices &scan_indices, int n_pts,
        int n_dets, TimestreamType) {
        if (n_dets != native_data.cols() || permissions.size() != 1 ||
            masks.size() != 1) {
            throw std::logic_error("invalid planned-gap test fixture");
        }
        const auto common = common_time
                                .segment(scan_indices(2, scan), n_pts)
                                .eval();
        const auto mask = masks.front()
                              .segment(scan_indices(2, scan), n_pts)
                              .eval();
        return engine_utils::interp_data_with_observation_resolved_admission(
            common, mask, native_time, native_data,
            permissions.front() != 0, cadence, half_cell);
    }
};

Eigen::MatrixXd affine_data(const Eigen::VectorXd &time,
                            Eigen::Index columns = 1) {
    Eigen::MatrixXd result(time.size(), columns);
    for (Eigen::Index row = 0; row < time.size(); ++row) {
        for (Eigen::Index column = 0; column < columns; ++column) {
            result(row, column) =
                2.0 + 3.0 * time(row) + static_cast<double>(column);
        }
    }
    return result;
}

citlali::pipeline::TimestreamAlignmentState single_interface_alignment(
    const Eigen::VectorXi &mask) {
    citlali::pipeline::TimestreamAlignmentState state;
    state.grid.initialized = true;
    state.grid.phase_sec = 0.0;
    state.grid.cadence_sec = 1.0;
    state.grid.exclusive_half_cell_sec = 0.5;
    state.grid.first_global_slot = 0;
    state.grid.last_global_slot = mask.size() - 1;
    state.common_time = Eigen::VectorXd::LinSpaced(
        mask.size(), 0.0, static_cast<double>(mask.size() - 1));
    state.masks.push_back(mask);
    citlali::pipeline::AlignmentInterfaceSummary interface;
    interface.interface_id = "toltec0";
    interface.roach_index = 0;
    state.interfaces.push_back(interface);
    state.support.nominal_slot_count =
        static_cast<std::uint64_t>(mask.size());
    state.support.acquired_original_count =
        static_cast<std::uint64_t>(mask.sum());
    state.support.timing_coordinate_valid_original_count =
        state.support.acquired_original_count;
    state.support.unavailable_count =
        state.support.nominal_slot_count -
        state.support.acquired_original_count;
    state.support.nominal_span_sec =
        static_cast<double>(mask.size());
    state.support.acquired_original_cadence_weighted_support_sec =
        static_cast<double>(mask.sum());
    return state;
}

citlali::pipeline::sci_align::ScanWindowPlan single_stable_scan_plan(
    Eigen::Index sample_count,
    citlali::pipeline::sci_align::HalfOpenInterval science,
    citlali::pipeline::sci_align::HalfOpenInterval context) {
    citlali::pipeline::sci_align::ScanWindowPlan plan;
    plan.policy = "test_expanded_context";
    plan.observation_sample_count = sample_count;
    plan.physical_records.push_back(
        {0, {0, sample_count}, "test_continuous_observation"});
    citlali::pipeline::sci_align::ScanWindowRecord record;
    record.stable_id = 0;
    record.physical_id = 0;
    record.identity_authority = "test_processing_chunk";
    record.processing = science;
    record.science = science;
    record.context = context;
    record.status = citlali::pipeline::sci_align::ScanStatus::usable;
    record.legacy_processing_admitted = true;
    record.compatibility_ordinal = 0;
    plan.records.push_back(record);
    plan.compatibility_to_stable_id.push_back(0);
    citlali::pipeline::sci_align::validate_scan_window_plan(plan);
    return plan;
}

}  // namespace

TEST(sci_align_gap_guard, assigns_with_round_half_up_and_fills_whole_affine_run) {
    const Eigen::VectorXd common =
        (Eigen::VectorXd(4) << 0.0, 1.0, 2.0, 3.0).finished();
    const Eigen::VectorXi mask =
        (Eigen::VectorXi(4) << 1, 0, 1, 1).finished();
    const Eigen::VectorXd native_time =
        (Eigen::VectorXd(3) << 0.49, 2.49, 3.49).finished();
    const Eigen::MatrixXd native = affine_data(native_time, 2);

    const auto aligned =
        engine_utils::interp_data(common, mask, native_time, native);

    EXPECT_TRUE(aligned.row(0).isApprox(native.row(0), 0.0));
    EXPECT_TRUE(aligned.row(2).isApprox(native.row(1), 0.0));
    EXPECT_TRUE(aligned.row(3).isApprox(native.row(2), 0.0));
    const double weight = (common(1) - common(0)) /
                          (common(2) - common(0));
    EXPECT_TRUE(aligned.row(1).isApprox(
        (1.0 - weight) * native.row(0) + weight * native.row(1),
        1.0e-14));
}

TEST(sci_align_gap_guard,
     rejects_malformed_shapes_and_output_product_before_allocation) {
    const Eigen::VectorXd common = Eigen::VectorXd::Zero(2);
    const Eigen::VectorXi wrong_mask = Eigen::VectorXi::Zero(1);
    const Eigen::VectorXd native_time = Eigen::VectorXd::Zero(1);
    const Eigen::MatrixXd native = Eigen::MatrixXd::Zero(1, 1);
    EXPECT_THROW(engine_utils::interp_data(
                     common, wrong_mask, native_time, native),
                 std::invalid_argument);

    EXPECT_THROW(
        engine_utils::require_interp_output_dimensions(
            std::numeric_limits<Eigen::Index>::max() / 2 + 1, 2),
        std::overflow_error);
}

TEST(sci_align_gap_guard, rejects_exact_half_cell_and_slot_collision) {
    const Eigen::VectorXd common =
        (Eigen::VectorXd(2) << 0.0, 1.0).finished();

    const Eigen::VectorXi boundary_mask =
        (Eigen::VectorXi(2) << 0, 1).finished();
    const Eigen::VectorXd boundary_time =
        (Eigen::VectorXd(1) << 0.5).finished();
    EXPECT_THROW(
        engine_utils::interp_data(common, boundary_mask, boundary_time,
                                  affine_data(boundary_time)),
        std::invalid_argument);

    const Eigen::VectorXi collision_mask =
        (Eigen::VectorXi(2) << 1, 0).finished();
    const Eigen::VectorXd collision_time =
        (Eigen::VectorXd(2) << 0.1, 0.2).finished();
    EXPECT_THROW(
        engine_utils::interp_data(common, collision_mask, collision_time,
                                  affine_data(collision_time)),
        std::invalid_argument);
}

TEST(sci_align_gap_guard, never_extrapolates_edges_or_partially_fills_columns) {
    const Eigen::VectorXd common =
        Eigen::VectorXd::LinSpaced(8, 0.0, 7.0);
    const Eigen::VectorXi edge_mask =
        (Eigen::VectorXi(8) << 0, 1, 1, 1, 1, 1, 1, 0).finished();
    const Eigen::VectorXd edge_time =
        (Eigen::VectorXd(6) << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0).finished();
    const auto edge_aligned = engine_utils::interp_data(
        common, edge_mask, edge_time, affine_data(edge_time, 2));
    EXPECT_TRUE(edge_aligned.row(0).isZero(0.0));
    EXPECT_TRUE(edge_aligned.row(7).isZero(0.0));

    const Eigen::VectorXd internal_common =
        (Eigen::VectorXd(5) << 0.0, 1.0, 2.0, 3.0, 4.0).finished();
    const Eigen::VectorXi internal_mask =
        (Eigen::VectorXi(5) << 1, 0, 1, 1, 1).finished();
    const Eigen::VectorXd internal_time =
        (Eigen::VectorXd(4) << 0.0, 2.0, 3.0, 4.0).finished();
    Eigen::MatrixXd invalid = affine_data(internal_time, 2);
    invalid(0, 1) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(engine_utils::interp_data(
                     internal_common, internal_mask, internal_time, invalid),
                 std::invalid_argument);

    const Eigen::VectorXi cross_chunk_mask =
        (Eigen::VectorXi(5) << 0, 1, 1, 1, 1).finished();
    const Eigen::VectorXd cross_chunk_time =
        (Eigen::VectorXd(5) << -1.0, 1.0, 2.0, 3.0, 4.0).finished();
    Eigen::MatrixXd invalid_external =
        affine_data(cross_chunk_time, 2);
    invalid_external(0, 0) =
        std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(engine_utils::interp_data(
                     internal_common, cross_chunk_mask, cross_chunk_time,
                     invalid_external),
                 std::invalid_argument);
}

TEST(sci_align_gap_guard, fills_cross_chunk_edge_only_with_external_native_endpoint) {
    const Eigen::VectorXd common =
        (Eigen::VectorXd(4) << 10.0, 11.0, 12.0, 13.0).finished();

    const Eigen::VectorXi left_mask =
        (Eigen::VectorXi(4) << 0, 1, 1, 1).finished();
    const Eigen::VectorXd left_source_time =
        (Eigen::VectorXd(4) << 9.0, 11.0, 12.0, 13.0).finished();
    const auto left = engine_utils::interp_data(
        common, left_mask, left_source_time, affine_data(left_source_time));
    EXPECT_DOUBLE_EQ(left(0, 0), 2.0 + 3.0 * 10.0);

    const Eigen::VectorXi right_mask =
        (Eigen::VectorXi(4) << 1, 1, 1, 0).finished();
    const Eigen::VectorXd right_source_time =
        (Eigen::VectorXd(4) << 10.0, 11.0, 12.0, 14.0).finished();
    const auto right = engine_utils::interp_data(
        common, right_mask, right_source_time,
        affine_data(right_source_time));
    EXPECT_DOUBLE_EQ(right(3, 0), 2.0 + 3.0 * 13.0);
}

TEST(sci_align_gap_guard, fills_cross_chunk_gap_from_nonadjacent_external_slot) {
    const Eigen::VectorXd common =
        (Eigen::VectorXd(4) << 10.0, 11.0, 12.0, 13.0).finished();

    const Eigen::VectorXi left_mask =
        (Eigen::VectorXi(4) << 0, 1, 1, 1).finished();
    const Eigen::VectorXd left_source_time =
        (Eigen::VectorXd(4) << 7.0, 11.0, 12.0, 13.0).finished();
    const auto left = engine_utils::interp_data(
        common, left_mask, left_source_time,
        affine_data(left_source_time));
    EXPECT_DOUBLE_EQ(left(0, 0), 2.0 + 3.0 * 10.0);

    const Eigen::VectorXi right_mask =
        (Eigen::VectorXi(4) << 1, 1, 1, 0).finished();
    const Eigen::VectorXd right_source_time =
        (Eigen::VectorXd(4) << 10.0, 11.0, 12.0, 16.0).finished();
    const auto right = engine_utils::interp_data(
        common, right_mask, right_source_time,
        affine_data(right_source_time));
    EXPECT_DOUBLE_EQ(right(3, 0), 2.0 + 3.0 * 13.0);
}

TEST(sci_align_gap_guard, rejects_nonuniform_common_grid) {
    const Eigen::VectorXd common =
        (Eigen::VectorXd(3) << 0.0, 1.0, 2.1).finished();
    const Eigen::VectorXi mask = Eigen::VectorXi::Ones(3);
    EXPECT_THROW(
        engine_utils::interp_data(common, mask, common,
                                  affine_data(common)),
        std::invalid_argument);
}

TEST(sci_align_gap_guard, consumes_realized_cadence_and_half_cell) {
    const Eigen::VectorXd common =
        (Eigen::VectorXd(3) << 0.0, 1.0, 2.0).finished();
    const Eigen::VectorXi mask = Eigen::VectorXi::Ones(3);
    const auto native = affine_data(common);

    const auto aligned = engine_utils::interp_data(
        common, mask, common, native, true, 1.0, 0.5);
    EXPECT_TRUE(aligned.isApprox(native, 0.0));

    EXPECT_THROW(
        engine_utils::interp_data(
            common, mask, common, native, true, 0.5, 0.25),
        std::invalid_argument);
    EXPECT_THROW(
        engine_utils::interp_data(
            common, mask, common, native, true, 1.0, 0.25),
        std::invalid_argument);
    EXPECT_THROW(
        engine_utils::interp_data(
            common, mask, common, native, true, 1.0),
        std::invalid_argument);
}

TEST(sci_align_gap_guard, requires_telescope_rate_to_match_realized_grid) {
    citlali::pipeline::AlignmentGridState grid;
    grid.initialized = true;
    grid.cadence_sec = 0.008192;
    grid.exclusive_half_cell_sec = 0.004096;

    EXPECT_NO_THROW(
        citlali::pipeline::require_consistent_gap_execution_grid(
            grid, 122.0703125));
    EXPECT_THROW(
        citlali::pipeline::require_consistent_gap_execution_grid(
            grid, 100.0),
        std::invalid_argument);

    auto invalid = grid;
    invalid.initialized = false;
    EXPECT_THROW(
        citlali::pipeline::require_consistent_gap_execution_grid(
            invalid, 122.0703125),
        std::invalid_argument);

    invalid = grid;
    invalid.exclusive_half_cell_sec = 0.003;
    EXPECT_THROW(
        citlali::pipeline::require_consistent_gap_execution_grid(
            invalid, 122.0703125),
        std::invalid_argument);
}

TEST(sci_align_gap_guard, permits_exact_quarter_but_skips_over_quarter_surrogate) {
    const Eigen::VectorXd common = Eigen::VectorXd::LinSpaced(8, 0.0, 7.0);

    const Eigen::VectorXi exact_mask =
        (Eigen::VectorXi(8) << 1, 1, 0, 0, 1, 1, 1, 1).finished();
    const Eigen::VectorXd exact_time =
        (Eigen::VectorXd(6) << 0.0, 1.0, 4.0, 5.0, 6.0, 7.0).finished();
    const auto exact = engine_utils::interp_data(
        common, exact_mask, exact_time, affine_data(exact_time));
    EXPECT_DOUBLE_EQ(exact(2, 0), 2.0 + 3.0 * 2.0);
    EXPECT_DOUBLE_EQ(exact(3, 0), 2.0 + 3.0 * 3.0);

    const Eigen::VectorXi over_mask =
        (Eigen::VectorXi(8) << 1, 1, 0, 0, 0, 1, 1, 1).finished();
    const Eigen::VectorXd over_time =
        (Eigen::VectorXd(5) << 0.0, 1.0, 5.0, 6.0, 7.0).finished();
    const auto over = engine_utils::interp_data(
        common, over_mask, over_time, affine_data(over_time));
    EXPECT_TRUE(over.block(2, 0, 3, 1).isZero(0.0));
}

TEST(sci_align_gap_guard, classifies_exact_missing_guard_and_unusable_separately) {
    const Eigen::VectorXi mask =
        (Eigen::VectorXi(8) << 1, 1, 0, 0, 1, 1, 1, 1).finished();
    const auto classification =
        citlali::pipeline::classify_gap_mask_chunk(mask, 0, 8, 1);

    EXPECT_EQ(classification.cumulative_missing, 2);
    EXPECT_EQ(classification.longest_missing_run, 2);
    EXPECT_FALSE(classification.network_chunk_unusable);
    const Eigen::VectorXi expected_missing =
        (Eigen::VectorXi(8) << 0, 0, 1, 1, 0, 0, 0, 0).finished();
    const Eigen::VectorXi expected_guard =
        (Eigen::VectorXi(8) << 0, 1, 0, 0, 1, 0, 0, 0).finished();
    EXPECT_TRUE((classification.exact_missing.array() ==
                 expected_missing.array()).all());
    EXPECT_TRUE((classification.processing_guard.array() ==
                 expected_guard.array()).all());
}

TEST(sci_align_gap_guard, just_over_quarter_flags_only_the_affected_network) {
    GapGuardRtcData rtc;
    rtc.flags.data = Eigen::MatrixXi::Zero(8, 4);
    GapGuardCalibration calibration{
        {{0, {0, 2}}, {1, {2, 4}}}};
    const std::map<int, Eigen::VectorXi> masks{
        {0, (Eigen::VectorXi(8) << 1, 1, 0, 0, 0, 1, 1, 1).finished()},
        {1, (Eigen::VectorXi(8) << 1, 1, 1, 1, 0, 1, 1, 1).finished()}};
    auto logger = std::make_shared<GapGuardLogger>();

    citlali::pipeline::apply_gap_masks_to_rtc_flags(
        rtc, calibration, masks, 0, 0, logger);

    EXPECT_TRUE((rtc.flags.data.leftCols(2).array() == 1).all());
    EXPECT_TRUE((rtc.flags.data.rightCols(2).colwise().sum().array() == 1)
                    .all());
    EXPECT_TRUE(rtc.flags.data.block(0, 2, 4, 2).isZero(0));
    EXPECT_TRUE(rtc.flags.data.block(5, 2, 3, 2).isZero(0));
}

TEST(sci_align_gap_guard, cumulative_short_runs_trigger_and_cross_chunk_guard_is_clipped) {
    const Eigen::VectorXi cumulative =
        (Eigen::VectorXi(12) << 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1)
            .finished();
    const auto cumulative_classification =
        citlali::pipeline::classify_gap_mask_chunk(cumulative, 0, 12, 0);
    EXPECT_EQ(cumulative_classification.longest_missing_run, 1);
    EXPECT_EQ(cumulative_classification.cumulative_missing, 3);
    EXPECT_FALSE(cumulative_classification.network_chunk_unusable);

    Eigen::VectorXi just_over = cumulative;
    just_over(8) = 0;
    const auto just_over_classification =
        citlali::pipeline::classify_gap_mask_chunk(just_over, 0, 12, 0);
    EXPECT_EQ(just_over_classification.longest_missing_run, 1);
    EXPECT_EQ(just_over_classification.cumulative_missing, 4);
    EXPECT_TRUE(just_over_classification.network_chunk_unusable);

    const Eigen::VectorXi cross_chunk =
        (Eigen::VectorXi(9) << 1, 1, 0, 0, 1, 1, 1, 1, 1).finished();
    const auto clipped =
        citlali::pipeline::classify_gap_mask_chunk(cross_chunk, 3, 4, 2);
    EXPECT_EQ(clipped.cumulative_missing, 1);
    EXPECT_EQ(clipped.longest_missing_run, 1);
    EXPECT_FALSE(clipped.network_chunk_unusable);
    const Eigen::VectorXi expected_clipped_missing =
        (Eigen::VectorXi(4) << 1, 0, 0, 0).finished();
    const Eigen::VectorXi expected_clipped_guard =
        (Eigen::VectorXi(4) << 0, 1, 1, 0).finished();
    EXPECT_TRUE((clipped.exact_missing.array() ==
                 expected_clipped_missing.array()).all());
    EXPECT_TRUE((clipped.processing_guard.array() ==
                 expected_clipped_guard.array()).all());
}

TEST(sci_align_gap_guard,
     expanded_context_cannot_change_science_window_quarter_admission) {
    // The stable science window contains exactly two missing cells out of
    // eight, which is admitted.  Four additional context-only missing cells
    // make the expanded context exceed one quarter but must not alter that
    // decision.
    const Eigen::VectorXi context_heavy =
        (Eigen::VectorXi(16) << 0, 0, 1, 1, 0, 0, 1, 1,
                                1, 1, 1, 1, 1, 1, 0, 0)
            .finished();
    auto admitted = single_interface_alignment(context_heavy);
    const auto admitted_plan = single_stable_scan_plan(
        16, {4, 12}, {0, 16});

    citlali::pipeline::finalize_alignment_gap_processing_plan(
        admitted, admitted_plan, 1, citlali::config::TodType::rs);

    ASSERT_EQ(admitted.chunk_dispositions.size(), 1U);
    const auto &admitted_disposition = admitted.chunk_dispositions.front();
    EXPECT_EQ(admitted_disposition.cumulative_missing_count, 2);
    EXPECT_EQ(admitted_disposition.longest_missing_run_count, 2);
    EXPECT_FALSE(admitted_disposition.full_network_unusable);
    EXPECT_EQ(admitted.support.guarded_original_count, 1U);
    EXPECT_EQ(admitted.support.gap_policy_eligible_original_count, 5U);
    EXPECT_NO_THROW(
        citlali::pipeline::validate_alignment_processing_support(
            admitted, &admitted_plan));

    // Conversely, a two-cell gap in a four-cell science window is unusable
    // even though the expanded context dilutes it below one quarter.
    const Eigen::VectorXi context_diluted =
        (Eigen::VectorXi(16) << 1, 1, 1, 1, 0, 0, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1)
            .finished();
    auto unusable = single_interface_alignment(context_diluted);
    const auto unusable_plan = single_stable_scan_plan(
        16, {4, 8}, {0, 16});

    citlali::pipeline::finalize_alignment_gap_processing_plan(
        unusable, unusable_plan, 0, citlali::config::TodType::rs);

    ASSERT_EQ(unusable.chunk_dispositions.size(), 1U);
    const auto &unusable_disposition = unusable.chunk_dispositions.front();
    EXPECT_EQ(unusable_disposition.cumulative_missing_count, 2);
    EXPECT_EQ(unusable_disposition.longest_missing_run_count, 2);
    EXPECT_TRUE(unusable_disposition.full_network_unusable);
    EXPECT_TRUE(unusable_disposition.synthesized_missing_runs.empty());
    EXPECT_TRUE(unusable_disposition.processing_guard_runs.empty());
    EXPECT_EQ(unusable.support.guarded_original_count, 0U);
    EXPECT_EQ(unusable.support.gap_policy_eligible_original_count, 0U);
    EXPECT_NO_THROW(
        citlali::pipeline::validate_alignment_processing_support(
            unusable, &unusable_plan));
}

TEST(sci_align_gap_guard,
     ordinary_scan_interface_state_is_an_implicit_sparse_default) {
    const Eigen::VectorXi mask = Eigen::VectorXi::Ones(8);
    auto alignment = single_interface_alignment(mask);
    const auto scan_plan = single_stable_scan_plan(8, {0, 8}, {0, 8});

    citlali::pipeline::finalize_alignment_gap_processing_plan(
        alignment, scan_plan, 2, citlali::config::TodType::xs);

    EXPECT_TRUE(alignment.chunk_dispositions.empty());
    EXPECT_EQ(alignment.support.synthesized_count, 0U);
    EXPECT_EQ(alignment.support.unavailable_count, 0U);
    EXPECT_EQ(alignment.support.guarded_original_count, 0U);
    EXPECT_EQ(alignment.support.gap_policy_eligible_original_count, 8U);
    const auto xs_permissions =
        citlali::pipeline::alignment_gap_synthesis_permissions(alignment, 0);
    ASSERT_EQ(xs_permissions.size(), 1U);
    EXPECT_EQ(xs_permissions.front(), 1U);
    EXPECT_NO_THROW(
        citlali::pipeline::validate_alignment_processing_support(
            alignment, &scan_plan));

    citlali::pipeline::AlignmentChunkDisposition spurious;
    spurious.stable_scan_id = 0;
    spurious.compatibility_ordinal = 0;
    spurious.interface_id = "toltec0";
    spurious.roach_index = 0;
    spurious.context_start = 0;
    spurious.context_stop = 8;
    spurious.continuity_surrogate_permitted = true;
    alignment.chunk_dispositions.push_back(spurious);
    EXPECT_THROW(
        citlali::pipeline::validate_alignment_processing_support(
            alignment, &scan_plan),
        std::logic_error);

    auto rs_alignment = single_interface_alignment(mask);
    citlali::pipeline::finalize_alignment_gap_processing_plan(
        rs_alignment, scan_plan, 2, citlali::config::TodType::rs);
    const auto rs_permissions =
        citlali::pipeline::alignment_gap_synthesis_permissions(
            rs_alignment, 0);
    EXPECT_EQ(rs_permissions.front(), 0U);
}

TEST(sci_align_gap_guard,
     compact_run_union_counts_repeated_context_actions_once) {
    const Eigen::VectorXi mask =
        (Eigen::VectorXi(8) << 1, 1, 1, 0, 1, 1, 1, 1).finished();
    auto alignment = single_interface_alignment(mask);
    alignment.exceptions.push_back({
        "toltec0", "detector_acquisition", 3, 4,
        "native_detector_gap", "unavailable_original",
        "bounded_continuity_candidate", "bounded_by_acquired_originals",
        2, 4});

    citlali::pipeline::sci_align::ScanWindowPlan plan;
    plan.policy = "test_overlapping_contexts";
    plan.observation_sample_count = 8;
    plan.physical_records.push_back(
        {0, {0, 8}, "test_continuous_observation"});
    for (Eigen::Index ordinal = 0; ordinal < 2; ++ordinal) {
        citlali::pipeline::sci_align::ScanWindowRecord record;
        record.stable_id = ordinal;
        record.physical_id = 0;
        record.identity_authority = "test_processing_chunk";
        record.processing = ordinal == 0
            ? citlali::pipeline::sci_align::HalfOpenInterval{0, 4}
            : citlali::pipeline::sci_align::HalfOpenInterval{4, 8};
        record.science = record.processing;
        record.context = ordinal == 0
            ? citlali::pipeline::sci_align::HalfOpenInterval{0, 5}
            : citlali::pipeline::sci_align::HalfOpenInterval{2, 8};
        record.status = citlali::pipeline::sci_align::ScanStatus::usable;
        record.legacy_processing_admitted = true;
        record.compatibility_ordinal = ordinal;
        plan.records.push_back(record);
        plan.compatibility_to_stable_id.push_back(ordinal);
    }
    citlali::pipeline::sci_align::validate_scan_window_plan(plan);

    citlali::pipeline::finalize_alignment_gap_processing_plan(
        alignment, plan, 0, citlali::config::TodType::xs);

    ASSERT_EQ(alignment.chunk_dispositions.size(), 2U);
    EXPECT_EQ(alignment.processing_support
                  .synthesized_processing_occurrence_count,
              2U);
    EXPECT_EQ(alignment.support.synthesized_count, 1U);
    EXPECT_EQ(alignment.support.gap_policy_eligible_original_count, 7U);
    EXPECT_NO_THROW(
        citlali::pipeline::validate_alignment_processing_support(
            alignment, &plan));
}

TEST(sci_align_gap_guard,
     production_scan_population_obeys_resolved_science_window_admission) {
    const Eigen::VectorXi mask =
        (Eigen::VectorXi(16) << 0, 0, 1, 1, 0, 0, 1, 1,
                                1, 1, 1, 1, 1, 1, 0, 0)
            .finished();
    auto alignment = single_interface_alignment(mask);
    const auto scan_plan = single_stable_scan_plan(
        16, {4, 12}, {0, 16});
    citlali::pipeline::finalize_alignment_gap_processing_plan(
        alignment, scan_plan, 0, citlali::config::TodType::xs);

    PlannedGapTelescope telescope;
    telescope.scan_indices =
        citlali::pipeline::sci_align::compatibility_scan_indices(scan_plan);
    PlannedGapKidsProc kidsproc;
    const Eigen::Index native_count = mask.sum();
    kidsproc.native_time.resize(native_count);
    Eigen::Index native_row = 0;
    for (Eigen::Index slot = 0; slot < mask.size(); ++slot) {
        if (mask(slot) != 0) {
            kidsproc.native_time(native_row++) =
                alignment.common_time(slot);
        }
    }
    kidsproc.native_data = affine_data(kidsproc.native_time);
    PlannedGapRtcData rtcdata;
    PlannedGapRawObs rawobs;

    citlali::pipeline::populate_rtc_scan_samples(
        rtcdata, kidsproc, rawobs, 0, telescope, alignment, true,
        16, 1, citlali::config::TodType::xs);

    ASSERT_EQ(rtcdata.scans.data.rows(), 16);
    EXPECT_EQ(alignment.support.gap_policy_eligible_original_count, 6U);
    EXPECT_EQ(alignment.support.synthesized_count, 2U);
    EXPECT_EQ(alignment.support.unavailable_count, 4U);
    EXPECT_DOUBLE_EQ(rtcdata.scans.data(4, 0), 2.0 + 3.0 * 4.0);
    EXPECT_DOUBLE_EQ(rtcdata.scans.data(5, 0), 2.0 + 3.0 * 5.0);
    EXPECT_TRUE(rtcdata.scans.data.topRows(2).isZero(0.0));
    EXPECT_TRUE(rtcdata.scans.data.bottomRows(2).isZero(0.0));
}
