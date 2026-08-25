#include "sci_align_native_gap_fixture.h"

#include <citlali/core/pipeline/timestream_rtc_run_adapter.h>
#include <citlali/core/pipeline/timestream_ptc_cohort_adapter.h>

#include <gtest/gtest.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace {

namespace fixture = citlali::test_support::sci_align;
namespace pipeline = citlali::pipeline;

pipeline::NativeRtcProcessedRun identity_body(
    const pipeline::NativeRtcRunInput &input) {
    return {input.measured_values, input.input_flag_bits};
}

fixture::NativeGapFixtureV1 complete_identical_time_fixture() {
    auto result = fixture::load_native_gap_fixture_v1();
    result.network(0).measured_values(0, 0) = -0.0;
    auto &network7 = result.network(7);
    network7.reconstructed_times_unix_sec =
        result.common_slot_reference_times_unix_sec;
    network7.packet_counters = {700, 701, 702, 703, 704};
    network7.legacy_presence_mask = Eigen::VectorXi::Ones(5);
    network7.expected_slot_native_rows = {700, 701, 702, 703, 704};
    network7.measured_values.resize(5, 2);
    network7.measured_values <<
        7000.0, 7001.0,
        7010.0, 7011.0,
        7020.0, 7021.0,
        7030.0, 7031.0,
        7040.0, 7041.0;
    network7.original_flag_bits.resize(5, 2);
    network7.original_flag_bits <<
        2, 16,
        8, 32,
        512, 1024,
        2048, 4096,
        8192, 16384;
    network7.expected_packet_contiguous_runs = {{700, 705}};
    result.expected_complete_cohort_slot_runs = {{0, 5}};
    return result;
}

struct ResultSnapshot {
    std::vector<std::tuple<std::size_t, pipeline::TimestreamNetworkId,
                           pipeline::TimestreamNativeRow,
                           pipeline::NativeDetectorFlagBits, double>> rows;

    friend bool operator==(const ResultSnapshot &,
                           const ResultSnapshot &) = default;
};

ResultSnapshot snapshot(const pipeline::NativeRtcDispatchResult &result) {
    ResultSnapshot frozen;
    for (const auto &run : result.runs) {
        for (Eigen::Index row = 0; row < run.selected_values.rows(); ++row) {
            for (Eigen::Index detector = 0;
                 detector < run.selected_values.cols(); ++detector) {
                frozen.rows.emplace_back(
                    run.input.segment_ordinal,
                    run.input.run.network_id,
                    run.support.at(static_cast<std::size_t>(row))
                        .selected_anchor.native_row(),
                    run.ored_flag_bits(row, detector),
                    run.selected_values(row, detector));
            }
        }
    }
    return frozen;
}

TEST(sci_align_rtc_run_adapter,
     frozen_gap_oracle_resets_factor2_anchors_and_never_bridges_gap) {
    const auto loaded = fixture::load_native_gap_fixture_v1();
    const auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    const auto result = pipeline::dispatch_native_rtc_runs(
        *scan, {2, false}, identity_body);

    ASSERT_EQ(result.runs.size(), 4U);
    EXPECT_EQ(result.output_row_count(), 4U);
    std::set<std::size_t> supported_slots;
    for (std::size_t index = 0;
         index < loaded.expected_stage4_factor2_support.size(); ++index) {
        const auto &expected =
            loaded.expected_stage4_factor2_support[index];
        const auto &actual = result.runs[index];
        EXPECT_EQ(actual.input.segment_ordinal,
                  expected.segment_ordinal);
        EXPECT_EQ(actual.input.run.network_id, expected.network_id);
        EXPECT_EQ(actual.input.first_common_slot,
                  expected.first_common_slot);
        EXPECT_EQ(actual.input.past_last_common_slot,
                  expected.past_last_common_slot);
        ASSERT_EQ(actual.support.size(), 1U);
        const auto &support = actual.support.front();
        EXPECT_EQ(support.selected_anchor.native_row(),
                  expected.selected_anchor_native_row);
        EXPECT_EQ(support.exact_common_slots,
                  (std::vector<std::size_t>{
                      expected.first_common_slot,
                      expected.first_common_slot + 1}));
        ASSERT_EQ(support.exact_native_support.size(), 2U);
        EXPECT_EQ(support.exact_native_support[0].native_row(),
                  expected.selected_anchor_native_row);
        EXPECT_EQ(support.exact_native_support[1].native_row(),
                  expected.selected_anchor_native_row + 1);
        EXPECT_FALSE(support.final_short_support);
        for (const auto slot : support.exact_common_slots) {
            supported_slots.insert(slot);
        }

        ASSERT_EQ(support.detector_columns.size(),
                  expected.original_flag_or_by_channel.size());
        for (std::size_t detector = 0;
             detector < support.detector_columns.size(); ++detector) {
            const auto column = support.detector_columns[detector];
            const auto raw_channel =
                scan->binding(column).raw_channel;
            const auto expected_bits =
                expected.original_flag_or_by_channel.at(
                    static_cast<std::size_t>(raw_channel));
            EXPECT_EQ(support.ored_flag_support[detector],
                      expected_bits);
            EXPECT_EQ(actual.ored_flag_bits(
                          0, static_cast<Eigen::Index>(detector)),
                      expected_bits);
            EXPECT_DOUBLE_EQ(
                actual.selected_values(
                    0, static_cast<Eigen::Index>(detector)),
                actual.input.measured_values(
                    0, static_cast<Eigen::Index>(detector)));
        }
    }
    EXPECT_EQ(supported_slots,
              (std::set<std::size_t>{0, 1, 3, 4}));

    const auto &nw0_first = result.runs[0];
    const auto &nw0_second = result.runs[2];
    EXPECT_EQ(nw0_first.input.run.past_last_native_row, 102);
    EXPECT_EQ(nw0_second.input.run.first_native_row, 103);
    EXPECT_FALSE(nw0_first.input.run.boundary_after
                     .counter_discontinuity.has_value());
    EXPECT_FALSE(nw0_second.input.run.boundary_before
                     .counter_discontinuity.has_value());
}

TEST(sci_align_rtc_run_adapter,
     complete_identical_times_are_exactly_legacy_rectangular) {
    const auto loaded = complete_identical_time_fixture();
    const auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    const auto native = pipeline::dispatch_native_rtc_runs(
        *scan, {2, false}, identity_body);
    ASSERT_EQ(native.runs.size(), 2U);

    Eigen::MatrixXd rectangular(5, 4);
    pipeline::NativeDetectorFlagBitsMatrix rectangular_flags(5, 4);
    for (std::size_t slot = 0; slot < 5; ++slot) {
        for (Eigen::Index detector = 0; detector < 4; ++detector) {
            const auto cell = scan->cell(slot, detector);
            ASSERT_TRUE(cell.mapped());
            rectangular(static_cast<Eigen::Index>(slot), detector) =
                *cell.measured_value();
            rectangular_flags(static_cast<Eigen::Index>(slot), detector) =
                cell.original_flag_bits();
        }
    }
    timestream::Downsampler legacy;
    legacy.factor = 2;
    Eigen::MatrixXd expected_values;
    legacy.downsample(rectangular, expected_values);
    pipeline::NativeDetectorFlagBitsMatrix expected_flags(3, 4);
    expected_flags.setZero();
    for (Eigen::Index output_row = 0; output_row < 3; ++output_row) {
        for (Eigen::Index row = output_row * 2;
             row < std::min<Eigen::Index>(output_row * 2 + 2, 5);
             ++row) {
            for (Eigen::Index detector = 0; detector < 4; ++detector) {
                expected_flags(output_row, detector) |=
                    rectangular_flags(row, detector);
            }
        }
    }

    Eigen::MatrixXd actual_values(3, 4);
    pipeline::NativeDetectorFlagBitsMatrix actual_flags(3, 4);
    for (const auto &run : native.runs) {
        ASSERT_EQ(run.support.size(), 3U);
        EXPECT_TRUE(run.support.back().final_short_support);
        EXPECT_EQ(run.support.back().exact_common_slots,
                  (std::vector<std::size_t>{4}));
        for (std::size_t local_detector = 0;
             local_detector < run.input.detector_columns.size();
             ++local_detector) {
            const auto detector =
                run.input.detector_columns[local_detector];
            actual_values.col(detector) = run.selected_values.col(
                static_cast<Eigen::Index>(local_detector));
            actual_flags.col(detector) = run.ored_flag_bits.col(
                static_cast<Eigen::Index>(local_detector));
        }
    }
    for (Eigen::Index row = 0; row < actual_values.rows(); ++row) {
        for (Eigen::Index detector = 0;
             detector < actual_values.cols(); ++detector) {
            EXPECT_EQ(std::bit_cast<std::uint64_t>(
                          actual_values(row, detector)),
                      std::bit_cast<std::uint64_t>(
                          expected_values(row, detector)));
        }
    }
    EXPECT_TRUE((actual_flags.array() == expected_flags.array()).all());
}

TEST(sci_align_rtc_run_adapter,
     outer_context_is_processed_but_only_inner_rows_are_published) {
    const auto loaded = complete_identical_time_fixture();
    const auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    std::size_t body_calls = 0;
    const auto result = pipeline::dispatch_native_rtc_runs(
        *scan, {1, false, 1U, 4U},
        [&](const pipeline::NativeRtcRunInput &input) {
            ++body_calls;
            EXPECT_EQ(input.measured_values.rows(), 5);
            EXPECT_EQ(input.selected_row_offset(), 1U);
            EXPECT_EQ(input.selected_row_count(), 3U);
            Eigen::MatrixXd selected(3, input.measured_values.cols());
            for (Eigen::Index row = 0; row < selected.rows(); ++row) {
                selected.row(row) =
                    input.measured_values.row(row) +
                    input.measured_values.row(row + 1) +
                    input.measured_values.row(row + 2);
            }
            return pipeline::NativeRtcProcessedRun{
                std::move(selected),
                input.input_flag_bits.middleRows(1, 3)};
        });

    ASSERT_EQ(body_calls, 2U);
    ASSERT_EQ(result.runs.size(), 2U);
    for (const auto &run : result.runs) {
        ASSERT_EQ(run.selected_values.rows(), 3);
        ASSERT_EQ(run.support.size(), 3U);
        EXPECT_EQ(run.support.front().exact_common_slots,
                  (std::vector<std::size_t>{1}));
        EXPECT_EQ(run.support.back().exact_common_slots,
                  (std::vector<std::size_t>{3}));
        for (Eigen::Index row = 0; row < 3; ++row) {
            for (Eigen::Index detector = 0;
                 detector < run.selected_values.cols(); ++detector) {
                EXPECT_DOUBLE_EQ(
                    run.selected_values(row, detector),
                    run.input.measured_values(row, detector) +
                        run.input.measured_values(row + 1, detector) +
                        run.input.measured_values(row + 2, detector));
            }
        }
    }
    const auto cohorts = pipeline::detail::make_native_ptc_rtc_cohort_segments(
        *scan, result);
    ASSERT_EQ(cohorts.size(), 1U);
    ASSERT_EQ(cohorts.front().rows.size(), 3U);
    EXPECT_EQ(cohorts.front().rows.front().exact_common_slots,
              (std::vector<std::size_t>{1}));
    EXPECT_EQ(cohorts.front().rows.back().exact_common_slots,
              (std::vector<std::size_t>{3}));
}

TEST(sci_align_rtc_run_adapter,
     global_cohort_body_runs_before_downsample_and_scatters_by_identity) {
    const auto loaded = complete_identical_time_fixture();
    const auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    int local_calls = 0;
    int cohort_calls = 0;
    const auto result = pipeline::dispatch_native_rtc_runs(
        *scan, {2, false},
        [&](const pipeline::NativeRtcRunInput &input) {
            ++local_calls;
            return identity_body(input);
        },
        pipeline::NativeRtcCohortNumericalBody{
            [&](const pipeline::NativeRtcCohortInput &input) {
                ++cohort_calls;
                auto values = input.values;
                for (Eigen::Index detector = 0;
                     detector < values.cols(); ++detector) {
                    values.col(detector).array() +=
                        1000.0 * static_cast<double>(detector + 1);
                }
                auto flags = input.flag_bits;
                flags(2, 3) |= pipeline::NativeDetectorFlagBits{1};
                Eigen::MatrixXd kernel = values.array() * 0.5;
                return pipeline::NativeRtcProcessedRun{
                    std::move(values), std::move(flags),
                    std::move(kernel)};
            }});

    EXPECT_EQ(local_calls, 0);
    EXPECT_EQ(cohort_calls, 1);
    ASSERT_EQ(result.runs.size(), 2U);
    for (const auto &run : result.runs) {
        ASSERT_TRUE(run.selected_kernel_values.has_value());
        for (std::size_t local = 0;
             local < run.input.detector_columns.size(); ++local) {
            const auto detector = run.input.detector_columns[local];
            for (Eigen::Index row = 0;
                 row < run.selected_values.rows(); ++row) {
                const auto expected = run.input.measured_values(
                    row * 2, static_cast<Eigen::Index>(local)) +
                    1000.0 * static_cast<double>(detector + 1);
                EXPECT_DOUBLE_EQ(
                    run.selected_values(
                        row, static_cast<Eigen::Index>(local)),
                    expected);
                EXPECT_DOUBLE_EQ(
                    (*run.selected_kernel_values)(
                        row, static_cast<Eigen::Index>(local)),
                    0.5 * expected);
            }
            if (detector == 3) {
                EXPECT_NE(
                    run.ored_flag_bits(1, static_cast<Eigen::Index>(local)) &
                        pipeline::NativeDetectorFlagBits{1},
                    pipeline::NativeDetectorFlagBits{0});
            }
        }
    }
}

TEST(sci_align_rtc_run_adapter,
     repeated_results_are_exact_at_openmp_thread_counts_1_2_4_8) {
    const auto loaded = fixture::load_native_gap_fixture_v1();
    const auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    std::optional<ResultSnapshot> reference;
    for (const int thread_count : std::array<int, 4>{1, 2, 4, 8}) {
#ifdef _OPENMP
        omp_set_num_threads(thread_count);
#else
        (void)thread_count;
#endif
        const auto result = pipeline::dispatch_native_rtc_runs(
            *scan, {2, false},
            [](const pipeline::NativeRtcRunInput &input) {
                auto processed = identity_body(input);
                processed.values.array() +=
                    static_cast<double>(input.segment_ordinal * 10) +
                    static_cast<double>(input.run.network_id);
                for (Eigen::Index row = 0;
                     row < processed.flag_bits.rows(); ++row) {
                    for (Eigen::Index detector = 0;
                         detector < processed.flag_bits.cols();
                         ++detector) {
                        processed.flag_bits(row, detector) |= 0x8000U;
                    }
                }
                return processed;
            });
        const auto current = snapshot(result);
        if (!reference.has_value()) reference = current;
        EXPECT_EQ(current, *reference);
    }
}

TEST(sci_align_rtc_run_adapter,
     rejected_runs_do_not_invoke_body_or_mutate_scan_ledger) {
    auto loaded = fixture::load_native_gap_fixture_v1();
    auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    pipeline::NativeMeasuredDetectorLedger ledger{scan};
    std::size_t body_calls = 0;
    auto body = [&](const pipeline::NativeRtcRunInput &input) {
        ++body_calls;
        return identity_body(input);
    };

    EXPECT_THROW(
        pipeline::dispatch_native_rtc_runs(*scan, {2, true}, body),
        std::logic_error);
    EXPECT_EQ(body_calls, 0U);
    EXPECT_FALSE(ledger.last_operation().has_value());
    EXPECT_THROW(
        pipeline::dispatch_native_rtc_runs(*scan, {0, false}, body),
        std::invalid_argument);
    EXPECT_EQ(body_calls, 0U);
    EXPECT_FALSE(ledger.last_operation().has_value());

    auto &network7 = loaded.network(7);
    network7.measured_values(0, 0) =
        std::numeric_limits<double>::quiet_NaN();
    scan = fixture::materialize_native_gap_measured_scan(loaded);
    pipeline::NativeMeasuredDetectorLedger invalid_ledger{scan};
    EXPECT_THROW(
        pipeline::dispatch_native_rtc_runs(*scan, {2, false}, body),
        std::logic_error);
    EXPECT_EQ(body_calls, 0U);
    EXPECT_FALSE(invalid_ledger.last_operation().has_value());

    EXPECT_THROW(
        pipeline::dispatch_native_rtc_runs(
            *scan, {1, false, 0U, scan->past_last_common_slot() + 1},
            body),
        std::invalid_argument);
}

TEST(sci_align_rtc_run_adapter,
     numerical_body_cannot_change_shape_drop_flags_or_emit_nonfinite_values) {
    const auto loaded = fixture::load_native_gap_fixture_v1();
    const auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    EXPECT_THROW(
        pipeline::dispatch_native_rtc_runs(
            *scan, {2, false},
            [](const pipeline::NativeRtcRunInput &input) {
                auto result = identity_body(input);
                result.values.conservativeResize(
                    result.values.rows() + 1, result.values.cols());
                return result;
            }),
        std::logic_error);
    EXPECT_THROW(
        pipeline::dispatch_native_rtc_runs(
            *scan, {2, false},
            [](const pipeline::NativeRtcRunInput &input) {
                auto result = identity_body(input);
                result.flag_bits.setZero();
                return result;
            }),
        std::logic_error);
    EXPECT_THROW(
        pipeline::dispatch_native_rtc_runs(
            *scan, {2, false},
            [](const pipeline::NativeRtcRunInput &input) {
                auto result = identity_body(input);
                result.values(0, 0) =
                    std::numeric_limits<double>::infinity();
                return result;
            }),
        std::logic_error);
}

}  // namespace
