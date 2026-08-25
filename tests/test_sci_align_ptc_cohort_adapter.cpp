#include "sci_align_native_gap_fixture.h"

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/pipeline/timestream_ptc_cohort_adapter.h>
#include <citlali/core/timestream/ptc/clean.h>

#include <gtest/gtest.h>
#include <spdlog/sinks/null_sink.h>

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

fixture::NativeGapFixtureV1 complete_identical_time_fixture() {
    auto result = fixture::load_native_gap_fixture_v1();
    auto &network7 = result.network(7);
    network7.reconstructed_times_unix_sec =
        result.common_slot_reference_times_unix_sec;
    network7.packet_counters = {700, 701, 702, 703, 704};
    network7.legacy_presence_mask = Eigen::VectorXi::Ones(5);
    network7.expected_slot_native_rows = {700, 701, 702, 703, 704};
    network7.measured_values.resize(5, 2);
    network7.measured_values <<
        7.1, 8.3,
        7.8, 9.4,
        9.2, 10.1,
        10.4, 11.8,
        12.7, 13.2;
    network7.original_flag_bits =
        pipeline::NativeDetectorFlagBitsMatrix::Zero(5, 2);
    result.network(0).original_flag_bits =
        pipeline::NativeDetectorFlagBitsMatrix::Zero(5, 2);
    network7.expected_packet_contiguous_runs = {{700, 705}};
    result.expected_complete_cohort_slot_runs = {{0, 5}};
    return result;
}

pipeline::NativeRtcProcessedRun identity_rtc_body(
    const pipeline::NativeRtcRunInput &input) {
    return {input.measured_values, input.input_flag_bits};
}

pipeline::NativeRtcDispatchResult identity_rtc(
    const pipeline::NativeMeasuredDetectorScan &scan,
    int downsample_factor = 1) {
    return pipeline::dispatch_native_rtc_runs(
        scan, {downsample_factor, false}, identity_rtc_body);
}

pipeline::NativeRtcDispatchResult offset_rtc(
    const pipeline::NativeMeasuredDetectorScan &scan, double offset) {
    return pipeline::dispatch_native_rtc_runs(
        scan, {1, false},
        [offset](const pipeline::NativeRtcRunInput &input) {
            return pipeline::NativeRtcProcessedRun{
                (input.measured_values.array() + offset).matrix(),
                input.input_flag_bits};
        });
}

pipeline::NativePtcCohortRequest request_for(
    std::string grouping, double placeholder) {
    return {
        std::move(grouping),
        pipeline::FinitePcaPlaceholder::checked(placeholder),
        {},
        {}, false, false};
}

const pipeline::NativePtcGroupWorkingSet &group_with_key(
    const pipeline::NativePtcPreparedOperation &prepared,
    std::size_t segment_ordinal, std::int64_t key) {
    const auto found = std::find_if(
        prepared.groups().begin(), prepared.groups().end(),
        [&](const auto &group) {
            return group.segment_ordinal() == segment_ordinal &&
                   group.group_key() == key;
        });
    if (found == prepared.groups().end()) {
        throw std::out_of_range("PTC test group key is absent");
    }
    return *found;
}

Eigen::MatrixXd placeholder_ignoring_body(
    const pipeline::NativePtcGroupWorkingSet &group) {
    Eigen::MatrixXd result = group.values();
    for (Eigen::Index detector = 0;
         detector < group.detector_count(); ++detector) {
        double sum = 0.0;
        std::size_t count = 0;
        for (Eigen::Index row = 0; row < group.slot_count(); ++row) {
            if (!group.exclusion_flags()(row, detector)) {
                sum += group.values()(row, detector);
                ++count;
            }
        }
        const double mean = count == 0 ? 0.0 : sum / count;
        for (Eigen::Index row = 0; row < group.slot_count(); ++row) {
            if (!group.exclusion_flags()(row, detector)) {
                result(row, detector) -= mean;
            }
        }
    }
    return result;
}

std::shared_ptr<spdlog::logger> ensure_sci_align_logger() {
    auto logger = spdlog::get("citlali_logger");
    if (logger == nullptr) {
        auto sink = std::make_shared<spdlog::sinks::null_sink_mt>();
        logger = std::make_shared<spdlog::logger>(
            "citlali_logger", sink);
        spdlog::register_logger(logger);
    }
    return logger;
}

Eigen::MatrixXd ordinary_pca_body(
    timestream::Cleaner &cleaner,
    const pipeline::NativePtcGroupWorkingSet &group) {
    constexpr Eigen::Index cut = 1;
    Eigen::VectorXi apt_exclusion_flags =
        group.apt_exclusion_flags();
    const auto [eigenvalues, eigenvectors] =
        cleaner.calc_eig_values<timestream::Cleaner::SpectraBackend>(
            group.values(), group.exclusion_flags(),
            apt_exclusion_flags,
            cut);
    Eigen::MatrixXd result = group.values();
    cleaner.remove_eig_values<timestream::Cleaner::SpectraBackend>(
        group.values(), group.exclusion_flags(), eigenvalues,
        eigenvectors, result, cut, -1,
        group.effective_grouping(),
        group.effective_grouping() == "nw" ? group.group_key() : -1,
        group.effective_grouping() == "array" ? group.group_key() : 0);
    return result;
}

TEST(sci_align_ptc_cohort_adapter,
     typed_interleaved_memberships_and_private_placeholders_are_exact) {
    const auto loaded = fixture::load_native_gap_fixture_v1();
    const auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    const auto rtc = offset_rtc(*scan, 11.0);

    pipeline::NativeMeasuredDetectorLedger nw_ledger{scan};
    const auto nw = pipeline::prepare_native_ptc_cohorts(
        nw_ledger, rtc, request_for("nw", -31.0));
    ASSERT_EQ(nw.segment_count(), 2U);
    ASSERT_EQ(nw.groups().size(), 4U);
    for (std::size_t segment = 0; segment < 2; ++segment) {
        EXPECT_EQ(group_with_key(nw, segment, 0).detector_columns(),
                  (std::vector<pipeline::TimestreamDetectorColumn>{1, 3}));
        EXPECT_EQ(group_with_key(nw, segment, 7).detector_columns(),
                  (std::vector<pipeline::TimestreamDetectorColumn>{0, 2}));
    }

    pipeline::NativeMeasuredDetectorLedger array_ledger{scan};
    const auto array = pipeline::prepare_native_ptc_cohorts(
        array_ledger, rtc, request_for("array", -31.0));
    ASSERT_EQ(array.groups().size(), 4U);
    for (std::size_t segment = 0; segment < 2; ++segment) {
        EXPECT_EQ(group_with_key(array, segment, 0).detector_columns(),
                  (std::vector<pipeline::TimestreamDetectorColumn>{1, 3}));
        EXPECT_EQ(group_with_key(array, segment, 1).detector_columns(),
                  (std::vector<pipeline::TimestreamDetectorColumn>{0, 2}));
    }

    pipeline::NativeMeasuredDetectorLedger low_ledger{scan};
    pipeline::NativeMeasuredDetectorLedger high_ledger{scan};
    const auto low = pipeline::prepare_native_ptc_cohorts(
        low_ledger, rtc, request_for("all", -1000.0));
    const auto high = pipeline::prepare_native_ptc_cohorts(
        high_ledger, rtc, request_for("all", 900000.0));
    ASSERT_EQ(low.groups().size(), 2U);
    ASSERT_EQ(high.groups().size(), 2U);
    const std::array<std::vector<std::size_t>, 2> expected_slots{
        std::vector<std::size_t>{0, 1},
        std::vector<std::size_t>{3, 4}};
    for (std::size_t segment = 0; segment < 2; ++segment) {
        const auto &low_group = group_with_key(low, segment, 0);
        const auto &high_group = group_with_key(high, segment, 0);
        ASSERT_EQ(low_group.slot_count(), 2);
        EXPECT_TRUE((low_group.exclusion_flags().array() ==
                     high_group.exclusion_flags().array()).all());
        for (Eigen::Index row = 0; row < low_group.slot_count(); ++row) {
            EXPECT_EQ(low_group.cell(row, 0).exact_common_slots,
                      (std::vector<std::size_t>{
                          expected_slots[segment][
                              static_cast<std::size_t>(row)]}));
            for (Eigen::Index detector = 0;
                 detector < low_group.detector_count(); ++detector) {
                if (low_group.exclusion_flags()(row, detector)) {
                    EXPECT_DOUBLE_EQ(low_group.values()(row, detector),
                                     -1000.0);
                    EXPECT_DOUBLE_EQ(high_group.values()(row, detector),
                                     900000.0);
                }
                else {
                    EXPECT_DOUBLE_EQ(low_group.values()(row, detector),
                                     high_group.values()(row, detector));
                }
            }
        }
    }

    const auto low_result = pipeline::run_native_ptc_groups(
        low, placeholder_ignoring_body);
    const auto high_result = pipeline::run_native_ptc_groups(
        high, placeholder_ignoring_body);
    pipeline::scatter_native_ptc_results_transactionally(
        low_ledger, low, low_result);
    pipeline::scatter_native_ptc_results_transactionally(
        high_ledger, high, high_result);
    for (const auto &group : low.groups()) {
        for (Eigen::Index row = 0; row < group.slot_count(); ++row) {
            for (Eigen::Index local = 0;
                 local < group.detector_count(); ++local) {
                const auto &cell = group.cell(row, local);
                if (cell.state !=
                    pipeline::CoincidenceCellState::mapped_invalid) {
                    continue;
                }
                const auto detector = group.detector_columns().at(
                    static_cast<std::size_t>(local));
                const auto record = low_ledger.record(
                    {cell.identity->key(), detector});
                EXPECT_DOUBLE_EQ(record.current_value,
                                 cell.preserved_input_value);
            }
        }
    }
    std::set<pipeline::NativeDetectorSampleKey> anchor_keys;
    for (const auto &run : rtc.runs) {
        for (const auto &support : run.support) {
            for (const auto detector : run.input.detector_columns) {
                anchor_keys.insert(
                    {support.selected_anchor.key(), detector});
            }
        }
    }
    for (std::size_t slot = 0; slot < scan->relational_slot_count(); ++slot) {
        for (std::size_t detector = 0; detector < scan->detector_count();
             ++detector) {
            const auto cell = scan->cell(
                scan->first_common_slot() + slot,
                static_cast<Eigen::Index>(detector));
            if (!cell.mapped()) continue;
            const pipeline::NativeDetectorSampleKey key{
                cell.identity()->key(), static_cast<Eigen::Index>(detector)};
            const auto low_record = low_ledger.record(key);
            const auto high_record = high_ledger.record(key);
            EXPECT_EQ(std::bit_cast<std::uint64_t>(low_record.current_value),
                      std::bit_cast<std::uint64_t>(high_record.current_value));
            const auto expected_revision = anchor_keys.contains(key) ? 1U : 0U;
            EXPECT_EQ(low_record.revision, expected_revision);
            EXPECT_EQ(high_record.revision, expected_revision);
        }
    }
}

TEST(sci_align_ptc_cohort_adapter,
     corr_nw_uses_noncontiguous_memberships_and_passes_ungrouped_columns) {
    const auto loaded = complete_identical_time_fixture();
    const auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    const auto rtc = offset_rtc(*scan, 2.0);
    pipeline::NativeMeasuredDetectorLedger ledger{scan};
    auto request = request_for("corr_nw", 0.0);
    request.corr_grouping_enabled = true;
    std::size_t grouping_calls = 0;
    const auto prepared = pipeline::prepare_native_ptc_cohorts(
        ledger, rtc, request,
        pipeline::NativePtcCorrGroupingBody{
            [&](const pipeline::NativePtcGroupWorkingSet &base) {
                ++grouping_calls;
                if (base.group_key() == 0) {
                    return pipeline::NativePtcCorrGroupingBody::Groups{
                        {0, 1}};
                }
                return pipeline::NativePtcCorrGroupingBody::Groups{};
            }});
    EXPECT_EQ(grouping_calls, 2U);
    ASSERT_EQ(prepared.groups().size(), 2U);
    EXPECT_EQ(prepared.groups()[0].detector_columns(),
              (std::vector<pipeline::TimestreamDetectorColumn>{1, 3}));
    EXPECT_EQ(prepared.groups()[0].role(),
              pipeline::NativePtcGroupRole::pca_clean);
    EXPECT_EQ(prepared.groups()[1].detector_columns(),
              (std::vector<pipeline::TimestreamDetectorColumn>{0, 2}));
    EXPECT_EQ(prepared.groups()[1].role(),
              pipeline::NativePtcGroupRole::pass_through);

    std::size_t cleaner_calls = 0;
    const auto processed = pipeline::run_native_ptc_groups(
        prepared, [&](const pipeline::NativePtcGroupWorkingSet &group) {
            ++cleaner_calls;
            return (group.values().array() + 5.0).matrix();
        });
    EXPECT_EQ(cleaner_calls, 1U);
    pipeline::scatter_native_ptc_results_transactionally(
        ledger, prepared, processed);
    for (std::size_t slot = 0; slot < 5; ++slot) {
        const auto pass = scan->cell(slot, 0);
        const auto clean = scan->cell(slot, 1);
        const auto pass_record = ledger.record(
            {pass.identity()->key(), 0});
        const auto clean_record = ledger.record(
            {clean.identity()->key(), 1});
        EXPECT_DOUBLE_EQ(pass_record.current_value,
                         *pass.measured_value() + 2.0);
        EXPECT_DOUBLE_EQ(clean_record.current_value,
                         *clean.measured_value() + 7.0);
        EXPECT_EQ(pass_record.revision, 1U);
        EXPECT_EQ(clean_record.revision, 1U);
    }
}

TEST(sci_align_ptc_cohort_adapter,
     optional_modes_and_incomplete_second_pass_cohorts_fail_before_operation) {
    const auto loaded = complete_identical_time_fixture();
    const auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    const auto rtc = identity_rtc(*scan);
    pipeline::NativeMeasuredDetectorLedger ledger{scan};

    auto incomplete_rtc = rtc;
    incomplete_rtc.runs.pop_back();
    EXPECT_THROW(
        pipeline::prepare_native_ptc_cohorts(
            ledger, incomplete_rtc, request_for("all", 0.0)),
        std::logic_error);
    EXPECT_FALSE(ledger.last_operation().has_value());

    auto request = request_for("corr_nw", 0.0);
    request.corr_grouping_enabled = true;
    const auto &first_run = rtc.runs.front();
    request.operation_exclusion_bits.emplace(
        pipeline::NativeDetectorSampleKey{
            first_run.support.front().selected_anchor.key(),
            first_run.input.detector_columns.front()},
        0x20U);
    request.optional_modes.null_model_active_for_operation = true;
    std::size_t grouping_calls = 0;
    EXPECT_THROW(
        pipeline::prepare_native_ptc_cohorts(
            ledger, rtc, request,
            pipeline::NativePtcCorrGroupingBody{
                [&](const auto &) {
                    ++grouping_calls;
                    return pipeline::NativePtcCorrGroupingBody::Groups{};
                }}),
        std::logic_error);
    EXPECT_EQ(grouping_calls, 0U);
    EXPECT_FALSE(ledger.last_operation().has_value());

    request.optional_modes = {};
    request.requires_second_pass_window = true;
    EXPECT_THROW(
        pipeline::prepare_native_ptc_cohorts(ledger, rtc, request),
        std::logic_error);
    EXPECT_FALSE(ledger.last_operation().has_value());

    auto complete_network = request_for("nw", 0.0);
    complete_network.requires_second_pass_window = true;
    EXPECT_NO_THROW(
        pipeline::prepare_native_ptc_cohorts(
            ledger, rtc, complete_network));
    EXPECT_TRUE(ledger.last_operation().has_value());

    auto unsupported = request_for("fg", 0.0);
    pipeline::NativeMeasuredDetectorLedger unsupported_ledger{scan};
    EXPECT_THROW(
        pipeline::prepare_native_ptc_cohorts(
            unsupported_ledger, rtc, unsupported),
        std::invalid_argument);
    EXPECT_FALSE(unsupported_ledger.last_operation().has_value());
}

TEST(sci_align_ptc_cohort_adapter,
     numerical_flags_are_append_only_and_survive_the_processed_group) {
    const auto loaded = complete_identical_time_fixture();
    const auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    const auto rtc = identity_rtc(*scan);
    pipeline::NativeMeasuredDetectorLedger ledger{scan};
    const auto prepared = pipeline::prepare_native_ptc_cohorts(
        ledger, rtc, request_for("nw", 0.0));

    const auto processed = pipeline::run_native_ptc_groups(
        prepared, [](const pipeline::NativePtcGroupWorkingSet &group) {
            auto flags = group.exclusion_flags();
            bool added = false;
            for (Eigen::Index row = 0; row < flags.rows() && !added; ++row) {
                for (Eigen::Index detector = 0;
                     detector < flags.cols(); ++detector) {
                    if (!flags(row, detector)) {
                        flags(row, detector) = true;
                        added = true;
                        break;
                    }
                }
            }
            if (!added) {
                throw std::logic_error(
                    "PTC append-only test fixture has no eligible sample");
            }
            return pipeline::NativePtcNumericalResult{
                group.values(), group.kernel_values(),
                group.exclusion_flags(), std::move(flags)};
        });
    ASSERT_EQ(processed.groups().size(), prepared.groups().size());
    for (std::size_t index = 0; index < processed.groups().size(); ++index) {
        ASSERT_TRUE(processed.groups()[index].exclusion_flags());
        EXPECT_GT(
            processed.groups()[index].exclusion_flags()->array().count(),
            prepared.groups()[index].exclusion_flags().array().count());
    }

    EXPECT_THROW(
        pipeline::run_native_ptc_groups(
            prepared,
            [](const pipeline::NativePtcGroupWorkingSet &group) {
                auto flags = group.exclusion_flags();
                bool removed = false;
                for (Eigen::Index row = 0;
                     row < flags.rows() && !removed; ++row) {
                    for (Eigen::Index detector = 0;
                         detector < flags.cols(); ++detector) {
                        if (flags(row, detector)) {
                            flags(row, detector) = false;
                            removed = true;
                            break;
                        }
                    }
                }
                if (!removed) {
                    throw std::logic_error(
                        "PTC append-only test fixture has no excluded sample");
                }
                return pipeline::NativePtcNumericalResult{
                    group.values(), group.kernel_values(),
                    group.exclusion_flags(), std::move(flags)};
            }),
        std::logic_error);
}

TEST(sci_align_ptc_cohort_adapter,
     stale_duplicate_and_nonfinite_scatter_reject_atomically_and_retry) {
    const auto loaded = complete_identical_time_fixture();
    const auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    const auto rtc = identity_rtc(*scan);
    pipeline::NativeMeasuredDetectorLedger ledger{scan};
    const auto prepared = pipeline::prepare_native_ptc_cohorts(
        ledger, rtc, request_for("all", 0.0));
    auto processed = pipeline::run_native_ptc_groups(
        prepared, [](const auto &group) { return group.values(); });
    const auto first_cell = scan->cell(0, 0);
    const pipeline::NativeDetectorSampleKey first_key{
        first_cell.identity()->key(), 0};
    processed.mutable_group_for_retry(0)
        .mutable_values_for_retry()(0, 0) =
        std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        pipeline::scatter_native_ptc_results_transactionally(
            ledger, prepared, processed),
        std::logic_error);
    EXPECT_FALSE(ledger.last_committed_operation().has_value());
    EXPECT_EQ(ledger.record(first_key).revision, 0U);

    processed.mutable_group_for_retry(0)
        .mutable_values_for_retry()(0, 0) = 123.5;
    pipeline::scatter_native_ptc_results_transactionally(
        ledger, prepared, processed);
    EXPECT_DOUBLE_EQ(ledger.record(first_key).current_value, 123.5);
    EXPECT_EQ(ledger.record(first_key).revision, 1U);
    ASSERT_TRUE(ledger.last_committed_operation().has_value());
    EXPECT_EQ(ledger.last_committed_operation()->sequence, 0U);

    EXPECT_THROW(
        pipeline::scatter_native_ptc_results_transactionally(
            ledger, prepared, processed),
        std::logic_error);
    EXPECT_DOUBLE_EQ(ledger.record(first_key).current_value, 123.5);
    EXPECT_EQ(ledger.record(first_key).revision, 1U);

    pipeline::NativeMeasuredDetectorLedger duplicate_ledger{scan};
    const auto operation = duplicate_ledger.issue_operation();
    const auto identity = scan->sample_identity(first_key);
    const auto update =
        pipeline::NativeMeasuredDetectorLedger::Update::replacement(
            identity, 0, 0, 44.0);
    EXPECT_THROW(
        duplicate_ledger.apply_transaction(operation, {update, update}),
        std::logic_error);
    EXPECT_FALSE(duplicate_ledger.last_committed_operation().has_value());
    EXPECT_EQ(duplicate_ledger.record(first_key).revision, 0U);
    duplicate_ledger.apply_transaction(operation, {update});
    EXPECT_DOUBLE_EQ(duplicate_ledger.record(first_key).current_value,
                     44.0);
}

TEST(sci_align_ptc_cohort_adapter,
     identical_times_match_existing_rectangular_ordinary_pca_exactly) {
    const auto loaded = complete_identical_time_fixture();
    const auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    const auto rtc = identity_rtc(*scan);
    pipeline::NativeMeasuredDetectorLedger ledger{scan};
    const auto prepared = pipeline::prepare_native_ptc_cohorts(
        ledger, rtc, request_for("nw", -77.0));

    timestream::Cleaner native_cleaner;
    native_cleaner.logger = ensure_sci_align_logger();
    native_cleaner.stddev_limit = 0.0;
    native_cleaner.n_calc = 0;
    native_cleaner.standard_pca.enabled = true;
    const auto processed = pipeline::run_native_ptc_groups(
        prepared, [&](const auto &group) {
            return ordinary_pca_body(native_cleaner, group);
        });

    Eigen::MatrixXd expected(5, 4);
    for (Eigen::Index row = 0; row < 5; ++row) {
        for (Eigen::Index detector = 0; detector < 4; ++detector) {
            expected(row, detector) =
                *scan->cell(static_cast<std::size_t>(row), detector)
                     .measured_value();
        }
    }
    timestream::Cleaner legacy_cleaner = native_cleaner;
    for (const auto &group : prepared.groups()) {
        const auto cleaned = ordinary_pca_body(legacy_cleaner, group);
        for (Eigen::Index local = 0; local < group.detector_count();
             ++local) {
            expected.col(group.detector_columns().at(
                static_cast<std::size_t>(local))) = cleaned.col(local);
        }
    }

    pipeline::scatter_native_ptc_results_transactionally(
        ledger, prepared, processed);
    for (Eigen::Index row = 0; row < 5; ++row) {
        for (Eigen::Index detector = 0; detector < 4; ++detector) {
            const auto cell = scan->cell(
                static_cast<std::size_t>(row), detector);
            const auto record = ledger.record(
                {cell.identity()->key(), detector});
            EXPECT_EQ(std::bit_cast<std::uint64_t>(record.current_value),
                      std::bit_cast<std::uint64_t>(
                          expected(row, detector)));
            EXPECT_EQ(record.revision, 1U);
        }
    }
}

using DeterminismRow =
    std::tuple<pipeline::NativeSampleKey,
               pipeline::TimestreamDetectorColumn, std::uint64_t,
               pipeline::TimestreamNativeRevision>;

TEST(sci_align_ptc_cohort_adapter,
     repeated_results_are_exact_at_openmp_thread_counts_1_2_4_8) {
    const auto loaded = complete_identical_time_fixture();
    const auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    const auto rtc = identity_rtc(*scan);
    std::optional<std::vector<DeterminismRow>> reference;
    for (const int thread_count : std::array<int, 4>{1, 2, 4, 8}) {
#ifdef _OPENMP
        omp_set_num_threads(thread_count);
#else
        (void)thread_count;
#endif
        pipeline::NativeMeasuredDetectorLedger ledger{scan};
        const auto prepared = pipeline::prepare_native_ptc_cohorts(
            ledger, rtc, request_for("array", 0.0));
        const auto processed = pipeline::run_native_ptc_groups(
            prepared, [](const auto &group) {
                Eigen::MatrixXd result = group.values();
                for (Eigen::Index row = 0; row < result.rows(); ++row) {
                    for (Eigen::Index detector = 0;
                         detector < result.cols(); ++detector) {
                        result(row, detector) +=
                            static_cast<double>(group.group_key() * 10) +
                            static_cast<double>(row + detector) / 16.0;
                    }
                }
                return result;
            });
        pipeline::scatter_native_ptc_results_transactionally(
            ledger, prepared, processed);
        std::vector<DeterminismRow> current;
        for (std::size_t slot = 0; slot < 5; ++slot) {
            for (Eigen::Index detector = 0; detector < 4; ++detector) {
                const auto cell = scan->cell(slot, detector);
                const auto record = ledger.record(
                    {cell.identity()->key(), detector});
                current.emplace_back(
                    cell.identity()->key(), detector,
                    std::bit_cast<std::uint64_t>(record.current_value),
                    record.revision);
            }
        }
        if (!reference.has_value()) reference = current;
        EXPECT_EQ(current, *reference);
    }
}

}  // namespace
