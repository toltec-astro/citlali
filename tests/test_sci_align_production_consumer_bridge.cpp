#include <citlali/core/pipeline/timestream_alignment_helpers.h>
#include <citlali/core/pipeline/timestream_native_consumer_bridge.h>
#include <citlali/core/pipeline/product_index_file.h>
#include <kids/toltec/toltec.h>
#include <citlali/core/timestream/ptc/ptcproc.h>
#include <citlali/core/timestream/rtc/rtcproc.h>

#include <gtest/gtest.h>
#include <spdlog/sinks/null_sink.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using citlali::pipeline::CoincidenceCellState;
using citlali::pipeline::CoincidenceAbsenceReason;
using citlali::pipeline::FinitePcaPlaceholder;
using citlali::pipeline::NativeAlignmentPlan;
using citlali::pipeline::NativeCohortSelection;
using citlali::pipeline::NativeDetectorBlock;
using citlali::pipeline::NativeDetectorFlagBits;
using citlali::pipeline::NativeDetectorFlagBitsMatrix;
using citlali::pipeline::NativeDetectorLedger;
using citlali::pipeline::NativeDetectorCoincidenceProvenance;
using citlali::pipeline::NativeDetectorRevisionAction;
using citlali::pipeline::NativeNetworkAlignment;
using citlali::pipeline::NativeOperationIdentity;
using citlali::pipeline::NativeSampleKey;
using citlali::pipeline::NativePreparedPcaOperation;
using citlali::pipeline::NativePreparedPcaGroupRole;
using citlali::pipeline::NativeSlotAssociation;
using citlali::pipeline::PcaCompatibilityHazard;
using citlali::pipeline::PcaCompatibilityInputs;
using citlali::pipeline::classify_native_detector_pca_compatibility;
using citlali::pipeline::packet_counters_from_timestream_matrix;
using citlali::pipeline::native_detector_revision_record_equal;
using citlali::pipeline::partition_native_contiguous_runs;
using citlali::pipeline::require_native_detector_pca_compatibility;
using citlali::pipeline::scatter_native_detector_pca_results_transactionally;
using citlali::pipeline::seed_native_detector_ledger;

std::shared_ptr<spdlog::logger> sci_align_production_logger() {
    auto logger = spdlog::get("citlali_logger");
    if (logger == nullptr) {
        auto sink = std::make_shared<spdlog::sinks::null_sink_mt>();
        logger =
            std::make_shared<spdlog::logger>("citlali_logger", sink);
        spdlog::register_logger(logger);
    }
    return logger;
}

std::uint64_t bits_of(double value) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

Eigen::MatrixXd make_ts(const std::vector<int> &nominal_slots) {
    Eigen::MatrixXd ts =
        Eigen::MatrixXd::Zero(
            static_cast<Eigen::Index>(nominal_slots.size()), 6);
    for (Eigen::Index row = 0; row < ts.rows(); ++row) {
        const auto slot = nominal_slots.at(static_cast<std::size_t>(row));
        // The production reconstruction applies the established second
        // boundary convention before adding the FPGA tick fraction.  A
        // delivered second field of 1001 therefore reconstructs the intended
        // 1000.0 + tick/1000 test timeline.
        ts(row, 0) = 1001.0;
        ts(row, 1) = 0.0;
        ts(row, 2) = static_cast<double>(slot * 10);
        ts(row, 3) = static_cast<double>(100 + slot);
        ts(row, 4) = 0.0;
        ts(row, 5) = 0.0;
    }
    return ts;
}

struct ProductionFixture {
    NativeAlignmentPlan plan;
    NativeCohortSelection selection;
    std::vector<NativeDetectorBlock> blocks;
};

ProductionFixture make_production_fixture() {
    std::vector<int> nw0_slots;
    std::vector<int> nw7_slots;
    for (int slot = 0; slot < 14; ++slot) {
        nw0_slots.push_back(slot);
        if (slot != 6) {
            nw7_slots.push_back(slot);
        }
    }
    const auto ts0 = make_ts(nw0_slots);
    const auto ts7 = make_ts(nw7_slots);
    const auto time0 =
        citlali::pipeline::network_time_from_timestream_matrix(
            ts0, 1000.0, 0.0);
    const auto time7 =
        citlali::pipeline::network_time_from_timestream_matrix(
            ts7, 1000.0, 0.0025);

    NativeNetworkAlignment network0{
        0, 100, time0, packet_counters_from_timestream_matrix(ts0)};
    NativeNetworkAlignment network7{
        7, 700, time7, packet_counters_from_timestream_matrix(ts7)};
    Eigen::VectorXd common(14);
    for (Eigen::Index slot = 0; slot < common.size(); ++slot) {
        common(slot) = 1000.0025 + 0.01 * static_cast<double>(slot);
    }
    Eigen::VectorXi mask0 = Eigen::VectorXi::Ones(14);
    Eigen::VectorXi mask7 = Eigen::VectorXi::Ones(14);
    mask7(6) = 0;
    NativeAlignmentPlan plan{
        {network0, network7}, common,
        {citlali::pipeline::make_gap_native_slot_associations(
             network0, common, mask0, 1000.0025, 0.01, 0.0049),
         citlali::pipeline::make_gap_native_slot_associations(
             network7, common, mask7, 1000.0025, 0.01, 0.0049)}};
    auto selection = plan.select_cohort(
        NativeOperationIdentity{41, 3}, 0, 14, 0);

    Eigen::MatrixXd values0(14, 2);
    NativeDetectorFlagBitsMatrix flags0 =
        NativeDetectorFlagBitsMatrix::Zero(14, 2);
    for (Eigen::Index row = 0; row < values0.rows(); ++row) {
        values0(row, 0) =
            std::sin(0.19 * static_cast<double>(row)) + 0.03;
        values0(row, 1) =
            std::cos(0.17 * static_cast<double>(row)) - 0.07;
    }
    flags0(8, 1) = 0x4U;

    Eigen::MatrixXd values7(13, 2);
    NativeDetectorFlagBitsMatrix flags7 =
        NativeDetectorFlagBitsMatrix::Zero(13, 2);
    for (Eigen::Index row = 0; row < values7.rows(); ++row) {
        const double x = static_cast<double>(nw7_slots.at(
            static_cast<std::size_t>(row)));
        values7(row, 0) = std::sin(0.13 * x) + 0.11;
        values7(row, 1) = std::cos(0.21 * x) - 0.02;
    }

    std::vector<NativeDetectorBlock> blocks;
    blocks.emplace_back(network0, 100, 0, values0, flags0);
    blocks.emplace_back(network7, 700, 2, values7, flags7);
    return ProductionFixture{
        std::move(plan), std::move(selection), std::move(blocks)};
}

struct FakeCalib {
    std::map<std::string, Eigen::VectorXd> apt;
    Eigen::VectorXi nws;
    Eigen::VectorXi arrays;
    Eigen::Index n_dets = 4;
    Eigen::Index n_nws = 2;
    Eigen::Index n_arrays = 2;
};

FakeCalib make_fake_calib() {
    FakeCalib calib;
    calib.apt["nw"].resize(4);
    calib.apt["nw"] << 0.0, 0.0, 7.0, 7.0;
    calib.apt["array"].resize(4);
    calib.apt["array"] << 0.0, 0.0, 1.0, 1.0;
    calib.apt["flag"] = Eigen::Vector4d::Zero();
    calib.apt["uid"].resize(4);
    calib.apt["uid"] << 1000.0, 1001.0, 7000.0, 7001.0;
    calib.nws.resize(2);
    calib.nws << 0, 7;
    calib.arrays.resize(2);
    calib.arrays << 0, 1;
    return calib;
}

struct CorrFixture {
    NativeAlignmentPlan plan;
    NativeCohortSelection selection;
    std::vector<NativeDetectorBlock> blocks;
    FakeCalib calib;
};

CorrFixture make_corr_fixture() {
    constexpr Eigen::Index rows = 240;
    constexpr Eigen::Index detectors = 7;
    Eigen::VectorXd times(rows);
    std::vector<citlali::pipeline::TimestreamPacketCounter> counters;
    counters.reserve(static_cast<std::size_t>(rows));
    for (Eigen::Index row = 0; row < rows; ++row) {
        times(row) = 2000.0 + 0.01 * static_cast<double>(row);
        counters.push_back(1000 + row);
    }
    NativeNetworkAlignment network{0, 300, times, std::move(counters)};
    NativeAlignmentPlan plan{
        {network}, times,
        {citlali::pipeline::make_direct_native_slot_associations(300, rows)}};
    auto selection = plan.select_cohort(
        NativeOperationIdentity{61, 5}, 0,
        static_cast<std::size_t>(rows), 0);

    Eigen::MatrixXd values(rows, detectors);
    for (Eigen::Index row = 0; row < rows; ++row) {
        const double x = static_cast<double>(row);
        const double a = std::sin(0.031 * x) + 0.07 * std::cos(0.013 * x);
        const double b = std::cos(0.027 * x) - 0.04 * std::sin(0.019 * x);
        values(row, 0) = a;
        values(row, 1) = b;
        values(row, 2) = a + 0.001 * std::sin(0.41 * x);
        values(row, 3) = b + 0.001 * std::cos(0.37 * x);
        values(row, 4) = a - 0.001 * std::cos(0.29 * x);
        values(row, 5) = b - 0.001 * std::sin(0.43 * x);
        values(row, 6) = std::sin(0.173 * x) + 0.23 * std::cos(0.097 * x);
    }
    NativeDetectorFlagBitsMatrix flags =
        NativeDetectorFlagBitsMatrix::Zero(rows, detectors);
    std::vector<NativeDetectorBlock> blocks;
    blocks.emplace_back(network, 300, 0, std::move(values), flags);

    FakeCalib calib;
    calib.n_dets = detectors;
    calib.n_nws = 1;
    calib.n_arrays = 1;
    calib.apt["nw"] = Eigen::VectorXd::Zero(detectors);
    calib.apt["array"] = Eigen::VectorXd::Zero(detectors);
    calib.apt["flag"] = Eigen::VectorXd::Zero(detectors);
    calib.apt["uid"].resize(detectors);
    for (Eigen::Index detector = 0; detector < detectors; ++detector) {
        calib.apt["uid"](detector) = 9100.0 + detector;
    }
    calib.nws = Eigen::VectorXi::Zero(1);
    calib.arrays = Eigen::VectorXi::Zero(1);
    return CorrFixture{
        std::move(plan), std::move(selection), std::move(blocks),
        std::move(calib)};
}

ProductionFixture make_complete_production_fixture(
    double network7_sync_offset = 0.0025) {
    std::vector<int> slots;
    for (int slot = 0; slot < 14; ++slot) {
        slots.push_back(slot);
    }
    const auto ts0 = make_ts(slots);
    const auto ts7 = make_ts(slots);
    const auto time0 =
        citlali::pipeline::network_time_from_timestream_matrix(
            ts0, 1000.0, 0.0);
    const auto time7 =
        citlali::pipeline::network_time_from_timestream_matrix(
            ts7, 1000.0, network7_sync_offset);
    NativeNetworkAlignment network0{
        0, 100, time0, packet_counters_from_timestream_matrix(ts0)};
    NativeNetworkAlignment network7{
        7, 700, time7, packet_counters_from_timestream_matrix(ts7)};
    Eigen::VectorXd common(14);
    for (Eigen::Index slot = 0; slot < common.size(); ++slot) {
        common(slot) = 1000.0 + 0.01 * static_cast<double>(slot);
    }
    NativeAlignmentPlan plan{
        {network0, network7}, common,
        {citlali::pipeline::make_direct_native_slot_associations(100, 14),
         citlali::pipeline::make_direct_native_slot_associations(700, 14)}};
    auto selection = plan.select_cohort(
        NativeOperationIdentity{51, 4}, 0, 14, 0);
    Eigen::MatrixXd values0(14, 2);
    Eigen::MatrixXd values7(14, 2);
    for (Eigen::Index row = 0; row < 14; ++row) {
        values0(row, 0) = std::sin(0.19 * row) + 0.03;
        values0(row, 1) = std::cos(0.17 * row) - 0.07;
        values7(row, 0) = std::sin(0.13 * row) + 0.11;
        values7(row, 1) = std::cos(0.21 * row) - 0.02;
    }
    NativeDetectorFlagBitsMatrix flags0 =
        NativeDetectorFlagBitsMatrix::Zero(14, 2);
    NativeDetectorFlagBitsMatrix flags7 =
        NativeDetectorFlagBitsMatrix::Zero(14, 2);
    std::vector<NativeDetectorBlock> blocks;
    blocks.emplace_back(network0, 100, 0, values0, flags0);
    blocks.emplace_back(network7, 700, 2, values7, flags7);
    return ProductionFixture{
        std::move(plan), std::move(selection), std::move(blocks)};
}

void configure_ptc_proc(timestream::PTCProc &proc) {
    proc.logger = sci_align_production_logger();
    proc.cleaner.logger = proc.logger;
    proc.cleaner.stddev_limit = 0.0;
    proc.cleaner.n_calc = 0;
    proc.cleaner.standard_pca.enabled = true;
    proc.cleaner.null_model.enabled = false;
    proc.cleaner.marchenko_pastur.enabled = false;
    proc.cleaner.adaptive_selector.enabled = false;
}

struct OrdinaryResult {
    Eigen::VectorXd eigenvalues;
    Eigen::MatrixXd projector;
    Eigen::MatrixXd cleaned;
};

OrdinaryResult run_ordinary_group(
    const Eigen::MatrixXd &values,
    const citlali::pipeline::NativeDetectorBooleanMatrix &flags,
    const Eigen::VectorXi &apt_flags,
    const std::string &grouping) {
    timestream::Cleaner cleaner;
    cleaner.logger = sci_align_production_logger();
    cleaner.stddev_limit = 0.0;
    cleaner.n_calc = 0;
    cleaner.standard_pca.enabled = true;
    cleaner.null_model.enabled = false;
    cleaner.marchenko_pastur.enabled = false;
    cleaner.adaptive_selector.enabled = false;
    Eigen::VectorXi apt_flags_local = apt_flags;
    constexpr Eigen::Index cut = 1;
    auto [eigenvalues, eigenvectors] =
        cleaner.calc_eig_values<timestream::Cleaner::SpectraBackend>(
            values, flags, apt_flags_local, cut);
    Eigen::MatrixXd cleaned = values;
    cleaner.remove_eig_values<timestream::Cleaner::SpectraBackend>(
        values, flags, eigenvalues, eigenvectors, cleaned, cut, -1,
        grouping, -1, 0);
    return OrdinaryResult{
        eigenvalues,
        eigenvectors.leftCols(1) * eigenvectors.leftCols(1).transpose(),
        std::move(cleaned)};
}

TEST(sci_align_production_consumer_bridge,
     reconstructs_native_times_and_maps_subcadence_drop_without_synthesis) {
    std::vector<int> complete_slots;
    std::vector<int> dropped_slots;
    for (int slot = 0; slot < 14; ++slot) {
        complete_slots.push_back(slot);
        if (slot != 6) {
            dropped_slots.push_back(slot);
        }
    }
    EXPECT_EQ(citlali::pipeline::count_packet_counter_gaps(
                  make_ts(complete_slots)),
              0);
    EXPECT_EQ(citlali::pipeline::count_packet_counter_gaps(
                  make_ts(dropped_slots)),
              1);

    auto fixture = make_production_fixture();
    const auto &network0 = fixture.plan.network(0);
    const auto &network7 = fixture.plan.network(7);

    EXPECT_EQ(network0.row_count(), 14);
    EXPECT_EQ(network7.row_count(), 13);
    EXPECT_NEAR(
        network0.identity(100).reconstructed_time_unix_sec(),
        1000.0, 1.0e-13);
    EXPECT_NEAR(
        network7.identity(706).reconstructed_time_unix_sec(),
        1000.0725, 1.0e-12);
    EXPECT_NE(
        bits_of(network0.identity(100).reconstructed_time_unix_sec()),
        bits_of(fixture.plan.common_slot_reference_times_unix_sec()(0)));

    EXPECT_TRUE(fixture.plan.association(0, 6).mapped());
    EXPECT_FALSE(fixture.plan.association(7, 6).mapped());
    EXPECT_EQ(fixture.plan.association(7, 7).native_row, 706);
    EXPECT_EQ(
        fixture.selection.cohort().cell_for_network(6, 7).state(),
        CoincidenceCellState::absent);
    EXPECT_EQ(
        fixture.selection.cohort()
            .cell_for_network(7, 7)
            .identity()
            ->native_row(),
        706);

    const auto runs = partition_native_contiguous_runs(
        network7, 700, 713);
    ASSERT_EQ(runs.size(), 2U);
    EXPECT_EQ(runs[0].first_native_row, 700);
    EXPECT_EQ(runs[0].past_last_native_row, 706);
    EXPECT_EQ(runs[1].first_native_row, 706);
    EXPECT_EQ(runs[1].past_last_native_row, 713);
    ASSERT_TRUE(runs[0].boundary_after.counter_discontinuity.has_value());
    EXPECT_EQ(
        runs[0].boundary_after.counter_discontinuity->before_counter,
        105);
    EXPECT_EQ(
        runs[0].boundary_after.counter_discontinuity->after_counter,
        107);

    auto ledger = seed_native_detector_ledger(fixture.blocks);
    EXPECT_EQ(ledger.size(), 54U);
    EXPECT_FALSE(ledger.contains(
        {{7, 713}, 2}));
    EXPECT_EQ(
        ledger.at({{0, 108}, 1}).original_flag_bits,
        NativeDetectorFlagBits{0x4U});
}

TEST(sci_align_production_consumer_bridge,
     production_nw_groups_and_ordinary_cleaner_ignore_private_placeholders) {
    auto fixture = make_production_fixture();
    auto ledger = seed_native_detector_ledger(fixture.blocks);
    auto calib = make_fake_calib();
    timestream::PTCProc low_proc;
    timestream::PTCProc high_proc;
    configure_ptc_proc(low_proc);
    configure_ptc_proc(high_proc);

    const auto production_groups =
        low_proc.get_grouping("nw", calib, calib.n_dets);
    ASSERT_EQ(production_groups.size(), 2U);
    EXPECT_EQ(std::get<0>(production_groups.at(0)), 0);
    EXPECT_EQ(std::get<1>(production_groups.at(0)), 2);
    EXPECT_EQ(std::get<0>(production_groups.at(7)), 2);
    EXPECT_EQ(std::get<1>(production_groups.at(7)), 4);

    NativeDetectorFlagBitsMatrix exclusion =
        NativeDetectorFlagBitsMatrix::Zero(14, 4);
    exclusion(8, 1) = 0x4U;
    auto low = low_proc.prepare_native_consumer_pca(
        ledger, fixture.selection, calib, "nw", exclusion,
        FinitePcaPlaceholder::checked(-17.0));
    auto high = high_proc.prepare_native_consumer_pca(
        ledger, fixture.selection, calib, "nw", exclusion,
        FinitePcaPlaceholder::checked(911000.0));
    ASSERT_EQ(low.groups.size(), high.groups.size());
    ASSERT_EQ(low.groups.size(), 2U);

    EXPECT_EQ(low.grouping, "nw");
    EXPECT_EQ(low.groups[0].group_key, 0);
    EXPECT_EQ(low.groups[0].subgroup_index, 0);
    EXPECT_EQ(low.groups[0].role,
              NativePreparedPcaGroupRole::pca_clean);
    EXPECT_EQ(low.groups[0].detector_columns,
              (std::vector<Eigen::Index>{0, 1}));
    EXPECT_EQ(low.groups[0].detector_uids,
              (std::vector<citlali::pipeline::TimestreamDetectorUid>{
                  1000, 1001}));
    EXPECT_EQ(low.groups[1].group_key, 7);
    EXPECT_EQ(low.groups[1].subgroup_index, 0);
    EXPECT_EQ(low.groups[1].role,
              NativePreparedPcaGroupRole::pca_clean);
    EXPECT_EQ(low.groups[1].detector_columns,
              (std::vector<Eigen::Index>{2, 3}));
    EXPECT_EQ(low.groups[1].detector_uids,
              (std::vector<citlali::pipeline::TimestreamDetectorUid>{
                  7000, 7001}));
    ASSERT_EQ(low.groups[0].working_set.detector_bindings().size(), 2U);
    EXPECT_EQ(low.groups[0].working_set.detector_bindings()[0].network_id,
              0);
    EXPECT_EQ(low.groups[0].working_set.detector_bindings()[0].detector_uid,
              1000);
    EXPECT_EQ(low.groups[0].working_set.detector_bindings()[1].network_id,
              0);
    ASSERT_EQ(low.groups[1].working_set.detector_bindings().size(), 2U);
    EXPECT_EQ(low.groups[1].working_set.detector_bindings()[0].network_id,
              7);
    EXPECT_EQ(low.groups[1].working_set.detector_bindings()[0].detector_uid,
              7000);

    const auto &low_nw0 = low.groups[0].working_set;
    const auto &high_nw0 = high.groups[0].working_set;
    constexpr std::size_t nw0_flagged_flat = 8U * 2U + 1U;
    EXPECT_EQ(low_nw0.provenance_states().at(nw0_flagged_flat),
              CoincidenceCellState::mapped_invalid);
    ASSERT_TRUE(low_nw0.mapped_identities().at(nw0_flagged_flat).has_value());
    EXPECT_EQ(
        low_nw0.mapped_identities().at(nw0_flagged_flat)->native_row(),
        108);
    ASSERT_TRUE(
        low_nw0.invalidity_provenance().at(nw0_flagged_flat).has_value());
    EXPECT_EQ(
        low_nw0.invalidity_provenance()
            .at(nw0_flagged_flat)
            ->delivered_flag_bits,
        0x4U);
    EXPECT_EQ(
        low_nw0.invalidity_provenance()
            .at(nw0_flagged_flat)
            ->operation_exclusion_bits,
        0x4U);
    EXPECT_EQ(bits_of(low_nw0.values()(8, 1)), bits_of(-17.0));
    EXPECT_EQ(bits_of(high_nw0.values()(8, 1)), bits_of(911000.0));

    const auto &low_nw7 = low.groups[1].working_set;
    const auto &high_nw7 = high.groups[1].working_set;
    for (const std::size_t absent_flat : {12U, 13U}) {
        EXPECT_EQ(low_nw7.provenance_states().at(absent_flat),
                  CoincidenceCellState::absent);
        EXPECT_FALSE(low_nw7.mapped_identities().at(absent_flat).has_value());
        ASSERT_TRUE(low_nw7.absence_reasons().at(absent_flat).has_value());
        EXPECT_EQ(*low_nw7.absence_reasons().at(absent_flat),
                  CoincidenceAbsenceReason::no_candidate);
    }
    EXPECT_EQ(bits_of(low_nw7.values()(6, 0)), bits_of(-17.0));
    EXPECT_EQ(bits_of(low_nw7.values()(6, 1)), bits_of(-17.0));
    EXPECT_EQ(bits_of(high_nw7.values()(6, 0)), bits_of(911000.0));
    EXPECT_EQ(bits_of(high_nw7.values()(6, 1)), bits_of(911000.0));
    constexpr std::size_t nw7_slot7_col2_flat = 7U * 2U;
    ASSERT_TRUE(
        low_nw7.mapped_identities().at(nw7_slot7_col2_flat).has_value());
    EXPECT_EQ(
        low_nw7.mapped_identities().at(nw7_slot7_col2_flat)->native_row(),
        706);
    EXPECT_EQ(
        bits_of(low_nw7.values()(7, 0)),
        bits_of(ledger.at({{7, 706}, 2}).current_value));

    for (std::size_t group = 0; group < low.groups.size(); ++group) {
        const auto &low_working = low.groups[group].working_set;
        const auto &high_working = high.groups[group].working_set;
        EXPECT_FALSE((low_working.exclusion_flags().array() !=
                      high_working.exclusion_flags().array())
                         .any());
        EXPECT_EQ(low_working.provenance_states(),
                  high_working.provenance_states());
        EXPECT_EQ(low_working.mapped_identities(),
                  high_working.mapped_identities());
        EXPECT_EQ(low_working.expected_revisions(),
                  high_working.expected_revisions());
        EXPECT_EQ(low_working.invalidity_provenance(),
                  high_working.invalidity_provenance());
        EXPECT_EQ(low_working.absence_reasons(),
                  high_working.absence_reasons());
        EXPECT_EQ(low_working.participant_indices(),
                  high_working.participant_indices());
        EXPECT_EQ(low_working.detector_bindings(),
                  high_working.detector_bindings());
        const auto low_result = run_ordinary_group(
            low_working.values(), low_working.exclusion_flags(),
            low.groups[group].apt_flags, "nw");
        const auto high_result = run_ordinary_group(
            high_working.values(), high_working.exclusion_flags(),
            high.groups[group].apt_flags, "nw");
        EXPECT_TRUE(low_result.eigenvalues.isApprox(
            high_result.eigenvalues, 1.0e-12));
        EXPECT_TRUE(low_result.projector.isApprox(
            high_result.projector, 1.0e-12));
        for (Eigen::Index row = 0;
             row < low_working.slot_count(); ++row) {
            for (Eigen::Index det = 0;
                 det < low_working.detector_count(); ++det) {
                if (!low_working.exclusion_flags()(row, det)) {
                    EXPECT_NEAR(
                        low_result.cleaned(row, det),
                        high_result.cleaned(row, det), 1.0e-12);
                }
                else {
                    EXPECT_EQ(bits_of(low_result.cleaned(row, det)),
                              bits_of(low_working.values()(row, det)));
                    EXPECT_EQ(bits_of(high_result.cleaned(row, det)),
                              bits_of(high_working.values()(row, det)));
                }
            }
        }
    }
}

TEST(sci_align_production_consumer_bridge,
     corr_nw_uses_exact_noncontiguous_memberships_and_pass_through_lineage) {
    auto fixture = make_corr_fixture();
    auto ledger = seed_native_detector_ledger(fixture.blocks);
    timestream::PTCProc proc;
    configure_ptc_proc(proc);
    proc.cleaner.corr_grouping.enabled = true;
    proc.cleaner.corr_grouping.metric = "signed";
    proc.cleaner.corr_grouping.corr_min = 0.995;
    proc.cleaner.corr_grouping.min_overlap = 200;
    proc.cleaner.corr_grouping.min_good_frac = 0.9;
    proc.cleaner.corr_grouping.min_group_size = 3;
    proc.cleaner.corr_grouping.max_samples = 0;
    proc.cleaner.corr_grouping.clean_residual = false;

    NativeDetectorFlagBitsMatrix exclusion =
        NativeDetectorFlagBitsMatrix::Zero(240, 7);
    auto prepared = proc.prepare_native_consumer_pca(
        ledger, fixture.selection, fixture.calib, "corr_nw", exclusion,
        FinitePcaPlaceholder::checked(-23.0));
    EXPECT_EQ(prepared.grouping, "corr_nw");
    prepared.require_complete_detector_partition();

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> direct_flags(240, 7);
    direct_flags.setConstant(false);
    Eigen::VectorXi direct_apt_flags = Eigen::VectorXi::Zero(7);
    const auto direct = proc.cleaner.get_corr_groups(
        fixture.blocks[0].measured_values(), direct_flags,
        direct_apt_flags);
    ASSERT_EQ(direct.groups.size(), 2U);

    std::vector<std::vector<Eigen::Index>> expected_groups = direct.groups;
    std::vector<std::vector<Eigen::Index>> prepared_groups;
    std::vector<Eigen::Index> pass_through;
    for (const auto &group : prepared.groups) {
        EXPECT_TRUE(group.group_identity_is_frozen());
        EXPECT_EQ(group.effective_grouping, "corr_nw");
        EXPECT_EQ(group.group_key, 0);
        ASSERT_EQ(group.detector_uids.size(), group.detector_columns.size());
        for (std::size_t index = 0; index < group.detector_uids.size();
             ++index) {
            EXPECT_EQ(group.detector_uids[index],
                      9100 + group.detector_columns[index]);
        }
        if (group.role == NativePreparedPcaGroupRole::pca_clean) {
            prepared_groups.push_back(group.detector_columns);
        }
        else {
            ASSERT_EQ(group.role,
                      NativePreparedPcaGroupRole::pass_through);
            pass_through = group.detector_columns;
        }
    }
    auto normalize_memberships = [](auto &groups) {
        for (auto &group : groups) {
            std::sort(group.begin(), group.end());
        }
        std::sort(groups.begin(), groups.end());
    };
    normalize_memberships(expected_groups);
    normalize_memberships(prepared_groups);
    EXPECT_EQ(prepared_groups, expected_groups);
    ASSERT_EQ(pass_through, (std::vector<Eigen::Index>{6}));

    for (auto &group : prepared.groups) {
        if (group.role == NativePreparedPcaGroupRole::pca_clean) {
            group.working_set.mutable_values_for_pca().array() += 5.0;
        }
    }
    const double pass_through_before =
        ledger.at({{0, 300}, 6}).current_value;
    scatter_native_detector_pca_results_transactionally(
        ledger, fixture.selection, prepared);
    const auto &pass_record = ledger.at({{0, 300}, 6});
    EXPECT_DOUBLE_EQ(pass_record.current_value, pass_through_before);
    EXPECT_EQ(pass_record.revision, 1U);
    ASSERT_EQ(pass_record.lineage.size(), 1U);
    EXPECT_EQ(pass_record.lineage[0].action,
              NativeDetectorRevisionAction::preserved_corr_ungrouped);
    EXPECT_EQ(pass_record.lineage[0].coincidence_provenance.detector_uid,
              9106);
    EXPECT_EQ(
        pass_record.lineage[0].coincidence_provenance.effective_grouping,
        "corr_nw");
    EXPECT_EQ(pass_record.lineage[0].coincidence_provenance.group_role,
              NativePreparedPcaGroupRole::pass_through);
    EXPECT_DOUBLE_EQ(ledger.at({{0, 300}, 0}).current_value,
                     fixture.blocks[0].measured_values()(0, 0) + 5.0);

    auto fallback_ledger = seed_native_detector_ledger(fixture.blocks);
    timestream::PTCProc fallback;
    configure_ptc_proc(fallback);
    auto fallback_prepared = fallback.prepare_native_consumer_pca(
        fallback_ledger, fixture.selection, fixture.calib, "corr_nw",
        exclusion, FinitePcaPlaceholder::checked(0.0));
    EXPECT_EQ(fallback_prepared.grouping, "nw");
    ASSERT_EQ(fallback_prepared.groups.size(), 1U);
    EXPECT_EQ(fallback_prepared.groups[0].detector_columns,
              (std::vector<Eigen::Index>{0, 1, 2, 3, 4, 5, 6}));
}

TEST(sci_align_production_consumer_bridge,
     identical_native_times_match_the_existing_rectangular_nw_result) {
    auto fixture = make_complete_production_fixture(0.0);
    for (Eigen::Index row = 0; row < 14; ++row) {
        const auto native_row =
            static_cast<citlali::pipeline::TimestreamNativeRow>(row);
        EXPECT_EQ(
            bits_of(fixture.plan.network(0)
                        .identity(100 + native_row)
                        .reconstructed_time_unix_sec()),
            bits_of(fixture.plan.network(7)
                        .identity(700 + native_row)
                        .reconstructed_time_unix_sec()));
        EXPECT_EQ(fixture.plan.association(0, row).native_row,
                  100 + native_row);
        EXPECT_EQ(fixture.plan.association(7, row).native_row,
                  700 + native_row);
    }

    Eigen::MatrixXd legacy_values(14, 4);
    legacy_values.leftCols(2) = fixture.blocks[0].measured_values();
    legacy_values.rightCols(2) = fixture.blocks[1].measured_values();
    Eigen::MatrixXd legacy_cleaned = legacy_values;
    NativeDetectorFlagBitsMatrix no_exclusion =
        NativeDetectorFlagBitsMatrix::Zero(14, 4);
    auto ledger = seed_native_detector_ledger(fixture.blocks);
    auto calib = make_fake_calib();
    timestream::PTCProc proc;
    configure_ptc_proc(proc);
    auto prepared = proc.prepare_native_consumer_pca(
        ledger, fixture.selection, calib, "nw", no_exclusion,
        FinitePcaPlaceholder::checked(-123.0));

    for (auto &group : prepared.groups) {
        const auto columns = static_cast<Eigen::Index>(
            group.detector_columns.size());
        Eigen::MatrixXd legacy_group(14, columns);
        for (Eigen::Index local = 0; local < columns; ++local) {
            legacy_group.col(local) = legacy_values.col(
                group.detector_columns.at(static_cast<std::size_t>(local)));
        }
        for (Eigen::Index row = 0; row < legacy_group.rows(); ++row) {
            for (Eigen::Index local = 0; local < legacy_group.cols();
                 ++local) {
                EXPECT_EQ(bits_of(group.working_set.values()(row, local)),
                          bits_of(legacy_group(row, local)));
            }
        }
        citlali::pipeline::NativeDetectorBooleanMatrix clean_flags(
            14, columns);
        clean_flags.setConstant(false);
        const auto native_result = run_ordinary_group(
            group.working_set.values(),
            group.working_set.exclusion_flags(), group.apt_flags, "nw");
        const auto legacy_result = run_ordinary_group(
            legacy_group, clean_flags, group.apt_flags, "nw");
        EXPECT_TRUE(native_result.eigenvalues.isApprox(
            legacy_result.eigenvalues, 1.0e-12));
        EXPECT_TRUE(native_result.projector.isApprox(
            legacy_result.projector, 1.0e-12));
        EXPECT_TRUE(native_result.cleaned.isApprox(
            legacy_result.cleaned, 1.0e-12));
        group.working_set.mutable_values_for_pca() = native_result.cleaned;
        for (Eigen::Index local = 0; local < columns; ++local) {
            legacy_cleaned.col(
                group.detector_columns.at(static_cast<std::size_t>(local))) =
                legacy_result.cleaned.col(local);
        }
    }

    scatter_native_detector_pca_results_transactionally(
        ledger, fixture.selection, prepared);
    EXPECT_EQ(ledger.size(), 56U);
    for (Eigen::Index row = 0; row < 14; ++row) {
        for (Eigen::Index detector = 0; detector < 4; ++detector) {
            const auto network_id = detector < 2 ? 0 : 7;
            const auto native_row =
                (detector < 2 ? 100 : 700) + row;
            const auto &record = ledger.at(
                {{network_id, native_row}, detector});
            EXPECT_EQ(bits_of(record.measured_value),
                      bits_of(legacy_values(row, detector)));
            EXPECT_NEAR(record.current_value,
                        legacy_cleaned(row, detector), 1.0e-12);
            EXPECT_EQ(record.revision, 1U);
        }
    }
}

TEST(sci_align_production_consumer_bridge,
     optional_modes_fail_closed_from_actual_detector_mask_before_cleaner) {
    auto fixture = make_complete_production_fixture();
    auto ledger = seed_native_detector_ledger(fixture.blocks);
    auto calib = make_fake_calib();
    NativeDetectorFlagBitsMatrix exclusion =
        NativeDetectorFlagBitsMatrix::Zero(14, 4);
    exclusion(3, 0) = 0x20U;

    for (std::size_t slot = 0;
         slot < fixture.selection.cohort().slot_count(); ++slot) {
        EXPECT_TRUE(
            fixture.selection.cohort().cell_for_network(slot, 0).pca_valid());
        EXPECT_TRUE(
            fixture.selection.cohort().cell_for_network(slot, 7).pca_valid());
    }

    timestream::PTCProc ordinary;
    configure_ptc_proc(ordinary);
    auto ordinary_prepared = ordinary.prepare_native_consumer_pca(
        ledger, fixture.selection, calib, "nw", exclusion,
        FinitePcaPlaceholder::checked(0.0));
    ASSERT_TRUE(
        ordinary_prepared.groups[0].working_set.exclusion_flags()(3, 0));

    constexpr const char *incompatibility_message =
        "actual native detector exclusions are incompatible with the "
        "requested optional PCA mode; no fallback is selected";
    const auto expect_fail_closed = [&](timestream::PTCProc &proc) {
        try {
            (void)proc.prepare_native_consumer_pca(
                ledger, fixture.selection, calib, "nw", exclusion,
                FinitePcaPlaceholder::checked(0.0));
            FAIL() << "incompatible optional PCA mode did not fail closed";
        }
        catch (const std::logic_error &error) {
            EXPECT_EQ(std::string{error.what()}, incompatibility_message);
        }
        catch (...) {
            FAIL() << "incompatible optional PCA mode threw the wrong type";
        }
    };

    timestream::PTCProc null_proc;
    configure_ptc_proc(null_proc);
    null_proc.cleaner.standard_pca.enabled = false;
    null_proc.cleaner.null_model.enabled = true;
    PcaCompatibilityInputs null_inputs;
    null_inputs.null_model_active_for_operation = true;
    const auto null_classification =
        classify_native_detector_pca_compatibility(
            ordinary_prepared.groups[0].working_set, null_inputs);
    EXPECT_TRUE(null_classification.has(PcaCompatibilityHazard::null_model));
    expect_fail_closed(null_proc);

    timestream::PTCProc adaptive_proc;
    configure_ptc_proc(adaptive_proc);
    adaptive_proc.cleaner.standard_pca.enabled = false;
    adaptive_proc.cleaner.adaptive_selector.enabled = true;
    PcaCompatibilityInputs adaptive_inputs;
    adaptive_inputs.adaptive_selector_active_for_operation = true;
    const auto adaptive_classification =
        classify_native_detector_pca_compatibility(
            ordinary_prepared.groups[0].working_set, adaptive_inputs);
    EXPECT_TRUE(adaptive_classification.has(
        PcaCompatibilityHazard::adaptive_selector));
    expect_fail_closed(adaptive_proc);

    timestream::PTCProc banded_mp_proc;
    configure_ptc_proc(banded_mp_proc);
    banded_mp_proc.cleaner.standard_pca.enabled = false;
    banded_mp_proc.cleaner.marchenko_pastur.enabled = true;
    banded_mp_proc.cleaner.marchenko_pastur.band_low_Hz = 0.1;
    PcaCompatibilityInputs banded_mp_inputs;
    banded_mp_inputs.marchenko_pastur_active_for_operation = true;
    banded_mp_inputs.marchenko_pastur_band_requested = true;
    const auto banded_mp_classification =
        classify_native_detector_pca_compatibility(
            ordinary_prepared.groups[0].working_set, banded_mp_inputs);
    EXPECT_TRUE(banded_mp_classification.has(
        PcaCompatibilityHazard::band_limited_marchenko_pastur));
    expect_fail_closed(banded_mp_proc);

    timestream::PTCProc unbanded_mp_proc;
    configure_ptc_proc(unbanded_mp_proc);
    unbanded_mp_proc.cleaner.standard_pca.enabled = false;
    unbanded_mp_proc.cleaner.marchenko_pastur.enabled = true;
    unbanded_mp_proc.cleaner.marchenko_pastur.band_low_Hz = 0.0;
    unbanded_mp_proc.cleaner.marchenko_pastur.band_high_Hz = 0.0;
    const auto unbanded_prepared =
        unbanded_mp_proc.prepare_native_consumer_pca(
            ledger, fixture.selection, calib, "nw", exclusion,
            FinitePcaPlaceholder::checked(0.0));
    EXPECT_EQ(unbanded_prepared.groups.size(), 2U);

    timestream::PTCProc group_gated_null_proc;
    configure_ptc_proc(group_gated_null_proc);
    group_gated_null_proc.cleaner.standard_pca.enabled = false;
    group_gated_null_proc.cleaner.null_model.enabled = true;
    group_gated_null_proc.cleaner.null_model.grouping = {"array"};
    const auto group_gated_prepared =
        group_gated_null_proc.prepare_native_consumer_pca(
            ledger, fixture.selection, calib, "nw", exclusion,
            FinitePcaPlaceholder::checked(0.0));
    EXPECT_EQ(group_gated_prepared.groups.size(), 2U);

    auto apt_exclusion_calib = make_fake_calib();
    apt_exclusion_calib.apt["flag"](0) = 3.0;
    NativeDetectorFlagBitsMatrix no_sample_exclusion =
        NativeDetectorFlagBitsMatrix::Zero(14, 4);
    auto apt_prepared = ordinary.prepare_native_consumer_pca(
        ledger, fixture.selection, apt_exclusion_calib, "nw",
        no_sample_exclusion, FinitePcaPlaceholder::checked(-31.0));
    const auto &apt_working = apt_prepared.groups[0].working_set;
    ASSERT_TRUE(apt_working.invalidity_provenance().at(0).has_value());
    EXPECT_EQ(apt_working.invalidity_provenance().at(0)->apt_flag_value, 3);
    EXPECT_EQ(
        apt_working.invalidity_provenance().at(0)->delivered_flag_bits, 0U);
    EXPECT_EQ(
        apt_working.invalidity_provenance().at(0)->operation_exclusion_bits,
        0U);
    EXPECT_EQ(bits_of(apt_working.values()(0, 0)), bits_of(-31.0));
    EXPECT_FALSE(ledger.last_operation().has_value());
}

TEST(sci_align_production_consumer_bridge,
     scatter_rejects_nonfinite_batch_atomically_then_allows_same_operation_retry) {
    auto fixture = make_production_fixture();
    auto ledger = seed_native_detector_ledger(fixture.blocks);
    auto calib = make_fake_calib();
    timestream::PTCProc proc;
    configure_ptc_proc(proc);
    NativeDetectorFlagBitsMatrix exclusion =
        NativeDetectorFlagBitsMatrix::Zero(14, 4);
    exclusion(8, 1) = 0x4U;
    auto prepared = proc.prepare_native_consumer_pca(
        ledger, fixture.selection, calib, "nw", exclusion,
        FinitePcaPlaceholder::checked(0.0));

    const auto before_snapshot = ledger.snapshot();
    const auto assert_snapshot_equal = [&](const NativeDetectorLedger &target,
                                           const auto &expected) {
        const auto actual = target.snapshot();
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t index = 0; index < expected.size(); ++index) {
            const auto &lhs = actual[index];
            const auto &rhs = expected[index];
            EXPECT_EQ(lhs.key, rhs.key);
            EXPECT_EQ(bits_of(lhs.measured_value),
                      bits_of(rhs.measured_value));
            EXPECT_EQ(bits_of(lhs.current_value),
                      bits_of(rhs.current_value));
            EXPECT_EQ(lhs.original_flag_bits,
                      rhs.original_flag_bits);
            EXPECT_EQ(lhs.original_flag_reason,
                      rhs.original_flag_reason);
            EXPECT_EQ(lhs.revision, rhs.revision);
            ASSERT_EQ(lhs.lineage.size(), rhs.lineage.size());
            for (std::size_t lineage = 0;
                 lineage < lhs.lineage.size(); ++lineage) {
                EXPECT_TRUE(native_detector_revision_record_equal(
                    lhs.lineage[lineage], rhs.lineage[lineage]));
            }
        }
    };

    std::vector<std::vector<NativeSlotAssociation>> altered_associations(2);
    for (std::size_t participant = 0; participant < 2; ++participant) {
        const auto network_id =
            fixture.plan.participant_network_ids().at(participant);
        altered_associations[participant].reserve(fixture.plan.slot_count());
        for (std::size_t slot = 0; slot < fixture.plan.slot_count(); ++slot) {
            altered_associations[participant].push_back(
                fixture.plan.association(network_id, slot));
        }
    }
    std::swap(altered_associations[1][7].native_row,
              altered_associations[1][8].native_row);
    NativeAlignmentPlan altered_plan{
        fixture.plan.networks(),
        fixture.plan.common_slot_reference_times_unix_sec(),
        std::move(altered_associations)};
    const auto altered_selection = altered_plan.select_cohort(
        NativeOperationIdentity{41, 3}, 0, 14, 0);
    EXPECT_THROW(
        scatter_native_detector_pca_results_transactionally(
            ledger, altered_selection, prepared),
        std::logic_error);
    EXPECT_FALSE(ledger.last_operation().has_value());
    assert_snapshot_equal(ledger, before_snapshot);

    auto stale_ledger = seed_native_detector_ledger(fixture.blocks);
    auto stale_prepared = proc.prepare_native_consumer_pca(
        stale_ledger, fixture.selection, calib, "nw", exclusion,
        FinitePcaPlaceholder::checked(0.0));
    const auto stale_identity = fixture.plan.network(7).identity(712);
    const auto stale_value = stale_ledger.at({{7, 712}, 3}).current_value;
    NativeDetectorCoincidenceProvenance stale_provenance{
        13, 1, 7, 3, 7001, "nw", 7, 0,
        NativePreparedPcaGroupRole::pca_clean, 0, 0, 0, {}};
    stale_ledger.apply_transaction(
        NativeOperationIdentity{40, 3},
        {NativeDetectorLedger::Update::replacement(
            stale_identity, 3, 0, stale_value,
            std::move(stale_provenance))});
    const auto stale_snapshot = stale_ledger.snapshot();
    EXPECT_THROW(
        scatter_native_detector_pca_results_transactionally(
            stale_ledger, fixture.selection, stale_prepared),
        std::logic_error);
    ASSERT_TRUE(stale_ledger.last_operation().has_value());
    EXPECT_EQ(stale_ledger.last_operation()->sequence, 40U);
    assert_snapshot_equal(stale_ledger, stale_snapshot);

    auto &last_group = prepared.groups.back().working_set;
    last_group.mutable_values_for_pca()(13, 1) =
        std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        scatter_native_detector_pca_results_transactionally(
            ledger, fixture.selection, prepared),
        std::logic_error);
    EXPECT_FALSE(ledger.last_operation().has_value());
    assert_snapshot_equal(ledger, before_snapshot);

    last_group.mutable_values_for_pca()(13, 1) = 123.5;
    prepared.groups.front().working_set.mutable_values_for_pca()(0, 0) =
        77.25;
    scatter_native_detector_pca_results_transactionally(
        ledger, fixture.selection, prepared);
    ASSERT_TRUE(ledger.last_operation().has_value());
    EXPECT_EQ(ledger.last_operation()->sequence, 41U);
    EXPECT_EQ(ledger.last_operation()->scan_index, 3);
    EXPECT_EQ(ledger.size(), 54U);
    EXPECT_FALSE(ledger.contains({{7, 713}, 2}));
    EXPECT_DOUBLE_EQ(ledger.at({{0, 100}, 0}).current_value, 77.25);
    EXPECT_DOUBLE_EQ(ledger.at({{7, 712}, 3}).current_value, 123.5);
    EXPECT_EQ(ledger.at({{0, 108}, 1}).original_flag_bits, 0x4U);
    EXPECT_EQ(
        ledger.at({{0, 108}, 1}).lineage.back().action,
        NativeDetectorRevisionAction::preserved_pca_invalid);

    for (const auto &entry : ledger.snapshot()) {
        EXPECT_EQ(entry.revision, 1U);
        ASSERT_EQ(entry.lineage.size(), 1U);
        EXPECT_EQ(entry.lineage[0].operation.sequence, 41U);
        EXPECT_EQ(entry.lineage[0].operation.scan_index, 3);
        EXPECT_EQ(entry.lineage[0].input_revision, 0U);
        EXPECT_EQ(entry.lineage[0].output_revision, 1U);
    }
    const auto &valid_record = ledger.at({{0, 100}, 0});
    ASSERT_EQ(valid_record.lineage.size(), 1U);
    EXPECT_EQ(valid_record.lineage[0].action,
              NativeDetectorRevisionAction::replaced_by_pca_result);
    const auto &valid_provenance =
        valid_record.lineage[0].coincidence_provenance;
    EXPECT_EQ(valid_provenance.common_slot, 0U);
    EXPECT_EQ(valid_provenance.participant_index, 0U);
    EXPECT_EQ(valid_provenance.participant_network_id, 0);
    EXPECT_EQ(valid_provenance.detector_column, 0);
    EXPECT_EQ(valid_provenance.detector_uid, 1000);
    EXPECT_EQ(valid_provenance.effective_grouping, "nw");
    EXPECT_EQ(valid_provenance.group_key, 0);
    EXPECT_EQ(valid_provenance.subgroup_index, 0);
    EXPECT_EQ(valid_provenance.group_role,
              NativePreparedPcaGroupRole::pca_clean);
    EXPECT_EQ(valid_provenance.delivered_flag_bits, 0U);
    EXPECT_EQ(valid_provenance.operation_exclusion_bits, 0U);

    const auto &invalid_record = ledger.at({{0, 108}, 1});
    EXPECT_EQ(bits_of(invalid_record.current_value),
              bits_of(invalid_record.measured_value));
    ASSERT_EQ(invalid_record.lineage.size(), 1U);
    const auto &invalid_provenance =
        invalid_record.lineage[0].coincidence_provenance;
    EXPECT_EQ(invalid_provenance.common_slot, 8U);
    EXPECT_EQ(invalid_provenance.participant_index, 0U);
    EXPECT_EQ(invalid_provenance.participant_network_id, 0);
    EXPECT_EQ(invalid_provenance.detector_column, 1);
    EXPECT_EQ(invalid_provenance.detector_uid, 1001);
    EXPECT_EQ(invalid_provenance.delivered_flag_bits, 0x4U);
    EXPECT_EQ(invalid_provenance.operation_exclusion_bits, 0x4U);
    EXPECT_EQ(invalid_provenance.apt_flag_value, 0);
    EXPECT_NE(invalid_provenance.exclusion_reason.find(
                  "delivered detector flag bits"),
              std::string::npos);
    EXPECT_NE(invalid_provenance.exclusion_reason.find(
                  "actual production PCA exclusion mask"),
              std::string::npos);

    const auto committed =
        ledger.at({{0, 100}, 0}).current_value;
    const auto committed_snapshot = ledger.snapshot();
    EXPECT_THROW(
        scatter_native_detector_pca_results_transactionally(
            ledger, fixture.selection, prepared),
        std::logic_error);
    EXPECT_DOUBLE_EQ(
        ledger.at({{0, 100}, 0}).current_value, committed);
    EXPECT_EQ(ledger.at({{0, 100}, 0}).revision, 1U);
    assert_snapshot_equal(ledger, committed_snapshot);
}

TEST(sci_align_production_consumer_bridge,
     rtc_downsamples_each_native_run_from_own_anchor_and_ors_flag_support) {
    Eigen::VectorXd times(6);
    times << 10.00, 10.01, 10.02, 10.04, 10.05, 10.06;
    NativeNetworkAlignment network{
        0, 200, times, {10, 11, 12, 14, 15, 16}};
    Eigen::MatrixXd values(6, 2);
    for (Eigen::Index row = 0; row < values.rows(); ++row) {
        values(row, 0) = 100.0 + static_cast<double>(row);
        values(row, 1) = 200.0 + static_cast<double>(row);
    }
    NativeDetectorFlagBitsMatrix original_flags =
        NativeDetectorFlagBitsMatrix::Zero(6, 2);
    original_flags(1, 0) = 0x8000U;
    original_flags(4, 1) = 0x4000U;
    NativeDetectorBlock block{network, 200, 0, values, original_flags};
    const auto runs = partition_native_contiguous_runs(network, 200, 206);
    ASSERT_EQ(runs.size(), 2U);
    EXPECT_TRUE(runs[0].boundary_before.scan_boundary);
    EXPECT_TRUE(runs[0].boundary_before.stream_boundary);
    EXPECT_FALSE(
        runs[0].boundary_before.counter_discontinuity.has_value());
    EXPECT_FALSE(runs[0].boundary_after.scan_boundary);
    EXPECT_FALSE(runs[0].boundary_after.stream_boundary);
    ASSERT_TRUE(runs[0].boundary_after.counter_discontinuity.has_value());
    EXPECT_EQ(
        runs[0].boundary_after.counter_discontinuity->before_counter, 12);
    EXPECT_EQ(
        runs[0].boundary_after.counter_discontinuity->after_counter, 14);
    ASSERT_TRUE(runs[1].boundary_before.counter_discontinuity.has_value());
    EXPECT_TRUE(runs[1].boundary_after.scan_boundary);
    EXPECT_TRUE(runs[1].boundary_after.stream_boundary);

    const auto interior_runs =
        partition_native_contiguous_runs(network, 201, 205);
    ASSERT_EQ(interior_runs.size(), 2U);
    EXPECT_TRUE(interior_runs[0].boundary_before.scan_boundary);
    EXPECT_FALSE(interior_runs[0].boundary_before.stream_boundary);
    EXPECT_FALSE(interior_runs[0].boundary_after.scan_boundary);
    ASSERT_TRUE(
        interior_runs[0].boundary_after.counter_discontinuity.has_value());
    ASSERT_TRUE(
        interior_runs[1].boundary_before.counter_discontinuity.has_value());
    EXPECT_TRUE(interior_runs[1].boundary_after.scan_boundary);
    EXPECT_FALSE(interior_runs[1].boundary_after.stream_boundary);

    timestream::RTCProc proc;
    proc.downsampler.factor = 2;
    Eigen::MatrixXd processed_first = values.topRows(3);
    Eigen::MatrixXd processed_second = values.bottomRows(3);
    processed_first.array() += 1000.0;
    processed_second.array() += 2000.0;
    NativeDetectorFlagBitsMatrix flags_first(3, 2);
    flags_first << 0x1U, 0x8U,
                   0x4U, 0x0U,
                   0x10U, 0x0U;
    NativeDetectorFlagBitsMatrix flags_second(3, 2);
    flags_second << 0x20U, 0x80U,
                    0x40U, 0x100U,
                    0x0U, 0x200U;
    const auto first = proc.downsample_native_consumer_run(
        block, runs[0], processed_first, flags_first, 0);
    const auto second = proc.downsample_native_consumer_run(
        block, runs[1], processed_second, flags_second, 1);

    ASSERT_EQ(first.support.size(), 2U);
    ASSERT_EQ(second.support.size(), 2U);
    EXPECT_EQ(first.support[0].run_ordinal, 0U);
    EXPECT_EQ(first.support[1].run_ordinal, 0U);
    EXPECT_EQ(second.support[0].run_ordinal, 1U);
    EXPECT_EQ(second.support[1].run_ordinal, 1U);
    EXPECT_EQ(first.support[0].factor, 2);
    EXPECT_EQ(second.support[0].factor, 2);
    EXPECT_EQ(first.support[0].selected_anchor.native_row(), 200);
    EXPECT_EQ(first.support[0].selected_anchor.network_id(), 0);
    EXPECT_DOUBLE_EQ(
        first.support[0].selected_anchor.reconstructed_time_unix_sec(),
        10.00);
    EXPECT_EQ(first.support[0].run_output_row, 0);
    EXPECT_EQ(first.support[1].run_output_row, 1);
    EXPECT_EQ(first.support[1].selected_anchor.native_row(), 202);
    EXPECT_EQ(second.support[0].selected_anchor.native_row(), 203);
    EXPECT_EQ(second.support[0].run_output_row, 0);
    EXPECT_EQ(second.support[1].run_output_row, 1);
    EXPECT_EQ(second.support[1].selected_anchor.native_row(), 205);
    EXPECT_DOUBLE_EQ(
        second.support[0].selected_anchor.reconstructed_time_unix_sec(),
        10.04);
    EXPECT_DOUBLE_EQ(
        second.support[1].selected_anchor.reconstructed_time_unix_sec(),
        10.06);
    EXPECT_EQ(first.support[0].first_support_native_row, 200);
    EXPECT_EQ(first.support[0].past_last_support_native_row, 202);
    EXPECT_EQ(first.support[1].first_support_native_row, 202);
    EXPECT_EQ(first.support[1].past_last_support_native_row, 203);
    EXPECT_TRUE(first.support[1].final_short_support);
    EXPECT_EQ(second.support[0].first_support_native_row, 203);
    EXPECT_EQ(second.support[0].past_last_support_native_row, 205);
    EXPECT_EQ(second.support[1].first_support_native_row, 205);
    EXPECT_EQ(second.support[1].past_last_support_native_row, 206);
    EXPECT_TRUE(second.support[1].final_short_support);
    EXPECT_FALSE(first.support[0].final_short_support);
    EXPECT_FALSE(second.support[0].final_short_support);
    ASSERT_EQ(first.support[0].exact_support_rows.size(), 2U);
    EXPECT_EQ(first.support[0].exact_support_rows[0].native_row(), 200);
    EXPECT_EQ(first.support[0].exact_support_rows[1].native_row(), 201);
    ASSERT_EQ(first.support[1].exact_support_rows.size(), 1U);
    EXPECT_EQ(first.support[1].exact_support_rows[0].native_row(), 202);
    ASSERT_EQ(second.support[0].exact_support_rows.size(), 2U);
    EXPECT_EQ(second.support[0].exact_support_rows[0].native_row(), 203);
    EXPECT_EQ(second.support[0].exact_support_rows[1].native_row(), 204);
    ASSERT_EQ(second.support[1].exact_support_rows.size(), 1U);
    EXPECT_EQ(second.support[1].exact_support_rows[0].native_row(), 205);
    EXPECT_EQ(first.support[0].detector_columns,
              (std::vector<Eigen::Index>{0, 1}));
    EXPECT_EQ(second.support[0].detector_columns,
              (std::vector<Eigen::Index>{0, 1}));
    EXPECT_EQ(first.support[0].ored_flag_support,
              (std::vector<NativeDetectorFlagBits>{0x5U, 0x8U}));
    EXPECT_EQ(first.support[1].ored_flag_support,
              (std::vector<NativeDetectorFlagBits>{0x10U, 0x0U}));
    EXPECT_EQ(second.support[0].ored_flag_support,
              (std::vector<NativeDetectorFlagBits>{0x60U, 0x180U}));
    EXPECT_EQ(second.support[1].ored_flag_support,
              (std::vector<NativeDetectorFlagBits>{0x0U, 0x200U}));
    EXPECT_TRUE(first.ored_flags(0, 0));
    EXPECT_TRUE(first.ored_flags(0, 1));
    EXPECT_TRUE(first.ored_flags(1, 0));
    EXPECT_FALSE(first.ored_flags(1, 1));
    EXPECT_TRUE(second.ored_flags(0, 0));
    EXPECT_TRUE(second.ored_flags(0, 1));
    EXPECT_FALSE(second.ored_flags(1, 0));
    EXPECT_TRUE(second.ored_flags(1, 1));
    EXPECT_DOUBLE_EQ(first.selected_values(0, 0), 1100.0);
    EXPECT_DOUBLE_EQ(first.selected_values(0, 1), 1200.0);
    EXPECT_DOUBLE_EQ(first.selected_values(1, 0), 1102.0);
    EXPECT_DOUBLE_EQ(first.selected_values(1, 1), 1202.0);
    EXPECT_DOUBLE_EQ(second.selected_values(0, 0), 2103.0);
    EXPECT_DOUBLE_EQ(second.selected_values(0, 1), 2203.0);
    EXPECT_DOUBLE_EQ(second.selected_values(1, 0), 2105.0);
    EXPECT_DOUBLE_EQ(second.selected_values(1, 1), 2205.0);

    Eigen::VectorXd discontinuous_times(4);
    discontinuous_times << 20.0, 20.01, 20.02, 20.03;
    NativeNetworkAlignment discontinuous_network{
        9, 900, discontinuous_times, {20, 20, 19, 20}};
    const auto discontinuous_runs = partition_native_contiguous_runs(
        discontinuous_network, 900, 904);
    ASSERT_EQ(discontinuous_runs.size(), 3U);
    EXPECT_EQ(discontinuous_runs[0].past_last_native_row, 901);
    EXPECT_EQ(discontinuous_runs[1].past_last_native_row, 902);
    EXPECT_EQ(discontinuous_runs[2].past_last_native_row, 904);
}

}  // namespace

TEST(sci_align_final_publication,
     required_products_gate_atomic_deterministic_index_replacement) {
    const auto root = std::filesystem::temp_directory_path() /
        "citlali_sci_align_final_publication_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const auto required = root / "native_cohort_product_provenance.yaml";
    { std::ofstream output(required); output << "complete: true\n"; }
    citlali::pipeline::write_final_product_index_file(root, {required});
    const auto index = root / "index.yaml";
    std::ifstream first(index); const std::string first_text{
        std::istreambuf_iterator<char>{first}, {}};
    EXPECT_FALSE(first_text.empty());
    std::filesystem::remove(required);
    EXPECT_THROW(citlali::pipeline::write_final_product_index_file(root, {required}),
                 std::logic_error);
    std::ifstream retained(index); const std::string retained_text{
        std::istreambuf_iterator<char>{retained}, {}};
    EXPECT_EQ(retained_text, first_text);
    std::filesystem::remove_all(root);
}

// B2a typed-identity evidence is appended below. The accepted B1 body above
// remains byte-for-byte unchanged, including its production-order KIDs
// prerequisite and all 18 assertions/tests.
#include <citlali/core/engine/calib.h>
#include <citlali/core/pipeline/apt_detector_relation.h>

#include <netcdf>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <string_view>

namespace {

namespace b2a_apt = citlali::pipeline::canonical_apt_v1;
namespace b2a_observation =
    citlali::pipeline::canonical_apt_observation_v1;
namespace b2a_publication =
    citlali::pipeline::canonical_artifact_publication;

b2a_apt::RegisteredField b2a_extension(std::string_view name) {
    for (const auto &field : b2a_apt::optional_extension_fields_v1()) {
        if (field.name == name) {
            return field;
        }
    }
    throw std::logic_error("unknown B2a canonical APT extension");
}

b2a_apt::Value b2a_field_value(
    const b2a_apt::RegisteredField &field, std::size_t row_index) {
    if (field.type == b2a_apt::ValueType::int64) {
        if (field.name == "flag") {
            return std::int64_t{0};
        }
        if (field.name == "kids_flag") {
            return static_cast<std::int64_t>(70 + row_index);
        }
        return static_cast<std::int64_t>(row_index);
    }
    if (field.type == b2a_apt::ValueType::float64) {
        if (field.name == "a_fwhm" || field.name == "b_fwhm") {
            return 10.0 + static_cast<double>(row_index);
        }
        if (field.name == "angle") {
            return 0.0;
        }
        return 0.25 + static_cast<double>(row_index);
    }
    if (field.type == b2a_apt::ValueType::boolean) {
        return row_index == 0;
    }
    return std::string("b2a-row-") + std::to_string(row_index);
}

b2a_apt::Document b2a_baseline_document(
    std::string occurrence = "occurrence:sci-align/b2a-baseline#001",
    bool include_kids_flag = false) {
    b2a_apt::Document document;
    document.envelope = {
        std::move(occurrence),
        "event:sci-align/b2a-baseline#001",
        std::string(b2a_apt::baseline_output_role_v1),
        "citlali",
        "110d36fe432e6475599607ea12bd60d14b64ff94",
        "config:sci-align/b2a-synthetic",
        "2026-08-15T12:00:00Z",
    };
    document.context = {
        "SCI-ALIGN-B2A", "synthetic typed identity",
        "2026-08-15T11:59:00Z", "altaz"};
    document.raw_manifest.observation = {148670, 0, 2};
    document.raw_manifest.inputs = {{0, "toltec0", 2}};
    document.registered_fields = b2a_apt::required_baseline_fields_v1();
    if (include_kids_flag) {
        document.registered_fields.push_back(b2a_extension("kids_flag"));
    }
    // Deliberately reverse presentation order and use sparse artifact-local
    // UIDs. Detector column remains the explicit (nw,kids_tone) relation.
    document.rows = {
        {b2a_apt::uid_v1_max, 1.1e9, 0, 0, 1, {}},
        {42, 1.0e9, 0, 0, 0, {}},
    };
    for (std::size_t row_index = 0; row_index < document.rows.size();
         ++row_index) {
        for (const auto &field : document.registered_fields) {
            document.rows[row_index].fields[field.name] =
                b2a_field_value(field, row_index);
        }
    }
    return document;
}

std::vector<citlali::pipeline::AptDetectorColumnAddress>
b2a_layout() {
    return {{0, 0, 0}, {1, 0, 1}};
}

struct B2aBaselineBytes {
    b2a_apt::Document document;
    std::string bytes;
    std::string receipt;
};

B2aBaselineBytes b2a_baseline_bytes(
    std::string occurrence = "occurrence:sci-align/b2a-baseline#001",
    bool include_kids_flag = false) {
    auto document =
        b2a_baseline_document(std::move(occurrence), include_kids_flag);
    const auto serialized = b2a_apt::serialize_ecsv(document);
    return {std::move(document), serialized.bytes,
            b2a_observation::canonical_baseline_receipt_bytes(
                serialized.transport)};
}

std::string b2a_sha256_reference(char digit) {
    return "sha256:" + std::string(64, digit);
}

b2a_observation::SourceArtifact b2a_source(
    std::int64_t key, std::string role, char digest,
    b2a_apt::ObservationIdentity observation,
    std::string exact_content_sha256 = {},
    std::uint64_t exact_byte_count = 0) {
    if (exact_content_sha256.empty()) {
        exact_content_sha256 = b2a_sha256_reference(digest);
        exact_byte_count = static_cast<std::uint64_t>(1000 + key);
    }
    return {key,
            std::move(role),
            "synthetic/sci-align/toltec0",
            std::move(exact_content_sha256),
            exact_byte_count,
            observation,
            0,
            "toltec0",
            2};
}

struct B2aObservationBytes {
    b2a_observation::VerifiedBaselineDescriptor baseline;
    b2a_observation::TargetManifest target;
    b2a_observation::MatchRelation relation;
    b2a_observation::MatchedOutput output;
    std::string bytes;
    std::string receipt;
};

B2aObservationBytes b2a_observation_bytes(
    std::string raw_content_sha256 = {},
    std::uint64_t raw_byte_count = 0) {
    const auto baseline_wire = b2a_baseline_bytes(
        "occurrence:sci-align/b2a-observation-parent#001", true);
    auto baseline = b2a_observation::verify_baseline_descriptor(
        baseline_wire.bytes, baseline_wire.receipt);

    b2a_observation::TargetManifest target;
    target.envelope = {
        "occurrence:sci-align/b2a-target#001",
        "event:sci-align/b2a-target#001",
        "tolproj-synthetic-b2a",
        "tolproj-config:sci-align-b2a",
        "2026-08-15T12:01:00Z",
    };
    target.observation = {148671, 0, 2};
    target.inputs = {{10,
                      0,
                      "toltec0",
                      2,
                      b2a_source(100, "raw", '1', target.observation,
                                 std::move(raw_content_sha256),
                                 raw_byte_count),
                      b2a_source(101, "kmp", '2', {148600, 0, 0})}};
    target.registered_fields =
        b2a_observation::canonical_target_fields_v1();
    target.rows = {
        {5, 10, 101, 0, 1.0e9, 1.0001e9, 0, 0, 0, {}},
        {11, 10, 101, 1, 1.1e9, 1.1001e9, 0, 0, 1, {}},
    };
    for (std::size_t row = 0; row < target.rows.size(); ++row) {
        target.rows[row].fields["kids_fr"] =
            target.rows[row].matching_frequency_hz;
        target.rows[row].fields["kids_f_out"] =
            target.rows[row].output_tone_frequency_hz;
        target.rows[row].fields["kids_Qr"] =
            40000.0 + static_cast<double>(row);
        target.rows[row].fields["kids_flag"] =
            static_cast<std::int64_t>(row);
    }
    target.target_source_sequence = {11, 5};
    target.target_application_sequence = {5, 11};
    b2a_observation::validate(target);

    const auto target_identity = b2a_observation::artifact_identity(target);
    const auto baseline_identity =
        b2a_observation::artifact_identity(baseline);
    const auto target_ref = [&](std::int64_t key) {
        return b2a_observation::row_reference(target_identity, key);
    };
    const auto seed_ref = [&](std::int64_t key) {
        return b2a_observation::row_reference(baseline_identity, key);
    };

    b2a_observation::MatchRelation relation;
    relation.envelope = {
        "occurrence:sci-align/b2a-relation#001",
        "event:sci-align/b2a-relation#001",
        "tolproj-synthetic-b2a",
        "tolproj-match-config:sci-align-b2a",
        "2026-08-15T12:02:00Z",
    };
    relation.baseline_parent =
        b2a_observation::baseline_reference(baseline);
    relation.target_parent = target_identity;
    relation.matcher = {
        "occurrence:sci-align/b2a-matcher#001",
        "tolproj-synthetic-b2a",
        "tolproj-match-config:sci-align-b2a",
        "astropy",
        "join-distance-v1",
    };
    relation.network_evidence = {{0, 0.0, 200000.0, 40000.0}};
    relation.pairs = {
        {900, target_ref(5), seed_ref(42), 0.0, true},
        {901, target_ref(11), seed_ref(b2a_apt::uid_v1_max), 0.0, true},
    };
    relation.target_dispositions = {
        {1000, target_ref(5),
         b2a_observation::EndpointDispositionState::matched, {900},
         "synthetic exact match"},
        {1001, target_ref(11),
         b2a_observation::EndpointDispositionState::matched, {901},
         "synthetic exact match"},
    };
    relation.seed_dispositions = {
        {2000, seed_ref(42),
         b2a_observation::EndpointDispositionState::matched, {900},
         "synthetic exact match"},
        {2001, seed_ref(b2a_apt::uid_v1_max),
         b2a_observation::EndpointDispositionState::matched, {901},
         "synthetic exact match"},
    };
    relation.seed_source_sequence = {b2a_apt::uid_v1_max, 42};
    b2a_observation::validate(relation, baseline, target);

    const auto contracts =
        b2a_observation::canonical_output_field_contracts_v1(
            baseline, target);
    std::vector<b2a_observation::MatchedOutputFieldSource> selections;
    for (const auto &target_row : target.rows) {
        const auto pair_key = target_row.row_key == 5 ? 900 : 901;
        for (const auto &contract : contracts) {
            if (contract.authorized_operation ==
                b2a_observation::TransformationOperation::
                    copy_baseline_when_matched_null_when_unmatched) {
                selections.push_back(
                    {target_row.row_key, contract.field.name, pair_key});
            }
        }
    }
    const auto output = b2a_observation::make_matched_observation_output_v1(
        {"occurrence:sci-align/b2a-output#001",
         "event:sci-align/b2a-output#001",
         "tolproj-synthetic-b2a",
         "tolproj-output-config:sci-align-b2a",
         "2026-08-15T12:03:00Z"},
        baseline, target, relation, selections);
    const auto serialized =
        b2a_observation::serialize_matched_observation_ecsv(
            output, baseline, target, relation);
    const auto receipt = b2a_publication::canonical_receipt_bytes(
        b2a_publication::make_receipt_binding(
            std::string(b2a_publication::receipt_schema_v1),
            std::string(
                b2a_observation::matched_output_byte_transport_scope_v1),
            serialized.digests.envelope_sha256, serialized.bytes));
    return {std::move(baseline), std::move(target), std::move(relation),
            output, serialized.bytes, receipt};
}

class B2aTemporaryDirectory {
public:
    B2aTemporaryDirectory() {
        const auto nonce =
            std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
            ("citlali-sci-align-b2a-" + std::to_string(nonce));
        std::filesystem::create_directories(path);
    }

    ~B2aTemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }

    B2aTemporaryDirectory(const B2aTemporaryDirectory &) = delete;
    B2aTemporaryDirectory &operator=(const B2aTemporaryDirectory &) = delete;

    std::filesystem::path path;
};

void b2a_write_bytes(const std::filesystem::path &path,
                     std::string_view bytes) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("unable to create B2a fixture file");
    }
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!stream) {
        throw std::runtime_error("unable to write B2a fixture file");
    }
}

std::filesystem::path b2a_write_baseline(
    const B2aTemporaryDirectory &temporary,
    const B2aBaselineBytes &wire,
    std::string_view stem = "baseline") {
    const auto artifact = temporary.path /
        (std::string(stem) + ".ecsv");
    b2a_write_bytes(artifact, wire.bytes);
    b2a_write_bytes(std::filesystem::path(artifact.string() + ".sha256"),
                    wire.receipt);
    return artifact;
}

std::filesystem::path b2a_write_raw(
    const B2aTemporaryDirectory &temporary, int network,
    std::size_t detector_count, std::string_view stem = "toltec0") {
    const auto path = temporary.path / (std::string(stem) + ".nc");
    netCDF::NcFile file(path.string(), netCDF::NcFile::replace);
    auto roach = file.addVar("Header.Toltec.RoachIndex", netCDF::ncInt);
    roach.putVar(&network);
    const auto samples = file.addDim("sample", 1);
    const auto detectors = file.addDim("detector", detector_count);
    file.addVar("Data.Toltec.Is", netCDF::ncDouble,
                std::vector<netCDF::NcDim>{samples, detectors});
    return path;
}

struct B2aCalibSnapshot {
    std::string filepath;
    std::string apt_meta;
    std::map<std::string, std::vector<std::uint64_t>> apt_bits;
    std::shared_ptr<const citlali::pipeline::AptDetectorRelation> relation;
    Eigen::Index n_dets = 0;
    Eigen::Index n_nws = 0;
    Eigen::Index n_arrays = 0;
    std::vector<Eigen::Index> fg;
    std::vector<Eigen::Index> nws;
    std::vector<Eigen::Index> arrays;
    std::map<Eigen::Index, std::tuple<Eigen::Index, Eigen::Index>> nw_limits;
    std::map<Eigen::Index, std::tuple<Eigen::Index, Eigen::Index>> array_limits;
    std::map<Eigen::Index, std::vector<Eigen::Index>> nw_indices;
    std::map<Eigen::Index, std::vector<Eigen::Index>> array_indices;
    std::map<Eigen::Index, std::pair<std::uint64_t, std::uint64_t>>
        nw_fwhm_bits;
    std::map<Eigen::Index, std::pair<std::uint64_t, std::uint64_t>>
        array_fwhm_bits;
    std::map<Eigen::Index, std::uint64_t> nw_pa_bits;
    std::map<Eigen::Index, std::uint64_t> array_pa_bits;
    std::map<Eigen::Index, std::uint64_t> nw_beam_bits;
    std::map<Eigen::Index, std::uint64_t> array_beam_bits;
    std::vector<std::uint64_t> flux_bits;
    std::map<std::string, std::uint64_t> mean_flux_bits;
};

B2aCalibSnapshot b2a_snapshot(const engine::Calib &calib) {
    B2aCalibSnapshot result;
    result.filepath = calib.apt_filepath;
    result.apt_meta = YAML::Dump(calib.apt_meta);
    for (const auto &[name, values] : calib.apt) {
        auto &bits = result.apt_bits[name];
        bits.reserve(static_cast<std::size_t>(values.size()));
        for (Eigen::Index index = 0; index < values.size(); ++index) {
            bits.push_back(bits_of(values(index)));
        }
    }
    result.relation = calib.apt_detector_relation_handle();
    result.n_dets = calib.n_dets;
    result.n_nws = calib.n_nws;
    result.n_arrays = calib.n_arrays;
    result.fg.assign(calib.fg.data(), calib.fg.data() + calib.fg.size());
    result.nws.assign(calib.nws.data(), calib.nws.data() + calib.nws.size());
    result.arrays.assign(calib.arrays.data(),
                         calib.arrays.data() + calib.arrays.size());
    result.nw_limits = calib.nw_limits;
    result.array_limits = calib.array_limits;
    result.nw_indices = calib.nw_detector_indices;
    result.array_indices = calib.array_detector_indices;
    for (const auto &[key, value] : calib.nw_fwhms) {
        result.nw_fwhm_bits.emplace(
            key, std::pair{bits_of(std::get<0>(value)),
                           bits_of(std::get<1>(value))});
    }
    for (const auto &[key, value] : calib.array_fwhms) {
        result.array_fwhm_bits.emplace(
            key, std::pair{bits_of(std::get<0>(value)),
                           bits_of(std::get<1>(value))});
    }
    for (const auto &[key, value] : calib.nw_pas) {
        result.nw_pa_bits.emplace(key, bits_of(value));
    }
    for (const auto &[key, value] : calib.array_pas) {
        result.array_pa_bits.emplace(key, bits_of(value));
    }
    for (const auto &[key, value] : calib.nw_beam_areas) {
        result.nw_beam_bits.emplace(key, bits_of(value));
    }
    for (const auto &[key, value] : calib.array_beam_areas) {
        result.array_beam_bits.emplace(key, bits_of(value));
    }
    for (Eigen::Index index = 0;
         index < calib.flux_conversion_factor.size(); ++index) {
        result.flux_bits.push_back(
            bits_of(calib.flux_conversion_factor(index)));
    }
    for (const auto &[key, value] : calib.mean_flux_conversion_factor) {
        result.mean_flux_bits.emplace(key, bits_of(value));
    }
    return result;
}

void b2a_expect_snapshot_eq(const B2aCalibSnapshot &expected,
                            const engine::Calib &actual) {
    const auto observed = b2a_snapshot(actual);
    EXPECT_EQ(observed.filepath, expected.filepath);
    EXPECT_EQ(observed.apt_meta, expected.apt_meta);
    EXPECT_EQ(observed.apt_bits, expected.apt_bits);
    EXPECT_EQ(observed.relation, expected.relation);
    EXPECT_EQ(observed.n_dets, expected.n_dets);
    EXPECT_EQ(observed.n_nws, expected.n_nws);
    EXPECT_EQ(observed.n_arrays, expected.n_arrays);
    EXPECT_EQ(observed.fg, expected.fg);
    EXPECT_EQ(observed.nws, expected.nws);
    EXPECT_EQ(observed.arrays, expected.arrays);
    EXPECT_EQ(observed.nw_limits, expected.nw_limits);
    EXPECT_EQ(observed.array_limits, expected.array_limits);
    EXPECT_EQ(observed.nw_indices, expected.nw_indices);
    EXPECT_EQ(observed.array_indices, expected.array_indices);
    EXPECT_EQ(observed.nw_fwhm_bits, expected.nw_fwhm_bits);
    EXPECT_EQ(observed.array_fwhm_bits, expected.array_fwhm_bits);
    EXPECT_EQ(observed.nw_pa_bits, expected.nw_pa_bits);
    EXPECT_EQ(observed.array_pa_bits, expected.array_pa_bits);
    EXPECT_EQ(observed.nw_beam_bits, expected.nw_beam_bits);
    EXPECT_EQ(observed.array_beam_bits, expected.array_beam_bits);
    EXPECT_EQ(observed.flux_bits, expected.flux_bits);
    EXPECT_EQ(observed.mean_flux_bits, expected.mean_flux_bits);
}

std::string b2a_replace_uid(std::string bytes, std::int64_t replacement) {
    const std::string needle =
        "\n" + std::to_string(b2a_apt::uid_v1_max) + ",";
    const auto position = bytes.find(needle);
    if (position == std::string::npos) {
        throw std::logic_error("B2a fixture cannot find its maximum UID row");
    }
    bytes.replace(position + 1,
                  std::to_string(b2a_apt::uid_v1_max).size(),
                  std::to_string(replacement));
    return bytes;
}

std::string b2a_receipt_for_tampered_baseline(
    std::string_view bytes, std::string_view envelope_sha256) {
    return b2a_observation::canonical_baseline_receipt_bytes(
        b2a_apt::make_byte_transport_hash(bytes, envelope_sha256));
}

TEST(sci_align_typed_apt_identity,
     baseline_admission_binds_verified_receipt_and_explicit_detector_columns) {
    const auto wire = b2a_baseline_bytes();
    const auto relation =
        citlali::pipeline::admit_published_baseline_apt_relation(
            wire.bytes, wire.receipt, b2a_layout());
    ASSERT_EQ(relation.scope_kind(),
              citlali::pipeline::AptDetectorScopeKind::published_artifact);
    ASSERT_EQ(relation.bindings().size(), 2U);
    EXPECT_EQ(relation.binding_for_column(0).uid, 42);
    EXPECT_EQ(relation.binding_for_column(0).network, 0);
    EXPECT_EQ(relation.binding_for_column(0).kids_tone, 0);
    EXPECT_EQ(relation.binding_for_column(1).uid,
              b2a_apt::uid_v1_max);
    EXPECT_EQ(relation.binding_for_column(1).kids_tone, 1);
    EXPECT_EQ(relation.published_scope().kind,
              citlali::pipeline::PublishedAptKind::canonical_baseline);
    EXPECT_EQ(relation.published_scope().artifact.occurrence,
              wire.document.envelope.occurrence);
    EXPECT_EQ(relation.published_scope().transport.byte_count,
              wire.bytes.size());
    EXPECT_EQ(relation.published_scope().receipt_sha256,
              "sha256:" + citlali::utils::sha256(wire.receipt));
    EXPECT_EQ(relation.published_scope().receipt_byte_count,
              wire.receipt.size());
    EXPECT_FALSE(relation.requires_published_artifact_join());

    B2aTemporaryDirectory temporary;
    const auto artifact = b2a_write_baseline(temporary, wire);
    const auto raw = b2a_write_raw(temporary, 0, 2);
    const std::vector<std::string> raw_files{raw.string()};
    const std::vector<std::string> interfaces{"toltec0"};
    engine::Calib calib;
    ASSERT_NO_THROW(calib.get_canonical_baseline_apt(
        artifact.string(), raw_files, interfaces));
    ASSERT_TRUE(calib.has_apt_detector_relation());
    const auto handle = calib.apt_detector_relation_handle();
    ASSERT_NE(handle, nullptr);
    EXPECT_EQ(&calib.require_apt_detector_relation(), handle.get());
    EXPECT_EQ(handle->binding_for_column(0).uid, 42);
    EXPECT_EQ(handle->binding_for_column(1).uid, b2a_apt::uid_v1_max);
    EXPECT_DOUBLE_EQ(calib.apt.at("uid")(0), 42.0);
    EXPECT_DOUBLE_EQ(calib.apt.at("uid")(1),
                     static_cast<double>(b2a_apt::uid_v1_max));
    EXPECT_DOUBLE_EQ(calib.apt.at("kids_tone")(0), 0.0);
    EXPECT_DOUBLE_EQ(calib.apt.at("kids_tone")(1), 1.0);
}

TEST(sci_align_typed_apt_identity,
     synthetic_observation_admission_binds_output_and_verified_parent_scopes) {
    B2aTemporaryDirectory temporary;
    const auto raw = b2a_write_raw(temporary, 0, 2);
    const auto fixture = b2a_observation_bytes(
        "sha256:" + citlali::utils::sha256_file(raw),
        std::filesystem::file_size(raw));
    const auto relation =
        citlali::pipeline::admit_published_observation_apt_relation(
            fixture.bytes, fixture.receipt, fixture.baseline, b2a_layout());
    const auto &scope = relation.published_scope();
    EXPECT_EQ(scope.kind,
              citlali::pipeline::PublishedAptKind::matched_observation);
    EXPECT_EQ(scope.artifact,
              b2a_observation::artifact_identity(
                  fixture.output, fixture.baseline, fixture.target,
                  fixture.relation));
    EXPECT_EQ(scope.baseline_parent, fixture.output.baseline_parent);
    ASSERT_TRUE(scope.target_parent.has_value());
    ASSERT_TRUE(scope.relation_parent.has_value());
    EXPECT_EQ(*scope.target_parent, fixture.output.target_parent);
    EXPECT_EQ(*scope.relation_parent, fixture.output.relation_parent);
    ASSERT_EQ(relation.bindings().size(), 2U);
    EXPECT_EQ(relation.binding_for_column(0).network, 0);
    EXPECT_EQ(relation.binding_for_column(0).kids_tone, 0);
    EXPECT_EQ(relation.binding_for_column(0).uid, 0);
    EXPECT_EQ(relation.binding_for_column(1).uid, 1);
    EXPECT_TRUE(relation.binding_for_column(0).flag.has_value());
    EXPECT_TRUE(relation.binding_for_column(1).flag.has_value());

    // This is deliberately synthetic APT-PROD-002 evidence. It does not
    // claim real matched-observation conformance.
    const auto baseline_path = temporary.path / "parent.ecsv";
    b2a_write_bytes(baseline_path, fixture.baseline.baseline_bytes());
    b2a_write_bytes(
        std::filesystem::path(baseline_path.string() + ".sha256"),
        fixture.baseline.receipt_bytes());
    const auto output_path = temporary.path / "observation.ecsv";
    b2a_write_bytes(output_path, fixture.bytes);
    b2a_write_bytes(
        std::filesystem::path(output_path.string() + ".sha256"),
        fixture.receipt);
    const std::vector<std::string> raw_files{raw.string()};
    const std::vector<std::string> interfaces{"toltec0"};
    engine::Calib calib;
    ASSERT_NO_THROW(calib.get_canonical_observation_apt(
        output_path.string(), baseline_path.string(), raw_files,
        interfaces));
    EXPECT_EQ(calib.require_apt_detector_relation().published_scope().artifact,
              scope.artifact);
}

TEST(sci_align_typed_apt_identity,
     exact_int64_boundary_is_preserved_and_out_of_range_keys_fail_atomically) {
    const auto wire = b2a_baseline_bytes();
    B2aTemporaryDirectory temporary;
    const auto artifact = b2a_write_baseline(temporary, wire);
    const auto raw = b2a_write_raw(temporary, 0, 2);
    const std::vector<std::string> raw_files{raw.string()};
    const std::vector<std::string> interfaces{"toltec0"};
    engine::Calib calib;
    ASSERT_NO_THROW(calib.get_canonical_baseline_apt(
        artifact.string(), raw_files, interfaces));
    const auto &binding =
        calib.require_apt_detector_relation().binding_for_column(1);
    EXPECT_EQ(binding.uid, 9007199254740991LL);
    EXPECT_EQ(binding.uid, b2a_apt::uid_v1_max);
    const auto snapshot = b2a_snapshot(calib);
    const auto envelope = b2a_apt::compute_digests(wire.document)
                              .envelope_sha256;

    for (const auto invalid_uid :
         {b2a_apt::uid_v1_max + 1,
          std::numeric_limits<std::int64_t>::max()}) {
        const auto invalid_bytes = b2a_replace_uid(wire.bytes, invalid_uid);
        b2a_write_bytes(artifact, invalid_bytes);
        b2a_write_bytes(
            std::filesystem::path(artifact.string() + ".sha256"),
            b2a_receipt_for_tampered_baseline(invalid_bytes, envelope));
        EXPECT_THROW(calib.get_canonical_baseline_apt(
                         artifact.string(), raw_files, interfaces),
                     std::exception);
        b2a_expect_snapshot_eq(snapshot, calib);
    }
}

TEST(sci_align_typed_apt_identity,
     presentation_permutations_preserve_explicit_native_relation) {
    auto first_document = b2a_baseline_document(
        "occurrence:sci-align/b2a-permutation#001");
    auto second_document = first_document;
    std::reverse(second_document.rows.begin(), second_document.rows.end());
    std::reverse(second_document.raw_manifest.inputs.begin(),
                 second_document.raw_manifest.inputs.end());
    std::reverse(second_document.registered_fields.begin(),
                 second_document.registered_fields.end());
    const auto first_wire = b2a_apt::serialize_ecsv(first_document);
    const auto second_wire = b2a_apt::serialize_ecsv(second_document);
    const auto first =
        citlali::pipeline::admit_published_baseline_apt_relation(
            first_wire.bytes,
            b2a_observation::canonical_baseline_receipt_bytes(
                first_wire.transport),
            b2a_layout());
    auto reversed_layout = b2a_layout();
    std::reverse(reversed_layout.begin(), reversed_layout.end());
    const auto second =
        citlali::pipeline::admit_published_baseline_apt_relation(
            second_wire.bytes,
            b2a_observation::canonical_baseline_receipt_bytes(
                second_wire.transport),
            reversed_layout);
    EXPECT_EQ(first.bindings(), second.bindings());
    EXPECT_EQ(first.binding_for_column(0).uid, 42);
    EXPECT_EQ(first.binding_for_column(1).uid, b2a_apt::uid_v1_max);
    EXPECT_EQ(first.binding_for_column(0).kids_tone, 0);
    EXPECT_EQ(first.binding_for_column(1).kids_tone, 1);
}

TEST(sci_align_typed_apt_identity,
     duplicate_wrong_network_channel_and_input_order_fail_atomically) {
    const auto wire = b2a_baseline_bytes();
    B2aTemporaryDirectory temporary;
    const auto artifact = b2a_write_baseline(temporary, wire);
    const auto raw = b2a_write_raw(temporary, 0, 2);
    const std::vector<std::string> raw_files{raw.string()};
    const std::vector<std::string> interfaces{"toltec0"};
    engine::Calib calib;
    ASSERT_NO_THROW(calib.get_canonical_baseline_apt(
        artifact.string(), raw_files, interfaces));
    const auto snapshot = b2a_snapshot(calib);

    auto duplicate_uid = wire.document;
    duplicate_uid.rows[1].uid = duplicate_uid.rows[0].uid;
    EXPECT_THROW(b2a_apt::serialize_ecsv(duplicate_uid), std::exception);
    b2a_expect_snapshot_eq(snapshot, calib);

    EXPECT_THROW(
        citlali::pipeline::admit_published_baseline_apt_relation(
            wire.bytes, wire.receipt, {{0, 0, 0}, {1, 0, 0}}),
        std::exception);
    EXPECT_THROW(
        citlali::pipeline::admit_published_baseline_apt_relation(
            wire.bytes, wire.receipt, {{0, 0, 0}, {1, 7, 0}}),
        std::exception);
    EXPECT_THROW(
        citlali::pipeline::admit_published_baseline_apt_relation(
            wire.bytes, wire.receipt,
            {{0, 0, 0}, {1, 0, 1}, {2, 0, 2}}),
        std::exception);
    b2a_expect_snapshot_eq(snapshot, calib);

    const std::vector<std::string> wrong_interface{"toltec1"};
    EXPECT_THROW(calib.get_canonical_baseline_apt(
                     artifact.string(), raw_files, wrong_interface),
                 std::exception);
    b2a_expect_snapshot_eq(snapshot, calib);

    const auto short_raw =
        b2a_write_raw(temporary, 0, 1, "toltec0-short");
    const std::vector<std::string> short_files{short_raw.string()};
    EXPECT_THROW(calib.get_canonical_baseline_apt(
                     artifact.string(), short_files, interfaces),
                 std::exception);
    b2a_expect_snapshot_eq(snapshot, calib);
}

TEST(sci_align_typed_apt_identity,
     receipt_tamper_stale_and_cross_scope_reuse_fail_atomically) {
    const auto wire = b2a_baseline_bytes();
    B2aTemporaryDirectory temporary;
    const auto artifact = b2a_write_baseline(temporary, wire);
    const auto raw = b2a_write_raw(temporary, 0, 2);
    const std::vector<std::string> raw_files{raw.string()};
    const std::vector<std::string> interfaces{"toltec0"};
    engine::Calib calib;
    ASSERT_NO_THROW(calib.get_canonical_baseline_apt(
        artifact.string(), raw_files, interfaces));
    const auto snapshot = b2a_snapshot(calib);

    auto tampered_receipt = wire.receipt;
    const auto digest = tampered_receipt.find("byte_sha256=sha256:");
    ASSERT_NE(digest, std::string::npos);
    const auto digit = digest + std::string("byte_sha256=sha256:").size();
    tampered_receipt[digit] = tampered_receipt[digit] == '0' ? '1' : '0';
    b2a_write_bytes(
        std::filesystem::path(artifact.string() + ".sha256"),
        tampered_receipt);
    EXPECT_THROW(calib.get_canonical_baseline_apt(
                     artifact.string(), raw_files, interfaces),
                 std::exception);
    b2a_expect_snapshot_eq(snapshot, calib);

    b2a_write_bytes(
        std::filesystem::path(artifact.string() + ".sha256"), wire.receipt);
    auto tampered_bytes = wire.bytes;
    const auto row = tampered_bytes.find("\n42,");
    ASSERT_NE(row, std::string::npos);
    tampered_bytes[row + 1] = '4';
    tampered_bytes[row + 2] = '3';
    b2a_write_bytes(artifact, tampered_bytes);
    b2a_write_bytes(
        std::filesystem::path(artifact.string() + ".sha256"),
        b2a_receipt_for_tampered_baseline(
            tampered_bytes,
            b2a_apt::compute_digests(wire.document).envelope_sha256));
    EXPECT_THROW(calib.get_canonical_baseline_apt(
                     artifact.string(), raw_files, interfaces),
                 std::exception);
    b2a_expect_snapshot_eq(snapshot, calib);

    const auto first =
        citlali::pipeline::admit_published_baseline_apt_relation(
            wire.bytes, wire.receipt, b2a_layout());
    const auto other_wire = b2a_baseline_bytes(
        "occurrence:sci-align/b2a-baseline#other");
    const auto other =
        citlali::pipeline::admit_published_baseline_apt_relation(
            other_wire.bytes, other_wire.receipt, b2a_layout());
    EXPECT_FALSE(first.same_scope(other));
    const auto first_reference = first.binding_reference_for_column(0);
    EXPECT_THROW(other.require_binding(first_reference), std::exception);

    const auto observation = b2a_observation_bytes();
    const auto foreign_parent_wire = b2a_baseline_bytes(
        "occurrence:sci-align/b2a-foreign-parent#001", true);
    const auto foreign_parent =
        b2a_observation::verify_baseline_descriptor(
            foreign_parent_wire.bytes, foreign_parent_wire.receipt);
    EXPECT_THROW(
        citlali::pipeline::admit_published_observation_apt_relation(
            observation.bytes, observation.receipt, foreign_parent,
            b2a_layout()),
        std::exception);
    b2a_expect_snapshot_eq(snapshot, calib);

    const auto output_path = temporary.path / "foreign-parent-output.ecsv";
    b2a_write_bytes(output_path, observation.bytes);
    b2a_write_bytes(
        std::filesystem::path(output_path.string() + ".sha256"),
        observation.receipt);
    const auto foreign_path =
        b2a_write_baseline(temporary, foreign_parent_wire,
                           "foreign-parent");
    EXPECT_THROW(calib.get_canonical_observation_apt(
                     output_path.string(), foreign_path.string(), raw_files,
                     interfaces),
                 std::exception);
    b2a_expect_snapshot_eq(snapshot, calib);
}

TEST(sci_align_typed_apt_identity,
     producer_raw_manifest_scope_is_explicit_nonfinal_and_injective) {
    b2a_apt::RawManifest manifest;
    manifest.observation = {148670, 0, 2};
    manifest.inputs = {{0, "toltec0", 2}};
    const std::vector<citlali::pipeline::AptDetectorBinding> bindings{
        {0, 42, 0, 0, 0},
        {1, b2a_apt::uid_v1_max, 0, 1, 0},
    };
    const auto relation =
        citlali::pipeline::admit_producer_raw_manifest_relation(
            "producer:sci-align/raw-manifest#001", manifest, bindings);
    EXPECT_EQ(relation.scope_kind(),
              citlali::pipeline::AptDetectorScopeKind::producer_raw_manifest);
    EXPECT_TRUE(relation.requires_published_artifact_join());
    EXPECT_TRUE(relation.producer_scope()
                    .requires_published_artifact_join());
    EXPECT_EQ(relation.bindings(), bindings);
    EXPECT_THROW(relation.published_scope(), std::exception);

    const auto other =
        citlali::pipeline::admit_producer_raw_manifest_relation(
            "producer:sci-align/raw-manifest#002", manifest, bindings);
    EXPECT_FALSE(relation.same_scope(other));
    EXPECT_THROW(
        other.require_binding(relation.binding_reference_for_column(0)),
        std::exception);

    auto invalid = bindings;
    invalid[1].uid = invalid[0].uid;
    EXPECT_THROW(citlali::pipeline::admit_producer_raw_manifest_relation(
                     "producer:sci-align/duplicate-uid", manifest, invalid),
                 std::exception);
    invalid = bindings;
    invalid[1].kids_tone = invalid[0].kids_tone;
    EXPECT_THROW(citlali::pipeline::admit_producer_raw_manifest_relation(
                     "producer:sci-align/duplicate-channel", manifest,
                     invalid),
                 std::exception);
    invalid = bindings;
    invalid[1].network = 7;
    EXPECT_THROW(citlali::pipeline::admit_producer_raw_manifest_relation(
                     "producer:sci-align/wrong-network", manifest, invalid),
                 std::exception);
    invalid = bindings;
    invalid[1].detector_column = 2;
    EXPECT_THROW(citlali::pipeline::admit_producer_raw_manifest_relation(
                     "producer:sci-align/gapped-column", manifest, invalid),
                 std::exception);
    invalid = bindings;
    invalid[1].flag.reset();
    EXPECT_THROW(citlali::pipeline::admit_producer_raw_manifest_relation(
                     "producer:sci-align/missing-flag", manifest, invalid),
                 std::exception);
    invalid = bindings;
    invalid[1].flag = 2;
    EXPECT_THROW(citlali::pipeline::admit_producer_raw_manifest_relation(
                     "producer:sci-align/invalid-flag", manifest, invalid),
                 std::exception);
    auto invalid_network_manifest = manifest;
    invalid_network_manifest.inputs = {{13, "toltec13", 2}};
    invalid = bindings;
    invalid[0].network = 13;
    invalid[1].network = 13;
    EXPECT_THROW(citlali::pipeline::admit_producer_raw_manifest_relation(
                     "producer:sci-align/invalid-network",
                     invalid_network_manifest, invalid),
                 std::exception);
    EXPECT_THROW(citlali::pipeline::admit_producer_raw_manifest_relation(
                     "producer:sci-align/missing", manifest,
                     std::vector<citlali::pipeline::AptDetectorBinding>{
                         bindings.front()}),
                 std::exception);
    EXPECT_EQ(relation.bindings(), bindings);
}

}  // namespace

#include <citlali/core/mapmaking/jinc_mm.h>
#include <citlali/core/mapmaking/naive_mm.h>
#include <citlali/core/pipeline/map_buffer_allocation.h>
#include <citlali/core/pipeline/science_map_identity.h>
#include <citlali/core/timestream/rtc/kernel.h>

namespace {

namespace b2b_apt = citlali::pipeline::canonical_apt_v1;

using B2bPtcData =
    timestream::TCData<timestream::TCDataKind::PTC, Eigen::MatrixXd>;
using B2bRtcData =
    timestream::TCData<timestream::TCDataKind::RTC, Eigen::MatrixXd>;
using B2bApt = std::map<std::string, Eigen::VectorXd>;

struct B2bMeasuredMaterial {
    Eigen::MatrixXd values;
    std::vector<citlali::pipeline::NativeDetectorBlock> blocks;
};

struct B2bNativeFixture {
    std::shared_ptr<const citlali::pipeline::NativeAlignmentPlan> alignment;
    std::shared_ptr<const citlali::pipeline::RawTelescopeTrajectory>
        raw_telescope;
    std::shared_ptr<const citlali::pipeline::NativePointingPlan> pointing;
    std::shared_ptr<const citlali::pipeline::AptDetectorRelation> relation;
    std::vector<citlali::pipeline::NativeCompleteCohortRun> runs;
    std::shared_ptr<const citlali::pipeline::NativeMeasuredScanState> state;
    B2bPtcData data;
    B2bApt apt;
};

citlali::pipeline::NativeTelescopeData b2b_telescope_series(
    const Eigen::VectorXd &times) {
    citlali::pipeline::NativeTelescopeData result;
    result["TelTime"] = times;
    result["TelUTC"] = times;
    for (const auto *key : {"TelAzAct", "TelElAct", "lat_phys",
                            "lon_phys", "alt_phys", "az_phys"}) {
        result[key].resize(times.size());
    }
    for (Eigen::Index row = 0; row < times.size(); ++row) {
        const double dt = times(row) - 1000.0;
        result["TelAzAct"](row) = 0.7 + 0.01 * dt;
        result["TelElAct"](row) = 0.4 - 0.02 * dt;
        result["lat_phys"](row) = 0.002 * dt;
        result["lon_phys"](row) = -0.0015 * dt;
        result["alt_phys"](row) = result["lat_phys"](row);
        result["az_phys"](row) = result["lon_phys"](row);
    }
    return result;
}

citlali::pipeline::NativePointingOffsetsArcsec b2b_pointing_offsets(
    const Eigen::VectorXd &times) {
    citlali::pipeline::NativePointingOffsetsArcsec result;
    result["az"].resize(times.size());
    result["alt"].resize(times.size());
    for (Eigen::Index row = 0; row < times.size(); ++row) {
        const double dt = times(row) - 1000.0;
        result["az"](row) = 0.25 * dt;
        result["alt"](row) = -0.1 * dt;
    }
    return result;
}

std::shared_ptr<const citlali::pipeline::AptDetectorRelation>
b2b_detector_relation(const std::string &scope, bool reversed = false) {
    b2b_apt::RawManifest manifest;
    manifest.observation = {148670, 0, 2};
    manifest.inputs = {{0, "toltec0", 1}, {7, "toltec7", 1}};
    std::vector<citlali::pipeline::AptDetectorBinding> bindings{
        {0, 42, 0, 0, 0},
        {1, b2b_apt::uid_v1_max, 7, 0, 0},
    };
    if (reversed) {
        std::reverse(bindings.begin(), bindings.end());
    }
    return std::make_shared<const citlali::pipeline::AptDetectorRelation>(
        citlali::pipeline::admit_producer_raw_manifest_relation(
            scope, std::move(manifest), std::move(bindings)));
}

std::vector<citlali::pipeline::NativeMeasuredScanSegment>
b2b_segments(
    const std::vector<citlali::pipeline::NativeCompleteCohortRun> &runs) {
    std::vector<citlali::pipeline::NativeMeasuredScanSegment> result;
    Eigen::Index first_output_row = 0;
    for (const auto &run : runs) {
        const Eigen::Index row_count = static_cast<Eigen::Index>(
            run.selection.relational_common_slots().size());
        result.emplace_back(
            run.run_ordinal, first_output_row,
            first_output_row + row_count,
            run.selection.relational_common_slots(), run.selection,
            run.participant_runs);
        first_output_row += row_count;
    }
    return result;
}

B2bMeasuredMaterial b2b_measured_material(
    const citlali::pipeline::NativeAlignmentPlan &alignment,
    const citlali::pipeline::AptDetectorRelation &relation,
    const std::vector<citlali::pipeline::NativeCompleteCohortRun> &runs) {
    Eigen::Index row_count = 0;
    for (const auto &run : runs) {
        row_count += static_cast<Eigen::Index>(
            run.selection.relational_common_slots().size());
    }
    B2bMeasuredMaterial result;
    result.values = Eigen::MatrixXd::Zero(
        row_count, static_cast<Eigen::Index>(relation.bindings().size()));

    Eigen::Index first_output_row = 0;
    for (const auto &run : runs) {
        const Eigen::Index run_rows = static_cast<Eigen::Index>(
            run.selection.relational_common_slots().size());
        for (const auto &participant : run.participant_runs) {
            std::vector<Eigen::Index> columns;
            for (const auto &binding : relation.bindings()) {
                if (binding.network == participant.network_id) {
                    columns.push_back(static_cast<Eigen::Index>(
                        binding.detector_column));
                }
            }
            if (columns.empty()) {
                throw std::logic_error(
                    "B2b test material lacks a participant detector");
            }
            Eigen::MatrixXd block_values(run_rows, columns.size());
            for (Eigen::Index local = 0; local < run_rows; ++local) {
                const auto native_row =
                    participant.first_native_row + local;
                for (Eigen::Index local_column = 0;
                     local_column < block_values.cols(); ++local_column) {
                    const auto column = columns.at(
                        static_cast<std::size_t>(local_column));
                    const double value =
                        1.0 + 0.01 * static_cast<double>(native_row) +
                        0.125 * static_cast<double>(column);
                    block_values(local, local_column) = value;
                    result.values(first_output_row + local, column) = value;
                }
            }
            auto flags =
                citlali::pipeline::NativeDetectorFlagBitsMatrix::Zero(
                    block_values.rows(), block_values.cols());
            result.blocks.emplace_back(
                alignment.network(participant.network_id),
                participant.first_native_row, columns.front(),
                std::move(block_values), std::move(flags));
        }
        first_output_row += run_rows;
    }
    return result;
}

B2bNativeFixture b2b_native_fixture(
    bool dropped_packet = false, bool identical_native_times = false,
    double common_slot_perturbation = 0.0,
    bool reverse_relation_presentation = false,
    std::string relation_scope =
        "producer:sci-align/b2b-native-fixture#001") {
    Eigen::VectorXd times0(5);
    times0 << 1000.0, 1000.01, 1000.02, 1000.03, 1000.04;
    std::vector<citlali::pipeline::TimestreamPacketCounter> counters0{
        100, 101, 102, 103, 104};

    Eigen::VectorXd times7(dropped_packet ? 4 : 5);
    std::vector<citlali::pipeline::TimestreamPacketCounter> counters7;
    if (dropped_packet) {
        times7 << 1000.0025, 1000.0125, 1000.0325, 1000.0425;
        counters7 = {700, 701, 703, 704};
    }
    else if (identical_native_times) {
        times7 = times0;
        counters7 = {700, 701, 702, 703, 704};
    }
    else {
        times7 = times0.array() + 0.0025;
        counters7 = {700, 701, 702, 703, 704};
    }

    citlali::pipeline::NativeNetworkAlignment network0{
        0, 100, times0, counters0};
    citlali::pipeline::NativeNetworkAlignment network7{
        7, 700, times7, counters7};
    Eigen::VectorXd common(5);
    common << 1000.0,
        1000.01 + common_slot_perturbation,
        1000.02 - common_slot_perturbation,
        1000.03 + common_slot_perturbation,
        1000.04;
    auto associations0 =
        citlali::pipeline::make_direct_native_slot_associations(100, 5);
    auto associations7 =
        citlali::pipeline::make_direct_native_slot_associations(700, 5);
    if (dropped_packet) {
        associations7[2] = {
            -1,
            citlali::pipeline::CoincidenceAbsenceReason::
                participant_unavailable};
        associations7[3].native_row = 702;
        associations7[4].native_row = 703;
    }
    auto alignment =
        std::make_shared<const citlali::pipeline::NativeAlignmentPlan>(
            std::vector<citlali::pipeline::NativeNetworkAlignment>{
                network0, network7},
            common,
            std::vector<std::vector<
                citlali::pipeline::NativeSlotAssociation>>{
                associations0, associations7});

    Eigen::VectorXd raw_times =
        Eigen::VectorXd::LinSpaced(9, 999.99, 1000.06);
    auto raw_telescope =
        std::make_shared<const citlali::pipeline::RawTelescopeTrajectory>(
            b2b_telescope_series(raw_times));
    std::vector<citlali::pipeline::NativeNetworkPointing> networks;
    for (const auto network_id : alignment->participant_network_ids()) {
        const auto &network = alignment->network(network_id);
        const auto &targets = network.reconstructed_times_unix_sec();
        networks.emplace_back(
            network_id, network.first_native_row(), targets,
            citlali::pipeline::evaluate_raw_telescope_trajectory_at(
                *raw_telescope, targets),
            b2b_pointing_offsets(targets));
    }
    auto pointing =
        std::make_shared<const citlali::pipeline::NativePointingPlan>(
            alignment, raw_telescope, std::move(networks));
    auto relation = b2b_detector_relation(
        relation_scope, reverse_relation_presentation);

    const citlali::pipeline::NativeOperationIdentity operation{0, 0};
    auto runs =
        citlali::pipeline::partition_complete_native_cohort_runs(
            *alignment, operation, 0, alignment->slot_count(), 0);
    auto segments = b2b_segments(runs);
    auto material = b2b_measured_material(*alignment, *relation, runs);
    const Eigen::Index admitted_rows = material.values.rows();
    auto state =
        citlali::pipeline::NativeMeasuredScanState::admit(
            operation, alignment, pointing, relation, 0, admitted_rows,
            std::move(segments), std::move(material.blocks));

    B2bPtcData data;
    data.scans.data = std::move(material.values);
    data.flags.data = Eigen::Matrix<
        bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(
        data.scans.data.rows(), data.scans.data.cols(), false);
    data.weights.data = Eigen::VectorXd::Ones(data.scans.data.cols());
    data.index.data = 0;
    data.native_science_mode =
        timestream::NativeScienceMode::native_required;
    data.native_scan = state;
    // These sentinels would make any accidental common-time fallback
    // immediately observable in the pointing and projection tests.
    data.tel_data.data["TelElAct"] =
        Eigen::VectorXd::Constant(data.scans.data.rows(), 0.0);
    data.tel_data.data["alt_phys"] =
        Eigen::VectorXd::Constant(data.scans.data.rows(), 0.5);
    data.tel_data.data["az_phys"] =
        Eigen::VectorXd::Constant(data.scans.data.rows(), -0.5);
    data.pointing_offsets_arcsec.data["az"] =
        Eigen::VectorXd::Constant(data.scans.data.rows(), 1.0e6);
    data.pointing_offsets_arcsec.data["alt"] =
        Eigen::VectorXd::Constant(data.scans.data.rows(), -1.0e6);

    B2bApt apt;
    apt["array"] = Eigen::VectorXd::Zero(data.scans.data.cols());
    apt["flag"] = Eigen::VectorXd::Zero(data.scans.data.cols());
    apt["x_t"] = Eigen::VectorXd::Zero(data.scans.data.cols());
    apt["y_t"] = Eigen::VectorXd::Zero(data.scans.data.cols());
    apt["uid"] = Eigen::VectorXd::Constant(
        data.scans.data.cols(),
        std::numeric_limits<double>::quiet_NaN());
    return B2bNativeFixture{
        std::move(alignment), std::move(raw_telescope),
        std::move(pointing), std::move(relation), std::move(runs),
        std::move(state), std::move(data), std::move(apt)};
}

B2bRtcData b2b_as_rtc(const B2bNativeFixture &fixture) {
    B2bRtcData result;
    result.scans.data = fixture.data.scans.data;
    result.flags.data = fixture.data.flags.data;
    result.index.data = fixture.data.index.data;
    result.native_science_mode = fixture.data.native_science_mode;
    result.native_scan = fixture.data.native_scan;
    result.tel_data.data = fixture.data.tel_data.data;
    result.pointing_offsets_arcsec.data =
        fixture.data.pointing_offsets_arcsec.data;
    return result;
}

mapmaking::MapBuffer b2b_map_buffer(bool jinc) {
    mapmaking::MapBuffer result{"omb"};
    result.n_rows = 65;
    result.n_cols = 65;
    result.n_noise = 0;
    result.pixel_size_rad = 5.0e-6;
    result.map_grouping = "array";
    result.parallel_policy = "seq";
    result.randomize_dets = false;
    result.cov_cut = 0.0;
    citlali::pipeline::allocate_map_matrices(
        result, 1, jinc, false, true, false, {}, jinc);
    return result;
}

void b2b_expect_vector_exact(const Eigen::VectorXd &lhs,
                             const Eigen::VectorXd &rhs) {
    ASSERT_EQ(lhs.size(), rhs.size());
    for (Eigen::Index row = 0; row < lhs.size(); ++row) {
        EXPECT_EQ(bits_of(lhs(row)), bits_of(rhs(row))) << "row=" << row;
    }
}

TEST(sci_align_native_pointing_consumers,
     subcadence_pointing_is_invariant_to_common_slot_perturbation) {
    auto first = b2b_native_fixture(false, false, 0.0);
    auto perturbed = b2b_native_fixture(
        false, false, 0.001, false,
        "producer:sci-align/b2b-common-perturb#001");
    std::string pixel_axes = "altaz";
    std::string grouping = "array";

    for (Eigen::Index detector = 0;
         detector < first.data.scans.data.cols(); ++detector) {
        auto [first_lat, first_lon] =
            engine_utils::calc_det_pointing_for_science_sample(
                first.data, detector, 0.0, 0.0, pixel_axes, grouping);
        auto [perturbed_lat, perturbed_lon] =
            engine_utils::calc_det_pointing_for_science_sample(
                perturbed.data, detector, 0.0, 0.0, pixel_axes,
                grouping);
        b2b_expect_vector_exact(first_lat, perturbed_lat);
        b2b_expect_vector_exact(first_lon, perturbed_lon);
        EXPECT_LT(first_lat.cwiseAbs().maxCoeff(), 1.0e-3);
        EXPECT_LT(first_lon.cwiseAbs().maxCoeff(), 1.0e-3);
    }

    auto [network0_lat, network0_lon] =
        engine_utils::calc_det_pointing_for_science_sample(
            first.data, 0, 0.0, 0.0, pixel_axes, grouping);
    auto [network7_lat, network7_lon] =
        engine_utils::calc_det_pointing_for_science_sample(
            first.data, 1, 0.0, 0.0, pixel_axes, grouping);
    EXPECT_NE(bits_of(network0_lat(0)), bits_of(network7_lat(0)));
    EXPECT_NE(bits_of(network0_lon(0)), bits_of(network7_lon(0)));
    EXPECT_DOUBLE_EQ(
        first.state->require_cell(0, 0)
            .identity.reconstructed_time_unix_sec(),
        1000.0);
    EXPECT_DOUBLE_EQ(
        first.state->require_cell(0, 1)
            .identity.reconstructed_time_unix_sec(),
        1000.0025);
}

TEST(sci_align_native_pointing_consumers,
     dropped_packet_gap_creates_nothing_and_cannot_bridge_native_runs) {
    auto fixture = b2b_native_fixture(true);
    ASSERT_EQ(fixture.runs.size(), 2U);
    ASSERT_EQ(fixture.state->admitted_segments().size(), 2U);
    EXPECT_EQ(fixture.state->row_count(), 4);
    EXPECT_EQ(fixture.alignment->slot_count(), 5U);
    EXPECT_EQ(fixture.runs[0].selection.relational_common_slots(),
              std::vector<std::size_t>({0, 1}));
    EXPECT_EQ(fixture.runs[1].selection.relational_common_slots(),
              std::vector<std::size_t>({3, 4}));
    EXPECT_TRUE(fixture.runs[0]
                    .participant_runs[1]
                    .boundary_after.cohort_boundary);
    EXPECT_TRUE(fixture.runs[1]
                    .participant_runs[1]
                    .boundary_before.cohort_boundary);
    ASSERT_TRUE(fixture.runs[1]
                    .participant_runs[1]
                    .boundary_before.counter_discontinuity.has_value());

    const auto first =
        citlali::pipeline::NativeMeasuredScanState::rtc_segment_view(
            fixture.state, 0);
    const auto second =
        citlali::pipeline::NativeMeasuredScanState::rtc_segment_view(
            fixture.state, 1);
    EXPECT_EQ(first->row_count(), 2);
    EXPECT_EQ(second->row_count(), 2);
    EXPECT_EQ(first->relational_common_slot(1), 1U);
    EXPECT_EQ(second->relational_common_slot(0), 3U);
    EXPECT_EQ(first->require_cell(1, 1).identity.native_row(), 701);
    EXPECT_EQ(second->require_cell(0, 1).identity.native_row(), 702);
    EXPECT_THROW(
        citlali::pipeline::NativeMeasuredScanState::rtc_segment_view(
            first, 0),
        std::exception);
    EXPECT_THROW(fixture.state->require_cell(4, 0), std::exception);
}

TEST(sci_align_native_pointing_consumers,
     identical_native_times_equal_the_explicit_legacy_rectangular_result) {
    auto fixture = b2b_native_fixture(false, true);
    auto legacy = fixture.data;
    legacy.native_science_mode =
        timestream::NativeScienceMode::legacy_inactive;
    legacy.native_scan.reset();
    legacy.tel_data.data =
        fixture.state->telescope_data_for_detector(0);
    legacy.pointing_offsets_arcsec.data =
        fixture.state->pointing_offsets_for_detector(0);
    std::string pixel_axes = "altaz";
    std::string grouping = "array";

    for (Eigen::Index detector = 0;
         detector < fixture.data.scans.data.cols(); ++detector) {
        auto [native_lat, native_lon] =
            engine_utils::calc_det_pointing_for_science_sample(
                fixture.data, detector, 0.0, 0.0, pixel_axes,
                grouping);
        auto [legacy_lat, legacy_lon] =
            engine_utils::calc_det_pointing_for_science_sample(
                legacy, detector, 0.0, 0.0, pixel_axes, grouping);
        b2b_expect_vector_exact(native_lat, legacy_lat);
        b2b_expect_vector_exact(native_lon, legacy_lon);
    }
}

TEST(sci_align_native_pointing_consumers,
     source_mask_and_kernel_model_use_the_same_native_pointing_pair) {
    auto fixture = b2b_native_fixture();
    auto rtc = b2b_as_rtc(fixture);
    std::string pixel_axes = "altaz";
    std::string grouping = "array";
    constexpr double sigma_arcsec = 4.0;
    constexpr double radius_arcsec = 8.0;

    auto [mask, info] = engine_utils::calc_source_protection_mask(
        rtc, fixture.apt, pixel_axes, grouping,
        "map_center_radius", radius_arcsec);
    timestream::Kernel kernel;
    kernel.map_grouping = grouping;
    kernel.sigma_rad = sigma_arcsec * ASEC_TO_RAD;
    kernel.sigma_limit = radius_arcsec / sigma_arcsec;
    kernel.create_symmetric_gaussian_kernel(
        rtc, pixel_axes, fixture.apt);

    EXPECT_TRUE(info.valid);
    EXPECT_GT(info.protected_samples, 0);
    EXPECT_LT(info.protected_samples, info.total_samples);
    for (Eigen::Index detector = 0;
         detector < rtc.scans.data.cols(); ++detector) {
        auto [lat, lon] =
            engine_utils::calc_det_pointing_for_science_sample(
                rtc, detector, 0.0, 0.0, pixel_axes, grouping);
        for (Eigen::Index row = 0; row < rtc.scans.data.rows(); ++row) {
            const double distance = std::hypot(lat(row), lon(row));
            const bool inside = distance <= radius_arcsec * ASEC_TO_RAD;
            EXPECT_EQ(mask(row, detector), inside);
            EXPECT_EQ(rtc.kernel.data(row, detector) > 0.0, inside);
            if (inside) {
                EXPECT_DOUBLE_EQ(
                    rtc.kernel.data(row, detector),
                    std::exp(-0.5 * std::pow(
                        distance / (sigma_arcsec * ASEC_TO_RAD), 2)));
            }
        }
    }
}

TEST(sci_align_native_pointing_consumers,
     naive_and_jinc_project_only_measured_native_samples) {
    auto fixture = b2b_native_fixture(true);
    Eigen::VectorXi map_indices =
        Eigen::VectorXi::Zero(fixture.data.scans.data.cols());
    std::string pixel_axes = "altaz";

    mapmaking::NaiveMapmaker naive;
    naive.logger = sci_align_production_logger();
    naive.run_polarization = false;
    auto naive_map = b2b_map_buffer(false);
    mapmaking::MapBuffer no_coadd{"cmb"};
    ASSERT_NO_THROW(naive.populate_maps_naive_science_contract(
        fixture.data, naive_map, no_coadd, map_indices, pixel_axes,
        fixture.apt, 100.0, true, false));
    EXPECT_GT(naive_map.weight[0].sum(), 0.0);
    EXPECT_DOUBLE_EQ(
        naive_map.coverage[0].sum(),
        static_cast<double>(fixture.state->row_count() *
                            fixture.state->detector_count()) /
            100.0);

    mapmaking::JincMapmaker jinc;
    jinc.logger = sci_align_production_logger();
    jinc.run_polarization = false;
    jinc.r_max = 3.0;
    jinc.subpixel_n = 1;
    jinc.array_names = {{0, "a1100"}, {1, "a1400"}, {2, "a2000"}};
    jinc.shape_params = {
        {0, Eigen::Vector3d(1.1, 1.67, 2.0)},
        {1, Eigen::Vector3d(1.1, 2.17, 2.0)},
        {2, Eigen::Vector3d(1.1, 3.17, 2.0)},
    };
    auto jinc_map = b2b_map_buffer(true);
    jinc.allocate_jinc_matrix(jinc_map.pixel_size_rad);
    ASSERT_NO_THROW(jinc.populate_maps_jinc(
        fixture.data, jinc_map, no_coadd, map_indices, pixel_axes,
        fixture.apt, 100.0, true, false));
    EXPECT_TRUE(
        (jinc_map.jinc_products.contributor_count[0].array() > 0).any());

    auto synthetic = fixture.data;
    Eigen::MatrixXd synthetic_values = Eigen::MatrixXd::Zero(
        fixture.data.scans.data.rows() + 1,
        fixture.data.scans.data.cols());
    synthetic_values.topRows(fixture.data.scans.data.rows()) =
        fixture.data.scans.data;
    synthetic_values.bottomRows(1).setConstant(12345.0);
    synthetic.scans.data = std::move(synthetic_values);
    synthetic.flags.data = Eigen::Matrix<
        bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(
        synthetic.scans.data.rows(), synthetic.scans.data.cols(), false);

    auto rejected_naive = b2b_map_buffer(false);
    EXPECT_THROW(naive.populate_maps_naive_science_contract(
                     synthetic, rejected_naive, no_coadd, map_indices,
                     pixel_axes, fixture.apt, 100.0, true, false),
                 std::exception);
    EXPECT_DOUBLE_EQ(rejected_naive.signal[0].sum(), 0.0);
    EXPECT_DOUBLE_EQ(rejected_naive.weight[0].sum(), 0.0);

    auto rejected_jinc = b2b_map_buffer(true);
    EXPECT_THROW(jinc.populate_maps_jinc(
                     synthetic, rejected_jinc, no_coadd, map_indices,
                     pixel_axes, fixture.apt, 100.0, true, false),
                 std::exception);
    EXPECT_DOUBLE_EQ(rejected_jinc.signal[0].sum(), 0.0);
    EXPECT_DOUBLE_EQ(rejected_jinc.weight[0].sum(), 0.0);
}

TEST(sci_align_native_pointing_consumers,
     typed_relation_permutation_preserves_complete_scoped_cells) {
    auto first = b2b_native_fixture(
        false, false, 0.0, false,
        "producer:sci-align/b2b-permutation#001");
    auto reversed = b2b_native_fixture(
        false, false, 0.0, true,
        "producer:sci-align/b2b-permutation#001");
    ASSERT_NO_THROW(citlali::pipeline::require_complete_native_cohort(
        first.runs.front().selection));
    EXPECT_EQ(first.relation->bindings(), reversed.relation->bindings());
    ASSERT_EQ(first.state->row_count(), reversed.state->row_count());
    ASSERT_EQ(first.state->detector_count(),
              reversed.state->detector_count());
    for (Eigen::Index row = 0; row < first.state->row_count(); ++row) {
        for (Eigen::Index column = 0;
             column < first.state->detector_count(); ++column) {
            const auto first_cell = first.state->require_cell(row, column);
            const auto reversed_cell =
                reversed.state->require_cell(row, column);
            EXPECT_EQ(first_cell.identity, reversed_cell.identity);
            EXPECT_EQ(first_cell.detector, reversed_cell.detector);
            EXPECT_EQ(
                first.state->require_detector_binding(column).uid,
                reversed.state->require_detector_binding(column).uid);
        }
    }
    EXPECT_EQ(first.state->require_detector_binding(1).uid,
              b2b_apt::uid_v1_max);
}

TEST(sci_align_native_pointing_consumers,
     incomplete_invalid_absent_duplicate_and_unequal_inputs_reject_atomically) {
    auto complete = b2b_native_fixture();
    auto published = complete.state;
    const auto *snapshot = published.get();

    auto absent = b2b_native_fixture(true);
    auto absent_cohort = absent.alignment->select_cohort(
        citlali::pipeline::NativeOperationIdentity{91, 0}, 0,
        absent.alignment->slot_count(), 0);
    EXPECT_THROW(citlali::pipeline::require_complete_native_cohort(
                     absent_cohort),
                 std::exception);
    EXPECT_EQ(published.get(), snapshot);

    std::map<std::pair<citlali::pipeline::TimestreamNetworkId,
                       citlali::pipeline::TimestreamNativeRow>,
             citlali::pipeline::NativeInvalidityProvenance>
        invalidities;
    invalidities.emplace(
        std::make_pair(7, 701),
        citlali::pipeline::NativeInvalidityProvenance{0x20U,
                                                       "delivered flag"});
    auto invalid_cohort = complete.alignment->select_cohort(
        citlali::pipeline::NativeOperationIdentity{92, 0}, 0,
        complete.alignment->slot_count(), 0, invalidities);
    EXPECT_THROW(citlali::pipeline::require_complete_native_cohort(
                     invalid_cohort),
                 std::exception);
    EXPECT_EQ(published.get(), snapshot);

    EXPECT_THROW(
        ([] {
            Eigen::VectorXd times(2);
            times << 1.0, 2.0;
            citlali::pipeline::NativeNetworkAlignment network{
                0, 0, times, {10, 11}};
            auto duplicate =
                citlali::pipeline::make_direct_native_slot_associations(
                    0, 2);
            duplicate[1].native_row = 0;
            (void)citlali::pipeline::NativeAlignmentPlan(
                {network}, times, {duplicate});
        }()),
        std::exception);
    EXPECT_EQ(published.get(), snapshot);

    EXPECT_THROW(
        ([] {
            b2b_apt::RawManifest manifest;
            manifest.observation = {148670, 0, 2};
            manifest.inputs = {
                {0, "toltec0", 1}, {7, "toltec7", 1}};
            (void)citlali::pipeline::
                admit_producer_raw_manifest_relation(
                    "producer:sci-align/b2b-unequal", manifest,
                    std::vector<citlali::pipeline::AptDetectorBinding>{
                        {0, 42, 0, 0, 0}});
        }()),
        std::exception);
    EXPECT_EQ(published.get(), snapshot);
}

TEST(sci_align_native_pointing_consumers,
     stale_cross_scope_nonfinite_and_synthetic_candidates_reject_atomically) {
    auto fixture = b2b_native_fixture();
    auto published = fixture.state;
    const auto *snapshot = published.get();

    auto other_relation = b2b_detector_relation(
        "producer:sci-align/b2b-cross-scope#other");
    const auto foreign_reference =
        fixture.relation->binding_reference_for_column(0);
    EXPECT_THROW(other_relation->require_binding(foreign_reference),
                 std::exception);
    EXPECT_EQ(published.get(), snapshot);

    auto nonfinite_raw = fixture.raw_telescope->telescope_data();
    nonfinite_raw["TelElAct"](1) =
        std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        (void)citlali::pipeline::RawTelescopeTrajectory{nonfinite_raw},
        std::exception);
    Eigen::Vector2d outside_support;
    outside_support << 999.0, 1000.0;
    EXPECT_THROW(citlali::pipeline::evaluate_raw_telescope_trajectory_at(
                     *fixture.raw_telescope, outside_support),
                 std::exception);
    EXPECT_EQ(published.get(), snapshot);

    auto stale_alignment =
        std::make_shared<const citlali::pipeline::NativeAlignmentPlan>(
            *fixture.alignment);
    auto stale_material = b2b_measured_material(
        *stale_alignment, *fixture.relation, fixture.runs);
    EXPECT_THROW(
        published = citlali::pipeline::NativeMeasuredScanState::admit(
            citlali::pipeline::NativeOperationIdentity{0, 0},
            stale_alignment, fixture.pointing, fixture.relation, 0,
            stale_material.values.rows(), b2b_segments(fixture.runs),
            std::move(stale_material.blocks)),
        std::exception);
    EXPECT_EQ(published.get(), snapshot);

    auto nonfinite_material = b2b_measured_material(
        *fixture.alignment, *fixture.relation, fixture.runs);
    const auto &first_run = fixture.runs.front().participant_runs.front();
    Eigen::MatrixXd bad_values = Eigen::MatrixXd::Ones(
        first_run.row_count(), 1);
    bad_values(0, 0) = std::numeric_limits<double>::quiet_NaN();
    auto bad_flags =
        citlali::pipeline::NativeDetectorFlagBitsMatrix::Zero(
            bad_values.rows(), bad_values.cols());
    nonfinite_material.blocks.front() =
        citlali::pipeline::NativeDetectorBlock(
            fixture.alignment->network(first_run.network_id),
            first_run.first_native_row, 0, std::move(bad_values),
            std::move(bad_flags));
    EXPECT_THROW(
        published = citlali::pipeline::NativeMeasuredScanState::admit(
            citlali::pipeline::NativeOperationIdentity{0, 0},
            fixture.alignment, fixture.pointing, fixture.relation, 0,
            nonfinite_material.values.rows(), b2b_segments(fixture.runs),
            std::move(nonfinite_material.blocks)),
        std::exception);
    EXPECT_EQ(published.get(), snapshot);

    std::vector<citlali::pipeline::NativeRtcOutputRowProvenance>
        duplicate_source_rows{
            {0, 0, fixture.state->relational_common_slot(0), {}},
            {1, 0, fixture.state->relational_common_slot(0), {}},
        };
    EXPECT_THROW(
        published =
            citlali::pipeline::NativeMeasuredScanState::
                rtc_output_projection(
                    fixture.state, std::move(duplicate_source_rows), 1),
        std::exception);
    EXPECT_EQ(published.get(), snapshot);

    struct IdentityMapIndices {
        Eigen::Index n_maps = 1;
        Eigen::VectorXi maps_to_arrays =
            Eigen::VectorXi::Zero(1);
        Eigen::VectorXi maps_to_stokes =
            Eigen::VectorXi::Zero(1);
    };
    struct IdentityMapBuffer {
        std::string map_grouping = "detector";
    };
    struct IdentityToltecIo {
        std::map<Eigen::Index, double> array_freq_map{
            {0, 280.0e9}};
    };
    struct IdentityRtc {
        bool run_polarization = false;
        struct {
            std::vector<std::string> stokes_params;
        } polarization;
    };
    struct IdentityAlignment {
        std::shared_ptr<const citlali::pipeline::NativeAlignmentPlan>
            native_consumer_plan;
        std::shared_ptr<const citlali::pipeline::NativePointingPlan>
            native_pointing_plan;
    };
    struct IdentityEngine {
        IdentityMapIndices map_indices;
        engine::Calib calib;
        IdentityMapBuffer omb;
        IdentityToltecIo toltec_io;
        IdentityRtc rtcproc;
        IdentityAlignment alignment;
    } identity_engine;

    identity_engine.calib.apt["uid"] =
        Eigen::VectorXd::Constant(1, 42.0);
    identity_engine.alignment.native_consumer_plan =
        fixture.alignment;
    identity_engine.alignment.native_pointing_plan =
        fixture.pointing;
    ASSERT_TRUE(identity_engine.alignment.native_pointing_plan->bound_to(
        identity_engine.alignment.native_consumer_plan));
    ASSERT_FALSE(identity_engine.calib.has_apt_detector_relation());
    const auto legacy_uid_snapshot = bits_of(
        identity_engine.calib.apt.at("uid")(0));
    const auto alignment_snapshot =
        identity_engine.alignment.native_consumer_plan.get();
    const auto pointing_snapshot =
        identity_engine.alignment.native_pointing_plan.get();
    std::optional<mapmaking::ScienceMapSlotIdentity>
        identity_publication;
    EXPECT_THROW(
        identity_publication.emplace(
            citlali::pipeline::science_map_slot_identity(
                identity_engine, 0)),
        std::exception);
    EXPECT_FALSE(identity_publication.has_value());
    EXPECT_FALSE(identity_engine.calib.has_apt_detector_relation());
    EXPECT_EQ(bits_of(identity_engine.calib.apt.at("uid")(0)),
              legacy_uid_snapshot);
    EXPECT_EQ(identity_engine.alignment.native_consumer_plan.get(),
              alignment_snapshot);
    EXPECT_EQ(identity_engine.alignment.native_pointing_plan.get(),
              pointing_snapshot);

    identity_engine.alignment.native_pointing_plan.reset();
    const auto incomplete_alignment_snapshot =
        identity_engine.alignment.native_consumer_plan.get();
    EXPECT_THROW(
        identity_publication.emplace(
            citlali::pipeline::science_map_slot_identity(
                identity_engine, 0)),
        std::exception);
    EXPECT_FALSE(identity_publication.has_value());
    EXPECT_EQ(identity_engine.alignment.native_consumer_plan.get(),
              incomplete_alignment_snapshot);
    EXPECT_EQ(identity_engine.alignment.native_pointing_plan.get(),
              nullptr);
    EXPECT_EQ(bits_of(identity_engine.calib.apt.at("uid")(0)),
              legacy_uid_snapshot);
}

}  // namespace

// B3a self-contained matched-observation admission evidence is appended below.
// The accepted 33-test B1/B2a/B2b prefix above remains byte-for-byte unchanged.
#include <citlali/core/pipeline/array_properties_table.h>

namespace {

struct B3aNonProductionRawObs;

static_assert(
    citlali::pipeline::is_production_raw_observation_v<::RawObs>);
static_assert(
    !citlali::pipeline::is_production_raw_observation_v<
        B3aNonProductionRawObs>);
static_assert(std::is_same_v<
              decltype(citlali::pipeline::AptDetectorBinding{}.uid),
              std::int64_t>);
static_assert(std::is_same_v<
              decltype(citlali::pipeline::AptDetectorBinding{}.network),
              std::int64_t>);
static_assert(std::is_same_v<
              decltype(citlali::pipeline::AptDetectorBinding{}.kids_tone),
              std::int64_t>);

B2aObservationBytes b3a_issued_observation_bytes(
    std::string raw_content_sha256 = {},
    std::uint64_t raw_byte_count = 0) {
    auto fixture = b2a_observation_bytes(
        std::move(raw_content_sha256), raw_byte_count);
    const auto serialized =
        b2a_observation::serialize_producer_issued_matched_observation_ecsv(
            fixture.output, fixture.target, fixture.relation);
    fixture.bytes = serialized.bytes;
    fixture.receipt = b2a_publication::canonical_receipt_bytes(
        b2a_publication::make_receipt_binding(
            std::string(b2a_publication::receipt_schema_v1),
            std::string(
                b2a_observation::matched_output_byte_transport_scope_v1),
            serialized.digests.envelope_sha256, serialized.bytes));
    return fixture;
}

std::string b3a_receipt_for_bytes(
    std::string_view bytes, std::string_view envelope_sha256) {
    return b2a_publication::canonical_receipt_bytes(
        b2a_publication::make_receipt_binding(
            std::string(b2a_publication::receipt_schema_v1),
            std::string(
                b2a_observation::matched_output_byte_transport_scope_v1),
            std::string(envelope_sha256), bytes));
}

std::string b3a_replace_once(
    std::string bytes, std::string_view before, std::string_view after) {
    const auto position = bytes.find(before);
    if (position == std::string::npos ||
        bytes.find(before, position + before.size()) != std::string::npos) {
        throw std::logic_error(
            "B3a fixture mutation requires one exact wire occurrence");
    }
    bytes.replace(position, before.size(), after);
    return bytes;
}

std::string b3a_nonfinite_output_tone(std::string bytes) {
    constexpr std::string_view row_prefix = "\n0,5,10,";
    const auto row = bytes.find(row_prefix);
    if (row == std::string::npos) {
        throw std::logic_error(
            "B3a fixture cannot find its first canonical output row");
    }
    const auto value = row + row_prefix.size();
    const auto delimiter = bytes.find(',', value);
    if (delimiter == std::string::npos) {
        throw std::logic_error(
            "B3a fixture cannot isolate its output tone value");
    }
    bytes.replace(value, delimiter - value, "nan");
    return bytes;
}

TEST(sci_align_typed_apt_consumer_admission,
     self_contained_parse_replays_exact_published_identity_and_closed_fields) {
    const auto fixture = b3a_issued_observation_bytes();
    const auto issued =
        b2a_observation::parse_issued_matched_observation_ecsv_with_receipt(
            fixture.bytes, fixture.receipt);
    const auto strict =
        b2a_observation::parse_matched_observation_ecsv_with_receipt(
            fixture.bytes, fixture.receipt, fixture.baseline);
    EXPECT_FALSE(issued.parent_content_revalidated);
    EXPECT_TRUE(strict.parent_content_revalidated);

    const auto target_identity =
        b2a_observation::artifact_identity(issued.target);
    const auto relation_identity =
        b2a_observation::producer_issued_artifact_identity(
            issued.relation, issued.target);
    const auto output_identity =
        b2a_observation::producer_issued_artifact_identity(
            issued.output, issued.target, issued.relation);
    EXPECT_EQ(issued.relation.target_parent, target_identity);
    EXPECT_EQ(issued.output.target_parent, target_identity);
    EXPECT_EQ(issued.output.relation_parent, relation_identity);
    EXPECT_EQ(issued.relation.baseline_parent,
              issued.output.baseline_parent);
    EXPECT_EQ(issued.output.baseline_parent,
              b2a_observation::baseline_reference(fixture.baseline));
    EXPECT_EQ(issued.output.envelope.occurrence,
              "occurrence:sci-align/b2a-output#001");

    const auto strict_relation_identity =
        b2a_observation::artifact_identity(
            strict.relation, fixture.baseline, strict.target);
    const auto strict_output_identity =
        b2a_observation::artifact_identity(
            strict.output, fixture.baseline, strict.target,
            strict.relation);
    EXPECT_EQ(strict_relation_identity, relation_identity);
    EXPECT_EQ(strict_output_identity, output_identity);
    EXPECT_EQ(strict.relation.baseline_parent,
              issued.relation.baseline_parent);
    EXPECT_EQ(strict.output.baseline_parent,
              issued.output.baseline_parent);
    EXPECT_EQ(strict.output.target_parent, issued.output.target_parent);
    EXPECT_EQ(strict.output.relation_parent,
              issued.output.relation_parent);

    const auto canonical =
        b2a_observation::serialize_producer_issued_matched_observation_ecsv(
            issued.output, issued.target, issued.relation);
    EXPECT_EQ(canonical.bytes, fixture.bytes);
    EXPECT_EQ(canonical.digests.semantic_sha256,
              issued.declared_digests.semantic_sha256);
    EXPECT_EQ(canonical.digests.envelope_sha256,
              issued.declared_digests.envelope_sha256);
    EXPECT_EQ(canonical.digests.semantic_sha256,
              output_identity.semantic_sha256);
    EXPECT_EQ(canonical.digests.envelope_sha256,
              output_identity.envelope_sha256);
    EXPECT_EQ(canonical.transport.scope, issued.computed_transport.scope);
    EXPECT_EQ(canonical.transport.envelope_sha256,
              issued.computed_transport.envelope_sha256);
    EXPECT_EQ(canonical.transport.sha256,
              issued.computed_transport.sha256);
    EXPECT_EQ(canonical.transport.byte_count,
              issued.computed_transport.byte_count);

    const auto receipt = b2a_publication::parse_canonical_receipt(
        fixture.receipt, b2a_publication::receipt_schema_v1,
        b2a_observation::matched_output_byte_transport_scope_v1);
    EXPECT_EQ(receipt.envelope_sha256, output_identity.envelope_sha256);
    EXPECT_EQ(receipt.byte_sha256, canonical.transport.sha256);
    EXPECT_EQ(receipt.byte_count, fixture.bytes.size());

    const auto expected_fields =
        b2a_observation::canonical_output_field_contracts_v1(
            fixture.baseline, issued.target);
    EXPECT_EQ(issued.output.registered_fields, expected_fields);
    for (const auto &row : issued.output.rows) {
        ASSERT_EQ(row.fields.size(), expected_fields.size());
        ASSERT_EQ(row.transformations.size(), expected_fields.size());
        std::set<std::string> transformed_fields;
        for (const auto &change : row.transformations) {
            const auto contract = std::find_if(
                expected_fields.begin(), expected_fields.end(),
                [&](const auto &candidate) {
                    return candidate.field.name == change.field_name;
                });
            ASSERT_NE(contract, expected_fields.end());
            EXPECT_TRUE(transformed_fields.insert(change.field_name).second);
            EXPECT_EQ(change.operation, contract->authorized_operation);
            EXPECT_EQ(change.after, row.fields.at(change.field_name));
            EXPECT_FALSE(change.authority_reference.empty());
            EXPECT_FALSE(change.provenance_reference.empty());
        }
        EXPECT_EQ(transformed_fields.size(), expected_fields.size());
    }

    ASSERT_EQ(issued.target.inputs.size(), 1U);
    EXPECT_EQ(issued.target.observation,
              (b2a_apt::ObservationIdentity{148671, 0, 2}));
    EXPECT_EQ(issued.target.inputs[0].network, 0);
    EXPECT_EQ(issued.target.inputs[0].interface_name, "toltec0");
    EXPECT_EQ(issued.target.inputs[0].channel_count, 2);
    ASSERT_EQ(issued.relation.seed_source_sequence.size(), 2U);
    EXPECT_EQ(issued.relation.seed_source_sequence.front(),
              b2a_apt::uid_v1_max);
}

TEST(sci_align_typed_apt_consumer_admission,
     production_rawobs_route_and_runtime_calib_use_only_self_contained_artifact) {
    EXPECT_TRUE(
        (citlali::pipeline::is_production_raw_observation_v<::RawObs>));
    EXPECT_FALSE(
        (citlali::pipeline::is_production_raw_observation_v<
            B3aNonProductionRawObs>));

    B2aTemporaryDirectory temporary;
    const auto raw = b2a_write_raw(temporary, 0, 2);
    const auto fixture = b3a_issued_observation_bytes(
        "sha256:" + citlali::utils::sha256_file(raw),
        std::filesystem::file_size(raw));
    const auto artifact = temporary.path / "issued-observation.ecsv";
    b2a_write_bytes(artifact, fixture.bytes);
    b2a_write_bytes(
        std::filesystem::path(artifact.string() + ".sha256"),
        fixture.receipt);
    const std::vector<std::string> raw_files{raw.string()};
    const std::vector<std::string> interfaces{"toltec0"};

    engine::Calib calib;
    ASSERT_NO_THROW(calib.get_canonical_observation_apt(
        artifact.string(), raw_files, interfaces));
    const auto &relation = calib.require_apt_detector_relation();
    const auto &scope = relation.published_scope();
    EXPECT_EQ(scope.kind,
              citlali::pipeline::PublishedAptKind::matched_observation);
    EXPECT_FALSE(scope.parent_content_revalidated);
    EXPECT_FALSE(relation.requires_published_artifact_join());
    EXPECT_EQ(scope.artifact,
              b2a_observation::producer_issued_artifact_identity(
                  fixture.output, fixture.target, fixture.relation));
    EXPECT_EQ(scope.baseline_parent, fixture.output.baseline_parent);
    EXPECT_EQ(scope.transport.sha256,
              "sha256:" + citlali::utils::sha256(fixture.bytes));
    EXPECT_EQ(scope.receipt_sha256,
              "sha256:" + citlali::utils::sha256(fixture.receipt));
    EXPECT_EQ(scope.receipt_byte_count, fixture.receipt.size());

    ASSERT_EQ(relation.bindings().size(), 2U);
    EXPECT_EQ(relation.binding_for_column(0).detector_column, 0U);
    EXPECT_EQ(relation.binding_for_column(0).uid, 0);
    EXPECT_EQ(relation.binding_for_column(0).network, 0);
    EXPECT_EQ(relation.binding_for_column(0).kids_tone, 0);
    EXPECT_EQ(relation.binding_for_column(1).detector_column, 1U);
    EXPECT_EQ(relation.binding_for_column(1).uid, 1);
    EXPECT_EQ(relation.binding_for_column(1).network, 0);
    EXPECT_EQ(relation.binding_for_column(1).kids_tone, 1);
    EXPECT_DOUBLE_EQ(calib.apt.at("uid")(0), 0.0);
    EXPECT_DOUBLE_EQ(calib.apt.at("uid")(1), 1.0);
    EXPECT_DOUBLE_EQ(calib.apt.at("kids_tone")(0), 0.0);
    EXPECT_DOUBLE_EQ(calib.apt.at("kids_tone")(1), 1.0);
}

TEST(sci_align_typed_apt_consumer_admission,
     tamper_stale_partial_swapped_duplicate_foreign_and_nonfinite_fail_atomically) {
    B2aTemporaryDirectory temporary;
    const auto raw = b2a_write_raw(temporary, 0, 2);
    const auto raw_sha256 =
        "sha256:" + citlali::utils::sha256_file(raw);
    const auto raw_byte_count = std::filesystem::file_size(raw);
    const auto fixture = b3a_issued_observation_bytes(
        raw_sha256, raw_byte_count);
    const auto output_identity =
        b2a_observation::producer_issued_artifact_identity(
            fixture.output, fixture.target, fixture.relation);
    const auto artifact = temporary.path / "atomic-observation.ecsv";
    const auto receipt_path =
        std::filesystem::path(artifact.string() + ".sha256");
    const std::vector<std::string> raw_files{raw.string()};
    const std::vector<std::string> interfaces{"toltec0"};
    b2a_write_bytes(artifact, fixture.bytes);
    b2a_write_bytes(receipt_path, fixture.receipt);

    engine::Calib calib;
    ASSERT_NO_THROW(calib.get_canonical_observation_apt(
        artifact.string(), raw_files, interfaces));
    const auto snapshot = b2a_snapshot(calib);
    const auto accepted_relation =
        citlali::pipeline::admit_published_observation_apt_relation(
            fixture.bytes, fixture.receipt, b2a_layout());

    const auto expect_atomic_failure = [&](
        std::string_view bytes, std::string_view receipt,
        const std::vector<std::string> &attempt_raw_files,
        const std::vector<std::string> &attempt_interfaces) {
        b2a_write_bytes(artifact, bytes);
        b2a_write_bytes(receipt_path, receipt);
        EXPECT_THROW(calib.get_canonical_observation_apt(
                         artifact.string(), attempt_raw_files,
                         attempt_interfaces),
                     std::exception);
        b2a_expect_snapshot_eq(snapshot, calib);
    };

    auto tampered_receipt = fixture.receipt;
    const auto receipt_digest =
        tampered_receipt.find("byte_sha256=sha256:");
    ASSERT_NE(receipt_digest, std::string::npos);
    const auto receipt_digit = receipt_digest +
        std::string("byte_sha256=sha256:").size();
    tampered_receipt[receipt_digit] =
        tampered_receipt[receipt_digit] == '0' ? '1' : '0';
    expect_atomic_failure(
        fixture.bytes, tampered_receipt, raw_files, interfaces);

    const auto stale_bytes = b3a_replace_once(
        fixture.bytes, "event:sci-align/b2a-target#001",
        "event:sci-align/b2a-target#009");
    expect_atomic_failure(
        stale_bytes,
        b3a_receipt_for_bytes(
            stale_bytes, output_identity.envelope_sha256),
        raw_files, interfaces);

    const auto partial_bytes =
        fixture.bytes.substr(0, fixture.bytes.size() - 1U);
    expect_atomic_failure(
        partial_bytes,
        b3a_receipt_for_bytes(
            partial_bytes, output_identity.envelope_sha256),
        raw_files, interfaces);

    const auto foreign = b3a_issued_observation_bytes(
        b2a_sha256_reference('9'), raw_byte_count + 1U);
    expect_atomic_failure(
        fixture.bytes, foreign.receipt, raw_files, interfaces);

    EXPECT_THROW(
        citlali::pipeline::admit_published_observation_apt_relation(
            fixture.bytes, fixture.receipt,
            {{0, 0, 0}, {1, 0, 0}}),
        std::exception);
    EXPECT_EQ(accepted_relation.bindings().size(), 2U);
    EXPECT_EQ(accepted_relation.binding_for_column(0).uid, 0);
    const std::vector<std::string> duplicate_raw_files{
        raw.string(), raw.string()};
    const std::vector<std::string> duplicate_interfaces{
        "toltec0", "toltec0"};
    expect_atomic_failure(
        fixture.bytes, fixture.receipt, duplicate_raw_files,
        duplicate_interfaces);

    expect_atomic_failure(
        foreign.bytes, foreign.receipt, raw_files, interfaces);
    const auto foreign_relation =
        citlali::pipeline::admit_published_observation_apt_relation(
            foreign.bytes, foreign.receipt, b2a_layout());
    EXPECT_FALSE(accepted_relation.same_scope(foreign_relation));
    EXPECT_THROW(
        foreign_relation.require_binding(
            accepted_relation.binding_reference_for_column(0)),
        std::exception);

    const auto nonfinite_bytes =
        b3a_nonfinite_output_tone(fixture.bytes);
    expect_atomic_failure(
        nonfinite_bytes,
        b3a_receipt_for_bytes(
            nonfinite_bytes, output_identity.envelope_sha256),
        raw_files, interfaces);
}

}  // namespace

// B3b observation-owned native-cohort product lineage evidence is appended
// below.  The accepted 36-test B1/B2a/B2b/B3a prefix above remains
// byte-for-byte unchanged.
#include <citlali/core/pipeline/native_cohort_product_provenance.h>
#include <citlali/core/pipeline/observation_buffers.h>
#include <citlali/core/pipeline/raw_timestream_provenance_lifecycle.h>

namespace {

struct B3bLineageFixture {
    B2aObservationBytes apt;
    std::shared_ptr<const citlali::pipeline::AptDetectorRelation> relation;
    std::shared_ptr<const citlali::pipeline::NativeAlignmentPlan> alignment;
    std::shared_ptr<const citlali::pipeline::RawTelescopeTrajectory>
        raw_telescope;
    std::shared_ptr<const citlali::pipeline::NativePointingPlan> pointing;
    citlali::pipeline::NativeDetectorBlock detector_block;
    std::vector<citlali::pipeline::NativeContiguousRun> runs;
    citlali::pipeline::NativeCohortObservationBinding binding;
};

struct B3bLineageCalibAuthority {
    std::shared_ptr<const citlali::pipeline::AptDetectorRelation> relation;

    std::shared_ptr<const citlali::pipeline::AptDetectorRelation>
    apt_detector_relation_handle() const noexcept {
        return relation;
    }
};

struct B3bLineageEngine {
    B3bLineageCalibAuthority calib;
    struct {
        std::shared_ptr<const citlali::pipeline::NativeAlignmentPlan>
            native_consumer_plan;
        std::shared_ptr<const citlali::pipeline::NativePointingPlan>
            native_pointing_plan;
    } alignment;
    struct {
        Eigen::MatrixXi scan_indices;
    } telescope;
    citlali::pipeline::RawTimestreamExecutionPlan raw_timestream_plan;
};

static_assert(
    citlali::pipeline::has_native_cohort_observation_sources_v<
        B3bLineageEngine>);

B3bLineageEngine b3b_lineage_engine(
    const B3bLineageFixture &fixture, Eigen::Index scan_count) {
    B3bLineageEngine result;
    result.calib.relation = fixture.relation;
    result.alignment.native_consumer_plan = fixture.alignment;
    result.alignment.native_pointing_plan = fixture.pointing;
    result.telescope.scan_indices =
        Eigen::MatrixXi::Zero(2, scan_count);
    result.raw_timestream_plan.initialized = true;
    result.raw_timestream_plan.observation.emplace();
    return result;
}

B3bLineageFixture b3b_lineage_fixture(
    double common_time_offset_sec = 0.0,
    std::string raw_content_sha256 = b2a_sha256_reference('1'),
    std::uint64_t raw_byte_count = 4096) {
    auto apt = b3a_issued_observation_bytes(
        std::move(raw_content_sha256), raw_byte_count);
    auto relation =
        std::make_shared<const citlali::pipeline::AptDetectorRelation>(
            citlali::pipeline::admit_published_observation_apt_relation(
                apt.bytes, apt.receipt, b2a_layout()));

    Eigen::VectorXd native_times(6);
    native_times << 1000.0, 1000.01, 1000.02,
        1000.04, 1000.05, 1000.06;
    const std::vector<citlali::pipeline::TimestreamPacketCounter> counters{
        100, 101, 102, 104, 105, 106};
    citlali::pipeline::NativeNetworkAlignment network{
        0, 100, native_times, counters};
    Eigen::VectorXd common_times =
        native_times.array() + common_time_offset_sec;
    auto alignment =
        std::make_shared<const citlali::pipeline::NativeAlignmentPlan>(
            std::vector<citlali::pipeline::NativeNetworkAlignment>{network},
            std::move(common_times),
            std::vector<std::vector<
                citlali::pipeline::NativeSlotAssociation>>{
                citlali::pipeline::make_direct_native_slot_associations(
                    100, 6)});

    Eigen::VectorXd telescope_times =
        Eigen::VectorXd::LinSpaced(10, 999.99, 1000.08);
    auto raw_telescope =
        std::make_shared<const citlali::pipeline::RawTelescopeTrajectory>(
            b2b_telescope_series(telescope_times));
    const auto &aligned_network = alignment->network(0);
    const auto &targets =
        aligned_network.reconstructed_times_unix_sec();
    std::vector<citlali::pipeline::NativeNetworkPointing> networks;
    networks.emplace_back(
        0, aligned_network.first_native_row(), targets,
        citlali::pipeline::evaluate_raw_telescope_trajectory_at(
            *raw_telescope, targets),
        b2b_pointing_offsets(targets));
    auto pointing =
        std::make_shared<const citlali::pipeline::NativePointingPlan>(
            alignment, raw_telescope, std::move(networks));

    Eigen::MatrixXd measured(6, 2);
    measured << 1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0,
        9.0, 10.0,
        11.0, 12.0;
    citlali::pipeline::NativeDetectorFlagBitsMatrix flags(6, 2);
    flags << 0x1, 0x2,
        0x4, 0x8,
        0x10, 0x20,
        0x40, 0x80,
        0x100, 0x200,
        0x400, 0x800;
    citlali::pipeline::NativeDetectorBlock detector_block{
        aligned_network, 100, 0, std::move(measured), std::move(flags)};
    auto runs = citlali::pipeline::partition_native_contiguous_runs(
        aligned_network, 100, 106);
    auto binding =
        citlali::pipeline::make_native_cohort_observation_binding(
            17, *relation, alignment, pointing);
    return {std::move(apt), std::move(relation), std::move(alignment),
            std::move(raw_telescope), std::move(pointing),
            std::move(detector_block), std::move(runs),
            std::move(binding)};
}

citlali::pipeline::NativeCohortScanProvenance b3b_scan_provenance(
    const B3bLineageFixture &fixture, std::int64_t scan_index,
    std::string method = "naive",
    citlali::pipeline::TimestreamNativeRevision input_revision = 12,
    citlali::pipeline::TimestreamNativeRevision output_revision = 13) {
    citlali::pipeline::NativeCohortScanProvenance record;
    record.observation_binding_digest = fixture.binding.digest();
    record.operation = citlali::pipeline::NativeOperationIdentity{
        static_cast<std::uint64_t>(900 + scan_index), scan_index};
    record.input_revision = input_revision;
    record.output_revision = output_revision;
    record.native_cell_action =
        "replaced-or-preserved-by-ptc-v1";

    for (std::size_t run_ordinal = 0;
         run_ordinal < fixture.runs.size(); ++run_ordinal) {
        const auto &run = fixture.runs.at(run_ordinal);
        const Eigen::Index first = static_cast<Eigen::Index>(
            run.first_native_row -
            fixture.detector_block.first_native_row());
        const Eigen::Index count = static_cast<Eigen::Index>(
            run.past_last_native_row - run.first_native_row);
        citlali::pipeline::NativeDetectorFlagBitsMatrix run_flags =
            fixture.detector_block.original_flag_bits().block(
                first, 0, count,
                fixture.detector_block.original_flag_bits().cols());
        auto supports = citlali::pipeline::make_native_stride_support(
            fixture.detector_block, run, run_flags, 2, run_ordinal);
        for (auto &support : supports) {
            citlali::pipeline::NativeCohortOutputRow row;
            row.output_row = static_cast<Eigen::Index>(record.rows.size());
            row.relational_common_slot = static_cast<std::size_t>(
                support.selected_anchor.native_row() - 100);
            row.participants.push_back(
                {support.selected_anchor, input_revision, output_revision,
                 citlali::pipeline::CoincidenceCellState::mapped_valid});
            row.participant_support.push_back(std::move(support));
            record.rows.push_back(std::move(row));
        }
    }

    if (!method.empty()) {
        record.map_join.mapmaking_enabled = true;
        record.map_join.method = method;
        record.map_join.eligible_input_digest =
            b2a_sha256_reference('3');
        record.map_join.ordered_map_indices = {5, 8, 13};
        record.map_join.product_identity_digest =
            b2a_sha256_reference('4');
        if (method == "jinc") {
            record.map_join.jinc_processing_configuration_digest =
                b2a_sha256_reference('5');
            record.map_join.jinc_scan_trace_digest =
                b2a_sha256_reference('6');
        }
    }
    return record;
}

TEST(sci_align_native_cohort_product_lineage,
     detector_relation_and_raw_manifest_digests_are_deterministic_and_scoped) {
    const auto fixture = b3a_issued_observation_bytes(
        b2a_sha256_reference('1'), 4096);
    const auto relation =
        citlali::pipeline::admit_published_observation_apt_relation(
            fixture.bytes, fixture.receipt, b2a_layout());
    auto reversed_layout = b2a_layout();
    std::reverse(reversed_layout.begin(), reversed_layout.end());
    const auto reversed =
        citlali::pipeline::admit_published_observation_apt_relation(
            fixture.bytes, fixture.receipt, reversed_layout);

    const auto relation_digest =
        citlali::pipeline::native_cohort_detector_relation_digest(
            relation);
    const auto raw_digest =
        citlali::pipeline::native_cohort_raw_manifest_digest(relation);
    EXPECT_EQ(
        relation_digest,
        citlali::pipeline::native_cohort_detector_relation_digest(
            reversed));
    EXPECT_EQ(
        raw_digest,
        citlali::pipeline::native_cohort_raw_manifest_digest(reversed));
    EXPECT_EQ(relation.binding_for_column(0).uid,
              reversed.binding_for_column(0).uid);
    EXPECT_EQ(relation.binding_for_column(1).uid,
              reversed.binding_for_column(1).uid);

    const auto foreign_fixture = b3a_issued_observation_bytes(
        b2a_sha256_reference('9'), 4097);
    const auto foreign =
        citlali::pipeline::admit_published_observation_apt_relation(
            foreign_fixture.bytes, foreign_fixture.receipt,
            b2a_layout());
    EXPECT_NE(
        relation_digest,
        citlali::pipeline::native_cohort_detector_relation_digest(
            foreign));
    EXPECT_NE(raw_digest,
              citlali::pipeline::native_cohort_raw_manifest_digest(
                  foreign));

    const auto strict =
        citlali::pipeline::admit_published_observation_apt_relation(
            fixture.bytes, fixture.receipt, fixture.baseline,
            b2a_layout());
    EXPECT_THROW(
        citlali::pipeline::native_cohort_detector_relation_digest(
            strict),
        std::exception);
    const auto producer = b2b_detector_relation(
        "producer:sci-align/b3b-disallowed#001");
    EXPECT_THROW(
        citlali::pipeline::native_cohort_detector_relation_digest(
            *producer),
        std::exception);
}

TEST(sci_align_native_cohort_product_lineage,
     complete_prepare_snapshot_binds_native_grouping_revision_and_product_state) {
    const auto fixture = b3b_lineage_fixture();
    const auto record = b3b_scan_provenance(fixture, 0);
    ASSERT_NO_THROW(record.validate(fixture.binding, 1));

    auto engine = b3b_lineage_engine(fixture, 1);
    ASSERT_NO_THROW(
        citlali::pipeline::begin_native_cohort_observation_if_available(
            engine, 17));
    ASSERT_TRUE(engine.raw_timestream_plan.observation
                    ->native_cohort_lineage);
    EXPECT_TRUE(
        citlali::pipeline::native_cohort_observation_bindings_equal(
            fixture.binding,
            engine.raw_timestream_plan.observation
                ->native_cohort_lineage->binding()));
    auto prepared =
        citlali::pipeline::
            prepare_native_cohort_scan_provenance_if_available(
                engine, record);
    ASSERT_TRUE(prepared.has_value());
    EXPECT_FALSE(
        engine.raw_timestream_plan.realized.native_cohort_provenance);
    EXPECT_THROW(
        engine.raw_timestream_plan.observation
            ->native_cohort_lineage->snapshot_complete(),
        std::exception);

    EXPECT_EQ(fixture.binding.observation_index, 17U);
    EXPECT_EQ(fixture.binding.raw_observation,
              (b2a_apt::ObservationIdentity{148671, 0, 2}));
    EXPECT_EQ(fixture.binding.detector_relation_digest,
              citlali::pipeline::native_cohort_detector_relation_digest(
                  *fixture.relation));
    EXPECT_EQ(fixture.binding.raw_manifest_digest,
              citlali::pipeline::native_cohort_raw_manifest_digest(
                  *fixture.relation));
    EXPECT_EQ(fixture.binding.alignment_plan_digest,
              citlali::pipeline::native_cohort_alignment_plan_digest(
                  *fixture.alignment));
    EXPECT_EQ(fixture.binding.pointing_plan_digest,
              citlali::pipeline::native_cohort_pointing_plan_digest(
                  *fixture.pointing));
    EXPECT_FALSE(
        fixture.binding.artifact_scope.parent_content_revalidated);
    EXPECT_EQ(record.observation_binding_digest,
              fixture.binding.digest());
    EXPECT_EQ(record.input_revision, 12U);
    EXPECT_EQ(record.output_revision, 13U);
    EXPECT_EQ(record.native_cell_action,
              "replaced-or-preserved-by-ptc-v1");
    ASSERT_EQ(record.rows.size(), 4U);
    for (std::size_t output = 0; output < record.rows.size(); ++output) {
        const auto &row = record.rows[output];
        EXPECT_EQ(row.output_row,
                  static_cast<Eigen::Index>(output));
        ASSERT_EQ(row.participants.size(), 1U);
        ASSERT_EQ(row.participant_support.size(), 1U);
        EXPECT_EQ(row.participants[0].cell_state,
                  citlali::pipeline::CoincidenceCellState::mapped_valid);
        EXPECT_EQ(row.participants[0].input_revision, 12U);
        EXPECT_EQ(row.participants[0].output_revision, 13U);
        EXPECT_TRUE(row.participants[0].identity ==
                    row.participant_support[0].selected_anchor);
    }
    EXPECT_STREQ(
        citlali::pipeline::native_cohort_common_slot_semantics,
        "relational-coincidence-grouping-only");
    EXPECT_TRUE(record.map_join.mapmaking_enabled);
    EXPECT_EQ(record.map_join.method, "naive");
    EXPECT_EQ(record.map_join.eligible_input_digest,
              b2a_sha256_reference('3'));
    EXPECT_EQ(record.map_join.ordered_map_indices,
              (std::vector<Eigen::Index>{5, 8, 13}));
    EXPECT_EQ(record.map_join.product_identity_digest,
              b2a_sha256_reference('4'));
    prepared.reset();
    EXPECT_THROW(
        engine.raw_timestream_plan.observation
            ->native_cohort_lineage->snapshot_complete(),
        std::exception);
}

TEST(sci_align_native_cohort_product_lineage,
     beammap_producer_mode_does_not_require_input_typed_apt_authority) {
    const auto fixture = b3b_lineage_fixture();

    auto consumer = b3b_lineage_engine(fixture, 1);
    consumer.calib.relation.reset();
    EXPECT_THROW(
        citlali::pipeline::begin_native_cohort_observation_if_available(
            consumer, 17),
        std::exception);

    auto beammap_producer = b3b_lineage_engine(fixture, 1);
    beammap_producer.calib.relation.reset();
    ASSERT_NO_THROW(
        citlali::pipeline::begin_native_cohort_observation_if_available<false>(
            beammap_producer, 17));
    EXPECT_FALSE(beammap_producer.raw_timestream_plan.observation
                     ->native_cohort_lineage);
}

TEST(sci_align_native_cohort_product_lineage,
     rtc_runs_reset_stride_and_preserve_exact_support_and_ored_flags) {
    const auto fixture = b3b_lineage_fixture();
    ASSERT_EQ(fixture.runs.size(), 2U);
    EXPECT_EQ(fixture.runs[0].first_native_row, 100);
    EXPECT_EQ(fixture.runs[0].past_last_native_row, 103);
    EXPECT_EQ(fixture.runs[1].first_native_row, 103);
    EXPECT_EQ(fixture.runs[1].past_last_native_row, 106);
    ASSERT_TRUE(
        fixture.runs[0].boundary_after.counter_discontinuity);
    ASSERT_TRUE(
        fixture.runs[1].boundary_before.counter_discontinuity);
    const citlali::pipeline::NativeCounterDiscontinuity expected_gap{
        102, 103, 102, 104};
    EXPECT_EQ(
        *fixture.runs[0].boundary_after.counter_discontinuity,
        expected_gap);
    EXPECT_EQ(
        *fixture.runs[1].boundary_before.counter_discontinuity,
        expected_gap);
    EXPECT_TRUE(fixture.runs[0].boundary_before.scan_boundary);
    EXPECT_TRUE(fixture.runs[0].boundary_before.stream_boundary);
    EXPECT_TRUE(fixture.runs[1].boundary_after.scan_boundary);
    EXPECT_TRUE(fixture.runs[1].boundary_after.stream_boundary);

    const auto record = b3b_scan_provenance(fixture, 0, "");
    ASSERT_NO_THROW(record.validate(fixture.binding, 1));
    ASSERT_EQ(record.rows.size(), 4U);
    const std::vector<citlali::pipeline::TimestreamNativeRow>
        expected_anchors{100, 102, 103, 105};
    const std::vector<std::size_t> expected_slots{0, 2, 3, 5};
    const std::vector<std::vector<
        citlali::pipeline::TimestreamNativeRow>> expected_support{
        {100, 101}, {102}, {103, 104}, {105}};
    const std::vector<std::vector<
        citlali::pipeline::NativeDetectorFlagBits>> expected_flags{
        {0x5, 0xa}, {0x10, 0x20},
        {0x140, 0x280}, {0x400, 0x800}};
    for (std::size_t output = 0; output < record.rows.size(); ++output) {
        const auto &row = record.rows[output];
        const auto &support = row.participant_support[0];
        EXPECT_EQ(row.relational_common_slot, expected_slots[output]);
        EXPECT_EQ(support.selected_anchor.native_row(),
                  expected_anchors[output]);
        EXPECT_EQ(support.factor, 2);
        EXPECT_EQ(support.run_output_row,
                  static_cast<citlali::pipeline::TimestreamNativeRow>(
                      output % 2));
        EXPECT_EQ(support.final_short_support, output % 2 == 1);
        ASSERT_EQ(support.exact_support_rows.size(),
                  expected_support[output].size());
        for (std::size_t index = 0;
             index < support.exact_support_rows.size(); ++index) {
            EXPECT_EQ(support.exact_support_rows[index].native_row(),
                      expected_support[output][index]);
        }
        EXPECT_EQ(support.detector_columns,
                  (std::vector<Eigen::Index>{0, 1}));
        EXPECT_EQ(support.ored_flag_support,
                  expected_flags[output]);
    }
    EXPECT_EQ(record.rows[0].participant_support[0].run_ordinal, 0U);
    EXPECT_EQ(record.rows[2].participant_support[0].run_ordinal, 1U);
    EXPECT_EQ(record.rows[0].participant_support[0]
                  .first_support_native_row,
              100);
    EXPECT_EQ(record.rows[2].participant_support[0]
                  .first_support_native_row,
              103);
}

TEST(sci_align_native_cohort_product_lineage,
     native_telescope_and_scan_level_map_jinc_joins_ignore_common_time_authority) {
    const auto first = b3b_lineage_fixture();
    const auto shifted_common = b3b_lineage_fixture(0.003);
    EXPECT_NE(
        first.alignment->common_slot_reference_times_unix_sec()(0),
        shifted_common.alignment
            ->common_slot_reference_times_unix_sec()(0));
    EXPECT_EQ(
        citlali::pipeline::native_cohort_alignment_plan_digest(
            *first.alignment),
        citlali::pipeline::native_cohort_alignment_plan_digest(
            *shifted_common.alignment));
    EXPECT_EQ(
        citlali::pipeline::native_cohort_pointing_plan_digest(
            *first.pointing),
        citlali::pipeline::native_cohort_pointing_plan_digest(
            *shifted_common.pointing));
    const auto &network = first.pointing->network(0);
    const auto &alignment_network = first.alignment->network(0);
    for (citlali::pipeline::TimestreamNativeRow native_row = 100;
         native_row < 106; ++native_row) {
        const auto identity = alignment_network.identity(native_row);
        EXPECT_TRUE(identity == network.identity(native_row));
        EXPECT_DOUBLE_EQ(
            network.telescope_series("TelTime")(
                network.local_row(native_row)),
            identity.reconstructed_time_unix_sec());
    }

    const auto naive = b3b_scan_provenance(first, 0, "naive");
    const auto jinc = b3b_scan_provenance(first, 0, "jinc");
    ASSERT_NO_THROW(naive.validate(first.binding, 1));
    ASSERT_NO_THROW(jinc.validate(first.binding, 1));
    EXPECT_EQ(naive.map_join.eligible_input_digest,
              jinc.map_join.eligible_input_digest);
    EXPECT_EQ(naive.map_join.ordered_map_indices,
              jinc.map_join.ordered_map_indices);
    EXPECT_EQ(jinc.map_join.method, "jinc");
    EXPECT_EQ(
        jinc.map_join.jinc_processing_configuration_digest,
        std::optional<std::string>{b2a_sha256_reference('5')});
    EXPECT_EQ(jinc.map_join.jinc_scan_trace_digest,
              std::optional<std::string>{
                  b2a_sha256_reference('6')});

    auto mislabeled = naive;
    mislabeled.map_join.jinc_processing_configuration_digest =
        b2a_sha256_reference('5');
    mislabeled.map_join.jinc_scan_trace_digest =
        b2a_sha256_reference('6');
    EXPECT_THROW(mislabeled.validate(first.binding, 1),
                 std::exception);
    auto incomplete_jinc = jinc;
    incomplete_jinc.map_join.jinc_scan_trace_digest.reset();
    EXPECT_THROW(incomplete_jinc.validate(first.binding, 1),
                 std::exception);
}

TEST(sci_align_native_cohort_product_lineage,
     stale_missing_foreign_and_partial_candidates_fail_before_slot_mutation) {
    const auto fixture = b3b_lineage_fixture();
    const auto valid = b3b_scan_provenance(fixture, 0);
    auto lineage =
        citlali::pipeline::NativeCohortObservationLineage::create(
            fixture.binding, 1);
    EXPECT_THROW(lineage->snapshot_complete(), std::exception);

    auto stale = valid;
    stale.observation_binding_digest = b2a_sha256_reference('0');
    EXPECT_THROW(lineage->reserve(stale), std::exception);

    const auto foreign_fixture = b3b_lineage_fixture(
        0.0, b2a_sha256_reference('9'), 4097);
    const auto foreign = b3b_scan_provenance(foreign_fixture, 0);
    EXPECT_NE(fixture.binding.digest(),
              foreign_fixture.binding.digest());
    EXPECT_THROW(lineage->reserve(foreign), std::exception);

    auto cross_observation = fixture.binding;
    ++cross_observation.raw_observation.observation;
    EXPECT_THROW(cross_observation.validate(), std::exception);
    EXPECT_THROW(
        citlali::pipeline::NativeCohortObservationLineage::create(
            cross_observation, 1),
        std::exception);

    auto missing = valid;
    missing.rows.clear();
    EXPECT_THROW(lineage->reserve(missing), std::exception);
    auto partial = valid;
    partial.rows[0].participant_support.clear();
    EXPECT_THROW(lineage->reserve(partial), std::exception);
    auto swapped = valid;
    std::swap(swapped.rows[0], swapped.rows[1]);
    EXPECT_THROW(lineage->reserve(swapped), std::exception);
    auto duplicated = valid;
    duplicated.rows[1].relational_common_slot =
        duplicated.rows[0].relational_common_slot;
    EXPECT_THROW(lineage->reserve(duplicated), std::exception);
    auto partial_jinc = b3b_scan_provenance(fixture, 0, "jinc");
    partial_jinc.map_join.jinc_scan_trace_digest.reset();
    EXPECT_THROW(lineage->reserve(partial_jinc), std::exception);

    EXPECT_THROW(lineage->snapshot_complete(), std::exception);
    auto reservation = lineage->reserve(valid);
    reservation.commit();
    const auto snapshot = lineage->snapshot_complete();
    ASSERT_EQ(snapshot.scans.size(), 1U);
    EXPECT_EQ(snapshot.scans[0].observation_binding_digest,
              fixture.binding.digest());
}

TEST(sci_align_native_cohort_product_lineage,
     reservation_rollback_retry_and_complete_snapshot_are_atomic) {
    using Lineage =
        citlali::pipeline::NativeCohortObservationLineage;
    static_assert(noexcept(
        std::declval<Lineage::Reservation &>().commit()));
    static_assert(noexcept(
        citlali::pipeline::commit_native_cohort_scan_provenance(
            std::declval<std::optional<Lineage::Reservation> &>())));

    const auto fixture = b3b_lineage_fixture();
    const auto scan0 = b3b_scan_provenance(fixture, 0, "naive");
    const auto scan1 = b3b_scan_provenance(fixture, 1, "jinc", 13, 14);
    auto engine = b3b_lineage_engine(fixture, 2);
    citlali::pipeline::begin_native_cohort_observation_if_available(
        engine, 17);
    const auto &lineage =
        engine.raw_timestream_plan.observation->native_cohort_lineage;
    ASSERT_TRUE(lineage);
    EXPECT_THROW(lineage->snapshot_complete(), std::exception);

    {
        auto abandoned =
            citlali::pipeline::
                prepare_native_cohort_scan_provenance_if_available(
                    engine, scan0);
        ASSERT_TRUE(abandoned.has_value());
        EXPECT_THROW(
            citlali::pipeline::
                prepare_native_cohort_scan_provenance_if_available(
                    engine, scan0),
            std::exception);
        EXPECT_THROW(lineage->snapshot_complete(), std::exception);
    }
    EXPECT_THROW(lineage->snapshot_complete(), std::exception);

    auto retry =
        citlali::pipeline::
            prepare_native_cohort_scan_provenance_if_available(
                engine, scan0);
    ASSERT_TRUE(retry.has_value());
    citlali::pipeline::commit_native_cohort_scan_provenance(retry);
    EXPECT_FALSE(retry.has_value());
    citlali::pipeline::commit_native_cohort_scan_provenance(retry);
    EXPECT_THROW(
        citlali::pipeline::
            prepare_native_cohort_scan_provenance_if_available(
                engine, scan0),
        std::exception);
    EXPECT_THROW(lineage->snapshot_complete(), std::exception);

    auto pending =
        citlali::pipeline::
            prepare_native_cohort_scan_provenance_if_available(
                engine, 17, scan1);
    ASSERT_TRUE(pending.has_value());
    EXPECT_THROW(
        citlali::pipeline::
            prepare_native_cohort_scan_provenance_if_available(
                engine, 17, scan1),
        std::exception);
    EXPECT_THROW(lineage->snapshot_complete(), std::exception);
    citlali::pipeline::commit_native_cohort_scan_provenance(pending);
    EXPECT_FALSE(pending.has_value());

    EXPECT_THROW(
        citlali::pipeline::complete_raw_timestream_observation(
            engine.raw_timestream_plan, 1, 3),
        std::exception);
    EXPECT_FALSE(engine.raw_timestream_plan.realized.execution_completed);
    EXPECT_FALSE(
        engine.raw_timestream_plan.realized.native_cohort_provenance);
    ASSERT_NO_THROW(
        citlali::pipeline::complete_raw_timestream_observation(
            engine.raw_timestream_plan, 2, 3));
    ASSERT_TRUE(
        engine.raw_timestream_plan.realized.native_cohort_provenance);
    const auto snapshot =
        *engine.raw_timestream_plan.realized.native_cohort_provenance;
    ASSERT_EQ(snapshot.scans.size(), 2U);
    ASSERT_NO_THROW(snapshot.validate_complete(2));
    EXPECT_EQ(snapshot.scans[0].operation.scan_index, 0);
    EXPECT_EQ(snapshot.scans[1].operation.scan_index, 1);
    EXPECT_EQ(snapshot.scans[0].output_revision, 13U);
    EXPECT_EQ(snapshot.scans[1].input_revision, 13U);
    EXPECT_EQ(snapshot.scans[1].output_revision, 14U);

    auto mutable_copy = snapshot;
    mutable_copy.scans[0].native_cell_action = "tampered-copy";
    const auto replay = lineage->snapshot_complete();
    EXPECT_EQ(replay.scans[0].native_cell_action,
              "replaced-or-preserved-by-ptc-v1");
    EXPECT_THROW(
        citlali::pipeline::
            prepare_native_cohort_scan_provenance_if_available(
                engine, scan1),
        std::exception);

    const auto persisted =
        citlali::pipeline::raw_timestream_provenance_node(
            engine.raw_timestream_plan);
    EXPECT_TRUE(persisted["realized"]
                    ["native_cohort_product_provenance_available"]
                        .as<bool>());
    const auto product = persisted["realized"]
        ["native_cohort_product_provenance"];
    EXPECT_EQ(product["observation_index"].as<std::size_t>(), 17U);
    EXPECT_EQ(product["detector_relation_digest"].as<std::string>(),
              fixture.binding.detector_relation_digest);
    EXPECT_EQ(product["raw_manifest_digest"].as<std::string>(),
              fixture.binding.raw_manifest_digest);
    EXPECT_EQ(product["alignment_plan_digest"].as<std::string>(),
              fixture.binding.alignment_plan_digest);
    EXPECT_EQ(product["pointing_plan_digest"].as<std::string>(),
              fixture.binding.pointing_plan_digest);
    ASSERT_EQ(product["scans"].size(), 2U);
    EXPECT_EQ(product["scans"][0]["map_join"]["method"]
                  .as<std::string>(),
              "naive");
    EXPECT_EQ(product["scans"][1]["map_join"]["method"]
                  .as<std::string>(),
              "jinc");
}

}  // namespace
