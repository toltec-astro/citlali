#include <citlali/core/pipeline/sci_align_scan_contract.h>
#include <citlali/core/pipeline/simulated_observation_indices.h>
#include <citlali/core/pipeline/raw_timestream_execution_plan.h>
#include <citlali/core/pipeline/timestream_output_provenance.h>
#include <citlali/core/pipeline/output_netcdf_metadata.h>
#include <citlali/core/engine/detail/beammap_detector_tod_scan_selection.h>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace {

using citlali::pipeline::sci_align::HalfOpenInterval;
using citlali::pipeline::sci_align::ScanStatus;

struct SimulatedDataItem {
    std::string interface_id;

    const std::string &interface() const { return interface_id; }
};

struct SimulatedRawObs {
    std::vector<SimulatedDataItem> items;
    std::optional<int> hwpr_item;

    std::vector<std::reference_wrapper<const SimulatedDataItem>>
    kidsdata() const {
        std::vector<std::reference_wrapper<const SimulatedDataItem>> result;
        result.reserve(items.size());
        for (const auto &item : items) {
            result.push_back(std::cref(item));
        }
        return result;
    }

    const std::optional<int> &hwpdata() const { return hwpr_item; }
};

struct SimulatedAlignmentEngine {
    struct {
        citlali::config::InterfaceSyncOffsetConfig interface_sync;
    } typed_config;
    citlali::pipeline::InterfaceSyncState interface_sync;
    citlali::pipeline::RawTimestreamExecutionPlan raw_timestream_plan;
    citlali::pipeline::TimestreamAlignmentState alignment;
    struct {
        bool run_hwpr = false;
    } calib;
    struct {
        double fsmp = 100.0;
        std::map<std::string, Eigen::VectorXd> tel_data;
    } telescope;
};

Eigen::VectorXd coordinate_axis(std::initializer_list<double> values) {
    Eigen::VectorXd result(static_cast<Eigen::Index>(values.size()));
    Eigen::Index row = 0;
    for (const double value : values) {
        result(row++) = value;
    }
    return result;
}

void seed_real_alignment_sentinel(
    citlali::pipeline::TimestreamAlignmentState &state) {
    state.common_time = coordinate_axis({1.0, 2.0});
    state.masks.push_back(Eigen::VectorXi::Zero(2));
    state.network_masks.emplace(12, Eigen::VectorXi::Zero(2));
    state.network_times.push_back(coordinate_axis({3.0, 4.0}));
    state.gaps.emplace("real-sentinel", 7);
    state.start_indices = {8};
    state.end_indices = {9};
    state.hwpr_start_index = 4;
    state.hwpr_end_index = 5;
    state.grid.initialized = true;
    state.interfaces.push_back({"real-sentinel"});
    state.telescope.initialized = true;
    state.hwpr =
        citlali::pipeline::bounded_nonpolarimetric_hwpr_summary(true);
    state.exceptions.push_back({"real-sentinel"});
    state.field_registry_version = "real-sentinel";
}

void begin_simulated_raw_observation(SimulatedAlignmentEngine &engine) {
    engine.raw_timestream_plan.reset_from_request(
        {}, engine.typed_config.interface_sync);
    (void)engine.raw_timestream_plan.begin_observation();
}

TEST(sci_align_t10, fixed_duration_is_half_open_and_retains_final_partial) {
    const auto plan =
        citlali::pipeline::sci_align::make_fixed_duration_scan_plan(
            7, 3.0, 1.0);

    ASSERT_EQ(plan.records.size(), 3U);
    EXPECT_TRUE(plan.physical_records.empty());
    EXPECT_FALSE(plan.records[0].physical_id.has_value());
    EXPECT_EQ(
        plan.records[0].identity_authority,
        "requested_processing_chunk_under_continuous_observation_no_physical_scan_authority");
    EXPECT_EQ(plan.records[0].processing,
              (HalfOpenInterval{0, 3}));
    EXPECT_EQ(plan.records[0].science.start, 0);
    EXPECT_EQ(plan.records[0].science.stop, 3);
    EXPECT_EQ(plan.records[1].science.start, 3);
    EXPECT_EQ(plan.records[1].science.stop, 6);
    EXPECT_EQ(plan.records[2].science.start, 6);
    EXPECT_EQ(plan.records[2].science.stop, 7);
    EXPECT_EQ(plan.records[2].status, ScanStatus::partial_support);
    EXPECT_FALSE(plan.records[2].legacy_processing_admitted);
    EXPECT_EQ(plan.records[2].compatibility_ordinal, -1);
    EXPECT_TRUE(plan.records[1].science.contains(3));
    EXPECT_FALSE(plan.records[0].science.contains(3));

    const auto legacy =
        citlali::pipeline::sci_align::compatibility_scan_indices(plan);
    ASSERT_EQ(legacy.cols(), 2);
    EXPECT_EQ(legacy(0, 0), 0);
    EXPECT_EQ(legacy(1, 0), 2);
    EXPECT_EQ(legacy(0, 1), 3);
    EXPECT_EQ(legacy(1, 1), 5);
}

TEST(sci_align_t10,
     governing_support_anchors_chunks_at_global_zero_without_union_edge_rephase) {
    constexpr Eigen::Index union_count = 7700;
    const HalfOpenInterval governing_support{1, 7698};
    const auto plan =
        citlali::pipeline::sci_align::make_fixed_duration_scan_plan(
            union_count, governing_support, 5.0, 0.008192, 32);

    ASSERT_EQ(plan.records.size(), 13U);
    EXPECT_EQ(plan.observation_sample_count, union_count);
    EXPECT_EQ(plan.records.front().processing,
              (HalfOpenInterval{1, 611}));
    EXPECT_EQ(plan.records.front().context,
              (HalfOpenInterval{1, 643}));
    EXPECT_EQ(plan.records[1].processing,
              (HalfOpenInterval{611, 1221}));
    EXPECT_EQ(plan.records.back().processing,
              (HalfOpenInterval{7321, 7698}));
    EXPECT_EQ(plan.records.back().context,
              (HalfOpenInterval{7289, 7698}));
    EXPECT_EQ(plan.records.back().status, ScanStatus::partial_support);
    EXPECT_FALSE(plan.records.back().legacy_processing_admitted);
    EXPECT_EQ(plan.compatibility_to_stable_id.size(), 12U);

    const auto default_plan =
        citlali::pipeline::sci_align::make_fixed_duration_scan_plan(
            union_count, 5.0, 0.008192, 32);
    EXPECT_EQ(default_plan.records.front().processing,
              (HalfOpenInterval{0, 610}));
    EXPECT_EQ(default_plan.records.front().context,
              (HalfOpenInterval{0, 642}));
}

TEST(sci_align_t10,
     governing_support_balances_fixed_count_without_admitting_union_edges) {
    const auto plan =
        citlali::pipeline::sci_align::make_number_scan_plan(
            10, HalfOpenInterval{1, 8}, 3, 1.0, 2);

    ASSERT_EQ(plan.records.size(), 3U);
    EXPECT_EQ(plan.records[0].processing, (HalfOpenInterval{1, 4}));
    EXPECT_EQ(plan.records[1].processing, (HalfOpenInterval{4, 6}));
    EXPECT_EQ(plan.records[2].processing, (HalfOpenInterval{6, 8}));
    EXPECT_EQ(plan.records[0].context, (HalfOpenInterval{1, 6}));
    EXPECT_EQ(plan.records[2].context, (HalfOpenInterval{4, 8}));
    const auto legacy =
        citlali::pipeline::sci_align::compatibility_scan_indices(plan);
    ASSERT_EQ(legacy.cols(), 3);
    EXPECT_EQ(legacy(0, 0), 1);
    EXPECT_EQ(legacy(1, 0), 2);
    EXPECT_EQ(legacy(0, 1), 3);
    EXPECT_EQ(legacy(1, 1), 4);
    EXPECT_EQ(legacy(0, 2), 5);
    EXPECT_EQ(legacy(1, 2), 6);
}

TEST(sci_align_t10, fixed_count_distributes_every_sample_deterministically) {
    const auto plan =
        citlali::pipeline::sci_align::make_number_scan_plan(7, 3, 1.0);

    ASSERT_EQ(plan.records.size(), 3U);
    EXPECT_TRUE(plan.physical_records.empty());
    EXPECT_FALSE(plan.records[2].physical_id.has_value());
    EXPECT_EQ(plan.records[2].processing,
              (HalfOpenInterval{5, 7}));
    EXPECT_EQ(plan.records[0].science.size(), 3);
    EXPECT_EQ(plan.records[1].science.size(), 2);
    EXPECT_EQ(plan.records[2].science.size(), 2);
    EXPECT_EQ(plan.records.front().science.start, 0);
    EXPECT_EQ(plan.records.back().science.stop, 7);
    const auto legacy =
        citlali::pipeline::sci_align::compatibility_scan_indices(plan);
    ASSERT_EQ(legacy.cols(), 3);
    EXPECT_EQ(legacy(0, 0), 0);
    EXPECT_EQ(legacy(1, 0), 1);
    EXPECT_EQ(legacy(0, 1), 2);
    EXPECT_EQ(legacy(1, 1), 3);
    EXPECT_EQ(legacy(0, 2), 4);
    EXPECT_EQ(legacy(1, 2), 5);
}

TEST(sci_align_t10,
     number_scan_count_rejects_malformed_or_out_of_support_values_before_narrowing) {
    using citlali::pipeline::sci_align::checked_number_scan_count;

    EXPECT_EQ(checked_number_scan_count(3.0, 7), 3);
    EXPECT_THROW(checked_number_scan_count(3.5, 7), std::runtime_error);
    EXPECT_THROW(checked_number_scan_count(8.0, 7), std::overflow_error);
    EXPECT_THROW(
        checked_number_scan_count(
            std::ldexp(1.0, std::numeric_limits<Eigen::Index>::digits), 7),
        std::overflow_error);
}

TEST(sci_align_t10, duration_uses_round_half_up) {
    const auto plan =
        citlali::pipeline::sci_align::make_fixed_duration_scan_plan(
            8, 2.5, 1.0);
    ASSERT_EQ(plan.records.size(), 3U);
    EXPECT_TRUE(plan.physical_records.empty());
    EXPECT_FALSE(plan.records[0].physical_id.has_value());
    EXPECT_EQ(plan.records[0].processing, (HalfOpenInterval{0, 3}));
    EXPECT_DOUBLE_EQ(plan.effective_duration_sec, 3.0);
    EXPECT_EQ(plan.records[0].science.size(), 3);
}

TEST(sci_align_t11, raster_restores_first_false_and_retains_short_identity) {
    const std::vector<unsigned char> composite{
        1, 0, 0, 1, 0, 0, 0, 1, 0,
    };
    const auto plan =
        citlali::pipeline::sci_align::make_raster_compatibility_scan_plan(
            composite, 1.0, 1, 2.0);

    ASSERT_EQ(plan.records.size(), 3U);
    EXPECT_TRUE(plan.physical_records.empty());
    EXPECT_FALSE(plan.records[0].physical_id.has_value());
    EXPECT_EQ(
        plan.records[0].identity_authority,
        "legacy_inferred_raster_compatibility_segment_not_physical");
    EXPECT_EQ(plan.records[0].processing, (HalfOpenInterval{1, 3}));
    EXPECT_EQ(plan.records[0].stable_id, 0);
    EXPECT_EQ(plan.records[0].science.start, 1);
    EXPECT_EQ(plan.records[0].science.stop, 3);
    EXPECT_EQ(plan.records[0].context.start, 0);
    EXPECT_EQ(plan.records[0].context.stop, 4);
    EXPECT_EQ(plan.records[0].science, (HalfOpenInterval{1, 3}));
    EXPECT_EQ(plan.records[2].stable_id, 2);
    EXPECT_EQ(plan.records[2].status, ScanStatus::short_support);
    EXPECT_FALSE(plan.records[2].legacy_processing_admitted);
    EXPECT_EQ(plan.records[2].compatibility_ordinal, -1);
    EXPECT_EQ(plan.compatibility_to_stable_id,
              (std::vector<Eigen::Index>{1}));

    const auto legacy =
        citlali::pipeline::sci_align::compatibility_scan_indices(plan);
    ASSERT_EQ(legacy.cols(), 1);
    EXPECT_EQ(legacy(0, 0), 5);
    EXPECT_EQ(legacy(1, 0), 6);
    EXPECT_EQ(legacy(2, 0), 4);
    EXPECT_EQ(legacy(3, 0), 7);
}

TEST(sci_align_t11,
     raster_keeps_corrected_identity_separate_from_legacy_edge_windows) {
    const std::vector<unsigned char> composite{
        1, 0, 0, 0, 1, 0, 0, 0, 1,
    };
    const auto plan =
        citlali::pipeline::sci_align::make_raster_compatibility_scan_plan(
            composite, 1.0, 1, 1.0, 1);

    ASSERT_EQ(plan.records.size(), 2U);
    EXPECT_EQ(plan.records[0].science, (HalfOpenInterval{1, 4}));
    EXPECT_EQ(plan.records[1].science, (HalfOpenInterval{5, 8}));
    EXPECT_EQ(*plan.records[0].compatibility_science,
              (HalfOpenInterval{3, 4}));
    EXPECT_EQ(*plan.records[1].compatibility_science,
              (HalfOpenInterval{6, 7}));
    EXPECT_EQ(*plan.records[0].compatibility_context,
              (HalfOpenInterval{1, 5}));
    EXPECT_EQ(*plan.records[1].compatibility_context,
              (HalfOpenInterval{5, 9}));

    const auto legacy =
        citlali::pipeline::sci_align::compatibility_scan_indices(plan);
    ASSERT_EQ(legacy.cols(), 2);
    EXPECT_EQ(legacy(0, 0), 3);
    EXPECT_EQ(legacy(1, 0), 3);
    EXPECT_EQ(legacy(2, 0), 1);
    EXPECT_EQ(legacy(3, 0), 4);
    EXPECT_EQ(legacy(0, 1), 6);
    EXPECT_EQ(legacy(1, 1), 6);
    EXPECT_EQ(legacy(2, 1), 5);
    EXPECT_EQ(legacy(3, 1), 8);
}

TEST(sci_align_t11,
     beammap_persisted_scan_samples_use_governing_local_origin) {
    Eigen::Matrix<Eigen::Index, 4, Eigen::Dynamic> scan_indices(4, 1);
    scan_indices << 101, 110, 95, 116;
    struct MockPtc {
        struct {
            Eigen::MatrixXd data;
        } scans;
    };
    std::vector<MockPtc> ptcs(1);
    ptcs[0].scans.data.resize(10, 1);
    const std::vector<double> distances{0.25};
    std::vector<int> scan_index(1, -1);
    std::vector<int> kind(1, -1);
    std::vector<int> n_samples(1, -1);
    std::vector<int> inner_start(1, -1);
    std::vector<int> inner_end(1, -1);
    std::vector<int> outer_start(1, -1);
    std::vector<int> outer_end(1, -1);
    std::vector<double> source_distance(1, -1.0);

    beammap_detector_tod_selection::fill_slot_scan_metadata(
        0, 0, 1, 0, 1, 1, scan_indices, ptcs, distances,
        scan_index, kind, n_samples, inner_start, inner_end,
        outer_start, outer_end, source_distance, 1);

    EXPECT_EQ(inner_start[0], 100);
    EXPECT_EQ(inner_end[0], 109);
    EXPECT_EQ(outer_start[0], 94);
    EXPECT_EQ(outer_end[0], 115);
    EXPECT_EQ(n_samples[0], 10);
    EXPECT_DOUBLE_EQ(source_distance[0], 0.25);
}

TEST(sci_align_t11,
     raw_hold_bits_scan_predicate_and_emitted_alias_remain_distinct) {
    Eigen::Matrix<std::uint64_t, Eigen::Dynamic, 1> raw_words(8);
    raw_words << 0, 2, 8, 10, 64, 66, 72, 74;
    const auto raw_words_before = raw_words;

    // Include fractional values on both sides of a transition.  They are the
    // legacy whole-word linear result, not typed raw words.
    Eigen::VectorXd aligned_numeric(11);
    aligned_numeric << 0.0, 1.0, 2.0, 5.0, 8.0, 36.0, 64.0, 69.0,
        74.0, 37.0, 0.0;
    const auto aligned_numeric_before = aligned_numeric;
    const std::vector<unsigned char> outside{
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    const Eigen::VectorXd emitted =
        citlali::pipeline::sci_align::
            legacy_hold_emitted_compatibility_view(aligned_numeric);

    const auto composite =
        citlali::pipeline::sci_align::compose_legacy_hold_and_outside(
            aligned_numeric, outside);

    EXPECT_EQ(composite,
              (std::vector<unsigned char>{1, 1, 1, 1, 1, 1, 1, 1,
                                          1, 1, 0}));
    EXPECT_EQ(emitted,
              (Eigen::VectorXd(11) << 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 0.0)
                  .finished());
    EXPECT_DOUBLE_EQ(aligned_numeric(9), 37.0);
    EXPECT_DOUBLE_EQ(emitted(9), 1.0);
    EXPECT_DOUBLE_EQ(emitted(10), 0.0);
    EXPECT_EQ(outside.front(), 1);
    EXPECT_DOUBLE_EQ(emitted(0), 0.0);
    EXPECT_EQ(aligned_numeric, aligned_numeric_before);
    EXPECT_EQ(raw_words, raw_words_before);
    EXPECT_EQ(raw_words(7) & std::uint64_t{0x02}, std::uint64_t{0x02});
    EXPECT_EQ(raw_words(7) & std::uint64_t{0x08}, std::uint64_t{0x08});
    EXPECT_EQ(raw_words(7) & std::uint64_t{0x40}, std::uint64_t{0x40});

    using citlali::pipeline::sci_align::TelescopeHoldReason;
    using citlali::pipeline::sci_align::native_hold_word_has_reason;
    using citlali::pipeline::sci_align::native_hold_word_science_valid;
    using citlali::pipeline::sci_align::native_hold_word_unknown_bits;
    EXPECT_EQ(citlali::pipeline::sci_align::telescope_hold_defined_mask,
              std::uint64_t{0x7e});
    EXPECT_TRUE(native_hold_word_science_valid(0));
    for (Eigen::Index i = 0; i < raw_words.size(); ++i) {
        EXPECT_EQ(
            citlali::pipeline::sci_align::
                legacy_hold_linear_any_nonzero_state(
                    static_cast<double>(raw_words(i))),
            !native_hold_word_science_valid(raw_words(i)));
        EXPECT_EQ(native_hold_word_unknown_bits(raw_words(i)), 0);
    }
    EXPECT_TRUE(native_hold_word_has_reason(
        std::uint64_t{74}, TelescopeHoldReason::pointing));
    EXPECT_TRUE(native_hold_word_has_reason(
        std::uint64_t{74}, TelescopeHoldReason::obs_pgm));
    EXPECT_TRUE(native_hold_word_has_reason(
        std::uint64_t{74}, TelescopeHoldReason::m3));
    EXPECT_FALSE(native_hold_word_has_reason(
        std::uint64_t{74}, TelescopeHoldReason::external));
    for (const auto word : {std::uint64_t{0x04}, std::uint64_t{0x10},
                            std::uint64_t{0x20}, std::uint64_t{0x01},
                            std::uint64_t{0x80}}) {
        EXPECT_FALSE(native_hold_word_science_valid(word));
    }
    EXPECT_EQ(native_hold_word_unknown_bits(std::uint64_t{0x01}),
              std::uint64_t{0x01});
    EXPECT_EQ(native_hold_word_unknown_bits(std::uint64_t{0x80}),
              std::uint64_t{0x80});
    ASSERT_EQ(
        citlali::pipeline::sci_align::telescope_hold_reason_definitions.size(),
        6U);
    constexpr std::array<std::uint64_t, 6> expected_masks{
        0x02, 0x04, 0x08, 0x10, 0x20, 0x40};
    constexpr std::array<const char *, 6> expected_names{
        "Pointing", "External", "ObsPgm", "M1", "M2", "M3"};
    constexpr std::array<bool, 6> expected_never_implemented{
        false, true, false, false, false, false};
    for (std::size_t i = 0; i < expected_masks.size(); ++i) {
        const auto &definition = citlali::pipeline::sci_align::
            telescope_hold_reason_definitions[i];
        EXPECT_EQ(citlali::pipeline::sci_align::telescope_hold_reason_mask(
                      definition.reason),
                  expected_masks[i]);
        EXPECT_STREQ(definition.producer_name, expected_names[i]);
        EXPECT_EQ(definition.declared_never_implemented,
                  expected_never_implemented[i]);
    }
    constexpr std::uint64_t mixed_defined_and_unknown = 0x83;
    EXPECT_TRUE(native_hold_word_has_reason(
        mixed_defined_and_unknown, TelescopeHoldReason::pointing));
    EXPECT_EQ(native_hold_word_unknown_bits(mixed_defined_and_unknown),
              std::uint64_t{0x81});
    EXPECT_FALSE(native_hold_word_science_valid(mixed_defined_and_unknown));
    EXPECT_STREQ(citlali::pipeline::sci_align::
                     telescope_hold_transition_side_authority,
                 "unresolved");

    Eigen::VectorXd invalid(1);
    invalid << std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        citlali::pipeline::sci_align::
            legacy_hold_emitted_compatibility_view(invalid),
        std::runtime_error);
}

TEST(sci_align_t11,
     routine_hold_output_metadata_remains_frozen_legacy_zero_one_view) {
    const auto path = std::filesystem::path(::testing::TempDir()) /
        "sci_align_hold_output_metadata.nc";
    std::filesystem::remove(path);

    {
        netCDF::NcFile file(path.string(), netCDF::NcFile::replace);
        const auto n_pts = file.addDim("n_pts", 3);
        const std::map<std::string, Eigen::VectorXd> telescope{
            {"Hold", (Eigen::VectorXd(3) << 0.0, 0.25, 0.0).finished()}};
        citlali::pipeline::add_telescope_data_vars(
            file, telescope, n_pts, netCDF::NcVar::nc_CHUNKED, {3});

        const auto hold = file.getVar("Hold");
        ASSERT_FALSE(hold.isNull());
        std::string units;
        std::string long_name;
        std::string encoding;
        std::string raw_word_availability;
        std::string comment;
        hold.getAtt("units").getValues(units);
        hold.getAtt("long_name").getValues(long_name);
        hold.getAtt("value_encoding").getValues(encoding);
        hold.getAtt("raw_word_availability")
            .getValues(raw_word_availability);
        hold.getAtt("comment").getValues(comment);
        EXPECT_EQ(units, "1");
        EXPECT_EQ(long_name,
                  "legacy_4x_linear_any_nonzero compatibility view");
        EXPECT_EQ(encoding, "0=false, 1=true");
        EXPECT_NE(raw_word_availability.find("internally"),
                  std::string::npos);
        EXPECT_NE(comment.find("post-nonzero 0/1 output"),
                  std::string::npos);
        EXPECT_NE(comment.find("not the raw word"), std::string::npos);
        // Product-contract v1 freezes this historical wording. Registry v2
        // and its digest-bound overlay, not this compatibility attribute, are
        // the current authority for native bit meanings and validity.
        EXPECT_NE(comment.find("not a producer-authoritative"),
                  std::string::npos);
        const auto frozen_attributes = hold.getAtts();
        for (const auto *attribute : {
                 "native_defined_bits", "native_semantics_authority",
                 "native_word_width_authority",
                 "native_word_science_validity",
                 "native_unknown_bit_policy", "transition_side_authority"}) {
            EXPECT_EQ(frozen_attributes.find(attribute),
                      frozen_attributes.end());
        }
    }

    std::filesystem::remove(path);
}

TEST(sci_align_t11, invalid_overlap_or_identity_is_rejected) {
    auto plan = citlali::pipeline::sci_align::make_number_scan_plan(
        7, 3, 1.0);
    plan.records[1].processing.start = 2;
    EXPECT_THROW(
        citlali::pipeline::sci_align::validate_scan_window_plan(plan),
        std::runtime_error);

    plan = citlali::pipeline::sci_align::make_number_scan_plan(7, 3, 1.0);
    plan.records[2].stable_id = 7;
    EXPECT_THROW(
        citlali::pipeline::sci_align::validate_scan_window_plan(plan),
        std::runtime_error);
}

TEST(sci_align_t16, successive_scan_plans_do_not_leak_identity_or_windows) {
    auto first = citlali::pipeline::sci_align::make_number_scan_plan(
        9, 2, 1.0, 2);
    ASSERT_EQ(first.records.size(), 2U);

    first.clear();
    EXPECT_TRUE(first.physical_records.empty());
    EXPECT_TRUE(first.records.empty());
    EXPECT_TRUE(first.compatibility_to_stable_id.empty());
    EXPECT_EQ(first.observation_sample_count, 0);

    const auto second =
        citlali::pipeline::sci_align::make_fixed_duration_scan_plan(
            4, 4.0, 1.0, 0);
    ASSERT_EQ(second.records.size(), 1U);
    EXPECT_EQ(second.records[0].stable_id, 0);
    EXPECT_EQ(second.records[0].science.start, 0);
    EXPECT_EQ(second.records[0].science.stop, 4);
}

TEST(sci_align_t16,
     real_to_sim_to_real_alignment_lifecycle_has_no_state_leakage) {
    SimulatedAlignmentEngine engine;
    seed_real_alignment_sentinel(engine.alignment);
    engine.typed_config.interface_sync.toltec_configured[0] = true;
    begin_simulated_raw_observation(engine);
    auto native_time = coordinate_axis({100.0, 100.01, 100.02});
    native_time(1) = std::nextafter(
        native_time(1), std::numeric_limits<double>::infinity());
    const double native_middle_representation = native_time(1);
    engine.telescope.tel_data["TelTime"] = native_time;
    engine.telescope.tel_data["TelUTC"] = native_time;
    const auto native_pps = coordinate_axis({7.0, 8.0, 9.0});
    engine.telescope.tel_data["PpsTime"] = native_pps;
    const SimulatedRawObs rawobs{{{"toltec0"}, {"toltec2"}}, 1};

    citlali::pipeline::reset_simulated_observation_indices(engine, rawobs);

    EXPECT_NO_THROW(
        citlali::pipeline::validate_compact_alignment_provenance(
            engine.alignment));
    EXPECT_TRUE(engine.alignment.grid.initialized);
    EXPECT_EQ(engine.alignment.grid.assignment_operator,
              "floor_q_plus_half_v1");
    EXPECT_EQ(engine.alignment.grid.physical_timestamp_semantics,
              "unavailable_no_integration_event_authority");
    EXPECT_DOUBLE_EQ(engine.alignment.grid.phase_sec, 100.0);
    EXPECT_DOUBLE_EQ(engine.alignment.grid.cadence_sec, 0.01);
    EXPECT_EQ(engine.alignment.grid.first_global_slot, 0);
    EXPECT_EQ(engine.alignment.grid.last_global_slot, 2);
    EXPECT_FALSE(
        engine.alignment.governing_compatibility_axis.initialized);
    ASSERT_EQ(engine.alignment.common_time.size(), 3);
    EXPECT_DOUBLE_EQ(engine.alignment.common_time(0), 100.0);
    EXPECT_DOUBLE_EQ(engine.alignment.common_time(2), 100.02);
    EXPECT_DOUBLE_EQ(engine.alignment.common_time(1),
                     native_middle_representation);
    EXPECT_EQ(engine.telescope.tel_data.at("TelTime"), native_time);
    EXPECT_EQ(engine.telescope.tel_data.at("TelUTC"), native_time);
    EXPECT_EQ(engine.telescope.tel_data.at("PpsTime"), native_pps);
    EXPECT_DOUBLE_EQ(
        citlali::pipeline::governing_compatibility_mean(
            engine.telescope.tel_data.at("TelTime"), engine.alignment),
        native_time.mean());
    EXPECT_EQ(
        citlali::pipeline::governing_consumer_sample_count(
            engine.alignment),
        native_time.size());
    EXPECT_EQ(engine.alignment.start_indices,
              (std::vector<Eigen::Index>{0, 0}));
    EXPECT_EQ(engine.alignment.end_indices,
              (std::vector<Eigen::Index>{2, 2}));
    EXPECT_EQ(engine.alignment.hwpr_start_index, -1);
    EXPECT_EQ(engine.alignment.hwpr_end_index, -1);
    EXPECT_TRUE(engine.alignment.gaps.empty());
    EXPECT_TRUE(engine.alignment.exceptions.empty());
    ASSERT_EQ(engine.alignment.interfaces.size(), 2U);
    EXPECT_EQ(engine.alignment.interfaces[0].interface_id, "toltec0");
    EXPECT_EQ(engine.alignment.interfaces[0].roach_index, 0);
    EXPECT_EQ(engine.alignment.interfaces[1].interface_id, "toltec2");
    EXPECT_EQ(engine.alignment.interfaces[1].roach_index, 2);
    ASSERT_EQ(engine.alignment.masks.size(), 2U);
    ASSERT_EQ(engine.alignment.network_times.size(), 2U);
    EXPECT_EQ(engine.alignment.masks[0].sum(), 3);
    EXPECT_EQ(engine.alignment.masks[1].sum(), 3);
    EXPECT_EQ(engine.alignment.network_masks.size(), 2U);
    EXPECT_EQ(engine.alignment.network_masks.at(0).sum(), 3);
    EXPECT_EQ(engine.alignment.network_masks.at(2).sum(), 3);
    EXPECT_EQ(engine.alignment.network_times[0],
              engine.alignment.common_time);
    EXPECT_EQ(engine.alignment.network_times[1],
              engine.alignment.common_time);
    EXPECT_TRUE(engine.alignment.telescope.initialized);
    EXPECT_EQ(engine.alignment.telescope.exact_target_count, 3U);
    EXPECT_EQ(engine.alignment.telescope.interpolated_target_count, 0U);
    EXPECT_EQ(engine.alignment.telescope.epoch_event_precision_authority,
              "unavailable");
    EXPECT_TRUE(engine.alignment.telescope.native_tel_utc_available);
    EXPECT_TRUE(engine.alignment.telescope.native_pps_time_available);
    EXPECT_TRUE(engine.alignment.hwpr.observation_resolved);
    EXPECT_TRUE(engine.alignment.hwpr.producer_input_present);
    EXPECT_FALSE(engine.alignment.hwpr.aligned_angle_available);
    EXPECT_TRUE(engine.alignment.hwpr.intensity_eligible);
    EXPECT_FALSE(engine.alignment.hwpr.polarization_eligible);
    EXPECT_EQ(engine.alignment.support.nominal_slot_count, 3U);
    EXPECT_EQ(engine.alignment.support.acquired_original_count, 6U);
    EXPECT_EQ(
        engine.alignment.support.timing_coordinate_valid_original_count,
        6U);
    EXPECT_EQ(
        engine.alignment.support.gap_policy_eligible_original_count, 0U);
    EXPECT_EQ(engine.alignment.support.unavailable_count, 0U);
    EXPECT_EQ(engine.alignment.field_registry_version,
              citlali::pipeline::sci_align::active_field_registry_version);

    const auto &toltec0 =
        citlali::pipeline::require_interface_offset_record(
            engine.interface_sync, "toltec0");
    EXPECT_TRUE(toltec0.applied_exactly_once);
    EXPECT_EQ(toltec0.availability,
              citlali::pipeline::OffsetAvailability::observation_resolved);
    const auto &toltec1 =
        citlali::pipeline::require_interface_offset_record(
            engine.interface_sync, "toltec1");
    EXPECT_FALSE(toltec1.applied_exactly_once);
    EXPECT_EQ(toltec1.availability,
              citlali::pipeline::OffsetAvailability::not_applicable);
    const auto &lmt = citlali::pipeline::require_interface_offset_record(
        engine.interface_sync, "lmt");
    EXPECT_TRUE(lmt.applied_exactly_once);
    ASSERT_TRUE(engine.raw_timestream_plan.observation.has_value());
    EXPECT_EQ(
        engine.raw_timestream_plan.observation->interface_offsets.size(),
        engine.interface_sync.lifecycle.size());

    // Real-observation preparation uses the same total reset.  Prove the
    // simulated runtime vectors and identities do not survive that boundary.
    citlali::pipeline::reset_alignment_observation_state(engine.alignment);
    EXPECT_FALSE(engine.alignment.grid.initialized);
    EXPECT_EQ(engine.alignment.common_time.size(), 0);
    EXPECT_TRUE(engine.alignment.masks.empty());
    EXPECT_TRUE(engine.alignment.network_masks.empty());
    EXPECT_TRUE(engine.alignment.network_times.empty());
    EXPECT_TRUE(engine.alignment.interfaces.empty());
    EXPECT_TRUE(engine.alignment.start_indices.empty());
    EXPECT_TRUE(engine.alignment.end_indices.empty());
    EXPECT_FALSE(engine.alignment.hwpr.observation_resolved);
    EXPECT_TRUE(engine.alignment.hwpr.policy.empty());
    EXPECT_TRUE(engine.alignment.field_registry_version.empty());

    seed_real_alignment_sentinel(engine.alignment);
    EXPECT_EQ(engine.alignment.network_masks.size(), 1U);
    EXPECT_EQ(engine.alignment.network_masks.count(0), 0U);
    EXPECT_EQ(engine.alignment.network_masks.count(2), 0U);
    EXPECT_EQ(engine.alignment.field_registry_version, "real-sentinel");
}

TEST(sci_align_t16,
     successive_simulations_replace_interface_and_window_identity) {
    SimulatedAlignmentEngine engine;
    begin_simulated_raw_observation(engine);
    engine.telescope.tel_data["TelTime"] =
        coordinate_axis({10.0, 10.01, 10.02});
    citlali::pipeline::reset_simulated_observation_indices(
        engine, SimulatedRawObs{{{"toltec0"}, {"toltec12"}}});

    engine.telescope.tel_data["TelTime"] =
        coordinate_axis({20.0, 20.01});
    engine.telescope.tel_data.erase("TelUTC");
    engine.telescope.tel_data.erase("PpsTime");
    citlali::pipeline::reset_simulated_observation_indices(
        engine, SimulatedRawObs{{{"toltec1"}}});

    EXPECT_NO_THROW(
        citlali::pipeline::validate_compact_alignment_provenance(
            engine.alignment));
    ASSERT_EQ(engine.alignment.interfaces.size(), 1U);
    EXPECT_EQ(engine.alignment.interfaces.front().interface_id, "toltec1");
    EXPECT_EQ(engine.alignment.network_masks.size(), 1U);
    EXPECT_EQ(engine.alignment.network_masks.count(0), 0U);
    EXPECT_EQ(engine.alignment.network_masks.count(12), 0U);
    EXPECT_EQ(engine.alignment.network_masks.at(1).size(), 2);
    EXPECT_EQ(engine.alignment.start_indices,
              (std::vector<Eigen::Index>{0}));
    EXPECT_EQ(engine.alignment.end_indices,
              (std::vector<Eigen::Index>{1}));
    EXPECT_FALSE(engine.alignment.telescope.native_tel_utc_available);
    EXPECT_FALSE(engine.alignment.telescope.native_pps_time_available);
    EXPECT_TRUE(engine.alignment.hwpr.observation_resolved);
    EXPECT_FALSE(engine.alignment.hwpr.producer_input_present);
    EXPECT_TRUE(engine.alignment.hwpr.intensity_eligible);
    EXPECT_FALSE(engine.alignment.hwpr.polarization_eligible);
    EXPECT_EQ(engine.telescope.tel_data.count("TelUTC"), 0U);
    EXPECT_EQ(engine.telescope.tel_data.count("PpsTime"), 0U);
    EXPECT_FALSE(citlali::pipeline::require_interface_offset_record(
                     engine.interface_sync, "toltec0")
                     .applied_exactly_once);
    EXPECT_TRUE(citlali::pipeline::require_interface_offset_record(
                    engine.interface_sync, "toltec1")
                    .applied_exactly_once);
}

TEST(sci_align_t16,
     malformed_simulator_axes_fail_closed_without_precision_guess) {
    SimulatedAlignmentEngine engine;
    const SimulatedRawObs rawobs{{{"toltec0"}}};

    auto expect_rejected = [&](const Eigen::VectorXd &axis, double fsmp) {
        citlali::pipeline::reset_alignment_observation_state(
            engine.alignment);
        seed_real_alignment_sentinel(engine.alignment);
        engine.telescope.fsmp = fsmp;
        engine.telescope.tel_data["TelTime"] = axis;
        EXPECT_THROW(
            citlali::pipeline::reset_simulated_observation_indices(
                engine, rawobs),
            std::exception);
        EXPECT_FALSE(engine.alignment.grid.initialized);
        EXPECT_TRUE(engine.alignment.field_registry_version.empty());
        EXPECT_TRUE(engine.alignment.gaps.empty());
        EXPECT_TRUE(engine.alignment.start_indices.empty());
        EXPECT_TRUE(engine.alignment.end_indices.empty());
        EXPECT_FALSE(engine.alignment.hwpr.observation_resolved);
        EXPECT_TRUE(engine.alignment.hwpr.policy.empty());
    };

    expect_rejected(Eigen::VectorXd{}, 100.0);
    expect_rejected(
        coordinate_axis({0.0, std::numeric_limits<double>::quiet_NaN()}),
        100.0);
    expect_rejected(coordinate_axis({0.0, 0.0}), 100.0);
    expect_rejected(coordinate_axis({0.0, 0.01, 0.021}), 100.0);
    expect_rejected(coordinate_axis({0.0, 0.01}), 0.0);

    citlali::pipeline::reset_alignment_observation_state(engine.alignment);
    seed_real_alignment_sentinel(engine.alignment);
    engine.telescope.tel_data.clear();
    engine.telescope.fsmp = 100.0;
    EXPECT_THROW(citlali::pipeline::reset_simulated_observation_indices(
                     engine, rawobs),
                 std::runtime_error);
    EXPECT_FALSE(engine.alignment.grid.initialized);
    EXPECT_TRUE(engine.alignment.field_registry_version.empty());
    EXPECT_FALSE(engine.alignment.hwpr.observation_resolved);
    EXPECT_TRUE(engine.alignment.hwpr.policy.empty());
    EXPECT_FALSE(engine.alignment.hwpr.observation_resolved);
    EXPECT_TRUE(engine.alignment.hwpr.policy.empty());
    EXPECT_TRUE(engine.alignment.gaps.empty());
}

TEST(sci_align_t16,
     simulator_rejects_hwpr_noncanonical_duplicate_and_unresolved_offsets) {
    SimulatedAlignmentEngine engine;
    engine.telescope.tel_data["TelTime"] =
        coordinate_axis({0.0, 0.01});
    seed_real_alignment_sentinel(engine.alignment);

    engine.calib.run_hwpr = true;
    EXPECT_THROW(citlali::pipeline::reset_simulated_observation_indices(
                     engine, SimulatedRawObs{{{"toltec0"}}}),
                 std::runtime_error);
    engine.calib.run_hwpr = false;
    EXPECT_FALSE(engine.alignment.grid.initialized);
    EXPECT_TRUE(engine.alignment.field_registry_version.empty());

    EXPECT_THROW(citlali::pipeline::reset_simulated_observation_indices(
                     engine, SimulatedRawObs{{{"nw0"}}}),
                 std::runtime_error);
    EXPECT_THROW(citlali::pipeline::reset_simulated_observation_indices(
                     engine,
                     SimulatedRawObs{{{"toltec0"}, {"toltec0"}}}),
                 std::runtime_error);

    engine.typed_config.interface_sync.toltec_offset_sec[0] = 0.25;
    engine.typed_config.interface_sync.toltec_configured[0] = true;
    citlali::pipeline::adapt_interface_sync_config_one_way(
        engine.typed_config.interface_sync, engine.interface_sync);
    EXPECT_THROW(citlali::pipeline::reset_simulated_observation_indices(
                     engine, SimulatedRawObs{{{"toltec0"}}}),
                 std::runtime_error);
    EXPECT_FALSE(engine.alignment.grid.initialized);
    EXPECT_TRUE(engine.alignment.field_registry_version.empty());
    const auto &failed_offset =
        citlali::pipeline::require_interface_offset_record(
            engine.interface_sync, "toltec0");
    EXPECT_FALSE(failed_offset.applied_exactly_once);
    EXPECT_EQ(failed_offset.availability,
              citlali::pipeline::OffsetAvailability::unavailable_authority);
}

}  // namespace
