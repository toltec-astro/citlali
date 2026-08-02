#include <citlali/core/config/reduction_config.h>
#include <citlali/core/pipeline/observation_exposure_time.h>
#include <citlali/core/pipeline/output_path_state.h>
#include <citlali/core/pipeline/timestream_output_provenance.h>
#include <citlali/core/pipeline/timestream_alignment_helpers.h>
#include <citlali/core/pipeline/tod_output_state.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

struct ScalarSeries {
    std::vector<double> values;

    double operator()(Eigen::Index index) const {
        return values.at(static_cast<std::size_t>(index));
    }

    Eigen::Index size() const {
        return static_cast<Eigen::Index>(values.size());
    }
};

struct TelescopeWithoutScanPlan {
    std::map<std::string, ScalarSeries> tel_data;
    Eigen::MatrixXI scan_indices;
};

struct ExposureOnlyEngine {
    TelescopeWithoutScanPlan telescope;
};

struct AlignedExposureEngine {
    TelescopeWithoutScanPlan telescope;
    citlali::pipeline::TimestreamAlignmentState alignment;
};

struct ProvenanceEngineWithoutAlignment {
    citlali::config::ReductionConfig typed_config;
    citlali::pipeline::TodOutputState tod_outputs;
    citlali::pipeline::OutputPathState output_paths;
    TelescopeWithoutScanPlan telescope;
};

struct ProvenanceEngine {
    citlali::config::ReductionConfig typed_config;
    citlali::pipeline::TodOutputState tod_outputs;
    citlali::pipeline::OutputPathState output_paths;
    TelescopeWithoutScanPlan telescope;
    citlali::pipeline::TimestreamAlignmentState alignment;
};

struct ScanPlanProvenanceEngine {
    citlali::config::ReductionConfig typed_config;
    citlali::pipeline::TodOutputState tod_outputs;
    citlali::pipeline::OutputPathState output_paths;
    struct {
        Eigen::MatrixXI scan_indices;
        citlali::pipeline::sci_align::ScanWindowPlan scan_plan;
    } telescope;
};

struct CompletedProvenanceEngine {
    citlali::config::ReductionConfig typed_config;
    citlali::pipeline::TodOutputState tod_outputs;
    citlali::pipeline::OutputPathState output_paths;
    struct {
        Eigen::MatrixXI scan_indices;
        citlali::pipeline::sci_align::ScanWindowPlan scan_plan;
    } telescope;
    citlali::pipeline::TimestreamAlignmentState alignment;
};

citlali::pipeline::TimestreamAlignmentState make_alignment_state() {
    citlali::pipeline::TimestreamAlignmentState state;
    state.grid.initialized = true;
    state.grid.phase_sec = 5.5;
    state.grid.cadence_sec = 0.5;
    state.grid.exclusive_half_cell_sec = 0.25;
    state.grid.first_global_slot = -1;
    state.grid.last_global_slot = 2;
    state.grid.physical_timestamp_semantics = "unavailable";
    state.common_time.resize(4);
    state.common_time << 5.0, 5.5, 6.0, 6.5;
    state.governing_compatibility_axis =
        citlali::pipeline::make_governing_gap_compatibility_axis(
            state.grid, 6.5);

    Eigen::VectorXi first(4);
    first << 1, 1, 0, 1;
    Eigen::VectorXi second(4);
    second << 0, 1, 1, 1;
    state.masks = {first, second};

    state.interfaces.push_back({
        "toltec0", 0, 3, 3, -0.01, 0.02, 0.02, -1, 2, 0, 0});
    state.interfaces.push_back({
        "toltec1", 1, 3, 3, -0.02, 0.01, 0.02, 0, 2, 1, 0});
    state.telescope.initialized = true;
    state.telescope.native_row_count = 5;
    state.telescope.native_first_coordinate_sec = 4.9;
    state.telescope.native_last_coordinate_sec = 6.6;
    state.telescope.interpolated_target_count = 4;
    state.telescope.minimum_used_bracket_span_sec = 0.4;
    state.telescope.maximum_used_bracket_span_sec = 0.5;
    state.telescope.native_tel_utc_available = true;
    state.telescope.native_pps_time_available = true;
    state.hwpr =
        citlali::pipeline::bounded_nonpolarimetric_hwpr_summary(true);
    state.exceptions.push_back({
        "toltec0", "detector_acquisition", 2, 3, "unavailable",
        "unavailable", "bounded_continuity_candidate", "packet_gap",
        1, 3});
    state.exceptions.push_back({
        "toltec1", "detector_acquisition", 0, 1, "unavailable",
        "unavailable", "none", "union_edge_no_extrapolation", -1, -1});

    state.support.nominal_slot_count = 4;
    state.support.acquired_original_count = 6;
    state.support.timing_coordinate_valid_original_count = 6;
    state.support.synthesized_count = 0;
    state.support.unavailable_count = 2;
    state.support.guarded_original_count = 0;
    state.support.gap_policy_eligible_original_count = 6;
    state.support.nominal_span_sec = 2.0;
    state.support.acquired_original_cadence_weighted_support_sec = 3.0;
    state.field_registry_version = "sci-align-active-field-registry-v1";
    return state;
}

citlali::pipeline::sci_align::ScanWindowPlan make_scan_plan() {
    citlali::pipeline::sci_align::ScanWindowPlan plan;
    plan.policy = "test_full_observation";
    plan.observation_sample_count = 4;
    plan.physical_records.push_back(
        {0, {0, 4}, "test_continuous_observation"});
    citlali::pipeline::sci_align::ScanWindowRecord record;
    record.stable_id = 0;
    record.physical_id = 0;
    record.identity_authority = "test_processing_chunk";
    record.processing = {0, 4};
    record.science = {0, 4};
    record.context = {0, 4};
    record.status = citlali::pipeline::sci_align::ScanStatus::usable;
    record.legacy_processing_admitted = true;
    record.compatibility_ordinal = 0;
    plan.records.push_back(record);
    plan.compatibility_to_stable_id.push_back(0);
    return plan;
}

ScanPlanProvenanceEngine make_raster_output_provenance_engine() {
    ScanPlanProvenanceEngine engine;
    auto &output = engine.typed_config.timestream.output;
    output.type = citlali::config::TodOutputType::both;
    output.raw_time_chunk.enabled = true;
    output.raw_time_chunk.mode =
        citlali::config::TodStreamOutputMode::mini_outer;
    output.processed_time_chunk.enabled = true;
    output.processed_time_chunk.mode =
        citlali::config::TodStreamOutputMode::mini_outer;

    const std::vector<unsigned char> composite{
        1, 0, 0, 1, 0, 0, 0, 1, 0,
    };
    engine.telescope.scan_plan =
        citlali::pipeline::sci_align::make_raster_compatibility_scan_plan(
            composite, 1.0, 1, 1.0);
    engine.telescope.scan_indices.resize(4, 2);
    engine.tod_outputs.rtc_scan_to_output_scan.resize(2);
    engine.tod_outputs.rtc_scan_to_output_scan << -1, 0;
    engine.tod_outputs.ptc_scan_to_output_scan.resize(2);
    engine.tod_outputs.ptc_scan_to_output_scan << 0, 1;
    engine.tod_outputs.n_rtc_output_scans = 1;
    engine.tod_outputs.n_ptc_output_scans = 2;
    return engine;
}

void resolve_processing_support(
    citlali::pipeline::TimestreamAlignmentState &state) {
    // One admitted scan produces one compact disposition per interface.
    citlali::pipeline::AlignmentChunkDisposition first;
    first.stable_scan_id = 0;
    first.compatibility_ordinal = 0;
    first.interface_id = "toltec0";
    first.roach_index = 0;
    first.context_start = 0;
    first.context_stop = 4;
    first.cumulative_missing_count = 1;
    first.longest_missing_run_count = 1;
    first.continuity_surrogate_permitted = true;
    first.synthesized_missing_runs.push_back({2, 3});

    citlali::pipeline::AlignmentChunkDisposition second;
    second.stable_scan_id = 0;
    second.compatibility_ordinal = 0;
    second.interface_id = "toltec1";
    second.roach_index = 1;
    second.context_start = 0;
    second.context_stop = 4;
    second.cumulative_missing_count = 1;
    second.longest_missing_run_count = 1;
    second.continuity_surrogate_permitted = true;
    second.unavailable_missing_runs.push_back({0, 1});

    state.chunk_dispositions = {first, second};
    state.processing_support.observation_resolved = true;
    state.processing_support.signal_domain = "xs";
    state.processing_support.synthesized_processing_occurrence_count = 1;
    state.processing_support.unavailable_processing_occurrence_count = 1;
    state.support.synthesized_count = 1;
    state.support.unavailable_count = 1;
}

template <class Engine>
void prepare_timestream_output_state(Engine &engine) {
    auto &output = engine.typed_config.timestream.output;
    output.type = citlali::config::TodOutputType::both;
    output.raw_time_chunk.enabled = true;
    output.processed_time_chunk.enabled = true;
    engine.telescope.scan_indices.resize(4, 2);
    engine.tod_outputs.rtc_scan_to_output_scan.resize(2);
    engine.tod_outputs.rtc_scan_to_output_scan << 0, 1;
    engine.tod_outputs.ptc_scan_to_output_scan.resize(2);
    engine.tod_outputs.ptc_scan_to_output_scan << 0, 1;
    engine.tod_outputs.n_rtc_output_scans = 2;
    engine.tod_outputs.n_ptc_output_scans = 2;
}

TEST(sci_align_exposure, preserves_endpoint_fallback_without_alignment) {
    ExposureOnlyEngine engine;
    engine.telescope.tel_data["TelTime"].values = {10.0, 12.5, 14.0};

    EXPECT_DOUBLE_EQ(
        citlali::pipeline::calculate_observation_exposure_time(engine), 4.0);
}

TEST(sci_align_exposure,
     records_compact_union_exposure_without_redefining_legacy_exptime) {
    AlignedExposureEngine engine;
    engine.alignment = make_alignment_state();
    engine.telescope.tel_data["TelTime"].values = {5.0, 5.5, 6.0, 6.5};
    const auto summary =
        citlali::pipeline::aligned_observation_exposure_summary(
            engine.alignment);

    EXPECT_TRUE(summary.alignment_initialized);
    EXPECT_EQ(summary.nominal_common_axis_slot_count, 4U);
    EXPECT_EQ(summary.acquired_original_interface_slot_count, 6U);
    EXPECT_EQ(
        summary.timing_coordinate_valid_original_interface_slot_count, 6U);
    EXPECT_EQ(summary.synthesized_interface_slot_count, 0U);
    EXPECT_EQ(summary.unavailable_interface_slot_count, 2U);
    EXPECT_EQ(summary.acquired_original_observation_union_slot_count, 4U);
    EXPECT_DOUBLE_EQ(summary.nominal_support_span_sec, 2.0);
    EXPECT_DOUBLE_EQ(
        summary.acquired_original_observation_cadence_weighted_support_sec,
        2.0);
    EXPECT_DOUBLE_EQ(
        citlali::pipeline::calculate_observation_exposure_time(engine), 1.0);
}

TEST(sci_align_exposure,
     governing_gap_axis_preserves_exact_linspaced_assigned_times) {
    citlali::pipeline::TimestreamAlignmentState state;
    state.grid.initialized = true;
    state.grid.phase_sec = 1771482887.3238158;
    state.grid.cadence_sec = 0.008192;
    state.grid.exclusive_half_cell_sec = 0.004096;
    state.grid.first_global_slot = -1;
    state.grid.last_global_slot = 7698;
    state.governing_compatibility_axis =
        citlali::pipeline::make_governing_gap_compatibility_axis(
            state.grid, 1771482950.3776398);

    state.common_time.resize(7700);
    for (Eigen::Index local = 0; local < state.common_time.size(); ++local) {
        const auto global = state.grid.first_global_slot + local;
        state.common_time(local) =
            state.grid.phase_sec +
            static_cast<double>(global) * state.grid.cadence_sec;
    }

    const Eigen::VectorXd expected = Eigen::VectorXd::LinSpaced(
        7697, state.grid.phase_sec,
        state.grid.phase_sec + state.grid.cadence_sec * 7696.0);
    Eigen::Index formula_difference_count = 0;
    for (Eigen::Index local = 0; local < expected.size(); ++local) {
        const double direct_formula =
            state.grid.phase_sec +
            static_cast<double>(local) * state.grid.cadence_sec;
        formula_difference_count += expected(local) != direct_formula;
    }
    EXPECT_GT(formula_difference_count, 0);

    citlali::pipeline::install_governing_compatibility_assigned_times(
        state);
    EXPECT_EQ(
        citlali::pipeline::governing_compatibility_sample_count(
            state.governing_compatibility_axis),
        7697);
    EXPECT_EQ(
        citlali::pipeline::governing_compatibility_segment(
            state.common_time, state),
        expected);
    EXPECT_DOUBLE_EQ(
        citlali::pipeline::governing_compatibility_mean(
            state.common_time, state),
        expected.mean());
    EXPECT_DOUBLE_EQ(
        citlali::pipeline::governing_compatibility_start_value(
            state.common_time, state),
        expected(0));
    EXPECT_DOUBLE_EQ(
        citlali::pipeline::governing_compatibility_stop_value(
            state.common_time, state),
        expected(expected.size() - 1));
    EXPECT_NO_THROW(
        citlali::pipeline::validate_governing_compatibility_assigned_times(
            state));

    const Eigen::VectorXd wrong_size = Eigen::VectorXd::Zero(7699);
    EXPECT_THROW(
        citlali::pipeline::governing_compatibility_mean(
            wrong_size, state),
        std::logic_error);
}

TEST(sci_align_exposure,
     rejects_timing_coordinate_validity_without_acquired_support) {
    auto alignment = make_alignment_state();
    alignment.support.timing_coordinate_valid_original_count = 5;
    alignment.support.gap_policy_eligible_original_count = 5;

    EXPECT_THROW(
        citlali::pipeline::aligned_observation_exposure_summary(alignment),
        std::logic_error);
}

TEST(sci_align_telescope_support,
     records_native_coordinate_and_adjacent_brackets_without_runtime_ceiling) {
    std::map<std::string, Eigen::VectorXd> telescope;
    const Eigen::VectorXd native_time =
        (Eigen::VectorXd(3) << 0.0, 1.0, 2.0).finished();
    for (const auto &entry :
         citlali::pipeline::sci_align::active_field_registry) {
        Eigen::VectorXd values = Eigen::VectorXd::Constant(3, 0.25);
        if (entry.canonical_name == "TelTime" ||
            entry.canonical_name == "TelUTC" ||
            entry.canonical_name == "PpsTime") {
            values = native_time;
        }
        else if (entry.canonical_name == "Hold") {
            values.setZero();
        }
        telescope[std::string{entry.canonical_name}] = std::move(values);
    }
    const Eigen::VectorXd common_time =
        (Eigen::VectorXd(2) << 0.5, 1.5).finished();
    citlali::pipeline::TimestreamAlignmentState state;

    citlali::pipeline::interpolate_telescope_data_to_common_time(
        telescope, common_time, false, &state);

    EXPECT_TRUE(state.telescope.initialized);
    EXPECT_EQ(state.telescope.native_row_count, 3);
    EXPECT_EQ(state.telescope.exact_target_count, 0U);
    EXPECT_EQ(state.telescope.interpolated_target_count, 2U);
    EXPECT_DOUBLE_EQ(state.telescope.minimum_used_bracket_span_sec, 1.0);
    EXPECT_DOUBLE_EQ(state.telescope.maximum_used_bracket_span_sec, 1.0);
    EXPECT_TRUE(state.telescope.native_tel_utc_available);
    EXPECT_TRUE(state.telescope.native_pps_time_available);
    EXPECT_EQ(state.telescope.epoch_event_precision_authority,
              "unavailable");
    EXPECT_TRUE(telescope.at("TelTime") == common_time);
    EXPECT_TRUE(telescope.at("TelUTC") == common_time);
}

TEST(sci_align_telescope_support,
     circular_alignment_preserves_native_numerical_branch) {
    std::map<std::string, Eigen::VectorXd> telescope;
    const Eigen::VectorXd native_time =
        (Eigen::VectorXd(3) << 0.0, 1.0, 2.0).finished();
    for (const auto &entry :
         citlali::pipeline::sci_align::active_field_registry) {
        Eigen::VectorXd values = Eigen::VectorXd::Constant(3, 0.25);
        if (entry.canonical_name == "TelTime" ||
            entry.canonical_name == "TelUTC" ||
            entry.canonical_name == "PpsTime") {
            values = native_time;
        }
        else if (entry.canonical_name == "Hold") {
            values.setZero();
        }
        else if (entry.permitted_operator ==
                 citlali::pipeline::sci_align::FieldOperator::
                     bracketed_shortest_arc) {
            values << -2.25, -2.20, -2.15;
        }
        telescope[std::string{entry.canonical_name}] = std::move(values);
    }
    const Eigen::VectorXd common_time =
        (Eigen::VectorXd(5) << 0.0, 0.5, 1.0, 1.5, 2.0).finished();

    citlali::pipeline::interpolate_telescope_data_to_common_time(
        telescope, common_time, false);

    for (const auto &entry :
         citlali::pipeline::sci_align::active_field_registry) {
        if (entry.permitted_operator !=
            citlali::pipeline::sci_align::FieldOperator::
                bracketed_shortest_arc) {
            continue;
        }
        const auto &aligned = telescope.at(
            std::string{entry.canonical_name});
        EXPECT_DOUBLE_EQ(aligned(0), -2.25);
        EXPECT_DOUBLE_EQ(aligned(1), -2.225);
        EXPECT_DOUBLE_EQ(aligned(2), -2.20);
        EXPECT_DOUBLE_EQ(aligned(3), -2.175);
        EXPECT_DOUBLE_EQ(aligned(4), -2.15);
    }
}

TEST(sci_align_provenance, serializes_compact_alignment_state) {
    ProvenanceEngine engine;
    prepare_timestream_output_state(engine);
    engine.alignment = make_alignment_state();

    const auto node =
        citlali::pipeline::timestream_output_provenance_node(engine);
    const auto alignment = node["realized"]["sci_align_alignment"];
    ASSERT_TRUE(alignment);
    EXPECT_TRUE(alignment["initialized"].as<bool>());
    EXPECT_EQ(alignment["representation"].as<std::string>(),
              "compact_generative_grid_plus_exception_runs_v1");
    EXPECT_FALSE(alignment["dense_mapping_persisted"].as<bool>());
    EXPECT_EQ(alignment["grid"]["physical_timestamp_semantics"]
                  .as<std::string>(),
              "unavailable");
    EXPECT_EQ(
        alignment["governing_compatibility_axis"]["availability"]
            .as<std::string>(),
        "available");
    EXPECT_EQ(
        alignment["governing_compatibility_axis"]
                 ["assigned_time_constructor"]
                     .as<std::string>(),
        "eigen_vectorxd_linspaced_9aae_gap_v1");
    EXPECT_EQ(
        alignment["governing_compatibility_axis"]["global_start"]
            .as<std::int64_t>(),
        0);
    EXPECT_EQ(
        alignment["governing_compatibility_axis"]["global_stop"]
            .as<std::int64_t>(),
        3);
    EXPECT_EQ(
        alignment["governing_compatibility_axis"]["union_local_start"]
            .as<Eigen::Index>(),
        1);
    EXPECT_FALSE(
        alignment["governing_compatibility_axis"]["dense_axis_persisted"]
            .as<bool>());
    EXPECT_EQ(alignment["telescope"]["native_coordinate_identity"]
                  .as<std::string>(),
              "Data.TelescopeBackend.TelTime");
    EXPECT_FALSE(
        alignment["telescope"]
                 ["general_numeric_runtime_bracket_limit_available"]
                     .as<bool>());
    EXPECT_EQ(alignment["hwpr"]["policy"].as<std::string>(),
              "bounded_nonpolarimetric_optional_hwpr_v1");
    EXPECT_TRUE(
        alignment["hwpr"]["producer_input_present"].as<bool>());
    EXPECT_FALSE(
        alignment["hwpr"]["aligned_angle_available"].as<bool>());
    EXPECT_TRUE(alignment["hwpr"]["intensity_eligible"].as<bool>());
    EXPECT_FALSE(
        alignment["hwpr"]["polarization_eligible"].as<bool>());
    EXPECT_EQ(
        alignment["hwpr"]["physical_timestamp_semantics"]
            .as<std::string>(),
        "unavailable_no_producer_integration_event_authority");
    EXPECT_EQ(alignment["hwpr"]["demodulation_semantics"]
                  .as<std::string>(),
              "unavailable_not_authorized_by_bounded_profile");
    EXPECT_FALSE(
        alignment["hwpr"]["dense_angle_mapping_persisted"].as<bool>());
    EXPECT_DOUBLE_EQ(
        alignment["telescope"]["maximum_used_bracket_span_sec"]
            .as<double>(),
        0.5);
    EXPECT_EQ(alignment["support"]["nominal_common_axis_slot_count"]
                  .as<std::uint64_t>(),
              4U);
    EXPECT_EQ(alignment["support"]
                  ["acquired_original_observation_union_slot_count"]
                      .as<std::uint64_t>(),
              4U);
    EXPECT_DOUBLE_EQ(
        alignment["support"]
                 ["acquired_original_observation_cadence_weighted_support_sec"]
                     .as<double>(),
        2.0);
    EXPECT_FALSE(
        alignment["support"]
                 ["acquired_original_aggregate_interface_exposure_sec"]);
    EXPECT_FALSE(alignment["support"]
                          ["physical_detector_integration_exposure_available"]
                              .as<bool>());
    EXPECT_EQ(alignment["interfaces"].size(), 2U);
    EXPECT_EQ(alignment["exception_runs"].size(), 2U);
    EXPECT_EQ(alignment["exception_run_contract"]["source_slot_identity"]
                  .as<std::string>(),
              "zero_based_observation_common_axis_slot");
    EXPECT_EQ(alignment["exception_run_contract"]["continuity_weight_rule"]
                  ["operator"]
                      .as<std::string>(),
              "linear_slot_coordinate_weights_v1");
    EXPECT_FALSE(
        alignment["exception_run_contract"]["continuity_weight_rule"]
                 ["dense_source_weights_persisted"]
                     .as<bool>());
    EXPECT_EQ(alignment["exception_runs"][0]["left_source_slot"]
                  .as<Eigen::Index>(),
              1);
    EXPECT_EQ(alignment["exception_runs"][0]["right_source_slot"]
                  .as<Eigen::Index>(),
              3);
    EXPECT_TRUE(alignment["exception_runs"][0]["source_slots_available"]
                    .as<bool>());
    EXPECT_EQ(alignment["exception_runs"][1]["left_source_slot"]
                  .as<Eigen::Index>(),
              -1);
    EXPECT_FALSE(alignment["exception_runs"][1]["source_slots_available"]
                     .as<bool>());
    EXPECT_EQ(alignment["availability"]["timing_covariance"]
                  .as<std::string>(),
              "unavailable_input");
    EXPECT_FALSE(alignment["common_time"]);
    EXPECT_FALSE(alignment["masks"]);
}

TEST(sci_align_provenance,
     serializes_observation_resolved_processing_as_plan_not_execution) {
    auto state = make_alignment_state();
    resolve_processing_support(state);
    const auto scan_plan = make_scan_plan();

    const auto alignment =
        citlali::pipeline::compact_alignment_provenance_node(
            state, &scan_plan);
    const auto processing = alignment["processing_support_plan"];
    ASSERT_TRUE(processing);
    EXPECT_EQ(
        alignment["support"]["gap_policy_eligible_count_scope"]
            .as<std::string>(),
        "unique_unguarded_original_interface_slots_within_admitted_science_windows");
    EXPECT_FALSE(
        alignment["support"]["final_science_eligibility_available"]
            .as<bool>());
    EXPECT_TRUE(processing["observation_resolved"].as<bool>());
    EXPECT_EQ(processing["evidence_stage"].as<std::string>(),
              "observation_resolved_planned_processing");
    EXPECT_FALSE(processing["execution_realized"].as<bool>());
    EXPECT_EQ(processing["realization_semantics"].as<std::string>(),
              "plan_only_no_execution_outcome_claim");
    EXPECT_EQ(processing["signal_domain"].as<std::string>(), "xs");
    EXPECT_EQ(
        processing["gap_admission_contract"]["support_reference"]
            .as<std::string>(),
        "sci_align_scan_plan.records[stable_scan_id].compatibility_science");
    EXPECT_EQ(processing["gap_admission_contract"]["exact_quarter"]
                  .as<std::string>(),
              "admitted");
    EXPECT_EQ(
        processing["planned_action_support_reference"].as<std::string>(),
        "chunk_dispositions[].context_expanded_support");
    EXPECT_EQ(processing["chunk_disposition_encoding"]["representation"]
                  .as<std::string>(),
              "sparse_exceptions_v1");
    EXPECT_EQ(processing["chunk_disposition_encoding"]["key_order"]
                  .as<std::string>(),
              "compatibility_ordinal_then_roach_index");
    EXPECT_EQ(processing["chunk_disposition_encoding"]["absent_default"]
                        ["support"]
                            .as<std::string>(),
              "all_acquired_original_zero_detector_gap");
    EXPECT_EQ(processing["planned_occurrence_counts"]
                        ["continuity_surrogate_missing"]
                            .as<std::uint64_t>(),
              1U);
    EXPECT_EQ(processing["planned_occurrence_counts"]
                        ["unavailable_missing"]
                            .as<std::uint64_t>(),
              1U);
    ASSERT_EQ(processing["chunk_dispositions"].size(), 2U);
    const auto first = processing["chunk_dispositions"][0];
    EXPECT_EQ(first["stable_scan_id"].as<Eigen::Index>(), 0);
    EXPECT_EQ(first["interface_id"].as<std::string>(), "toltec0");
    EXPECT_EQ(first["context"]["start"].as<Eigen::Index>(), 0);
    EXPECT_EQ(first["context"]["stop"].as<Eigen::Index>(), 4);
    EXPECT_EQ(first["planned_actions"]["continuity_surrogate_missing"]
                   ["action"]
                       .as<std::string>(),
              "bounded_continuity_surrogate");
    const auto run =
        first["planned_actions"]["continuity_surrogate_missing"]
             ["runs"][0];
    EXPECT_EQ(run["start"].as<Eigen::Index>(), 2);
    EXPECT_EQ(run["stop"].as<Eigen::Index>(), 3);
    const auto weights =
        citlali::pipeline::alignment_exception_linear_source_weights(
            state.exceptions[0], 2);
    EXPECT_DOUBLE_EQ(weights.first, 0.5);
    EXPECT_DOUBLE_EQ(weights.second, 0.5);
}

TEST(sci_align_provenance,
     persists_only_nondefault_scan_interface_dispositions) {
    auto state = make_alignment_state();
    state.masks[1].setOnes();
    state.interfaces[1].native_row_count = 4;
    state.interfaces[1].accepted_row_count = 4;
    state.interfaces[1].first_global_slot = -1;
    state.interfaces[1].last_global_slot = 2;
    state.interfaces[1].leading_unavailable_count = 0;
    state.exceptions.pop_back();
    state.support.acquired_original_count = 7;
    state.support.timing_coordinate_valid_original_count = 7;
    state.support.synthesized_count = 0;
    state.support.unavailable_count = 1;
    state.support.acquired_original_cadence_weighted_support_sec = 3.5;
    const auto scan_plan = make_scan_plan();

    citlali::pipeline::finalize_alignment_gap_processing_plan(
        state, scan_plan, 0, citlali::config::TodType::xs);
    const auto alignment =
        citlali::pipeline::compact_alignment_provenance_node(
            state, &scan_plan);
    const auto processing = alignment["processing_support_plan"];

    ASSERT_EQ(state.chunk_dispositions.size(), 1U);
    ASSERT_EQ(processing["chunk_dispositions"].size(), 1U);
    EXPECT_EQ(processing["chunk_dispositions"][0]["interface_id"]
                  .as<std::string>(),
              "toltec0");
    EXPECT_EQ(alignment["support"]
                       ["gap_policy_eligible_original_interface_slot_count"]
                           .as<std::uint64_t>(),
              7U);
    EXPECT_EQ(processing["chunk_disposition_encoding"]["persisted_rows"]
                  .as<std::string>(),
              "nondefault_scan_interface_dispositions_only");
}

TEST(sci_align_provenance,
     completed_stage_records_compact_execution_only_after_completion) {
    auto state = make_alignment_state();
    resolve_processing_support(state);
    const auto scan_plan = make_scan_plan();

    const auto alignment =
        citlali::pipeline::compact_alignment_provenance_node(
            state, &scan_plan,
            citlali::pipeline::TimestreamOutputProvenanceStage::
                observation_execution_completed);
    const auto processing = alignment["processing_support_plan"];
    EXPECT_TRUE(processing["execution_realized"].as<bool>());
    EXPECT_EQ(processing["evidence_stage"].as<std::string>(),
              "observation_execution_completed_compact_result");
    EXPECT_EQ(
        processing["realization_semantics"].as<std::string>(),
        "required_processing_and_outputs_completed_compact_plan_result");
    EXPECT_FALSE(processing["processing_results"]);
}

TEST(sci_align_provenance,
     binds_selected_output_rows_to_admitted_raster_windows) {
    const auto engine = make_raster_output_provenance_engine();
    const auto node =
        citlali::pipeline::timestream_output_provenance_node(engine);

    const auto plan = node["realized"]["sci_align_scan_plan"];
    ASSERT_EQ(plan["records"].size(), 3U);
    EXPECT_TRUE(plan["records"][0]["physical_id"].IsNull());
    EXPECT_FALSE(plan["records"][2]["legacy_processing_admitted"]
                     .as<bool>());

    const auto raw = node["realized"]["raw_time_chunk"]
                         ["selected_output_windows"];
    ASSERT_EQ(raw.size(), 1U);
    EXPECT_EQ(raw[0]["stable_processing_record_id"].as<Eigen::Index>(), 1);
    EXPECT_EQ(raw[0]["compatibility_ordinal"].as<Eigen::Index>(), 1);
    EXPECT_EQ(raw[0]["output_row"].as<Eigen::Index>(), 0);
    EXPECT_EQ(raw[0]["output_interval"]["start"].as<Eigen::Index>(), 4);
    EXPECT_EQ(raw[0]["output_interval"]["stop"].as<Eigen::Index>(), 8);
    EXPECT_EQ(raw[0]["interval_convention"].as<std::string>(),
              "half_open_start_stop");
    EXPECT_EQ(raw[0]["interval_authority"].as<std::string>(),
              "context_outer");

    const auto processed = node["realized"]["processed_time_chunk"]
                               ["selected_output_windows"];
    ASSERT_EQ(processed.size(), 2U);
    EXPECT_EQ(processed[0]["stable_processing_record_id"]
                  .as<Eigen::Index>(),
              0);
    EXPECT_EQ(processed[0]["output_interval"]["start"]
                  .as<Eigen::Index>(),
              2);
    EXPECT_EQ(processed[0]["output_interval"]["stop"]
                  .as<Eigen::Index>(),
              3);
    EXPECT_EQ(processed[1]["stable_processing_record_id"]
                  .as<Eigen::Index>(),
              1);
    EXPECT_EQ(processed[1]["output_interval"]["start"]
                  .as<Eigen::Index>(),
              5);
    EXPECT_EQ(processed[1]["output_interval"]["stop"]
                  .as<Eigen::Index>(),
              7);
    EXPECT_EQ(processed[1]["interval_authority"].as<std::string>(),
              "science_inner");
    for (const auto &record : processed) {
        EXPECT_NE(record["stable_processing_record_id"].as<Eigen::Index>(),
                  2);
    }
}

TEST(sci_align_provenance,
     raw_nonouter_and_processed_outer_mode_both_use_science_windows) {
    auto engine = make_raster_output_provenance_engine();
    engine.typed_config.timestream.output.raw_time_chunk.mode =
        citlali::config::TodStreamOutputMode::mini;

    const auto node =
        citlali::pipeline::timestream_output_provenance_node(engine);
    const auto raw = node["realized"]["raw_time_chunk"]
                         ["selected_output_windows"][0];
    EXPECT_EQ(raw["output_interval"]["start"].as<Eigen::Index>(), 5);
    EXPECT_EQ(raw["output_interval"]["stop"].as<Eigen::Index>(), 7);
    EXPECT_EQ(raw["interval_authority"].as<std::string>(),
              "science_inner");
    EXPECT_EQ(node["realized"]["processed_time_chunk"]
                  ["selected_output_windows"][1]["interval_authority"]
                      .as<std::string>(),
              "science_inner");
}

TEST(sci_align_provenance,
     rejects_nonbijective_selected_output_window_rows) {
    auto duplicate = make_raster_output_provenance_engine();
    duplicate.tod_outputs.ptc_scan_to_output_scan << 0, 0;
    duplicate.tod_outputs.n_ptc_output_scans = 1;
    EXPECT_THROW(
        citlali::pipeline::timestream_output_provenance_node(duplicate),
        std::logic_error);

    auto missing = make_raster_output_provenance_engine();
    missing.tod_outputs.ptc_scan_to_output_scan << 0, -1;
    EXPECT_THROW(
        citlali::pipeline::timestream_output_provenance_node(missing),
        std::logic_error);
}

TEST(sci_align_provenance,
     rejects_missing_or_spurious_exception_source_slot_identities) {
    auto missing_sources = make_alignment_state();
    missing_sources.exceptions[0].left_source_slot = -1;
    missing_sources.exceptions[0].right_source_slot = -1;
    EXPECT_THROW(
        citlali::pipeline::validate_compact_alignment_provenance(
            missing_sources),
        std::logic_error);

    auto edge_sources = make_alignment_state();
    edge_sources.exceptions[1].left_source_slot = 0;
    edge_sources.exceptions[1].right_source_slot = 2;
    EXPECT_THROW(
        citlali::pipeline::validate_compact_alignment_provenance(
            edge_sources),
        std::logic_error);

    EXPECT_THROW(
        citlali::pipeline::alignment_exception_linear_source_weights(
            make_alignment_state().exceptions[1], 0),
        std::logic_error);
}

TEST(sci_align_provenance,
     optional_hwpr_absence_is_intensity_eligible_but_not_polarization_eligible) {
    auto state = make_alignment_state();
    state.hwpr =
        citlali::pipeline::bounded_nonpolarimetric_hwpr_summary(false);

    EXPECT_NO_THROW(
        citlali::pipeline::validate_compact_alignment_provenance(state));
    const auto node =
        citlali::pipeline::compact_alignment_provenance_node(state);
    EXPECT_FALSE(node["hwpr"]["producer_input_present"].as<bool>());
    EXPECT_FALSE(node["hwpr"]["aligned_angle_available"].as<bool>());
    EXPECT_TRUE(node["hwpr"]["intensity_eligible"].as<bool>());
    EXPECT_FALSE(node["hwpr"]["polarization_eligible"].as<bool>());
    EXPECT_EQ(node["hwpr"]["availability_reason"].as<std::string>(),
              "producer_input_absent_optional_nonfatal");
}

TEST(sci_align_provenance, rejects_stale_or_claimed_hwpr_availability) {
    auto unresolved = make_alignment_state();
    unresolved.hwpr = {};
    EXPECT_THROW(
        citlali::pipeline::validate_compact_alignment_provenance(unresolved),
        std::logic_error);

    auto claimed_angle = make_alignment_state();
    claimed_angle.hwpr.aligned_angle_available = true;
    EXPECT_THROW(
        citlali::pipeline::validate_compact_alignment_provenance(
            claimed_angle),
        std::logic_error);

    auto stale_indices = make_alignment_state();
    stale_indices.hwpr_start_index = 0;
    stale_indices.hwpr_end_index = 3;
    EXPECT_THROW(
        citlali::pipeline::validate_compact_alignment_provenance(
            stale_indices),
        std::logic_error);
}

TEST(sci_align_provenance,
     rejects_incomplete_or_incoherent_admitted_processing_plan) {
    const auto scan_plan = make_scan_plan();

    auto missing_interface = make_alignment_state();
    resolve_processing_support(missing_interface);
    missing_interface.chunk_dispositions.pop_back();
    missing_interface.processing_support
        .unavailable_processing_occurrence_count = 0;
    EXPECT_THROW(
        citlali::pipeline::validate_compact_alignment_provenance(
            missing_interface, &scan_plan),
        std::logic_error);

    auto mismatched_scan = make_alignment_state();
    resolve_processing_support(mismatched_scan);
    mismatched_scan.chunk_dispositions[0].context_stop = 3;
    EXPECT_THROW(
        citlali::pipeline::validate_compact_alignment_provenance(
            mismatched_scan, &scan_plan),
        std::logic_error);

    auto stale_summary = make_alignment_state();
    resolve_processing_support(stale_summary);
    ++stale_summary.processing_support
          .guarded_original_processing_occurrence_count;
    EXPECT_THROW(
        citlali::pipeline::validate_compact_alignment_provenance(
            stale_summary, &scan_plan),
        std::logic_error);

    auto stale_science_eligibility = make_alignment_state();
    resolve_processing_support(stale_science_eligibility);
    --stale_science_eligibility.support
          .gap_policy_eligible_original_count;
    EXPECT_THROW(
        citlali::pipeline::validate_compact_alignment_provenance(
            stale_science_eligibility, &scan_plan),
        std::logic_error);

    auto reordered = make_alignment_state();
    resolve_processing_support(reordered);
    std::swap(reordered.chunk_dispositions[0],
              reordered.chunk_dispositions[1]);
    EXPECT_THROW(
        citlali::pipeline::validate_compact_alignment_provenance(
            reordered, &scan_plan),
        std::logic_error);
}

TEST(sci_align_provenance,
     rejects_unbounded_or_policy_inconsistent_continuity_action) {
    auto state = make_alignment_state();
    resolve_processing_support(state);
    const auto scan_plan = make_scan_plan();
    auto &edge = state.chunk_dispositions[1];
    edge.synthesized_missing_runs = edge.unavailable_missing_runs;
    edge.unavailable_missing_runs.clear();
    state.processing_support.synthesized_processing_occurrence_count = 2;
    state.processing_support.unavailable_processing_occurrence_count = 0;

    EXPECT_THROW(
        citlali::pipeline::validate_compact_alignment_provenance(
            state, &scan_plan),
        std::logic_error);
}

TEST(sci_align_provenance, mock_without_alignment_remains_supported) {
    ProvenanceEngineWithoutAlignment engine;
    prepare_timestream_output_state(engine);

    const auto node =
        citlali::pipeline::timestream_output_provenance_node(engine);
    EXPECT_FALSE(node["realized"]["sci_align_alignment"]);
}

TEST(sci_align_provenance, uninitialized_alignment_is_explicitly_unrealized) {
    ProvenanceEngine engine;
    prepare_timestream_output_state(engine);

    const auto node =
        citlali::pipeline::timestream_output_provenance_node(engine);
    const auto alignment = node["realized"]["sci_align_alignment"];
    ASSERT_TRUE(alignment);
    EXPECT_FALSE(alignment["initialized"].as<bool>());
    EXPECT_EQ(alignment["availability"]["alignment"].as<std::string>(),
              "not_realized");
}

TEST(sci_align_provenance, rejects_stale_state_when_grid_is_uninitialized) {
    ProvenanceEngine engine;
    prepare_timestream_output_state(engine);
    engine.alignment.support.nominal_slot_count = 1;

    EXPECT_THROW(
        citlali::pipeline::timestream_output_provenance_node(engine),
        std::logic_error);
}

TEST(sci_align_provenance,
     rejects_stale_processing_plan_when_grid_is_uninitialized) {
    ProvenanceEngine engine;
    prepare_timestream_output_state(engine);
    engine.alignment.processing_support.signal_domain = "xs";

    EXPECT_THROW(
        citlali::pipeline::timestream_output_provenance_node(engine),
        std::logic_error);
}

TEST(sci_align_provenance, incomplete_state_fails_before_atomic_output) {
    ProvenanceEngine engine;
    prepare_timestream_output_state(engine);
    engine.alignment = make_alignment_state();
    engine.alignment.field_registry_version.clear();
    const auto output_dir = std::filesystem::path(::testing::TempDir()) /
                            "sci_align_incomplete_provenance";
    std::filesystem::remove_all(output_dir);
    std::filesystem::create_directories(output_dir);
    engine.output_paths.obsnum_dir_name = output_dir.string();
    const auto output_path =
        citlali::pipeline::timestream_output_provenance_path(output_dir);

    EXPECT_THROW(
        citlali::pipeline::write_timestream_output_provenance_file(engine),
        std::logic_error);
    EXPECT_FALSE(std::filesystem::exists(output_path));
    EXPECT_FALSE(std::filesystem::exists(output_path.string() + ".tmp"));
    std::filesystem::remove_all(output_dir);
}

TEST(sci_align_provenance,
     completion_atomically_rewrites_plan_and_failed_rewrite_stays_incomplete) {
    CompletedProvenanceEngine engine;
    auto &output = engine.typed_config.timestream.output;
    output.type = citlali::config::TodOutputType::both;
    output.raw_time_chunk.enabled = true;
    output.processed_time_chunk.enabled = true;
    engine.telescope.scan_plan = make_scan_plan();
    engine.telescope.scan_indices.resize(4, 1);
    engine.tod_outputs.rtc_scan_to_output_scan.resize(1);
    engine.tod_outputs.rtc_scan_to_output_scan << 0;
    engine.tod_outputs.ptc_scan_to_output_scan.resize(1);
    engine.tod_outputs.ptc_scan_to_output_scan << 0;
    engine.tod_outputs.n_rtc_output_scans = 1;
    engine.tod_outputs.n_ptc_output_scans = 1;
    engine.alignment = make_alignment_state();
    resolve_processing_support(engine.alignment);

    const auto output_dir = std::filesystem::path(::testing::TempDir()) /
                            "sci_align_completed_provenance";
    std::filesystem::remove_all(output_dir);
    std::filesystem::create_directories(output_dir);
    engine.output_paths.obsnum_dir_name = output_dir.string();
    const auto output_path =
        citlali::pipeline::timestream_output_provenance_path(output_dir);

    citlali::pipeline::write_timestream_output_provenance_file(engine);
    auto stored = YAML::LoadFile(output_path.string());
    EXPECT_FALSE(stored["realized"]["execution_completed"].as<bool>());
    EXPECT_EQ(stored["realized"]["evidence_stage"].as<std::string>(),
              "observation_setup_plan");
    EXPECT_FALSE(stored["realized"]["sci_align_alignment"]
                       ["processing_support_plan"]["execution_realized"]
                           .as<bool>());

    const auto completed_path =
        citlali::pipeline::publish_completed_timestream_output_provenance(
            engine);
    ASSERT_TRUE(completed_path.has_value());
    EXPECT_EQ(*completed_path, output_path);
    stored = YAML::LoadFile(output_path.string());
    EXPECT_TRUE(stored["realized"]["execution_completed"].as<bool>());
    EXPECT_EQ(stored["realized"]["evidence_stage"].as<std::string>(),
              "observation_execution_completed");
    EXPECT_TRUE(stored["realized"]["sci_align_alignment"]
                      ["processing_support_plan"]["execution_realized"]
                          .as<bool>());

    citlali::pipeline::write_timestream_output_provenance_file(engine);
    engine.alignment.field_registry_version.clear();
    EXPECT_THROW(
        citlali::pipeline::publish_completed_timestream_output_provenance(
            engine),
        std::logic_error);
    stored = YAML::LoadFile(output_path.string());
    EXPECT_FALSE(stored["realized"]["execution_completed"].as<bool>());
    EXPECT_FALSE(std::filesystem::exists(output_path.string() + ".tmp"));
    std::filesystem::remove_all(output_dir);
}

TEST(sci_align_provenance, atomic_write_failure_propagates_without_temp_file) {
    ProvenanceEngine engine;
    prepare_timestream_output_state(engine);
    engine.alignment = make_alignment_state();
    const auto missing_dir = std::filesystem::path(::testing::TempDir()) /
                             "sci_align_missing_provenance" / "nested";
    std::filesystem::remove_all(missing_dir.parent_path());
    engine.output_paths.obsnum_dir_name = missing_dir.string();
    const auto output_path =
        citlali::pipeline::timestream_output_provenance_path(missing_dir);

    EXPECT_THROW(
        citlali::pipeline::write_timestream_output_provenance_file(engine),
        std::ios_base::failure);
    EXPECT_FALSE(std::filesystem::exists(output_path));
    EXPECT_FALSE(std::filesystem::exists(output_path.string() + ".tmp"));
}

}  // namespace
