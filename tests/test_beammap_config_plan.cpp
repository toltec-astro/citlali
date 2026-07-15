#include <citlali_config/default_config.h>
#include <citlali/core/config/beammap_config_validation.h>
#include <citlali/core/pipeline/beammap_config_loading.h>
#include <citlali/core/pipeline/beammap_config_serialization.h>
#include <citlali/core/pipeline/beammap_provenance.h>
#include <citlali/core/pipeline/beammap_provenance_lifecycle.h>
#include <citlali/core/pipeline/config_diagnostics_state.h>
#include <citlali/core/pipeline/beammap_execution_plan.h>
#include <citlali/core/pipeline/mapmaking_provenance_lifecycle.h>
#include <citlali/core/pipeline/post_processing_provenance_lifecycle.h>

#include <gtest/gtest.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>
#include <tula/config/yamlconfig.h>

#include <filesystem>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

void ensure_citlali_test_logger() {
    if (!spdlog::get("citlali_logger")) {
        auto sink = std::make_shared<spdlog::sinks::null_sink_mt>();
        spdlog::register_logger(
            std::make_shared<spdlog::logger>("citlali_logger", sink));
    }
}

void collect_yaml_leaf_paths(const YAML::Node &node, const std::string &prefix,
                             std::set<std::string> &paths) {
    if (node.IsMap()) {
        for (const auto &item : node) {
            const auto key = item.first.as<std::string>();
            collect_yaml_leaf_paths(
                item.second, prefix.empty() ? key : prefix + "." + key,
                paths);
        }
        return;
    }
    if (node.IsSequence()) {
        for (std::size_t index = 0; index < node.size(); ++index) {
            collect_yaml_leaf_paths(
                node[index], prefix + "." + std::to_string(index), paths);
        }
        return;
    }
    paths.insert(prefix);
}

citlali::config::BeammapConfig complete_beammap_request() {
    citlali::config::BeammapConfig config;
    config.iteration.max_iterations = 4;
    config.iteration.tolerance = 0.25;
    config.phase_strategy.locator_iter = 2;
    config.phase_strategy.measurement_start_iter = 1;
    config.detector_weighting_mode =
        citlali::config::BeammapDetectorWeightingMode::ptc_after_iter0;
    config.priors.enabled = true;
    config.priors.filepath = "priors.ecsv";
    config.priors.max_d2 = 12.0;
    config.priors.score_lambda = 3.0;
    config.priors.alignment_scope =
        citlali::config::BeammapPriorAlignmentScope::common;
    config.priors.alignment_common_support =
        citlali::config::BeammapPriorAlignmentSupport::overlap_box;
    config.split_fits_by_flag.flag_values = {5, 1, 5};
    config.flagging.array_lower_fwhm_arcsec = {1.0, 2.0, 3.0};
    config.flagging.array_upper_fwhm_arcsec = {4.0, 5.0, 6.0};
    config.flagging.array_lower_sig2noise = {7.0, 8.0, 9.0};
    config.flagging.array_upper_sig2noise = {10.0, 11.0, 12.0};
    config.flagging.array_max_dist_arcsec = {13.0, 14.0, 15.0};
    config.flagging.array_network_robust_z = {16.0, 17.0, 18.0};
    config.flagging.sens_factors = {0.5, 2.0};
    config.flagging.sens_psd_limits_hz = {1.0, 4.0};
    return config;
}

citlali::config::BeammapConfig valid_complete_beammap_config() {
    auto config = complete_beammap_request();
    config.phase_strategy.locator_iter = 0;
    config.phase_strategy.measurement_start_iter = 1;
    return config;
}

citlali::config::BeammapPhotometryConfig complete_beammap_photometry() {
    citlali::config::BeammapPhotometryConfig photometry;
    photometry.fluxes = {
        {"a1100", 1000.0, 10.0},
        {"a1400", 900.0, 9.0},
        {"a2000", 800.0, 8.0},
    };
    return photometry;
}

citlali::pipeline::MapmakingExecutionPlan completed_beammap_mapmaking_plan(
    std::size_t map_count = 5) {
    citlali::config::MapmakingConfig request;
    request.enabled = true;
    request.grouping = citlali::config::MapGrouping::detector;
    citlali::pipeline::MapmakingExecutionPlan plan;
    plan.reset_from_request(
        request, citlali::config::ReductionType::beammap);
    plan.begin_iteration();
    plan.begin_observation(0, "148670", map_count, 1.0e-5, map_count);
    citlali::pipeline::complete_mapmaking_observation(plan);
    citlali::pipeline::record_mapmaking_run_completed(plan);
    return plan;
}

citlali::pipeline::PostProcessingExecutionPlan
completed_beammap_post_processing_plan(
    const citlali::pipeline::MapmakingExecutionPlan &mapmaking,
    std::size_t fit_context_count, std::size_t map_count = 5) {
    citlali::config::PostProcessingConfig request;
    request.map_filtering.enabled = false;
    request.source_finding.enabled = false;
    citlali::pipeline::PostProcessingExecutionPlan plan;
    plan.reset_from_request(
        request, citlali::config::ReductionType::beammap, true, false);
    plan.begin_iteration();
    for (std::size_t index = 0; index < fit_context_count; ++index) {
        citlali::pipeline::record_post_processing_beammap_fits_completed(
            plan, map_count, map_count - 1);
    }
    citlali::pipeline::record_post_processing_run_completed(
        plan, mapmaking);
    return plan;
}

citlali::pipeline::BeammapExecutionPlan completed_beammap_plan() {
    auto request = valid_complete_beammap_config();
    request.iteration.max_iterations = 3;
    citlali::pipeline::BeammapExecutionPlan plan;
    plan.reset_from_request(request, {}, true);
    plan.begin_iteration();
    plan.begin_observation(
        0, "148670", complete_beammap_photometry(), 5, 5, 198);

    plan.begin_internal_iteration(
        0, citlali::pipeline::BeammapIterationPhase::locator, 5);
    plan.record_source_aware_rtc_rerun(false);
    plan.record_mapmaking_pass_completed();
    plan.record_fitting_completed();
    plan.complete_internal_iteration(
        0, citlali::pipeline::BeammapTerminationReason::none);

    plan.begin_internal_iteration(
        1, citlali::pipeline::BeammapIterationPhase::measurement_start,
        5);
    plan.record_source_aware_rtc_rerun(true);
    plan.record_mapmaking_pass_completed();
    plan.record_mapmaking_pass_completed();
    plan.record_fitting_completed();
    plan.complete_internal_iteration(
        2, citlali::pipeline::BeammapTerminationReason::none);

    plan.begin_internal_iteration(
        2, citlali::pipeline::BeammapIterationPhase::measurement, 3);
    plan.record_source_aware_rtc_rerun(false);
    plan.record_mapmaking_pass_completed();
    plan.record_fitting_completed();
    plan.complete_internal_iteration(
        5,
        citlali::pipeline::BeammapTerminationReason::maximum_iterations);
    plan.complete_observation();
    return plan;
}

TEST(BeammapConfigSerialization, CoversFrozenSeventyFourLeafSurface) {
    auto config = complete_beammap_request();
    config.split_fits_by_flag.flag_values = {5, 1};
    const auto node = citlali::pipeline::beammap_config_node(config);
    std::set<std::string> paths;
    collect_yaml_leaf_paths(node, "beammap", paths);

    EXPECT_EQ(paths.size(), 74U);
    EXPECT_NE(paths.find("beammap.flagging.array_lower_fwhm_arcsec.2"),
              paths.end());
    EXPECT_NE(paths.find("beammap.flagging.sens_factors.1"), paths.end());
    EXPECT_NE(paths.find("beammap.sens_psd_limits_Hz.1"), paths.end());
    EXPECT_EQ(node["detector_weighting"]["mode"].as<std::string>(),
              "ptc_after_iter0");
    EXPECT_EQ(node["priors"]["alignment_scope"].as<std::string>(),
              "common");
    EXPECT_EQ(node["priors"]["alignment_common_support"].as<std::string>(),
              "overlap_box");
}

TEST(BeammapExecutionPlan, PreservesRequestAndRecordsEffectivePolicy) {
    const auto request = complete_beammap_request();
    citlali::pipeline::BeammapRequestPresence presence;
    presence.split_flag_values = true;
    citlali::pipeline::BeammapExecutionPlan plan;

    plan.reset_from_request(request, presence, true);

    EXPECT_TRUE(plan.initialized());
    EXPECT_EQ(plan.requested().phase_strategy.locator_iter, 2);
    EXPECT_EQ(plan.requested().split_fits_by_flag.flag_values,
              (std::vector<int>{5, 1, 5}));
    EXPECT_EQ(plan.effective().phase_strategy.locator_iter, 0);
    EXPECT_EQ(plan.effective().phase_strategy.measurement_start_iter, 1);
    EXPECT_EQ(plan.effective().split_fits_by_flag.flag_values,
              (std::vector<int>{1, 5}));
    EXPECT_DOUBLE_EQ(plan.effective().priors.max_d2_iter0, 12.0);
    EXPECT_DOUBLE_EQ(plan.effective().priors.max_d2_after_iter0, 12.0);
    EXPECT_DOUBLE_EQ(plan.effective().priors.score_lambda_iter0, 3.0);
    EXPECT_DOUBLE_EQ(plan.effective().priors.score_lambda_after_iter0, 3.0);

    const auto &resolution = plan.resolution();
    EXPECT_TRUE(resolution.locator_iter_forced_zero);
    EXPECT_FALSE(resolution.measurement_start_iter_adjusted);
    EXPECT_TRUE(resolution.measurement_pass_available);
    EXPECT_TRUE(resolution.convergence_check_available);
    EXPECT_TRUE(resolution.convergence_active);
    EXPECT_TRUE(resolution.split_flag_values_sorted);
    EXPECT_TRUE(resolution.split_flag_values_deduplicated);
    EXPECT_EQ(resolution.requested_split_flag_count, 3U);
    EXPECT_EQ(resolution.effective_split_flag_count, 2U);
}

TEST(BeammapExecutionPlan, ResolvesMissingValuesWithoutMutatingRequest) {
    auto request = complete_beammap_request();
    request.iteration.max_iterations = 7;
    request.phase_strategy.measurement_start_iter = 0;
    request.priors.filepath = "null";
    request.split_fits_by_flag.flag_values = {9};
    citlali::pipeline::BeammapExecutionPlan plan;

    plan.reset_from_request(request, {}, false);

    EXPECT_EQ(plan.requested().iteration.max_iterations, 7);
    EXPECT_TRUE(plan.requested().priors.enabled);
    EXPECT_EQ(plan.requested().split_fits_by_flag.flag_values,
              (std::vector<int>{9}));
    EXPECT_EQ(plan.effective().iteration.max_iterations, 1);
    EXPECT_EQ(plan.effective().phase_strategy.measurement_start_iter, 1);
    EXPECT_FALSE(plan.effective().priors.enabled);
    EXPECT_EQ(plan.effective().split_fits_by_flag.flag_values,
              (std::vector<int>{0, 1}));

    const auto &resolution = plan.resolution();
    EXPECT_TRUE(resolution.max_iterations_forced_without_mapmaking);
    EXPECT_TRUE(resolution.measurement_start_iter_adjusted);
    EXPECT_TRUE(resolution.priors_disabled_by_missing_path);
    EXPECT_TRUE(resolution.split_flag_values_defaulted);
    EXPECT_EQ(resolution.requested_split_flag_count, 0U);
    EXPECT_FALSE(resolution.measurement_pass_available);
    EXPECT_FALSE(resolution.convergence_active);
}

TEST(BeammapExecutionPlan, ResetClearsPriorResolutionState) {
    auto first = complete_beammap_request();
    first.priors.filepath = "null";
    citlali::pipeline::BeammapExecutionPlan plan;
    plan.reset_from_request(first, {}, false);
    ASSERT_TRUE(plan.resolution().priors_disabled_by_missing_path);

    auto second = complete_beammap_request();
    citlali::pipeline::BeammapRequestPresence presence;
    presence.max_d2_iter0 = true;
    presence.max_d2_after_iter0 = true;
    presence.score_lambda_iter0 = true;
    presence.score_lambda_after_iter0 = true;
    presence.split_flag_values = true;
    plan.reset_from_request(second, presence, true);

    EXPECT_FALSE(plan.resolution().priors_disabled_by_missing_path);
    EXPECT_FALSE(plan.resolution().max_d2_iter0_inherited);
    EXPECT_FALSE(plan.resolution().split_flag_values_defaulted);
    EXPECT_EQ(plan.resolution().requested_split_flag_count, 3U);
}

TEST(BeammapExecutionPlan, RecordsCompletedObservationLifecycle) {
    auto plan = completed_beammap_plan();
    const auto mapmaking = completed_beammap_mapmaking_plan();
    const auto post_processing =
        completed_beammap_post_processing_plan(mapmaking, 3);

    citlali::pipeline::record_beammap_run_completed(
        plan, mapmaking, post_processing);

    ASSERT_EQ(plan.observations().size(), 1U);
    const auto &observation = plan.observations().front();
    ASSERT_EQ(observation.iterations.size(), 3U);
    EXPECT_EQ(observation.detector_count, 5U);
    EXPECT_EQ(observation.scan_count, 198U);
    EXPECT_EQ(observation.terminal_iteration, 2U);
    EXPECT_EQ(
        observation.termination_reason,
        citlali::pipeline::BeammapTerminationReason::maximum_iterations);
    EXPECT_EQ(observation.iterations[1].mapmaking_pass_count, 2U);
    EXPECT_EQ(observation.iterations[1].newly_converged_map_count, 2U);
    EXPECT_EQ(observation.iterations[2].newly_converged_map_count, 3U);
    EXPECT_TRUE(plan.realized().reduction_completed);
    EXPECT_TRUE(plan.realized().beammap_executed);
    EXPECT_EQ(plan.realized().completed_observation_count, 1U);
    EXPECT_EQ(plan.realized().completed_iteration_count, 3U);
}

TEST(BeammapExecutionPlan, RequiresEveryInternalStageBeforeCompletion) {
    auto request = valid_complete_beammap_config();
    request.iteration.max_iterations = 1;
    citlali::pipeline::BeammapExecutionPlan plan;
    plan.reset_from_request(request, {}, true);
    plan.begin_iteration();
    plan.begin_observation(
        0, "148670", complete_beammap_photometry(), 5, 5, 198);
    plan.begin_internal_iteration(
        0, citlali::pipeline::BeammapIterationPhase::locator, 5);
    plan.record_mapmaking_pass_completed();
    plan.record_fitting_completed();

    EXPECT_THROW(
        plan.complete_internal_iteration(
            0,
            citlali::pipeline::BeammapTerminationReason::maximum_iterations),
        std::logic_error);
}

TEST(BeammapExecutionPlan, RejectsFitContextCardinalityMismatch) {
    auto plan = completed_beammap_plan();
    const auto mapmaking = completed_beammap_mapmaking_plan();
    const auto post_processing =
        completed_beammap_post_processing_plan(mapmaking, 2);

    EXPECT_THROW(
        citlali::pipeline::record_beammap_run_completed(
            plan, mapmaking, post_processing),
        std::logic_error);
}

TEST(BeammapExecutionPlan, RecordsEarlyConvergenceTermination) {
    auto request = valid_complete_beammap_config();
    request.iteration.max_iterations = 3;
    citlali::pipeline::BeammapExecutionPlan plan;
    plan.reset_from_request(request, {}, true);
    plan.begin_iteration();
    plan.begin_observation(
        0, "148670", complete_beammap_photometry(), 5, 5, 198);
    plan.begin_internal_iteration(
        0, citlali::pipeline::BeammapIterationPhase::locator, 5);
    plan.record_source_aware_rtc_rerun(false);
    plan.record_mapmaking_pass_completed();
    plan.record_fitting_completed();
    plan.complete_internal_iteration(
        0, citlali::pipeline::BeammapTerminationReason::none);
    plan.begin_internal_iteration(
        1, citlali::pipeline::BeammapIterationPhase::measurement_start,
        5);
    plan.record_source_aware_rtc_rerun(true);
    plan.record_mapmaking_pass_completed();
    plan.record_fitting_completed();
    plan.complete_internal_iteration(
        5,
        citlali::pipeline::BeammapTerminationReason::all_maps_converged);
    plan.complete_observation();
    const auto mapmaking = completed_beammap_mapmaking_plan();
    const auto post_processing =
        completed_beammap_post_processing_plan(mapmaking, 2);

    citlali::pipeline::record_beammap_run_completed(
        plan, mapmaking, post_processing);

    EXPECT_EQ(
        plan.observations().front().termination_reason,
        citlali::pipeline::BeammapTerminationReason::all_maps_converged);
    EXPECT_EQ(plan.realized().completed_iteration_count, 2U);
}

TEST(BeammapExecutionPlan, RecordsDisabledExecutionWithoutFakeProducts) {
    citlali::config::MapmakingConfig mapmaking_request;
    mapmaking_request.enabled = false;
    citlali::pipeline::MapmakingExecutionPlan mapmaking;
    mapmaking.reset_from_request(
        mapmaking_request, citlali::config::ReductionType::beammap);
    mapmaking.begin_iteration();
    citlali::pipeline::record_mapmaking_run_completed(mapmaking);

    citlali::config::PostProcessingConfig post_request;
    citlali::pipeline::PostProcessingExecutionPlan post_processing;
    post_processing.reset_from_request(
        post_request, citlali::config::ReductionType::beammap,
        false, false);
    post_processing.begin_iteration();
    citlali::pipeline::record_post_processing_run_completed(
        post_processing, mapmaking);

    citlali::pipeline::BeammapExecutionPlan plan;
    plan.reset_from_request(valid_complete_beammap_config(), {}, false);
    plan.begin_iteration();
    citlali::pipeline::record_beammap_run_completed(
        plan, mapmaking, post_processing);

    EXPECT_TRUE(plan.observations().empty());
    EXPECT_TRUE(plan.realized().reduction_completed);
    EXPECT_FALSE(plan.realized().beammap_executed);
    EXPECT_EQ(plan.realized().completed_observation_count, 0U);
    EXPECT_EQ(plan.realized().completed_iteration_count, 0U);
}

TEST(BeammapExecutionPlan, RequiresExactlyOneEnabledDetectorTodWrite) {
    auto request = valid_complete_beammap_config();
    request.iteration.max_iterations = 1;
    request.detector_tod_output.enabled = true;

    citlali::pipeline::BeammapExecutionPlan missing;
    missing.reset_from_request(request, {}, true);
    missing.begin_iteration();
    missing.begin_observation(
        0, "148670", complete_beammap_photometry(), 5, 5, 198);
    missing.begin_internal_iteration(
        0, citlali::pipeline::BeammapIterationPhase::locator, 5);
    missing.record_source_aware_rtc_rerun(false);
    missing.record_mapmaking_pass_completed();
    missing.record_fitting_completed();
    missing.complete_internal_iteration(
        0,
        citlali::pipeline::BeammapTerminationReason::maximum_iterations);
    EXPECT_THROW(missing.complete_observation(), std::logic_error);

    citlali::pipeline::BeammapExecutionPlan complete;
    complete.reset_from_request(request, {}, true);
    complete.begin_iteration();
    complete.begin_observation(
        0, "148670", complete_beammap_photometry(), 5, 5, 198);
    complete.begin_internal_iteration(
        0, citlali::pipeline::BeammapIterationPhase::locator, 5);
    complete.record_source_aware_rtc_rerun(false);
    complete.record_mapmaking_pass_completed();
    complete.record_fitting_completed();
    complete.record_detector_tod_written(0, 5, 8, 1200);
    EXPECT_THROW(
        complete.record_detector_tod_written(0, 5, 8, 1200),
        std::logic_error);
    complete.complete_internal_iteration(
        0,
        citlali::pipeline::BeammapTerminationReason::maximum_iterations);
    complete.complete_observation();

    const auto &detector_tod =
        complete.observations().front().detector_tod;
    EXPECT_TRUE(detector_tod.required);
    EXPECT_EQ(detector_tod.completed_write_count, 1U);
    EXPECT_EQ(detector_tod.output_iteration, 0U);
    EXPECT_EQ(detector_tod.detector_count, 5U);
    EXPECT_EQ(detector_tod.slot_count, 8U);
    EXPECT_EQ(detector_tod.maximum_sample_count, 1200U);
}

TEST(BeammapExecutionPlan, RepeatedIterationResetClearsRealizedState) {
    auto plan = completed_beammap_plan();

    plan.begin_iteration();

    EXPECT_TRUE(plan.observations().empty());
    EXPECT_FALSE(plan.realized().reduction_completed);
    EXPECT_EQ(plan.realized().completed_observation_count, 0U);
    EXPECT_EQ(plan.realized().completed_iteration_count, 0U);
}

TEST(BeammapProvenance, SerializesRequestedEffectiveAndRealizedState) {
    auto plan = completed_beammap_plan();
    const auto mapmaking = completed_beammap_mapmaking_plan();
    const auto post_processing =
        completed_beammap_post_processing_plan(mapmaking, 3);
    citlali::pipeline::record_beammap_run_completed(
        plan, mapmaking, post_processing);

    const auto node = citlali::pipeline::beammap_provenance_node(plan);

    EXPECT_EQ(node["schema_version"].as<std::string>(),
              "citlali-beammap-provenance-v2");
    EXPECT_EQ(node["requested"]["iter_max"].as<int>(), 3);
    EXPECT_EQ(node["effective"]["config"]["iter_max"].as<int>(), 3);
    EXPECT_TRUE(node["effective"]["resolution"]["mapmaking_enabled"]
                    .as<bool>());
    ASSERT_EQ(node["observations"].size(), 1U);
    EXPECT_EQ(
        node["observations"][0]["source_identity_authority"]
            .as<std::string>(),
        "telescope_data");
    EXPECT_EQ(
        node["observations"][0]["photometry"]
            ["required_flux_policy"].as<std::string>(),
        "fail_reduction");
    EXPECT_EQ(
        node["observations"][0]["photometry"]
            ["calibrator_flux_authority"].as<std::string>(),
        "tolproj");
    EXPECT_EQ(
        node["observations"][0]["photometry"]
            ["flux_input_path"].as<std::string>(),
        "beammap_source.fluxes");
    ASSERT_EQ(
        node["observations"][0]["photometry"]["fluxes"].size(), 3U);
    EXPECT_EQ(
        node["observations"][0]["photometry"]["fluxes"][0]
            ["array_name"].as<std::string>(),
        "a1100");
    ASSERT_EQ(node["observations"][0]["iterations"].size(), 3U);
    EXPECT_FALSE(node["observations"][0]["detector_tod"]["required"]
                     .as<bool>());
    EXPECT_EQ(node["observations"][0]["detector_tod"]
                  ["completed_write_count"]
                      .as<std::size_t>(),
              0U);
    EXPECT_EQ(node["observations"][0]["iterations"][1]
                  ["mapmaking_pass_count"]
                      .as<std::size_t>(),
              2U);
    EXPECT_EQ(node["realized"]["completed_iteration_count"]
                  .as<std::size_t>(),
              3U);
}

TEST(BeammapProvenance, RequiresCompletionAndPropagatesWriteFailure) {
    auto incomplete = completed_beammap_plan();
    const auto directory = std::filesystem::temp_directory_path() /
        "citlali_beammap_provenance_test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);

    EXPECT_THROW(
        citlali::pipeline::write_beammap_provenance_file(
            directory, incomplete),
        std::logic_error);

    const auto mapmaking = completed_beammap_mapmaking_plan();
    const auto post_processing =
        completed_beammap_post_processing_plan(mapmaking, 3);
    citlali::pipeline::record_beammap_run_completed(
        incomplete, mapmaking, post_processing);
    EXPECT_NO_THROW(
        citlali::pipeline::write_beammap_provenance_file(
            directory, incomplete));
    EXPECT_TRUE(std::filesystem::exists(
        directory / citlali::pipeline::beammap_provenance_filename));
    EXPECT_THROW(
        citlali::pipeline::write_beammap_provenance_file(
            directory / "missing", incomplete),
        std::ios_base::failure);
    EXPECT_FALSE(std::filesystem::exists(
        directory / "missing" /
        (std::string{citlali::pipeline::beammap_provenance_filename} +
         ".tmp")));
    std::filesystem::remove_all(directory);
}

TEST(BeammapExecutionPlan, ReadsRawRequestBeforeResolvingEffectivePolicy) {
    ensure_citlali_test_logger();
    auto root = YAML::Load(citlali::citlali_default_config_content);
    auto beammap = root["beammap"];
    beammap["iter_max"] = 6;
    beammap["phase_strategy"]["locator_iter"] = 2;
    beammap["phase_strategy"]["measurement_start_iter"] = 1;
    beammap["split_fits_by_flag"]["flag_values"] =
        std::vector<int>{5, 1, 5};
    beammap["priors"]["enabled"] = true;
    beammap["priors"]["filepath"] = "null";
    beammap["priors"]["max_d2"] = 12.0;
    beammap["priors"]["score_lambda"] = 3.0;
    beammap["priors"]["max_d2_iter0"] = 7.0;
    beammap["priors"]["max_d2_after_iter0"] = 8.0;
    beammap["priors"]["score_lambda_iter0"] = 9.0;
    beammap["priors"]["score_lambda_after_iter0"] = 10.0;
    auto yaml_config =
        tula::config::YamlConfig::from_str(YAML::Dump(root));
    citlali::pipeline::ConfigDiagnosticsState diagnostics;

    const auto read = citlali::pipeline::read_beammap_request_config(
        yaml_config, diagnostics, 3);

    ASSERT_FALSE(diagnostics.has_errors());
    EXPECT_EQ(read.request.iteration.max_iterations, 6);
    EXPECT_EQ(read.request.phase_strategy.locator_iter, 2);
    EXPECT_EQ(read.request.phase_strategy.measurement_start_iter, 1);
    EXPECT_TRUE(read.request.priors.enabled);
    EXPECT_EQ(read.request.split_fits_by_flag.flag_values,
              (std::vector<int>{5, 1, 5}));
    EXPECT_TRUE(read.presence.max_d2_iter0);
    EXPECT_TRUE(read.presence.max_d2_after_iter0);
    EXPECT_TRUE(read.presence.score_lambda_iter0);
    EXPECT_TRUE(read.presence.score_lambda_after_iter0);
    EXPECT_TRUE(read.presence.split_flag_values);

    citlali::pipeline::BeammapExecutionPlan plan;
    plan.reset_from_request(read.request, read.presence, true);
    EXPECT_EQ(plan.requested().phase_strategy.locator_iter, 2);
    EXPECT_TRUE(plan.requested().priors.enabled);
    EXPECT_EQ(plan.effective().phase_strategy.locator_iter, 0);
    EXPECT_DOUBLE_EQ(plan.effective().priors.max_d2_iter0, 7.0);
    EXPECT_DOUBLE_EQ(plan.effective().priors.max_d2_after_iter0, 8.0);
    EXPECT_DOUBLE_EQ(plan.effective().priors.score_lambda_iter0, 9.0);
    EXPECT_DOUBLE_EQ(plan.effective().priors.score_lambda_after_iter0, 10.0);
    EXPECT_FALSE(plan.effective().priors.enabled);
    EXPECT_EQ(plan.effective().split_fits_by_flag.flag_values,
              (std::vector<int>{1, 5}));

    citlali::config::BeammapConfig compatibility;
    citlali::pipeline::install_beammap_effective_compatibility_config(
        plan, compatibility);
    EXPECT_EQ(compatibility.phase_strategy.locator_iter, 0);
    EXPECT_FALSE(compatibility.priors.enabled);
    EXPECT_EQ(compatibility.split_fits_by_flag.flag_values,
              (std::vector<int>{1, 5}));
}

TEST(BeammapConfigValidation, AcceptsDefaultAndCompleteShapes) {
    citlali::config::ValidationReport default_report;
    citlali::config::validate(citlali::config::BeammapConfig{}, default_report);
    EXPECT_TRUE(default_report.ok()) << default_report.format_for_cli();

    citlali::config::ValidationReport complete_report;
    citlali::config::validate(valid_complete_beammap_config(), complete_report);
    EXPECT_TRUE(complete_report.ok()) << complete_report.format_for_cli();
}

TEST(BeammapConfigValidation, RejectsNonFiniteAndInconsistentVectors) {
    auto config = valid_complete_beammap_config();
    config.iteration.tolerance = std::numeric_limits<double>::infinity();
    config.priors.min_snr = std::numeric_limits<double>::quiet_NaN();
    config.flagging.array_upper_fwhm_arcsec = {1.0, 2.0};
    config.flagging.array_max_dist_arcsec[1] =
        std::numeric_limits<double>::infinity();
    config.flagging.sens_factors = {1.0};
    config.flagging.sens_psd_limits_hz[0] =
        std::numeric_limits<double>::quiet_NaN();
    citlali::config::ValidationReport report;

    citlali::config::validate(config, report);

    EXPECT_EQ(report.error_count(), 6U) << report.format_for_cli();
    EXPECT_NE(report.format_for_cli().find("beammap.iter_tolerance"),
              std::string::npos);
    EXPECT_NE(report.format_for_cli().find(
                  "beammap.flagging.array_max_dist_arcsec.1"),
              std::string::npos);
    EXPECT_NE(report.format_for_cli().find("beammap.sens_psd_limits_Hz.0"),
              std::string::npos);
}

TEST(BeammapConfigValidation, RequiresTodSubdirectoryWhenEnabled) {
    citlali::config::BeammapConfig config;
    config.detector_tod_output.enabled = true;
    config.detector_tod_output.subdir_name.clear();
    citlali::config::ValidationReport report;

    citlali::config::validate(config, report);

    EXPECT_EQ(report.error_count(), 1U) << report.format_for_cli();
    EXPECT_NE(report.format_for_cli().find(
                  "beammap.detector_tod_output.subdir_name"),
              std::string::npos);
}

TEST(BeammapConfigValidation, RejectsPartiallyPopulatedArrayVectors) {
    auto config = valid_complete_beammap_config();
    config.flagging.array_network_robust_z.clear();
    citlali::config::ValidationReport report;

    citlali::config::validate(config, report);

    EXPECT_EQ(report.error_count(), 1U) << report.format_for_cli();
    EXPECT_NE(report.format_for_cli().find(
                  "beammap.flagging.array_network_robust_z"),
              std::string::npos);
}

}  // namespace
