#include <citlali/core/config/beammap_config_validation.h>
#include <citlali/core/pipeline/beammap_config_serialization.h>
#include <citlali/core/pipeline/beammap_execution_plan.h>

#include <gtest/gtest.h>

#include <limits>
#include <set>
#include <string>
#include <vector>

namespace {

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
