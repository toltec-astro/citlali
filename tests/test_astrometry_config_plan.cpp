#include <citlali/core/pipeline/astrometry_provenance.h>
#include <citlali/core/pipeline/telescope_pointing_operations.h>

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

citlali::config::AstrometryConfig astrometry_request(
    std::vector<double> az, std::vector<double> alt,
    std::vector<double> mjd = {0.0, 0.0}) {
    citlali::config::AstrometryConfig config;
    config.pointing_offsets.enabled = true;
    config.pointing_offsets.az_arcsec = std::move(az);
    config.pointing_offsets.alt_arcsec = std::move(alt);
    config.pointing_offsets.modified_julian_date = std::move(mjd);
    return config;
}

struct FakeTelescope {
    std::unordered_map<std::string, std::vector<double>> tel_data;
    void calc_tan_pointing() {}
};

struct FakeAstrometryEngine {
    citlali::pipeline::AstrometryExecutionPlan astrometry_plan;
    FakeTelescope telescope;
};

struct FakeAstrometryTodProc {
    FakeAstrometryEngine state;
    std::size_t interpolation_count = 0;

    FakeAstrometryEngine &engine() { return state; }
    void interp_pointing() { ++interpolation_count; }
};

struct NullLogger {
    template <class... Args>
    void info(Args &&...) const {}
};

TEST(AstrometryExecutionPlan, ResolvesApplicationModesWithoutChangingValues) {
    const auto constant = astrometry_request({1.0}, {2.0});
    const auto span = astrometry_request({1.0, 3.0}, {2.0, 4.0});
    const auto explicit_mjd = astrometry_request(
        {1.0, 3.0}, {2.0, 4.0}, {60000.0, 60001.0});

    EXPECT_EQ(citlali::pipeline::resolve_astrometry_application(constant)
                  .application_mode,
              citlali::pipeline::AstrometryApplicationMode::constant);
    EXPECT_EQ(citlali::pipeline::resolve_astrometry_application(span)
                  .application_mode,
              citlali::pipeline::AstrometryApplicationMode::
                  observation_span_linear);
    const auto explicit_resolution =
        citlali::pipeline::resolve_astrometry_application(explicit_mjd);
    EXPECT_EQ(explicit_resolution.application_mode,
              citlali::pipeline::AstrometryApplicationMode::
                  explicit_mjd_linear);
    EXPECT_TRUE(explicit_resolution.explicit_mjd_support);
}

TEST(AstrometryExecutionPlan, RecordsRepeatedObservationLifecycle) {
    citlali::pipeline::AstrometryExecutionPlan plan;
    const auto first = astrometry_request({1.0}, {2.0});
    const auto second = astrometry_request(
        {3.0, 5.0}, {4.0, 6.0}, {60000.0, 60001.0});
    plan.reset(2);

    citlali::pipeline::record_astrometry_request(plan, 0, 152389, first);
    citlali::pipeline::record_astrometry_installed(plan);
    citlali::pipeline::record_astrometry_applied(plan, 101);
    citlali::pipeline::record_astrometry_request(plan, 1, 152390, second);
    citlali::pipeline::record_astrometry_installed(plan);
    citlali::pipeline::record_astrometry_applied(plan, 202);
    citlali::pipeline::record_astrometry_request(plan, 0, 152389, first);
    citlali::pipeline::record_astrometry_installed(plan);
    citlali::pipeline::record_astrometry_applied(plan, 101);
    citlali::pipeline::record_astrometry_request(plan, 1, 152390, second);
    citlali::pipeline::record_astrometry_installed(plan);
    citlali::pipeline::record_astrometry_applied(plan, 202);
    citlali::pipeline::record_astrometry_reduction_completed(plan);

    EXPECT_TRUE(plan.reduction_completed);
    ASSERT_EQ(plan.observations.size(), 2U);
    EXPECT_EQ(plan.observations[0].realized.installation_count, 2U);
    EXPECT_EQ(plan.observations[0].realized.application_count, 2U);
    EXPECT_EQ(plan.observations[1].realized.telescope_sample_count, 202U);
    EXPECT_FALSE(plan.active_observation_index.has_value());
}

TEST(AstrometryExecutionPlan, RejectsIdentityChangesAndIncompleteCompletion) {
    citlali::pipeline::AstrometryExecutionPlan plan;
    const auto request = astrometry_request({1.0}, {2.0});
    plan.reset(1);
    citlali::pipeline::record_astrometry_request(plan, 0, 152389, request);

    EXPECT_THROW(
        citlali::pipeline::record_astrometry_request(
            plan, 0, 152390, request),
        std::logic_error);
    EXPECT_THROW(
        citlali::pipeline::record_astrometry_reduction_completed(plan),
        std::logic_error);
}

TEST(AstrometryExecutionPlan, RecordsApplicationAtPointingBoundary) {
    FakeAstrometryTodProc todproc;
    NullLogger logger;
    auto &plan = todproc.engine().astrometry_plan;
    plan.reset(1);
    citlali::pipeline::record_astrometry_request(
        plan, 0, 152389, astrometry_request({1.0}, {2.0}));
    citlali::pipeline::record_astrometry_installed(plan);
    todproc.engine().telescope.tel_data["TelTime"] = {1.0, 2.0, 3.0};

    citlali::pipeline::interpolate_pointing_offsets(todproc, &logger);

    EXPECT_EQ(todproc.interpolation_count, 1U);
    EXPECT_EQ(plan.observations[0].realized.application_count, 1U);
    EXPECT_EQ(plan.observations[0].realized.telescope_sample_count, 3U);
}

TEST(AstrometryExecutionPlan, RejectsChangedRealizedSampleCount) {
    citlali::pipeline::AstrometryExecutionPlan plan;
    plan.reset(1);
    citlali::pipeline::record_astrometry_request(
        plan, 0, 152389, astrometry_request({1.0}, {2.0}));
    citlali::pipeline::record_astrometry_installed(plan);
    citlali::pipeline::record_astrometry_applied(plan, 303);

    EXPECT_THROW(
        citlali::pipeline::record_astrometry_applied(plan, 304),
        std::logic_error);
}

TEST(AstrometryProvenance, SerializesAuthorityIdentityAndRealizedState) {
    citlali::pipeline::AstrometryExecutionPlan plan;
    plan.reset(1);
    citlali::pipeline::record_astrometry_request(
        plan, 0, 152389, astrometry_request({1.0}, {2.0}));
    citlali::pipeline::record_astrometry_installed(plan);
    citlali::pipeline::record_astrometry_applied(plan, 303);
    citlali::pipeline::record_astrometry_reduction_completed(plan);

    const auto node = citlali::pipeline::astrometry_provenance_node(plan);
    EXPECT_EQ(node["schema_version"].as<std::string>(),
              "citlali-astrometry-provenance-v1");
    EXPECT_EQ(node["authority"]["calibration_selection"].as<std::string>(),
              "tolteca");
    EXPECT_FALSE(node["authority"]["support_origin_metadata_available"]
                     .as<bool>());
    EXPECT_EQ(node["observations"][0]["effective"]["resolution"]
                  ["application_mode"]
                      .as<std::string>(),
              "constant");
    EXPECT_EQ(node["observations"][0]["realized"]
                  ["telescope_sample_count"]
                      .as<std::size_t>(),
              303U);
}

TEST(AstrometryProvenance, RequiresCompletionAndWritesAtomically) {
    namespace fs = std::filesystem;
    citlali::pipeline::AstrometryExecutionPlan plan;
    plan.reset(1);
    const auto output_dir =
        fs::path(::testing::TempDir()) / "citlali_astrometry_provenance";
    fs::remove_all(output_dir);
    fs::create_directories(output_dir);

    EXPECT_THROW(
        citlali::pipeline::write_astrometry_provenance_file(output_dir, plan),
        std::logic_error);

    citlali::pipeline::record_astrometry_request(
        plan, 0, 152389, astrometry_request({1.0}, {2.0}));
    citlali::pipeline::record_astrometry_installed(plan);
    citlali::pipeline::record_astrometry_applied(plan, 303);
    citlali::pipeline::record_astrometry_reduction_completed(plan);
    citlali::pipeline::write_astrometry_provenance_file(output_dir, plan);

    const auto output_path =
        citlali::pipeline::astrometry_provenance_path(output_dir);
    ASSERT_TRUE(fs::exists(output_path));
    EXPECT_EQ(YAML::LoadFile(output_path.string())["schema_version"]
                  .as<std::string>(),
              "citlali-astrometry-provenance-v1");
    EXPECT_FALSE(fs::exists(output_path.string() + ".tmp"));
    fs::remove_all(output_dir);
}

}  // namespace
