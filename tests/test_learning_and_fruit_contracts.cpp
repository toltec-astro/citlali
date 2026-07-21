#include <citlali/core/engine/learning.h>
#include <citlali/core/pipeline/fits_image_metadata.h>
#include <citlali/core/pipeline/fruit_loop_activation_validation.h>
#include <citlali/core/pipeline/learning_housekeeping_qa.h>

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

ReductionLearningState make_learning_state(bool diagnostics_enabled = true,
                                           int diagnostic_cap = 1) {
    ReductionLearningState state;
    ReductionLearningState::Options options;
    options.enabled = true;
    options.diagnostics_enabled = diagnostics_enabled;
    options.max_records_per_type = diagnostic_cap;
    state.configure(options);
    state.begin_iteration(0, false, "science");
    return state;
}

ReductionLearningState::LearnedSampleMask sample_mask(
    std::string obsnum, int uid, long long start, long long stop,
    int iter = 0) {
    ReductionLearningState::LearnedSampleMask record;
    record.obsnum = std::move(obsnum);
    record.producer = "test";
    record.reason = "test_pathology";
    record.iter = iter;
    record.scan = 4;
    record.uid = uid;
    record.ptc_start = start;
    record.ptc_stop = stop;
    return record;
}

ReductionLearningState::DetectorPenalty detector_penalty(
    std::string obsnum, int uid, int iter = 0) {
    ReductionLearningState::DetectorPenalty record;
    record.obsnum = std::move(obsnum);
    record.producer = "ptc_second_pass";
    record.reason = "busy_vetoed_residual";
    record.iter = iter;
    record.scan = 4;
    record.uid = uid;
    record.nw = 1;
    record.array = 0;
    record.factor = 0.0;
    record.scan_local = true;
    return record;
}

struct HousekeepingQaDataItem {
    std::string interface_name;
    std::string path;
    const std::string &interface() const { return interface_name; }
    const std::string &filepath() const { return path; }
};

struct HousekeepingQaRawObs {
    std::vector<HousekeepingQaDataItem> items;
    const auto &data_items() const { return items; }
    auto kidsdata() const {
        std::vector<std::reference_wrapper<const HousekeepingQaDataItem>> kids;
        for (const auto &item : items) {
            if (item.interface_name.rfind("toltec", 0) == 0 &&
                item.interface_name != "toltec_hk") {
                kids.push_back(std::cref(item));
            }
        }
        return kids;
    }
};

struct HousekeepingQaEngine {
    ReductionLearningState learning;
    struct {
        std::string redu_dir_name;
    } output_paths;
    struct {
        int fruit_iter = 0;
    } iteration;
    struct {
        std::string obsnum;
    } observation_identity;
};

TEST(ReductionLearningState, DiagnosticCapNeverTruncatesEffectiveState) {
    auto state = make_learning_state();
    state.record_learned_sample_mask(sample_mask("152390", 10, 20, 22));
    state.record_learned_sample_mask(sample_mask("152433", 11, 30, 31));

    EXPECT_EQ(state.learned_sample_mask_events.size(), 1U);
    EXPECT_EQ(state.dropped_learned_sample_masks, 1U);
    EXPECT_EQ(state.effective_sample_mask_interval_count(), 2U);
    EXPECT_EQ(state.effective_sample_masks_for("152390", 4, false, 2).size(),
              1U);
    EXPECT_EQ(state.effective_sample_masks_for("152433", 4, false, 2).size(),
              1U);
}

TEST(ReductionLearningState, MergesRepeatedAndAdjacentMaskEventsOnline) {
    auto state = make_learning_state(true, 20);
    state.record_learned_sample_mask(sample_mask("152390", 10, 13, 15));
    state.record_learned_sample_mask(sample_mask("152390", 10, 10, 12));
    state.begin_iteration(1, true, "science");
    state.record_learned_sample_mask(sample_mask("152390", 10, 11, 14, 1));

    const auto effective =
        state.effective_sample_masks_for("152390", 4, false, 2);
    ASSERT_EQ(effective.size(), 1U);
    EXPECT_EQ(effective.front().start, 10);
    EXPECT_EQ(effective.front().stop, 15);
    EXPECT_EQ(effective.front().iter, 0);
}

TEST(ReductionLearningState, DiagnosticsCanBeDisabledWithoutDisablingLearning) {
    auto state = make_learning_state(false);
    state.record_learned_sample_mask(sample_mask("152390", 10, 20, 22));
    state.record_detector_penalty(detector_penalty("152390", 10));

    EXPECT_TRUE(state.learned_sample_mask_events.empty());
    EXPECT_TRUE(state.detector_penalty_events.empty());
    EXPECT_EQ(state.effective_sample_mask_interval_count(), 1U);
    EXPECT_EQ(state.effective_detector_penalty_records().size(), 1U);
}

TEST(ReductionLearningState, DetectorPenaltyEventsDeduplicateInEffectiveState) {
    auto state = make_learning_state(true, 1);
    state.record_detector_penalty(detector_penalty("152390", 10));
    state.record_detector_penalty(detector_penalty("152433", 11));
    state.begin_iteration(1, true, "science");
    state.record_detector_penalty(detector_penalty("152390", 10, 1));

    EXPECT_EQ(state.detector_penalty_events.size(), 1U);
    EXPECT_EQ(state.dropped_detector_penalties, 2U);
    EXPECT_EQ(state.effective_detector_penalty_records().size(), 2U);
}

TEST(FruitLoopActivation, RejectsEnabledNoOpConfiguration) {
    citlali::config::TimestreamFruitLoopsConfig fruit_loops;
    fruit_loops.enabled = true;
    fruit_loops.max_iters = 10;
    citlali::config::NoiseConfig noise;

    const auto report =
        citlali::pipeline::validate_fruit_loop_activation(
            fruit_loops, noise, citlali::config::ReductionType::science);
    EXPECT_EQ(report.error_count(), 1U);
    EXPECT_EQ(report.errors().front().path,
              (citlali::config::ConfigPath{"timestream", "fruit_loops"}));
}

TEST(FruitLoopActivation, RequiresEmpiricalProductsForSnrGate) {
    citlali::config::TimestreamFruitLoopsConfig fruit_loops;
    fruit_loops.enabled = true;
    fruit_loops.max_iters = 2;
    fruit_loops.sig2noise_limit = 3.0;
    citlali::config::NoiseConfig noise;

    EXPECT_FALSE(
        citlali::pipeline::validate_fruit_loop_activation(
            fruit_loops, noise, citlali::config::ReductionType::science)
            .ok());
    noise.enabled = true;
    noise.n_noise_maps = 10;
    noise.products_enabled = true;
    EXPECT_TRUE(
        citlali::pipeline::validate_fruit_loop_activation(
            fruit_loops, noise, citlali::config::ReductionType::science)
            .ok());
}

TEST(FruitLoopActivation, AcceptsNonzeroFluxGateWithoutNoiseProducts) {
    citlali::config::TimestreamFruitLoopsConfig fruit_loops;
    fruit_loops.enabled = true;
    fruit_loops.max_iters = 2;
    fruit_loops.array_flux_limit = {0.1, 0.2, 0.3};
    citlali::config::NoiseConfig noise;

    EXPECT_TRUE(
        citlali::pipeline::validate_fruit_loop_activation(
            fruit_loops, noise, citlali::config::ReductionType::science)
            .ok());
}

TEST(FruitLoopActivation, LeavesBeammapInternalIterationPolicyAlone) {
    citlali::config::TimestreamFruitLoopsConfig fruit_loops;
    fruit_loops.enabled = true;
    fruit_loops.max_iters = 1;
    citlali::config::NoiseConfig noise;

    EXPECT_TRUE(
        citlali::pipeline::validate_fruit_loop_activation(
            fruit_loops, noise, citlali::config::ReductionType::beammap)
            .ok());
}

TEST(MapImageSemantics, FormalStandardizedSignalIsNotNamedSnr) {
    EXPECT_EQ(citlali::pipeline::formal_standardized_signal_map_hdu_name(
                  "a1100", "_I"),
              "formal_standardized_signal_a1100_I");
    EXPECT_EQ(citlali::pipeline::formal_standardized_signal_estimator_type(),
              std::string{"formal_weight_standardized"});
    EXPECT_NE(std::string{
                  citlali::pipeline::formal_standardized_signal_map_description()}
                  .find("not a statistical significance map"),
              std::string::npos);
}

TEST(LearningHousekeepingQa, MatchesNearestSampleAndPublishesLocalChange) {
    const std::vector<double> times{100.0, 160.0, 220.0};
    const std::vector<double> temperatures{0.100, 0.110, 0.102};

    const auto match =
        citlali::pipeline::match_learning_housekeeping_sample(
            times, temperatures, 158.0);

    EXPECT_EQ(match.status, "matched");
    EXPECT_DOUBLE_EQ(match.sample_time_unix_sec, 160.0);
    EXPECT_DOUBLE_EQ(match.sample_offset_sec, 2.0);
    EXPECT_DOUBLE_EQ(match.sample_age_sec, 2.0);
    EXPECT_DOUBLE_EQ(match.delta_from_previous, 0.010);
    EXPECT_DOUBLE_EQ(match.delta_to_next, -0.008);
    EXPECT_NEAR(match.local_excursion, 0.009, 1e-12);
}

TEST(LearningHousekeepingQa, RejectsUnavailableAndOutOfRangeMatches) {
    const std::vector<double> times{100.0, 160.0, 220.0};
    const std::vector<double> unavailable{0.100, -1.0, 0.102};

    const auto unavailable_match =
        citlali::pipeline::match_learning_housekeeping_sample(
            times, unavailable, 158.0);
    EXPECT_EQ(unavailable_match.status,
              "nearest_value_invalid_or_unavailable");
    EXPECT_FALSE(std::isfinite(unavailable_match.value));

    const auto outside_match =
        citlali::pipeline::match_learning_housekeeping_sample(
            times, unavailable, 99.0);
    EXPECT_EQ(outside_match.status, "event_outside_housekeeping_range");
    EXPECT_FALSE(std::isfinite(outside_match.sample_time_unix_sec));
}

TEST(LearningHousekeepingQa, WritesRequiredSidecarFromExplicitInput) {
    namespace fs = std::filesystem;
    const auto suffix = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto directory = fs::temp_directory_path() /
                           ("citlali-learning-housekeeping-" + suffix);
    fs::create_directories(directory);
    const auto hk_path = directory / "toltec_hk_test_152433_00.nc";
    {
        netCDF::NcFile file(hk_path.string(), netCDF::NcFile::replace);
        const auto sample = file.addDim("sample", 3);
        const auto time = file.addVar(
            "Data.ToltecThermetry.Time4", netCDF::ncDouble, sample);
        const auto temperature = file.addVar(
            "Data.ToltecThermetry.Temperature4", netCDF::ncDouble, sample);
        const std::vector<double> times{100.0, 160.0, 220.0};
        const std::vector<double> temperatures{0.100, 0.110, 0.102};
        time.putVar(times.data());
        temperature.putVar(temperatures.data());
    }

    HousekeepingQaEngine engine;
    engine.output_paths.redu_dir_name = directory.string();
    engine.observation_identity.obsnum = "152433";
    engine.learning = make_learning_state(true, 20);
    auto event = detector_penalty("152433", -1);
    event.reason = "busy_network_pathology";
    event.scan = 36;
    event.nw = 9;
    event.array = 2;
    event.score = 139.0;
    event.event_time_unix_sec = 158.0;
    engine.learning.record_detector_penalty(event);
    const HousekeepingQaRawObs rawobs{{
        {"toltec_hk", hk_path.string()},
    }};

    EXPECT_NO_THROW(
        citlali::pipeline::write_learning_housekeeping_qa_if_available(
            engine, rawobs, true, spdlog::default_logger()));

    const auto output_path = directory / "learning_housekeeping_iter_0.csv";
    std::ifstream input(output_path);
    ASSERT_TRUE(input.good());
    const std::string contents{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    EXPECT_NE(contents.find("citlali-learning-housekeeping-qa-v1"),
              std::string::npos);
    EXPECT_NE(contents.find("\"matched\",160,2,2,0.11"),
              std::string::npos);
    EXPECT_NE(contents.find("\"channel_missing\""), std::string::npos);

    const HousekeepingQaRawObs sibling_discovery{{
        {"toltec0", (directory / "toltec0_test_152433.nc").string()},
    }};
    const auto discovered =
        citlali::pipeline::find_learning_housekeeping_files(
            sibling_discovery, "152433");
    ASSERT_EQ(discovered.size(), 1U);
    EXPECT_EQ(discovered.front(), hk_path.string());

    std::error_code ignored;
    fs::remove_all(directory, ignored);
}

}  // namespace
