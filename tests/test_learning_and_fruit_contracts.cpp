#include <citlali/core/engine/learning.h>
#include <citlali/core/pipeline/fits_image_metadata.h>
#include <citlali/core/pipeline/fruit_loop_activation_validation.h>

#include <gtest/gtest.h>

#include <string>

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

}  // namespace
