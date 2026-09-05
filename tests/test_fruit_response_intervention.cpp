#include <citlali/core/fruit/response_intervention.h>
#include <gtest/gtest.h>
#include <citlali/core/engine/learning.h>
#include <citlali/core/pipeline/fruit_response_hooks.h>

#include <limits>

namespace {
using namespace citlali::fruit;

ResponseCandidate candidate(int uid, double ratio = 2.0) {
    ResponseCandidate value;
    value.key = {"synthetic", 0, uid, 0};
    value.response.ratio = ratio;
    value.response.footprint = 1;
    value.response.conditioned = 1;
    return value;
}

ResponseInterventionState selected(const std::string &arm) {
    ResponseInterventionState state;
    state.configure(arm);
    state.begin(0, false);
    auto value = candidate(100);
    state.resolve({value}, {{0, value.response}});
    return state;
}

TEST(FruitResponseIntervention, DisabledAndEmptyCensusHaveNoAction) {
    ResponseInterventionState disabled;
    disabled.begin(50, true);
    EXPECT_EQ(disabled.coefficient(candidate(100).key), 1.0);
    ResponseInterventionState state;
    state.configure("Half");
    state.begin(0, false);
    state.resolve({}, {});
    EXPECT_TRUE(state.assignments().empty());
    auto copy = ResponseInterventionState::restore(state.serialize(), "Half", 0);
    EXPECT_EQ(copy.serialize(), state.serialize());
}

TEST(FruitResponseIntervention, EitherStagePermitsHalfAndNeverStacks) {
    for (const auto &arm : {"Half", "Hold", "H"}) {
        for (const bool raw_permits : {false, true}) {
            for (const bool processed_permits : {false, true}) {
                auto state = selected(arm);
                state.begin(1, true);
                const auto key = candidate(100).key;
                EXPECT_THROW(state.coefficient(key), std::runtime_error);
                state.record_stage(key, 0, raw_permits ? 4 : 3,
                                   raw_permits ? 0.02 : 0.03, 0.02);
                state.record_stage(key, 1, processed_permits ? 4 : 3,
                                   processed_permits ? 0.01 : 0.03, 0.02);
                const double expected = std::string(arm) == "Half" &&
                    (raw_permits || processed_permits) ? 0.5 : 1.0;
                EXPECT_EQ(state.coefficient(key), expected);
                EXPECT_THROW(state.record_stage(key, 1, 4, 0.01, 0.02), std::runtime_error);
                state.resolve({}, {});
                const auto bytes = state.serialize();
                auto copy = ResponseInterventionState::restore(bytes, arm, 1);
                EXPECT_EQ(copy.coefficient(key), expected);
                copy.begin(2, false);
                EXPECT_EQ(copy.coefficient(key), 1.0);  // no permission carryover
            }
        }
    }
}

TEST(FruitResponseIntervention, AllKeysAndJointRiskDriveSelectionWithoutUidRanking) {
    ResponseInterventionState state;
    state.configure("Half");
    state.begin(0, false);
    auto first = candidate(987, 0.2);
    auto second = candidate(5, 0.3);
    ResponseScore joint;
    joint.support_risk = true;
    joint.lost = 1;
    state.resolve({first, second}, {{0, joint}});
    EXPECT_EQ(state.assignments().size(), 2U);
    ResponseInterventionState reordered;
    reordered.configure("Half");
    reordered.begin(0, false);
    reordered.resolve({second, first}, {{0, joint}});
    EXPECT_EQ(reordered.serialize(), state.serialize());
    first.key.uid = 2000;
    second.key.uid = 3000;
    ResponseInterventionState renamed;
    renamed.configure("Half");
    renamed.begin(0, false);
    renamed.resolve({first, second}, {{0, joint}});
    EXPECT_EQ(renamed.assignments().size(), 2U);
}

TEST(FruitResponseIntervention, IndependentExclusionsRemainOutsideSelection) {
    auto independent = candidate(100, 1e9);
    independent.disposition = "independent_exclusion";
    ResponseInterventionState state;
    state.configure("Hold");
    state.begin(0, false);
    state.resolve({independent}, {});
    EXPECT_FALSE(state.suppresses(independent.key));
    EXPECT_FALSE(state.census().front().selected);
}

TEST(FruitResponseIntervention, InvalidAndIncompleteCensusCannotPartiallyAssign) {
    ResponseInterventionState state;
    state.configure("Half");
    state.begin(0, false);
    auto value = candidate(100);
    EXPECT_THROW(state.resolve({value, value}, {{0, value.response}}), std::runtime_error);
    EXPECT_THROW(state.resolve({value}, {}), std::runtime_error);
    value.response.ratio = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(state.resolve({value}, {{0, value.response}}), std::runtime_error);
    EXPECT_TRUE(state.assignments().empty());
    EXPECT_FALSE(state.completed());
}

TEST(FruitResponseIntervention, PopulationAndAssignmentCapsNeverTruncate) {
    ResponseInterventionState state;
    state.configure("Half");
    state.begin(0, false);
    std::vector<ResponseCandidate> all;
    for (int i = 0; i < 17; ++i) all.push_back(candidate(i));
    EXPECT_THROW(state.resolve(all, {{0, all.front().response}}), std::runtime_error);
    EXPECT_TRUE(state.assignments().empty());
    all.pop_back();
    state.resolve(all, {{0, all.front().response}});
    for (int iteration = 1; iteration <= 4; ++iteration) {
        state.begin(iteration, false);
        for (int i = 0; i < 16; ++i) all[i] = candidate(iteration * 16 + i);
        if (iteration < 4) state.resolve(all, {{0, all.front().response}});
        else EXPECT_THROW(state.resolve(all, {{0, all.front().response}}), std::runtime_error);
    }
    EXPECT_EQ(state.assignments().size(), 64U);
}

TEST(FruitResponseIntervention, RestartRejectsMissingChangedAndBeyondHorizonState) {
    auto state = selected("Half");
    const auto bytes = state.serialize();
    EXPECT_THROW(ResponseInterventionState::restore(bytes, "Hold", 0), std::runtime_error);
    EXPECT_THROW(ResponseInterventionState::restore(bytes, "Half", 1), std::runtime_error);
    EXPECT_THROW(ResponseInterventionState::restore(bytes.substr(0, bytes.size() / 2), "Half", 0), std::runtime_error);
    EXPECT_THROW(ResponseInterventionState::restore(bytes + "extra", "Half", 0), std::runtime_error);
    for (int iteration = 1; iteration <= 6; ++iteration) {
        state.begin(iteration, false);
        state.resolve({}, {});
    }
    EXPECT_THROW(state.begin(7, false), std::runtime_error);
    EXPECT_THROW(ResponseInterventionState::restore(state.serialize(), "Half", 6), std::runtime_error);
}

TEST(FruitResponseIntervention, HighInfluenceDoesNotClaimBeneficialRescue) {
    // Fixed-processed-state deletion removes a harmful +100 contribution
    // from a zero-truth map. Influence is large even though hard exclusion
    // is useful. The selector selects it; only the later science protections
    // can decide whether replacing the hard action was beneficial.
    const double truth = 0.0;
    const double with_bad_contribution = 100.0;
    const double after_hard_exclusion = 0.0;
    auto value = candidate(100, std::abs(after_hard_exclusion - with_bad_contribution));
    ResponseInterventionState state;
    state.configure("Hold");
    state.begin(0, false);
    state.resolve({value}, {{0, value.response}});
    EXPECT_TRUE(state.census().front().selected);
    EXPECT_GT(std::abs(with_bad_contribution - truth), std::abs(after_hard_exclusion - truth));
}
}  // namespace


TEST(FruitResponseCensus, IncludesAllNewKeysAndKeepsExclusionReasonsWithoutOracleInputs) {
    ReductionLearningState learning;
    ReductionLearningState::Options options;
    options.enabled = true;
    options.fruit_response_arm = "Half";
    learning.configure(options);
    auto record = [&](int uid, int iteration, std::string reason, double factor = 0.0) {
        ReductionLearningState::DetectorPenalty value;
        value.obsnum = "123424";
        value.array = 0; value.uid = uid; value.nw = 0; value.scan = 0;
        value.iter = iteration; value.scan_local = true; value.factor = factor;
        value.reason = reason; value.producer = "mapdiag:raw_obs";
        learning.record_detector_penalty(value, true);
    };
    const std::string reason = "map_pixel_outlier_detector_dominance";
    learning.begin_iteration(0, false, "pointing");
    record(10, 0, reason);
    learning.fruit_response.resolve({}, {});
    learning.finalize_iteration(0);
    learning.begin_iteration(1, true, "pointing");
    record(900, 1, reason);
    record(3, 1, reason);
    record(3, 1, reason); // repeated proposal is one opportunity
    record(11, 1, reason);
    record(11, 1, "busy_vetoed_residual");
    record(12, 1, reason, 0.5); // non-hard records are outside the population
    const auto census = citlali::pipeline::fruit_response_census(learning, "123424", 1);
    ASSERT_EQ(census.size(), 4U);
    std::map<int, std::string> dispositions;
    for (const auto &entry : census) dispositions[entry.key.uid] = entry.disposition;
    EXPECT_EQ(dispositions.at(3), "eligible");
    EXPECT_EQ(dispositions.at(900), "eligible");
    EXPECT_EQ(dispositions.at(10), "entry_hard_exclusion");
    EXPECT_EQ(dispositions.at(11), "independent_exclusion");
}
