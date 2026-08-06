#include <gtest/gtest.h>

#include <citlali/core/pipeline/beammap_mapmaking_policy.h>
#include <citlali/core/pipeline/noise_execution_plan.h>
#include <citlali/core/pipeline/noise_provenance.h>
#include <citlali/core/pipeline/noise_realization_identity.h>
#include <citlali/core/pipeline/timestream_scan_generation.h>

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using citlali::pipeline::NoiseAssignmentContext;

NoiseAssignmentContext ordinary_context(
    std::string observation_id = "152390", int iteration = 2,
    std::string pass = "ordinary_mapmaking", int n_realizations = 4,
    std::size_t unit_count = 7, std::size_t channel_count = 12,
    bool randomize_channels = true) {
    return citlali::pipeline::make_noise_assignment_context(
        std::move(observation_id), iteration, std::move(pass),
        n_realizations, unit_count, channel_count, randomize_channels);
}

std::vector<int> flattened_signs(const NoiseAssignmentContext &context) {
    std::vector<int> signs(
        static_cast<std::size_t>(context.n_realizations) *
        context.coherence_unit_count * context.channel_count);
    for (int realization = 0; realization < context.n_realizations;
         ++realization) {
        for (std::size_t unit = 0; unit < context.coherence_unit_count;
             ++unit) {
            for (std::size_t channel = 0; channel < context.channel_count;
                 ++channel) {
                const auto offset =
                    (static_cast<std::size_t>(realization) *
                         context.coherence_unit_count +
                     unit) *
                        context.channel_count +
                    channel;
                signs[offset] = citlali::pipeline::noise_realization_sign(
                    context, realization, unit, channel);
            }
        }
    }
    return signs;
}

TEST(noise_realization_contract,
     deterministic_assignments_ignore_traversal_and_openmp_schedule) {
    const auto context = ordinary_context();
    const auto sequential = flattened_signs(context);
    std::vector<int> reverse(sequential.size());
    for (std::size_t offset = sequential.size(); offset-- > 0;) {
        const auto channel = offset % context.channel_count;
        const auto quotient = offset / context.channel_count;
        const auto unit = quotient % context.coherence_unit_count;
        const auto realization = static_cast<int>(
            quotient / context.coherence_unit_count);
        reverse[offset] = citlali::pipeline::noise_realization_sign(
            context, realization, unit, channel);
    }
    EXPECT_EQ(reverse, sequential);

    std::vector<int> parallel(sequential.size());
#pragma omp parallel for schedule(dynamic, 1)
    for (std::int64_t signed_offset = 0;
         signed_offset < static_cast<std::int64_t>(parallel.size());
         ++signed_offset) {
        const auto offset = static_cast<std::size_t>(signed_offset);
        const auto channel = offset % context.channel_count;
        const auto quotient = offset / context.channel_count;
        const auto unit = quotient % context.coherence_unit_count;
        const auto realization = static_cast<int>(
            quotient / context.coherence_unit_count);
        parallel[offset] = citlali::pipeline::noise_realization_sign(
            context, realization, unit, channel);
    }
    EXPECT_EQ(parallel, sequential);
}

TEST(noise_realization_contract,
     distinct_observations_have_independent_namespaces_and_assignments) {
    const auto first = ordinary_context("152390");
    const auto second = ordinary_context("152391");

    EXPECT_NE(citlali::pipeline::noise_assignment_namespace_digest(first),
              citlali::pipeline::noise_assignment_namespace_digest(second));
    EXPECT_NE(flattened_signs(first), flattened_signs(second));
}

TEST(noise_realization_contract,
     versioned_key_digest_covers_every_realized_identity_dimension) {
    const auto base = ordinary_context();
    const auto next_iteration = ordinary_context(
        "152390", 3, "ordinary_mapmaking", 4, 7, 12, true);
    const auto named_pass = ordinary_context(
        "152390", 2, "ordinary_rebuild", 4, 7, 12, true);
    const auto second_named_pass =
        citlali::pipeline::make_noise_assignment_context(
            "152390", 2, "ordinary_mapmaking", 4, 7, 12, true, 1);
    const auto key = citlali::pipeline::noise_realization_key_digest(
        base, 0, 0, 0);

    EXPECT_NE(key, citlali::pipeline::noise_realization_key_digest(
                       ordinary_context("152391"), 0, 0, 0));
    EXPECT_NE(key, citlali::pipeline::noise_realization_key_digest(
                       next_iteration, 0, 0, 0));
    EXPECT_NE(key, citlali::pipeline::noise_realization_key_digest(
                       named_pass, 0, 0, 0));
    EXPECT_NE(key, citlali::pipeline::noise_realization_key_digest(
                       second_named_pass, 0, 0, 0));
    EXPECT_NE(key, citlali::pipeline::noise_realization_key_digest(
                       base, 1, 0, 0));
    EXPECT_NE(key, citlali::pipeline::noise_realization_key_digest(
                       base, 0, 1, 0));
    EXPECT_NE(key, citlali::pipeline::noise_realization_key_digest(
                       base, 0, 0, 1));

    const auto shared = ordinary_context(
        "152390", 2, "ordinary_mapmaking", 4, 7, 12, false);
    EXPECT_EQ(citlali::pipeline::noise_realization_key_digest(
                  shared, 0, 0, 0),
              citlali::pipeline::noise_realization_key_digest(
                  shared, 0, 0, 11));
    EXPECT_THROW(
        citlali::pipeline::noise_realization_key_digest(
            shared, 0, 0, shared.channel_count),
        std::out_of_range);
}

TEST(noise_realization_contract,
     channel_identity_is_observation_scoped_and_stable_under_shape_growth) {
    const auto smaller = ordinary_context(
        "152390", 2, "ordinary_mapmaking", 4, 7, 12, true);
    const auto larger = ordinary_context(
        "152390", 2, "ordinary_mapmaking", 4, 7, 16, true);

    EXPECT_EQ(citlali::pipeline::noise_assignment_namespace_digest(smaller),
              citlali::pipeline::noise_assignment_namespace_digest(larger));
    for (int realization = 0; realization < smaller.n_realizations;
         ++realization) {
        for (std::size_t unit = 0; unit < smaller.coherence_unit_count;
             ++unit) {
            for (std::size_t channel = 0; channel < smaller.channel_count;
                 ++channel) {
                EXPECT_EQ(
                    citlali::pipeline::noise_realization_sign(
                        smaller, realization, unit, channel),
                    citlali::pipeline::noise_realization_sign(
                        larger, realization, unit, channel));
            }
        }
    }

    const auto shared = ordinary_context(
        "152390", 2, "ordinary_mapmaking", 4, 7, 12, false);
    EXPECT_EQ(citlali::pipeline::noise_realization_sign(shared, 1, 3, 0),
              citlali::pipeline::noise_realization_sign(shared, 1, 3, 11));
}

struct FakeNoiseBlock {
    Eigen::MatrixXi data;
};

struct FakePtc {
    FakeNoiseBlock noise;
};

struct FakeBeammapBuffer {
    int n_noise = 3;
    std::vector<Eigen::MatrixXd> signal{
        Eigen::MatrixXd::Ones(2, 2), Eigen::MatrixXd::Ones(2, 2)};
    std::vector<Eigen::MatrixXd> weight{
        Eigen::MatrixXd::Ones(2, 2), Eigen::MatrixXd::Ones(2, 2)};
    std::vector<Eigen::MatrixXd> grid_weight;
    std::vector<Eigen::MatrixXd> coverage;
    std::vector<Eigen::MatrixXd> kernel;
    std::vector<Eigen::MatrixXd> noise{
        Eigen::MatrixXd::Ones(2, 2), Eigen::MatrixXd::Ones(2, 2)};

    void clear_contribution_diag() {}
};

TEST(noise_realization_contract,
     beammap_named_pass_is_reused_across_active_map_order_and_history) {
    std::vector<FakePtc> ptcs(3);
    const auto primary = ordinary_context(
        "152390", 4, "beammap_primary", 3, ptcs.size(), 5, true);
    citlali::pipeline::populate_beammap_noise_signs(ptcs, true, primary);
    std::vector<Eigen::MatrixXi> expected;
    for (const auto &ptc : ptcs) {
        expected.push_back(ptc.noise.data);
    }

    FakeBeammapBuffer maps;
    Eigen::Matrix<bool, Eigen::Dynamic, 1> active(2);
    active << true, false;
    citlali::pipeline::reset_beammap_mapmaking_buffers(
        maps, 2, false, &active);
    active << false, true;
    citlali::pipeline::reset_beammap_mapmaking_buffers(
        maps, 2, false, &active);
    for (std::size_t unit = 0; unit < ptcs.size(); ++unit) {
        EXPECT_EQ(ptcs[unit].noise.data, expected[unit]);
    }

    std::reverse(ptcs.begin(), ptcs.end());
    std::reverse(ptcs.begin(), ptcs.end());
    citlali::pipeline::populate_beammap_noise_signs(ptcs, true, primary);
    for (std::size_t unit = 0; unit < ptcs.size(); ++unit) {
        EXPECT_EQ(ptcs[unit].noise.data, expected[unit]);
    }

    const auto rebuild = ordinary_context(
        "152390", 4, "beammap_scan_band_rebuild", 3, ptcs.size(), 5,
        true);
    EXPECT_NE(citlali::pipeline::noise_assignment_namespace_digest(primary),
              citlali::pipeline::noise_assignment_namespace_digest(rebuild));
}

TEST(noise_realization_contract,
     compact_provenance_reconstructs_assignments_and_product_digest_join) {
    citlali::config::NoiseConfig request;
    request.enabled = true;
    request.n_noise_maps = 4;
    citlali::pipeline::NoiseExecutionPlan plan;
    plan.reset_from_request(request, true);
    const auto context = ordinary_context();

    citlali::pipeline::record_noise_assignment_completed(plan, context);
    ASSERT_EQ(plan.assignments.size(), 1U);
    const auto &record = plan.assignments.front();
    EXPECT_EQ(record.key_policy_version,
              citlali::pipeline::noise_realization_key_policy_version);
    EXPECT_EQ(record.ensemble_mode,
              citlali::pipeline::noise_ensemble_mode_source_imprinted_current);
    EXPECT_EQ(record.completed_realization_ids,
              (std::vector<std::size_t>{0, 1, 2, 3}));
    EXPECT_FALSE(record.namespace_digest.empty());
    EXPECT_FALSE(record.partition_digest.empty());
    EXPECT_FALSE(record.reconstruction_digest.empty());
    EXPECT_TRUE(record.compact());

    const auto identity =
        citlali::pipeline::noise_realization_product_identity(
            plan, "152390", false, 2);
    EXPECT_EQ(identity.ensemble_mode,
              citlali::pipeline::noise_ensemble_mode_source_imprinted_current);
    EXPECT_EQ(identity.realization_id, 2U);
    EXPECT_EQ(identity.assignment_digest, record.reconstruction_digest);
    EXPECT_FALSE(identity.product_digest_join.empty());

    const auto node = citlali::pipeline::noise_provenance_node(plan);
    EXPECT_EQ(node["schema_version"].as<std::string>(),
              "citlali-noise-products-provenance-v2");
    EXPECT_EQ(node["assignment_policy"]["key_policy_version"]
                  .as<std::string>(),
              citlali::pipeline::noise_realization_key_policy_version);
    EXPECT_EQ(node["assignments"][0]["ensemble_mode"].as<std::string>(),
              citlali::pipeline::noise_ensemble_mode_source_imprinted_current);
    EXPECT_EQ(node["assignments"][0]["completed_realization_ids"].size(),
              4U);
    EXPECT_EQ(node["assignment_summary"]["record_count"].as<std::size_t>(),
              1U);
    EXPECT_FALSE(node["assignment_summary"]["digest"].as<std::string>()
                     .empty());
}

TEST(noise_realization_contract,
     provenance_and_coadd_digest_join_are_stable_under_observation_order) {
    citlali::config::NoiseConfig request;
    request.enabled = true;
    request.n_noise_maps = 4;
    citlali::pipeline::NoiseExecutionPlan forward;
    citlali::pipeline::NoiseExecutionPlan reverse;
    forward.reset_from_request(request, true);
    reverse.reset_from_request(request, true);
    const auto first = ordinary_context("152390");
    const auto second = ordinary_context("152391");

    citlali::pipeline::record_noise_assignment_completed(forward, first);
    citlali::pipeline::record_noise_assignment_completed(forward, second);
    citlali::pipeline::record_noise_assignment_completed(reverse, second);
    citlali::pipeline::record_noise_assignment_completed(reverse, first);

    ASSERT_EQ(forward.assignments.size(), 2U);
    ASSERT_EQ(reverse.assignments.size(), 2U);
    EXPECT_EQ(forward.assignments[0].observation_id, "152390");
    EXPECT_EQ(reverse.assignments[0].observation_id, "152390");
    EXPECT_EQ(citlali::pipeline::noise_assignment_records_digest(
                  forward.assignments),
              citlali::pipeline::noise_assignment_records_digest(
                  reverse.assignments));
    EXPECT_EQ(citlali::pipeline::noise_realization_product_identity(
                  forward, "", true, 3)
                  .product_digest_join,
              citlali::pipeline::noise_realization_product_identity(
                  reverse, "", true, 3)
                  .product_digest_join);
}

}  // namespace
