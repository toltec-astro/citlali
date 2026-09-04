#include <citlali/core/pipeline/timestream_d2_native_measurement.h>

#include "timestream_successor_identity_test_support.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace pipeline = citlali::pipeline;
namespace support = citlali::test::timestream_successor;

struct D2Fixture {
    std::shared_ptr<const pipeline::NativePairedReadoutObservation> parent;
    std::shared_ptr<const pipeline::ValSnapshot> val;
    pipeline::D2RouteProfileIdentity route;
    pipeline::D2ResidualRealizationIdentity realization;
};

D2Fixture make_fixture(double time_offset = 0.0) {
    std::vector<pipeline::NativePairedReadoutNetwork> networks;
    networks.push_back(support::make_network(
        0, 10,
        {100.0 + time_offset, 101.0 + time_offset,
         102.0 + time_offset},
        {20, 21, 40}, 2, 1.0, 101.0));
    networks.push_back(support::make_network(
        7, 30, {200.0 + time_offset, 201.0 + time_offset},
        {80, 81}, 1, 701.0, 801.0));
    auto parent = support::make_observation(std::move(networks), {0, 7});
    return {parent, pipeline::ValSnapshot::initial(parent),
            pipeline::D2RouteProfileIdentity::admit(
                "timestream-successor", "identity", "identity-v1"),
            pipeline::D2ResidualRealizationIdentity::admit(
                "rtc-residual-realization:17", "identity-rtc:0.1",
                "effective-config:identity-v1", "network")};
}

std::vector<pipeline::D2ResidualNetworkPayload> make_residuals(
    const D2Fixture &fixture,
    std::shared_ptr<const pipeline::ValSnapshot> snapshot,
    bool make_network_7_absent = false,
    bool make_nonfinite = false) {
    std::vector<pipeline::D2ResidualNetworkPayload> result;
    auto x0 = support::matrix(3, 2, 1000.0);
    if (make_nonfinite) {
        x0(1, 1) = std::numeric_limits<double>::quiet_NaN();
    }
    result.push_back(pipeline::D2ResidualNetworkPayload::realize(
        fixture.parent, fixture.route, snapshot, fixture.realization, 0,
        pipeline::D2ResidualCoordinatePayload::complete(std::move(x0)),
        pipeline::D2ResidualCoordinatePayload::complete(
            support::matrix(3, 2, 2000.0))));
    if (make_network_7_absent) {
        result.push_back(pipeline::D2ResidualNetworkPayload::realize(
            fixture.parent, fixture.route, snapshot, fixture.realization, 7,
            pipeline::D2ResidualCoordinatePayload::absent(),
            pipeline::D2ResidualCoordinatePayload::absent()));
    } else {
        result.push_back(pipeline::D2ResidualNetworkPayload::realize(
            fixture.parent, fixture.route, snapshot, fixture.realization, 7,
            pipeline::D2ResidualCoordinatePayload::complete(
                support::matrix(2, 1, 3000.0)),
            pipeline::D2ResidualCoordinatePayload::complete(
                support::matrix(2, 1, 4000.0))));
    }
    return result;
}

std::vector<std::shared_ptr<const pipeline::D2SourceMaskEvidence>>
make_source_masks(const D2Fixture &fixture) {
    std::vector<std::shared_ptr<const pipeline::D2SourceMaskEvidence>> result;
    result.push_back(pipeline::D2SourceMaskEvidence::admit(
        fixture.parent, fixture.route, 0, "source-mask:network-0",
        pipeline::D2SourceMaskDisposition::applied,
        {0, 0, 0, 1, 0, 0}));
    result.push_back(pipeline::D2SourceMaskEvidence::admit(
        fixture.parent, fixture.route, 7, "source-mask:network-7",
        pipeline::D2SourceMaskDisposition::approved_not_applicable, {}));
    return result;
}

std::vector<std::shared_ptr<const pipeline::D2LineOperatorEvidence>>
make_line_operators(const D2Fixture &fixture) {
    std::vector<std::shared_ptr<const pipeline::D2LineOperatorEvidence>>
        result;
    result.push_back(pipeline::D2LineOperatorEvidence::admit(
        fixture.parent, fixture.route, 0, "line-evidence:network-0",
        pipeline::D2LineOperatorDisposition::applied,
        {{"line-10Hz", 9.5, 10.5, true, "operator:line-10Hz"}}));
    result.push_back(pipeline::D2LineOperatorEvidence::admit(
        fixture.parent, fixture.route, 7, "line-evidence:network-7",
        pipeline::D2LineOperatorDisposition::complete_no_lines, {}));
    return result;
}

pipeline::D2NativeMeasurementPublicationInput make_publication_input(
    const D2Fixture &fixture,
    std::shared_ptr<const pipeline::ValSnapshot> payload_snapshot,
    std::shared_ptr<const pipeline::ValSnapshot> publication_snapshot,
    bool make_network_7_absent = false,
    bool make_nonfinite = false) {
    return {fixture.parent, fixture.route, std::move(publication_snapshot),
            fixture.realization,
            make_residuals(fixture, std::move(payload_snapshot),
                           make_network_7_absent, make_nonfinite),
            make_source_masks(fixture), make_line_operators(fixture)};
}

TEST(timestream_d2_native_measurement,
     publishes_exact_native_axes_zero_copy_prefilter_and_owned_residuals) {
    const auto fixture = make_fixture();
    const auto product = pipeline::D2NativeMeasurement::publish(
        make_publication_input(fixture, fixture.val, fixture.val));

    EXPECT_EQ(product->parent_handle().get(), fixture.parent.get());
    EXPECT_EQ(product->val_snapshot_handle().get(), fixture.val.get());
    EXPECT_EQ(product->val_generation(), pipeline::ValGeneration{0});
    EXPECT_EQ(product->route_profile(), fixture.route);
    EXPECT_EQ(product->residual_realization().realization_id(),
              "rtc-residual-realization:17");
    EXPECT_EQ(product->residual_realization().operator_id(),
              "identity-rtc:0.1");
    EXPECT_EQ(product->residual_realization().effective_config_id(),
              "effective-config:identity-v1");
    EXPECT_EQ(product->residual_realization().grouping_id(), "network");
    EXPECT_EQ(product->network_count(), 2U);

    const auto &network = product->network(0);
    const auto &parent_network = fixture.parent->network(0);
    EXPECT_EQ(network.parent_handle().get(), fixture.parent.get());
    EXPECT_EQ(network.val_snapshot_handle().get(), fixture.val.get());
    EXPECT_EQ(network.occurrence_axis_handle().get(),
              parent_network.occurrence_axis_handle().get());
    EXPECT_EQ(&network.detectors(), &parent_network.detectors());
    EXPECT_EQ(network.prefilter_values(
                  pipeline::NativeReadoutCoordinate::x).data(),
              parent_network.values(
                  pipeline::NativeReadoutCoordinate::x).data());
    EXPECT_EQ(product->prefilter_values(
                  0, pipeline::NativeReadoutCoordinate::r).data(),
              parent_network.values(
                  pipeline::NativeReadoutCoordinate::r).data());
    EXPECT_THROW(
        (void)product->prefilter_values(
            99, pipeline::NativeReadoutCoordinate::x),
        std::out_of_range);
    EXPECT_EQ(network.residual(
                  pipeline::NativeReadoutCoordinate::x).values()(2, 1),
              1021.0);
    EXPECT_EQ(network.sampling_relation(),
              pipeline::D2SamplingRelation::native_occurrence_axis_unchanged);
    EXPECT_EQ(network.grid_relation(),
              pipeline::D2GridRelation::
                  network_native_detector_axis_unchanged);

    const auto runs = network.contiguous_runs();
    ASSERT_EQ(runs.size(), 2U);
    EXPECT_EQ(runs[0].first_native_row, 10);
    EXPECT_EQ(runs[0].past_last_native_row, 12);
    EXPECT_EQ(runs[1].first_native_row, 12);
    EXPECT_EQ(runs[1].past_last_native_row, 13);
    ASSERT_TRUE(runs[0].boundary_after.counter_discontinuity.has_value());
    EXPECT_EQ(runs[0].boundary_after.counter_discontinuity->before_counter,
              21U);
    EXPECT_EQ(runs[0].boundary_after.counter_discontinuity->after_counter,
              40U);

    EXPECT_TRUE(product->source_mask(0).excluded_from_processing(11, 1));
    EXPECT_EQ(product->source_mask(7).disposition(),
              pipeline::D2SourceMaskDisposition::approved_not_applicable);
    EXPECT_EQ(product->line_operator(0).records().size(), 1U);
    EXPECT_EQ(product->line_operator(7).disposition(),
              pipeline::D2LineOperatorDisposition::complete_no_lines);

    const auto memory = product->memory_evidence();
    EXPECT_EQ(memory.residual_numeric_bytes,
              2U * (6U + 2U) * sizeof(double));
    EXPECT_EQ(memory.validation_state_bytes, 0U);
    EXPECT_EQ(memory.referenced_paired_product_count, 1U);
    EXPECT_EQ(memory.referenced_val_snapshot_count, 1U);
    EXPECT_EQ(memory.referenced_native_axis_count, 2U);
    EXPECT_GT(memory.referenced_processing_evidence_bytes, 0U);
}

TEST(timestream_d2_native_measurement,
     requires_exact_snapshot_handle_not_equal_generation) {
    const auto fixture = make_fixture();
    const auto equal_generation_different_snapshot =
        pipeline::ValSnapshot::initial(fixture.parent);
    ASSERT_EQ(equal_generation_different_snapshot->generation(),
              fixture.val->generation());
    ASSERT_NE(equal_generation_different_snapshot.get(), fixture.val.get());

    EXPECT_THROW(
        pipeline::D2NativeMeasurement::publish(make_publication_input(
            fixture, fixture.val, equal_generation_different_snapshot)),
        std::invalid_argument);

    // Creating another snapshot cannot replace the explicitly supplied one.
    // Publication retains the realization-time handle and performs no lookup
    // for an ambient or later "current" VAL state.
    const auto product = pipeline::D2NativeMeasurement::publish(
        make_publication_input(fixture, fixture.val, fixture.val));
    EXPECT_EQ(product->val_snapshot_handle().get(), fixture.val.get());
    EXPECT_NE(product->val_snapshot_handle().get(),
              equal_generation_different_snapshot.get());
}

TEST(timestream_d2_native_measurement,
     nonfinite_residual_is_mechanical_payload_not_local_validity) {
    const auto fixture = make_fixture();
    const auto product = pipeline::D2NativeMeasurement::publish(
        make_publication_input(fixture, fixture.val, fixture.val, true, true));

    EXPECT_TRUE(std::isnan(product->network(0)
                               .residual(pipeline::NativeReadoutCoordinate::x)
                               .values()(1, 1)));
    const auto &absent = product->network(7).residual(
        pipeline::NativeReadoutCoordinate::x);
    EXPECT_EQ(absent.state(), pipeline::D2ResidualPayloadState::absent);
    EXPECT_FALSE(absent.present());
    EXPECT_FALSE(absent.structurally_complete());
    EXPECT_THROW((void)absent.values(), std::logic_error);
    EXPECT_EQ(product->val_snapshot_handle()->generation(),
              pipeline::ValGeneration{0});
}

TEST(timestream_d2_native_measurement,
     residual_realization_rejects_foreign_parent_snapshot_or_native_shape) {
    const auto fixture = make_fixture();
    const auto foreign = make_fixture(50.0);

    EXPECT_THROW(
        pipeline::D2ResidualRealizationIdentity::admit(
            "", "identity-rtc:0.1", "effective-config:identity-v1",
            "network"),
        std::invalid_argument);
    EXPECT_THROW(
        pipeline::D2ResidualNetworkPayload::realize(
            fixture.parent, fixture.route, foreign.val,
            fixture.realization, 0,
            pipeline::D2ResidualCoordinatePayload::complete(
                support::matrix(3, 2, 1.0)),
            pipeline::D2ResidualCoordinatePayload::complete(
                support::matrix(3, 2, 2.0))),
        std::invalid_argument);
    EXPECT_THROW(
        pipeline::D2ResidualNetworkPayload::realize(
            fixture.parent, fixture.route, fixture.val,
            fixture.realization, 0,
            pipeline::D2ResidualCoordinatePayload::complete(
                support::matrix(2, 2, 1.0)),
            pipeline::D2ResidualCoordinatePayload::complete(
                support::matrix(3, 2, 2.0))),
        std::invalid_argument);
    EXPECT_THROW(
        pipeline::D2ResidualNetworkPayload::realize(
            fixture.parent, fixture.route, fixture.val,
            fixture.realization, 0,
            pipeline::D2ResidualCoordinatePayload::complete(
                support::matrix(3, 1, 1.0)),
            pipeline::D2ResidualCoordinatePayload::complete(
                support::matrix(3, 2, 2.0))),
        std::invalid_argument);
}

TEST(timestream_d2_native_measurement,
     publication_rejects_incomplete_duplicate_or_mismatched_evidence) {
    const auto fixture = make_fixture();

    auto missing = make_publication_input(fixture, fixture.val, fixture.val);
    missing.residuals.pop_back();
    EXPECT_THROW(
        pipeline::D2NativeMeasurement::publish(std::move(missing)),
        std::invalid_argument);

    auto duplicate = make_publication_input(fixture, fixture.val, fixture.val);
    duplicate.residuals.pop_back();
    duplicate.residuals.push_back(
        pipeline::D2ResidualNetworkPayload::realize(
            fixture.parent, fixture.route, fixture.val,
            fixture.realization, 0,
            pipeline::D2ResidualCoordinatePayload::complete(
                support::matrix(3, 2, 1.0)),
            pipeline::D2ResidualCoordinatePayload::complete(
                support::matrix(3, 2, 2.0))));
    EXPECT_THROW(
        pipeline::D2NativeMeasurement::publish(std::move(duplicate)),
        std::invalid_argument);

    auto wrong_route =
        make_publication_input(fixture, fixture.val, fixture.val);
    wrong_route.line_operators[0] =
        pipeline::D2LineOperatorEvidence::admit(
            fixture.parent,
            pipeline::D2RouteProfileIdentity::admit(
                "timestream-successor", "non-identity", "v1"),
            0, "line-evidence:wrong-route",
            pipeline::D2LineOperatorDisposition::complete_no_lines, {});
    EXPECT_THROW(
        pipeline::D2NativeMeasurement::publish(std::move(wrong_route)),
        std::invalid_argument);

    auto wrong_realization =
        make_publication_input(fixture, fixture.val, fixture.val);
    wrong_realization.residual_realization =
        pipeline::D2ResidualRealizationIdentity::admit(
            "rtc-residual-realization:18", "identity-rtc:0.1",
            "effective-config:identity-v1", "network");
    EXPECT_THROW(
        pipeline::D2NativeMeasurement::publish(
            std::move(wrong_realization)),
        std::invalid_argument);

    const auto foreign = make_fixture();
    auto foreign_mask =
        make_publication_input(fixture, fixture.val, fixture.val);
    foreign_mask.source_masks[0] =
        pipeline::D2SourceMaskEvidence::admit(
            foreign.parent, fixture.route, 0, "source-mask:foreign-parent",
            pipeline::D2SourceMaskDisposition::applied,
            {0, 0, 0, 0, 0, 0});
    EXPECT_THROW(
        pipeline::D2NativeMeasurement::publish(std::move(foreign_mask)),
        std::invalid_argument);
}

TEST(timestream_d2_native_measurement,
     source_mask_is_exact_processing_evidence_only) {
    const auto fixture = make_fixture();
    EXPECT_THROW(
        pipeline::D2SourceMaskEvidence::admit(
            fixture.parent, fixture.route, 0, "short-mask",
            pipeline::D2SourceMaskDisposition::applied, {0, 1}),
        std::invalid_argument);
    EXPECT_THROW(
        pipeline::D2SourceMaskEvidence::admit(
            fixture.parent, fixture.route, 0, "non-binary-mask",
            pipeline::D2SourceMaskDisposition::applied,
            {0, 0, 0, 2, 0, 0}),
        std::invalid_argument);
    EXPECT_THROW(
        pipeline::D2SourceMaskEvidence::admit(
            fixture.parent, fixture.route, 0, "not-applicable-with-data",
            pipeline::D2SourceMaskDisposition::approved_not_applicable,
            {0, 0, 0, 0, 0, 0}),
        std::invalid_argument);

    const auto evidence = pipeline::D2SourceMaskEvidence::admit(
        fixture.parent, fixture.route, 0, "mask",
        pipeline::D2SourceMaskDisposition::applied,
        {0, 0, 0, 1, 0, 0});
    EXPECT_TRUE(evidence->excluded_from_processing(11, 1));
    EXPECT_FALSE(evidence->excluded_from_processing(10, 0));
    EXPECT_THROW(evidence->excluded_from_processing(13, 0),
                 std::out_of_range);
    EXPECT_EQ(evidence->occurrence_axis_handle().get(),
              fixture.parent->network(0).occurrence_axis_handle().get());
    EXPECT_EQ(fixture.val->generation(), pipeline::ValGeneration{0});
    EXPECT_EQ(fixture.val->committed_delta_findings().size(), 0U);
}

TEST(timestream_d2_native_measurement,
     line_dispositions_require_effective_predecimation_operator_evidence) {
    const auto fixture = make_fixture();
    EXPECT_THROW(
        pipeline::D2LineOperatorEvidence::admit(
            fixture.parent, fixture.route, 0, "empty-applied",
            pipeline::D2LineOperatorDisposition::applied, {}),
        std::invalid_argument);
    EXPECT_THROW(
        pipeline::D2LineOperatorEvidence::admit(
            fixture.parent, fixture.route, 0, "pending-with-record",
            pipeline::D2LineOperatorDisposition::pending,
            {{"line", 1.0, 2.0, true, "operator"}}),
        std::invalid_argument);
    EXPECT_THROW(
        pipeline::D2LineOperatorEvidence::admit(
            fixture.parent, fixture.route, 0, "not-effective",
            pipeline::D2LineOperatorDisposition::applied,
            {{"line", 1.0, 2.0, false, "operator"}}),
        std::invalid_argument);
    EXPECT_THROW(
        pipeline::D2LineOperatorEvidence::admit(
            fixture.parent, fixture.route, 0, "missing-operator",
            pipeline::D2LineOperatorDisposition::applied,
            {{"line", 1.0, 2.0, true, ""}}),
        std::invalid_argument);
    EXPECT_THROW(
        pipeline::D2LineOperatorEvidence::admit(
            fixture.parent, fixture.route, 0, "negative-frequency",
            pipeline::D2LineOperatorDisposition::applied,
            {{"line", -1.0, 2.0, true, "operator"}}),
        std::invalid_argument);
    EXPECT_THROW(
        pipeline::D2LineOperatorEvidence::admit(
            fixture.parent, fixture.route, 0, "overlap",
            pipeline::D2LineOperatorDisposition::applied,
            {{"line-a", 1.0, 3.0, true, "operator-a"},
             {"line-b", 2.0, 4.0, true, "operator-b"}}),
        std::invalid_argument);

    const auto evidence = pipeline::D2LineOperatorEvidence::admit(
        fixture.parent, fixture.route, 0, "applied",
        pipeline::D2LineOperatorDisposition::applied,
        {{"line-b", 3.0, 4.0, true, "operator-b"},
         {"line-a", 1.0, 2.0, true, "operator-a"}});
    ASSERT_EQ(evidence->records().size(), 2U);
    EXPECT_EQ(evidence->records()[0].line_id, "line-a");
    EXPECT_EQ(evidence->records()[1].line_id, "line-b");
    EXPECT_EQ(evidence->occurrence_axis_handle().get(),
              fixture.parent->network(0).occurrence_axis_handle().get());
    EXPECT_EQ(evidence->contiguous_runs().size(), 2U);
    EXPECT_EQ(fixture.val->generation(), pipeline::ValGeneration{0});
    EXPECT_EQ(fixture.val->committed_delta_findings().size(), 0U);
}

}  // namespace
