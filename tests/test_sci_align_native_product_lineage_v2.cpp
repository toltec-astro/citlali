#include "sci_align_native_gap_fixture.h"

#include <citlali/core/pipeline/native_cohort_product_provenance_v2.h>
#include <citlali/core/pipeline/raw_timestream_provenance.h>
#include <citlali/core/pipeline/raw_timestream_provenance_lifecycle.h>

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>

namespace {

namespace fixture = citlali::test_support::sci_align;
namespace pipeline = citlali::pipeline;

struct LineageFixture {
    std::shared_ptr<const pipeline::NativeMeasuredDetectorScan> scan;
    pipeline::NativeMeasuredDetectorLedger ledger;
    pipeline::NativeRtcDispatchResult rtc;
    pipeline::NativePtcPreparedOperation prepared;
    pipeline::NativeScienceProjection projection;
    pipeline::NativeCohortObservationBindingV2 binding;
    pipeline::NativeCohortScanProvenanceV2 record;
};

LineageFixture make_lineage_fixture(bool mapmaking_enabled = true) {
    auto loaded = fixture::load_native_gap_fixture_v1();
    loaded.scan_index = 0;
    auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    pipeline::NativeMeasuredDetectorLedger ledger{scan};
    auto rtc = pipeline::dispatch_native_rtc_runs(
        *scan, {1, false},
        [](const pipeline::NativeRtcRunInput &input) {
            return pipeline::NativeRtcProcessedRun{
                input.measured_values, input.input_flag_bits};
        });
    pipeline::NativePtcCohortRequest ptc_request{
        "all", pipeline::FinitePcaPlaceholder::checked(-71.0),
        {}, {}, false, false};
    auto prepared = pipeline::prepare_native_ptc_cohorts(
        ledger, rtc, ptc_request);
    auto processed = pipeline::run_native_ptc_groups(
        prepared, [](const auto &group) { return group.values(); });
    pipeline::scatter_native_ptc_results_transactionally(
        ledger, prepared, processed);

    pipeline::NativeScienceProjectionRequest projection_request;
    projection_request.pixel_axes = "altaz";
    projection_request.map_grouping = "detector";
    for (std::size_t detector = 0; detector < scan->detector_count();
         ++detector) {
        const auto column = static_cast<Eigen::Index>(detector);
        const auto &binding = scan->binding(column);
        projection_request.detectors.push_back({
            column, binding.output_uid, binding.array,
            binding.network_id, binding.apt_flag, column, 0.0, 0.0});
    }
    auto projection = pipeline::make_native_science_projection(
        ledger, prepared, std::move(projection_request));
    auto binding = pipeline::make_native_cohort_observation_binding_v2(
        0, *scan->relation_handle(), scan->carriers_handle());
    auto record = pipeline::make_native_cohort_scan_provenance_v2(
        binding, ledger, rtc, prepared, projection,
        mapmaking_enabled
            ? pipeline::NativeCohortMapPublicationRequestV2{
                  true, "naive", "urn:citlali:test:map:occurrence",
                  "sha256:test-map-product", "sha256:test-map-weights"}
            : pipeline::NativeCohortMapPublicationRequestV2{});
    return {std::move(scan), std::move(ledger), std::move(rtc),
            std::move(prepared), std::move(projection),
            std::move(binding), std::move(record)};
}

TEST(SciAlignNativeProductLineageV2,
     CompactRelationAndRawManifestDigestsAreDeterministicAndDistinct) {
    const auto fixture = make_lineage_fixture();
    EXPECT_EQ(
        fixture.binding.detector_relation_digest,
        pipeline::native_cohort_detector_relation_digest_v2(
            *fixture.scan->relation_handle()));
    EXPECT_EQ(
        fixture.binding.raw_manifest_digest,
        pipeline::native_cohort_raw_manifest_digest_v2(
            *fixture.scan->relation_handle()));
    EXPECT_NE(fixture.binding.detector_relation_digest,
              fixture.binding.raw_manifest_digest);
    EXPECT_EQ(fixture.binding.digest(), fixture.record.observation_binding_digest);
}

TEST(SciAlignNativeProductLineageV2,
     CommitPublishesOneCompletePreparedSnapshot) {
    auto fixture = make_lineage_fixture();
    auto lineage = pipeline::NativeCohortObservationLineageV2::create(
        fixture.binding, 1);
    auto reservation = lineage->reserve(fixture.record);
    EXPECT_THROW(lineage->snapshot_complete(), std::logic_error);
    reservation.commit();
    const auto snapshot = lineage->snapshot_complete();
    ASSERT_EQ(snapshot.scans.size(), 1U);
    EXPECT_EQ(snapshot.scans.front().rtc_support.size(),
              fixture.record.rtc_support.size());
    EXPECT_EQ(snapshot.scans.front().ptc_groups.size(),
              fixture.record.ptc_groups.size());
    EXPECT_EQ(snapshot.scans.front().revisions.size(),
              static_cast<std::size_t>(fixture.projection.row_count()) *
                  static_cast<std::size_t>(
                      fixture.projection.detector_count()));
    EXPECT_EQ(snapshot.scans.front().map_occurrence.method, "naive");
}

TEST(SciAlignNativeProductLineageV2,
     RollbackLeavesNoPartialStateAndAllowsExactRetry) {
    auto fixture = make_lineage_fixture(false);
    auto lineage = pipeline::NativeCohortObservationLineageV2::create(
        fixture.binding, 1);
    {
        auto reservation = lineage->reserve(fixture.record);
        reservation.rollback();
    }
    EXPECT_THROW(lineage->snapshot_complete(), std::logic_error);
    auto retry = lineage->reserve(fixture.record);
    retry.commit();
    const auto snapshot = lineage->snapshot_complete();
    EXPECT_FALSE(snapshot.scans.front().map_occurrence.mapmaking_enabled);
}

TEST(SciAlignNativeProductLineageV2,
     MissingForeignAndDuplicateLineageRejectBeforePublication) {
    auto fixture = make_lineage_fixture();
    auto lineage = pipeline::NativeCohortObservationLineageV2::create(
        fixture.binding, 2);

    auto foreign = fixture.record;
    foreign.observation_binding_digest = "sha256:foreign";
    EXPECT_THROW(lineage->reserve(std::move(foreign)), std::logic_error);
    EXPECT_THROW(lineage->snapshot_complete(), std::logic_error);

    auto first = lineage->reserve(fixture.record);
    first.commit();
    EXPECT_THROW(lineage->reserve(fixture.record), std::logic_error);
    EXPECT_THROW(lineage->snapshot_complete(), std::logic_error);
}

TEST(SciAlignNativeProductLineageV2,
     IncompleteMapOccurrenceRejectsBeforeReservation) {
    auto fixture = make_lineage_fixture(false);
    fixture.record.map_occurrence.mapmaking_enabled = true;
    fixture.record.map_occurrence.method = "jinc";
    auto lineage = pipeline::NativeCohortObservationLineageV2::create(
        fixture.binding, 1);
    EXPECT_THROW(lineage->reserve(std::move(fixture.record)),
                 std::logic_error);
    EXPECT_THROW(lineage->snapshot_complete(), std::logic_error);
}

TEST(SciAlignNativeProductLineageV2,
     JincOccurrenceRequiresAndSerializesConfigurationAndScanTrace) {
    auto fixture = make_lineage_fixture(false);
    auto record = pipeline::make_native_cohort_scan_provenance_v2(
        fixture.binding, fixture.ledger, fixture.rtc, fixture.prepared,
        fixture.projection,
        {true, "jinc", "urn:citlali:test:jinc:occurrence",
         "sha256:test-jinc-product",
         "sha256:test-jinc-weights",
         std::string{"sha256:test-jinc-processing-config"},
         std::string{"sha256:test-jinc-scan-trace"}});
    EXPECT_EQ(
        record.map_occurrence.jinc_processing_configuration_digest,
        std::optional<std::string>{
            "sha256:test-jinc-processing-config"});
    EXPECT_EQ(
        record.map_occurrence.jinc_scan_trace_digest,
        std::optional<std::string>{"sha256:test-jinc-scan-trace"});

    auto lineage = pipeline::NativeCohortObservationLineageV2::create(
        fixture.binding, 1);
    auto reservation = lineage->reserve(std::move(record));
    reservation.commit();
    EXPECT_NO_THROW(lineage->snapshot_complete());
    pipeline::RawTimestreamExecutionPlan plan;
    plan.initialized = true;
    auto &observation = plan.begin_observation();
    observation.native_consumer_route =
        pipeline::NativeConsumerRoute::native_required;
    observation.native_cohort_lineage = std::move(lineage);
    pipeline::complete_raw_timestream_observation(plan, 1, 0);
    const auto node = pipeline::raw_timestream_provenance_node(plan);
    const auto map = node["realized"]["native_cohort_provenance"]
                         ["value"]["scans"][0]["map_occurrence"];
    EXPECT_EQ(
        map["product_identity_digest"].as<std::string>(),
        "sha256:test-jinc-product");
    EXPECT_EQ(
        map["jinc_processing_configuration_digest"].as<std::string>(),
        "sha256:test-jinc-processing-config");
    EXPECT_EQ(
        map["jinc_scan_trace_digest"].as<std::string>(),
        "sha256:test-jinc-scan-trace");
}

TEST(SciAlignNativeProductLineageV2,
     ObservationCompletionIsAtomicAndSerializesCommittedLineage) {
    auto fixture = make_lineage_fixture();
    pipeline::RawTimestreamExecutionPlan plan;
    plan.initialized = true;
    auto &observation = plan.begin_observation();
    observation.native_consumer_route =
        pipeline::NativeConsumerRoute::native_required;
    observation.native_cohort_lineage =
        pipeline::NativeCohortObservationLineageV2::create(
            fixture.binding, 1);

    EXPECT_THROW(
        pipeline::complete_raw_timestream_observation(plan, 1, 2),
        std::logic_error);
    EXPECT_FALSE(plan.realized.execution_completed);
    EXPECT_FALSE(plan.realized.native_cohort_provenance.has_value());

    auto reservation =
        observation.native_cohort_lineage->reserve(fixture.record);
    reservation.commit();
    EXPECT_THROW(
        pipeline::complete_raw_timestream_observation(plan, 2, 2),
        std::logic_error);
    EXPECT_FALSE(plan.realized.execution_completed);

    ASSERT_NO_THROW(
        pipeline::complete_raw_timestream_observation(plan, 1, 2));
    ASSERT_TRUE(plan.realized.native_cohort_provenance.has_value());
    const auto node = pipeline::raw_timestream_provenance_node(plan);
    EXPECT_EQ(
        node["observation"]["value"]["native_consumer_route"]
            .as<std::string>(),
        "native_required");
    EXPECT_TRUE(
        node["realized"]["native_cohort_provenance"]["available"]
            .as<bool>());
    const auto product =
        node["realized"]["native_cohort_provenance"]["value"];
    EXPECT_EQ(
        product["schema_version"].as<std::string>(),
        pipeline::native_cohort_product_provenance_v2_schema);
    ASSERT_EQ(product["scans"].size(), 1U);
    EXPECT_EQ(
        product["scans"][0]["map_occurrence"]["method"]
            .as<std::string>(),
        "naive");
}

}  // namespace
