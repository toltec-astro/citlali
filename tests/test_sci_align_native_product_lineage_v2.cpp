#include "sci_align_native_gap_fixture.h"

#include <citlali/core/pipeline/native_cohort_product_provenance_v3.h>
#include <citlali/core/pipeline/native_cohort_debug_trace.h>
#include <citlali/core/pipeline/raw_timestream_provenance.h>
#include <citlali/core/pipeline/raw_timestream_provenance_lifecycle.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
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
    pipeline::NativeCohortScanProvenanceV3 record;
};

LineageFixture make_lineage_fixture(bool mapmaking_enabled = true,
                                    bool operation_exclusion = false,
                                    bool rtc_processing_flag = false,
                                    bool force_weight_zero = false) {
    auto loaded = fixture::load_native_gap_fixture_v1();
    loaded.scan_index = 0;
    auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    pipeline::NativeMeasuredDetectorLedger ledger{scan};
    auto rtc = pipeline::dispatch_native_rtc_runs(
        *scan, {1, false},
        [rtc_processing_flag](const pipeline::NativeRtcRunInput &input) {
            auto flags = input.input_flag_bits;
            if (rtc_processing_flag) {
                flags(0, 0) |=
                    pipeline::native_cohort_rtc_processing_flag_bit_v3;
            }
            return pipeline::NativeRtcProcessedRun{
                input.measured_values, std::move(flags)};
        });
    pipeline::NativePtcCohortRequest ptc_request{
        "all", pipeline::FinitePcaPlaceholder::checked(-71.0),
        {}, {}, false, false};
    if (operation_exclusion) {
        const auto &first_run = rtc.runs.front();
        for (const auto &support : first_run.support) {
            ptc_request.operation_exclusion_bits.emplace(
                pipeline::NativeDetectorSampleKey{
                    support.selected_anchor.key(),
                    first_run.input.detector_columns.front()},
                pipeline::native_cohort_duplicate_tone_exclusion_bit_v3);
        }
    }
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
    std::size_t positive_weights = 0;
    std::size_t zero_weights = 0;
    std::vector<pipeline::TimestreamDetectorColumn>
        zero_weight_detector_columns;
    for (Eigen::Index detector = 0;
         detector < projection.detector_count(); ++detector) {
        bool eligible = false;
        for (Eigen::Index row = 0; row < projection.row_count(); ++row) {
            eligible = eligible || !projection.flags()(row, detector);
        }
        if (force_weight_zero && detector == 0) {
            eligible = false;
        }
        if (eligible) {
            ++positive_weights;
        }
        else {
            ++zero_weights;
            zero_weight_detector_columns.push_back(detector);
        }
    }
    pipeline::NativeCohortMapPublicationRequestV3 map_request;
    if (mapmaking_enabled) {
        map_request.mapmaking_enabled = true;
        map_request.method = "naive";
        map_request.product_occurrence =
            "urn:citlali:test:map:occurrence";
        map_request.product_identity_digest =
            "sha256:test-map-product";
        map_request.eligible_weight_digest =
            "sha256:test-map-weights";
        map_request.positive_weight_detector_count = positive_weights;
        map_request.zero_weight_detector_count = zero_weights;
        map_request.zero_weight_detector_columns =
            std::move(zero_weight_detector_columns);
    }
    auto record = pipeline::make_native_cohort_scan_provenance_v3(
        binding, ledger, rtc, prepared, projection,
        projection.flags(), projection.flags(), projection.flags(),
        std::move(map_request));
    return {std::move(scan), std::move(ledger), std::move(rtc),
            std::move(prepared), std::move(projection),
            std::move(binding), std::move(record)};
}

pipeline::RawTimestreamCanonicalRunIdentity make_run_identity() {
    const std::string digest =
        "sha256:0123456789abcdef0123456789abcdef"
        "0123456789abcdef0123456789abcdef";
    return {digest, digest, digest, {{0, "/test/config.yaml", 17, digest}}};
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
    auto lineage = pipeline::NativeCohortObservationLineageV3::create(
        fixture.binding, *fixture.scan->relation_handle(), 1);
    auto reservation = lineage->reserve(fixture.record);
    EXPECT_THROW(lineage->snapshot_complete(), std::logic_error);
    reservation.commit();
    const auto snapshot = lineage->snapshot_complete();
    ASSERT_EQ(snapshot.scans.size(), 1U);
    EXPECT_EQ(snapshot.scans.front().rtc.output_row_count,
              fixture.record.rtc.output_row_count);
    EXPECT_EQ(snapshot.scans.front().ptc.group_count,
              fixture.record.ptc.group_count);
    EXPECT_EQ(snapshot.scans.front().population.detector_sample_count,
              static_cast<std::size_t>(fixture.projection.row_count()) *
                  static_cast<std::size_t>(
                      fixture.projection.detector_count()));
    EXPECT_EQ(snapshot.scans.front().map_occurrence.method, "naive");
}

TEST(SciAlignNativeProductLineageV2,
     RollbackLeavesNoPartialStateAndAllowsExactRetry) {
    auto fixture = make_lineage_fixture(false);
    auto lineage = pipeline::NativeCohortObservationLineageV3::create(
        fixture.binding, *fixture.scan->relation_handle(), 1);
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
    auto lineage = pipeline::NativeCohortObservationLineageV3::create(
        fixture.binding, *fixture.scan->relation_handle(), 2);

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
    auto lineage = pipeline::NativeCohortObservationLineageV3::create(
        fixture.binding, *fixture.scan->relation_handle(), 1);
    EXPECT_THROW(lineage->reserve(std::move(fixture.record)),
                 std::logic_error);
    EXPECT_THROW(lineage->snapshot_complete(), std::logic_error);
}

TEST(SciAlignNativeProductLineageV2,
     JincOccurrenceRequiresConfigurationWithoutBindingExecutionTrace) {
    auto fixture = make_lineage_fixture(false);
    pipeline::NativeCohortMapPublicationRequestV3 request;
    request.mapmaking_enabled = true;
    request.method = "jinc";
    request.product_occurrence = "urn:citlali:test:jinc:occurrence";
    request.product_identity_digest = "sha256:test-jinc-product";
    request.eligible_weight_digest = "sha256:test-jinc-weights";
    request.positive_weight_detector_count =
        static_cast<std::size_t>(fixture.projection.detector_count());
    request.jinc_processing_configuration_digest =
        "sha256:test-jinc-processing-config";
    Eigen::MatrixXi noise_signs(2, 1);
    noise_signs << 1, -1;
    request.noise_assignment =
        pipeline::make_native_noise_assignment_summary_v3(
            noise_signs, true, false, 2,
            static_cast<std::size_t>(fixture.projection.detector_count()));
    request.fruit_loop_feedback.enabled = true;
    request.fruit_loop_feedback.source_model_available = true;
    request.fruit_loop_feedback.noise_map_pass_applied = true;
    request.fruit_loop_feedback.keep_source_subtracted_weights = true;
    request.fruit_loop_feedback.iteration = 2;
    request.fruit_loop_feedback.model_map_count = 4;
    request.fruit_loop_feedback.subtraction_sample_count = 11;
    request.fruit_loop_feedback.addback_sample_count = 9;
    request.fruit_loop_feedback.interpolation_mode = "jinc";
    request.fruit_loop_feedback.support_authority =
        pipeline::native_fruit_loop_feedback_authority_v3;
    auto record = pipeline::make_native_cohort_scan_provenance_v3(
        fixture.binding, fixture.ledger, fixture.rtc, fixture.prepared,
        fixture.projection, fixture.projection.flags(),
        fixture.projection.flags(), fixture.projection.flags(),
        std::move(request));
    EXPECT_EQ(
        record.map_occurrence.jinc_processing_configuration_digest,
        std::optional<std::string>{
            "sha256:test-jinc-processing-config"});

    auto lineage = pipeline::NativeCohortObservationLineageV3::create(
        fixture.binding, *fixture.scan->relation_handle(), 1);
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
    EXPECT_FALSE(map["jinc_scan_trace_digest"]);
    ASSERT_TRUE(map["noise_assignment"]["enabled"].as<bool>());
    EXPECT_EQ(
        map["noise_assignment"]["realization_count"].as<std::size_t>(),
        2U);
    EXPECT_EQ(
        map["noise_assignment"]["assignment_column_count"]
            .as<std::size_t>(),
        1U);
    EXPECT_FALSE(
        map["noise_assignment"]["assignment_values_serialized"].as<bool>());
    ASSERT_TRUE(map["fruit_loop_feedback"]["enabled"].as<bool>());
    EXPECT_EQ(
        map["fruit_loop_feedback"]["subtraction_sample_count"]
            .as<std::size_t>(),
        11U);
    EXPECT_EQ(
        map["fruit_loop_feedback"]["addback_sample_count"]
            .as<std::size_t>(),
        9U);
    EXPECT_FALSE(
        map["fruit_loop_feedback"]["projected_values_serialized"]
            .as<bool>());
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
        pipeline::NativeCohortObservationLineageV3::create(
            fixture.binding, *fixture.scan->relation_handle(), 1);

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
        pipeline::native_cohort_product_provenance_v3_schema);
    ASSERT_EQ(product["scans"].size(), 1U);
    EXPECT_EQ(
        product["scans"][0]["map_occurrence"]["method"]
            .as<std::string>(),
        "naive");
}

TEST(SciAlignNativeProductLineageV2,
     NativePublicationRequiresCompleteCanonicalRunIdentity) {
    auto fixture = make_lineage_fixture();
    pipeline::RawTimestreamExecutionPlan plan;
    plan.initialized = true;
    auto &observation = plan.begin_observation();
    observation.native_consumer_route =
        pipeline::NativeConsumerRoute::native_required;
    observation.native_cohort_lineage =
        pipeline::NativeCohortObservationLineageV3::create(
            fixture.binding, *fixture.scan->relation_handle(), 1);
    auto reservation =
        observation.native_cohort_lineage->reserve(fixture.record);
    reservation.commit();
    pipeline::complete_raw_timestream_observation(plan, 1, 0);

    const auto output_dir = std::filesystem::path(testing::TempDir()) /
        "citlali_bounded_native_publication_identity_test";
    std::filesystem::remove_all(output_dir);
    std::filesystem::create_directories(output_dir);
    EXPECT_THROW(
        pipeline::write_raw_timestream_provenance_file(output_dir, plan),
        std::logic_error);
    EXPECT_FALSE(std::filesystem::exists(
        pipeline::raw_timestream_provenance_path(output_dir)));

    plan.canonical_run_identity = make_run_identity();
    ASSERT_NO_THROW(
        pipeline::write_raw_timestream_provenance_file(output_dir, plan));
    const auto stored = YAML::LoadFile(
        pipeline::raw_timestream_provenance_path(output_dir).string());
    EXPECT_TRUE(stored["canonical_run_identity"]["available"].as<bool>());
    EXPECT_EQ(
        stored["canonical_run_identity"]["config_sources"][0]["path"]
            .as<std::string>(),
        "/test/config.yaml");
    EXPECT_EQ(
        stored["software_identity"]["citlali_revision"].as<std::string>(),
        CITLALI_GIT_REVISION);
    EXPECT_TRUE(
        stored["canonical_publication"]["required"].as<bool>());
    EXPECT_EQ(
        stored["canonical_publication"]["status"].as<std::string>(),
        "validated_complete");
    EXPECT_TRUE(stored["canonical_publication"]
                      ["bounded_provenance_validated"]
                          .as<bool>());
    std::filesystem::remove_all(output_dir);
}

TEST(SciAlignNativeProductLineageV2,
     OperationExclusionCauseIsNamedAndSerializedAtScanScope) {
    auto fixture = make_lineage_fixture(true, true);
    const auto found = std::find_if(
        fixture.record.scoped_causes.begin(),
        fixture.record.scoped_causes.end(),
        [](const auto &candidate) {
            return candidate.cause == "duplicate_tone";
        });
    ASSERT_NE(found, fixture.record.scoped_causes.end());
    const auto &cause = *found;
    EXPECT_EQ(cause.scope, "scan_summary");
    EXPECT_EQ(cause.authority,
              "citlali.native_duplicate_tone_policy_v2");
    EXPECT_EQ(cause.cause, "duplicate_tone");
    EXPECT_EQ(cause.flag_bits,
              pipeline::native_cohort_duplicate_tone_exclusion_bit_v3);
    EXPECT_EQ(cause.count_unit, "detector_samples");
    EXPECT_EQ(cause.detector_columns.size(), 1U);
    EXPECT_GT(cause.affected_count, 0U);

    auto lineage = pipeline::NativeCohortObservationLineageV3::create(
        fixture.binding, *fixture.scan->relation_handle(), 1);
    auto reservation = lineage->reserve(std::move(fixture.record));
    reservation.commit();
    pipeline::RawTimestreamExecutionPlan plan;
    plan.initialized = true;
    auto &observation = plan.begin_observation();
    observation.native_consumer_route =
        pipeline::NativeConsumerRoute::native_required;
    observation.native_cohort_lineage = std::move(lineage);
    pipeline::complete_raw_timestream_observation(plan, 1, 0);
    const auto causes =
        pipeline::raw_timestream_provenance_node(plan)
            ["realized"]["native_cohort_provenance"]["value"]
            ["scans"][0]["scoped_causes"];
    bool serialized_duplicate = false;
    for (const auto &serialized : causes) {
        if (serialized["cause"].as<std::string>() == "duplicate_tone") {
            serialized_duplicate = true;
            EXPECT_EQ(serialized["scope"].as<std::string>(),
                      "scan_summary");
        }
    }
    EXPECT_TRUE(serialized_duplicate);
    EXPECT_FALSE(
        pipeline::raw_timestream_provenance_node(plan)
            ["realized"]["native_cohort_provenance"]["value"]
            ["scans"][0]["revision_transitions"]);
}

TEST(SciAlignNativeProductLineageV2,
     RawAndRtcFlagsRetainDistinctNamedAuthorities) {
    const auto fixture = make_lineage_fixture(false, false, true);
    const auto find_authority = [&](const std::string &authority) {
        return std::find_if(
            fixture.record.scoped_causes.begin(),
            fixture.record.scoped_causes.end(),
            [&](const auto &cause) {
                return cause.authority == authority;
            });
    };
    const auto rtc = find_authority(
        "citlali.native_rtc_processing_policy_v2");
    ASSERT_NE(rtc, fixture.record.scoped_causes.end());
    EXPECT_EQ(rtc->cause, "rtc_processing_generated_flag");
    EXPECT_EQ(rtc->scope, "scan_summary");
    EXPECT_EQ(rtc->flag_bits,
              pipeline::native_cohort_rtc_processing_flag_bit_v3);
    EXPECT_EQ(
        fixture.record.population.rtc_processing_flagged_sample_count,
        rtc->affected_count);

    const auto raw = find_authority("raw_kids_input.sample_flags");
    if (raw != fixture.record.scoped_causes.end()) {
        EXPECT_EQ(raw->cause, "raw_input_flag_bits");
        EXPECT_EQ(raw->scope, "scan_summary");
        ASSERT_TRUE(raw->flag_bits.has_value());
        EXPECT_EQ(*raw->flag_bits &
                      pipeline::native_cohort_rtc_processing_flag_bit_v3,
                  0U);
    }
}

TEST(SciAlignNativeProductLineageV2,
     AdditionalZeroWeightDetectorHasNamedWeightAuthority) {
    const auto fixture = make_lineage_fixture(true, false, false, true);
    const auto found = std::find_if(
        fixture.record.scoped_causes.begin(),
        fixture.record.scoped_causes.end(),
        [](const auto &cause) {
            return cause.authority ==
                "citlali.weighting.detector_weight_contract_v1";
        });
    ASSERT_NE(found, fixture.record.scoped_causes.end());
    EXPECT_EQ(found->scope, "scan_detector");
    EXPECT_EQ(found->cause, "nonpositive_final_detector_weight");
    EXPECT_EQ(found->count_unit, "detectors");
    EXPECT_FALSE(found->flag_bits.has_value());
    EXPECT_EQ(found->affected_count, found->detector_columns.size());
    EXPECT_TRUE(std::binary_search(found->detector_columns.begin(),
                                   found->detector_columns.end(), 0));
    auto lineage = pipeline::NativeCohortObservationLineageV3::create(
        fixture.binding, *fixture.scan->relation_handle(), 1);
    auto reservation = lineage->reserve(fixture.record);
    reservation.commit();
    EXPECT_NO_THROW(lineage->snapshot_complete());
}

TEST(SciAlignNativeProductLineageV2,
     RuntimeFlagsBecomeBoundedNamedIntervalsWithoutSampleLedger) {
    auto fixture = make_lineage_fixture(false);
    auto ptc_flags = fixture.projection.flags();
    auto final_flags = ptc_flags;
    std::vector<std::pair<Eigen::Index, Eigen::Index>> eligible;
    for (Eigen::Index detector = 0;
         detector < ptc_flags.cols(); ++detector) {
        for (Eigen::Index row = 0; row < ptc_flags.rows(); ++row) {
            if (!ptc_flags(row, detector)) {
                eligible.emplace_back(row, detector);
            }
        }
    }
    ASSERT_GE(eligible.size(), 1U);
    ptc_flags(eligible[0].first, eligible[0].second) = true;
    final_flags = ptc_flags;

    auto record = pipeline::make_native_cohort_scan_provenance_v3(
        fixture.binding, fixture.ledger, fixture.rtc, fixture.prepared,
        fixture.projection, fixture.projection.flags(), ptc_flags,
        final_flags, {});
    EXPECT_EQ(record.population.ptc_second_pass_excluded_sample_count, 1U);
    EXPECT_EQ(record.population.postclean_outlier_excluded_sample_count, 0U);
    EXPECT_EQ(
        record.population.final_excluded_sample_count,
            static_cast<std::size_t>(fixture.projection.flags().array().count()) +
            1U);
    const auto second_pass = std::find_if(
        record.scoped_causes.begin(), record.scoped_causes.end(),
        [](const auto &cause) {
            return cause.authority ==
                "citlali.ptc.second_pass_local_v1";
        });
    ASSERT_NE(second_pass, record.scoped_causes.end());
    EXPECT_EQ(second_pass->scope, "scan_detector_interval");
    EXPECT_EQ(second_pass->affected_count, 1U);
    EXPECT_EQ(second_pass->start_row, second_pass->end_row);
    auto outlier_flags = fixture.projection.flags();
    outlier_flags(eligible[0].first, eligible[0].second) = true;
    auto outlier_record = pipeline::make_native_cohort_scan_provenance_v3(
        fixture.binding, fixture.ledger, fixture.rtc, fixture.prepared,
        fixture.projection, fixture.projection.flags(),
        fixture.projection.flags(), outlier_flags, {});
    EXPECT_EQ(
        outlier_record.population.postclean_outlier_excluded_sample_count,
        1U);
    const auto postclean = std::find_if(
        outlier_record.scoped_causes.begin(),
        outlier_record.scoped_causes.end(),
        [](const auto &cause) {
            return cause.authority ==
                "citlali.ptc.postclean_outlier_policy_v1";
        });
    ASSERT_NE(postclean, outlier_record.scoped_causes.end());
    EXPECT_EQ(postclean->affected_count, 1U);

    auto invalid_final = final_flags;
    invalid_final(eligible[0].first, eligible[0].second) = false;
    EXPECT_THROW(
        pipeline::make_native_cohort_scan_provenance_v3(
            fixture.binding, fixture.ledger, fixture.rtc,
            fixture.prepared, fixture.projection,
            fixture.projection.flags(), ptc_flags,
            invalid_final, {}),
        std::logic_error);
}

TEST(SciAlignNativeProductLineageV2,
     CanonicalSerializationDoesNotScaleWithDetectorSampleCardinality) {
    auto fixture = make_lineage_fixture(false);
    auto lineage = pipeline::NativeCohortObservationLineageV3::create(
        fixture.binding, *fixture.scan->relation_handle(), 1);
    auto reservation = lineage->reserve(std::move(fixture.record));
    reservation.commit();
    auto product = lineage->snapshot_complete();
    const auto baseline = YAML::Dump(
        pipeline::native_cohort_product_provenance_node_v3(product));

    auto &scan = product.scans.front();
    scan.population.row_count = 1000000U;
    scan.population.detector_sample_count =
        scan.population.row_count * scan.population.detector_count;
    scan.population.mapped_valid_sample_count =
        scan.population.detector_sample_count;
    scan.population.mapped_invalid_sample_count = 0;
    scan.population.delivered_flagged_sample_count = 0;
    scan.population.raw_input_flagged_sample_count = 0;
    scan.population.rtc_processing_flagged_sample_count = 0;
    scan.population.learned_rtc_excluded_sample_count = 0;
    scan.population.operation_excluded_sample_count = 0;
    scan.population.apt_excluded_sample_count = 0;
    scan.population.ptc_second_pass_excluded_sample_count = 0;
    scan.population.learned_ptc_excluded_sample_count = 0;
    scan.population.postclean_outlier_excluded_sample_count = 0;
    scan.population.final_excluded_sample_count = 0;
    scan.population.replaced_by_pca_sample_count =
        scan.population.detector_sample_count;
    scan.population.preserved_pca_invalid_sample_count = 0;
    scan.population.preserved_pass_through_sample_count = 0;
    scan.scoped_causes.clear();
    const auto long_observation = YAML::Dump(
        pipeline::native_cohort_product_provenance_node_v3(product));

    EXPECT_LT(baseline.size(), 16384U);
    EXPECT_LT(long_observation.size(), baseline.size() + 128U);
    EXPECT_EQ(long_observation.find("revision_transitions"),
              std::string::npos);
    EXPECT_EQ(long_observation.find("exact_native_support"),
              std::string::npos);
}

TEST(SciAlignNativeProductLineageV2,
     DebugTraceRequiresExplicitSelectionAndHonorsHardRecordBound) {
    auto fixture = make_lineage_fixture();
    pipeline::NativeCohortDebugTraceRequestV1 invalid;
    invalid.enabled = true;
    invalid.max_records = 2;
    EXPECT_THROW(
        pipeline::make_native_cohort_debug_trace_v1(
            fixture.ledger, fixture.prepared, invalid),
        std::invalid_argument);

    pipeline::NativeCohortDebugTraceRequestV1 request;
    request.enabled = true;
    request.max_records = 2;
    request.scan_index = 0;
    const auto trace = pipeline::make_native_cohort_debug_trace_v1(
        fixture.ledger, fixture.prepared, request);
    EXPECT_EQ(trace.records.size(), 2U);
    EXPECT_GT(trace.matching_record_count, trace.records.size());
    EXPECT_TRUE(trace.truncated);
    const auto node = pipeline::native_cohort_debug_trace_node_v1(trace);
    EXPECT_EQ(node["artifact_class"].as<std::string>(),
              "diagnostic_not_canonical");
    EXPECT_FALSE(node["retention_required"].as<bool>());
    EXPECT_EQ(node["selection"]["scan_index"].as<std::int64_t>(), 0);
    auto overfilled = trace;
    overfilled.records.push_back(trace.records.front());
    EXPECT_THROW(
        pipeline::native_cohort_debug_trace_node_v1(overfilled),
        std::logic_error);
}

}  // namespace
