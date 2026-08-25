#include "sci_align_native_gap_fixture.h"

#include <citlali/core/engine/engine.h>
#include <citlali/core/pipeline/native_consumer_execution.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>

namespace {

namespace fixture = citlali::test_support::sci_align;
namespace pipeline = citlali::pipeline;

void configure_native_test_engine(Engine &engine,
                                  const pipeline::NativeMeasuredDetectorScan &scan) {
    engine.logger = spdlog::default_logger();
    const auto n_dets = static_cast<Eigen::Index>(scan.detector_count());
    engine.calib.n_dets = n_dets;
    const auto add = [&](const std::string &name, double value) {
        engine.calib.apt[name] = Eigen::VectorXd::Constant(n_dets, value);
    };
    add("uid", 0.0);
    add("nw", 0.0);
    add("array", 0.0);
    add("fg", 0.0);
    add("flag", 0.0);
    add("a_fwhm", 10.0);
    add("b_fwhm", 10.0);
    add("angle", 0.0);
    add("x_t", 0.0);
    add("y_t", 0.0);
    add("sens", 1.0);
    add("flxscale", 1.0);
    add("duplicate_tone", 0.0);
    for (Eigen::Index detector = 0; detector < n_dets; ++detector) {
        const auto &binding = scan.binding(detector);
        engine.calib.apt["uid"](detector) =
            static_cast<double>(binding.output_uid);
        engine.calib.apt["nw"](detector) =
            static_cast<double>(binding.network_id);
        engine.calib.apt["array"](detector) =
            static_cast<double>(binding.array);
        engine.calib.apt["flag"](detector) =
            static_cast<double>(binding.apt_flag.value_or(1));
    }
    engine.calib.flux_conversion_factor = Eigen::VectorXd::Ones(n_dets);

    engine.telescope.fsmp = 100.0;
    engine.telescope.d_fsmp = 100.0;
    engine.telescope.tau_225_GHz = 0.0;
    engine.telescope.pixel_axes = "altaz";
    engine.omb.pixel_size_rad = 1.0e-5;
    engine.typed_config.mapmaking.grouping =
        citlali::config::MapGrouping::detector;

    engine.rtcproc.run_timestream = true;
    engine.rtcproc.run_pointing = false;
    engine.rtcproc.run_polarization = false;
    engine.rtcproc.run_kernel = false;
    engine.rtcproc.run_despike = false;
    engine.rtcproc.run_tod_filter = false;
    engine.rtcproc.run_tod_notch = false;
    engine.rtcproc.run_tod_iir_highpass = false;
    engine.rtcproc.run_downsample = false;
    engine.rtcproc.run_calibrate = false;
    engine.rtcproc.run_extinction = false;
    engine.rtcproc.line_audit.enabled = false;
    engine.rtcproc.filter_edge_guard.enabled = false;
    engine.rtcproc.altaz_destripe.enabled = false;
    engine.rtcproc.network_step_mask.enabled = false;
    engine.rtcproc.impulsive_capture.enabled = false;
    engine.rtcproc.impulsive_coincidence.enabled = false;
    engine.rtcproc.coherent_iq_mode_observer_enabled = false;
    engine.rtcproc.logger = spdlog::default_logger();

    engine.ptcproc.run_clean = false;
    engine.ptcproc.run_fruit_loops = false;
    engine.ptcproc.mask_radius_arcsec = 0.0;
    engine.ptcproc.source_mask_radius_arcsec = 0.0;
    engine.ptcproc.second_pass_local.enabled = false;
    engine.ptcproc.weight_validation.enabled = false;
    engine.ptcproc.weighting_type = "const";
    engine.ptcproc.med_weight_factor = 0.0;
    engine.ptcproc.lower_weight_factor = 0.0;
    engine.ptcproc.upper_weight_factor = 0.0;
    engine.ptcproc.logger = spdlog::default_logger();
    engine.ptcproc.cleaner.logger = spdlog::default_logger();
}

void configure_native_learning_apply_test(Engine &engine,
                                          const std::string &obsnum,
                                          int scan_index) {
    ReductionLearningState::Options options;
    options.enabled = true;
    options.diagnostics_enabled = true;
    options.learn_iters = 1;
    options.apply_start_iter = 1;
    options.apply_sample_masks_enabled = true;
    options.apply_max_new_flagged_fraction = 1.0;
    options.busy_detector_exclusion_enabled = true;
    options.scan_network_pathology_enabled = true;
    options.scan_network_pathology_apply_pre_rtc = false;
    options.scan_network_pathology_apply_pre_ptc = false;
    options.scan_network_pathology_apply_pre_mapmaking = true;
    options.scan_network_pathology_max_new_flagged_fraction = 1.0;
    engine.learning.configure(options);
    engine.observation_identity.obsnum = obsnum;

    engine.learning.begin_iteration(0, false, "science");
    const auto add_sample_mask = [&](int uid, long long start,
                                     long long stop, bool pre_rtc,
                                     int record_iter) {
        ReductionLearningState::LearnedSampleMask record;
        record.obsnum = obsnum;
        record.producer = "test";
        record.reason = "prior_iteration_pathology";
        record.iter = record_iter;
        record.scan = scan_index;
        record.uid = uid;
        record.apply_pre_rtc = pre_rtc;
        if (pre_rtc) {
            record.raw_start = start;
            record.raw_stop = stop;
        }
        else {
            record.ptc_start = start;
            record.ptc_stop = stop;
        }
        engine.learning.record_learned_sample_mask(std::move(record));
    };
    // The iter=1 records are deliberately present in effective state but must
    // not be applied to iteration 1.  Learning is strictly prior-iteration.
    add_sample_mask(0, 0, 0, true, 0);
    add_sample_mask(0, 3, 3, true, 1);
    add_sample_mask(1, 2, 2, false, 0);
    add_sample_mask(1, 0, 0, false, 1);

    ReductionLearningState::DetectorPenalty network;
    network.obsnum = obsnum;
    network.producer = "ptc_second_pass";
    network.reason = "busy_network_pathology";
    network.iter = 0;
    network.scan = scan_index;
    network.uid = -1;
    network.nw = 7;
    network.array = 0;
    network.factor = 0.0;
    network.scan_local = true;
    engine.learning.record_detector_penalty(std::move(network));

    engine.iteration.fruit_iter = 1;
    engine.learning.begin_iteration(1, true, "science");
}

TEST(SciAlignNativeConsumerExecution,
     GapFixtureRunsEstablishedBodiesAndProducesCompleteMapCandidate) {
    auto loaded = fixture::load_native_gap_fixture_v1();
    loaded.scan_index = 0;
    auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    Engine engine;
    configure_native_test_engine(engine, *scan);
    pipeline::raw_time_chunk_config(engine).kernel.enabled = true;
    pipeline::raw_time_chunk_config(engine).kernel.type = "gaussian";
    engine.rtcproc.run_kernel = true;
    engine.rtcproc.kernel.type = "gaussian";
    engine.rtcproc.kernel.fwhm_rad = 0.0;
    engine.rtcproc.kernel.sigma_rad = 0.0;
    engine.rtcproc.kernel.map_grouping = "detector";

    timestream::TCData<timestream::TCDataKind::RTC, Eigen::MatrixXd>
        source;
    source.native_runtime =
        std::make_shared<pipeline::NativeScanRuntimeState>(scan);
    source.noise.data.resize(0, 0);
    Eigen::VectorXI map_indices(
        static_cast<Eigen::Index>(scan->detector_count()));
    for (Eigen::Index detector = 0; detector < map_indices.size();
         ++detector) {
        map_indices(detector) = detector;
    }

    auto prepared = pipeline::prepare_native_consumer_map_scan(
        engine, source, map_indices);
    ASSERT_TRUE(prepared.runtime->rtc.has_value());
    ASSERT_TRUE(prepared.runtime->ptc_prepared.has_value());
    ASSERT_TRUE(prepared.runtime->science_projection.has_value());
    ASSERT_TRUE(prepared.runtime->jinc_processing_trace.has_value());
    EXPECT_NO_THROW(
        pipeline::native_jinc_processing_scan_trace_digest_v2(
            *prepared.runtime->jinc_processing_trace));
    EXPECT_EQ(
        prepared.ptcdata.scans.data.rows(),
        prepared.runtime->science_projection->row_count());
    EXPECT_EQ(
        prepared.ptcdata.scans.data.cols(),
        static_cast<Eigen::Index>(scan->detector_count()));
    EXPECT_TRUE(prepared.ptcdata.scans.data.array().isFinite().all());
    EXPECT_EQ(prepared.ptcdata.kernel.data.rows(),
              prepared.ptcdata.scans.data.rows());
    EXPECT_EQ(prepared.ptcdata.kernel.data.cols(),
              prepared.ptcdata.scans.data.cols());
    EXPECT_TRUE(prepared.ptcdata.kernel.data.array().isFinite().all());
    EXPECT_TRUE(prepared.ptcdata.weights.data.array().isFinite().all());
    EXPECT_TRUE((prepared.ptcdata.weights.data.array() > 0.0).all());

    const auto binding =
        pipeline::make_native_cohort_observation_binding_v2(
            0, *scan->relation_handle(), scan->carriers_handle());
    auto record = pipeline::make_native_cohort_scan_provenance_v3(
        binding, prepared.runtime->ledger(), *prepared.runtime->rtc,
        *prepared.runtime->ptc_prepared,
        *prepared.runtime->science_projection,
        *prepared.runtime->ptc_preclean_flags,
        *prepared.runtime->ptc_flags,
        prepared.ptcdata.flags.data,
        {true, "naive", "urn:citlali:test:native-execution",
         "sha256:test-native-execution-product",
         "sha256:test-native-execution-weights",
         static_cast<std::size_t>(prepared.ptcdata.weights.data.size()),
         0});
    auto lineage = pipeline::NativeCohortObservationLineageV3::create(
        binding, *scan->relation_handle(), 1);
    auto reservation = lineage->reserve(std::move(record));
    reservation.commit();
    EXPECT_NO_THROW(lineage->snapshot_complete());
}

TEST(SciAlignNativeConsumerExecution,
     GlobalRtcCohortSupportsInterleavedDetectorNetworks) {
    auto loaded = fixture::load_native_gap_fixture_v1();
    loaded.scan_index = 0;
    auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    Engine engine;
    configure_native_test_engine(engine, *scan);
    pipeline::NativeScanRuntimeState runtime{scan};

    pipeline::raw_time_chunk_config(engine).line_audit.enabled = true;
    const auto rtc =
        pipeline::run_native_rtc_numerical_bodies(engine, runtime);
    EXPECT_EQ(rtc.output_row_count(), 8U);
    EXPECT_TRUE(runtime.jinc_processing_trace.has_value());
    EXPECT_EQ(runtime.fcf.size(), 4);
    EXPECT_FALSE(runtime.ledger().last_operation().has_value());
}

TEST(SciAlignNativeConsumerExecution,
     ExtinctionFcfContractUsesExactRunSampleCounts) {
    pipeline::NativeDetectorRunFcfContract contract(3, true);
    contract.observe({0, 2}, (Eigen::Vector2d{} << 2.0, 6.0).finished(), 2);
    contract.observe({0, 1, 2},
                     (Eigen::Vector3d{} << 4.0, 8.0, 10.0).finished(), 3);
    const auto result = contract.finish();
    EXPECT_DOUBLE_EQ(result(0), 3.2);
    EXPECT_DOUBLE_EQ(result(1), 8.0);
    EXPECT_DOUBLE_EQ(result(2), 8.4);
}

TEST(SciAlignNativeConsumerExecution,
     NonExtinctionFcfContractRetainsExactEquality) {
    pipeline::NativeDetectorRunFcfContract contract(1, false);
    contract.observe({0}, Eigen::VectorXd::Constant(1, 2.0), 2);
    contract.observe({0}, Eigen::VectorXd::Constant(1, 2.0), 3);
    EXPECT_DOUBLE_EQ(contract.finish()(0), 2.0);

    pipeline::NativeDetectorRunFcfContract mismatch(1, false);
    mismatch.observe({0}, Eigen::VectorXd::Constant(1, 2.0), 2);
    EXPECT_THROW(
        mismatch.observe({0}, Eigen::VectorXd::Constant(1, 2.5), 3),
        std::logic_error);
}

TEST(SciAlignNativeConsumerExecution,
     ObservationPreflightRejectsLegacyTodOutputAndInvalidOuterContext) {
    auto loaded = fixture::load_native_gap_fixture_v1();
    auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    Engine engine;
    configure_native_test_engine(engine, *scan);
    engine.telescope.scan_indices.resize(4, 1);
    engine.telescope.scan_indices.col(0) << 4, 11, 4, 11;

    EXPECT_NO_THROW(
        pipeline::require_supported_native_consumer_observation(engine));

    engine.typed_config.timestream.output.type =
        citlali::config::TodOutputType::rtc;
    EXPECT_THROW(
        pipeline::require_supported_native_consumer_observation(engine),
        std::logic_error);

    engine.typed_config.timestream.output.type =
        citlali::config::TodOutputType::none;
    engine.telescope.scan_indices.col(0) << 4, 11, 2, 13;
    EXPECT_NO_THROW(
        pipeline::require_supported_native_consumer_observation(engine));

    engine.telescope.scan_indices.col(0) << 4, 11, 5, 13;
    EXPECT_THROW(
        pipeline::require_supported_native_consumer_observation(engine),
        std::logic_error);
}

TEST(SciAlignNativeConsumerExecution,
     ObservationPreflightAdmitsCohortOutliersAndRejectsUnsupportedState) {
    auto loaded = fixture::load_native_gap_fixture_v1();
    auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    Engine engine;
    configure_native_test_engine(engine, *scan);
    engine.telescope.scan_indices.resize(4, 1);
    engine.telescope.scan_indices.col(0) << 4, 11, 4, 11;

    pipeline::raw_time_chunk_config(engine)
        .flagging.lower_tod_inv_var_factor = 0.5;
    EXPECT_NO_THROW(
        pipeline::require_supported_native_consumer_observation(engine));
    pipeline::raw_time_chunk_config(engine)
        .flagging.lower_tod_inv_var_factor = 0.0;

    pipeline::noise_config(engine).enabled = true;
    EXPECT_NO_THROW(
        pipeline::require_supported_native_consumer_observation(engine));
    pipeline::noise_config(engine).enabled = false;

    pipeline::fruit_loops_config(engine).enabled = true;
    EXPECT_NO_THROW(
        pipeline::require_supported_native_consumer_observation(engine));
    pipeline::fruit_loops_config(engine).enabled = false;

    engine.calib.apt["duplicate_tone"](0) = 0.5;
    EXPECT_THROW(
        pipeline::require_supported_native_consumer_observation(engine),
        std::logic_error);
}

TEST(SciAlignNativeConsumerExecution,
     FruitLoopFeedbackPreservesMatureOrderAndBoundedRealization) {
    auto loaded = fixture::load_native_gap_fixture_v1();
    loaded.scan_index = 0;
    auto reference_scan =
        fixture::materialize_native_gap_measured_scan(loaded);
    auto fruit_scan =
        fixture::materialize_native_gap_measured_scan(loaded);
    Engine reference_engine;
    configure_native_test_engine(reference_engine, *reference_scan);
    Engine fruit_engine;
    configure_native_test_engine(fruit_engine, *fruit_scan);
    std::set<std::int64_t> arrays;
    for (Eigen::Index detector = 0;
         detector < static_cast<Eigen::Index>(fruit_scan->detector_count());
         ++detector) {
        arrays.insert(fruit_scan->binding(detector).array);
    }
    fruit_engine.calib.arrays.resize(
        static_cast<Eigen::Index>(arrays.size()));
    Eigen::Index array_index = 0;
    for (const auto array : arrays) {
        fruit_engine.calib.arrays(array_index++) = array;
    }

    auto &fruit = pipeline::fruit_loops_config(fruit_engine);
    fruit.enabled = true;
    fruit.recompute_weights_after_addback = false;
    fruit_engine.iteration.fruit_iter = 1;
    fruit_engine.ptcproc.fruit_loops_interp_mode = "nearest";
    fruit_engine.ptcproc.fruit_mode = "upper";
    fruit_engine.ptcproc.fruit_loops_flux = Eigen::VectorXd::Constant(
        fruit_engine.calib.arrays.size(), 0.1);
    fruit_engine.ptcproc.fruit_loops_kernel_feedback_enabled = false;
    auto &model = fruit_engine.ptcproc.tod_mb;
    model.n_rows = 21;
    model.n_cols = 21;
    model.pixel_size_rad = 1.0;
    model.signal.assign(
        fruit_scan->detector_count(),
        Eigen::MatrixXd::Constant(21, 21, 0.5));

    Eigen::VectorXI map_indices(
        static_cast<Eigen::Index>(fruit_scan->detector_count()));
    for (Eigen::Index detector = 0; detector < map_indices.size();
         ++detector) {
        map_indices(detector) = detector;
    }
    timestream::TCData<timestream::TCDataKind::RTC, Eigen::MatrixXd>
        reference_source;
    reference_source.native_runtime =
        std::make_shared<pipeline::NativeScanRuntimeState>(reference_scan);
    timestream::TCData<timestream::TCDataKind::RTC, Eigen::MatrixXd>
        fruit_source;
    fruit_source.native_runtime =
        std::make_shared<pipeline::NativeScanRuntimeState>(fruit_scan);

    const auto reference = pipeline::prepare_native_consumer_map_scan(
        reference_engine, reference_source, map_indices);
    const auto realized = pipeline::prepare_native_consumer_map_scan(
        fruit_engine, fruit_source, map_indices);

    ASSERT_TRUE(realized.runtime->fruit_loop_feedback.has_value());
    const auto &summary = *realized.runtime->fruit_loop_feedback;
    EXPECT_TRUE(summary.enabled);
    EXPECT_TRUE(summary.source_model_available);
    EXPECT_FALSE(summary.noise_map_pass_applied);
    EXPECT_TRUE(summary.keep_source_subtracted_weights);
    EXPECT_EQ(summary.iteration, 1);
    EXPECT_EQ(summary.model_map_count, fruit_scan->detector_count());
    EXPECT_GT(summary.subtraction_sample_count, 0U);
    EXPECT_EQ(summary.addback_sample_count,
              summary.subtraction_sample_count);
    EXPECT_EQ(summary.interpolation_mode, "nearest");
    EXPECT_EQ(summary.support_authority,
              pipeline::native_fruit_loop_feedback_authority_v3);
    EXPECT_TRUE(realized.runtime->map_projection.has_value());
    EXPECT_FALSE(realized.ptcdata.scans.data.isApprox(
        reference.ptcdata.scans.data, 1.0e-14));
    EXPECT_TRUE(realized.runtime->map_projection->values().isApprox(
        realized.ptcdata.scans.data, 0.0));
    EXPECT_FALSE(realized.runtime->science_projection->values().isApprox(
        realized.ptcdata.scans.data, 0.0));
    EXPECT_NO_THROW(summary.validate());
}

TEST(SciAlignNativeConsumerExecution,
     ObservationPreflightAdmitsEnabledLearningState) {
    auto loaded = fixture::load_native_gap_fixture_v1();
    auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    Engine engine;
    configure_native_test_engine(engine, *scan);
    engine.telescope.scan_indices.resize(4, 1);
    engine.telescope.scan_indices.col(0) << 4, 11, 4, 11;
    configure_native_learning_apply_test(
        engine, std::to_string(loaded.scope.observation),
        static_cast<int>(loaded.scan_index));

    EXPECT_NO_THROW(
        pipeline::require_supported_native_consumer_observation(engine));
}

TEST(SciAlignNativeConsumerExecution,
     NoiseAssignmentsAreExactBoundedAndProjectionOwned) {
    auto loaded = fixture::load_native_gap_fixture_v1();
    loaded.scan_index = 0;
    auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    Engine engine;
    configure_native_test_engine(engine, *scan);
    pipeline::noise_config(engine).enabled = true;
    pipeline::noise_config(engine).n_noise_maps = 2;
    pipeline::noise_config(engine).randomize_dets = true;

    timestream::TCData<timestream::TCDataKind::RTC, Eigen::MatrixXd>
        source;
    source.native_runtime =
        std::make_shared<pipeline::NativeScanRuntimeState>(scan);
    source.noise.data.resize(2, 4);
    source.noise.data << 1, -1, 1, -1,
                        -1, -1, 1, 1;
    Eigen::VectorXI map_indices(4);
    map_indices << 0, 1, 2, 3;

    auto prepared = pipeline::prepare_native_consumer_map_scan(
        engine, source, map_indices);
    ASSERT_TRUE(prepared.runtime->noise_assignment.has_value());
    const auto &noise = *prepared.runtime->noise_assignment;
    EXPECT_TRUE(noise.enabled);
    EXPECT_TRUE(noise.randomize_detectors);
    EXPECT_EQ(noise.realization_count, 2U);
    EXPECT_EQ(noise.assignment_column_count, 4U);
    EXPECT_EQ(noise.assignment_count, 8U);
    EXPECT_EQ(noise.positive_sign_count, 4U);
    EXPECT_EQ(noise.negative_sign_count, 4U);
    EXPECT_FALSE(noise.assignment_digest.empty());
    EXPECT_EQ(noise.support_authority,
              pipeline::native_noise_support_authority_v3);

    auto bad_signs = source.noise.data;
    bad_signs(0, 0) = 0;
    EXPECT_THROW(
        pipeline::make_native_noise_assignment_summary_v3(
            bad_signs, true, true, 2, 4),
        std::logic_error);

    pipeline::NativeCohortMapPublicationRequestV3 map_request;
    map_request.mapmaking_enabled = true;
    map_request.method = "naive";
    map_request.product_occurrence = "urn:citlali:test:native-noise";
    map_request.product_identity_digest =
        "sha256:test-native-noise-product";
    map_request.eligible_weight_digest =
        mapmaking::jinc_matrix_digest(prepared.ptcdata.weights.data);
    map_request.noise_assignment = noise;
    for (Eigen::Index detector = 0;
         detector < prepared.ptcdata.weights.data.size(); ++detector) {
        if (prepared.ptcdata.weights.data(detector) > 0.0) {
            ++map_request.positive_weight_detector_count;
        }
        else {
            ++map_request.zero_weight_detector_count;
            map_request.zero_weight_detector_columns.push_back(detector);
        }
    }
    const auto binding =
        pipeline::make_native_cohort_observation_binding_v2(
            0, *scan->relation_handle(), scan->carriers_handle());
    const auto record = pipeline::make_native_cohort_scan_provenance_v3(
        binding, prepared.runtime->ledger(), *prepared.runtime->rtc,
        *prepared.runtime->ptc_prepared,
        *prepared.runtime->science_projection,
        *prepared.runtime->ptc_preclean_flags,
        *prepared.runtime->ptc_flags,
        prepared.ptcdata.flags.data, std::move(map_request));
    EXPECT_EQ(record.map_occurrence.noise_assignment.assignment_digest,
              noise.assignment_digest);
}

TEST(SciAlignNativeConsumerExecution,
     AppliesOnlyPriorLearningAcrossRtcPtcAndMapWithBoundedLineage) {
    auto loaded = fixture::load_native_gap_fixture_v1();
    loaded.network(0).original_flag_bits(0, 0) = 0;
    loaded.network(0).original_flag_bits(3, 0) = 0;
    loaded.network(0).original_flag_bits(0, 1) = 0;
    loaded.network(0).original_flag_bits(3, 1) = 0;
    auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    Engine engine;
    configure_native_test_engine(engine, *scan);
    configure_native_learning_apply_test(
        engine, std::to_string(loaded.scope.observation),
        static_cast<int>(loaded.scan_index));

    timestream::TCData<timestream::TCDataKind::RTC, Eigen::MatrixXd>
        source;
    source.native_runtime =
        std::make_shared<pipeline::NativeScanRuntimeState>(scan);
    Eigen::VectorXI map_indices(
        static_cast<Eigen::Index>(scan->detector_count()));
    for (Eigen::Index detector = 0; detector < map_indices.size();
         ++detector) {
        map_indices(detector) = detector;
    }

    auto prepared = pipeline::prepare_native_consumer_map_scan(
        engine, source, map_indices);
    ASSERT_TRUE(prepared.runtime->rtc.has_value());
    ASSERT_TRUE(prepared.runtime->ptc_prepared.has_value());
    ASSERT_TRUE(prepared.runtime->science_projection.has_value());
    ASSERT_TRUE(prepared.runtime->ptc_preclean_flags.has_value());
    ASSERT_TRUE(prepared.runtime->ptc_flags.has_value());

    // Detector column 1 has uid=0.  Common slot 0 is from iteration 0 and is
    // learned; common slot 1 is tagged iteration 1 and must remain untouched.
    bool learned_rtc_slot_zero = false;
    bool learned_rtc_slot_three = false;
    for (const auto &run : prepared.runtime->rtc->runs) {
        for (Eigen::Index row = 0; row < run.ored_flag_bits.rows(); ++row) {
            for (Eigen::Index local = 0;
                 local < run.ored_flag_bits.cols(); ++local) {
                if (run.input.detector_columns.at(
                        static_cast<std::size_t>(local)) != 1) {
                    continue;
                }
                const auto &support = run.support.at(
                    static_cast<std::size_t>(row));
                const bool learned =
                    (run.ored_flag_bits(row, local) &
                     pipeline::native_learned_rtc_flag_bit_v2) != 0;
                ASSERT_EQ(support.exact_common_slots.size(), 1U);
                if (support.exact_common_slots.front() == 0) {
                    learned_rtc_slot_zero = learned;
                }
                if (support.exact_common_slots.front() == 3) {
                    learned_rtc_slot_three = learned;
                }
            }
        }
    }
    EXPECT_TRUE(learned_rtc_slot_zero);
    EXPECT_FALSE(learned_rtc_slot_three);

    // PTC rows are concatenated segment-locally: rows 0-1 precede the native
    // gap and rows 2-3 follow it.  Only the prior-iteration row 2 mask applies.
    EXPECT_TRUE((*prepared.runtime->ptc_preclean_flags)(2, 3));
    EXPECT_FALSE((*prepared.runtime->ptc_preclean_flags)(0, 3));
    EXPECT_TRUE((*prepared.runtime->ptc_flags)(2, 3));
    EXPECT_FALSE((*prepared.runtime->ptc_flags)(0, 3));

    const std::vector<pipeline::TimestreamDetectorColumn>
        expected_learned_map_columns{0, 2};
    EXPECT_EQ(
        prepared.runtime->learned_map_zero_weight_detector_columns,
        expected_learned_map_columns);
    EXPECT_DOUBLE_EQ(prepared.ptcdata.weights.data(0), 0.0);
    EXPECT_DOUBLE_EQ(prepared.ptcdata.weights.data(2), 0.0);
    int matched_pre_rtc_applications = 0;
    for (const auto &summary : engine.learning.learned_mask_applications) {
        if (summary.stage == "pre_rtc" && summary.matched_records > 0) {
            ++matched_pre_rtc_applications;
        }
    }
    EXPECT_EQ(matched_pre_rtc_applications, 1);

    pipeline::NativeCohortMapPublicationRequestV3 map_request;
    map_request.mapmaking_enabled = true;
    map_request.method = "naive";
    map_request.product_occurrence =
        "urn:citlali:test:native-learning";
    map_request.product_identity_digest =
        "sha256:test-native-learning-product";
    map_request.eligible_weight_digest =
        mapmaking::jinc_matrix_digest(prepared.ptcdata.weights.data);
    for (Eigen::Index detector = 0;
         detector < prepared.ptcdata.weights.data.size(); ++detector) {
        if (prepared.ptcdata.weights.data(detector) > 0.0) {
            ++map_request.positive_weight_detector_count;
        }
        else {
            ++map_request.zero_weight_detector_count;
            map_request.zero_weight_detector_columns.push_back(detector);
        }
    }
    map_request.learned_map_zero_weight_detector_columns =
        prepared.runtime->learned_map_zero_weight_detector_columns;

    const auto binding =
        pipeline::make_native_cohort_observation_binding_v2(
            0, *scan->relation_handle(), scan->carriers_handle());
    auto record = pipeline::make_native_cohort_scan_provenance_v3(
        binding, prepared.runtime->ledger(), *prepared.runtime->rtc,
        *prepared.runtime->ptc_prepared,
        *prepared.runtime->science_projection,
        *prepared.runtime->ptc_preclean_flags,
        *prepared.runtime->ptc_flags,
        prepared.ptcdata.flags.data, std::move(map_request));
    EXPECT_EQ(record.population.learned_rtc_excluded_sample_count, 1U);
    EXPECT_EQ(record.population.learned_ptc_excluded_sample_count, 1U);

    std::set<std::string> learned_authorities;
    for (const auto &cause : record.scoped_causes) {
        if (cause.authority.rfind("citlali.learning.", 0) == 0) {
            learned_authorities.insert(cause.authority);
        }
        if (cause.authority ==
            "citlali.learning.native_ptc_application_v1") {
            EXPECT_EQ(cause.scope, "scan_detector_interval");
            EXPECT_EQ(cause.start_row, 2U);
            EXPECT_EQ(cause.end_row, 2U);
            EXPECT_EQ(cause.affected_count, 1U);
            EXPECT_EQ(cause.detector_columns,
                      std::vector<pipeline::TimestreamDetectorColumn>{3});
        }
    }
    EXPECT_EQ(
        learned_authorities,
        (std::set<std::string>{
            "citlali.learning.native_map_application_v1",
            "citlali.learning.native_ptc_application_v1",
            "citlali.learning.native_rtc_application_v1"}));
}

TEST(SciAlignNativeConsumerExecution,
     DuplicateTonesUseExactPtcExclusionAndFlagEveryMappedSample) {
    auto loaded = fixture::load_native_gap_fixture_v1();
    loaded.scan_index = 0;
    auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    Engine engine;
    configure_native_test_engine(engine, *scan);
    engine.ptcproc.weighting_type = "full";
    engine.calib.apt["duplicate_tone"](0) = 1.0;
    engine.telescope.scan_indices.resize(4, 1);
    engine.telescope.scan_indices.col(0) << 4, 11, 4, 11;

    EXPECT_NO_THROW(
        pipeline::require_supported_native_consumer_observation(engine));

    timestream::TCData<timestream::TCDataKind::RTC, Eigen::MatrixXd>
        source;
    source.native_runtime =
        std::make_shared<pipeline::NativeScanRuntimeState>(scan);
    Eigen::VectorXI map_indices(
        static_cast<Eigen::Index>(scan->detector_count()));
    for (Eigen::Index detector = 0; detector < map_indices.size();
         ++detector) {
        map_indices(detector) = detector;
    }

    auto prepared = pipeline::prepare_native_consumer_map_scan(
        engine, source, map_indices);
    ASSERT_TRUE(prepared.runtime->ptc_prepared.has_value());
    EXPECT_TRUE(prepared.ptcdata.flags.data.col(0).array().all());
    EXPECT_DOUBLE_EQ(prepared.ptcdata.weights.data(0), 0.0);
    for (const auto &group : prepared.runtime->ptc_prepared->groups()) {
        for (Eigen::Index local = 0;
             local < group.detector_count(); ++local) {
            for (Eigen::Index row = 0; row < group.slot_count(); ++row) {
                const auto &cell = group.cell(row, local);
                const auto detector = group.detector_columns().at(
                    static_cast<std::size_t>(local));
                EXPECT_EQ(
                    cell.operation_exclusion_bits,
                    detector == 0
                        ? pipeline::native_duplicate_tone_exclusion_bit_v2
                        : 0U);
            }
        }
    }
}

TEST(SciAlignNativeConsumerExecution,
     TypedAptExclusionPreservesFiniteMeasuredRtcPayloadAndFlagsIt) {
    Eigen::MatrixXd measured(3, 2);
    measured << 1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0;
    Eigen::MatrixXd processed = measured;
    processed.col(0).setConstant(
        std::numeric_limits<double>::quiet_NaN());
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> flags(3, 2);
    flags.setConstant(false);
    double excluded_fcf = std::numeric_limits<double>::quiet_NaN();

    EXPECT_NO_THROW(pipeline::reconcile_native_rtc_detector_result(
        measured, 0, std::nullopt, processed, flags, excluded_fcf));
    EXPECT_TRUE((processed.col(0).array() ==
                 measured.col(0).array()).all());
    EXPECT_TRUE(flags.col(0).array().all());
    EXPECT_DOUBLE_EQ(excluded_fcf, 1.0);

    auto invalid_eligible = processed;
    invalid_eligible(0, 1) =
        std::numeric_limits<double>::quiet_NaN();
    double eligible_fcf = 1.0;
    EXPECT_THROW(pipeline::reconcile_native_rtc_detector_result(
        measured, 1, std::optional<std::int64_t>{0}, invalid_eligible,
        flags, eligible_fcf), std::logic_error);
}

}  // namespace
