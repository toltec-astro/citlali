#include "sci_align_native_gap_fixture.h"

#include <citlali/core/engine/engine.h>
#include <citlali/core/pipeline/native_consumer_execution.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>

namespace {

namespace fixture = citlali::test_support::sci_align;
namespace pipeline = citlali::pipeline;

void configure_native_test_engine(Engine &engine,
                                  const pipeline::NativeMeasuredDetectorScan &scan) {
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

TEST(SciAlignNativeConsumerExecution,
     GapFixtureRunsEstablishedBodiesAndProducesCompleteMapCandidate) {
    auto loaded = fixture::load_native_gap_fixture_v1();
    loaded.scan_index = 0;
    auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    Engine engine;
    configure_native_test_engine(engine, *scan);

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
    EXPECT_TRUE(prepared.ptcdata.weights.data.array().isFinite().all());
    EXPECT_TRUE((prepared.ptcdata.weights.data.array() > 0.0).all());

    const auto binding =
        pipeline::make_native_cohort_observation_binding_v2(
            0, *scan->relation_handle(), scan->carriers_handle());
    auto record = pipeline::make_native_cohort_scan_provenance_v3(
        binding, prepared.runtime->ledger(), *prepared.runtime->rtc,
        *prepared.runtime->ptc_prepared,
        *prepared.runtime->science_projection,
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
     UnsupportedKernelAndExtinctionFailBeforeLedgerMutation) {
    auto loaded = fixture::load_native_gap_fixture_v1();
    loaded.scan_index = 0;
    auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    Engine engine;
    configure_native_test_engine(engine, *scan);
    pipeline::NativeScanRuntimeState runtime{scan};

    pipeline::raw_time_chunk_config(engine).kernel.enabled = true;
    EXPECT_THROW(
        pipeline::run_native_rtc_numerical_bodies(engine, runtime),
        std::logic_error);
    EXPECT_FALSE(runtime.ledger().last_operation().has_value());
    pipeline::raw_time_chunk_config(engine).kernel.enabled = false;
    pipeline::raw_time_chunk_config(engine).extinction_correction_enabled =
        true;
    EXPECT_THROW(
        pipeline::run_native_rtc_numerical_bodies(engine, runtime),
        std::logic_error);
    EXPECT_FALSE(runtime.ledger().last_operation().has_value());
}

TEST(SciAlignNativeConsumerExecution,
     ObservationPreflightRejectsLegacyTodOutputAndOuterContext) {
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
    EXPECT_THROW(
        pipeline::require_supported_native_consumer_observation(engine),
        std::logic_error);
}

TEST(SciAlignNativeConsumerExecution,
     ObservationPreflightRejectsUnrepresentedDetectorSelection) {
    auto loaded = fixture::load_native_gap_fixture_v1();
    auto scan = fixture::materialize_native_gap_measured_scan(loaded);
    Engine engine;
    configure_native_test_engine(engine, *scan);
    engine.telescope.scan_indices.resize(4, 1);
    engine.telescope.scan_indices.col(0) << 4, 11, 4, 11;

    pipeline::raw_time_chunk_config(engine)
        .flagging.lower_tod_inv_var_factor = 0.5;
    EXPECT_THROW(
        pipeline::require_supported_native_consumer_observation(engine),
        std::logic_error);
    pipeline::raw_time_chunk_config(engine)
        .flagging.lower_tod_inv_var_factor = 0.0;

    pipeline::noise_config(engine).enabled = true;
    EXPECT_THROW(
        pipeline::require_supported_native_consumer_observation(engine),
        std::logic_error);
    pipeline::noise_config(engine).enabled = false;

    engine.calib.apt["duplicate_tone"](0) = 0.5;
    EXPECT_THROW(
        pipeline::require_supported_native_consumer_observation(engine),
        std::logic_error);
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
