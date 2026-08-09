#include <kids/toltec/toltec.h>

#include <citlali/core/timestream/rtc/rtcproc.h>

#include <gtest/gtest.h>

#include <map>
#include <set>
#include <string>

namespace {

struct TestCalib {
    Eigen::Index n_dets = 2;
    bool run_hwpr = false;
    std::map<std::string, Eigen::VectorXd> apt;
    Eigen::VectorXd flux_conversion_factor = Eigen::VectorXd::Ones(2);
    Eigen::VectorXd arrays;
    Eigen::VectorXd fg;
};

struct TestTelescope {
    double fsmp = 8.0;
    double tau_225_GHz = 0.0;
    std::string pixel_axes = "altaz";
    bool sim_obs = false;
};

using RtcData =
    timestream::TCData<timestream::TCDataKind::RTC, Eigen::MatrixXd>;
using PtcData =
    timestream::TCData<timestream::TCDataKind::PTC, Eigen::MatrixXd>;

TestCalib make_calib(bool include_source_identity = true) {
    TestCalib calib;
    calib.apt["nw"] = Eigen::VectorXd::Zero(2);
    calib.apt["flag"] = Eigen::VectorXd::Zero(2);
    calib.apt["x_t"] = Eigen::VectorXd::Zero(2);
    calib.apt["y_t"] = Eigen::VectorXd::Zero(2);
    calib.apt["flxscale"] = Eigen::VectorXd::Ones(2);
    calib.apt["array"] = Eigen::VectorXd::Zero(2);
    if (include_source_identity) {
        calib.apt["uid"].resize(2);
        calib.apt["uid"] << 101.0, 102.0;
    }
    return calib;
}

RtcData make_rtc_data(Eigen::Index scan_id, bool with_coordinates = false) {
    RtcData data;
    data.index.data = scan_id;
    data.scans.data.resize(9, 2);
    for (Eigen::Index row = 0; row < data.scans.data.rows(); ++row) {
        data.scans.data(row, 0) = static_cast<double>(row);
        data.scans.data(row, 1) = row == 4 ? 1.0 : 0.0;
    }
    data.flags.data.resize(9, 2);
    data.flags.data.setConstant(false);
    data.flags.data(3, 0) = true;
    data.scan_indices.data.resize(4);
    const Eigen::Index absolute_start = scan_id * 9;
    data.scan_indices.data << absolute_start, absolute_start + 8,
                              absolute_start, absolute_start + 8;
    data.tel_data.data["TelTime"] = Eigen::VectorXd::LinSpaced(
        9, static_cast<double>(absolute_start) / 8.0,
        static_cast<double>(absolute_start + 8) / 8.0);
    if (with_coordinates) {
        data.tel_data.data["TelElAct"] = Eigen::VectorXd::Zero(9);
        data.tel_data.data["alt_phys"] =
            Eigen::VectorXd::LinSpaced(9, -1.0e-5, 1.0e-5);
        data.tel_data.data["az_phys"] = Eigen::VectorXd::Zero(9);
        data.pointing_offsets_arcsec.data["az"] =
            Eigen::VectorXd::Zero(9);
        data.pointing_offsets_arcsec.data["alt"] =
            Eigen::VectorXd::Zero(9);
    }
    return data;
}

void configure_minimal_proc(timestream::RTCProc &proc,
                            bool downsample = true) {
    proc.logger = spdlog::default_logger();
    proc.run_timestream = true;
    proc.run_pointing = false;
    proc.run_polarization = false;
    proc.run_kernel = false;
    proc.run_despike = false;
    proc.run_tod_filter = false;
    proc.run_tod_notch = false;
    proc.run_tod_iir_highpass = false;
    proc.run_downsample = downsample;
    proc.run_calibrate = false;
    proc.run_extinction = false;
    proc.downsampler.factor = downsample ? 2 : 1;
    proc.altaz_destripe.enabled = false;
    proc.network_step_mask.enabled = false;
    proc.impulsive_coincidence.enabled = false;
    proc.coherent_iq_mode_observer_enabled = false;
}

PtcData run_one(timestream::RTCProc &proc, Eigen::Index scan_id,
                bool simulated, bool keep_outer = false) {
    auto input = make_rtc_data(scan_id);
    TestCalib calib = make_calib();
    TestTelescope telescope;
    telescope.sim_obs = simulated;
    proc.bind_assigned_grid_segment(
        scan_id, scan_id * 9, scan_id * 9, 9, 9, 2,
        telescope.fsmp, proc.run_downsample ? proc.downsampler.factor : 1,
        simulated, "project:152389");
    PtcData output;
    RtcData outer;
    proc.run(input, output, calib, telescope, 1.0e-5, "nw",
             keep_outer ? &outer : nullptr);
    return output;
}

TEST(RtcPhaseIndependentProductionPaths,
     DirectRtcRunEnforcesInfluenceAndPublishesOuterInnerStages) {
    timestream::RTCProc proc;
    configure_minimal_proc(proc);
    const auto output = run_one(proc, 0, false, true);

    ASSERT_EQ(output.scans.data.rows(), 5);
    ASSERT_EQ(output.flags.data.rows(), 5);
    EXPECT_TRUE(output.flags.data(1, 0));

    const auto stage = proc.phase_independent_stage_for_scan(0);
    EXPECT_EQ(stage.stage_view, "inner");
    EXPECT_EQ(stage.output_sample_count, 5);
    EXPECT_EQ(stage.downsample_factor, 2);
    EXPECT_EQ(stage.phase_label, "assigned_index_phase_zero");
    EXPECT_EQ(stage.physical_event_semantics, "unavailable");
    EXPECT_EQ(stage.representative_assigned_time_rule,
              "phase_zero_first_cell_compatibility_value");
    EXPECT_FALSE(stage.representative_assigned_time_hex.empty());
    EXPECT_FALSE(stage.assigned_time_values_digest.empty());
    EXPECT_GT(stage.influenced_sample_count, 0u);
    EXPECT_FALSE(stage.complete_response_available);

    const auto snapshot = proc.snapshot_phase_independent_state();
    ASSERT_EQ(snapshot.stages.size(), 2u);
    std::set<std::string> views;
    for (const auto &value : snapshot.stages) {
        views.insert(value.stage_view);
    }
    EXPECT_EQ(views, (std::set<std::string>{"inner", "outer"}));
}

TEST(RtcPhaseIndependentProductionPaths,
     MissingAssignedGridContextFailsBeforeRtcExecution) {
    timestream::RTCProc proc;
    configure_minimal_proc(proc, false);
    auto input = make_rtc_data(0);
    PtcData output;
    TestCalib calib = make_calib();
    TestTelescope telescope;
    EXPECT_THROW(
        proc.run(input, output, calib, telescope, 1.0e-5, "nw"),
        std::logic_error);
}

TEST(RtcPhaseIndependentProductionPaths,
     SourceMaskAdmissionValidatesFrameCoordinatesRadiusShapeAndIdentity) {
    auto data = make_rtc_data(0, true);
    auto calib = make_calib();

    const auto valid = timestream::admit_rtc_source_mask(
        data, calib.apt, "altaz", "nw", "map_center_radius", 30.0);
    ASSERT_TRUE(valid.admitted());
    ASSERT_FALSE(valid.identity.empty());
    EXPECT_EQ(valid.frame, "altaz");

    const auto invalid_frame = timestream::admit_rtc_source_mask(
        data, calib.apt, "unknown", "nw", "map_center_radius", 30.0);
    EXPECT_EQ(invalid_frame.status,
              timestream::RTCSourceMaskAdmissionStatus::unavailable_frame);

    const auto invalid_radius = timestream::admit_rtc_source_mask(
        data, calib.apt, "altaz", "nw", "map_center_radius", 0.0);
    EXPECT_EQ(invalid_radius.status,
              timestream::RTCSourceMaskAdmissionStatus::unavailable_radius);

    auto missing_coordinate = data;
    missing_coordinate.tel_data.data.erase("az_phys");
    const auto invalid_coordinate = timestream::admit_rtc_source_mask(
        missing_coordinate, calib.apt, "altaz", "nw",
        "map_center_radius", 30.0);
    EXPECT_EQ(
        invalid_coordinate.status,
        timestream::RTCSourceMaskAdmissionStatus::unavailable_coordinates);

    auto wrong_shape = data;
    wrong_shape.pointing_offsets_arcsec.data["alt"].conservativeResize(8);
    const auto invalid_shape = timestream::admit_rtc_source_mask(
        wrong_shape, calib.apt, "altaz", "nw", "map_center_radius", 30.0);
    EXPECT_EQ(
        invalid_shape.status,
        timestream::RTCSourceMaskAdmissionStatus::unavailable_coordinates);

    auto permuted = calib;
    std::swap(permuted.apt["uid"](0), permuted.apt["uid"](1));
    const auto permutation = timestream::admit_rtc_source_mask(
        data, permuted.apt, "altaz", "nw", "map_center_radius", 30.0);
    ASSERT_TRUE(permutation.admitted());
    EXPECT_NE(valid.identity, permutation.identity);

    auto duplicate = calib;
    duplicate.apt["uid"](1) = duplicate.apt["uid"](0);
    const auto invalid_identity = timestream::admit_rtc_source_mask(
        data, duplicate.apt, "altaz", "nw", "map_center_radius", 30.0);
    EXPECT_EQ(invalid_identity.status,
              timestream::RTCSourceMaskAdmissionStatus::
                  unavailable_detector_identity);
}

TEST(RtcPhaseIndependentProductionPaths,
     RequestedSourceMaskFailsClosedInTheActualRtcPath) {
    timestream::RTCProc proc;
    configure_minimal_proc(proc, false);
    proc.run_despike = true;
    proc.despiker.source_protection_enabled = true;
    proc.despiker.source_protection_radius_arcsec = 30.0;

    auto input = make_rtc_data(0, true);
    PtcData output;
    TestCalib calib = make_calib(false);
    TestTelescope telescope;
    proc.bind_assigned_grid_segment(
        0, 0, 0, 9, 9, 2, telescope.fsmp, 1, false,
        "project:152389");
    EXPECT_THROW(
        proc.run(input, output, calib, telescope, 1.0e-5, "nw"),
        std::runtime_error);
}

TEST(RtcPhaseIndependentProductionPaths,
     SequentialAndOpenMpActualRunsHaveTheSameStageIdentities) {
    constexpr Eigen::Index n_scans = 4;
    timestream::RTCProc sequential;
    configure_minimal_proc(sequential);
    for (Eigen::Index scan = 0; scan < n_scans; ++scan) {
        (void)run_one(sequential, scan, false);
    }

    timestream::RTCProc parallel;
    configure_minimal_proc(parallel);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (Eigen::Index scan = 0; scan < n_scans; ++scan) {
        (void)run_one(parallel, scan, false);
    }

    const auto sequential_snapshot =
        sequential.snapshot_phase_independent_state();
    const auto parallel_snapshot = parallel.snapshot_phase_independent_state();
    ASSERT_EQ(sequential_snapshot.stages.size(),
              static_cast<std::size_t>(n_scans));
    ASSERT_EQ(parallel_snapshot.stages.size(),
              sequential_snapshot.stages.size());
    for (std::size_t index = 0;
         index < sequential_snapshot.stages.size(); ++index) {
        EXPECT_EQ(sequential_snapshot.stages[index].stage_identity,
                  parallel_snapshot.stages[index].stage_identity);
    }
}

TEST(RtcPhaseIndependentProductionPaths,
     RealAndSimulationInterfacesPreserveContractButNotIdentity) {
    timestream::RTCProc real;
    configure_minimal_proc(real);
    const auto real_output = run_one(real, 0, false);
    timestream::RTCProc simulated;
    configure_minimal_proc(simulated);
    const auto simulated_output = run_one(simulated, 0, true);

    EXPECT_EQ(real_output.scans.data, simulated_output.scans.data);
    const auto real_stage = real.phase_independent_stage_for_scan(0);
    const auto simulated_stage =
        simulated.phase_independent_stage_for_scan(0);
    EXPECT_EQ(real_stage.physical_event_semantics, "unavailable");
    EXPECT_EQ(simulated_stage.physical_event_semantics, "unavailable");
    EXPECT_EQ(real_stage.signal_stage_bits,
              simulated_stage.signal_stage_bits);
    EXPECT_NE(real_stage.stage_identity,
              simulated_stage.stage_identity);
}

}  // namespace
