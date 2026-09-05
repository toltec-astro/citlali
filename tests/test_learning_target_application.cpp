#include <gtest/gtest.h>

#include <citlali/core/engine/engine.h>
#include <citlali/core/mapmaking/map.h>

#include <spdlog/sinks/null_sink.h>

#include <map>
#include <memory>
#include <stdexcept>

namespace {

ReductionLearningState::Options targeted_options() {
    ReductionLearningState::Options options;
    options.enabled = true;
    options.diagnostics_enabled = true;
    options.map_pixel_outlier_diagnostics_enabled = true;
    options.map_pixel_outlier_contributor_diagnostics_enabled = false;
    options.map_pixel_outlier_targeted_contributor_diagnostics_enabled = true;
    options.map_pixel_outlier_detector_exclusion_enabled = true;
    options.map_pixel_outlier_targeted_contributor_max_pixels = 2;
    return options;
}

ReductionLearningState::MapPixelOutlier outlier(int row, int col,
                                                double score) {
    ReductionLearningState::MapPixelOutlier record;
    record.obsnum = "152390";
    record.producer = "mapdiag:raw_obs";
    record.reason = "extreme_pixel_no_contributor";
    record.iter = 4;
    record.map_index = 0;
    record.row = row;
    record.col = col;
    record.leave_one_out_z = score;
    return record;
}

Engine configured_engine() {
    Engine engine;
    engine.logger = std::make_shared<spdlog::logger>(
        "learning-target-application-test",
        std::make_shared<spdlog::sinks::null_sink_mt>());
    engine.learning.configure(targeted_options());
    engine.learning.begin_iteration(4, true, "pointing");
    engine.observation_identity.obsnum = "152390";
    engine.iteration.fruit_iter = 5;
    return engine;
}

mapmaking::MapBuffer target_map() {
    mapmaking::MapBuffer map;
    map.n_rows = 12;
    map.n_cols = 13;
    map.signal.resize(2);
    return map;
}

struct ExclusionTestChunk {
    struct {
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> data;
    } flags;
    struct {
        int data = 0;
    } index;
    struct {
        Eigen::MatrixXd data;
    } scans;
    struct {
        std::map<std::string, Eigen::VectorXd> data;
    } tel_data;
    struct {
        std::map<std::string, Eigen::VectorXd> data;
    } pointing_offsets_arcsec;
};

struct ExclusionTestCalibration {
    std::map<std::string, Eigen::VectorXd> apt;
};

Engine exclusion_engine(
    citlali::config::MapPixelOutlierDetectorExclusionApplication
        application,
    const std::string &producer,
    const std::string &reason) {
    Engine engine;
    engine.logger = std::make_shared<spdlog::logger>(
        "learning-exclusion-application-test",
        std::make_shared<spdlog::sinks::null_sink_mt>());
    auto options = targeted_options();
    options.learn_iters = 1;
    options.apply_start_iter = 1;
    options.apply_max_new_flagged_fraction = 1.0;
    options.map_pixel_outlier_detector_exclusion_application = application;
    engine.learning.configure(options);
    engine.observation_identity.obsnum = "152390";
    engine.learning.begin_iteration(0, false, "pointing");
    ReductionLearningState::DetectorPenalty penalty;
    penalty.obsnum = "152390";
    penalty.producer = producer;
    penalty.reason = reason;
    penalty.iter = 0;
    penalty.scan = 0;
    penalty.uid = 4460;
    penalty.nw = 9;
    penalty.array = 1;
    penalty.factor = 0.0;
    penalty.scan_local = true;
    engine.learning.record_detector_penalty(penalty, true);
    engine.learning.begin_iteration(1, true, "pointing");
    engine.iteration.fruit_iter = 1;
    return engine;
}

ExclusionTestChunk exclusion_chunk() {
    ExclusionTestChunk chunk;
    chunk.flags.data.resize(3, 2);
    chunk.flags.data.setConstant(false);
    chunk.scans.data = Eigen::MatrixXd::Zero(3, 2);
    return chunk;
}

ExclusionTestCalibration exclusion_calibration() {
    ExclusionTestCalibration calibration;
    calibration.apt["uid"] = Eigen::VectorXd(2);
    calibration.apt["uid"] << 4460.0, 4461.0;
    calibration.apt["nw"] = Eigen::VectorXd::Constant(2, 9.0);
    calibration.apt["array"] = Eigen::VectorXd::Constant(2, 1.0);
    calibration.apt["flag"] = Eigen::VectorXd::Zero(2);
    calibration.apt["x_t"] = Eigen::VectorXd::Zero(2);
    calibration.apt["y_t"] = Eigen::VectorXd::Zero(2);
    return calibration;
}

TEST(LearningTargetApplication,
     AppliesResolvedBoundaryTargetsToRealContributionTracer) {
    auto engine = configured_engine();
    engine.learning.record_map_pixel_outlier(outlier(2, 3, 12.0));
    engine.learning.record_map_pixel_outlier(outlier(4, 5, 10.0));
    engine.learning.record_map_pixel_outlier(outlier(6, 7, 8.0));
    engine.learning.resolve_map_pixel_targets_for_next_iteration(
        "152390", "mapdiag:raw_obs", 4, 2, 12, 13);
    engine.learning.finalize_map_pixel_target_state(
        {"152390"}, "mapdiag:raw_obs", 4);

    auto map = target_map();
    ASSERT_NO_THROW(
        engine.configure_map_pixel_contribution_targets(map, "raw_obs"));
    EXPECT_TRUE(map.contribution_diag_enabled);
    EXPECT_TRUE(map.contribution_diag_targeted);
    EXPECT_TRUE(map.contribution_target_enabled(0, 2, 3));
    EXPECT_TRUE(map.contribution_target_enabled(0, 4, 5));
    EXPECT_FALSE(map.contribution_target_enabled(0, 6, 7));
}

TEST(LearningTargetApplication, MissingRequiredBoundaryStateFailsClosed) {
    auto engine = configured_engine();
    auto map = target_map();
    EXPECT_THROW(
        engine.configure_map_pixel_contribution_targets(map, "raw_obs"),
        std::runtime_error);
}

TEST(LearningTargetApplication,
     IncompatibleMapGridFailsClosed) {
    auto engine = configured_engine();
    engine.learning.record_map_pixel_outlier(outlier(2, 3, 12.0));
    engine.learning.resolve_map_pixel_targets_for_next_iteration(
        "152390", "mapdiag:raw_obs", 4, 2, 12, 13);
    engine.learning.finalize_map_pixel_target_state(
        {"152390"}, "mapdiag:raw_obs", 4);
    auto map = target_map();
    map.n_rows = 11;
    EXPECT_THROW(
        engine.configure_map_pixel_contribution_targets(map, "raw_obs"),
        std::runtime_error);
}

TEST(LearningTargetApplication,
     MapDiagnosticExclusionDefaultsToSharedCleaningStages) {
    auto engine = exclusion_engine(
        citlali::config::
            MapPixelOutlierDetectorExclusionApplication::pre_cleaning,
        "mapdiag:raw_obs", "map_pixel_outlier_detector_dominance");
    auto calibration = exclusion_calibration();
    auto rtc = exclusion_chunk();
    engine.apply_learned_rtc_sample_masks(rtc, calibration);
    EXPECT_TRUE(rtc.flags.data.col(0).all());
    EXPECT_FALSE(rtc.flags.data.col(1).any());

    auto map_input = exclusion_chunk();
    engine.apply_learned_mapmaking_detector_exclusions(
        map_input, calibration);
    EXPECT_FALSE(map_input.flags.data.col(0).any());
}

TEST(LearningTargetApplication,
     MapDiagnosticExclusionCanMoveAfterSharedCleaning) {
    auto engine = exclusion_engine(
        citlali::config::
            MapPixelOutlierDetectorExclusionApplication::pre_mapmaking,
        "mapdiag:raw_obs", "map_pixel_outlier_detector_dominance");
    auto calibration = exclusion_calibration();
    auto rtc = exclusion_chunk();
    engine.apply_learned_rtc_sample_masks(rtc, calibration);
    EXPECT_FALSE(rtc.flags.data.col(0).any());

    auto ptc = exclusion_chunk();
    engine.apply_learned_ptc_detector_exclusions(ptc, calibration);
    EXPECT_FALSE(ptc.flags.data.col(0).any());

    auto map_input = exclusion_chunk();
    engine.apply_learned_mapmaking_detector_exclusions(
        map_input, calibration);
    EXPECT_TRUE(map_input.flags.data.col(0).all());
    EXPECT_FALSE(map_input.flags.data.col(1).any());
}

TEST(LearningTargetApplication,
     BusyDetectorExclusionPlacementDoesNotMoveWithMapDiagnosticSetting) {
    auto engine = exclusion_engine(
        citlali::config::
            MapPixelOutlierDetectorExclusionApplication::pre_mapmaking,
        "ptc_second_pass", "busy_vetoed_residual");
    auto calibration = exclusion_calibration();
    auto rtc = exclusion_chunk();
    engine.apply_learned_rtc_sample_masks(rtc, calibration);
    EXPECT_TRUE(rtc.flags.data.col(0).all());

    auto map_input = exclusion_chunk();
    engine.apply_learned_mapmaking_detector_exclusions(
        map_input, calibration);
    EXPECT_FALSE(map_input.flags.data.col(0).any());
}


// Synthetic boundary states only: no observation data or experimental action.
Engine synthetic_cap_engine(int proposed_detectors) {
    Engine engine;
    engine.logger = std::make_shared<spdlog::logger>(
        "learning-cap-boundary-test",
        std::make_shared<spdlog::sinks::null_sink_mt>());
    auto options = targeted_options();
    options.learn_iters = 1;
    options.apply_start_iter = 1;
    options.apply_max_new_flagged_fraction = 0.02;
    options.map_pixel_outlier_detector_exclusion_application =
        citlali::config::
            MapPixelOutlierDetectorExclusionApplication::pre_cleaning;
    engine.learning.configure(options);
    engine.observation_identity.obsnum = "synthetic-cap-boundary";
    engine.learning.begin_iteration(0, false, "pointing");
    for (int detector = 0; detector < proposed_detectors; ++detector) {
        ReductionLearningState::DetectorPenalty penalty;
        penalty.obsnum = engine.observation_identity.obsnum;
        penalty.producer = "mapdiag:raw_obs";
        penalty.reason = "map_pixel_outlier_detector_dominance";
        penalty.iter = 0;
        penalty.scan = 0;
        penalty.uid = 1000 + detector;
        penalty.nw = 0;
        penalty.array = 0;
        penalty.factor = 0.0;
        penalty.scan_local = true;
        engine.learning.record_detector_penalty(penalty, true);
    }
    engine.learning.begin_iteration(1, true, "pointing");
    engine.iteration.fruit_iter = 1;
    return engine;
}

ExclusionTestChunk synthetic_cap_chunk(int samples, int detectors) {
    ExclusionTestChunk chunk;
    chunk.flags.data.resize(samples, detectors);
    chunk.flags.data.setConstant(false);
    chunk.scans.data = Eigen::MatrixXd::Zero(samples, detectors);
    return chunk;
}

ExclusionTestCalibration synthetic_cap_calibration(int first_detector) {
    const int detectors = 100 - first_detector;
    ExclusionTestCalibration calibration;
    calibration.apt["uid"] = Eigen::VectorXd::LinSpaced(
        detectors, 1000 + first_detector, 1099);
    for (const auto *column : {"nw", "array", "flag", "x_t", "y_t"}) {
        calibration.apt[column] = Eigen::VectorXd::Zero(detectors);
    }
    return calibration;
}

TEST(LearningTargetApplication,
     HistoricalCapCanRejectBeforeRtcAndAcceptBeforePtc) {
    // With no intervening removals both stages reject. Removing two of the
    // three proposed detectors changes the second gate, without changing its
    // cap or records. These are supplied boundary states, not an RTC replay.
    for (const int removed : {0, 2}) {
        SCOPED_TRACE(removed);
        auto engine = synthetic_cap_engine(3);
        auto raw_calibration = synthetic_cap_calibration(0);
        auto rtc = synthetic_cap_chunk(10, 100);
        engine.apply_learned_rtc_sample_masks(rtc, raw_calibration);
        ASSERT_EQ(engine.learning.learned_mask_applications.size(), 1U);
        const auto rtc_receipt = engine.learning.learned_mask_applications.back();
        EXPECT_EQ(rtc_receipt.stage, "pre_rtc_detector_exclusion");
        EXPECT_DOUBLE_EQ(rtc_receipt.newly_flagged_fraction, 0.03);
        EXPECT_FALSE(rtc_receipt.applied);
        EXPECT_FALSE(rtc.flags.data.any());
        EXPECT_TRUE(raw_calibration.apt.at("flag").isZero());

        auto ptc_calibration = synthetic_cap_calibration(removed);
        auto ptc = synthetic_cap_chunk(5, 100 - removed);
        engine.apply_learned_ptc_detector_exclusions(ptc, ptc_calibration);
        ASSERT_EQ(engine.learning.learned_mask_applications.size(), 2U);
        const auto ptc_receipt = engine.learning.learned_mask_applications.back();
        EXPECT_EQ(ptc_receipt.stage, "pre_ptc_detector_exclusion");
        EXPECT_EQ(ptc_receipt.candidate_records, 3);
        EXPECT_EQ(ptc_receipt.matched_records, 3 - removed);
        EXPECT_EQ(ptc_receipt.invalid_records, removed);
        EXPECT_DOUBLE_EQ(ptc_receipt.newly_flagged_fraction,
                         double(3 - removed) / (100 - removed));
        EXPECT_EQ(ptc_receipt.applied, removed == 2);
        if (removed == 2) {
            EXPECT_TRUE(ptc.flags.data.col(0).all());
            EXPECT_FALSE(ptc.flags.data.rightCols(97).any());
            EXPECT_DOUBLE_EQ(ptc_calibration.apt.at("flag")(0), 1.0);
        } else {
            EXPECT_FALSE(ptc.flags.data.any());
            EXPECT_TRUE(ptc_calibration.apt.at("flag").isZero());
        }
    }
}

TEST(LearningTargetApplication, HistoricalCapAcceptsItsExactBoundary) {
    auto engine = synthetic_cap_engine(2);
    auto calibration = synthetic_cap_calibration(0);
    auto rtc = synthetic_cap_chunk(10, 100);
    engine.apply_learned_rtc_sample_masks(rtc, calibration);
    ASSERT_EQ(engine.learning.learned_mask_applications.size(), 1U);
    const auto receipt = engine.learning.learned_mask_applications.back();
    EXPECT_DOUBLE_EQ(receipt.newly_flagged_fraction, 0.02);
    EXPECT_TRUE(receipt.applied);
    EXPECT_TRUE(rtc.flags.data.leftCols(2).all());
    EXPECT_FALSE(rtc.flags.data.rightCols(98).any());
}



TEST(LearningTargetApplication, FruitResponseNativeCapPrecedence) {
    using namespace citlali::fruit;
    for (const std::string arm : {"H", "Half", "Hold"}) {
        for (const bool independent : {false, true}) {
            SCOPED_TRACE(arm + (independent ? ":independent" : ":map-only"));
            auto engine = synthetic_cap_engine(3);
            const ResponseKey key{"synthetic-cap-boundary", 0, 1002, 0};
            if (independent) {
                ReductionLearningState::DetectorPenalty penalty;
                penalty.obsnum = key.observation;
                penalty.producer = "ptc_second_pass";
                penalty.reason = "busy_vetoed_residual";
                penalty.iter = 0; penalty.scan = 0; penalty.uid = 1002;
                penalty.nw = 0; penalty.array = 0; penalty.factor = 0;
                penalty.scan_local = true;
                engine.learning.record_detector_penalty(penalty, true);
            }
            auto &state = engine.learning.fruit_response;
            state.configure(arm);
            state.begin(0, false);
            ResponseCandidate proposal;
            proposal.key = key;
            proposal.response.ratio = 2;
            proposal.response.footprint = proposal.response.conditioned = 1;
            state.resolve({proposal}, {{0, proposal.response}});
            state.begin(1, true);
            auto raw_calibration = synthetic_cap_calibration(0);
            auto rtc = synthetic_cap_chunk(10, 100);
            engine.apply_learned_rtc_sample_masks(rtc, raw_calibration);
            EXPECT_FALSE(rtc.flags.data.any());
            EXPECT_EQ(state.receipts().at(key)[0].status, 3);
            auto processed_calibration = synthetic_cap_calibration(2);
            auto ptc = synthetic_cap_chunk(5, 98);
            engine.apply_learned_ptc_detector_exclusions(ptc, processed_calibration);
            const bool hard = arm == "H" || independent;
            EXPECT_EQ(ptc.flags.data.col(0).all(), hard);
            EXPECT_EQ(processed_calibration.apt.at("flag")(0), hard ? 1 : 0);
            EXPECT_FALSE(ptc.flags.data.rightCols(97).any());
            EXPECT_EQ(state.receipts().at(key)[1].status, 4);
            EXPECT_EQ(state.receipts().at(key)[1].suppressed, !hard);
            EXPECT_EQ(state.receipts().at(key)[1].independent_reason, independent);
            EXPECT_EQ(state.coefficient(key), arm == "Half" ? 0.5 : 1.0);
            EXPECT_EQ(state.first_applications().at(key), hard && arm != "H" ? -1 : 1);
        }
    }
}

TEST(LearningTargetApplication, FruitResponseNativeJincHalfPropagatesEveryCoefficient) {
    using namespace citlali::fruit;
    const auto directory = std::filesystem::path(testing::TempDir()) /
        ("fruit-native-jinc-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::filesystem::remove_all(path); }
    } cleanup{directory};
    for (const double input_weight : {1.0, 0.731}) for (const bool clip : {false, true}) {
        mapmaking::JincMapmaker maker;
        maker.run_polarization = false;
        maker.parallel_policy = "seq";
        maker.subpixel_n = 2;
        Eigen::MatrixXd base(3, 3);
        base << 0, -0.25, 0, -0.25, 1, -0.25, 0, -0.25, 0;
        maker.jinc_weights_mat[0] = base;
        maker.jinc_weights_sq_mat[0] = base.array().square().matrix();
        for (int i = 0; i < 4; ++i) {
            maker.jinc_weights_mat_subpix[0].push_back(base * (i + 1));
            maker.jinc_weights_sq_mat_subpix[0].push_back((base * (i + 1)).array().square().matrix());
        }
        timestream::TCData<timestream::TCDataKind::PTC, Eigen::MatrixXd> data;
        data.index.data = 0;
        data.scans.data.resize(1, 2); data.scans.data << 20, 0;
        data.kernel.data.resize(1, 2); data.kernel.data << 2, 3;
        data.weights.data = Eigen::VectorXd::Constant(2, input_weight);
        data.flags.data.resize(1, 2); data.flags.data.setConstant(false);
        data.noise.data.resize(1, 2); data.noise.data << 1, -1;
        data.tel_data.data["TelElAct"] = Eigen::VectorXd::Zero(1);
        data.tel_data.data["alt_phys"] = Eigen::VectorXd::Constant(1, (clip ? -0.8 : 0.2) * 1e-5);
        data.tel_data.data["az_phys"] = Eigen::VectorXd::Constant(1, (clip ? -1.2 : -0.2) * 1e-5);
        data.pointing_offsets_arcsec.data["az"] = Eigen::VectorXd::Zero(1);
        data.pointing_offsets_arcsec.data["alt"] = Eigen::VectorXd::Zero(1);
        std::map<std::string, Eigen::VectorXd> apt;
        for (const auto *name : {"array", "flag", "x_t", "y_t"}) apt[name] = Eigen::VectorXd::Zero(2);
        apt["uid"] = Eigen::VectorXd(2); apt["uid"] << 1000, 1001;
        Eigen::VectorXi indices = Eigen::VectorXi::Zero(2);
        std::string axes = "altaz";
        auto make_buffer = [] {
            mapmaking::MapBuffer buffer;
            buffer.n_rows = buffer.n_cols = 3;
            buffer.pixel_size_rad = 1e-5;
            buffer.map_grouping = "array";
            buffer.parallel_policy = "seq";
            buffer.sig_unit = "mJy/beam";
            buffer.cov_cut = 0;
            buffer.signal = buffer.grid_weight = buffer.weight = buffer.kernel = buffer.coverage =
                std::vector<Eigen::MatrixXd>{Eigen::MatrixXd::Zero(3, 3)};
            buffer.n_noise = 1;
            buffer.randomize_dets = true;
            buffer.noise.emplace_back(3, 3, 1); buffer.noise.back().setZero();
            return buffer;
        };
        auto control = make_buffer(), half = make_buffer();
        mapmaking::MapBuffer coadd;
        ResponseInterventionState state;
        state.configure("Half"); state.begin(0, false);
        ResponseCandidate proposal;
        proposal.key = {"synthetic", 0, 1000, 0};
        proposal.response.ratio = 2;
        proposal.response.footprint = proposal.response.conditioned = 1;
        state.resolve({proposal}, {{0, proposal.response}});
        state.begin(1, true);
        state.record_stage(proposal.key, 0, 4, 0.01, 0.02);
        state.record_stage(proposal.key, 1, 4, 0.01, 0.02);
        ResponseOccurrenceLedger::KernelBank kernels{{0, maker.jinc_weights_mat_subpix.at(0)}};
        ResponseOccurrenceLedger::KernelBank squares{{0, maker.jinc_weights_sq_mat_subpix.at(0)}};
        half.fruit_response_state = &state;
        half.fruit_response_ledger = std::make_shared<ResponseOccurrenceLedger>(
            directory / ((clip ? "clip-" : "center-") + std::to_string(input_weight) + ".bin"), "synthetic", 1, 3, 3,
            std::vector<int>{0}, kernels, squares);
        maker.populate_maps_jinc(data, control, coadd, indices, axes, apt, 2.0, true, false);
        maker.populate_maps_jinc(data, half, coadd, indices, axes, apt, 2.0, true, false);
        EXPECT_TRUE((half.signal[0].array() == control.signal[0].array() * 0.5).all());
        EXPECT_TRUE(half.grid_weight[0].isApprox(control.grid_weight[0] * 0.75, 64 * std::numeric_limits<double>::epsilon()));
        EXPECT_TRUE(half.weight[0].isApprox(control.weight[0] * 0.625, 64 * std::numeric_limits<double>::epsilon()));
        EXPECT_TRUE(half.kernel[0].isApprox(control.kernel[0] * 0.8, 64 * std::numeric_limits<double>::epsilon()));
        EXPECT_TRUE((half.coverage[0].array() == control.coverage[0].array()).all());
        EXPECT_EQ(half.fruit_response_ledger->occurrence_count(), 2U);
        // A separate diagnostic noise pass uses the same multiplier but adds
        // no second science occurrence or duplicate scan boundary.
        maker.populate_maps_jinc(data, control, coadd, indices, axes, apt, 2.0, false, true);
        maker.populate_maps_jinc(data, half, coadd, indices, axes, apt, 2.0, false, true);
        EXPECT_EQ(half.fruit_response_ledger->occurrence_count(), 2U);
        for (Eigen::Index i = 0; i < half.noise[0].size(); ++i)
            EXPECT_DOUBLE_EQ(half.noise[0].data()[i], control.noise[0].data()[i] * 0.5);
        control.normalize_maps(); half.normalize_maps();
        const Eigen::Index center = clip ? 0 : 1;
        EXPECT_DOUBLE_EQ(control.signal[0](center, center), 10.0);
        EXPECT_DOUBLE_EQ(half.signal[0](center, center), 10.0 / 1.5);
        EXPECT_DOUBLE_EQ(control.weight[0](center, center), 2.0 * input_weight);
        EXPECT_DOUBLE_EQ(half.weight[0](center, center), 1.8 * input_weight);
        half.fruit_response_ledger->finish(state, {}, half.signal, half.weight, half.weight, 0.0);
        EXPECT_EQ(state.first_applications().at(proposal.key), 1);
    }
}

}  // namespace
