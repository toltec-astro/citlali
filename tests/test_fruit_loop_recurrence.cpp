#include <gtest/gtest.h>

#include <citlali/core/mapmaking/naive_mm.h>
#include <citlali/core/pipeline/fruit_loop_injected_source_test.h>
#include <citlali/core/pipeline/mapdiag_penalty_evidence.h>
#include <citlali/core/timestream/timestream.h>

#include <cmath>
#include <map>
#include <string>

namespace {

using PtcData =
    timestream::TCData<timestream::TCDataKind::PTC, Eigen::MatrixXd>;

struct SyntheticCalib {
    Eigen::VectorXi arrays;
    std::map<std::string, Eigen::VectorXd> apt;
};

struct SyntheticKernelCenter {
    bool uniform_source_center_enabled = false;
    double uniform_source_lat_rad = 0.0;
    double uniform_source_lon_rad = 0.0;

    void clear_uniform_source_center() {
        uniform_source_center_enabled = false;
        uniform_source_lat_rad = 0.0;
        uniform_source_lon_rad = 0.0;
    }

    void set_uniform_source_center(double lat_rad, double lon_rad) {
        uniform_source_center_enabled = true;
        uniform_source_lat_rad = lat_rad;
        uniform_source_lon_rad = lon_rad;
    }
};

constexpr Eigen::Index kRows = 9;
constexpr Eigen::Index kCols = 9;
constexpr double kPixelSizeRad = 1.0e-5;

SyntheticCalib make_calib() {
    SyntheticCalib calib;
    calib.arrays.resize(1);
    calib.arrays << 0;
    calib.apt["array"] = Eigen::VectorXd::Zero(1);
    calib.apt["flag"] = Eigen::VectorXd::Zero(1);
    calib.apt["x_t"] = Eigen::VectorXd::Zero(1);
    calib.apt["y_t"] = Eigen::VectorXd::Zero(1);
    return calib;
}

PtcData make_scan() {
    const Eigen::Index n_samples = kRows * kCols;
    PtcData data;
    data.scans.data = Eigen::MatrixXd::Zero(n_samples, 1);
    data.kernel.data = Eigen::MatrixXd::Zero(n_samples, 1);
    data.flags.data.resize(n_samples, 1);
    data.flags.data.setConstant(false);
    data.weights.data = Eigen::VectorXd::Ones(1);
    data.index.data = 7;

    Eigen::VectorXd lat(n_samples);
    Eigen::VectorXd lon(n_samples);
    Eigen::Index sample = 0;
    for (Eigen::Index row = 0; row < kRows; ++row) {
        for (Eigen::Index col = 0; col < kCols; ++col) {
            lat(sample) =
                (static_cast<double>(row) - (kRows - 1) / 2.0) *
                kPixelSizeRad;
            lon(sample) =
                (static_cast<double>(col) - (kCols - 1) / 2.0) *
                kPixelSizeRad;
            ++sample;
        }
    }
    data.tel_data.data["TelElAct"] = Eigen::VectorXd::Zero(n_samples);
    data.tel_data.data["alt_phys"] = lat;
    data.tel_data.data["az_phys"] = lon;
    data.pointing_offsets_arcsec.data["az"] =
        Eigen::VectorXd::Zero(n_samples);
    data.pointing_offsets_arcsec.data["alt"] =
        Eigen::VectorXd::Zero(n_samples);
    return data;
}

mapmaking::MapBuffer make_map_buffer(bool with_kernel) {
    mapmaking::MapBuffer maps{"synthetic"};
    maps.n_rows = kRows;
    maps.n_cols = kCols;
    maps.pixel_size_rad = kPixelSizeRad;
    maps.map_grouping = "array";
    maps.parallel_policy = "seq";
    maps.cov_cut = 0.0;
    maps.wcs.naxis = {static_cast<int>(kCols), static_cast<int>(kRows), 0, 0};
    maps.wcs.cdelt = {-1.0F, 1.0F, 0.0F, 0.0F};
    maps.wcs.crpix = {4.0F, 4.0F, 0.0F, 0.0F};
    maps.wcs.crval = {0.0F, 0.0F, 0.0F, 0.0F};
    maps.wcs.cunit = {"deg", "deg"};
    maps.signal = {Eigen::MatrixXd::Zero(kRows, kCols)};
    maps.weight = {Eigen::MatrixXd::Zero(kRows, kCols)};
    if (with_kernel) {
        maps.kernel = {Eigen::MatrixXd::Zero(kRows, kCols)};
    }
    return maps;
}

Eigen::MatrixXd gaussian(double amplitude, double sigma_pixels) {
    Eigen::MatrixXd result(kRows, kCols);
    const double center_row = (kRows - 1) / 2.0;
    const double center_col = (kCols - 1) / 2.0;
    for (Eigen::Index row = 0; row < kRows; ++row) {
        for (Eigen::Index col = 0; col < kCols; ++col) {
            const double dy = static_cast<double>(row) - center_row;
            const double dx = static_cast<double>(col) - center_col;
            result(row, col) =
                amplitude *
                std::exp(-(dx * dx + dy * dy) /
                         (2.0 * sigma_pixels * sigma_pixels));
        }
    }
    return result;
}

timestream::TCProc make_processor() {
    timestream::TCProc processor;
    processor.fruit_loops_interp_mode = "bilinear";
    processor.fruit_mode = "upper";
    processor.fruit_loops_flux.resize(1);
    processor.fruit_loops_flux << 1.0e-100;
    processor.fruit_loops_kernel_feedback_enabled = true;
    processor.fruit_loops_diagnostics_enabled = false;
    return processor;
}

mapmaking::MapBuffer bin_scan(
    PtcData &data, SyntheticCalib &calib, Eigen::VectorXi &map_indices) {
    auto maps = make_map_buffer(data.kernel.data.size() != 0);
    mapmaking::MapBuffer coadd{"unused"};
    mapmaking::NaiveMapmaker mapmaker;
    mapmaker.run_polarization = false;
    std::string pixel_axes = "altaz";
    mapmaker.populate_maps_naive(
        data, maps, coadd, map_indices, pixel_axes, calib.apt, 1.0,
        true, false);
    maps.normalize_maps();
    return maps;
}

double map_rms(const Eigen::MatrixXd &map) {
    return std::sqrt(map.array().square().mean());
}

double radial_sigma_pixels(const Eigen::MatrixXd &map) {
    const double center_row = (kRows - 1) / 2.0;
    const double center_col = (kCols - 1) / 2.0;
    double weighted_radius2 = 0.0;
    double sum = 0.0;
    for (Eigen::Index row = 0; row < kRows; ++row) {
        for (Eigen::Index col = 0; col < kCols; ++col) {
            const double value = std::max(0.0, map(row, col));
            const double dy = static_cast<double>(row) - center_row;
            const double dx = static_cast<double>(col) - center_col;
            weighted_radius2 += value * (dx * dx + dy * dy);
            sum += value;
        }
    }
    return std::sqrt(weighted_radius2 / (2.0 * sum));
}

TEST(fruit_loop_recurrence, subtract_addback_round_trip_restores_signal_and_kernel) {
    auto calib = make_calib();
    auto data = make_scan();
    auto processor = make_processor();
    auto model = make_map_buffer(true);
    model.signal[0] = gaussian(7.5, 1.25);
    model.kernel[0] = gaussian(0.8, 1.5);
    Eigen::VectorXi map_indices(1);
    map_indices << 0;

    for (Eigen::Index sample = 0; sample < data.scans.data.rows(); ++sample) {
        data.scans.data(sample, 0) =
            std::sin(0.31 * static_cast<double>(sample)) + 2.0;
        data.kernel.data(sample, 0) =
            std::cos(0.19 * static_cast<double>(sample)) - 0.2;
    }
    data.flags.data(0, 0) = true;
    const Eigen::MatrixXd original_signal = data.scans.data;
    const Eigen::MatrixXd original_kernel = data.kernel.data;

    processor.map_to_tod<timestream::TCProc::SourceType::NegativeMap>(
        model, data, calib, map_indices, "altaz", "array");
    processor.map_to_tod<timestream::TCProc::SourceType::Map>(
        model, data, calib, map_indices, "altaz", "array");

    EXPECT_TRUE(data.scans.data.isApprox(original_signal, 2.0e-15));
    EXPECT_TRUE(data.kernel.data.isApprox(original_kernel, 2.0e-15));
    EXPECT_DOUBLE_EQ(data.scans.data(0, 0), original_signal(0, 0));
    EXPECT_DOUBLE_EQ(data.kernel.data(0, 0), original_kernel(0, 0));
}

TEST(fruit_loop_recurrence,
     alpha_one_uses_unmodified_complete_product_path_bitwise) {
    auto complete = make_map_buffer(true);
    complete.signal[0] = gaussian(7.0, 1.2);
    complete.kernel[0] = gaussian(0.8, 1.5);
    complete.weight[0].setConstant(3.0);
    complete.median_rms = Eigen::VectorXd::Constant(1, 0.25);
    const auto signal = complete.signal[0];
    const auto kernel = complete.kernel[0];
    const auto weight = complete.weight[0];
    citlali::fruit::FruitLoopRelaxedFeedbackState state;

    citlali::fruit::update_fruit_loop_relaxed_feedback_state(
        state, complete, "152389", 0, 1.0);

    EXPECT_FALSE(state.stored);
    EXPECT_DOUBLE_EQ(state.alpha, 1.0);
    EXPECT_TRUE(complete.signal[0].isApprox(signal, 0.0));
    EXPECT_TRUE(complete.kernel[0].isApprox(kernel, 0.0));
    EXPECT_TRUE(complete.weight[0].isApprox(weight, 0.0));
}

TEST(fruit_loop_recurrence,
     relaxed_state_updates_signal_and_kernel_and_leaves_q_state_authoritative) {
    auto first = make_map_buffer(true);
    first.signal[0].setConstant(2.0);
    first.kernel[0].setConstant(0.5);
    first.weight[0].setConstant(3.0);
    first.median_rms = Eigen::VectorXd::Constant(1, 0.25);
    citlali::fruit::FruitLoopRelaxedFeedbackState state;

    citlali::fruit::update_fruit_loop_relaxed_feedback_state(
        state, first, "152389", 0, 1.25);
    ASSERT_TRUE(state.stored);
    EXPECT_EQ(state.wcs_naxis.size(), 2U);
    EXPECT_EQ(state.wcs_cdelt.size(), 2U);
    EXPECT_EQ(state.wcs_crpix.size(), 2U);
    EXPECT_EQ(state.wcs_crval.size(), 2U);
    EXPECT_EQ(state.wcs_cunit.size(), 2U);

    auto second = first;
    second.signal[0].setConstant(6.0);
    second.kernel[0].setConstant(1.5);
    second.weight[0].setConstant(9.0);
    second.median_rms(0) = 0.125;
    citlali::fruit::update_fruit_loop_relaxed_feedback_state(
        state, second, "152389", 1, 1.25);

    auto loaded = second;
    const double reloaded_weight = std::nextafter(9.0, 10.0);
    const double reloaded_fits_medrms = 15.4269797465861;
    loaded.weight[0].setConstant(reloaded_weight);
    loaded.median_rms(0) = reloaded_fits_medrms;
    citlali::fruit::apply_fruit_loop_relaxed_feedback_state(
        state, loaded, "152389", 1);
    EXPECT_TRUE(loaded.signal[0].isConstant(7.0));
    EXPECT_TRUE(loaded.kernel[0].isConstant(1.75));
    EXPECT_TRUE(loaded.weight[0].isConstant(reloaded_weight));
    EXPECT_DOUBLE_EQ(loaded.median_rms(0), reloaded_fits_medrms);
}

TEST(fruit_loop_recurrence,
     relaxed_state_rejects_grid_and_finite_support_changes) {
    auto first = make_map_buffer(true);
    first.signal[0].setConstant(2.0);
    first.kernel[0].setConstant(0.5);
    first.weight[0].setOnes();
    citlali::fruit::FruitLoopRelaxedFeedbackState state;
    citlali::fruit::update_fruit_loop_relaxed_feedback_state(
        state, first, "152389", 0, 1.5);

    auto changed_grid = first;
    changed_grid.wcs.crval[0] = 1.0F;
    EXPECT_THROW(
        citlali::fruit::update_fruit_loop_relaxed_feedback_state(
            state, changed_grid, "152389", 1, 1.5),
        std::invalid_argument);

    auto changed_support = first;
    changed_support.signal[0](0, 0) =
        std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        citlali::fruit::update_fruit_loop_relaxed_feedback_state(
            state, changed_support, "152389", 1, 1.5),
        std::invalid_argument);
}

TEST(fruit_loop_recurrence, injected_gaussian_converges_through_controlled_cleaner) {
    constexpr double truth_amplitude = 10.0;
    constexpr double truth_sigma = 1.2;
    constexpr double cleaner_transfer = 0.4;

    auto calib = make_calib();
    auto processor = make_processor();
    auto truth = make_map_buffer(false);
    truth.signal[0] = gaussian(truth_amplitude, truth_sigma);
    Eigen::VectorXi map_indices(1);
    map_indices << 0;

    auto raw = make_scan();
    raw.kernel.data.resize(0, 0);
    processor.map_to_tod<timestream::TCProc::SourceType::Map>(
        truth, raw, calib, map_indices, "altaz", "array");

    auto seed_data = raw;
    seed_data.scans.data *= cleaner_transfer;
    auto estimate = bin_scan(seed_data, calib, map_indices);
    double previous_delta_rms = std::numeric_limits<double>::infinity();
    double previous_error =
        std::abs(estimate.signal[0](kRows / 2, kCols / 2) -
                 truth_amplitude);

    for (int iteration = 1; iteration <= 6; ++iteration) {
        auto iter_data = raw;
        processor.map_to_tod<timestream::TCProc::SourceType::NegativeMap>(
            estimate, iter_data, calib, map_indices, "altaz", "array");
        iter_data.scans.data *= cleaner_transfer;
        processor.map_to_tod<timestream::TCProc::SourceType::Map>(
            estimate, iter_data, calib, map_indices, "altaz", "array");

        auto next = bin_scan(iter_data, calib, map_indices);
        const double delta_rms =
            map_rms(next.signal[0] - estimate.signal[0]);
        const double amplitude =
            next.signal[0](kRows / 2, kCols / 2);
        const double error = std::abs(amplitude - truth_amplitude);

        EXPECT_LT(delta_rms, previous_delta_rms);
        EXPECT_LT(error, previous_error);
        EXPECT_NEAR(
            radial_sigma_pixels(next.signal[0]), truth_sigma, 5.0e-3);
        previous_delta_rms = delta_rms;
        previous_error = error;
        estimate = std::move(next);
    }

    EXPECT_NEAR(
        estimate.signal[0](kRows / 2, kCols / 2),
        truth_amplitude * (1.0 - std::pow(1.0 - cleaner_transfer, 7)),
        1.0e-10);
}

TEST(fruit_loop_recurrence,
     injected_source_test_scales_pristine_kernel_by_array) {
    PtcData data;
    data.scans.data = Eigen::MatrixXd::Constant(2, 3, 5.0);
    data.kernel.data.resize(2, 3);
    data.kernel.data << 0.0, 0.5, 1.0,
                        1.0, 0.0, 0.25;

    SyntheticCalib calib;
    calib.arrays.resize(3);
    calib.arrays << 0, 1, 2;
    calib.apt["array"].resize(3);
    calib.apt["array"] << 0.0, 1.0, 2.0;

    citlali::config::FruitLoopsInjectedSourceTestConfig config;
    config.enabled = true;
    config.start_iteration = 4;
    config.array_amplitude_mjy_beam = {10.0, 20.0, 30.0};

    const auto original_kernel = data.kernel.data;
    const auto summary =
        citlali::pipeline::inject_fruit_loop_test_source(
            data, calib, config, 4, "mJy/beam");

    EXPECT_TRUE(summary.applied);
    EXPECT_EQ(summary.projected_samples, 4);
    EXPECT_EQ(summary.arrays[0].projected_samples, 1);
    EXPECT_EQ(summary.arrays[1].projected_samples, 1);
    EXPECT_EQ(summary.arrays[2].projected_samples, 2);
    EXPECT_TRUE(data.kernel.data.isApprox(original_kernel, 0.0));
    EXPECT_DOUBLE_EQ(data.scans.data(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(data.scans.data(1, 0), 15.0);
    EXPECT_DOUBLE_EQ(data.scans.data(0, 1), 15.0);
    EXPECT_DOUBLE_EQ(data.scans.data(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(data.scans.data(0, 2), 35.0);
    EXPECT_DOUBLE_EQ(data.scans.data(1, 2), 12.5);
}

TEST(fruit_loop_recurrence,
     injected_source_explicit_zero_offset_retains_legacy_kernel_center) {
    SyntheticKernelCenter kernel;
    kernel.set_uniform_source_center(1.0, -1.0);
    citlali::config::FruitLoopsInjectedSourceTestConfig config;
    config.enabled = true;
    config.start_iteration = 1;
    config.array_amplitude_mjy_beam = {100.0};
    config.az_offset_arcsec = 0.0;
    config.el_offset_arcsec = 0.0;

    citlali::pipeline::configure_fruit_loop_injected_source_kernel_center(
        kernel, config, 1, ASEC_TO_RAD);

    EXPECT_FALSE(kernel.uniform_source_center_enabled);
    EXPECT_DOUBLE_EQ(kernel.uniform_source_lat_rad, 0.0);
    EXPECT_DOUBLE_EQ(kernel.uniform_source_lon_rad, 0.0);
}

TEST(fruit_loop_recurrence,
     injected_source_offset_configures_map_world_kernel_center) {
    SyntheticKernelCenter kernel;
    citlali::config::FruitLoopsInjectedSourceTestConfig config;
    config.enabled = true;
    config.start_iteration = 1;
    config.array_amplitude_mjy_beam = {100.0};
    config.az_offset_arcsec = kPixelSizeRad / (ASEC_TO_RAD);
    config.el_offset_arcsec = -2.0 * kPixelSizeRad / (ASEC_TO_RAD);

    citlali::pipeline::configure_fruit_loop_injected_source_kernel_center(
        kernel, config, 1, ASEC_TO_RAD);
    ASSERT_TRUE(kernel.uniform_source_center_enabled);
    EXPECT_NEAR(kernel.uniform_source_lon_rad, kPixelSizeRad, 1.0e-20);
    EXPECT_NEAR(
        kernel.uniform_source_lat_rad, -2.0 * kPixelSizeRad, 1.0e-20);
}

TEST(fruit_loop_recurrence,
     injected_source_inactive_iteration_clears_offset_kernel_state) {
    SyntheticKernelCenter kernel;
    kernel.set_uniform_source_center(1.0, -1.0);
    citlali::config::FruitLoopsInjectedSourceTestConfig config;
    config.enabled = true;
    config.start_iteration = 2;
    config.array_amplitude_mjy_beam = {100.0};
    config.az_offset_arcsec = 10.0;
    config.el_offset_arcsec = -10.0;

    citlali::pipeline::configure_fruit_loop_injected_source_kernel_center(
        kernel, config, 1, ASEC_TO_RAD);

    EXPECT_FALSE(kernel.uniform_source_center_enabled);
    EXPECT_DOUBLE_EQ(kernel.uniform_source_lat_rad, 0.0);
    EXPECT_DOUBLE_EQ(kernel.uniform_source_lon_rad, 0.0);
}

TEST(fruit_loop_recurrence,
     injected_source_test_is_exactly_inactive_before_start) {
    auto data = make_scan();
    const auto original_signal = data.scans.data;
    const auto original_kernel = data.kernel.data;
    auto calib = make_calib();

    citlali::config::FruitLoopsInjectedSourceTestConfig config;
    config.enabled = true;
    config.start_iteration = 3;
    config.array_amplitude_mjy_beam = {10.0};

    const auto summary =
        citlali::pipeline::inject_fruit_loop_test_source(
            data, calib, config, 2, "mJy/beam");

    EXPECT_FALSE(summary.applied);
    EXPECT_EQ(summary.projected_samples, 0);
    EXPECT_TRUE(data.scans.data.isApprox(original_signal, 0.0));
    EXPECT_TRUE(data.kernel.data.isApprox(original_kernel, 0.0));
}

TEST(fruit_loop_recurrence,
     injected_source_test_rejects_missing_kernel_and_wrong_units) {
    auto data = make_scan();
    auto calib = make_calib();
    citlali::config::FruitLoopsInjectedSourceTestConfig config;
    config.enabled = true;
    config.start_iteration = 1;
    config.array_amplitude_mjy_beam = {10.0};

    EXPECT_THROW(
        citlali::pipeline::inject_fruit_loop_test_source(
            data, calib, config, 1, "K"),
        citlali::error::Error);
    data.kernel.data.resize(0, 0);
    EXPECT_THROW(
        citlali::pipeline::inject_fruit_loop_test_source(
            data, calib, config, 1, "mJy/beam"),
        citlali::error::Error);
}

TEST(fruit_loop_recurrence,
     mapdiag_feedback_excluded_evidence_subtracts_exact_carried_model) {
    Eigen::MatrixXd complete(2, 3);
    complete << 9.0, 7.0, 4.0,
                -1.0, 3.0, 8.0;
    const Eigen::MatrixXd original = complete;
    Eigen::MatrixXd feedback(2, 3);
    feedback << 2.0, 5.0, -1.0,
                -4.0, 3.5, 6.0;

    const Eigen::MatrixXd evidence =
        citlali::pipeline::make_mapdiag_feedback_excluded_signal(
            complete, feedback);

    EXPECT_TRUE(evidence.isApprox(complete - feedback, 0.0));
    EXPECT_TRUE(complete.isApprox(original, 0.0));
}

TEST(fruit_loop_recurrence,
     mapdiag_feedback_excluded_evidence_rejects_grid_or_support_mismatch) {
    Eigen::MatrixXd complete = Eigen::MatrixXd::Ones(2, 2);
    Eigen::MatrixXd wrong_grid = Eigen::MatrixXd::Ones(3, 2);
    EXPECT_THROW(
        citlali::pipeline::make_mapdiag_feedback_excluded_signal(
            complete, wrong_grid),
        std::runtime_error);

    Eigen::MatrixXd wrong_support = Eigen::MatrixXd::Ones(2, 2);
    wrong_support(0, 1) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
        citlali::pipeline::make_mapdiag_feedback_excluded_signal(
            complete, wrong_support),
        std::runtime_error);
}

}  // namespace
