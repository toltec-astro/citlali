#include <gtest/gtest.h>

#include <citlali/core/mapmaking/naive_mm.h>
#include <citlali/core/pipeline/fruit_loop_injected_source_test.h>
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
     precomputed_detector_pointing_reuses_exact_feedback_arithmetic) {
    auto calib = make_calib();
    auto legacy = make_scan();
    for (Eigen::Index sample = 0; sample < legacy.scans.data.rows();
         ++sample) {
        legacy.scans.data(sample, 0) =
            std::sin(0.17 * static_cast<double>(sample)) + 1.5;
        legacy.kernel.data(sample, 0) =
            std::cos(0.23 * static_cast<double>(sample)) - 0.4;
    }
    legacy.flags.data(3, 0) = true;
    auto native = legacy;
    native.pointing.data["lat"] = legacy.tel_data.data.at("alt_phys");
    native.pointing.data["lon"] = legacy.tel_data.data.at("az_phys");

    auto model = make_map_buffer(true);
    model.signal[0] = gaussian(5.5, 1.35);
    model.kernel[0] = gaussian(0.6, 1.7);
    Eigen::VectorXi map_indices(1);
    map_indices << 0;
    auto legacy_processor = make_processor();
    auto native_processor = make_processor();
    long long legacy_count = -1;
    long long native_count = -1;

    legacy_processor.map_to_tod<
        timestream::TCProc::SourceType::NegativeMap>(
        model, legacy, calib, map_indices, "altaz", "array",
        &legacy_count);
    native_processor.map_to_tod<
        timestream::TCProc::SourceType::NegativeMap>(
        model, native, calib, map_indices, "altaz", "array",
        &native_count);

    EXPECT_TRUE(native.scans.data.isApprox(legacy.scans.data, 0.0));
    EXPECT_TRUE(native.kernel.data.isApprox(legacy.kernel.data, 0.0));
    EXPECT_EQ(native_count, legacy_count);
    EXPECT_GT(native_count, 0);

    native.pointing.data.erase("lon");
    EXPECT_THROW(
        native_processor.map_to_tod<
            timestream::TCProc::SourceType::Map>(
            model, native, calib, map_indices, "altaz", "array"),
        std::logic_error);
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

}  // namespace
