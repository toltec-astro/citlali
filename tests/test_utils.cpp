#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <tula/logging.h>
#include <tula/algorithm/ei_iterclip.h>
#include <tula/algorithm/index.h>
#include <tula/formatter/matrix.h>
#if defined(CITLALI_USE_WIENER_FILTER_OMP)
#include "citlali/core/mapmaking/wiener_filter_omp.h"
#else
#include "citlali/core/mapmaking/wiener_filter.h"
#endif
#include "citlali/core/mapmaking/map.h"
#include "citlali/core/pipeline/fits_image_metadata.h"
#include "citlali/core/pipeline/map_image_output_helpers.h"
#include "citlali/core/timestream/rtc/calibrate.h"
#include "citlali/core/utils/pointing.h"
#include "citlali/core/utils/utils.h"
#include "citlali/core/timestream/rtc/filter.h"
#include <spdlog/sinks/null_sink.h>

namespace {
using namespace ::testing;

std::shared_ptr<spdlog::logger> ensure_citlali_logger() {
    auto logger = spdlog::get("citlali_logger");
    if (logger == nullptr) {
        auto sink = std::make_shared<spdlog::sinks::null_sink_mt>();
        logger = std::make_shared<spdlog::logger>("citlali_logger", sink);
        spdlog::register_logger(logger);
    }
    return logger;
}

void configure_small_wiener_denominator(mapmaking::WienerFilter &filter) {
    filter.logger = ensure_citlali_logger();
    filter.n_rows = 4;
    filter.n_cols = 4;
    filter.uniform_weight = false;
    filter.denom_limit = -1e300;
    filter.max_loops = 100;

    filter.filter_template.resize(4, 4);
    filter.filter_template <<
        1.0, 0.6, 0.3, 0.1,
        0.6, 0.4, 0.2, 0.1,
        0.3, 0.2, 0.1, 0.05,
        0.1, 0.1, 0.05, 0.02;

    filter.rr.resize(4, 4);
    filter.rr <<
        1.0, 1.1, 0.9, 1.2,
        0.8, 1.3, 1.0, 0.7,
        1.2, 0.9, 1.4, 1.1,
        0.7, 1.0, 1.2, 0.8;

    filter.vvq.resize(4, 4);
    filter.vvq <<
        1.0, 1.4, 0.8, 1.7,
        1.2, 0.9, 1.6, 1.1,
        0.7, 1.5, 1.3, 0.95,
        1.8, 1.05, 0.85, 1.25;
}

struct KernelTemplateMap {
    int n_rows = 5;
    int n_cols = 5;
    Eigen::VectorXd rows_tan_vec;
    Eigen::VectorXd cols_tan_vec;
    std::vector<Eigen::MatrixXd> kernel;
    std::vector<Eigen::MatrixXd> weight;

    KernelTemplateMap() {
        rows_tan_vec.resize(5);
        cols_tan_vec.resize(5);
        rows_tan_vec << -2.0, -1.0, 0.0, 1.0, 2.0;
        cols_tan_vec << -2.0, -1.0, 0.0, 1.0, 2.0;
        kernel.emplace_back(Eigen::MatrixXd::Ones(5, 5));
        weight.emplace_back(Eigen::MatrixXd::Ones(5, 5));
    }
};

struct DummyCalibData {};

struct CalibrationFixture {
    Eigen::VectorXd flux_conversion_factor;
    std::map<std::string, Eigen::VectorXd> apt;
};

struct MetadataHdu {
    std::map<std::string, std::string> string_keys;
    std::map<std::string, double> double_keys;
    std::map<std::string, long long> integer_keys;
    std::map<std::string, bool> bool_keys;

    void addKey(const std::string &name, const std::string &value,
                const std::string &) {
        string_keys[name] = value;
    }

    void addKey(const std::string &name, double value,
                const std::string &) {
        double_keys[name] = value;
    }

    void addKey(const std::string &name, long long value,
                const std::string &) {
        integer_keys[name] = value;
    }

    void addKey(const std::string &name, int value,
                const std::string &) {
        integer_keys[name] = value;
    }

    void addKey(const std::string &name, bool value,
                const std::string &) {
        bool_keys[name] = value;
    }
};

struct MetadataFitsEntry {
    std::string filepath = "metadata-test.fits";
    std::vector<std::string> hdu_names;
    std::vector<std::shared_ptr<MetadataHdu>> hdus;

    template <class Data>
    void add_hdu(const std::string &name, Data &) {
        hdu_names.push_back(name);
        hdus.push_back(std::make_shared<MetadataHdu>());
    }

    template <class Hdu, class Wcs>
    void add_wcs(Hdu &, const Wcs &, double) {}
};

mapmaking::MapBuffer make_source_finder_map(const Eigen::MatrixXd &signal) {
    mapmaking::MapBuffer map{"source-finder-test"};
    map.n_rows = signal.rows();
    map.n_cols = signal.cols();
    map.pixel_size_rad = ASEC_TO_RAD;
    map.cov_cut = 0.0;
    map.source_sigma = 5.0;
    map.source_window_rad = 2.0 * ASEC_TO_RAD;
    map.signal = {signal};
    map.weight = {Eigen::MatrixXd::Ones(signal.rows(), signal.cols())};
    map.n_sources = {0};
    map.row_source_locs.resize(1);
    map.col_source_locs.resize(1);
    return map;
}

mapmaking::MapBuffer make_noise_product_map(
    const std::vector<double> &realizations) {
    mapmaking::MapBuffer map{"noise-product-test"};
    map.n_rows = 1;
    map.n_cols = 1;
    map.n_noise = static_cast<Eigen::Index>(realizations.size());
    map.cov_cut = 0.0;
    map.signal = {Eigen::MatrixXd::Ones(1, 1)};
    map.weight = {Eigen::MatrixXd::Ones(1, 1)};
    map.noise.emplace_back(1, 1, map.n_noise);
    for (Eigen::Index i = 0; i < map.n_noise; ++i) {
        map.noise[0](0, 0, i) = realizations[static_cast<std::size_t>(i)];
    }
    return map;
}

TEST(calibration, beam_flux_to_rj_temperature_has_expected_scale) {
    const double result = engine_utils::mJy_beam_to_uK(1.0, 270e9, 10.0);

    EXPECT_NEAR(result, 167.643394204913, 1e-9);
    EXPECT_NEAR(engine_utils::mJy_beam_to_uK(2.0, 270e9, 10.0),
                2.0 * result, 1e-9);
    EXPECT_NEAR(engine_utils::mJy_beam_to_uK(1.0, 540e9, 10.0),
                result / 4.0, 1e-9);
    EXPECT_NEAR(engine_utils::mJy_beam_to_uK(1.0, 270e9, 20.0),
                result / 4.0, 1e-9);
    EXPECT_THROW(engine_utils::mJy_beam_to_uK(1.0, 0.0, 10.0),
                 std::runtime_error);
}

TEST(calibration, applies_detector_specific_flux_factors) {
    timestream::TCData<timestream::TCDataKind::RTC, Eigen::MatrixXd> data;
    data.scans.data.resize(2, 2);
    data.scans.data << 1.0, 10.0,
                       2.0, 20.0;
    data.fcf.data.resize(2);

    CalibrationFixture calib;
    calib.flux_conversion_factor.resize(2);
    calib.flux_conversion_factor << 2.0, 3.0;
    calib.apt["flxscale"].resize(2);
    calib.apt["flxscale"] << 5.0, 7.0;

    timestream::Calibration calibration;
    calibration.calibrate_tod(data, calib);

    EXPECT_TRUE(data.fcf.data.isApprox(calib.flux_conversion_factor));
    EXPECT_TRUE(data.scans.data.col(0).isApprox(
        (Eigen::Vector2d() << 10.0, 20.0).finished()));
    EXPECT_TRUE(data.scans.data.col(1).isApprox(
        (Eigen::Vector2d() << 210.0, 420.0).finished()));
}

TEST(pointing, rotates_detector_offsets_in_altaz_frame) {
    std::map<std::string, Eigen::VectorXd> telescope;
    telescope["TelElAct"] = Eigen::Vector2d::Zero();
    telescope["alt_phys"] =
        (Eigen::Vector2d() << 0.1, 0.2).finished();
    telescope["az_phys"] =
        (Eigen::Vector2d() << -0.1, -0.2).finished();
    std::map<std::string, Eigen::VectorXd> offsets;
    offsets["az"] = Eigen::Vector2d::Constant(1.0);
    offsets["alt"] = Eigen::Vector2d::Constant(-1.0);

    auto [lat, lon] = engine_utils::calc_det_pointing(
        telescope, 2.0, 3.0, std::string{"altaz"}, offsets,
        citlali::config::MapGrouping::array);

    const Eigen::VectorXd expected_lat =
        telescope["alt_phys"].array() + 2.0 * ASEC_TO_RAD;
    const Eigen::VectorXd expected_lon =
        telescope["az_phys"].array() + 3.0 * ASEC_TO_RAD;
    EXPECT_TRUE(lat.isApprox(expected_lat));
    EXPECT_TRUE(lon.isApprox(expected_lon));
}

TEST(source_finder, negative_no_detection_does_not_mutate_signal) {
    const Eigen::MatrixXd signal = Eigen::MatrixXd::Ones(3, 3);
    auto map = make_source_finder_map(signal);
    map.source_finder_mode = "negative";

    EXPECT_FALSE(map.find_sources(0));
    EXPECT_TRUE(map.signal[0].isApprox(signal));
}

TEST(source_finder, detects_source_on_map_edge_without_out_of_bounds_access) {
    Eigen::MatrixXd signal = Eigen::MatrixXd::Zero(3, 3);
    signal(0, 0) = 10.0;
    auto map = make_source_finder_map(signal);
    map.source_finder_mode = "positive";

    EXPECT_TRUE(map.find_sources(0));
    ASSERT_EQ(map.n_sources[0], 1);
    ASSERT_EQ(map.row_source_locs[0].size(), 1);
    EXPECT_EQ(map.row_source_locs[0](0), 0);
    EXPECT_EQ(map.col_source_locs[0](0), 0);
}

TEST(utils, fft2_into_matches_fft2_forward_and_inverse) {
    Eigen::MatrixXcd in(4, 3);
    for (Eigen::Index i = 0; i < in.rows(); ++i) {
        for (Eigen::Index j = 0; j < in.cols(); ++j) {
            in(i, j) = std::complex<double>(0.2 * static_cast<double>(i + 1),
                                            -0.15 * static_cast<double>(j + 2));
        }
    }

    fftw_complex *a = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * in.size()));
    fftw_complex *b = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * in.size()));
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    fftw_plan pf = fftw_plan_dft_2d(in.rows(), in.cols(), a, b, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan pr = fftw_plan_dft_2d(in.rows(), in.cols(), a, b, FFTW_BACKWARD, FFTW_ESTIMATE);
    ASSERT_NE(pf, nullptr);
    ASSERT_NE(pr, nullptr);

    Eigen::MatrixXcd forward_expected = engine_utils::fft2<engine_utils::forward>(in, pf, a, b);
    Eigen::MatrixXcd forward_actual;
    engine_utils::fft2_into<engine_utils::forward>(in, forward_actual, pf, a, b);
    EXPECT_LT((forward_actual - forward_expected).norm(), 1e-12);

    Eigen::MatrixXcd inverse_expected = engine_utils::fft2<engine_utils::inverse>(forward_expected, pr, a, b);
    Eigen::MatrixXcd inverse_actual;
    engine_utils::fft2_into<engine_utils::inverse>(forward_expected, inverse_actual, pr, a, b);
    EXPECT_LT((inverse_actual - inverse_expected).norm(), 1e-12);

    fftw_destroy_plan(pf);
    fftw_destroy_plan(pr);
    fftw_free(a);
    fftw_free(b);
}

TEST(wiener_filter, denominator_honors_max_denom_iters) {
    mapmaking::WienerFilter filter;
    configure_small_wiener_denominator(filter);
    filter.denom_rel_tol = 0.0;
    filter.tail_frac_tol = 0.0;
    filter.denom_check_iters = 16;
    filter.max_denom_iters = 2;

    filter.calc_denominator();

    EXPECT_EQ(filter.last_denom_iters, 2);
    EXPECT_EQ(filter.last_denom_stop_reason, "max_denom_iters");
    EXPECT_TRUE(filter.denom.allFinite());
}

TEST(wiener_filter, denominator_rel_tol_can_stop_before_tail_cap) {
    mapmaking::WienerFilter filter;
    configure_small_wiener_denominator(filter);
    filter.denom_rel_tol = 2.0;
    filter.tail_frac_tol = 0.0;
    filter.denom_check_iters = 1;
    filter.max_denom_iters = 0;

    filter.calc_denominator();

    EXPECT_EQ(filter.last_denom_iters, 1);
    EXPECT_EQ(filter.last_denom_stop_reason, "converged");
    EXPECT_GT(filter.last_denom_tail_frac, filter.tail_frac_tol);
}

TEST(wiener_filter, kernel_template_tail_zero_truncates_extrapolated_pixels) {
    mapmaking::WienerFilter filter;
    filter.logger = ensure_citlali_logger();
    filter.n_rows = 5;
    filter.n_cols = 5;
    filter.diff_rows = 1.0;
    filter.diff_cols = 1.0;
    filter.kernel_template_tail_mode = "zero";

    KernelTemplateMap mb;
    DummyCalibData calib_data;
    filter.make_kernel_template(mb, 0, calib_data);

    EXPECT_DOUBLE_EQ(filter.filter_template.minCoeff(), 0.0);
    EXPECT_DOUBLE_EQ(filter.filter_template.maxCoeff(), 1.0);
}

TEST(wiener_filter, unit_sum_convolution_rejects_compensated_template) {
    mapmaking::WienerFilter filter;
    filter.logger = ensure_citlali_logger();
    filter.n_rows = 2;
    filter.n_cols = 2;
    filter.filter_type = "wiener_filter";
    filter.template_type = "kernel";
    filter.filter_template.resize(2, 2);
    filter.filter_template <<
        1.0, -0.98,
        0.0, 0.0;
    filter.filtered_map = Eigen::MatrixXd::Ones(2, 2);

    try {
        filter.run_convolve();
        FAIL() << "expected compensated convolution template to fail";
    } catch (const citlali::error::Error &error) {
        EXPECT_EQ(error.code(), citlali::error::Code::runtime);
        EXPECT_THAT(error.what(),
                    HasSubstr("unsafe unit-sum convolution template"));
        EXPECT_THAT(error.what(), HasSubstr("dc_fraction"));
        EXPECT_THAT(error.what(), HasSubstr("lowpass_only: false"));
    }
}

TEST(wiener_filter, unit_sum_convolution_preserves_well_conditioned_behavior) {
    mapmaking::WienerFilter filter;
    filter.logger = ensure_citlali_logger();
    filter.n_rows = 2;
    filter.n_cols = 2;
    filter.filter_type = "convolve";
    filter.template_type = "gaussian";
    filter.filter_template = Eigen::MatrixXd::Ones(2, 2);
    filter.filtered_map = Eigen::MatrixXd::Constant(2, 2, 3.0);

    ASSERT_NO_THROW(filter.run_convolve());
    EXPECT_LT((filter.nume.array() - 3.0).abs().maxCoeff(), 1e-12);
}

TEST(wiener_filter, convolve_delta_kernel_is_signal_and_variance_identity) {
    mapmaking::WienerFilter filter;
    filter.logger = ensure_citlali_logger();
    filter.n_rows = 2;
    filter.n_cols = 2;
    filter.filter_type = "convolve";
    filter.template_type = "gaussian";
    filter.edge_guard_enabled = false;
    filter.filter_template = Eigen::MatrixXd::Zero(2, 2);
    filter.filter_template(0, 0) = 1.0;

    mapmaking::MapBuffer map{"convolve-delta-test"};
    map.n_rows = 2;
    map.n_cols = 2;
    map.cov_cut = 0.9;
    Eigen::MatrixXd signal(2, 2);
    signal << 1.0, 2.0, 3.0, 4.0;
    Eigen::MatrixXd weight(2, 2);
    weight << 1.0, 2.0, 4.0, 8.0;
    map.signal = {signal};
    map.weight = {weight};

    filter.filter_maps(map, 0);

    EXPECT_LT((map.signal[0] - signal).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_LT((map.weight[0] - weight).cwiseAbs().maxCoeff(), 1e-12);
}

TEST(wiener_filter, convolve_matches_hand_nonuniform_signal_and_variance) {
    mapmaking::WienerFilter filter;
    filter.logger = ensure_citlali_logger();
    filter.n_rows = 2;
    filter.n_cols = 2;
    filter.filter_type = "convolve";
    filter.template_type = "gaussian";
    filter.edge_guard_enabled = false;
    filter.filter_template = Eigen::MatrixXd::Zero(2, 2);
    filter.filter_template(0, 0) = 1.0;
    filter.filter_template(0, 1) = 1.0;

    mapmaking::MapBuffer map{"convolve-nonuniform-test"};
    map.n_rows = 2;
    map.n_cols = 2;
    map.cov_cut = 0.0;
    map.signal.resize(1);
    map.signal[0].resize(2, 2);
    map.signal[0] << 2.0, 6.0, 10.0, 14.0;
    map.weight.resize(1);
    map.weight[0].resize(2, 2);
    map.weight[0] << 1.0, 0.25, 4.0, 1.0;

    filter.filter_maps(map, 0);

    Eigen::MatrixXd expected_signal(2, 2);
    expected_signal << 4.0, 4.0, 12.0, 12.0;
    Eigen::MatrixXd expected_weight(2, 2);
    expected_weight << 0.8, 0.8, 3.2, 3.2;
    EXPECT_LT((map.signal[0] - expected_signal).cwiseAbs().maxCoeff(),
              1e-12);
    EXPECT_LT((map.weight[0] - expected_weight).cwiseAbs().maxCoeff(),
              1e-12);
}

TEST(wiener_filter, convolve_propagates_uniform_diagonal_variance) {
    mapmaking::WienerFilter filter;
    filter.logger = ensure_citlali_logger();
    filter.n_rows = 2;
    filter.n_cols = 2;
    filter.filter_type = "convolve";
    filter.template_type = "gaussian";
    filter.edge_guard_enabled = false;
    filter.filter_template = Eigen::MatrixXd::Ones(2, 2);

    mapmaking::MapBuffer map{"convolve-weight-test"};
    map.n_rows = 2;
    map.n_cols = 2;
    map.cov_cut = 0.0;
    map.signal = {Eigen::MatrixXd::Zero(2, 2)};
    map.weight = {Eigen::MatrixXd::Constant(2, 2, 2.0)};

    filter.filter_maps(map, 0);

    // Unit-sum 2x2 kernel: sum(k^2)=1/4, so W_out=W_in/sum(k^2)=8.
    EXPECT_LT((map.weight[0].array() - 8.0).abs().maxCoeff(), 1e-10);
}

TEST(wiener_filter, convolve_does_not_renormalize_variance_by_valid_support) {
    mapmaking::WienerFilter filter;
    filter.logger = ensure_citlali_logger();
    filter.n_rows = 2;
    filter.n_cols = 2;
    filter.filter_type = "convolve";
    filter.template_type = "gaussian";
    filter.edge_guard_enabled = false;
    filter.filter_template = Eigen::MatrixXd::Ones(2, 2);

    mapmaking::MapBuffer map{"convolve-support-test"};
    map.n_rows = 2;
    map.n_cols = 2;
    map.cov_cut = 0.0;
    map.signal = {Eigen::MatrixXd::Zero(2, 2)};
    map.weight = {Eigen::MatrixXd::Zero(2, 2)};
    map.weight[0](0, 0) = 2.0;

    filter.filter_maps(map, 0);

    // Each periodic output receives k^2 Var = (1/16)(1/2), so W_out=32.
    EXPECT_LT((map.weight[0].array() - 32.0).abs().maxCoeff(), 1e-10);
}

TEST(wiener_filter, convolve_variance_includes_all_positive_weight_inputs) {
    mapmaking::WienerFilter filter;
    filter.logger = ensure_citlali_logger();
    filter.n_rows = 2;
    filter.n_cols = 2;
    filter.filter_type = "convolve";
    filter.template_type = "gaussian";
    filter.edge_guard_enabled = false;
    filter.filter_template = Eigen::MatrixXd::Ones(2, 2);

    mapmaking::MapBuffer map{"convolve-cov-cut-test"};
    map.n_rows = 2;
    map.n_cols = 2;
    map.cov_cut = 0.5;
    map.signal = {Eigen::MatrixXd::Zero(2, 2)};
    map.signal[0](1, 1) = 8.0;
    map.weight = {Eigen::MatrixXd::Constant(2, 2, 100.0)};
    map.weight[0](1, 1) = 1.0;

    filter.filter_maps(map, 0);

    // The below-cov_cut sample enters the fixed signal operator too:
    // every periodic output is (1/4) * 8 = 2.
    EXPECT_LT((map.signal[0].array() - 2.0).abs().maxCoeff(), 1e-12);
    // The low-weight sample is still part of the convolution, so its
    // variance must be propagated even though it is below cov_cut * max(W).
    // Var_out=(1/16)(3/100 + 1), hence W_out=16/1.03.
    EXPECT_LT((map.weight[0].array() - (16.0 / 1.03)).abs().maxCoeff(),
              1e-10);
}

TEST(wiener_filter, convolve_zero_weight_value_is_conditioned_as_fixed) {
    mapmaking::WienerFilter filter;
    filter.logger = ensure_citlali_logger();
    filter.n_rows = 2;
    filter.n_cols = 2;
    filter.filter_type = "convolve";
    filter.template_type = "gaussian";
    filter.edge_guard_enabled = false;
    filter.filter_template = Eigen::MatrixXd::Ones(2, 2);

    mapmaking::MapBuffer map{"convolve-zero-weight-test"};
    map.n_rows = 2;
    map.n_cols = 2;
    map.cov_cut = 0.0;
    map.signal = {Eigen::MatrixXd::Zero(2, 2)};
    map.signal[0](1, 1) = 8.0;
    map.weight = {Eigen::MatrixXd::Constant(2, 2, 2.0)};
    map.weight[0](1, 1) = 0.0;

    filter.filter_maps(map, 0);

    // Low-level filtering retains the deterministic value in the affine
    // signal offset, so every periodic output is 2.  Formal covariance has
    // no variance model for weight zero and therefore includes only the
    // three stochastic W=2 contributors: V=3*(1/16)*(1/2)=3/32.
    EXPECT_LT((map.signal[0].array() - 2.0).abs().maxCoeff(), 1e-12);
    EXPECT_LT((map.weight[0].array() - (32.0 / 3.0)).abs().maxCoeff(),
              1e-10);
}

TEST(wiener_filter, convolve_core_median_fill_and_output_mask_match_equation) {
    mapmaking::WienerFilter filter;
    filter.logger = ensure_citlali_logger();
    filter.n_rows = 3;
    filter.n_cols = 3;
    filter.filter_type = "convolve";
    filter.template_type = "gaussian";
    filter.edge_guard_enabled = true;
    filter.edge_guard_radius_fwhm = 1.0;
    filter.init_fwhm = 1.0;
    filter.edge_taper_mode = "none";
    filter.edge_fill_mode = "core_median";
    filter.filter_template = Eigen::MatrixXd::Ones(3, 3);

    mapmaking::MapBuffer map{"convolve-edge-fill-test"};
    map.n_rows = 3;
    map.n_cols = 3;
    map.cov_cut = 0.5;
    map.signal.resize(1);
    map.signal[0].resize(3, 3);
    map.signal[0] << 100.0, 2.0, 101.0,
                     3.0, 10.0, 4.0,
                     102.0, 5.0, 103.0;
    map.weight = {Eigen::MatrixXd::Zero(3, 3)};
    map.weight[0](1, 1) = 4.0;

    filter.filter_maps(map, 0);

    // The science median is 10.  Radius-one binary support is the center
    // plus its four axial neighbors.  Corners are filled with 10 before the
    // global 3x3 circular mean, giving (10+2+3+4+5+4*10)/9=64/9, then the
    // same binary window masks output corners.
    Eigen::MatrixXd expected_signal = Eigen::MatrixXd::Zero(3, 3);
    expected_signal(0, 1) = 64.0 / 9.0;
    expected_signal(1, 0) = 64.0 / 9.0;
    expected_signal(1, 1) = 64.0 / 9.0;
    expected_signal(1, 2) = 64.0 / 9.0;
    expected_signal(2, 1) = 64.0 / 9.0;
    Eigen::MatrixXd expected_weight = Eigen::MatrixXd::Zero(3, 3);
    expected_weight(0, 1) = 324.0;
    expected_weight(1, 0) = 324.0;
    expected_weight(1, 1) = 324.0;
    expected_weight(1, 2) = 324.0;
    expected_weight(2, 1) = 324.0;
    EXPECT_DOUBLE_EQ(map.edge_guard_background_level[0], 10.0);
    EXPECT_LT((map.signal[0] - expected_signal).cwiseAbs().maxCoeff(),
              1e-10);
    EXPECT_LT((map.weight[0] - expected_weight).cwiseAbs().maxCoeff(),
              1e-8);
}

TEST(wiener_filter, convolve_zero_fill_is_deterministic_and_output_masked) {
    mapmaking::WienerFilter filter;
    filter.logger = ensure_citlali_logger();
    filter.n_rows = 3;
    filter.n_cols = 3;
    filter.filter_type = "convolve";
    filter.template_type = "gaussian";
    filter.edge_guard_enabled = true;
    filter.edge_guard_radius_fwhm = 1.0;
    filter.init_fwhm = 1.0;
    filter.edge_taper_mode = "none";
    filter.edge_fill_mode = "zero";
    filter.filter_template = Eigen::MatrixXd::Ones(3, 3);

    mapmaking::MapBuffer map{"convolve-zero-fill-test"};
    map.n_rows = 3;
    map.n_cols = 3;
    map.cov_cut = 0.5;
    map.signal.resize(1);
    map.signal[0].resize(3, 3);
    map.signal[0] << 100.0, 2.0, 101.0,
                     3.0, 10.0, 4.0,
                     102.0, 5.0, 103.0;
    map.weight = {Eigen::MatrixXd::Zero(3, 3)};
    map.weight[0](1, 1) = 4.0;

    filter.filter_maps(map, 0);

    // Exterior values are deterministically replaced by zero, so the global
    // mean is (2+3+10+4+5)/9=8/3 before output masking.
    Eigen::MatrixXd expected_signal = Eigen::MatrixXd::Zero(3, 3);
    expected_signal(0, 1) = 8.0 / 3.0;
    expected_signal(1, 0) = 8.0 / 3.0;
    expected_signal(1, 1) = 8.0 / 3.0;
    expected_signal(1, 2) = 8.0 / 3.0;
    expected_signal(2, 1) = 8.0 / 3.0;
    EXPECT_DOUBLE_EQ(map.edge_guard_background_level[0], 0.0);
    EXPECT_LT((map.signal[0] - expected_signal).cwiseAbs().maxCoeff(),
              1e-10);
}

TEST(wiener_filter, convolve_noise_uses_the_same_binary_stochastic_operator) {
    mapmaking::WienerFilter filter;
    filter.logger = ensure_citlali_logger();
    filter.n_rows = 3;
    filter.n_cols = 3;
    filter.filter_type = "convolve";
    filter.template_type = "gaussian";
    filter.filter_template = Eigen::MatrixXd::Ones(3, 3);

    mapmaking::MapBuffer map{"convolve-noise-edge-test"};
    map.n_rows = 3;
    map.n_cols = 3;
    map.n_noise = 1;
    map.noise.emplace_back(3, 3, 1);
    Eigen::MatrixXd input(3, 3);
    input << 100.0, 2.0, 101.0,
             3.0, 10.0, 4.0,
             102.0, 5.0, 103.0;
    for (Eigen::Index r = 0; r < 3; ++r) {
        for (Eigen::Index c = 0; c < 3; ++c) {
            map.noise[0](r, c, 0) = input(r, c);
        }
    }
    Eigen::MatrixXd edge_window = Eigen::MatrixXd::Zero(3, 3);
    edge_window(0, 1) = 1.0;
    edge_window(1, 0) = 1.0;
    edge_window(1, 1) = 1.0;
    edge_window(1, 2) = 1.0;
    edge_window(2, 1) = 1.0;
    map.edge_guard_window = {edge_window};

    filter.filter_noise(map, 0, 0);

    // Noise uses O C_k T n.  It is not given the signal map's deterministic
    // median fill, so the retained cross sums to 24 and the circular 3x3
    // mean is 8/3 before the same output window is applied.
    Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(3, 3);
    expected(0, 1) = 8.0 / 3.0;
    expected(1, 0) = 8.0 / 3.0;
    expected(1, 1) = 8.0 / 3.0;
    expected(1, 2) = 8.0 / 3.0;
    expected(2, 1) = 8.0 / 3.0;
    Eigen::Map<Eigen::MatrixXd> actual(map.noise[0].data(), 3, 3);
    EXPECT_LT((actual - expected).cwiseAbs().maxCoeff(), 1e-10);
}

TEST(wiener_filter, numerical_support_floor_is_strict_and_not_scientific) {
    constexpr double kernel_square_sum = 0.25;
    const double boundary =
        mapmaking::convolve_numerical_support_floor(kernel_square_sum);

    EXPECT_FALSE(mapmaking::convolve_has_numerical_variance_support(
        boundary, kernel_square_sum));
    EXPECT_TRUE(mapmaking::convolve_has_numerical_variance_support(
        std::nextafter(boundary, std::numeric_limits<double>::infinity()),
        kernel_square_sum));
    EXPECT_FALSE(mapmaking::convolve_has_numerical_variance_support(
        std::numeric_limits<double>::quiet_NaN(), kernel_square_sum));
    EXPECT_DOUBLE_EQ(
        boundary / kernel_square_sum,
        mapmaking::convolve_numerical_support_fraction_floor);
}

TEST(map_noise_products, mean_subtracted_variance_uses_n_minus_one) {
    auto map = make_noise_product_map({1.0, 3.0});

    map.calc_noise_products(false);

    EXPECT_DOUBLE_EQ(map.noise_mean[0](0, 0), 2.0);
    EXPECT_NEAR(map.noise_variance[0](0, 0), 2.0, 1e-12);
    EXPECT_NEAR(map.point_source_uncertainty[0](0, 0), std::sqrt(2.0),
                1e-12);
}

TEST(map_noise_products, mean_subtracted_variance_is_stable_at_large_offset) {
    auto map = make_noise_product_map({1.0e12 + 1.0, 1.0e12 + 3.0});

    map.calc_noise_products(false);

    EXPECT_DOUBLE_EQ(map.noise_mean[0](0, 0), 1.0e12 + 2.0);
    EXPECT_NEAR(map.noise_variance[0](0, 0), 2.0, 1e-12);
}

TEST(map_noise_products, mean_subtracted_products_require_two_realizations) {
    auto map = make_noise_product_map({3.0});

    EXPECT_THROW(map.calc_noise_products(false), std::invalid_argument);
}

TEST(map_noise_products, known_zero_mean_second_moment_allows_one_realization) {
    auto map = make_noise_product_map({3.0});

    ASSERT_NO_THROW(map.calc_noise_products(false, false));
    EXPECT_DOUBLE_EQ(map.noise_mean[0](0, 0), 3.0);
    EXPECT_DOUBLE_EQ(map.noise_variance[0](0, 0), 9.0);
}

TEST(map_noise_products, convolved_amplitude_metadata_is_not_photometric) {
    MetadataHdu flux_hdu;
    MetadataHdu uncertainty_hdu;
    MetadataHdu snr_hdu;

    citlali::pipeline::add_point_source_flux_map_metadata(
        flux_hdu, "mJy/beam", false);
    citlali::pipeline::add_point_source_uncertainty_map_metadata(
        uncertainty_hdu, "mJy/beam", false);
    citlali::pipeline::add_point_source_snr_map_metadata(snr_hdu, false);

    EXPECT_EQ(flux_hdu.string_keys["BUNIT"], "mJy/beam");
    EXPECT_EQ(flux_hdu.string_keys["TYPE"], "convolved_amplitude");
    EXPECT_THAT(flux_hdu.string_keys["DESCRIP"],
                HasSubstr("no point-source response normalization"));
    EXPECT_EQ(uncertainty_hdu.string_keys["BUNIT"], "mJy/beam");
    EXPECT_EQ(uncertainty_hdu.string_keys["TYPE"],
              "convolved_amplitude");
    EXPECT_EQ(snr_hdu.string_keys["BUNIT"], "N/A");
    EXPECT_EQ(snr_hdu.string_keys["TYPE"], "convolved_amplitude");
}

TEST(map_noise_products, convolved_products_record_conditional_contract) {
    MetadataHdu hdu;

    citlali::pipeline::add_signal_map_metadata(
        hdu, "mJy/beam", true);
    citlali::pipeline::add_filtered_map_operator_identity_key(
        hdu, "unit_sum_convolution");
    citlali::pipeline::add_convolved_map_contract_keys(
        hdu, true, true, "core_median", true);

    EXPECT_EQ(hdu.string_keys["BUNIT"], "mJy/beam");
    EXPECT_EQ(hdu.string_keys["TYPE"], "convolved_amplitude");
    EXPECT_EQ(hdu.string_keys["BOUNDARY"], "circular");
    EXPECT_EQ(hdu.string_keys["FILLMODE"], "core_median");
    EXPECT_EQ(hdu.string_keys["FILTEROP"], "unit_sum_convolution");
    EXPECT_TRUE(hdu.bool_keys["CONDMASK"]);
    EXPECT_TRUE(hdu.bool_keys["CONDFILL"]);
    EXPECT_TRUE(hdu.bool_keys["ZEROWFIX"]);
    EXPECT_TRUE(hdu.bool_keys["COVDIAG"]);
    EXPECT_FALSE(hdu.bool_keys["RESPCORR"]);
    EXPECT_FALSE(hdu.bool_keys["FLFBACK"]);
    EXPECT_EQ(hdu.string_keys["FLWHY"],
              "support_contract_unresolved");
}

TEST(map_noise_products, weight_metadata_distinguishes_formal_and_calibrated) {
    MetadataHdu formal_hdu;
    MetadataHdu calibrated_hdu;

    citlali::pipeline::add_formal_weight_map_metadata(
        formal_hdu, "1/(mJy/beam)^2");
    citlali::pipeline::add_formal_weight_provenance_key(formal_hdu);
    citlali::pipeline::add_weight_map_metadata(
        calibrated_hdu, "1/(mJy/beam)^2", true);
    citlali::pipeline::add_empirical_weight_calibration_model_key(
        calibrated_hdu);

    EXPECT_EQ(formal_hdu.string_keys["TYPE"], "formal");
    EXPECT_EQ(formal_hdu.string_keys["WPROV"],
              "stage_input_snapshot");
    EXPECT_THAT(formal_hdu.string_keys["DESCRIP"],
                HasSubstr("conditional diagonal"));
    EXPECT_EQ(calibrated_hdu.string_keys["TYPE"], "empirical");
    EXPECT_EQ(calibrated_hdu.string_keys["CALMODEL"],
              "global_scalar");
    EXPECT_THAT(calibrated_hdu.string_keys["DESCRIP"],
                HasSubstr("global jackknife scalar"));
}

TEST(map_noise_products, coverage_mask_metadata_disclaims_support) {
    MetadataHdu hdu;

    citlali::pipeline::add_coverage_mask_map_metadata(hdu);

    EXPECT_EQ(hdu.string_keys["BUNIT"], "N/A");
    EXPECT_THAT(hdu.string_keys["DESCRIP"],
                HasSubstr("not convolution support"));
    EXPECT_THAT(hdu.string_keys["DESCRIP"],
                HasSubstr("not complete validity"));
}

TEST(map_noise_products, formal_standardized_signal_uses_formal_weight_snapshot) {
    mapmaking::MapBuffer map{"formal-standardized-signal-test"};
    map.n_rows = 1;
    map.n_cols = 2;
    map.signal = {Eigen::MatrixXd::Constant(1, 2, 3.0)};
    map.weight = {Eigen::MatrixXd::Constant(1, 2, 4.0)};
    map.weight_formal = {Eigen::MatrixXd::Ones(1, 2)};

    const auto &formal_weight =
        citlali::pipeline::formal_weight_for_standardized_signal(map, 0);
    const Eigen::MatrixXd standardized =
        citlali::pipeline::standardized_signal_from_weight(
            map.signal[0], formal_weight);

    EXPECT_EQ(&formal_weight, &map.weight_formal[0]);
    EXPECT_TRUE(standardized.isApprox(map.signal[0]));
}

TEST(map_noise_products, formal_standardized_signal_falls_back_to_current_weight) {
    mapmaking::MapBuffer map{"formal-standardized-signal-fallback-test"};
    map.n_rows = 1;
    map.n_cols = 1;
    map.signal = {Eigen::MatrixXd::Constant(1, 1, 3.0)};
    map.weight = {Eigen::MatrixXd::Constant(1, 1, 4.0)};

    const auto &formal_weight =
        citlali::pipeline::formal_weight_for_standardized_signal(map, 0);
    const Eigen::MatrixXd standardized =
        citlali::pipeline::standardized_signal_from_weight(
            map.signal[0], formal_weight);

    EXPECT_EQ(&formal_weight, &map.weight[0]);
    EXPECT_DOUBLE_EQ(standardized(0, 0), 6.0);
}

TEST(fruit_loop_feedback, explicit_product_withholding_is_enforced) {
    EXPECT_THROW(
        citlali::pipeline::require_filtered_fruit_loop_feedback_product_contract(
            true, false, true, "unit_sum_convolution", "signal_I"),
        citlali::error::Error);
    EXPECT_NO_THROW(
        citlali::pipeline::require_filtered_fruit_loop_feedback_product_contract(
            true, true, true, "wiener_filter", "signal_I"));
    EXPECT_THROW(
        citlali::pipeline::require_filtered_fruit_loop_feedback_product_contract(
            false, false, true, "unit_sum_convolution", "signal_I"),
        citlali::error::Error);
    EXPECT_NO_THROW(
        citlali::pipeline::require_filtered_fruit_loop_feedback_product_contract(
            false, false, true, "wiener_filter", "signal_I"));
    EXPECT_THROW(
        citlali::pipeline::require_filtered_fruit_loop_feedback_product_contract(
            false, false, true, "destripe", "signal_I"),
        citlali::error::Error);
    EXPECT_THROW(
        citlali::pipeline::require_filtered_fruit_loop_feedback_product_contract(
            false, false, false, "", "legacy_signal_I"),
        citlali::error::Error);
}

TEST(map_noise_products, filtered_operator_identity_is_writer_routing_fact) {
    EXPECT_TRUE(citlali::pipeline::filtered_map_operator_identity(
                    false, false, "wiener_filter")
                    .empty());
    EXPECT_EQ(citlali::pipeline::filtered_map_operator_identity(
                  true, true, "convolve"),
              "unit_sum_convolution");
    EXPECT_EQ(citlali::pipeline::filtered_map_operator_identity(
                  true, true, "wiener_filter"),
              "unit_sum_convolution");
    EXPECT_EQ(citlali::pipeline::filtered_map_operator_identity(
                  true, false, "wiener_filter"),
              "wiener_filter");
}

TEST(map_noise_products, primary_writer_routes_filtered_operator_contract) {
    auto map = std::make_shared<mapmaking::MapBuffer>("writer-routing-test");
    map->n_rows = 1;
    map->n_cols = 1;
    map->n_noise = 0;
    map->sig_unit = "mJy/beam";
    map->signal = {Eigen::MatrixXd::Constant(1, 1, 2.0)};
    map->weight = {Eigen::MatrixXd::Constant(1, 1, 4.0)};
    map->median_err = Eigen::VectorXd::Zero(1);
    const int unused_wcs = 0;

    MetadataFitsEntry convolved_entry;
    citlali::pipeline::add_primary_map_image_hdus(
        convolved_entry, map, 0, "", "I", unused_wcs, 2000.0,
        false, false, false, "unit_sum_convolution",
        citlali::pipeline::ConvolvedMapOutputContract{
            true, false, false, "none"},
        ensure_citlali_logger());
    ASSERT_EQ(convolved_entry.hdus.size(), 2);
    EXPECT_EQ(convolved_entry.hdu_names[0], "signal_I");
    EXPECT_EQ(convolved_entry.hdus[0]->string_keys["FILTEROP"],
              "unit_sum_convolution");
    EXPECT_FALSE(convolved_entry.hdus[0]->bool_keys["FLFBACK"]);

    MetadataFitsEntry full_wiener_entry;
    citlali::pipeline::add_primary_map_image_hdus(
        full_wiener_entry, map, 0, "", "I", unused_wcs, 2000.0,
        false, false, false, "wiener_filter",
        citlali::pipeline::ConvolvedMapOutputContract{},
        ensure_citlali_logger());
    ASSERT_EQ(full_wiener_entry.hdus.size(), 2);
    EXPECT_EQ(full_wiener_entry.hdus[0]->string_keys["FILTEROP"],
              "wiener_filter");
    EXPECT_EQ(full_wiener_entry.hdus[0]->bool_keys.count("FLFBACK"), 0);
}

TEST(map_noise_products, coverage_bool_is_exact_legacy_weight_comparison) {
    Eigen::MatrixXd weight(1, 3);
    weight << 0.0, std::numeric_limits<double>::quiet_NaN(), 1.0;

    const Eigen::MatrixXd at_zero =
        citlali::pipeline::coverage_mask_from_weight(weight, 0.0);
    const Eigen::MatrixXd at_half =
        citlali::pipeline::coverage_mask_from_weight(weight, 0.5);

    EXPECT_DOUBLE_EQ(at_zero(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(at_zero(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(at_zero(0, 2), 1.0);
    EXPECT_DOUBLE_EQ(at_half(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(at_half(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(at_half(0, 2), 1.0);
}

TEST(map_noise_products, noise_realization_metadata_has_physical_unit) {
    MetadataHdu hdu;

    citlali::pipeline::add_noise_image_summary_keys(
        hdu, "mJy/beam", 2.5);

    EXPECT_EQ(hdu.string_keys["UNIT"], "mJy/beam");
    EXPECT_EQ(hdu.string_keys["BUNIT"], "mJy/beam");
    EXPECT_EQ(hdu.string_keys["TYPE"], "noise_realization");
    EXPECT_DOUBLE_EQ(hdu.double_keys["MEDRMS"], 2.5);
}

TEST(map_noise_products, point_source_flux_compatibility_alias_is_exact) {
    mapmaking::MapBuffer map{"convolved-alias-test"};
    map.signal.resize(1);
    map.signal[0].resize(2, 2);
    map.signal[0] << 1.0, 2.0, 3.0, 4.0;

    auto &alias =
        citlali::pipeline::convolved_amplitude_compatibility_alias(map, 0);

    EXPECT_EQ(&alias, &map.signal[0]);
    EXPECT_LT((alias - map.signal[0]).cwiseAbs().maxCoeff(), 1e-15);
}

TEST(map_noise_products, empirical_variance_metadata_records_estimator) {
    MetadataHdu hdu;

    citlali::pipeline::add_empirical_variance_estimator_keys(hdu, 17);

    EXPECT_EQ(hdu.integer_keys["NNOISE"], 17);
    EXPECT_EQ(hdu.string_keys["VAREST"], "central_sample_variance");
    EXPECT_EQ(hdu.integer_keys["VARDDOF"], 1);
    EXPECT_TRUE(hdu.bool_keys["MEANSUB"]);
}

TEST(timestream_filter, notch_settle_samples_are_positive_for_narrow_notches) {
    auto samples = timestream::Filter::notch_settle_samples_for_width(
        122.0703125, 0.25, 0.01);
    EXPECT_GT(samples, 0);
}

TEST(timestream_filter, notch_settle_samples_increase_for_narrower_widths) {
    auto narrow = timestream::Filter::notch_settle_samples_for_width(
        122.0703125, 0.25, 0.01);
    auto broad = timestream::Filter::notch_settle_samples_for_width(
        122.0703125, 1.0, 0.01);
    EXPECT_GT(narrow, broad);
}

TEST(timestream_filter, zero_phase_notch_preserves_constant_timestream) {
    timestream::Filter filter;
    filter.notch_zero_phase = true;
    filter.w0s = {11.03, 15.23, 13.23, 2.71};
    filter.qs = {
        11.03 / 0.25,
        15.23 / 0.30,
        13.23 / 0.25,
        2.71 / 0.20,
    };
    filter.make_notch_filter(122.0703125);

    Eigen::MatrixXd data = Eigen::MatrixXd::Constant(512, 3, 1.0);
    filter.iir(data);

    EXPECT_LT((data.array() - 1.0).abs().maxCoeff(), 1e-10);
}

TEST(timestream_filter, zero_phase_notch_preserves_constant_edges) {
    timestream::Filter filter;
    filter.notch_zero_phase = true;
    filter.w0s = {2.71};
    filter.qs = {2.71 / 0.20};
    filter.make_notch_filter(122.0703125);

    Eigen::MatrixXd data = Eigen::MatrixXd::Constant(64, 2, 3.5);
    filter.iir(data);

    EXPECT_NEAR(data(0, 0), 3.5, 1e-10);
    EXPECT_NEAR(data(data.rows() - 1, 0), 3.5, 1e-10);
    EXPECT_LT((data.array() - 3.5).abs().maxCoeff(), 1e-10);
}

class UtilsTest : public Test {
public:
    UtilsTest() : mat(Eigen::MatrixXd::Constant(10, 2, std::nan(""))) {
        const Eigen::VectorXd values =
            Eigen::VectorXd::LinSpaced(10, 1.0, 0.0);
        mat.block(5, 0, 5, 2) =
            Eigen::Map<const Eigen::MatrixXd>(values.data(), 5, 2);
    }

    auto block() { return mat.block(5, 0, 5, 2); }

    Eigen::MatrixXd mat;
};

TEST_F(UtilsTest, generate_chunks) {
    const auto chunks = tula::alg::indexchunks(0, 11, 2, 2);

    EXPECT_EQ(chunks,
              (std::vector<std::pair<int, int>>{{0, 7}, {5, 11}}));
}

TEST_F(UtilsTest, eigen_to_stdvec_iterates_in_storage_order) {
    auto blk = block();
    const auto values = tula::eigen_utils::to_stdvec(blk);

    EXPECT_EQ(std::distance(values.begin(), values.end()), blk.size());
    EXPECT_DOUBLE_EQ(*values.begin(), 1.0);
    EXPECT_DOUBLE_EQ(*(values.end() - 1), 0.0);
}

TEST_F(UtilsTest, eigen_to_stdvec_supports_standard_algorithms) {
    auto blk = block();
    auto values = tula::eigen_utils::to_stdvec(blk);

    EXPECT_TRUE(
        std::is_sorted(values.begin(), values.end(), std::greater<>()));
    std::sort(values.begin(), values.end());
    EXPECT_TRUE(std::is_sorted(values.begin(), values.end()));
}

TEST_F(UtilsTest, vector) {
    auto blk = block();
    const auto column_major = tula::eigen_utils::to_stdvec(blk);
    const auto row_major =
        tula::eigen_utils::to_stdvec(blk, Eigen::RowMajor);

    ASSERT_EQ(column_major.size(), 10U);
    ASSERT_EQ(row_major.size(), 10U);
    EXPECT_DOUBLE_EQ(column_major.front(), 1.0);
    EXPECT_DOUBLE_EQ(column_major.back(), 0.0);
    EXPECT_DOUBLE_EQ(row_major[0], 1.0);
    EXPECT_DOUBLE_EQ(row_major[1], 4.0 / 9.0);
}

TEST_F(UtilsTest, meanstd) {
    auto blk = block();
    auto [m, s] = tula::alg::meanstd(blk);
    std::vector<double> expected{0.5, 0.31914236925211265};
    EXPECT_THAT(expected, ElementsAre(DoubleEq(m), DoubleEq(s)));
    std::tie(std::ignore, s) = tula::alg::meanstd(blk, 1);
    EXPECT_DOUBLE_EQ(0.33640559489972127, s);
}

TEST_F(UtilsTest, medianmad) {
    auto blk = block();
    auto [m, s] = tula::alg::medmad(blk);
    std::vector<double> expected{0.5, 0.27777777777777773};
    EXPECT_THAT(expected, ElementsAre(DoubleEq(m), DoubleEq(s)));
}

TEST_F(UtilsTest, indexofthresh) {
    auto blk = block();
    const auto values = tula::eigen_utils::to_stdvec(blk);
    const Eigen::VectorXd data = Eigen::Map<const Eigen::VectorXd>(
        values.data(), static_cast<Eigen::Index>(values.size()));
    auto func = tula::alg::iterclip(
            [](auto&& m) {
        return tula::alg::meanstd(m);
    },
            [](auto v, auto m, auto s) {
        return std::abs(v - m) > s;
    }
            );
    auto [indices, converged, center, stddev] = func(data);

    EXPECT_TRUE(converged);
    EXPECT_FALSE(indices.empty());
    for (const auto index : indices) {
        EXPECT_GT(std::abs(data.coeff(index) - center), stddev);
    }
}
}  // namespace
