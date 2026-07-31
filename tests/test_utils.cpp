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
    map.weight = {Eigen::MatrixXd::Constant(2, 2, 100.0)};
    map.weight[0](1, 1) = 1.0;

    filter.filter_maps(map, 0);

    // The low-weight sample is still part of the convolution, so its
    // variance must be propagated even though it is below cov_cut * max(W).
    // Var_out=(1/16)(3/100 + 1), hence W_out=16/1.03.
    EXPECT_LT((map.weight[0].array() - (16.0 / 1.03)).abs().maxCoeff(),
              1e-10);
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
