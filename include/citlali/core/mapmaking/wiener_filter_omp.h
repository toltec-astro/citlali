#pragma once

#include <string>
#include <chrono>
#include <algorithm>
#include <map>
#include <vector>
#include <cmath>
#include <omp.h>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/Splines>

#include <boost/math/special_functions/bessel.hpp>

#include <tula/algorithm/mlinterp/mlinterp.hpp>
#include <tula/logging.h>

#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/utils.h>

#include <citlali/core/utils/gauss_models.h>
#include <citlali/core/utils/fitting.h>
#include <citlali/core/utils/toltec_io.h>

namespace mapmaking {

class WienerFilter {
public:
    struct FFTWContext {
        int n_rows = 0;
        int n_cols = 0;
        fftw_complex *a = nullptr;
        fftw_complex *b = nullptr;
        fftw_plan pf = nullptr;
        fftw_plan pr = nullptr;

        void reset() {
            if (pf != nullptr) {
                fftw_destroy_plan(pf);
                pf = nullptr;
            }
            if (pr != nullptr) {
                fftw_destroy_plan(pr);
                pr = nullptr;
            }
            if (a != nullptr) {
                fftw_free(a);
                a = nullptr;
            }
            if (b != nullptr) {
                fftw_free(b);
                b = nullptr;
            }
            n_rows = 0;
            n_cols = 0;
        }

        ~FFTWContext() { reset(); }
    };

    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // filter template
    std::string template_type, filter_type;
    // normalize filtered map errors
    bool normalize_error;
    // uniform weighting
    bool uniform_weight;
    // lowpass only
    bool run_lowpass;

    // number of loops in denom calc
    int n_loops;
    // maximum number of loops for denom calc
    int max_loops = 500;
    // lower limit to zero out denom values
    double denom_limit = 1.e-4;
    // psd limit
    double psd_lim = 1.e-4;
    // denominator convergence tolerances
    double denom_rel_tol = 1e-4;
    double tail_frac_tol = 5e-2;

    // guess fwhm for kernel map filtering
    double init_fwhm;

    // fwhms for gaussian template
    std::map<std::string, double> template_fwhm_rad;

    // size of maps
    int n_rows, n_cols;

    // size of pixel in each dimension
    double diff_rows, diff_cols;

    // parallelization for ffts
    std::string parallel_policy;

    // matrices for main calculations from each function
    Eigen::MatrixXd rr, vvq, denom, nume;
    // temporarily holds the filtered map
    Eigen::MatrixXd filtered_map;
    // filter template
    Eigen::MatrixXd filter_template;
    // cached FFTs of the current filter template
    Eigen::MatrixXcd filter_template_fft;
    Eigen::MatrixXcd filter_template_fft_scaled;
    Eigen::MatrixXcd filter_template_fft_normalized_scaled;
    bool filter_template_fft_valid = false;
    bool filter_template_fft_scaled_valid = false;
    bool filter_template_fft_normalized_scaled_valid = false;

    // declare fitter class
    engine_utils::mapFitter map_fitter;

    // get config file
    template <typename config_t>
    void get_config(config_t &, std::vector<std::vector<std::string>> &, std::vector<std::vector<std::string>> &);

    template<class MB>
    void make_gaussian_template(MB &mb, const double);

    template<class MB>
    void make_airy_template(MB &mb, const double);

    template<class MB, class CD>
    void make_kernel_template(MB &mb, const int, CD &);

    template<class MB, class CD>
    void make_template(MB &mb, CD &calib_data, const double template_fwhm_rad, const int map_index) {
        // map dimensions
        n_rows = mb.n_rows;
        n_cols = mb.n_cols;

        if (n_rows < 2 || n_cols < 2 ||
            mb.rows_tan_vec.size() < 2 || mb.cols_tan_vec.size() < 2) {
            logger->error("invalid map geometry for Wiener template: map_index={} n_rows={} n_cols={} rows_size={} cols_size={}",
                          map_index, n_rows, n_cols,
                          static_cast<long long>(mb.rows_tan_vec.size()),
                          static_cast<long long>(mb.cols_tan_vec.size()));
            std::exit(EXIT_FAILURE);
        }

        // x and y spacing should be equal
        diff_rows = std::abs(mb.rows_tan_vec(1) - mb.rows_tan_vec(0));
        diff_cols = std::abs(mb.cols_tan_vec(1) - mb.cols_tan_vec(0));
        if (!std::isfinite(diff_rows) || !std::isfinite(diff_cols) || diff_rows <= 0.0 || diff_cols <= 0.0) {
            logger->error("invalid tangent-plane spacing for Wiener template: map_index={} diff_rows={} diff_cols={}",
                          map_index, diff_rows, diff_cols);
            std::exit(EXIT_FAILURE);
        }

        // highpass template
        if (template_type=="highpass") {
            logger->info("creating highpass template");
            filter_template.setZero(n_rows,n_cols);
            filter_template(0,0) = 1;
        }

        // gaussian template
        else if (template_type=="gaussian") {
            logger->info("creating gaussian template");
            make_gaussian_template(mb, template_fwhm_rad);
        }

        // airy template
        else if (template_type=="airy") {
            logger->info("creating airy template");
            make_airy_template(mb, template_fwhm_rad);
        }

        // kernel template
        else {
            logger->info("creating template from kernel map");
            if (map_index < 0 || map_index >= static_cast<int>(mb.kernel.size()) ||
                map_index >= static_cast<int>(mb.weight.size())) {
                logger->error("kernel template requested but kernel/weight map index is invalid: map_index={} kernel_size={} weight_size={}",
                              map_index,
                              static_cast<long long>(mb.kernel.size()),
                              static_cast<long long>(mb.weight.size()));
                std::exit(EXIT_FAILURE);
            }
            make_kernel_template(mb, map_index, calib_data);
        }
        invalidate_template_fft_cache();
    }

    template<class MB>
    void calc_rr(MB &mb, const int map_index) {
        if (uniform_weight) {
            rr = Eigen::MatrixXd::Ones(n_rows,n_cols);
        }
        else {
            rr = sqrt(mb.weight[map_index].array());
        }
    }

    template <class MB>
    void calc_vvq(MB &, const int);
    void calc_numerator();
    void calc_denominator();
    void run_convolve(bool normalize=true);
    static FFTWContext &get_thread_fft_context(int, int);
    void invalidate_template_fft_cache() {
        filter_template_fft_valid = false;
        filter_template_fft_scaled_valid = false;
        filter_template_fft_normalized_scaled_valid = false;
    }
    const Eigen::MatrixXcd &get_filter_template_fft();
    const Eigen::MatrixXcd &get_filter_template_fft_scaled(bool);
    Eigen::MatrixXd calc_numerator_from_input(const Eigen::MatrixXd &);
    Eigen::MatrixXd run_convolve_on_input(const Eigen::MatrixXd &, bool);
    Eigen::MatrixXd divide_by_denom(const Eigen::MatrixXd &, const Eigen::MatrixXd &) const;
    void destripe(double);

    template<class MB>
    void run_filter(MB &mb, const int map_index) {
        const auto t0 = std::chrono::steady_clock::now();
        SPDLOG_DEBUG("calculating rr");
        calc_rr(mb, map_index);
        SPDLOG_DEBUG("rr {}", rr);

        const auto t1 = std::chrono::steady_clock::now();
        SPDLOG_DEBUG("calculating vvq");
        calc_vvq(mb, map_index);
        SPDLOG_DEBUG("vvq {}", vvq);

        const auto t2 = std::chrono::steady_clock::now();
        SPDLOG_DEBUG("calculating denominator");
        calc_denominator();
        SPDLOG_DEBUG("denominator {}", denom);

        const auto t3 = std::chrono::steady_clock::now();
        SPDLOG_DEBUG("calculating numerator");
        calc_numerator();
        SPDLOG_DEBUG("numerator {}", nume);
        const auto t4 = std::chrono::steady_clock::now();
        logger->info(
            "Wiener core timings map_index={} rr_s={} vvq_s={} denom_s={} numer_s={} uniform_weight={}",
            map_index,
            std::chrono::duration<double>(t1 - t0).count(),
            std::chrono::duration<double>(t2 - t1).count(),
            std::chrono::duration<double>(t3 - t2).count(),
            std::chrono::duration<double>(t4 - t3).count(),
            uniform_weight);
    }

    template<class MB>
    void filter_maps(MB &mb, const int map_index) {
        const bool use_convolve = (filter_type=="convolve") || (filter_type=="wiener_filter" && run_lowpass);
        Eigen::MatrixXd weight_input;
        if (use_convolve) {
            weight_input = mb.weight[map_index];
        }

        // filter kernel
        if (!mb.kernel.empty()) {
            SPDLOG_INFO("filtering kernel");
            filtered_map = mb.kernel[map_index];
            if (use_convolve) {
                run_convolve();
            }
            else if (filter_type=="wiener_filter") {
                uniform_weight = true;
                run_filter(mb, map_index);
            }

            // divide by filtered weight
            for (Eigen::Index i=0; i<n_rows; ++i) {
                for (Eigen::Index j=0; j<n_cols; ++j) {
                    if (denom(i,j) != 0.0) {
                        mb.kernel[map_index](i,j)=nume(i,j)/denom(i,j);
                    }
                    else {
                        mb.kernel[map_index](i,j)= 0.0;
                    }
                }
            }

            SPDLOG_INFO("kernel filtering done");
        }

        SPDLOG_INFO("filtering signal");
        filtered_map = mb.signal[map_index];
        if (use_convolve) {
            run_convolve();
        }
        else if (filter_type=="wiener_filter") {
            uniform_weight = false;
            run_filter(mb, map_index);
        }

        // divide by filtered weight
        for (Eigen::Index i=0; i<n_rows; ++i) {
            for (Eigen::Index j=0; j<n_cols; ++j) {
                if (denom(i,j) != 0.0) {
                    mb.signal[map_index](i,j) = nume(i,j)/denom(i,j);
                }
                else {
                    mb.signal[map_index](i,j)= 0.0;
                }
            }
        }

        if (filter_type=="wiener_filter" && !use_convolve) {
            mb.weight[map_index] = denom;
        }
        else if (use_convolve) {
            // propagate inverse-variance through smoothing: Var_smooth = (k^2) ⊗ Var
            Eigen::MatrixXd kernel = filter_template;
            double kernel_sum = kernel.sum();
            if (kernel_sum == 0.0 || !std::isfinite(kernel_sum)) {
                SPDLOG_WARN("convolve kernel sum is zero/invalid; skipping weight propagation");
            }
            else {
                kernel /= kernel_sum;
                Eigen::MatrixXd kernel_sq = kernel.array().square().matrix();
                double kernel_sq_sum = kernel_sq.sum();
                if (kernel_sq_sum == 0.0 || !std::isfinite(kernel_sq_sum)) {
                    SPDLOG_WARN("convolve kernel^2 sum is zero/invalid; skipping weight propagation");
                }
                else {
                    Eigen::MatrixXd var_map(n_rows, n_cols);
                    Eigen::MatrixXd mask_map(n_rows, n_cols);

                    double weight_threshold = 0.0;
                    if (mb.cov_cut > 0.0) {
                        weight_threshold = engine_utils::find_weight_threshold(weight_input, mb.cov_cut);
                    }
                    if (!std::isfinite(weight_threshold) || weight_threshold < 0.0) {
                        weight_threshold = 0.0;
                    }

                    for (Eigen::Index i=0; i<n_rows; ++i) {
                        for (Eigen::Index j=0; j<n_cols; ++j) {
                            double w = weight_input(i,j);
                            if (w > 0.0 && std::isfinite(w) && w >= weight_threshold) {
                                var_map(i,j) = 1.0 / w;
                                mask_map(i,j) = 1.0;
                            }
                            else {
                                var_map(i,j) = 0.0;
                                mask_map(i,j) = 0.0;
                            }
                        }
                    }

                    Eigen::MatrixXd template_backup = filter_template;
                    filter_template = kernel_sq;
                    invalidate_template_fft_cache();

                    filtered_map = var_map;
                    run_convolve(false);
                    Eigen::MatrixXd var_smooth = nume;

                    filtered_map = mask_map;
                    run_convolve(false);
                    Eigen::MatrixXd mask_smooth = nume;

                    constexpr double mask_floor = 1e-6;
                    for (Eigen::Index i=0; i<n_rows; ++i) {
                        for (Eigen::Index j=0; j<n_cols; ++j) {
                            double m = mask_smooth(i,j);
                            if (m > mask_floor && std::isfinite(m)) {
                                double v = var_smooth(i,j) / m;
                                if (v > 0.0 && std::isfinite(v)) {
                                    mb.weight[map_index](i,j) = 1.0 / v;
                                }
                                else {
                                    mb.weight[map_index](i,j) = 0.0;
                                }
                            }
                            else {
                                mb.weight[map_index](i,j) = 0.0;
                            }
                        }
                    }
                    filter_template = template_backup;
                    invalidate_template_fft_cache();
                }
            }
        }

        SPDLOG_INFO("signal/weight map filtering done");
    }

    template<class MB>
    void filter_noise(MB &mb, const int map_index, const int noise_num) {
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(
            mb.noise[map_index].data() + noise_num * mb.n_rows * mb.n_cols, mb.n_rows, mb.n_cols);
        filtered_map = noise_matrix;

        const bool use_convolve = (filter_type=="convolve") || (filter_type=="wiener_filter" && run_lowpass);
        if (use_convolve) {
            run_convolve();
        }
        else if (filter_type=="wiener_filter") {
            calc_numerator();
        }
        else {
            nume.setZero(n_rows, n_cols);
        }
        Eigen::MatrixXd ratio = use_convolve ? nume : divide_by_denom(nume, denom);
        noise_matrix.noalias() = ratio;
    }

    template<class MB>
    void filter_noise_threadsafe(MB &mb, const int map_index, const int noise_num) {
        const bool use_convolve = (filter_type=="convolve") || (filter_type=="wiener_filter" && run_lowpass);
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(
            mb.noise[map_index].data() + noise_num * mb.n_rows * mb.n_cols, mb.n_rows, mb.n_cols);

        Eigen::MatrixXd local_input = noise_matrix;
        Eigen::MatrixXd local_nume = Eigen::MatrixXd::Zero(n_rows, n_cols);
        if (use_convolve) {
            local_nume = run_convolve_on_input(local_input, true);
        }
        else if (filter_type=="wiener_filter") {
            local_nume = calc_numerator_from_input(local_input);
        }
        Eigen::MatrixXd ratio = use_convolve ? local_nume : divide_by_denom(local_nume, denom);
        noise_matrix.noalias() = ratio;
    }
};

inline WienerFilter::FFTWContext &WienerFilter::get_thread_fft_context(int rows, int cols) {
    static thread_local FFTWContext ctx;
    if (ctx.n_rows != rows || ctx.n_cols != cols || ctx.a == nullptr || ctx.b == nullptr ||
        ctx.pf == nullptr || ctx.pr == nullptr) {
        #pragma omp critical (wfFFTWPlanCache)
        {
            if (ctx.n_rows != rows || ctx.n_cols != cols || ctx.a == nullptr || ctx.b == nullptr ||
                ctx.pf == nullptr || ctx.pr == nullptr) {
                ctx.reset();
                ctx.a = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows * cols);
                ctx.b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rows * cols);
                ctx.pf = fftw_plan_dft_2d(rows, cols, ctx.a, ctx.b, FFTW_FORWARD, FFTW_ESTIMATE);
                ctx.pr = fftw_plan_dft_2d(rows, cols, ctx.a, ctx.b, FFTW_BACKWARD, FFTW_ESTIMATE);
                if (ctx.a == nullptr || ctx.b == nullptr || ctx.pf == nullptr || ctx.pr == nullptr) {
                    SPDLOG_ERROR("failed to create FFTW thread context for rows={} cols={}", rows, cols);
                    std::exit(EXIT_FAILURE);
                }
                ctx.n_rows = rows;
                ctx.n_cols = cols;
            }
        }
    }
    return ctx;
}

inline const Eigen::MatrixXcd &WienerFilter::get_filter_template_fft() {
    if (!filter_template_fft_valid) {
        auto &ctx = get_thread_fft_context(n_rows, n_cols);
        Eigen::MatrixXcd in(n_rows, n_cols);
        Eigen::MatrixXcd out(n_rows, n_cols);
        in.real() = filter_template;
        in.imag().setZero();
        out = engine_utils::fft2<engine_utils::forward>(in, ctx.pf, ctx.a, ctx.b);
        filter_template_fft = std::move(out);
        filter_template_fft_valid = true;
    }
    return filter_template_fft;
}

inline const Eigen::MatrixXcd &WienerFilter::get_filter_template_fft_scaled(bool normalize) {
    const double scale = static_cast<double>(n_rows) * static_cast<double>(n_cols);
    if (normalize) {
        if (!filter_template_fft_normalized_scaled_valid) {
            auto &ctx = get_thread_fft_context(n_rows, n_cols);
            Eigen::MatrixXcd in(n_rows, n_cols);
            Eigen::MatrixXcd out(n_rows, n_cols);
            Eigen::MatrixXd kernel = filter_template;
            const double kernel_sum = kernel.sum();
            if (kernel_sum != 0.0 && std::isfinite(kernel_sum)) {
                kernel /= kernel_sum;
            }
            in.real() = kernel;
            in.imag().setZero();
            out = engine_utils::fft2<engine_utils::forward>(in, ctx.pf, ctx.a, ctx.b);
            out *= scale;
            filter_template_fft_normalized_scaled = std::move(out);
            filter_template_fft_normalized_scaled_valid = true;
        }
        return filter_template_fft_normalized_scaled;
    }
    if (!filter_template_fft_scaled_valid) {
        filter_template_fft_scaled = get_filter_template_fft();
        filter_template_fft_scaled *= scale;
        filter_template_fft_scaled_valid = true;
    }
    return filter_template_fft_scaled;
}

inline Eigen::MatrixXd WienerFilter::calc_numerator_from_input(const Eigen::MatrixXd &input_map) {
    auto &ctx = get_thread_fft_context(n_rows, n_cols);

    struct ComplexScratch {
        Eigen::MatrixXcd in;
        Eigen::MatrixXcd out;
        Eigen::MatrixXcd qqq;
    };
    static thread_local ComplexScratch scratch;
    if (scratch.in.rows() != n_rows || scratch.in.cols() != n_cols) {
        scratch.in.resize(n_rows, n_cols);
        scratch.out.resize(n_rows, n_cols);
        scratch.qqq.resize(n_rows, n_cols);
    }
    auto &in = scratch.in;
    auto &out = scratch.out;
    auto &qqq = scratch.qqq;

    in.real() = rr.array() * input_map.array();
    in.imag().setZero();
    out = engine_utils::fft2<engine_utils::forward>(in, ctx.pf, ctx.a, ctx.b);

    in.real() = out.real().array() / vvq.array();
    in.imag() = out.imag().array() / vvq.array();
    out = engine_utils::fft2<engine_utils::inverse>(in, ctx.pr, ctx.a, ctx.b);

    in.real() = out.real().array() * rr.array();
    in.imag().setZero();
    out = engine_utils::fft2<engine_utils::forward>(in, ctx.pf, ctx.a, ctx.b);
    qqq = out;

    const auto &template_fft = get_filter_template_fft();
    in.real() = template_fft.real().array() * qqq.real().array() + template_fft.imag().array() * qqq.imag().array();
    in.imag() = -template_fft.imag().array() * qqq.real().array() + template_fft.real().array() * qqq.imag().array();
    out = engine_utils::fft2<engine_utils::inverse>(in, ctx.pr, ctx.a, ctx.b);

    return out.real();
}

inline Eigen::MatrixXd WienerFilter::run_convolve_on_input(const Eigen::MatrixXd &input_map, bool normalize) {
    auto &ctx = get_thread_fft_context(n_rows, n_cols);

    struct ConvolveScratch {
        Eigen::MatrixXcd in;
        Eigen::MatrixXcd out;
    };
    static thread_local ConvolveScratch scratch;
    if (scratch.in.rows() != n_rows || scratch.in.cols() != n_cols) {
        scratch.in.resize(n_rows, n_cols);
        scratch.out.resize(n_rows, n_cols);
    }
    auto &in = scratch.in;
    auto &out = scratch.out;
    const auto &fft_filter = get_filter_template_fft_scaled(normalize);

    in.real() = input_map;
    in.imag().setZero();
    out = engine_utils::fft2<engine_utils::forward>(in, ctx.pf, ctx.a, ctx.b);
    out = out * n_rows * n_cols;

    in.real() = out.real().array() * fft_filter.real().array() - out.imag().array() * fft_filter.imag().array();
    in.imag() = out.imag().array() * fft_filter.real().array() + out.real().array() * fft_filter.imag().array();
    out = engine_utils::fft2<engine_utils::inverse>(in, ctx.pr, ctx.a, ctx.b);
    out = out / n_rows / n_cols;

    return out.real();
}

inline Eigen::MatrixXd WienerFilter::divide_by_denom(const Eigen::MatrixXd &numerator,
                                                     const Eigen::MatrixXd &denominator) const {
    Eigen::MatrixXd ratio = Eigen::MatrixXd::Zero(numerator.rows(), numerator.cols());
    ratio.array() = (denominator.array() != 0.0).select(numerator.array() / denominator.array(), 0.0);
    return ratio;
}

// get config file
template <typename config_t>
void WienerFilter::get_config(config_t &config, std::vector<std::vector<std::string>> &missing_keys,
                              std::vector<std::vector<std::string>> &invalid_keys) {

    // for array names
    engine_utils::toltecIO toltec_io;

    // get filter type
    get_config_value(config, filter_type, missing_keys, invalid_keys,
                     std::tuple{"post_processing","map_filtering","type"},{"wiener_filter","convolve","destripe"});
    // get template type
    get_config_value(config, template_type, missing_keys, invalid_keys,
                     std::tuple{"wiener_filter","template_type"},{"kernel","gaussian","airy","highpass"});
    // run lowpass only?
    get_config_value(config, run_lowpass, missing_keys, invalid_keys,
                     std::tuple{"wiener_filter","lowpass_only"});
    // re-normalize weight maps?
    get_config_value(config, normalize_error, missing_keys, invalid_keys,
                     std::tuple{"post_processing","map_filtering","normalize_errors"});
    // denominator convergence thresholds
    get_config_value(config, denom_rel_tol, missing_keys, invalid_keys,
                     std::tuple{"wiener_filter","denom_rel_tol"}, {}, {0.0}, {1.0});
    get_config_value(config, tail_frac_tol, missing_keys, invalid_keys,
                     std::tuple{"wiener_filter","tail_frac_tol"}, {}, {0.0}, {1.0});
    get_config_value(config, max_loops, missing_keys, invalid_keys,
                     std::tuple{"wiener_filter","max_loops"}, {}, {1});

    // gaussian or airy template fwhms
    if (template_type=="gaussian" || template_type=="airy") {
        for (auto const& [arr_index, arr_name] : toltec_io.array_name_map) {
            get_config_value(config, template_fwhm_rad[arr_name], missing_keys, invalid_keys,
                             std::tuple{"wiener_filter","template_fwhm_arcsec",arr_name});
        }
        for (auto const& pair : template_fwhm_rad) {
            template_fwhm_rad[pair.first] = template_fwhm_rad[pair.first]*ASEC_TO_RAD;
        }
    }
}

template<class MB>
void WienerFilter::make_gaussian_template(MB &mb, const double gaussian_template_fwhm_rad) {
    // distance from tangent point
    Eigen::MatrixXd dist(n_rows,n_cols);

    // calculate distance
    for (Eigen::Index i=0; i<n_cols; i++) {
        for (Eigen::Index j=0; j<n_rows; j++) {
            dist(j,i) = sqrt(pow(mb.rows_tan_vec(j),2) +
                             pow(mb.cols_tan_vec(i),2));
        }
    }

    Eigen::Index row_index, col_index;

    // minimum distance
    double min_dist = dist.minCoeff(&row_index,&col_index);
    // standard deviation
    double sigma = gaussian_template_fwhm_rad*FWHM_TO_STD;

    // shift indices
    std::vector<Eigen::Index> shift_indices = {-row_index, -col_index};

    // calculate template
    filter_template = exp(-0.5 * pow(dist.array() / sigma, 2.));
    // shift template
    filter_template = engine_utils::shift_2D(filter_template, shift_indices);
}

template<class MB>
void WienerFilter::make_airy_template(MB &mb, const double gaussian_template_fwhm_rad) {
    // distance from tangent point
    Eigen::MatrixXd dist(n_rows,n_cols);

    // calculate distance
    for (Eigen::Index i=0; i<n_cols; i++) {
        for (Eigen::Index j=0; j<n_rows; j++) {
            dist(j,i) = sqrt(pow(mb.rows_tan_vec(j),2) +
                             pow(mb.cols_tan_vec(i),2));
        }
    }

    // to hold minimum distance
    Eigen::Index row_index, col_index;

    // minimum distance
    double min_dist = dist.minCoeff(&row_index,&col_index);

    // shift indices
    std::vector<Eigen::Index> shift_indices = {-row_index, -col_index};

    // calculate template
    double factor = pi*(1.028/gaussian_template_fwhm_rad);

    // resize template
    filter_template.resize(n_rows, n_cols);

    // populate template
    for (Eigen::Index i=0; i<n_cols; i++) {
        for (Eigen::Index j=0; j<n_rows; j++) {
            if (dist(j,i)!=0) {
            filter_template(j,i) = pow(2*boost::math::cyl_bessel_j(1,factor*dist(j,i))/(factor*dist(j,i)),2);
            }
            else {
                filter_template(j,i) = 1;
            }
        }
    }

    // shift template
    filter_template = engine_utils::shift_2D(filter_template, shift_indices);
}


template<class MB, class CD>
void WienerFilter::make_kernel_template(MB &mb, const int map_index, CD &calib_data) {
    logger->info("building kernel template internals for map_index={}", map_index);
    if (map_index < 0 || map_index >= static_cast<int>(mb.kernel.size()) ||
        map_index >= static_cast<int>(mb.weight.size())) {
        logger->error("invalid map index for kernel template: map_index={} kernel_size={} weight_size={}",
                      map_index,
                      static_cast<long long>(mb.kernel.size()),
                      static_cast<long long>(mb.weight.size()));
        std::exit(EXIT_FAILURE);
    }
    if (mb.kernel[map_index].rows() != n_rows || mb.kernel[map_index].cols() != n_cols ||
        mb.weight[map_index].rows() != n_rows || mb.weight[map_index].cols() != n_cols) {
        logger->error("kernel/weight dimensions do not match map geometry for map_index={}: kernel=({}, {}) weight=({}, {}) expected=({}, {})",
                      map_index,
                      static_cast<long long>(mb.kernel[map_index].rows()), static_cast<long long>(mb.kernel[map_index].cols()),
                      static_cast<long long>(mb.weight[map_index].rows()), static_cast<long long>(mb.weight[map_index].cols()),
                      n_rows, n_cols);
        std::exit(EXIT_FAILURE);
    }

    // collect what we need
    Eigen::MatrixXd temp_kernel = mb.kernel[map_index];

    // Center kernel deterministically using the peak absolute response.
    // This avoids unstable Gaussian fitting failures in some coadd/kernel combinations.
    Eigen::Index peak_row = 0;
    Eigen::Index peak_col = 0;
    const double peak_abs = temp_kernel.cwiseAbs().maxCoeff(&peak_row, &peak_col);
    if (!std::isfinite(peak_abs)) {
        logger->error("kernel template peak is non-finite for map_index={}", map_index);
        std::exit(EXIT_FAILURE);
    }
    const Eigen::Index center_row = n_rows / 2;
    const Eigen::Index center_col = n_cols / 2;
    Eigen::Index shift_row = center_row - peak_row;
    Eigen::Index shift_col = center_col - peak_col;
    logger->info("kernel template centering via peak: map_index={} peak_row={} peak_col={} shift_row={} shift_col={}",
                 map_index, peak_row, peak_col, shift_row, shift_col);

    std::vector<Eigen::Index> shift_indices = {shift_row,shift_col};
    temp_kernel = engine_utils::shift_2D(temp_kernel, shift_indices);

    // calculate distance
    Eigen::MatrixXd dist(n_rows,n_cols);
    for (Eigen::Index i=0; i<n_cols; i++) {
        for (Eigen::Index j=0; j<n_rows; j++) {
            dist(j,i) = sqrt(pow(mb.rows_tan_vec(j),2) +
                             pow(mb.cols_tan_vec(i),2));
        }
    }

    // pixel closet to tangent point
    Eigen::Index row_index, col_index;
    auto min_dist = dist.minCoeff(&row_index,&col_index);

    // create new bins based on diff_rows
    int n_bins = static_cast<int>(std::floor(std::abs(mb.rows_tan_vec(n_rows-1) / diff_rows))) + 1;
    n_bins = std::max(n_bins, 2);
    Eigen::VectorXd bin_low = Eigen::VectorXd::LinSpaced(n_bins,0,n_bins-1)*diff_rows;

    Eigen::VectorXd kernel_interp(n_bins-1);
    kernel_interp.setZero();
    Eigen::VectorXd dist_interp(n_bins-1);
    dist_interp.setZero();

    std::vector<double> kernel_valid;
    std::vector<double> dist_valid;
    kernel_valid.reserve(n_bins - 1);
    dist_valid.reserve(n_bins - 1);

    // radial averages
    for (Eigen::Index i=0; i<n_bins-1; i++) {
        int c = 0;
        for (Eigen::Index j=0; j<n_cols; j++) {
            for (Eigen::Index k=0; k<n_rows; k++) {
                if (dist(k,j) >= bin_low(i) && dist(k,j) < bin_low(i+1)){
                    c++;
                    kernel_interp(i) += temp_kernel(k,j);
                    dist_interp(i) += dist(k,j);
                }
            }
        }
        if (c > 0) {
            kernel_interp(i) /= c;
            dist_interp(i) /= c;
            kernel_valid.push_back(kernel_interp(i));
            dist_valid.push_back(dist_interp(i));
        }
    }

    if (dist_valid.size() < 2) {
        SPDLOG_WARN("kernel template radial averages are undersampled; using shifted kernel map directly");
        filter_template = temp_kernel;
        return;
    }

    // now spline interpolate to generate new template array
    filter_template.resize(n_rows,n_cols);

    // create spline function
    Eigen::VectorXd kernel_interp_valid = Eigen::Map<Eigen::VectorXd>(kernel_valid.data(), kernel_valid.size());
    Eigen::VectorXd dist_interp_valid = Eigen::Map<Eigen::VectorXd>(dist_valid.data(), dist_valid.size());

    engine_utils::SplineFunction s(dist_interp_valid, kernel_interp_valid);

    // carry out the interpolation
    for (Eigen::Index i=0; i<n_cols; i++) {
        for (Eigen::Index j=0; j<n_rows; j++) {
            Eigen::Index tj = (j-row_index)%n_rows;
            Eigen::Index ti = (i-col_index)%n_cols;
            Eigen::Index shiftj = (tj < 0) ? n_rows+tj : tj;
            Eigen::Index shifti = (ti < 0) ? n_cols+ti : ti;

            // if within limits
            if (dist(j,i) <= s.x_max && dist(j,i) >= s.x_min) {
                filter_template(shiftj,shifti) = s(dist(j,i));
            }
            // if above x limit
            else if (dist(j,i) > s.x_max) {
                filter_template(shiftj,shifti) = kernel_interp_valid(kernel_interp_valid.size()-1);
            }
            // if below x limit
            else if (dist(j,i) < s.x_min) {
                filter_template(shiftj,shifti) = kernel_interp_valid(0);
            }
        }
    }
}

template <class MB>
void WienerFilter::calc_vvq(MB &mb, const int map_index) {
    // resize psd_q
    Eigen::MatrixXd psd_q(n_rows,n_cols);

    // set constant if lowpass only
    if (run_lowpass) {
        psd_q.setOnes();
    }
    else if (mb.noise_psds.empty() || mb.noise_psd_freqs.empty() ||
             map_index >= static_cast<int>(mb.noise_psds.size()) ||
             map_index >= static_cast<int>(mb.noise_psd_freqs.size())) {
        SPDLOG_WARN("noise PSDs missing for map {}; falling back to lowpass-only response", map_index);
        psd_q.setOnes();
    }
    else {
        // psd and psd freq vectors
        Eigen::VectorXd psd = mb.noise_psds[map_index];
        Eigen::VectorXd psd_freq = mb.noise_psd_freqs[map_index];

        // size of psd and psd freq vectors
        Eigen::Index n_psd = psd.size();

        // modify the psd array to take out lowpassing and highpassing
        Eigen::Index max_psd_index;
        double max_psd = psd.maxCoeff(&max_psd_index);
        double psd_freq_break = 0.;
        double psd_break = 0.;

        if (!std::isfinite(max_psd) || max_psd <= 0.0) {
            SPDLOG_WARN("noise PSD invalid for map {}; falling back to lowpass-only response", map_index);
            psd_q.setOnes();
        }
        else {

        for (Eigen::Index i=0; i<n_psd; i++) {
            if (psd(i)/max_psd < psd_lim){
                psd_freq_break = psd_freq(i);
                break;
            }
        }

        // flatten the response above the lowpass break
        int count = (psd_freq.array() <= 0.8*psd_freq_break).count();

        if (count > 0) {
            for (Eigen::Index i=0; i<n_psd; i++) {
                if (psd_freq_break > 0) {
                    if (psd_freq(i) <= 0.8*psd_freq_break) {
                        psd_break = psd(i);
                    }

                    if (psd_freq(i) > 0.8*psd_freq_break) {
                        psd(i) = psd_break;
                    }
                }
            }
        }

        // flatten highpass response if present
        if (max_psd_index > 0) {
            psd.head(max_psd_index).setConstant(max_psd);
        }

        // get spacing
        double diff_qr = 1. / (n_rows * diff_rows);
        double diff_qc = 1. / (n_cols * diff_cols);

        Eigen::MatrixXd qmap(n_rows,n_cols);

        // q_row
        Eigen::VectorXd q_row = Eigen::VectorXd::LinSpaced(n_rows, -n_rows / 2, n_rows / 2) * diff_qr;
        // q_col
        Eigen::VectorXd q_col = Eigen::VectorXd::LinSpaced(n_cols, -n_cols / 2, n_cols / 2) * diff_qc;

        // shift q_row
        std::vector<Eigen::Index> shift_1 = {-n_rows/2};
        q_row = engine_utils::shift_1D(q_row, shift_1);
        // shift q_col
        std::vector<Eigen::Index> shift_2 = {n_cols/2};
        q_col = engine_utils::shift_1D(q_col, shift_2);

        for (Eigen::Index i=0; i<n_rows; ++i) {
            for (Eigen::Index j=0; j<n_cols; ++j) {
                qmap(i,j) = sqrt(pow(q_row(i),2)+pow(q_col(j),2));
            }
        }

        psd_q.setZero();

        Eigen::Matrix<Eigen::Index, 1, 1> n_psd_matrix;
        n_psd_matrix << n_psd;

        // interpolate onto psd_q
        Eigen::Index interp_pts = 1;
        for (Eigen::Index i=0; i<n_cols; i++) {
            for (Eigen::Index j=0; j<n_rows; j++) {
                if ((qmap(j,i) <= psd_freq(psd_freq.size() - 1)) && (qmap(j,i) >= psd_freq(0))) {
                    mlinterp::interp<mlinterp::rnatord>(n_psd_matrix.data(), interp_pts,
                                     psd.data(), psd_q.data() + n_rows * i + j,
                                     psd_freq.data(), qmap.data() + n_rows * i + j);
                }
                else if (qmap(j,i) > psd_freq(n_psd - 1)) {
                    psd_q(j,i) = psd(n_psd- 1);
                }
                else if (qmap(j,i) < psd_freq(0)) {
                    psd_q(j,i) = psd(0);
                }
            }
        }

        // find the minimum value of psd
        auto psd_min = psd.minCoeff();
        if (!std::isfinite(psd_min) || psd_min <= 0.0) {
            psd_min = std::max(psd_lim * max_psd, 1e-12);
        }

        for (Eigen::Index i=0; i<n_rows; ++i) {
            for (Eigen::Index j=0; j<n_cols; ++j) {
                if (psd_q(i,j) < psd_min) {
                    psd_q(i,j) = psd_min;
                }
            }
        }
        }
    }

    // normalize the power spectrum psd_q and place into vvq
    double psd_sum = psd_q.sum();
    if (!std::isfinite(psd_sum) || psd_sum <= 0.0) {
        vvq.setOnes(n_rows, n_cols);
        vvq /= static_cast<double>(n_rows * n_cols);
    }
    else {
        vvq = psd_q/psd_sum;
    }
}

void WienerFilter::calc_numerator() {
    nume = calc_numerator_from_input(filtered_map);
}

void WienerFilter::run_convolve(bool normalize) {
    nume = run_convolve_on_input(filtered_map, normalize);
    denom.setOnes(n_rows,n_cols);
}

void WienerFilter::destripe(double threshold_factor) {

    // allocate FFTW plans and buffers
    fftw_complex *in, *out;
    fftw_plan p_forward, p_backward;

    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n_rows * n_cols);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n_rows * n_cols);

    // create plans
    p_forward = fftw_plan_dft_2d(n_rows, n_cols, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    p_backward = fftw_plan_dft_2d(n_rows, n_cols, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);

    // copy the image to the in buffer and set imaginary parts to zero
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            in[i * n_cols + j][0] = filtered_map(i, j);
            in[i * n_cols + j][1] = 0.0;
        }
    }

    // perform the forward FFT
    fftw_execute(p_forward);

    // compute the magnitude and find the maximum
    double max_magnitude = 0.0;
    for (int i = 0; i < n_rows * n_cols; ++i) {
        double magnitude = std::sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]);
        if (magnitude > max_magnitude) {
            max_magnitude = magnitude;
        }
    }

    // threshold and zero out coefficients below threshold
    double threshold = threshold_factor * max_magnitude;
    int n_pixels = 0;
    for (int i = 0; i < n_rows * n_cols; ++i) {
        double magnitude = std::sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]);
        if (magnitude < threshold) {
            out[i][0] = 0.0;
            out[i][1] = 0.0;
            n_pixels++;
        }
    }

    SPDLOG_INFO("number of pixels below threshold {}", n_pixels);

    // perform the inverse FFT
    fftw_execute(p_backward);

    // copy the normalized real part back to the Eigen matrix
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            filtered_map(i, j) = in[i * n_cols + j][0] / (n_rows * n_cols);
        }
    }

    // cleanup
    fftw_destroy_plan(p_forward);
    fftw_destroy_plan(p_backward);
    fftw_free(in);
    fftw_free(out);
}

void WienerFilter::calc_denominator() {
    // set up fftw
    fftw_complex *a;
    fftw_complex *b;
    fftw_plan pf, pr;

    a = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_rows*n_cols);
    b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_rows*n_cols);

    pf = fftw_plan_dft_2d(n_rows, n_cols, a, b, FFTW_FORWARD, FFTW_ESTIMATE);
    pr = fftw_plan_dft_2d(n_rows, n_cols, a, b, FFTW_BACKWARD, FFTW_ESTIMATE);

    // resize denominator
    denom.setZero(n_rows,n_cols);

    // inputs and outputs to ffts
    Eigen::MatrixXcd in(n_rows,n_cols);
    Eigen::MatrixXcd out(n_rows,n_cols);

    // using uniform weights only
    if (uniform_weight) {
        const auto &template_fft = get_filter_template_fft();
        out = template_fft;

        // set denominator
        denom.setConstant(((out.real().array() * out.real().array() + out.imag().array() * out.imag().array()) / vvq.array()).sum());

        // destroy fftw plans
        fftw_destroy_plan(pf);
        fftw_destroy_plan(pr);

        fftw_free(a);
        fftw_free(b);
    }

    else {
        // initialize denominator
        denom.setZero();

        in.real() = pow(vvq.array(), -1);
        in.imag().setZero();

        //out = engine_utils::fft<engine_utils::inverse>(in, parallel_policy);
        out = engine_utils::fft2<engine_utils::inverse>(in, pr, a, b);

        // destroy fftw plans
        fftw_free(a);
        fftw_free(b);

        fftw_destroy_plan(pf);
        fftw_destroy_plan(pr);

        Eigen::VectorXd zz2d(n_rows * n_cols);

        // vector of real components of IFFT(1/VVQ)
        for (Eigen::Index i=0; i<n_cols; i++) {
            for (Eigen::Index j=0; j<n_rows;j++) {
                int ii = n_rows*i+j;
                zz2d(ii) = (out.real()(j,i));
            }
        }

        // sort absolute values in ascending order
        Eigen::VectorXd Z_abs = zz2d.array().abs();
        auto sorted = engine_utils::sorter(Z_abs);

        // number of iterations for convergence
        n_loops = n_rows * n_cols / 100;
        if (n_loops < 100) {
            n_loops = 100;
        }
        const double Z_abs_total = Z_abs.sum();
        double Z_abs_done = 0.0;

        // flag for convergence
        bool done = false;

        tula::logging::progressbar pb(
            [&](const auto &msg) { logger->info("{}", msg); }, 90,
            "calculating denom");
        const Eigen::Index total_iters = n_rows * n_cols;
        const Eigen::Index pb_stride = std::max<Eigen::Index>(total_iters / 100, 1);
        const auto denom_start = std::chrono::steady_clock::now();
        double last_log_s = 0.0;
        const double denom_rel_tol_local = denom_rel_tol;
        const double tail_frac_tol_local = tail_frac_tol;
        const double inv_npix = 1.0 / static_cast<double>(n_rows * n_cols);
        const Eigen::Index chunk_size = std::max<Eigen::Index>(n_loops, 100);
        const int max_checks = std::max(max_loops, 1);
        int checks_done = 0;
        const int max_threads = omp_get_max_threads();
        std::vector<Eigen::MatrixXd> partial_deltas;
        partial_deltas.reserve(max_threads);
        for (int thread = 0; thread < max_threads; ++thread) {
            partial_deltas.emplace_back(Eigen::MatrixXd::Zero(n_rows, n_cols));
        }
        std::vector<double> partial_z_abs_done(max_threads, 0.0);
        for (Eigen::Index chunk_start = 0; chunk_start < total_iters && !done; chunk_start += chunk_size) {
            const Eigen::Index chunk_end = std::min(chunk_start + chunk_size, total_iters);
            Eigen::MatrixXd chunk_delta = Eigen::MatrixXd::Zero(n_rows, n_cols);
            double chunk_z_abs_done = 0.0;
            for (int thread = 0; thread < max_threads; ++thread) {
                partial_deltas[thread].setZero();
                partial_z_abs_done[thread] = 0.0;
            }

            #pragma omp parallel shared(sorted, zz2d, Z_abs, partial_deltas, partial_z_abs_done, chunk_start, chunk_end, total_iters, n_rows, n_cols, filter_template, rr, inv_npix) default (none)
            {
                const int thread_index = omp_get_thread_num();
                auto &ctx = get_thread_fft_context(n_rows, n_cols);
                Eigen::MatrixXcd in_local(n_rows, n_cols);
                Eigen::MatrixXcd out_local(n_rows, n_cols);
                Eigen::MatrixXcd ffdq(n_rows, n_cols);
                Eigen::MatrixXd in_prod(n_rows, n_cols);
                Eigen::MatrixXd shifted_template(n_rows, n_cols);
                Eigen::MatrixXd shifted_rr(n_rows, n_cols);
                auto &local_delta = partial_deltas[thread_index];
                double &local_z_abs_done = partial_z_abs_done[thread_index];

                auto shift_into = [&](const Eigen::MatrixXd &src, Eigen::Index shift_row, Eigen::Index shift_col, Eigen::MatrixXd &dst) {
                    for (Eigen::Index col = 0; col < n_cols; ++col) {
                        const Eigen::Index shifted_col = (col + shift_col) >= 0 ?
                            (col + shift_col) % n_cols :
                            (n_cols + ((col + shift_col) % n_cols)) % n_cols;
                        for (Eigen::Index row = 0; row < n_rows; ++row) {
                            const Eigen::Index shifted_row = (row + shift_row) >= 0 ?
                                (row + shift_row) % n_rows :
                                (n_rows + ((row + shift_row) % n_rows)) % n_rows;
                            dst(shifted_row, shifted_col) = src(row, col);
                        }
                    }
                };

                #pragma omp for schedule(dynamic)
                for (Eigen::Index kk = chunk_start; kk < chunk_end; ++kk) {
                    auto shift_index = std::get<1>(sorted[total_iters - kk - 1]);
                    const Eigen::Index shift_row = -static_cast<Eigen::Index>(shift_index % n_rows);
                    const Eigen::Index shift_col = -static_cast<Eigen::Index>(shift_index / n_rows);

                    shift_into(filter_template, shift_row, shift_col, shifted_template);
                    in_prod = filter_template.array() * shifted_template.array();
                    in_local.real() = in_prod;
                    in_local.imag().setZero();
                    out_local = engine_utils::fft2<engine_utils::forward>(in_local, ctx.pf, ctx.a, ctx.b);
                    ffdq = out_local;

                    shift_into(rr, shift_row, shift_col, shifted_rr);
                    in_prod = rr.array() * shifted_rr.array();
                    in_local.real() = in_prod;
                    in_local.imag().setZero();
                    out_local = engine_utils::fft2<engine_utils::forward>(in_local, ctx.pf, ctx.a, ctx.b);

                    in_local.real() = ffdq.real().array() * out_local.real().array() + ffdq.imag().array() * out_local.imag().array();
                    in_local.imag() = -ffdq.imag().array() * out_local.real().array() + ffdq.real().array() * out_local.imag().array();
                    out_local = engine_utils::fft2<engine_utils::inverse>(in_local, ctx.pr, ctx.a, ctx.b);

                    const double scale = zz2d(shift_index) * inv_npix;
                    local_delta.array() += scale * out_local.real().array();
                    local_z_abs_done += Z_abs(shift_index);
                }
            }
            for (int thread = 0; thread < max_threads; ++thread) {
                chunk_delta += partial_deltas[thread];
                chunk_z_abs_done += partial_z_abs_done[thread];
            }

            denom += chunk_delta;
            Z_abs_done += chunk_z_abs_done;
            for (Eigen::Index kk = chunk_start; kk < chunk_end; ++kk) {
                pb.count(total_iters, pb_stride);
            }

            const double denom_norm = denom.norm();
            const double delta_norm = chunk_delta.norm();
            const double rel_update = delta_norm / std::max(denom_norm, 1e-12);
            const double tail_frac = (Z_abs_total > 0.0) ? ((Z_abs_total - Z_abs_done) / Z_abs_total) : 0.0;
            const double elapsed_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - denom_start).count();
            const double step_s = elapsed_s - last_log_s;
            last_log_s = elapsed_s;

            logger->info("{} iteration(s) complete. rel_update={} tail_frac={} elapsed_s={} step_s={}",
                         chunk_end, static_cast<float>(rel_update), static_cast<float>(tail_frac),
                         static_cast<float>(elapsed_s), static_cast<float>(step_s));

            ++checks_done;
            if (rel_update < denom_rel_tol_local && tail_frac < tail_frac_tol_local) {
                done = true;
            }
            else if (checks_done >= max_checks) {
                logger->info("reached Wiener denominator max_loops={} after {} iteration(s); stopping early",
                             max_checks, chunk_end);
                done = true;
            }
        }
        for (Eigen::Index i=0; i<n_rows; i++) {
            for (Eigen::Index j=0; j<n_cols; j++) {
                if (denom(i,j) < denom_limit) {
                    denom(i,j) = 0;
                }
            }
        }
    }
}

} // namespace mapmaking
