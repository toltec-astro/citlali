#pragma once

#include <citlali/core/mapmaking/convolve_variance_contract.h>

#include <string>
#include <complex>
#include <chrono>
#include <algorithm>
#include <exception>
#include <map>
#include <memory>
#include <vector>
#include <cmath>
#include <limits>

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
#include <citlali/core/mapmaking/edge_guard_state.h>
#include <citlali/core/pipeline/wiener_filter_validation.h>

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
    std::string kernel_template_tail_mode = "constant";
    // normalize filtered map errors
    bool normalize_error;
    // uniform weighting
    bool uniform_weight;
    // lowpass only
    bool run_lowpass;
    // optional pre-filter edge guard derived from weight/coverage maps
    bool edge_guard_enabled = true;
    std::string edge_weight_threshold_mode = "coverage_cut";
    std::string edge_hits_threshold_mode = "core_median_fraction";
    std::string edge_fill_mode = "core_median";
    std::string edge_taper_mode = "none";
    double edge_hits_core_fraction = 0.15;
    double edge_guard_radius_fwhm = 1.0;
    double edge_taper_min_fraction = 0.25;

    // number of loops in denom calc
    int n_loops;
    // maximum number of loops for denom calc
    int max_loops = 500;
    // optional actual denominator iteration cadence/cap
    int denom_check_iters = 0;
    int max_denom_iters = 0;
    // lower limit to zero out denom values
    double denom_limit = 1.e-4;
    // psd limit
    double psd_lim = 1.e-4;
    // denominator convergence tolerances
    double denom_rel_tol = 1e-4;
    double tail_frac_tol = 5e-2;
    Eigen::Index last_denom_iters = 0;
    double last_denom_rel_update = std::numeric_limits<double>::quiet_NaN();
    double last_denom_tail_frac = std::numeric_limits<double>::quiet_NaN();
    std::string last_denom_stop_reason = "not_run";

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

        citlali::pipeline::require_wiener_template_geometry(
            n_rows, n_cols, mb.rows_tan_vec.size(), mb.cols_tan_vec.size());

        // x and y spacing should be equal
        diff_rows = std::abs(mb.rows_tan_vec(1) - mb.rows_tan_vec(0));
        diff_cols = std::abs(mb.cols_tan_vec(1) - mb.cols_tan_vec(0));
        citlali::pipeline::require_wiener_pixel_spacing(diff_rows, diff_cols);

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
            citlali::pipeline::require_wiener_kernel_weight_index(
                map_index, mb.kernel.size(), mb.weight.size());
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
        mapmaking::ensure_edge_guard_storage(mb);

        const auto m_idx = static_cast<std::size_t>(map_index);
        mapmaking::reset_edge_guard_map(mb, m_idx);

        Eigen::MatrixXd guarded_weight = mb.weight[map_index];
        if (edge_guard_enabled) {
            const Eigen::MatrixXd original_signal = mb.signal[map_index];
            double weight_threshold = 0.0;
            if (edge_weight_threshold_mode == "coverage_cut" && mb.cov_cut > 0.0) {
                weight_threshold = engine_utils::find_weight_threshold(mb.weight[map_index], mb.cov_cut);
            }
            if (!std::isfinite(weight_threshold) || weight_threshold < 0.0) {
                weight_threshold = 0.0;
            }
            mb.edge_guard_weight_threshold[m_idx] = weight_threshold;

            Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> science_mask =
                ((mb.weight[map_index].array() >= weight_threshold) &&
                 (mb.weight[map_index].array() > 0.0));
            const auto provisional_count = static_cast<int>(science_mask.count());

            double hits_threshold = mapmaking::edge_guard_fill_double();
            if (!mb.coverage.empty() &&
                map_index < static_cast<int>(mb.coverage.size()) &&
                edge_hits_threshold_mode == "core_median_fraction" &&
                provisional_count > 0) {
                std::vector<double> coverage_values;
                coverage_values.reserve(static_cast<std::size_t>(provisional_count));
                for (Eigen::Index r = 0; r < mb.coverage[map_index].rows(); ++r) {
                    for (Eigen::Index c = 0; c < mb.coverage[map_index].cols(); ++c) {
                        const double value = mb.coverage[map_index](r, c);
                        if (science_mask(r, c) && std::isfinite(value) && value > 0.0) {
                            coverage_values.push_back(value);
                        }
                    }
                }
                if (!coverage_values.empty()) {
                    Eigen::Map<Eigen::VectorXd> cov_vec(coverage_values.data(),
                                                        static_cast<Eigen::Index>(coverage_values.size()));
                    const double core_median = tula::alg::median(cov_vec);
                    if (std::isfinite(core_median) && core_median > 0.0) {
                        hits_threshold = edge_hits_core_fraction * core_median;
                        science_mask = science_mask &&
                            ((mb.coverage[map_index].array() >= hits_threshold) &&
                             (mb.coverage[map_index].array() > 0.0));
                    }
                }
            }
            mb.edge_guard_hits_threshold[m_idx] = hits_threshold;

            if (science_mask.count() == 0) {
                science_mask = ((mb.weight[map_index].array() >= weight_threshold) &&
                                (mb.weight[map_index].array() > 0.0));
            }

            const int science_npix = static_cast<int>(science_mask.count());
            mb.edge_guard_science_npix[m_idx] = science_npix;
            const double n_pix = static_cast<double>(mb.n_rows) * static_cast<double>(mb.n_cols);
            mb.edge_guard_science_frac[m_idx] = (n_pix > 0.0) ? static_cast<double>(science_npix) / n_pix : 0.0;

            Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> support_mask = science_mask;
            const int support_radius_pix = std::max(
                1, static_cast<int>(std::ceil(edge_guard_radius_fwhm * std::max(init_fwhm, 1.0))));
            mb.edge_guard_support_radius_pix[m_idx] = support_radius_pix;

            if (science_npix > 0 && support_radius_pix > 0) {
                for (Eigen::Index r = 0; r < mb.n_rows; ++r) {
                    for (Eigen::Index c = 0; c < mb.n_cols; ++c) {
                        if (support_mask(r, c)) {
                            continue;
                        }
                        const Eigen::Index r0 = std::max<Eigen::Index>(0, r - support_radius_pix);
                        const Eigen::Index r1 = std::min<Eigen::Index>(mb.n_rows - 1, r + support_radius_pix);
                        const Eigen::Index c0 = std::max<Eigen::Index>(0, c - support_radius_pix);
                        const Eigen::Index c1 = std::min<Eigen::Index>(mb.n_cols - 1, c + support_radius_pix);
                        bool found = false;
                        for (Eigen::Index rr = r0; rr <= r1 && !found; ++rr) {
                            for (Eigen::Index cc = c0; cc <= c1; ++cc) {
                                if (!science_mask(rr, cc)) {
                                    continue;
                                }
                                const Eigen::Index dr = rr - r;
                                const Eigen::Index dc = cc - c;
                                if (dr * dr + dc * dc <= support_radius_pix * support_radius_pix) {
                                    support_mask(r, c) = true;
                                    found = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> guardband_mask =
                support_mask && (!science_mask);

            std::vector<double> science_values;
            science_values.reserve(static_cast<std::size_t>(std::max(science_npix, 0)));
            for (Eigen::Index r = 0; r < mb.signal[map_index].rows(); ++r) {
                for (Eigen::Index c = 0; c < mb.signal[map_index].cols(); ++c) {
                    const double value = mb.signal[map_index](r, c);
                    if (science_mask(r, c) && std::isfinite(value)) {
                        science_values.push_back(value);
                    }
                }
            }
            double background_level = 0.0;
            if (!science_values.empty() && edge_fill_mode == "core_median") {
                Eigen::Map<Eigen::VectorXd> signal_vec(science_values.data(),
                                                       static_cast<Eigen::Index>(science_values.size()));
                background_level = tula::alg::median(signal_vec);
            }
            mb.edge_guard_background_level[m_idx] = background_level;

            Eigen::MatrixXd edge_window = Eigen::MatrixXd::Zero(mb.n_rows, mb.n_cols);
            for (Eigen::Index r = 0; r < mb.n_rows; ++r) {
                for (Eigen::Index c = 0; c < mb.n_cols; ++c) {
                    if (science_mask(r, c)) {
                        edge_window(r, c) = 1.0;
                    }
                }
            }

            if (edge_taper_mode == "cosine" && science_npix > 0 && support_radius_pix > 0) {
                Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> taper_guard_mask = guardband_mask;
                const double taper_weight_floor = edge_taper_min_fraction * weight_threshold;
                taper_guard_mask = taper_guard_mask &&
                    ((mb.weight[map_index].array() > 0.0) &&
                     (mb.weight[map_index].array() >= taper_weight_floor));
                if (std::isfinite(hits_threshold) &&
                    !mb.coverage.empty() &&
                    map_index < static_cast<int>(mb.coverage.size())) {
                    const double taper_hits_floor = edge_taper_min_fraction * hits_threshold;
                    taper_guard_mask = taper_guard_mask &&
                        ((mb.coverage[map_index].array() > 0.0) &&
                         (mb.coverage[map_index].array() >= taper_hits_floor));
                }

                for (Eigen::Index r = 0; r < mb.n_rows; ++r) {
                    for (Eigen::Index c = 0; c < mb.n_cols; ++c) {
                        if (!taper_guard_mask(r, c)) {
                            continue;
                        }
                        double min_dist2 = std::numeric_limits<double>::infinity();
                        const Eigen::Index r0 = std::max<Eigen::Index>(0, r - support_radius_pix);
                        const Eigen::Index r1 = std::min<Eigen::Index>(mb.n_rows - 1, r + support_radius_pix);
                        const Eigen::Index c0 = std::max<Eigen::Index>(0, c - support_radius_pix);
                        const Eigen::Index c1 = std::min<Eigen::Index>(mb.n_cols - 1, c + support_radius_pix);
                        for (Eigen::Index rr = r0; rr <= r1; ++rr) {
                            for (Eigen::Index cc = c0; cc <= c1; ++cc) {
                                if (!science_mask(rr, cc)) {
                                    continue;
                                }
                                const double dr = static_cast<double>(rr - r);
                                const double dc = static_cast<double>(cc - c);
                                const double dist2 = dr * dr + dc * dc;
                                if (dist2 < min_dist2) {
                                    min_dist2 = dist2;
                                }
                            }
                        }
                        if (!std::isfinite(min_dist2)) {
                            continue;
                        }
                        const double frac = std::min(std::sqrt(min_dist2) / static_cast<double>(support_radius_pix), 1.0);
                        edge_window(r, c) = 0.5 * (1.0 + std::cos(M_PI * frac));
                    }
                }
            }
            else {
                for (Eigen::Index r = 0; r < mb.n_rows; ++r) {
                    for (Eigen::Index c = 0; c < mb.n_cols; ++c) {
                        if (guardband_mask(r, c)) {
                            edge_window(r, c) = 1.0;
                        }
                    }
                }
            }

            mb.edge_guard_support_npix[m_idx] =
                static_cast<int>((edge_window.array() > 0.0).count());
            mb.edge_guard_guardband_npix[m_idx] =
                std::max(0, mb.edge_guard_support_npix[m_idx] - science_npix);
            mb.edge_guard_support_frac[m_idx] =
                (n_pix > 0.0) ? static_cast<double>(mb.edge_guard_support_npix[m_idx]) / n_pix : 0.0;

            const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> effective_guardband_mask =
                (edge_window.array() > 0.0) && (!science_mask);
            const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> effective_exterior_mask =
                (edge_window.array() <= 0.0);

            auto calc_region_rms = [&](const Eigen::MatrixXd &matrix,
                                       const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> &mask) {
                double sumsq = 0.0;
                std::size_t count = 0;
                for (Eigen::Index r = 0; r < matrix.rows(); ++r) {
                    for (Eigen::Index c = 0; c < matrix.cols(); ++c) {
                        const double value = matrix(r, c);
                        if (mask(r, c) && std::isfinite(value)) {
                            sumsq += value * value;
                            ++count;
                        }
                    }
                }
                if (count == 0) {
                    return mapmaking::edge_guard_fill_double();
                }
                return std::sqrt(sumsq / static_cast<double>(count));
            };

            auto calc_region_max_abs = [&](const Eigen::MatrixXd &matrix,
                                           const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> &mask) {
                double max_abs = mapmaking::edge_guard_fill_double();
                for (Eigen::Index r = 0; r < matrix.rows(); ++r) {
                    for (Eigen::Index c = 0; c < matrix.cols(); ++c) {
                        const double value = matrix(r, c);
                        if (mask(r, c) && std::isfinite(value)) {
                            const double abs_value = std::abs(value);
                            if (!std::isfinite(max_abs) || abs_value > max_abs) {
                                max_abs = abs_value;
                            }
                        }
                    }
                }
                return max_abs;
            };

            mb.edge_guard_guardband_rms_pre[m_idx] = calc_region_rms(original_signal, effective_guardband_mask);
            mb.edge_guard_exterior_rms_pre[m_idx] = calc_region_rms(original_signal, effective_exterior_mask);
            mb.edge_guard_exterior_max_abs_pre[m_idx] = calc_region_max_abs(original_signal, effective_exterior_mask);

            for (Eigen::Index r = 0; r < mb.n_rows; ++r) {
                for (Eigen::Index c = 0; c < mb.n_cols; ++c) {
                    const double taper = edge_window(r, c);
                    if (taper >= 1.0) {
                        continue;
                    }
                    mb.signal[map_index](r, c) =
                        background_level + (mb.signal[map_index](r, c) - background_level) * taper;
                    guarded_weight(r, c) *= taper * taper;
                    if (!mb.kernel.empty()) {
                        mb.kernel[map_index](r, c) *= taper;
                    }
                }
            }
            mb.weight[map_index] = guarded_weight;
            mb.edge_guard_guardband_rms_post[m_idx] = calc_region_rms(mb.signal[map_index], effective_guardband_mask);
            mb.edge_guard_exterior_rms_post[m_idx] = calc_region_rms(mb.signal[map_index], effective_exterior_mask);
            mb.edge_guard_exterior_max_abs_post[m_idx] = calc_region_max_abs(mb.signal[map_index], effective_exterior_mask);
            if (m_idx < mb.edge_guard_window.size()) {
                mb.edge_guard_window[m_idx] = edge_window;
            }
            mb.edge_guard_applied[m_idx] = 1;
        }

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
            const double kernel_sum =
                citlali::pipeline::require_wiener_unit_sum_kernel(
                    kernel.sum(), kernel.cwiseAbs().sum(), filter_type,
                    template_type);
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

                    for (Eigen::Index i=0; i<n_rows; ++i) {
                        for (Eigen::Index j=0; j<n_cols; ++j) {
                            double w = weight_input(i,j);
                            // Every finite positive-weight sample passed to the
                            // signal/noise convolution must contribute to the
                            // propagated variance.  A coverage-cut threshold
                            // may classify output support, but it cannot
                            // silently remove noise from an input that still
                            // enters the fixed convolution.
                            if (mapmaking::convolve_stochastic_input_weight(w)) {
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

                    for (Eigen::Index i=0; i<n_rows; ++i) {
                        for (Eigen::Index j=0; j<n_cols; ++j) {
                            double m = mask_smooth(i,j);
                            if (mapmaking::convolve_has_numerical_variance_support(
                                    m, kernel_sq_sum)) {
                                // The signal is a fixed unit-sum convolution;
                                // it is not renormalized by the locally valid
                                // kernel support.  Its diagonal input-noise
                                // variance is therefore sum(k^2 * variance),
                                // not the support-normalized average variance.
                                double v = var_smooth(i,j);
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

        if (map_index < static_cast<int>(mb.edge_guard_window.size()) &&
            mb.edge_guard_window[map_index].rows() == mb.n_rows &&
            mb.edge_guard_window[map_index].cols() == mb.n_cols) {
            const auto &edge_window = mb.edge_guard_window[map_index];
            mb.signal[map_index].array() *= edge_window.array();
            mb.weight[map_index].array() *= edge_window.array().square();
            if (!mb.kernel.empty()) {
                mb.kernel[map_index].array() *= edge_window.array();
            }
        }

        SPDLOG_INFO("signal/weight map filtering done");
    }

    template<class MB>
    void filter_noise(MB &mb, const int map_index, const int noise_num) {
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(
            mb.noise[map_index].data() + noise_num * mb.n_rows * mb.n_cols, mb.n_rows, mb.n_cols);
        filtered_map = noise_matrix;
        if (map_index < static_cast<int>(mb.edge_guard_window.size()) &&
            mb.edge_guard_window[map_index].rows() == mb.n_rows &&
            mb.edge_guard_window[map_index].cols() == mb.n_cols) {
            filtered_map.array() *= mb.edge_guard_window[map_index].array();
        }

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
        if (map_index < static_cast<int>(mb.edge_guard_window.size()) &&
            mb.edge_guard_window[map_index].rows() == mb.n_rows &&
            mb.edge_guard_window[map_index].cols() == mb.n_cols) {
            ratio.array() *= mb.edge_guard_window[map_index].array();
        }
        noise_matrix.noalias() = ratio;
    }

    template<class MB>
    void filter_noise_threadsafe(MB &mb, const int map_index, const int noise_num) {
        const bool use_convolve = (filter_type=="convolve") || (filter_type=="wiener_filter" && run_lowpass);
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(
            mb.noise[map_index].data() + noise_num * mb.n_rows * mb.n_cols, mb.n_rows, mb.n_cols);

        Eigen::MatrixXd local_input = noise_matrix;
        if (map_index < static_cast<int>(mb.edge_guard_window.size()) &&
            mb.edge_guard_window[map_index].rows() == mb.n_rows &&
            mb.edge_guard_window[map_index].cols() == mb.n_cols) {
            local_input.array() *= mb.edge_guard_window[map_index].array();
        }
        Eigen::MatrixXd local_nume = Eigen::MatrixXd::Zero(n_rows, n_cols);
        if (use_convolve) {
            local_nume = run_convolve_on_input(local_input, true);
        }
        else if (filter_type=="wiener_filter") {
            local_nume = calc_numerator_from_input(local_input);
        }
        Eigen::MatrixXd ratio = use_convolve ? local_nume : divide_by_denom(local_nume, denom);
        if (map_index < static_cast<int>(mb.edge_guard_window.size()) &&
            mb.edge_guard_window[map_index].rows() == mb.n_rows &&
            mb.edge_guard_window[map_index].cols() == mb.n_cols) {
            ratio.array() *= mb.edge_guard_window[map_index].array();
        }
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
                if (ctx.a == nullptr || ctx.b == nullptr) {
                    ctx.reset();
                    citlali::pipeline::require_wiener_fftw_context(
                        false, rows, cols);
                }
                ctx.pf = fftw_plan_dft_2d(rows, cols, ctx.a, ctx.b, FFTW_FORWARD, FFTW_ESTIMATE);
                ctx.pr = fftw_plan_dft_2d(rows, cols, ctx.a, ctx.b, FFTW_BACKWARD, FFTW_ESTIMATE);
                if (ctx.pf == nullptr || ctx.pr == nullptr) {
                    ctx.reset();
                    citlali::pipeline::require_wiener_fftw_context(
                        false, rows, cols);
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
        engine_utils::fft2_into<engine_utils::forward>(in, out, ctx.pf, ctx.a, ctx.b);
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
            const double kernel_sum =
                citlali::pipeline::require_wiener_unit_sum_kernel(
                    kernel.sum(), kernel.cwiseAbs().sum(), filter_type,
                    template_type);
            kernel /= kernel_sum;
            in.real() = kernel;
            in.imag().setZero();
            engine_utils::fft2_into<engine_utils::forward>(in, out, ctx.pf, ctx.a, ctx.b);
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
    engine_utils::fft2_into<engine_utils::forward>(in, out, ctx.pf, ctx.a, ctx.b);

    in.real() = out.real().array() / vvq.array();
    in.imag() = out.imag().array() / vvq.array();
    engine_utils::fft2_into<engine_utils::inverse>(in, out, ctx.pr, ctx.a, ctx.b);

    in.real() = out.real().array() * rr.array();
    in.imag().setZero();
    engine_utils::fft2_into<engine_utils::forward>(in, out, ctx.pf, ctx.a, ctx.b);
    qqq = out;

    const auto &template_fft = get_filter_template_fft();
    in.real() = template_fft.real().array() * qqq.real().array() + template_fft.imag().array() * qqq.imag().array();
    in.imag() = -template_fft.imag().array() * qqq.real().array() + template_fft.real().array() * qqq.imag().array();
    engine_utils::fft2_into<engine_utils::inverse>(in, out, ctx.pr, ctx.a, ctx.b);

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
    engine_utils::fft2_into<engine_utils::forward>(in, out, ctx.pf, ctx.a, ctx.b);
    out = out * n_rows * n_cols;

    in.real() = out.real().array() * fft_filter.real().array() - out.imag().array() * fft_filter.imag().array();
    in.imag() = out.imag().array() * fft_filter.real().array() + out.real().array() * fft_filter.imag().array();
    engine_utils::fft2_into<engine_utils::inverse>(in, out, ctx.pr, ctx.a, ctx.b);
    out = out / n_rows / n_cols;

    return out.real();
}

inline Eigen::MatrixXd WienerFilter::divide_by_denom(const Eigen::MatrixXd &numerator,
                                                     const Eigen::MatrixXd &denominator) const {
    Eigen::MatrixXd ratio = Eigen::MatrixXd::Zero(numerator.rows(), numerator.cols());
    ratio.array() = (denominator.array() != 0.0).select(numerator.array() / denominator.array(), 0.0);
    return ratio;
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
    citlali::pipeline::require_wiener_kernel_weight_index(
        map_index, mb.kernel.size(), mb.weight.size());
    citlali::pipeline::require_wiener_kernel_geometry(
        map_index,
        mb.kernel[map_index].rows(), mb.kernel[map_index].cols(),
        mb.weight[map_index].rows(), mb.weight[map_index].cols(),
        n_rows, n_cols);

    // collect what we need
    Eigen::MatrixXd temp_kernel = mb.kernel[map_index];

    // Center kernel deterministically using the peak absolute response.
    // This avoids unstable Gaussian fitting failures in some coadd/kernel combinations.
    Eigen::Index peak_row = 0;
    Eigen::Index peak_col = 0;
    const double peak_abs = temp_kernel.cwiseAbs().maxCoeff(&peak_row, &peak_col);
    citlali::pipeline::require_finite_wiener_kernel_peak(
        peak_abs, map_index);
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
    const double tail_value = kernel_interp_valid(kernel_interp_valid.size() - 1);
    const double kernel_peak_abs = std::max(kernel_interp_valid.cwiseAbs().maxCoeff(), 1e-300);
    const double max_dist = dist.maxCoeff();
    const Eigen::Index tail_pixels = (dist.array() > s.x_max).count();
    logger->info("kernel template radial tail mode={} x_max={} max_dist={} tail_pixels={} tail_fraction={:.4f} tail_value={} tail_rel_peak={:.4g}",
                 kernel_template_tail_mode, s.x_max, max_dist,
                 static_cast<long long>(tail_pixels),
                 static_cast<double>(tail_pixels) / static_cast<double>(n_rows * n_cols),
                 tail_value, std::abs(tail_value) / kernel_peak_abs);
    auto radial_tail_value = [&](double radius) {
        if (kernel_template_tail_mode == "zero") {
            return 0.0;
        }
        if (kernel_template_tail_mode == "cosine" && max_dist > s.x_max) {
            constexpr double pi = 3.141592653589793238462643383279502884;
            const double frac = std::clamp((radius - s.x_max) / (max_dist - s.x_max), 0.0, 1.0);
            return tail_value * 0.5 * (1.0 + std::cos(pi * frac));
        }
        return tail_value;
    };

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
                filter_template(shiftj,shifti) = radial_tail_value(dist(j,i));
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
    last_denom_iters = 0;
    last_denom_rel_update = std::numeric_limits<double>::quiet_NaN();
    last_denom_tail_frac = std::numeric_limits<double>::quiet_NaN();
    last_denom_stop_reason = "not_run";

    // inputs and outputs to ffts
    Eigen::MatrixXcd in(n_rows,n_cols);
    Eigen::MatrixXcd out(n_rows,n_cols);

    // using uniform weights only
    if (uniform_weight) {
        const auto &template_fft = get_filter_template_fft();
        out = template_fft;

        // set denominator
        denom.setConstant(((out.real().array() * out.real().array() + out.imag().array() * out.imag().array()) / vvq.array()).sum());
        last_denom_rel_update = 0.0;
        last_denom_tail_frac = 0.0;
        last_denom_stop_reason = "uniform_weight";

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
        engine_utils::fft2_into<engine_utils::inverse>(in, out, pr, a, b);

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
        const Eigen::Index total_iters = n_rows * n_cols;
        std::vector<Eigen::Index> shift_rows_desc(total_iters);
        std::vector<Eigen::Index> shift_cols_desc(total_iters);
        std::vector<double> scales_desc(total_iters);
        std::vector<double> tail_fracs_desc(total_iters);

        // number of iterations for convergence
        n_loops = total_iters / 100;
        if (n_loops < 100) {
            n_loops = 100;
        }
        const double denom_rel_tol_local = denom_rel_tol;
        const double tail_frac_tol_local = tail_frac_tol;
        const double Z_abs_total = Z_abs.sum();
        double Z_abs_done = 0.0;
        Eigen::Index tail_cap_iters = total_iters;
        for (Eigen::Index kk=0; kk<total_iters; ++kk) {
            auto shift_index = std::get<1>(sorted[total_iters - kk - 1]);
            shift_rows_desc[kk] = -static_cast<Eigen::Index>(shift_index % n_rows);
            shift_cols_desc[kk] = -static_cast<Eigen::Index>(shift_index / n_rows);
            scales_desc[kk] = zz2d(shift_index) / static_cast<double>(n_rows * n_cols);
            Z_abs_done += Z_abs(shift_index);
            const double tail_frac = (Z_abs_total > 0.0) ? ((Z_abs_total - Z_abs_done) / Z_abs_total) : 0.0;
            tail_fracs_desc[kk] = tail_frac;
            if (tail_cap_iters == total_iters && tail_frac <= tail_frac_tol_local) {
                tail_cap_iters = kk + 1;
            }
        }
        // flag for convergence
        bool done = false;

        const auto denom_start = std::chrono::steady_clock::now();
        double last_log_s = 0.0;
        const Eigen::Index check_iters = denom_check_iters > 0 ? denom_check_iters : n_loops;
        const int max_checks = std::max(max_loops, 1);
        int checks_done = 0;
        const Eigen::Index requested_max_iters = max_denom_iters > 0 ? std::min<Eigen::Index>(max_denom_iters, total_iters) : total_iters;
        const Eigen::Index max_iters = max_denom_iters > 0 ? std::min<Eigen::Index>(requested_max_iters, tail_cap_iters) : tail_cap_iters;
        if (max_denom_iters > 0 && max_iters < tail_cap_iters) {
            logger->warn(
                "configured max_denom_iters={} is below Wiener denominator tail_cap_iters={}; "
                "using the configured cap with tail_frac_at_cap={:.4f}",
                static_cast<long long>(max_iters),
                static_cast<long long>(tail_cap_iters),
                tail_fracs_desc[max_iters - 1]);
        }
        logger->info("Wiener denominator total_iters={} tail_cap_iters={} max_iters={} check_iters={}",
                     static_cast<long long>(total_iters), static_cast<long long>(tail_cap_iters),
                     static_cast<long long>(max_iters), static_cast<long long>(check_iters));

        const int n_threads = std::max(omp_get_max_threads(), 1);
        std::vector<Eigen::MatrixXd> denom_partials;
        denom_partials.reserve(n_threads);
        for (int thread_id = 0; thread_id < n_threads; ++thread_id) {
            denom_partials.emplace_back(Eigen::MatrixXd::Zero(n_rows, n_cols));
        }
        Eigen::MatrixXd delta_since_check = Eigen::MatrixXd::Zero(n_rows, n_cols);

        for (Eigen::Index chunk_start = 0; chunk_start < max_iters && !done; chunk_start += check_iters) {
            const Eigen::Index chunk_end = std::min<Eigen::Index>(chunk_start + check_iters, max_iters);
            for (auto &partial : denom_partials) {
                partial.setZero();
            }

            std::exception_ptr fftw_context_error;
            #pragma omp parallel shared(shift_rows_desc, shift_cols_desc, scales_desc, chunk_start, chunk_end, n_rows, n_cols, denom_partials, filter_template, rr, fftw_context_error) default (none)
            {
                FFTWContext *ctx = nullptr;
                try {
                    ctx = &get_thread_fft_context(n_rows, n_cols);
                } catch (...) {
                    #pragma omp critical (wfFFTWContextError)
                    {
                        if (!fftw_context_error) {
                            fftw_context_error = std::current_exception();
                        }
                    }
                }

                #pragma omp barrier
                if (!fftw_context_error) {
                    const int thread_id = omp_get_thread_num();
                    auto &denom_local = denom_partials[thread_id];
                    Eigen::MatrixXcd in_local(n_rows, n_cols);
                    Eigen::MatrixXcd out_local(n_rows, n_cols);
                    Eigen::MatrixXcd ffdq(n_rows, n_cols);
                    Eigen::MatrixXd in_prod(n_rows, n_cols);
                    Eigen::MatrixXd shifted_template(n_rows, n_cols);
                    Eigen::MatrixXd shifted_rr(n_rows, n_cols);

                    #pragma omp for schedule(static)
                    for (Eigen::Index kk = chunk_start; kk < chunk_end; ++kk) {
                        const auto shift_row = shift_rows_desc[kk];
                        const auto shift_col = shift_cols_desc[kk];

                        engine_utils::shift_2D_into(filter_template, shift_row, shift_col, shifted_template);
                        in_prod = filter_template.array() * shifted_template.array();
                        in_local.real() = in_prod;
                        in_local.imag().setZero();
                        engine_utils::fft2_into<engine_utils::forward>(in_local, out_local, ctx->pf, ctx->a, ctx->b);
                        ffdq = out_local;

                        engine_utils::shift_2D_into(rr, shift_row, shift_col, shifted_rr);
                        in_prod = rr.array() * shifted_rr.array();
                        in_local.real() = in_prod;
                        in_local.imag().setZero();
                        engine_utils::fft2_into<engine_utils::forward>(in_local, out_local, ctx->pf, ctx->a, ctx->b);

                        in_local.real() = ffdq.real().array() * out_local.real().array() + ffdq.imag().array() * out_local.imag().array();
                        in_local.imag() = -ffdq.imag().array() * out_local.real().array() + ffdq.real().array() * out_local.imag().array();
                        engine_utils::fft2_into<engine_utils::inverse>(in_local, out_local, ctx->pr, ctx->a, ctx->b);

                        denom_local.array() += scales_desc[kk] * out_local.real().array();
                    }
                }
            }
            if (fftw_context_error) {
                std::rethrow_exception(fftw_context_error);
            }

            delta_since_check.setZero();
            for (const auto &partial : denom_partials) {
                denom.array() += partial.array();
                delta_since_check.array() += partial.array();
            }
            const double denom_norm = denom.norm();
            const double delta_norm = delta_since_check.norm();
            const double rel_update = delta_norm / std::max(denom_norm, 1e-12);
            const double tail_frac = tail_fracs_desc[chunk_end - 1];
            const double elapsed_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - denom_start).count();
            const double step_s = elapsed_s - last_log_s;
            last_log_s = elapsed_s;
            const double progress_pct = 100.0 * static_cast<double>(chunk_end) / static_cast<double>(max_iters);

            logger->info("{} iteration(s) complete. progress={:.1f}% rel_update={:.4g} tail_frac={:.4f} elapsed_s={:.2f} step_s={:.2f}",
                         static_cast<long long>(chunk_end), progress_pct, rel_update, tail_frac, elapsed_s, step_s);

            ++checks_done;
            last_denom_iters = chunk_end;
            last_denom_rel_update = rel_update;
            last_denom_tail_frac = tail_frac;
            if (rel_update < denom_rel_tol_local) {
                logger->info("Wiener denominator converged after {} iteration(s); rel_update={:.4g} tail_frac={:.4f}",
                             static_cast<long long>(chunk_end), rel_update, tail_frac);
                last_denom_stop_reason = "converged";
                done = true;
            }
            else if (checks_done >= max_checks) {
                logger->info("reached Wiener denominator max_loops={} after {} iteration(s); stopping early",
                             max_checks, static_cast<long long>(chunk_end));
                last_denom_stop_reason = "max_loops";
                done = true;
            }
            else if (chunk_end >= max_iters) {
                if (max_denom_iters > 0 && max_iters < tail_cap_iters) {
                    logger->info("reached configured Wiener denominator max_denom_iters={} before tail_cap_iters={}; stopping",
                                 static_cast<long long>(max_iters), static_cast<long long>(tail_cap_iters));
                    last_denom_stop_reason = "max_denom_iters";
                }
                else {
                    logger->info("reached Wiener denominator tail_cap_iters={} and stopping",
                                 static_cast<long long>(max_iters));
                    last_denom_stop_reason = "tail_cap_iters";
                }
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
