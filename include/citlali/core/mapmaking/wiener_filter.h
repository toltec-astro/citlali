#pragma once

#include <string>
#include <cmath>
#include <limits>
#include <algorithm>
#include <chrono>

#include <boost/math/special_functions/bessel.hpp>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/Splines>

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
    // optional pre-filter edge guard derived from weight/coverage maps
    bool edge_guard_enabled = true;
    std::string edge_weight_threshold_mode = "coverage_cut";
    std::string edge_hits_threshold_mode = "core_median_fraction";
    std::string edge_fill_mode = "core_median";
    std::string edge_taper_mode = "none";
    double edge_hits_core_fraction = 0.15;
    double edge_guard_radius_fwhm = 1.0;

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

    // make a symmetric Gaussian to use as a template
    template<class MB>
    void make_gaussian_template(MB &mb, const double);

    // make an Airy pattern to use as a template
    template<class MB>
    void make_airy_template(MB &mb, const double);

    // use a symmetric version of the kernel as a template
    template<class MB, class CD>
    void make_kernel_template(MB &mb, const int, CD &);

    // main function to determine what template to make
    template<class MB, class CD>
    void make_template(MB &, CD &c, const double, const int);

    // calculate standard deviations of each pixel
    template<class MB>
    void calc_rr(MB &, const int);

    // calculate normalized noise psd
    template <class MB>
    void calc_vvq(MB &, const int);

    // calculate the numerator
    void calc_numerator();

    // calculate the denominator
    void calc_denominator();

    // run the filter on the signal, weight, and kernel maps
    template<class MB>
    void run_filter(MB &, const int);

    // simple convolution with template
    void run_convolve(bool normalize=true);
    void invalidate_template_fft_cache() {
        filter_template_fft_valid = false;
        filter_template_fft_scaled_valid = false;
        filter_template_fft_normalized_scaled_valid = false;
    }
    const Eigen::MatrixXcd &get_filter_template_fft();
    const Eigen::MatrixXcd &get_filter_template_fft_scaled(bool);

    // test destriper
    void destripe(double);

    // filter a map
    template<class MB>
    void filter_maps(MB &, const int);

    // filter the noise maps
    template<class MB>
    void filter_noise(MB &mb, const int, const int);
};

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
    get_config_value(config, edge_guard_enabled, missing_keys, invalid_keys,
                     std::tuple{"post_processing","map_filtering","edge_guard","enabled"});
    get_config_value(config, edge_weight_threshold_mode, missing_keys, invalid_keys,
                     std::tuple{"post_processing","map_filtering","edge_guard","weight_threshold_mode"},
                     {"coverage_cut"});
    get_config_value(config, edge_hits_threshold_mode, missing_keys, invalid_keys,
                     std::tuple{"post_processing","map_filtering","edge_guard","hits_threshold_mode"},
                     {"core_median_fraction"});
    get_config_value(config, edge_hits_core_fraction, missing_keys, invalid_keys,
                     std::tuple{"post_processing","map_filtering","edge_guard","hits_core_fraction"},
                     {}, {0.0});
    get_config_value(config, edge_guard_radius_fwhm, missing_keys, invalid_keys,
                     std::tuple{"post_processing","map_filtering","edge_guard","guard_radius_fwhm"},
                     {}, {0.0});
    get_config_value(config, edge_fill_mode, missing_keys, invalid_keys,
                     std::tuple{"post_processing","map_filtering","edge_guard","fill_mode"},
                     {"core_median"});
    get_config_value(config, edge_taper_mode, missing_keys, invalid_keys,
                     std::tuple{"post_processing","map_filtering","edge_guard","taper_mode"},
                     {"none"});
    // denominator convergence thresholds
    get_config_value(config, denom_rel_tol, missing_keys, invalid_keys,
                     std::tuple{"wiener_filter","denom_rel_tol"}, {}, {0.0}, {1.0});
    get_config_value(config, tail_frac_tol, missing_keys, invalid_keys,
                     std::tuple{"wiener_filter","tail_frac_tol"}, {}, {0.0}, {1.0});
    get_config_value(config, max_loops, missing_keys, invalid_keys,
                     std::tuple{"wiener_filter","max_loops"}, {}, {1});
    get_config_value(config, denom_check_iters, missing_keys, invalid_keys,
                     std::tuple{"wiener_filter","denom_check_iters"}, {}, {0});
    get_config_value(config, max_denom_iters, missing_keys, invalid_keys,
                     std::tuple{"wiener_filter","max_denom_iters"}, {}, {0});

    // gaussian or airy template fwhms
    if (template_type=="gaussian" || template_type=="airy") {
        // loop through array names and get fwhms
        for (auto const& [arr_index, arr_name] : toltec_io.array_name_map) {
            get_config_value(config, template_fwhm_rad[arr_name], missing_keys, invalid_keys,
                             std::tuple{"wiener_filter","template_fwhm_arcsec",arr_name});
        }
        // convert to radians
        for (auto const& pair : template_fwhm_rad) {
            template_fwhm_rad[pair.first] = template_fwhm_rad[pair.first]*ASEC_TO_RAD;
        }
    }
}

template<class MB>
void WienerFilter::make_gaussian_template(MB &mb, const double template_fwhm_rad) {
    // distance from tangent point
    Eigen::MatrixXd dist(n_rows,n_cols);

    // calculate distance
    for (Eigen::Index i=0; i<n_rows; ++i) {
        for (Eigen::Index j=0; j<n_cols; ++j) {
            dist(i,j) = sqrt(pow(mb.rows_tan_vec(i),2) + pow(mb.cols_tan_vec(j),2));
        }
    }

    // to hold minimum distance
    Eigen::Index row_index, col_index;

    // minimum distance
    double min_dist = dist.minCoeff(&row_index,&col_index);
    // standard deviation
    double sigma = template_fwhm_rad*FWHM_TO_STD;

    // shift indices
    std::vector<Eigen::Index> shift_indices = {-row_index, -col_index};

    // calculate template
    filter_template = exp(-0.5 * pow(dist.array() / sigma, 2.));
    // shift template
    filter_template = engine_utils::shift_2D(filter_template, shift_indices);
}

template<class MB>
void WienerFilter::make_airy_template(MB &mb, const double template_fwhm_rad) {
    // distance from tangent point
    Eigen::MatrixXd dist(n_rows,n_cols);

    // calculate distance
    for (Eigen::Index i=0; i<n_rows; ++i) {
        for (Eigen::Index j=0; j<n_cols; ++j) {
            dist(i,j) = sqrt(pow(mb.rows_tan_vec(i),2) + pow(mb.cols_tan_vec(j),2));
        }
    }

    // to hold minimum distance
    Eigen::Index row_index, col_index;

    // minimum distance
    double min_dist = dist.minCoeff(&row_index,&col_index);

    // shift indices
    std::vector<Eigen::Index> shift_indices = {-row_index, -col_index};

    // calculate template
    double factor = pi*(1.028/template_fwhm_rad);

    // resize template
    filter_template.resize(n_rows, n_cols);

    // populate template
    for (Eigen::Index i=0; i<n_rows; ++i) {
        for (Eigen::Index j=0; j<n_cols; ++j) {
            if (dist(i,j)!=0) {
            filter_template(i,j) = pow(2*boost::math::cyl_bessel_j(1,factor*dist(i,j))/(factor*dist(i,j)),2);
            }
            else {
                filter_template(i,j) = 1;
            }
        }
    }

    // shift template
    filter_template = engine_utils::shift_2D(filter_template, shift_indices);
}

template<class MB, class CD>
void WienerFilter::make_kernel_template(MB &mb, const int map_index, CD &calib_data) {
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
    // calculate distance
    for (Eigen::Index i=0; i<n_rows; ++i) {
        for (Eigen::Index j=0; j<n_cols; ++j) {
            dist(i,j) = sqrt(pow(mb.rows_tan_vec(i),2) + pow(mb.cols_tan_vec(j),2));
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
    for (Eigen::Index i=0; i<n_bins-1; ++i) {
        int c = 0;
        for (Eigen::Index j=0; j<n_cols; ++j) {
            for (Eigen::Index k=0; k<n_rows; ++k) {
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
        logger->warn("kernel template radial averages are undersampled; using shifted kernel map directly");
        filter_template = temp_kernel;
        return;
    }

    Eigen::VectorXd kernel_interp_valid = Eigen::Map<Eigen::VectorXd>(kernel_valid.data(), kernel_valid.size());
    Eigen::VectorXd dist_interp_valid = Eigen::Map<Eigen::VectorXd>(dist_valid.data(), dist_valid.size());

    // now spline interpolate to generate new template array
    filter_template.resize(n_rows,n_cols);

    // create spline function
    engine_utils::SplineFunction s(dist_interp_valid, kernel_interp_valid);

    // carry out the interpolation
    for (Eigen::Index i=0; i<n_cols; ++i) {
        Eigen::Index ti = (i-col_index)%n_cols;
        Eigen::Index shifti = (ti < 0) ? n_cols+ti : ti;
        for (Eigen::Index j=0; j<n_rows; ++j) {
            Eigen::Index tj = (j-row_index)%n_rows;
            Eigen::Index shiftj = (tj < 0) ? n_rows+tj : tj;

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

// calculate standard deviations of each pixel
template<class MB>
void WienerFilter::calc_rr(MB &mb, const int map_index) {
    if (uniform_weight) {
        rr = Eigen::MatrixXd::Ones(n_rows,n_cols);
    }
    else {
        rr = sqrt(mb.weight[map_index].array());
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
        logger->warn("noise PSDs missing for map {}; falling back to lowpass-only response", map_index);
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
            logger->warn("noise PSD invalid for map {}; falling back to lowpass-only response", map_index);
            psd_q.setOnes();
        }
        else {

        for (Eigen::Index i=0; i<n_psd; ++i) {
            if (psd(i)/max_psd < psd_lim) {
                psd_freq_break = psd_freq(i);
                break;
            }
        }

        // number of frequency samples below lowpass break
        int count = (psd_freq.array() <= 0.8*psd_freq_break).count();

        // flatten the response above the lowpass break
        if (count > 0) {
            for (Eigen::Index i=0; i<n_psd; ++i) {
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

        Eigen::MatrixXd q_map(n_rows,n_cols);

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
                q_map(i,j) = sqrt(pow(q_row(i),2)+pow(q_col(j),2));
            }
        }

        // set psd q to zero
        psd_q.setZero();

        Eigen::Matrix<Eigen::Index, 1, 1> n_psd_matrix;
        n_psd_matrix << n_psd;

        // interpolate onto psd_q
        Eigen::Index interp_pts = 1;
        for (Eigen::Index i=0; i<n_cols; ++i) {
            for (Eigen::Index j=0; j<n_rows; ++j) {
                if ((q_map(j,i) <= psd_freq(psd_freq.size() - 1)) && (q_map(j,i) >= psd_freq(0))) {
                    mlinterp::interp<mlinterp::rnatord>(n_psd_matrix.data(), interp_pts,
                                     psd.data(), psd_q.data() + n_rows * i + j,
                                     psd_freq.data(), q_map.data() + n_rows * i + j);
                }
                else if (q_map(j,i) > psd_freq(n_psd - 1)) {
                    psd_q(j,i) = psd(n_psd- 1);
                }
                else if (q_map(j,i) < psd_freq(0)) {
                    psd_q(j,i) = psd(0);
                }
            }
        }

        // find the minimum value of psd and clamp
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
        //psd_q = psd_q.array().max(psd_min).matrix();
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
    // set up fftw
    fftw_complex *a;
    fftw_complex *b;
    fftw_plan pf, pr;

    // allocate space for 2d ffts
    a = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_rows*n_cols);
    b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_rows*n_cols);

    // fftw plans
    pf = fftw_plan_dft_2d(n_rows, n_cols, a, b, FFTW_FORWARD, FFTW_ESTIMATE);
    pr = fftw_plan_dft_2d(n_rows, n_cols, a, b, FFTW_BACKWARD, FFTW_ESTIMATE);

    // set up inputs and outputs
    Eigen::MatrixXcd in(n_rows,n_cols), out(n_rows,n_cols);
    // d x RR
    in.real() = rr.array() * filtered_map.array();
    in.imag().setZero();

    // fft(d x RR)
    out = engine_utils::fft2<engine_utils::forward>(in, pf, a, b);

    // fft(d x RR) x 1/VV
    in.real() = out.real().array() / vvq.array();
    in.imag() = out.imag().array() / vvq.array();

    // ifft(fft(d x RR) x 1/VV)
    out = engine_utils::fft2<engine_utils::inverse>(in, pr, a, b);

    // Q = ifft(fft(d x RR) x 1/VV) x RR
    in.real() = out.real().array() * rr.array();
    in.imag().setZero();

    // fft(Q)
    out = engine_utils::fft2<engine_utils::forward>(in, pf, a, b);

    // copy of fft(Q)
    Eigen::MatrixXcd Q = out;

    // cached fft(f(x))
    const auto &template_fft = get_filter_template_fft();

    // fft(f(x)) x fft(Q) (convolution)
    in.real() = template_fft.real().array() * Q.real().array() + template_fft.imag().array() * Q.imag().array();
    in.imag() = -template_fft.imag().array() * Q.real().array() + template_fft.real().array() * Q.imag().array();

    // ifft(fft(f(x)) x fft(Q))
    out = engine_utils::fft2<engine_utils::inverse>(in, pr, a, b);

    // populate numerator with real(ifft(fft(f(x)) x fft(Q)))
    nume = out.real();

    // destroy fftw plans
    fftw_destroy_plan(pf);
    fftw_destroy_plan(pr);
    // free fftw vectors
    fftw_free(a);
    fftw_free(b);
}

inline const Eigen::MatrixXcd &WienerFilter::get_filter_template_fft() {
    if (!filter_template_fft_valid) {
        fftw_complex *a = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_rows*n_cols);
        fftw_complex *b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_rows*n_cols);
        fftw_plan pf = fftw_plan_dft_2d(n_rows, n_cols, a, b, FFTW_FORWARD, FFTW_ESTIMATE);

        Eigen::MatrixXcd in(n_rows, n_cols), out(n_rows, n_cols);
        in.real() = filter_template;
        in.imag().setZero();
        out = engine_utils::fft2<engine_utils::forward>(in, pf, a, b);
        filter_template_fft = std::move(out);
        filter_template_fft_valid = true;

        fftw_destroy_plan(pf);
        fftw_free(a);
        fftw_free(b);
    }
    return filter_template_fft;
}

inline const Eigen::MatrixXcd &WienerFilter::get_filter_template_fft_scaled(bool normalize) {
    const double scale = static_cast<double>(n_rows) * static_cast<double>(n_cols);
    if (normalize) {
        if (!filter_template_fft_normalized_scaled_valid) {
            fftw_complex *a = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_rows*n_cols);
            fftw_complex *b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_rows*n_cols);
            fftw_plan pf = fftw_plan_dft_2d(n_rows, n_cols, a, b, FFTW_FORWARD, FFTW_ESTIMATE);

            Eigen::MatrixXcd in(n_rows, n_cols), out(n_rows, n_cols);
            Eigen::MatrixXd kernel = filter_template;
            const double kernel_sum = kernel.sum();
            if (kernel_sum != 0.0 && std::isfinite(kernel_sum)) {
                kernel /= kernel_sum;
            }
            in.real() = kernel;
            in.imag().setZero();
            out = engine_utils::fft2<engine_utils::forward>(in, pf, a, b);
            out *= scale;
            filter_template_fft_normalized_scaled = std::move(out);
            filter_template_fft_normalized_scaled_valid = true;

            fftw_destroy_plan(pf);
            fftw_free(a);
            fftw_free(b);
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

void WienerFilter::calc_denominator() {
    // set up fftw
    fftw_complex *a;
    fftw_complex *b;
    fftw_plan pf, pr;

    // allocate space for 2d ffts
    a = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_rows*n_cols);
    b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_rows*n_cols);

    // fftw plans
    pf = fftw_plan_dft_2d(n_rows, n_cols, a, b, FFTW_FORWARD, FFTW_ESTIMATE);
    pr = fftw_plan_dft_2d(n_rows, n_cols, a, b, FFTW_BACKWARD, FFTW_ESTIMATE);

    // resize denominator
    denom.setZero(n_rows,n_cols);

    // inputs and outputs to ffts
    Eigen::MatrixXcd in(n_rows,n_cols), out(n_rows,n_cols);

    // using uniform weights only
    if (uniform_weight) {
        const auto &template_fft = get_filter_template_fft();
        out = template_fft;

        // set denominator = abs(fft(f(x))/VV
        denom.setConstant(((out.real().array() * out.real().array() + out.imag().array() * out.imag().array()) / vvq.array()).sum());
    }
    else {
        // initialize denominator
        denom.setZero();

        // 1/VV
        in.real() = pow(vvq.array(),-1);
        in.imag().setZero();

        // Z = ifft(1/VV)
        out = engine_utils::fft2<engine_utils::inverse>(in, pr, a, b);

        // flattened real(Z) array
        Eigen::VectorXd Z(n_rows * n_cols);

        // real(Z).  do loop to make sure colmajor is preserved
        for (Eigen::Index i=0; i<n_cols; ++i) {
            for (Eigen::Index j=0; j<n_rows;++j) {
                int ii = n_rows*i+j;
                Z(ii) = (out.real()(j,i));
            }
        }

        // sort absolute values of Z in ascending order
        Eigen::VectorXd Z_abs = Z.array().abs();
        auto Z_indices_sorted = engine_utils::sorter(Z_abs);
        const Eigen::Index total_iters = n_rows * n_cols;
        std::vector<Eigen::Index> shift_indices_desc(total_iters);
        std::vector<Eigen::Index> shift_rows_desc(total_iters);
        std::vector<Eigen::Index> shift_cols_desc(total_iters);
        std::vector<double> scales_desc(total_iters);
        std::vector<double> tail_fracs_desc(total_iters);

        // number of iterations between convergence checks
        n_loops = total_iters / 100;
        if (n_loops < 100) {
            n_loops = 100;
        }
        const double Z_abs_total = Z_abs.sum();
        double Z_abs_done = 0.0;
        Eigen::Index tail_cap_iters = total_iters;
        for (Eigen::Index kk=0; kk<total_iters; ++kk) {
            auto shift_index = std::get<1>(Z_indices_sorted[total_iters - kk - 1]);
            shift_indices_desc[kk] = shift_index;
            shift_rows_desc[kk] = -static_cast<Eigen::Index>(shift_index % n_rows);
            shift_cols_desc[kk] = -static_cast<Eigen::Index>(shift_index / n_rows);
            scales_desc[kk] = Z(shift_index) / static_cast<double>(n_rows * n_cols);
            Z_abs_done += Z_abs(shift_index);
            const double tail_frac = (Z_abs_total > 0.0) ? ((Z_abs_total - Z_abs_done) / Z_abs_total) : 0.0;
            tail_fracs_desc[kk] = tail_frac;
            if (tail_cap_iters == total_iters && tail_frac <= tail_frac_tol) {
                tail_cap_iters = kk + 1;
            }
        }
        Z_abs_done = 0.0;

        // flag for convergence
        bool done = false;

        const auto denom_start = std::chrono::steady_clock::now();
        double last_log_s = 0.0;
        const Eigen::Index check_iters = denom_check_iters > 0 ? denom_check_iters : n_loops;
        const int max_checks = std::max(max_loops, 1);
        int checks_done = 0;
        const Eigen::Index requested_max_iters = max_denom_iters > 0 ? std::min<Eigen::Index>(max_denom_iters, total_iters) : total_iters;
        const Eigen::Index max_iters = std::min(requested_max_iters, tail_cap_iters);
        tula::logging::progressbar pb(
            [&](const auto &msg) { logger->info("{}", msg); }, 90,
            "calculating denom");
        const Eigen::Index pb_stride = std::max<Eigen::Index>(max_iters / 100, 1);
        logger->info("Wiener denominator pre-cap total_iters={} tail_cap_iters={} max_iters={} check_iters={}",
                     static_cast<long long>(total_iters), static_cast<long long>(tail_cap_iters),
                     static_cast<long long>(max_iters), static_cast<long long>(check_iters));

        Eigen::MatrixXcd in(n_rows,n_cols), out(n_rows,n_cols);
        Eigen::MatrixXcd ffdq(n_rows,n_cols);
        Eigen::MatrixXd in_prod(n_rows,n_cols);
        Eigen::MatrixXd shifted_template(n_rows,n_cols);
        Eigen::MatrixXd shifted_rr(n_rows,n_cols);

        // loop through cols and rows
        for (Eigen::Index k=0; k<n_cols; ++k) {
            for (Eigen::Index l=0; l<n_rows; ++l) {
                const Eigen::Index kk = n_rows * k + l;
                if (kk >= max_iters) {
                    done = true;
                    break;
                }
                if (!done) {
                    const auto shift_index = shift_indices_desc[kk];
                    const auto shift_row = shift_rows_desc[kk];
                    const auto shift_col = shift_cols_desc[kk];

                    // f(x) x f(x-x_d)
                    engine_utils::shift_2D_into(filter_template, shift_row, shift_col, shifted_template);
                    in_prod = filter_template.array() * shifted_template.array();

                    // populate matrices for fft
                    in.real() = in_prod;
                    in.imag().setZero();

                    // fft(f(x) x f(x-x_d))
                    out = engine_utils::fft2<engine_utils::forward>(in, pf, a, b);

                    // copy of fft(f(x) x f(x-x_d))
                    ffdq = out;

                    // R(x) x R(x-x_d)
                    engine_utils::shift_2D_into(rr, shift_row, shift_col, shifted_rr);
                    in_prod = rr.array() * shifted_rr.array();

                    // populate matrices for fft
                    in.real() = in_prod;
                    in.imag().setZero();

                    // fft(R(x) x R(x-x_d))
                    out = engine_utils::fft2<engine_utils::forward>(in, pf, a, b);

                    // fft(f(x) x f(x-x_d)) x fft(R(x) x R(x-x_d))
                    in.real() = ffdq.real().array() * out.real().array() + ffdq.imag().array() * out.imag().array();
                    in.imag() = -ffdq.imag().array() * out.real().array() + ffdq.real().array() * out.imag().array();

                    // G = ifft(fft(f(x) x f(x-x_d)) x fft(R(x) x R(x-x_d)))
                    out = engine_utils::fft2<engine_utils::inverse>(in, pr, a, b);

                    // Z(x_d) x G/n_pixels
                    Eigen::MatrixXd delta_denom = scales_desc[kk] * out.real();

                    // D = D + Z(x_d) x G/n_pixels
                    denom = denom.array() + delta_denom.array();
                    Z_abs_done += Z_abs(shift_index);
                    pb.count(max_iters, pb_stride);

                    // update status
                    if ((kk % check_iters) == 1) {
                        const double denom_norm = denom.norm();
                        const double delta_norm = delta_denom.norm();
                        const double rel_update = delta_norm / std::max(denom_norm, 1e-12);
                        const double tail_frac = tail_fracs_desc[kk];

                        const double elapsed_s = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - denom_start).count();
                        const double step_s = elapsed_s - last_log_s;
                        last_log_s = elapsed_s;

                        logger->info("{} iteration(s) complete. rel_update={} tail_frac={} elapsed_s={} step_s={}",
                                     kk, static_cast<float>(rel_update), static_cast<float>(tail_frac),
                                     static_cast<float>(elapsed_s), static_cast<float>(step_s));

                        ++checks_done;
                        if (rel_update < denom_rel_tol && tail_frac < tail_frac_tol) {
                            done = true;
                        }
                        else if (checks_done >= max_checks) {
                            logger->info("reached Wiener denominator max_loops={} after {} iteration(s); stopping early",
                                         max_checks, kk);
                            done = true;
                        }
                    }
                    else if (kk + 1 >= max_iters) {
                        logger->info("reached Wiener denominator max_denom_iters={} and stopping",
                                     static_cast<long long>(max_iters));
                        done = true;
                    }
                }
            }
            if (done) {
                break;
            }
        }

        // zero out extremely small denom values
        for (Eigen::Index i=0; i<n_rows; ++i) {
            for (Eigen::Index j=0; j<n_cols; ++j) {
                if (denom(i,j) < denom_limit) {
                    denom(i,j) = 0;
                }
            }
        }
    }

    // destroy fftw plans
    fftw_destroy_plan(pf);
    fftw_destroy_plan(pr);
    // free fftw vectors
    fftw_free(a);
    fftw_free(b);
}

template<class MB, class CD>
void WienerFilter::make_template(MB &mb, CD &calib_data, const double template_fwhm_rad, const int map_index) {
    // make sure filtered maps have even dimensions
    n_rows = mb.n_rows;
    n_cols = mb.n_cols;

    logger->info(
        "make_template precheck: map_index={} template_type={} n_rows={} n_cols={} rows_size={} cols_size={} kernel_size={}",
        map_index, template_type, static_cast<long long>(n_rows), static_cast<long long>(n_cols),
        static_cast<long long>(mb.rows_tan_vec.size()), static_cast<long long>(mb.cols_tan_vec.size()),
        static_cast<long long>(mb.kernel.size()));

    if (n_rows < 2 || n_cols < 2 ||
        mb.rows_tan_vec.size() < 2 || mb.cols_tan_vec.size() < 2) {
        logger->error(
            "invalid map geometry for Wiener template: n_rows={} n_cols={} rows_size={} cols_size={}",
            static_cast<long long>(n_rows), static_cast<long long>(n_cols),
            static_cast<long long>(mb.rows_tan_vec.size()), static_cast<long long>(mb.cols_tan_vec.size()));
        std::exit(EXIT_FAILURE);
    }

    // x and y spacing should be equal
    diff_rows = std::abs(mb.rows_tan_vec(1) - mb.rows_tan_vec(0));
    diff_cols = std::abs(mb.cols_tan_vec(1) - mb.cols_tan_vec(0));
    if (!std::isfinite(diff_rows) || !std::isfinite(diff_cols) || diff_rows <= 0.0 || diff_cols <= 0.0) {
        logger->error("invalid tangent-plane pixel spacing: diff_rows={} diff_cols={}", diff_rows, diff_cols);
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

    // symmetric version of kernel template
    else {
        logger->info("creating template from kernel map");
        if (mb.kernel.empty() || map_index < 0 || map_index >= static_cast<int>(mb.kernel.size())) {
            logger->error(
                "kernel template requested but kernel map is unavailable: map_index={} kernel_size={}",
                map_index, static_cast<long long>(mb.kernel.size()));
            std::exit(EXIT_FAILURE);
        }
        make_kernel_template(mb, map_index, calib_data);
    }
    invalidate_template_fft_cache();
}

template<class MB>
void WienerFilter::run_filter(MB &mb, const int map_index) {
    const auto t0 = std::chrono::steady_clock::now();
    // calculate pixel standard deviations
    logger->debug("calculating rr");
    calc_rr(mb, map_index);
    logger->debug("rr {}", rr);

    const auto t1 = std::chrono::steady_clock::now();
    // calculate normalized psd
    logger->debug("calculating vvq");
    calc_vvq(mb, map_index);
    logger->debug("vvq {}", vvq);

    const auto t2 = std::chrono::steady_clock::now();
    // calculate denominator
    logger->debug("calculating denominator");
    calc_denominator();
    logger->debug("denominator {}", denom);

    const auto t3 = std::chrono::steady_clock::now();
    // calculate numerator
    logger->debug("calculating numerator");
    calc_numerator();
    logger->debug("numerator {}", nume);
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

void WienerFilter::run_convolve(bool normalize) {
    // set up fftw
    fftw_complex *a;
    fftw_complex *b;
    fftw_plan pf, pr;

    // allocate space for 2d ffts
    a = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_rows*n_cols);
    b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_rows*n_cols);

    // fftw plans
    pf = fftw_plan_dft_2d(n_rows, n_cols, a, b, FFTW_FORWARD, FFTW_ESTIMATE);
    pr = fftw_plan_dft_2d(n_rows, n_cols, a, b, FFTW_BACKWARD, FFTW_ESTIMATE);

    // inputs and outputs to ffts
    Eigen::MatrixXcd in(n_rows,n_cols), out(n_rows,n_cols);

    const auto &fft_filter = get_filter_template_fft_scaled(normalize);

    in.real() = filtered_map;
    in.imag().setZero();

    out = engine_utils::fft2<engine_utils::forward>(in, pf, a, b);
    out = out*n_rows*n_cols;

    // convolution
    in.real() = out.real().array() * fft_filter.real().array() - out.imag().array() * fft_filter.imag().array();
    in.imag() = out.imag().array() * fft_filter.real().array() + out.real().array() * fft_filter.imag().array();

    out = engine_utils::fft2<engine_utils::inverse>(in, pr, a, b);
    out = out/n_rows/n_cols;

    nume = out.real();
    denom.setOnes(n_rows,n_cols);

    // destroy fftw plans
    fftw_destroy_plan(pf);
    fftw_destroy_plan(pr);
    // free fftw vectors
    fftw_free(a);
    fftw_free(b);
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

    logger->info("number of pixels below threshold {}", n_pixels);

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

template<class MB>
void WienerFilter::filter_maps(MB &mb, const int map_index) {
    const bool use_convolve = (filter_type=="convolve") || (filter_type=="wiener_filter" && run_lowpass);
    if (mb.edge_guard_applied.size() != static_cast<std::size_t>(mb.signal.size())) {
        const auto n_maps_local = static_cast<std::size_t>(mb.signal.size());
        mb.edge_guard_applied.assign(n_maps_local, 0);
        mb.edge_guard_support_radius_pix.assign(n_maps_local, 0);
        mb.edge_guard_science_npix.assign(n_maps_local, 0);
        mb.edge_guard_support_npix.assign(n_maps_local, 0);
        mb.edge_guard_guardband_npix.assign(n_maps_local, 0);
        mb.edge_guard_weight_threshold.assign(n_maps_local, std::numeric_limits<double>::quiet_NaN());
        mb.edge_guard_hits_threshold.assign(n_maps_local, std::numeric_limits<double>::quiet_NaN());
        mb.edge_guard_background_level.assign(n_maps_local, std::numeric_limits<double>::quiet_NaN());
        mb.edge_guard_science_frac.assign(n_maps_local, std::numeric_limits<double>::quiet_NaN());
        mb.edge_guard_support_frac.assign(n_maps_local, std::numeric_limits<double>::quiet_NaN());
    }

    const auto m_idx = static_cast<std::size_t>(map_index);
    mb.edge_guard_applied[m_idx] = 0;
    mb.edge_guard_support_radius_pix[m_idx] = 0;
    mb.edge_guard_science_npix[m_idx] = 0;
    mb.edge_guard_support_npix[m_idx] = 0;
    mb.edge_guard_guardband_npix[m_idx] = 0;
    mb.edge_guard_weight_threshold[m_idx] = std::numeric_limits<double>::quiet_NaN();
    mb.edge_guard_hits_threshold[m_idx] = std::numeric_limits<double>::quiet_NaN();
    mb.edge_guard_background_level[m_idx] = std::numeric_limits<double>::quiet_NaN();
    mb.edge_guard_science_frac[m_idx] = std::numeric_limits<double>::quiet_NaN();
    mb.edge_guard_support_frac[m_idx] = std::numeric_limits<double>::quiet_NaN();

    Eigen::MatrixXd guarded_weight = mb.weight[map_index];
    if (edge_guard_enabled) {
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

        double hits_threshold = std::numeric_limits<double>::quiet_NaN();
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
        mb.edge_guard_support_npix[m_idx] = static_cast<int>(support_mask.count());
        mb.edge_guard_guardband_npix[m_idx] = static_cast<int>(guardband_mask.count());
        mb.edge_guard_support_frac[m_idx] =
            (n_pix > 0.0) ? static_cast<double>(mb.edge_guard_support_npix[m_idx]) / n_pix : 0.0;

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

        for (Eigen::Index r = 0; r < mb.n_rows; ++r) {
            for (Eigen::Index c = 0; c < mb.n_cols; ++c) {
                if (!support_mask(r, c)) {
                    mb.signal[map_index](r, c) = background_level;
                    guarded_weight(r, c) = 0.0;
                    if (!mb.kernel.empty()) {
                        mb.kernel[map_index](r, c) = 0.0;
                    }
                }
            }
        }
        mb.weight[map_index] = guarded_weight;
        mb.edge_guard_applied[m_idx] = 1;
    }

    Eigen::MatrixXd weight_input;
    if (use_convolve) {
        weight_input = mb.weight[map_index];
    }
    /*if (filter_type=="destripe") {
        filtered_map = mb.signal[map_index];
        destripe(0.5);
        mb.signal[map_index] = filtered_map;
    }*/

    // filter kernel (only if present)
    if (!mb.kernel.empty()) {
        logger->info("filtering kernel");
        filtered_map = mb.kernel[map_index];
        // run all filter steps
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
                    mb.kernel[map_index](i,j) = nume(i,j)/denom(i,j);
                }
                else {
                    mb.kernel[map_index](i,j) = 0.0;
                }
            }
        }

        logger->info("kernel filtering done");
    }

    logger->info("filtering signal");
    // filter signal
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
                mb.signal[map_index](i,j) = 0.0;
            }
        }
    }
    if (filter_type=="wiener_filter" && !use_convolve) {
        // weight map is the denominator
        mb.weight[map_index] = denom;
    }
    else if (use_convolve) {
        // propagate inverse-variance through smoothing: Var_smooth = (k^2) ⊗ Var
        Eigen::MatrixXd kernel = filter_template;
        double kernel_sum = kernel.sum();
        if (kernel_sum == 0.0 || !std::isfinite(kernel_sum)) {
            logger->warn("convolve kernel sum is zero/invalid; skipping weight propagation");
        }
        else {
            kernel /= kernel_sum;
            Eigen::MatrixXd kernel_sq = kernel.array().square().matrix();
            double kernel_sq_sum = kernel_sq.sum();
            if (kernel_sq_sum == 0.0 || !std::isfinite(kernel_sq_sum)) {
                logger->warn("convolve kernel^2 sum is zero/invalid; skipping weight propagation");
            }
            else {
                Eigen::MatrixXd var_map(n_rows, n_cols);
                Eigen::MatrixXd mask_map(n_rows, n_cols);

                // use coverage-based threshold to avoid huge variances from tiny weights
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

    logger->info("signal/weight map filtering done");
}

template<class MB>
void WienerFilter::filter_noise(MB &mb, const int map_index, const int noise_num) {
    // get noise map
    filtered_map = Eigen::Map<Eigen::MatrixXd>(mb.noise[map_index].data() + noise_num * mb.n_rows * mb.n_cols,
                                               mb.n_rows, mb.n_cols);

    const bool use_convolve = (filter_type=="convolve") || (filter_type=="wiener_filter" && run_lowpass);
    // don't need to run through the whole filter, just the numerator
    if (use_convolve) {
        run_convolve();
    }
    else if (filter_type=="wiener_filter") {
        calc_numerator();
    }

    Eigen::MatrixXd ratio(n_rows,n_cols);

    // divide by filtered weight
    for (Eigen::Index i=0; i<n_rows; ++i) {
        for (Eigen::Index j=0; j<n_cols; ++j) {
            if (denom(i,j) != 0.0) {
                ratio(i,j) = nume(i,j)/denom(i,j);
            }
            else {
                ratio(i,j) = 0.0;
            }
        }
    }

    // map to tensor
    Eigen::TensorMap<Eigen::Tensor<double, 2>> in_tensor(ratio.data(), ratio.rows(), ratio.cols());
    mb.noise[map_index].chip(noise_num,2) = in_tensor;
}

} // namespace mapmaking
