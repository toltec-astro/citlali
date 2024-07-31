#pragma once

#include <string>

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

class GaussianFilter {
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // fwhms for gaussian template
    std::map<std::string, double> template_fwhm_rad;

    // size of maps
    int n_rows, n_cols;

    // filter template
    Eigen::MatrixXd filter_template;

    // get config file
    template <typename config_t>
    void get_config(config_t &, std::vector<std::vector<std::string>> &, std::vector<std::vector<std::string>> &);

    // make a symmetric Gaussian to use as a template
    template<class MB>
    void make_gaussian_template(MB &mb, const double);

    template<class MB>
    void filter(MB &, const int);
};

// get config file
template <typename config_t>
void GaussianFilter::get_config(config_t &config, std::vector<std::vector<std::string>> &missing_keys,
                              std::vector<std::vector<std::string>> &invalid_keys) {

    // for array names
    engine_utils::toltecIO toltec_io;

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

template<class MB>
void GaussianFilter::make_gaussian_template(MB &mb, const double template_fwhm_rad) {
    // distance from tangent point
    Eigen::MatrixXd dist(n_rows,n_cols);

    // calculate distance
    for (Eigen::Index i=0; i<n_rows; ++i) {
        for (Eigen::Index j=0; j<n_cols; ++j) {
            dist(i,j) = sqrt(pow(mb.rows_tan_vec(i)+0.5*mb.pixel_size_rad,2) + pow(mb.cols_tan_vec(j)+0.5*mb.pixel_size_rad,2));
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
void WienerFilter::run_gaussian_filter(MB &mb, const int map_index) {
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

    in.real() = filter_template;
    in.imag().setZero();

    // fft(f(x))
    out = engine_utils::fft2<engine_utils::forward>(in, pf, a, b);

    Eigen::MatrixXcd fft_filter = out;

    in.real() = filtered_map;
    in.imag().setZero();

    out = engine_utils::fft2<engine_utils::forward>(in, pf, a, b);

    // fft(f(x)) x fft(Q) (convolution)
    in.real() = out.real().array() * fft_filter.real().array() + out.imag().array() * fft_filter.imag().array();
    in.imag() = -out.imag().array() * fft_filter.real().array() + out.real().array() * fft_filter.imag().array();

    out = engine_utils::fft2<engine_utils::inverse>(in, pr, a, b);

    nume = out.real();

    // free fftw vectors
    fftw_free(a);
    fftw_free(b);

    // destroy fftw plans
    fftw_destroy_plan(pf);
    fftw_destroy_plan(pr);
}
