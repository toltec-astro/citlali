#pragma once

#include <string>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/Splines>

#include <tula/algorithm/mlinterp/mlinterp.hpp>
#include <tula/logging.h>

#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/utils.h>

#include <citlali/core/utils/gauss_models.h>
#include <citlali/core/utils/fitting.h>

namespace mapmaking {

class WienerFilter {
public:
    // filter template
    std::string template_type;
    // normalize filtered map errors
    bool normalize_error;
    // uniform weighting
    bool uniform_weight;
    // lowpass only
    bool run_lowpass;

    // if kernel is enabled
    bool run_kernel;
    // number of loops in denom calc
    int n_loops;
    // maximum number of loops for denom calc
    int max_loops = 500;
    // lower limit to zero out denom values
    double denom_limit = 1.e-4;

    // kernel map filtering
    double init_fwhm;

    // fwhms for gaussian tempalte
    std::map<std::string, double> gaussian_template_fwhm_rad;

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

    // declare fitter class
    engine_utils::mapFitter map_fitter;

    template<class MB>
    void make_gaussian_template(MB &mb, const double);

    template<class MB>
    void make_airy_template(MB &mb, const double);

    template<class MB, class CD>
    void make_kernel_template(MB &mb, const int, CD &);

    template<class MB, class CD>
    void make_template(MB &mb, CD &calib_data, const double gaussian_template_fwhm_rad, const int map_index) {
        // make sure filtered maps have even dimensions
        n_rows = 2*(mb.n_rows/2);
        n_cols = 2*(mb.n_cols/2);

        // x and y spacing should be equal
        diff_rows = abs(mb.rows_tan_vec(1) - mb.rows_tan_vec(0));
        diff_cols = abs(mb.cols_tan_vec(1) - mb.cols_tan_vec(0));

        // highpass template
        if (template_type=="highpass") {
            SPDLOG_INFO("creating template with highpass only");
            filter_template.setZero(n_rows,n_cols);
            filter_template(0,0) = 1;
        }

        // gaussian template
        else if (template_type=="gaussian") {
            SPDLOG_INFO("creating gaussian template");
            make_gaussian_template(mb, gaussian_template_fwhm_rad);
        }

        // airy template
        else if (template_type=="airy") {
            SPDLOG_INFO("creating airy template");
            make_airy_template(mb, gaussian_template_fwhm_rad);
        }

        // kernel template
        else {
            make_kernel_template(mb, map_index, calib_data);
        }
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

    template<class MB>
    void run_filter(MB &mb, const int map_index) {
        SPDLOG_DEBUG("calculating rr");
        calc_rr(mb, map_index);
        SPDLOG_DEBUG("rr {}", rr);

        SPDLOG_DEBUG("calculating vvq");
        calc_vvq(mb, map_index);
        SPDLOG_DEBUG("vvq {}", vvq);

        SPDLOG_DEBUG("calculating denominator");
        calc_denominator();
        SPDLOG_DEBUG("denominator {}", denom);

        SPDLOG_DEBUG("calculating numerator");
        calc_numerator();
        SPDLOG_DEBUG("numerator {}", nume);
    }

    template<class MB>
    void filter_maps(MB &mb, const int map_index) {
        // filter kernel
        if (run_kernel) {
            SPDLOG_INFO("filtering kernel");
            filtered_map = mb.kernel[map_index];
            uniform_weight = true;
            run_filter(mb, map_index);

            // divide by filtered weight
            for (Eigen::Index i=0; i<n_cols; i++) {
                for (Eigen::Index j=0; j<n_rows; j++) {
                    if (denom(j,i) != 0.0) {
                        mb.kernel[map_index](j,i)=nume(j,i)/denom(j,i);
                    }
                    else {
                        mb.kernel[map_index](j,i)= 0.0;
                    }
                }
            }

            SPDLOG_INFO("done");
        }

        SPDLOG_INFO("filtering signal");
        // filter signal
        filtered_map = mb.signal[map_index];
        uniform_weight = false;
        // run all filter steps
        run_filter(mb, map_index);

        // divide by filtered weight
        for (Eigen::Index i=0; i<n_cols; i++) {
            for (Eigen::Index j=0; j<n_rows; j ++) {
                if (denom(j,i) != 0.0) {
                    mb.signal[map_index](j,i) = nume(j,i)/denom(j,i);
                }
                else {
                    mb.signal[map_index](j,i)= 0.0;
                }
            }
        }

        SPDLOG_INFO("done");

        // weight map is the denominator
        mb.weight[map_index] = denom;
    }

    template<class MB>
    void filter_noise(MB &mb, const int map_index, const int noise_num) {
        Eigen::Tensor<double,2> out = mb.noise[map_index].chip(noise_num,2);
        filtered_map = Eigen::Map<Eigen::MatrixXd>(out.data(),out.dimension(0),out.dimension(1));
        calc_numerator();

        Eigen::MatrixXd ratio(n_rows,n_cols);

        // divide by filtered weight
        for (Eigen::Index i=0; i<n_cols; i++) {
            for (Eigen::Index j=0; j<n_rows; j++) {
                if (denom(j,i) != 0.0) {
                    ratio(j,i) = nume(j,i)/denom(j,i);
                }
                else {
                    ratio(j,i)= 0.0;
                }
            }
        }

        // map to tensor
        Eigen::TensorMap<Eigen::Tensor<double, 2>> in_tensor(ratio.data(), ratio.rows(), ratio.cols());
        mb.noise[map_index].chip(noise_num,2) = in_tensor;
    }
};

template<class MB>
void WienerFilter::make_gaussian_template(MB &mb, const double gaussian_template_fwhm_rad) {
    // distance from tangent point
    Eigen::MatrixXd dist(n_rows,n_cols);

    // calculate distance
    for (Eigen::Index i=0; i<n_cols; i++) {
        for (Eigen::Index j=0; j<n_rows; j++) {
            dist(j,i) = sqrt(pow(mb.rows_tan_vec(j),2) + pow(mb.cols_tan_vec(i),2));
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
            dist(j,i) = sqrt(pow(mb.rows_tan_vec(j),2) + pow(mb.cols_tan_vec(i),2));
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
    // collect what we need
    Eigen::MatrixXd temp_kernel = mb.kernel[map_index];

    // carry out fit to kernel
    auto [map_params, map_perror, good_fit] =
        map_fitter.fit_to_gaussian<engine_utils::mapFitter::pointing>(mb.kernel[map_index], mb.weight[map_index], init_fwhm);

    if (!good_fit) {
        SPDLOG_ERROR("fit to kernel map failed. try setting a small fitting_region_arcsec value.");
        std::exit(EXIT_FAILURE);
    }

    // rescale parameters to on-sky units
    map_params(1) = mb.pixel_size_rad*(map_params(1) - (n_cols)/2);
    map_params(2) = mb.pixel_size_rad*(map_params(2) - (n_rows)/2);

    Eigen::Index shift_row = -std::round(map_params(2)/diff_rows);
    Eigen::Index shift_col = -std::round(map_params(1)/diff_cols);

    std::vector<Eigen::Index> shift_indices = {shift_row,shift_col};
    temp_kernel = engine_utils::shift_2D(temp_kernel, shift_indices);

    // calculate distance
    Eigen::MatrixXd dist(n_rows,n_cols);
    for (Eigen::Index i=0; i<n_cols; i++) {
        for (Eigen::Index j=0; j<n_rows; j++) {
            dist(j,i) = sqrt(pow(mb.rows_tan_vec(j),2)+pow(mb.cols_tan_vec(i),2));
        }
    }

    // pixel closet to tangent point
    Eigen::Index row_index, col_index;
    auto min_dist = dist.minCoeff(&row_index,&col_index);

    // create new bins based on diff_rows
    int n_bins = mb.rows_tan_vec(n_rows-1)/diff_rows;
    Eigen::VectorXd bin_low = Eigen::VectorXd::LinSpaced(n_bins,0,n_bins-1)*diff_rows;

    Eigen::VectorXd kernel_interp(n_bins-1);
    kernel_interp.setZero();
    Eigen::VectorXd dist_interp(n_bins-1);
    dist_interp.setZero();

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
        kernel_interp(i) /= c;
        dist_interp(i) /= c;
    }

    // now spline interpolate to generate new template array
    filter_template.resize(n_rows,n_cols);

    // create spline function
    engine_utils::SplineFunction s(dist_interp, kernel_interp);

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
                filter_template(shiftj,shifti) = kernel_interp(kernel_interp.size()-1);
            }
            // if below x limit
            else if (dist(j,i) < s.x_min) {
                filter_template(shiftj,shifti) = kernel_interp(0);
            }
        }
    }
}

template <class MB>
void WienerFilter::calc_vvq(MB &mb, const int map_index) {
    // resize psd_q
    Eigen::MatrixXd psd_q(n_rows,n_cols);

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

    for (Eigen::Index i=0; i<n_psd; i++) {
        if (psd(i)/max_psd < 1.e-4){
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

    // set up q-space
    double row_size = n_rows * diff_rows;
    double col_size = n_cols * diff_cols;
    double diff_qr = 1. / row_size;
    double diff_qc = 1. / col_size;

    Eigen::MatrixXd qmap(n_rows,n_cols);

    // shift q_row
    Eigen::VectorXd q_row = Eigen::VectorXd::LinSpaced(n_rows, -(n_rows - 1) / 2, (n_rows - 1) / 2 + 1) * diff_qr;
    // shift q_col
    Eigen::VectorXd q_col = Eigen::VectorXd::LinSpaced(n_cols, -(n_cols - 1) / 2, (n_cols - 1) / 2 + 1) * diff_qc;

    std::vector<Eigen::Index> shift_1 = {-(n_rows-1)/2};
    engine_utils::shift_1D(q_row, shift_1);

    std::vector<Eigen::Index> shift_2 = {-(n_cols-1)/2};
    engine_utils::shift_1D(q_col, shift_2);

    for (Eigen::Index i=0; i<n_cols; i++) {
        for (Eigen::Index j=0; j<n_rows; j++) {
            qmap(j,i) = sqrt(pow(q_row(j),2)+pow(q_col(i),2));
        }
    }

    // set constant if lowpass only
    if (run_lowpass) {
        psd_q.setOnes();
    }
    else {
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

        for (Eigen::Index i=0; i<n_cols; i++) {
            for (Eigen::Index j=0; j<n_rows; j++) {
                if (psd_q(j,i) < psd_min) {
                    psd_q(j,i) = psd_min;
                }
            }
        }
    }

    // normalize the power spectrum psd_q and place into vvq
    vvq = psd_q/psd_q.sum();
}

void WienerFilter::calc_numerator() {
    // set up fftw
    fftw_complex *a;
    fftw_complex *b;
    fftw_plan pf, pr;

    a = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_rows*n_cols);
    b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_rows*n_cols);

    pf = fftw_plan_dft_2d(n_rows, n_cols, a, b, FFTW_FORWARD, FFTW_ESTIMATE);
    pr = fftw_plan_dft_2d(n_rows, n_cols, a, b, FFTW_BACKWARD, FFTW_ESTIMATE);

    // set up inputs and outputs
    Eigen::MatrixXcd in(n_rows,n_cols);
    Eigen::MatrixXcd out(n_rows,n_cols);

    in.real() = rr.array() * filtered_map.array();
    in.imag().setZero();

    //out = engine_utils::fft<engine_utils::forward>(in, parallel_policy);
    out = engine_utils::fft2<engine_utils::forward>(in, pf, a, b);

    in.real() = out.real().array() / vvq.array();
    in.imag() = out.imag().array() / vvq.array();

    //out = engine_utils::fft<engine_utils::inverse>(in, parallel_policy);
    out = engine_utils::fft2<engine_utils::inverse>(in, pr, a, b);

    in.real() = out.real().array() * rr.array();
    in.imag().setZero();

    //out = engine_utils::fft<engine_utils::forward>(in, parallel_policy);
    out = engine_utils::fft2<engine_utils::forward>(in, pf, a, b);

    // copy of out
    Eigen::MatrixXcd qqq = out;

    in.real() = filter_template;
    in.imag().setZero();

    //out = engine_utils::fft<engine_utils::forward>(in, parallel_policy);
    out = engine_utils::fft2<engine_utils::forward>(in, pf, a, b);

    in.real() = out.real().array() * qqq.real().array() + out.imag().array() * qqq.imag().array();
    in.imag() = -out.imag().array() * qqq.real().array() + out.real().array() * qqq.imag().array();

    //out = engine_utils::fft<engine_utils::inverse>(in, parallel_policy);
    out = engine_utils::fft2<engine_utils::inverse>(in, pr, a, b);

    // populate numerator
    nume = out.real();

    // destroy fftw plans
    fftw_destroy_plan(pf);
    fftw_destroy_plan(pr);
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
    denom.resize(n_rows,n_cols);

    // inputs and outputs to ffts
    Eigen::MatrixXcd in(n_rows,n_cols);
    Eigen::MatrixXcd out(n_rows,n_cols);

    // using uniform weights only
    if (uniform_weight) {
        in.real() = filter_template;
        in.imag().setZero();

        //out = engine_utils::fft<engine_utils::forward>(in, parallel_policy);
        out = engine_utils::fft2<engine_utils::forward>(in, pf, a, b);

        // set denominator
        denom.setConstant(((out.real().array() * out.real().array() + out.imag().array() * out.imag().array()) / vvq.array()).sum());

        // destroy fftw plans
        fftw_destroy_plan(pf);
        fftw_destroy_plan(pr);
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
        Eigen::VectorXd ss_ord = zz2d.array().abs();
        auto sorted = engine_utils::sorter(ss_ord);

        // number of iterations for convergence
        n_loops = n_rows * n_cols / 100;

        // flag for convergence
        bool done = false;

        tula::logging::progressbar pb(
            [](const auto &msg) { SPDLOG_INFO("{}", msg); }, 90,
            "calculating denom");

        Eigen::Index ii;
        // loop through cols and rows
        for (Eigen::Index k=0; k<n_cols; k++) {
        #pragma omp parallel for schedule (dynamic) ordered shared (sorted, n_loops, zz2d, k, done, pb) private (ii) default (none)
            for (Eigen::Index l=0; l<n_rows; l++) {
                #pragma omp flush (done)
                if (!done) {
                    fftw_complex *a,*b;
                    fftw_plan pf, pr;
                    int kk = n_rows*k + l;
                    if (kk >= n_loops) {
                        continue;
                    }
                    #pragma omp critical (wfFFTW)
                    {
                        a = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_rows*n_cols);
                        b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n_rows*n_cols);
                        pf = fftw_plan_dft_2d(n_rows, n_cols, a, b, FFTW_FORWARD, FFTW_ESTIMATE);
                        pr = fftw_plan_dft_2d(n_rows, n_cols, a, b, FFTW_BACKWARD, FFTW_ESTIMATE);
                    }
                    Eigen::MatrixXcd in(n_rows,n_cols);
                    Eigen::MatrixXcd out(n_rows,n_cols);

                    // current element in flattened 1d vector
                    //int kk = n_rows * k + l;
                    // get index in reverse order
                    auto shift_index = std::get<1>(sorted[n_rows * n_cols - kk - 1]);

                    double r_shift_n = shift_index / n_rows;
                    double c_shift_n = shift_index % n_rows;

                    Eigen::Index shift_1 = -r_shift_n;
                    Eigen::Index shift_2 = -c_shift_n;

                    std::vector<Eigen::Index> shift_indices = {shift_1, shift_2};

                    Eigen::MatrixXd in_prod = filter_template.array() * engine_utils::shift_2D(filter_template, shift_indices).array();

                    in.real() = in_prod;
                    in.imag().setZero();

                    //out = engine_utils::fft<engine_utils::forward>(in, parallel_policy);
                    out = engine_utils::fft2<engine_utils::forward>(in, pf, a, b);

                    Eigen::MatrixXcd ffdq = out;

                    in_prod = rr.array() * engine_utils::shift_2D(rr, shift_indices).array();

                    in.real() = in_prod;
                    in.imag().setZero();

                    //out = engine_utils::fft<engine_utils::forward>(in, parallel_policy);
                    out = engine_utils::fft2<engine_utils::forward>(in, pf, a, b);

                    in.real() = ffdq.real().array() * out.real().array() + ffdq.imag().array() * out.imag().array();
                    in.imag() = -ffdq.imag().array() * out.real().array() + ffdq.real().array() * out.imag().array();

                    //out = engine_utils::fft<engine_utils::inverse>(in, parallel_policy);
                    out = engine_utils::fft2<engine_utils::inverse>(in, pr, a, b);

                    #pragma omp ordered
                    {
                        Eigen::MatrixXd updater = zz2d(shift_index) * out.real()/n_rows/n_cols;

                        // update denominator
                        denom = denom + updater;

                        #pragma omp critical (wfFFTW)
                        {
                            fftw_free(a);
                            fftw_free(b);
                            fftw_destroy_plan(pf);
                            fftw_destroy_plan(pr);
                        }

                        // update progress bar
                        pb.count(n_loops, n_loops / 100);

                        // update status
                        if ((kk % 100) == 1) {
                            double max_ratio = -1;
                            double max_denom = denom.maxCoeff();

                            for (Eigen::Index i=0; i<n_rows; i++) {
                                for (Eigen::Index j=0; j<n_cols; j++) {
                                    if (denom(i, j) > 0.01 * max_denom) {
                                        if (abs(updater(i, j) / denom(i, j)) > max_ratio)
                                            max_ratio = abs(updater(i, j) / denom(i, j));
                                    }
                                }
                            }

                            // check if done
                            if (((kk >= max_loops) && (max_ratio < 0.0002)) || max_ratio < 1e-10) {
                                done = true;
                            #pragma omp flush(done)
                            }
                        }
                    }
                }
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
