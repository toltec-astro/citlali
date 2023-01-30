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

    // kernel map filtering
    double bounding_box_pix, init_fwhm;

    // fwhms for gaussian tempalte
    std::map<std::string, double> gaussian_template_fwhm_rad;

    // size of maps
    int n_rows, n_cols;

    // size of pixel in each dimension
    double diff_rows, diff_cols;
    // lower limit to zero out denom values
    double denom_limit = 1.e-4;

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

    template<class MB, class CD>
    void make_symmetric_template(MB &mb, const int, CD &);

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

        // kernel template
        else {
            make_symmetric_template(mb, map_index, calib_data);
        }
    }

    template<class MB>
    void calc_rr(MB &mb, const int map_index) {
        if (uniform_weight) {
            rr = Eigen::MatrixXd::Ones(n_rows,n_cols);
        }
        else {
            rr = sqrt(mb.weight.at(map_index).array());
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
        SPDLOG_DEBUG("calculating vvq");
        calc_vvq(mb, map_index);
        SPDLOG_DEBUG("calculating denominator");
        calc_denominator();
        SPDLOG_DEBUG("calculating numerator");
        calc_numerator();
    }

    template<class MB>
    void filter_maps(MB &mb, const int map_index) {
        if (run_kernel) {
            SPDLOG_INFO("filtering kernel");
            filtered_map = mb.kernel.at(map_index);
            uniform_weight = true;
            run_filter(mb, map_index);

            for (Eigen::Index i=0; i<n_cols; i++) {
                for (Eigen::Index j=0; j<n_rows; j++) {
                    if (denom(j,i) != 0.0) {
                        mb.kernel.at(map_index)(j,i)=nume(j,i)/denom(j,i);
                    }
                    else {
                        mb.kernel.at(map_index)(j,i)= 0.0;
                    }
                }
            }
        }

        SPDLOG_INFO("filtering signal");
        filtered_map = mb.signal.at(map_index);
        uniform_weight = false;
        run_filter(mb, map_index);

        for (Eigen::Index i=0; i<n_cols; i++) {
            for (Eigen::Index j=0; j<n_rows; j ++) {
                if (denom(j,i) != 0.0) {
                    mb.signal.at(map_index)(j,i) = nume(j,i)/denom(j,i);
                }
                else {
                    mb.signal.at(map_index)(j,i)= 0.0;
                }
            }
        }
        // weight map is the denominator
        mb.weight.at(map_index) = denom;
    }

    template<class MB>
    void filter_noise(MB &mb, const int map_index, const int noise_num) {
        Eigen::Tensor<double,2> out = mb.noise.at(map_index).chip(noise_num,2);
        filtered_map = Eigen::Map<Eigen::MatrixXd>(out.data(),out.dimension(0),out.dimension(1));
        calc_numerator();

        //Eigen::MatrixXd ratio = (denom.array() == 0).select(0, nume.array() / denom.array());
        Eigen::MatrixXd ratio(n_rows,n_cols);

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

        Eigen::TensorMap<Eigen::Tensor<double, 2>> in_tensor(ratio.data(), ratio.rows(), ratio.cols());
        mb.noise.at(map_index).chip(noise_num,2) = in_tensor;
    }
};

template<class MB>
void WienerFilter::make_gaussian_template(MB &mb, const double gaussian_template_fwhm_rad) {

    Eigen::VectorXd rgcut = mb.rows_tan_vec;
    Eigen::VectorXd cgcut = mb.cols_tan_vec;

    Eigen::MatrixXd dist(n_rows,n_cols);
    /*dist = (xgcut.replicate(1,n_cols).array().pow(2.0).matrix()
            + ygcut.replicate(n_rows, 1).array().pow(2.0).matrix())
               .array()
               .sqrt();*/

    for (Eigen::Index i=0; i<n_cols; i++) {
        for (Eigen::Index j=0; j<n_rows; j++) {
            dist(j,i) = sqrt(pow(rgcut(j),2) + pow(cgcut(i),2));
        }
    }

    Eigen::Index rcind, ccind;

    double min_dist = dist.minCoeff(&rcind, &ccind);
    double sigma = gaussian_template_fwhm_rad*FWHM_TO_STD;

    std::vector<Eigen::Index> shift_indices = {-rcind, -ccind};

    filter_template = exp(-0.5 * pow(dist.array() / sigma, 2.));
    filter_template = engine_utils::shift_2D(filter_template, shift_indices);
}

template<class MB, class CD>
void WienerFilter::make_symmetric_template(MB &mb, const int map_index, CD &calib_data) {
    // collect what we need
    Eigen::VectorXd rgcut = mb.rows_tan_vec;
    Eigen::VectorXd cgcut = mb.cols_tan_vec;
    Eigen::MatrixXd temp = mb.kernel[map_index];

    // set n_params for fit
    Eigen::Index n_params = 6;

    auto [det_params, det_perror, good_fit] =
        map_fitter.fit_to_gaussian<engine_utils::mapFitter::pointing>(mb.signal[map_index], mb.weight[map_index], init_fwhm);

    det_params(1) = mb.pixel_size_rad*(det_params(1) - (n_cols)/2);
    det_params(2) = mb.pixel_size_rad*(det_params(2) - (n_rows)/2);

    Eigen::Index shift_row = -std::round(det_params(2)/diff_rows);
    Eigen::Index shift_col = -std::round(det_params(1)/diff_cols);

    std::vector<Eigen::Index> shift_indices = {shift_row,shift_col};
    temp = engine_utils::shift_2D(temp, shift_indices);

    Eigen::MatrixXd dist(n_rows,n_cols);
    for (Eigen::Index i=0; i<n_cols; i++) {
        for(Eigen::Index j=0; j<n_rows; j++) {
            dist(j,i) = sqrt(pow(rgcut(j),2)+pow(cgcut(i),2));
        }
    }

    Eigen::Index rcind, ccind;
    auto min_dist = dist.minCoeff(&rcind,&ccind);

    // create new bins based on diff_rows
    int n_bins = rgcut(n_rows-1)/diff_rows;
    Eigen::VectorXd bin_low(n_bins);
    for(Eigen::Index i=0;i<n_bins;i++) {
        bin_low(i) = (double) (i*diff_rows);
    }

    Eigen::VectorXd kone(n_bins-1);
    kone.setZero();
    Eigen::VectorXd done(n_bins-1);
    done.setZero();
    for (Eigen::Index i=0; i<n_bins-1; i++) {
        int c=0;
        for (Eigen::Index j=0;j<n_cols;j++) {
            for (int k=0; k<n_rows; k++) {
                if (dist(k,j) >= bin_low(i) && dist(k,j) < bin_low(i+1)){
                    c++;
                    kone(i) += temp(k,j);
                    done(i) += dist(k,j);
                }
            }
        }
        kone(i) /= c;
        done(i) /= c;
    }

    // now spline interpolate to generate new template array
    filter_template.resize(n_rows,n_cols);

    // create spline function
    engine_utils::SplineFunction s(done, kone);

    for (Eigen::Index i=0; i<n_cols; i++) {
        for (Eigen::Index j=0; j<n_rows; j++) {
            Eigen::Index tj = (j-rcind)%n_rows;
            Eigen::Index ti = (i-ccind)%n_cols;
            Eigen::Index shiftj = (tj < 0) ? n_rows+tj : tj;
            Eigen::Index shifti = (ti < 0) ? n_cols+ti : ti;

            if (dist(j,i) <= s.x_max && dist(j,i) >= s.x_min) {
                filter_template(shiftj,shifti) = s(dist(j,i));
            }
            else if (dist(j,i) > s.x_max) {
                filter_template(shiftj,shifti) = kone(kone.size()-1);
            }
            else if (dist(j,i) < s.x_min) {
                filter_template(shiftj,shifti) = kone(0);
            }
        }
    }
}

template <class MB>
void WienerFilter::calc_vvq(MB &mb, const int map_index) {
    // psd and psd freq vectors
    Eigen::VectorXd psd = mb.psds[map_index];
    Eigen::VectorXd psd_freq = mb.psd_freqs[map_index];

    // size of psd and psd freq vectors
    Eigen::Index n_psd = psd.size();

    // modify the psd array to take out lowpassing and highpassing
    double max_psd = -1.;
    int max_psd_index = 0;
    double psd_freq_break = 0.;
    double psd_break = 0.;
    for (Eigen::Index i=0; i<n_psd; i++) {
        if (psd(i) > max_psd) {
            max_psd = psd(i);
            max_psd_index = i;
        }
    }
    for (Eigen::Index i=0; i<n_psd; i++) {
        if (psd(i)/max_psd < 1.e-4){
            psd_freq_break = psd_freq(i);
            break;
        }
    }
    // flatten the response above the lowpass break
    int count = 0;
    for (Eigen::Index i=0; i<n_psd; i++) {
        if (psd_freq(i) <= 0.8*psd_freq_break) {
            count++;
        }
    }
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
       for (Eigen::Index i=0; i<max_psd_index; i++) {
           psd(i) = max_psd;
       }
    }

    // modify the psd array to take out lowpassing+highpassing
    /*Eigen::Index max_psd_index;
    double qf_break = 0.;
    double psd_break = 0.;

    // get max value and index of hp
    auto max_psd = hp.maxCoeff(&max_psd_index);

    if (max_psd < -1) {
        max_psd = -1;
        max_psd_index = 0;
    }

    // find the frequency where hp falls to 1e-4 its max value
    for (Eigen::Index i=0; i<n_psd; i++)
        if (hp(i) / max_psd < 1.e-4) {
            qf_break = qf(i);
            break;
        }
    // get the total number of points where the frequency is less than 0.8*qf_break
    auto count = (qf.array() <= 0.8 * qf_break).count();

    // set the region of hp corresponding to the frequency range above 0.8*qf_break to
    // the value of psd at 0.8*qf_break (i.e. flatten it).  We can do this as a vector
    // since the frequencies are monotonically increasing.
    if (count > 0) {
        if (qf_break > 0) {
            Eigen::Index start = 0.8 * qf_break;
            Eigen::Index size = hp.size() - 0.8 * qf_break;
            psd.segment(start, size).setConstant(hp(start));
        }
    }
    */

    // do the same for the region below the maximum hp
    //psd.head(max_psd_index).setConstant(max_psd);

    // set up q-space
    double row_size = n_rows * diff_rows;
    double col_size = n_cols * diff_cols;
    double diffqr = 1. / row_size;
    double diffqc = 1. / col_size;

    Eigen::MatrixXd qmap(n_rows,n_cols);

    // shift qr
    Eigen::VectorXd q_row = Eigen::VectorXd::LinSpaced(n_rows, -(n_rows - 1) / 2, (n_rows - 1) / 2 + 1) * diffqr;
    // shift qc
    Eigen::VectorXd q_col = Eigen::VectorXd::LinSpaced(n_cols, -(n_cols - 1) / 2, (n_cols - 1) / 2 + 1) * diffqc;

    std::vector<Eigen::Index> shift_1 = {-(n_rows-1)/2};
    engine_utils::shift_1D(q_row, shift_1);

    std::vector<Eigen::Index> shift_2 = {-(n_cols-1)/2};
    engine_utils::shift_1D(q_col, shift_2);

    // make qmap by replicating qc and qr ncols and n_rowsows times respectively.  Faster than a for loop for expected
    // map dimensions
    //qmap = (qr.replicate(1,n_cols).array().pow(2.0).matrix() + qc.replicate(n_rows, 1).array().pow(2.0).matrix()).array().sqrt();

    for (Eigen::Index i=0; i<n_cols; i++) {
        for (Eigen::Index j=0; j<n_rows; j++) {
            qmap(j,i) = sqrt(pow(q_row(j),2)+pow(q_col(i),2));
        }
    }

    Eigen::MatrixXd psd_q;

    if (run_lowpass) {
        psd_q.setOnes();
    }
    else {
        psd_q.setZero(n_rows,n_cols);

        Eigen::Matrix<Eigen::Index, 1, 1> n_psd_matrix;
        n_psd_matrix << psd.size();

        // interpolate onto psd_q
        Eigen::Index interp_pts = 1;
        for (Eigen::Index i=0; i<n_cols; i++) {
            for (Eigen::Index j=0; j<n_rows; j++) {
                if ((qmap(j, i) <= psd_freq(psd_freq.size() - 1)) && (qmap(j, i) >= psd_freq(0))) {
                    mlinterp::interp<mlinterp::rnatord>(n_psd_matrix.data(), interp_pts,
                                     psd.data(), psd_q.data() + n_rows * i + j,
                                     psd_freq.data(), qmap.data() + n_rows * i + j);
                }
                else if (qmap(j, i) > psd_freq(psd_freq.size() - 1)) {
                    psd_q(j, i) = psd(psd.size() - 1);
                }
                else if (qmap(j, i) < psd_freq(0)) {
                    psd_q(j, i) = psd(0);
                }
            }
        }

        // find the minimum value of psd
        auto psd_min = psd.minCoeff();

        // set all the points in psd_q smaller than lowval to lowval
        //(psd_q.array() < lowval).select(lowval, psd_q);

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
    nume.setZero(n_rows,n_cols);
    // normalization for fft
    double fft_norm = 1. / n_rows / n_cols;

    Eigen::MatrixXcd in(n_rows,n_cols);
    Eigen::MatrixXcd out(n_rows,n_cols);

    //Eigen::Map<Eigen::VectorXd> rr_vec(rr.data(),rr.rows(),rr.cols());
    //Eigen::Map<Eigen::VectorXd> filtered_map_vec(filtered_map.data(),filtered_map.rows(),filtered_map.cols());

    in.real() = rr.array() * filtered_map.array();
    in.imag().setZero();

    out = engine_utils::fft<engine_utils::forward>(in, parallel_policy);
    out = out * fft_norm;

    in.real() = out.real().array() / vvq.array();
    in.imag() = out.imag().array() / vvq.array();

    out = engine_utils::fft<engine_utils::inverse>(in, parallel_policy);

    in.real() = out.real().array() * rr.array();
    in.imag().setZero();

    out = engine_utils::fft<engine_utils::forward>(in, parallel_policy);
    out = out * fft_norm;

    // copy of out
    Eigen::MatrixXcd qqq = out;

    in.real() = filter_template;
    in.imag().setZero();

    out = engine_utils::fft<engine_utils::forward>(in, parallel_policy);
    out = out * fft_norm;

    in.real() = out.real().array() * qqq.real().array() + out.imag().array() * qqq.imag().array();
    in.imag() = -out.imag().array() * qqq.real().array() + out.real().array() * qqq.imag().array();

    out = engine_utils::fft<engine_utils::inverse>(in, parallel_policy);

    // populate numerator
    nume = out.real();
}

void WienerFilter::calc_denominator() {
    double fft_norm = 1. / n_rows / n_cols;
    denom.setZero(n_rows,n_cols);
    //Eigen::VectorXcd in(n_rows * nc);
    //Eigen::VectorXcd out(n_rows * nc);

    Eigen::MatrixXcd in(n_rows,n_cols);
    Eigen::MatrixXcd out(n_rows,n_cols);

    if (uniform_weight) {
        in.real() = filter_template;
        in.imag().setZero();

        out = engine_utils::fft<engine_utils::forward>(in, parallel_policy);
        out = out * fft_norm;

       // auto d = ((out.real().array() * out.real().array() + out.imag().array() * out.imag().array()) / vvq.array()).sum();

        double d=0;
        for (Eigen::Index i=0; i<n_cols; i++) {
            for (Eigen::Index j=0; j<n_rows; j++) {
                d += (out.real()(j,i)*out.real()(j,i) + out.imag()(j,i)*out.imag()(j,i))/vvq(j,i);
            }
        }
        denom.setConstant(d);
    }

    else {
        //Eigen::Map<Eigen::VectorXd> vvq_vec(vvq.data(), vvq.rows(),vvq.cols());
        //in.real() = pow(vvq_vec.array(),-1.);
        in.real() = pow(vvq.array(), -1);
        in.imag().setZero();

        out = engine_utils::fft<engine_utils::inverse>(in, parallel_policy);

        Eigen::VectorXd zz2d(n_rows * n_cols);
        //Eigen::Map<Eigen::VectorXd> out_vec(out.real().data(), n_rows*nc);

        for(Eigen::Index i=0;i<n_cols;i++)
            for(Eigen::Index j=0;j<n_rows;j++){
              int ii = n_rows*i+j;
              zz2d(ii) = (out.real()(j,i));
            }

        //zz2d = abs(out_vec.array());

        // sort
        Eigen::VectorXd ss_ord = zz2d;
        auto sorted = engine_utils::sorter(ss_ord);

        for (Eigen::Index i=0; i<n_cols; i++) {
            for (Eigen::Index j=0; j<n_rows; j++) {
                int ii = n_rows*i+j;
                zz2d(ii) = (out.real()(j,i));
            }
        }

        // number of iterations for convergence
        n_loops = n_rows * n_cols / 100;

        // flag for convergence
        bool done = false;

        tula::logging::progressbar pb0(
            [](const auto &msg) { SPDLOG_INFO("{}", msg); }, 60,
            "wiener filter denominator calculation progress ");

        for (int k=0; k<n_cols; k++) {
            for (int l=0; l<n_rows; l++) {
                if (!done) {
                    Eigen::MatrixXcd in2(n_rows,n_cols);
                    Eigen::MatrixXcd out2(n_rows,n_cols);

                    /* May need to flip directions due to order of matrix storage order */
                    int kk = n_rows * k + l;
                    if (kk >= n_loops) {
                        continue;
                    }

                    auto shift_index = std::get<1>(sorted[n_rows * n_cols - kk - 1]);

                    // changed n_rows
                    double r_shift_n = shift_index / n_rows;
                    double c_shift_n = shift_index % n_rows;

                    Eigen::Index shift_1 = -r_shift_n;
                    Eigen::Index shift_2 = -c_shift_n;

                    std::vector<Eigen::Index> shift_indices = {shift_1, shift_2};

                    Eigen::MatrixXd in_prod = filter_template.array() * engine_utils::shift_2D(filter_template, shift_indices).array();

                    /*in2.real() = Eigen::Map<Eigen::VectorXd>(
                        (in_prod).data(),
                        n_rows * nc);
                    in2.imag().setZero();
                    */

                    in2.real() = in_prod;
                    in2.imag().setZero();

                    out2 = engine_utils::fft<engine_utils::forward>(in2, parallel_policy);
                    out2 = out2 * fft_norm;

                    Eigen::MatrixXcd ffdq(n_rows,n_cols);

                    ffdq.real() = out2.real();
                    ffdq.imag() = out2.imag();

                    in_prod = rr.array() * engine_utils::shift_2D(rr, shift_indices).array();

                    //in2.real() = Eigen::Map<Eigen::VectorXd>(in_prod.data(), n_rows * nc);
                    //in2.imag().setZero();

                    in2.real() = in_prod;
                    in2.imag().setZero();

                    out2 = engine_utils::fft<engine_utils::forward>(in2, parallel_policy);
                    out2 = out2 * fft_norm;

                    in2.real() = ffdq.real().array() * out2.real().array() + ffdq.imag().array() * out2.imag().array();
                    in2.imag() = -ffdq.imag().array() * out2.real().array() + ffdq.real().array() * out2.imag().array();

                    out2 = engine_utils::fft<engine_utils::inverse>(in2, parallel_policy);

                    Eigen::MatrixXd updater = zz2d(shift_index) * out2.real() * fft_norm;

                    denom = denom + updater;

                    pb0.count(n_loops, n_loops / 100);

                    if ((kk % 100) == 1) {
                        double max_ratio = -1;
                        double maxdenom = denom.maxCoeff();

                        for (Eigen::Index i=0; i<n_rows; i++) {
                            for (Eigen::Index j=0; j<n_cols; j++) {
                                if (denom(i, j) > 0.01 * maxdenom) {
                                    if (abs(updater(i, j) / denom(i, j)) > max_ratio)
                                        max_ratio = abs(updater(i, j) / denom(i, j));
                                }
                            }
                        }

                        if (((kk >= max_loops) && (max_ratio < 0.0002)) || max_ratio < 1e-10) {
                            SPDLOG_INFO("done.  max_ratio={} {} iterations", max_ratio, kk);
                            done = true;
                        }
                        else {
                            //SPDLOG_INFO("completed iteration {} of {}. max_ratio={}",
                                        //kk, n_loops, max_ratio);
                        }
                    }
                }
            }
        }

        SPDLOG_DEBUG("zeroing out any values < {} in denominator", denom_limit);
        //(denom.array() < denom_limit).select(0, denom);
        for (Eigen::Index i=0;i<n_rows;i++) {
            for (Eigen::Index j=0;j<n_cols;j++) {
                if (denom(i,j) < denom_limit) {
                    denom(i,j) = 0;
                }
            }
        }
        SPDLOG_DEBUG("min denom {}",denom.minCoeff());
    }
}

} // namespace mapmaking
