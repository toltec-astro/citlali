#pragma once

#include <Eigen/Core>

#include <string>
#include <Eigen/Core>

namespace mapmaking {

class WienerFilter {
public:
    std::string template_type;
    bool run_highpass, run_lowpass;
    bool use_uniform_weight, normalize_error;

    double diff_rows, diff_cols;
    Eigen::Index n_rows, n_cols;

    Eigen::MatrixXd rr, vvq, denom, nume;
    Eigen::MatrixXd filtered_map, filter_template;

    double psd_limit = 1e-4;

    //void make_gaussian_template() {
        //filter_template.setZero(n_rows, n_cols);
    //}

    void make_gaussian_template();
    void make_kernel_template();

    template <class map_buffer_t>
    void make_template(map_buffer_t &);

    template <typename Derived>
    void calc_rr(Eigen::DenseBase<Derived> &);

    template <typename Derived>
    void calc_vvq(Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &);

    void calc_numerator();
    void calc_denominator();

    template <class map_buffer_t>
    void run(map_buffer_t &mb) {
        // loop through maps
        for (Eigen::Index i=0; i<mb.n_maps; i++) {
            use_uniform_weight = true;
            calc_rr(mb[i].weight);
            calc_vvq(mb.noise_psds[i], mb.noise_psd_freqs[i]);

            // check if noise maps are requested
            if (!mb.noise.empty()) {
                // loop through noise maps
                for (Eigen::Index j=0; j<mb.n_noise; j++) {
                }
            }
        }
    }
};

template <typename Derived>
void WienerFilter::calc_rr(Eigen::DenseBase<Derived> &weight) {
    if (use_uniform_weight) {
        rr = Eigen::MatrixXd::Ones(n_rows, n_cols);
    }
    else {
        rr = sqrt(weight.array());
    }
}

template <class map_buffer_type>
void WienerFilter::make_template(map_buffer_type &mb) {
    // make sure new wiener filtered maps have even dimensions
    n_rows = 2*(mb.n_rows/2);
    n_cols = 2*(mb.n_cols/2);

         // calculate spacing for rows and cols
    diff_rows = abs(mb.rows_tan_vec(1) - mb.rows_tan_vec(0));
    diff_cols = abs(mb.cols_tan_vec(1) - mb.cols_tan_vec(0));

    if (run_highpass) {
        filter_template.setZero(n_rows, n_cols);
        filter_template(0,0) = 1;
    }

    else if (template_type=="gaussian") {
        make_gaussian_template();
    }

    else if (template_type=="kernel"){
        make_kernel_template();
    }
}

template <typename Derived>
void WienerFilter::calc_vvq(Eigen::DenseBase<Derived> &psd, Eigen::DenseBase<Derived> &psd_freq) {
    // psd size
    Eigen::Index n_psd = psd.size();
    // get max psd value
    Eigen::Index max_psd_indx;
    auto max_psd = psd.maxCoeff(&max_psd_indx);

    double psd_freq_break = 0;

    for (Eigen::Index i=0; i<n_psd; i++) {
        if (psd(i)/max_psd < psd_limit) {
            psd_freq_break = psd_freq(i);
            break;
        }
    }

    double psd_break;

    auto count = (psd_freq.array() > 0.8*psd_freq_break).count();

    if (count>0) {
        for (Eigen::Index i=0; i<n_psd; i++) {
            if (psd_freq_break > 0) {
                if (psd_freq(i) <= 0.8*psd_freq_break) {
                    psd_break = psd(i);
                }
                else {
                    psd(i) = psd_break;
                }
            }
        }
    }

    // flatten highpass response if present
    if (max_psd_indx > 0) {
        for (Eigen::Index i=0; i<max_psd_indx; i++) {
            psd(i) = max_psd_indx;
        }
    }

    double row_size = n_rows*diff_rows;
    double col_size = n_cols*diff_cols;
    double diffq_rows = 1./row_size;
    double diffq_cols = 1./col_size;

    Eigen::VectorXd psd_freq_rows = Eigen::VectorXd::LinSpaced(n_rows, diffq_rows*(-(n_rows-1)/2), diffq_rows*(n_rows-1-(n_rows-1)/2));
    Eigen::VectorXd psd_freq_cols = Eigen::VectorXd::LinSpaced(n_cols, diffq_cols*(-(n_rows-1)/2), diffq_cols*(n_rows-1-(n_cols-1)/2));

}



} // namespace mapmaking
