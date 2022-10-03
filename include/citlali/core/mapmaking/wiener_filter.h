#pragma once

#include <string>
#include <Eigen/Core>

namespace mapmaking {

class WienerFilter {
public:
    std::string template_type;
    bool run_highpass, run_lowpass;
    bool uniform_weight, normalize_error;

    double diff_rows, diff_cols;
    Eigen::Index n_rows, n_cols;

    Eigen::MatrixXd rr, vvq, denom, nume;
    Eigen::MatrixXd data, filter_template;

    void make_gaussian_template() {
        filter_template.setZero(n_rows, n_cols);
    }
    void make_kernel_template();

    template <class map_buffer_type>
    void make_template(map_buffer_type &mb) {
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

    /*void calc_rr() {
        if (uniform_weight) {
            rr = Eigen::MatrixXd::Ones(n_rows, n_cols);
        }
        else {
            rr = sqrt(weight.array());
        }
    }*/

    void calc_vvq();
    void calc_numerator();
    void calc_denominator();

    void filter_obs() {
        //calc_rr();
        calc_vvq();
        calc_denominator();
        calc_numerator();
    }

    void filter_noise();

    void run() {
        filter_obs();

        filter_noise();
    }
};

} // namespace mapmaking
