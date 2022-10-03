#pragma once

#include <string>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <citlali/core/utils/utils.h>

namespace mapmaking {

// wcs information
struct WCS {
    // pixel size
    std::vector<double> cdelt;

    // map size in pixels
    std::vector<double> naxis;

    // reference pixels
    std::vector<double> crpix;

    // reference sky value
    std::vector<double> crval;

    // map unit
    std::vector<std::string> cunit;

    // coord type
    std::vector<std::string> ctype;
};

class ObsMapBuffer {
public:

    // wcs
    WCS wcs;
    // number of rows and columns
    Eigen::Index n_rows, n_cols;
    // number of noise maps
    Eigen::Index n_noise;
    // pixel size in radians
    double pixel_size_rad;
    // tangent plane pixel positions
    Eigen::VectorXd rows_tan_vec, cols_tan_vec;
    // signal map units
    std::string sig_unit;
    // maps
    std::vector<Eigen::MatrixXd> signal, weight, kernel, coverage;
    // noise maps (n_rows, n_cols, n_noise) of length nmaps
    std::vector<Eigen::Tensor<double,3>> noise;

    // coverage cut
    double cov_cut;

    void normalize_maps();
    void calc_map_psd(std::string);
    void calc_map_hist();
};

void ObsMapBuffer::normalize_maps() {
    // normalize maps
    for (Eigen::Index i=0; i<signal.size(); i++) {
        for (Eigen::Index j=0; j<n_rows; j++) {
            for (Eigen::Index k=0; k<n_cols; k++) {
                double sig_weight = weight[i](j,k);
                if (sig_weight != 0.) {
                    signal[i](j,k) = signal[i](j,k) / sig_weight;

                    if(!kernel.empty()) {
                        kernel[i](j,k) = kernel[i](j,k) / sig_weight;
                    }
                }
                else {
                    signal[i](j,k) = 0;

                    if(!kernel.empty()) {
                        kernel[i](j,k) = 0;
                    }
                }
            }
        }
    }

    // normalize noise maps
    for (Eigen::Index i=0; i<noise.size(); i++) {
        for (Eigen::Index j=0; j<signal.size(); j++) {
            for (Eigen::Index k=0; k<n_rows; k++) {
                for (Eigen::Index l=0; l<n_cols; l++) {
                    double sig_weight = weight.at(i)(j,k);
                    if (sig_weight != 0.) {
                        noise.at(i)(j,k,l) = (noise.at(i)(j,k,l)) / sig_weight;
                    }

                    else {
                        noise.at(i)(j,k,l) = 0;
                    }
                }
            }
        }
    }
}

void ObsMapBuffer::calc_map_psd(std::string exmode) {
    // loop through maps
    for (Eigen::Index i=0; i<signal.size(); i++) {
        // calculate weight threshold
        double weight_threshold;

        // calculate coverage cut ranges
        Eigen::VectorXd cut_row_range(2), cut_col_range(2);

        Eigen::Index cut_n_rows = cut_row_range(1) - cut_row_range(0) + 1;
        Eigen::Index cut_n_cols = cut_col_range(1) - cut_col_range(0) + 1;

        // ensure even rows and cols
        if (cut_n_rows % 2 == 1) {
            cut_row_range(1)--;
            cut_n_rows--;
        }

        if (cut_n_cols % 2 == 1) {
            cut_col_range(1)--;
            cut_n_cols--;
        }

        double diff_rows = rows_tan_vec(1) - rows_tan_vec(0);
        double diff_cols = cols_tan_vec(1) - cols_tan_vec(0);
        double rsize = diff_rows * cut_n_rows;
        double csize = diff_cols * cut_n_cols;
        double diffq_rows = 1. / rsize;
        double diffq_cols = 1. / csize;

        Eigen::MatrixXcd in(cut_n_rows, cut_n_cols);
        in.real() = signal[i].block(cut_row_range(0), cut_col_range(0), cut_n_rows, cut_n_cols);
        in.imag().setZero();

        // apply hanning window
        in.real() = in.real().array()*engine_utils::hanning_window(cut_n_rows, cut_n_cols).array();

        // do fft
        auto out = engine_utils::fft<engine_utils::forward>(in, exmode);
        out = out/cut_n_rows/cut_n_cols;

        // make vectors for frequencies
        Eigen::VectorXd q_rows(cut_n_rows), q_cols(cut_n_cols);

        // smooth the psd
    }



}

void ObsMapBuffer::calc_map_hist() {

}

} // namespace mapmaking
