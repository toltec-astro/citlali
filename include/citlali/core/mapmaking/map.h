#pragma once

#include <string>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <citlali/core/utils/utils.h>
#include <citlali/core/mapmaking/wiener_filter.h>

namespace mapmaking {

enum MapType {
    RawObs = 0,
    FilteredObs = 1,
    RawCoadd = 2,
    FilteredCoadd = 3,
};

// wcs information
struct WCS {
    // pixel size
    std::vector<float> cdelt;

    // map size in pixels
    std::vector<float> naxis;

    // reference pixels
    std::vector<float> crpix;

    // reference sky value
    std::vector<float> crval;

    // map unit
    std::vector<std::string> cunit;

    // coord type
    std::vector<std::string> ctype;
};

class ObsMapBuffer {
public:
    // parallel policy for fft
    std::string parallel_policy;
    //obsnums
    std::vector<std::string> obsnums;
    // wcs
    WCS wcs;
    // map grouping
    std::string map_grouping;
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
    // exposure time
    double exposure_time = 0;
    // maps
    std::vector<Eigen::MatrixXd> signal, weight, kernel, coverage, flag;
    // noise maps (n_rows, n_cols, n_noise) of length n_maps
    std::vector<Eigen::Tensor<double,3>> noise;

    // coverage cut
    double cov_cut;

    // smoothing window for psd
    int smooth_window = 10;

    // number of bins for histogram
    int hist_n_bins;

    // vector to hold psds
    std::vector<Eigen::VectorXd> psds;
    // vector to hold psd freqs
    std::vector<Eigen::VectorXd> psd_freqs;

    // vector to hold 2D psds
    std::vector<Eigen::MatrixXd> psd_2ds;
    // vector to hold 2D psd freqs
    std::vector<Eigen::MatrixXd> psd_2d_freqs;

    // vector to hold hists
    std::vector<Eigen::VectorXd> hists;
    // vector to hold hist bins
    std::vector<Eigen::VectorXd> hist_bins;

    // vector to hold noise psds
    std::vector<Eigen::VectorXd> noise_psds;
    // vector to hold noise psd freqs
    std::vector<Eigen::VectorXd> noise_psd_freqs;

    // vector to hold noise 2D psds
    std::vector<Eigen::MatrixXd> noise_psd_2ds;
    // vector to hold noise 2D psd freqs
    std::vector<Eigen::MatrixXd> noise_psd_2d_freqs;

    // vector to hold noise hists
    std::vector<Eigen::VectorXd> noise_hists;
    // vector to hold noise hist bins
    std::vector<Eigen::VectorXd> noise_hist_bins;

    void normalize_maps();

    template <typename Derived>
    auto calc_cov_region(Eigen::DenseBase<Derived> &);

    void calc_map_psd();
    void calc_map_hist();

    void wiener_filter_maps();
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
    for (Eigen::Index i=0; i<signal.size(); i++) {
        for (Eigen::Index j=0; j<n_rows; j++) {
            for (Eigen::Index k=0; k<n_cols; k++) {
                double sig_weight = weight.at(i)(j,k);
                if (sig_weight != 0.) {
                    for (Eigen::Index l=0; l<noise.size(); l++) {
                        noise[i](j,k,l) = (noise.at(i)(j,k,l)) / sig_weight;
                    }
                }
                else {
                    for (Eigen::Index l=0; l<noise.size(); l++) {
                        noise[i](j,k,l) = 0;
                    }
                }
            }
        }
    }
}

template <typename Derived>
auto ObsMapBuffer::calc_cov_region(Eigen::DenseBase<Derived> &w) {
    // calculate weight threshold
    double weight_threshold = engine_utils::find_weight_threshold(w, cov_cut);

    // calculate coverage ranges
    Eigen::MatrixXd cov_ranges = engine_utils::set_cov_cov_ranges(w, weight_threshold);

    Eigen::Index cov_n_rows = cov_ranges(1,0) - cov_ranges(0,0) + 1;
    Eigen::Index cov_n_cols = cov_ranges(1,1) - cov_ranges(0,1) + 1;

    return std::tuple<double, Eigen::MatrixXd, Eigen::Index, Eigen::Index>(weight_threshold, cov_ranges, cov_n_rows, cov_n_cols);
}

// loop through maps
void ObsMapBuffer::calc_map_psd() {
    // clear storage vectors
    psds.clear();
    psd_freqs.clear();
    psd_2ds.clear();
    psd_2d_freqs.clear();

    noise_psds.clear();
    noise_psd_freqs.clear();
    noise_psd_2ds.clear();
    noise_psd_2d_freqs.clear();

    // loop through maps
    for (Eigen::Index i=0; i<signal.size(); i++) {
        // calculate weight threshold
        auto [weight_threshold, cov_ranges, cov_n_rows, cov_n_cols] = calc_cov_region(weight[i]);

        // ensure even rows
        if (cov_n_rows % 2 == 1) {
            cov_ranges(1,0)--;
            cov_n_rows--;
        }

        // ensure even cols
        if (cov_n_cols % 2 == 1) {
            cov_ranges(1,1)--;
            cov_n_cols--;
        }

        // explicit copy
        Eigen::MatrixXd sig = signal[i].block(cov_ranges(0,0), cov_ranges(0,1), n_rows, n_cols);
        // calculate psds
        auto [p, pf, p_2d, pf_2d] = engine_utils::calc_2D_psd(sig, rows_tan_vec, cols_tan_vec, cov_n_rows, cov_n_cols,
                                                           smooth_window, parallel_policy);

        // move current map psd values into vectors
        psds.push_back(std::move(p));
        psd_freqs.push_back(std::move(pf));

        psd_2ds.push_back(std::move(p_2d));
        psd_2d_freqs.push_back(std::move(pf_2d));

        // get average noise psd if noise maps are requested
        if (!noise.empty()) {
            for (Eigen::Index j=0; j<n_noise; j++) {
                // get noise map
                Eigen::Tensor<double, 2> noise_tensor = noise[i].chip(j, 2);
                // map to eigen matrix
                Eigen::Map<Eigen::MatrixXd> noise_matrix(noise_tensor.data(), noise_tensor.dimension(0), noise_tensor.dimension(1));
                sig = noise_matrix.block(cov_ranges(0,0), cov_ranges(0,1), n_rows, n_cols);

                // calculate psds
                auto [noise_p, noise_pf, noise_p_2d, noise_pf_2d] = engine_utils::calc_2D_psd(sig, rows_tan_vec, cols_tan_vec, cov_n_rows,
                                                                                           cov_n_cols, smooth_window, parallel_policy);

                // just copy if on first noise map
                if (j==0) {
                    noise_psds.push_back(std::move(noise_p));
                    noise_psd_freqs.push_back(std::move(noise_pf));

                    noise_psd_2ds.push_back(std::move(noise_p_2d));
                    noise_psd_2d_freqs.push_back(std::move(noise_pf_2d));
                }

                // otherwise add to existing vector
                else {
                    noise_psds.back() = noise_psds.back() + noise_p;
                    noise_psd_2ds.back() = noise_psd_2ds.back() + noise_p_2d/n_noise;
                    noise_psd_2d_freqs.back() = noise_psd_2d_freqs.back() + noise_pf_2d/n_noise;
                }
            }
            noise_psds.back() = noise_psds.back()/n_noise;
        }
    }
}

void ObsMapBuffer::calc_map_hist() {
    // clear storage vectors
    hists.clear();
    hist_bins.clear();

    noise_hists.clear();
    noise_hist_bins.clear();

    // loop through maps
    for (Eigen::Index i=0; i<signal.size(); i++) {
        // calculate weight threshold
        auto [weight_threshold, cov_ranges, cov_n_rows, cov_n_cols] = calc_cov_region(weight[i]);

        // setup input signal data
        Eigen::MatrixXd sig = signal[i].block(cov_ranges(0,0),cov_ranges(0,1), cov_n_rows, cov_n_cols);

        // calculate histogram and bins
        auto [h, h_bins] = engine_utils::calc_hist(sig, hist_n_bins);

        hists.push_back(std::move(h));
        hist_bins.push_back(std::move(h_bins));

        // get average noise psd if noise maps are requested
        if (!noise.empty()) {
            for (Eigen::Index j=0; j<n_noise; j++) {
                // get noise map
                Eigen::Tensor<double, 2> noise_tensor = noise[i].chip(j,2);
                // map to eigen matrix
                Eigen::Map<Eigen::MatrixXd> noise_matrix(noise_tensor.data(), noise_tensor.dimension(0), noise_tensor.dimension(1));
                sig = noise_matrix.block(cov_ranges(0,0), cov_ranges(0,1), n_rows, n_cols);

                // calculate histogram and bins
                auto [noise_h, noise_h_bins] = engine_utils::calc_hist(sig, hist_n_bins);

                // just copy if on first noise map
                if (j==0) {
                    noise_hists.push_back(std::move(noise_h));
                    noise_hist_bins.push_back(std::move(noise_h_bins));
                }
                // otherwise add to existing vector
                else {
                    noise_hists.back() = noise_hists.back() + noise_h;
                }
            }
            noise_hists.back() = noise_hists.back()/n_noise;
        }
    }
}

void ObsMapBuffer::wiener_filter_maps() {
    // loop through maps
    for (Eigen::Index i=0; i<signal.size(); i++) {
        //WienerFilter wiener_filter;
        //wiener_filter.make_template();
        //wiener_filter.run();
    }
}

} // namespace mapmaking
