#pragma once

#include <string>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <tula/config/flatconfig.h>
#include <tula/config/yamlconfig.h>

#include <citlali/core/engine/config.h>
#include <citlali/core/utils/utils.h>
#include <citlali/core/utils/fitting.h>

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
    std::vector<int> naxis;

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
    // wcs object
    WCS wcs;

    // name of map buffer (i.e. omb, cmb)
    std::string name;

    // reference sky value
    std::vector<float> crval_config;
    // parallel policy for fft
    std::string parallel_policy;
    //obsnums
    std::vector<std::string> obsnums;
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

    // maps (n_rows, n_cols) of length n_maps
    std::vector<Eigen::MatrixXd> signal, weight, kernel, coverage;

    // noise maps (n_rows, n_cols, n_noise) of length n_maps
    std::vector<Eigen::Tensor<double,3>> noise;

    // pointing matrix (M in Benton 2015)
    std::vector<Eigen::Tensor<double,3>> pointing;

    // randomize noise maps on detectors
    bool randomize_dets;

    // coverage cut
    double cov_cut;

    // smoothing window for psd
    int smooth_window = 10;

    // number of bins for histogram
    int hist_n_bins;

    // vector to hold psds
    std::vector<Eigen::VectorXd> psds, psd_freqs;

    // vector to hold 2D psds
    std::vector<Eigen::MatrixXd> psd_2ds, psd_2d_freqs;

    // vector to hold hists
    std::vector<Eigen::VectorXd> hists, hist_bins;

    // vector to hold noise psds
    std::vector<Eigen::VectorXd> noise_psds, noise_psd_freqs;

    // vector to hold noise 2D psds
    std::vector<Eigen::MatrixXd> noise_psd_2ds, noise_psd_2d_freqs;

    // vector to hold noise hists
    std::vector<Eigen::VectorXd> noise_hists, noise_hist_bins;

    // vector to hold mean rms values
    Eigen::VectorXd mean_rms, mean_err;

    // number of sources found by source finder
    std::vector<int> n_sources;

    // source finding mode
    std::string source_finder_mode;

    // minimum source sigma
    double source_sigma;
    // mask window around source
    double source_window_rad;

    // hold source row/col locations
    std::vector<Eigen::VectorXi> row_source_locs, col_source_locs;

    // fitted source parameters and errors [n_sources x n_params]
    Eigen::MatrixXd source_params, source_perror;

    // constructor
    ObsMapBuffer(std::string);

    // get config file
    void get_config(tula::config::YamlConfig &, std::vector<std::vector<std::string>> &,
                    std::vector<std::vector<std::string>> &, std::string, std::string);

    // normalize signal and noise maps by the weight maps
    void normalize_maps();

    // calculate map coverage region
    std::tuple<double, Eigen::MatrixXd, Eigen::Index, Eigen::Index> calc_cov_region(Eigen::Index);

    // calculate map psds
    void calc_map_psd();
    // calculate map histograms
    void calc_map_hist();

    // calculate mean square error of weight maps
    void calc_mean_err();
    // calculate average rms of noise maps
    void calc_mean_rms();
    // find sources in maps
    bool find_sources(Eigen::Index);
};

} // namespace mapmaking
