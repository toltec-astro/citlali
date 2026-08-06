#pragma once

#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <citlali/core/utils/utils.h>
#include <citlali/core/utils/fitting.h>
#include <citlali/core/mapmaking/science_map_contract.h>

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

class MapBuffer {
public:
    // wcs object
    WCS wcs;

    // placeholder vectors for grppi map
    std::vector<int> map_in_vec, map_out_vec;

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

    // maps (n_rows, n_cols) of length n_maps. `weight` is the realized
    // normalization coefficient, not a precision by default. `coverage` is a
    // compatibility alias of science_products.retained_exposure.
    std::vector<Eigen::MatrixXd> signal, weight, kernel, coverage;

    // SCI-MAP-001 typed hit/exposure/support/validity products and immutable
    // full-precision bundle identity. This state is owned by the map buffer
    // lifecycle rather than synchronized through Engine.
    ScienceMapProducts science_products;
    // Once a downstream filter is about to mutate map-domain planes, this
    // const snapshot carries the already-validated raw F010 authority and raw
    // parent identities. Filtered computation never rewrites it.
    std::shared_ptr<const ScienceMapProducts> raw_science_parent;

    // Conditional products of the completed source_imprinted_current stack.
    // The legacy member names remain storage-level compatibility aliases:
    // noise_variance is S_R/R conditional finite-stack scatter,
    // sig2noise_pixel is coefficient-standardized signal,
    // point_source_uncertainty is filtered-pixel stack scatter, and
    // sig2noise_point_source is a conditional stack-scatter ratio. None is a
    // physical-noise variance, precision, significance, or aperture error.
    std::vector<Eigen::MatrixXd> weight_formal, noise_mean, noise_variance,
                                weight_empirical, sig2noise_pixel,
                                point_source_uncertainty, sig2noise_point_source;
    // Per-map product-level validity. Pixel-level invalid denominators are
    // represented by non-finite values in the corresponding ratio plane.
    Eigen::VectorXi noise_stack_scatter_valid,
                    noise_uncertainty_use_valid,
                    noise_weight_scale_valid,
                    noise_pooled_stack_scale_valid;
    // Compatibility storage for the global nonprecision scale diagnostic,
    // pooled stack-scale diagnostic, and calibration-region pixel count.
    Eigen::VectorXd noise_weight_median_ratio, noise_weight_scale,
                    noise_s2n_sigma, noise_valid_pixels;

    // optional memo-style gridding denominator used before finalizing inverse-variance weights
    std::vector<Eigen::MatrixXd> grid_weight;

    // diagnostic largest single weighted sample contribution per map pixel
    bool contribution_diag_enabled = false;
    bool contribution_diag_targeted = false;
    std::vector<std::vector<std::pair<Eigen::Index, Eigen::Index>>> contribution_targets;
    std::vector<Eigen::MatrixXd> contribution_max_abs, contribution_signal,
                                contribution_weight, contribution_variance_weight,
                                contribution_total_signal, contribution_total_weight,
                                contribution_total_variance_weight;
    std::vector<Eigen::MatrixXi> contribution_uid, contribution_scan,
                                contribution_sample;

    struct NormalizeSupportDiag {
        Eigen::Index map_index = -1;
        Eigen::Index n_total = 0;
        Eigen::Index n_retained = 0;
        Eigen::Index n_masked = 0;
        Eigen::Index n_masked_no_accum_weight = 0;
        Eigen::Index n_masked_bad_grid_weight_with_accum_weight = 0;
        Eigen::Index n_masked_by_support_threshold = 0;
        Eigen::Index n_masked_raw_signal_nonzero = 0;
        Eigen::Index n_masked_adjacent_support = 0;
        double support_weight_threshold = std::numeric_limits<double>::quiet_NaN();
        double max_masked_abs_raw_signal = std::numeric_limits<double>::quiet_NaN();
        double max_masked_neighbor_weight = std::numeric_limits<double>::quiet_NaN();
        Eigen::Index max_neighbor_row = -1;
        Eigen::Index max_neighbor_col = -1;
        int max_neighbor_cause = 0; // 1=no accum weight, 2=bad grid weight, 3=support threshold
        bool use_grid_weight = false;
    };
    std::vector<NormalizeSupportDiag> normalize_support_diag;

    // noise maps (n_rows, n_cols, n_noise) of length n_maps
    std::vector<Eigen::Tensor<double,3>> noise;

    // pointing matrix (n_pixels, 9) of length n_maps (M in Benton 2015)
    std::vector<Eigen::MatrixXd> pointing;

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
    Eigen::VectorXd median_rms, median_err;

    // realized edge-guard diagnostics for filtered map products
    std::vector<int> edge_guard_applied;
    std::vector<int> edge_guard_support_radius_pix;
    std::vector<int> edge_guard_science_npix;
    std::vector<int> edge_guard_support_npix;
    std::vector<int> edge_guard_guardband_npix;
    std::vector<double> edge_guard_weight_threshold;
    std::vector<double> edge_guard_hits_threshold;
    std::vector<double> edge_guard_background_level;
    std::vector<double> edge_guard_science_frac;
    std::vector<double> edge_guard_support_frac;
    std::vector<double> edge_guard_guardband_rms_pre;
    std::vector<double> edge_guard_guardband_rms_post;
    std::vector<double> edge_guard_exterior_rms_pre;
    std::vector<double> edge_guard_exterior_rms_post;
    std::vector<double> edge_guard_exterior_max_abs_pre;
    std::vector<double> edge_guard_exterior_max_abs_post;
    std::vector<Eigen::MatrixXd> edge_guard_window;

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
    MapBuffer();
    MapBuffer(std::string);

    // Normalize accumulated maps and finalize the numerical coefficient and
    // the distinct normalization/science-policy/validity products.
    void normalize_maps(const Eigen::Matrix<bool, Eigen::Dynamic, 1> *active_maps = nullptr);
    // Refresh the post-normalization support/provenance authority after the
    // existing optional global empirical coefficient rescale. This never
    // divides or otherwise renormalizes signal, kernel, or realizations.
    void refresh_science_products_after_coefficient_rescale(Eigen::Index);
    void freeze_raw_science_parent();
    void ensure_contribution_diag(Eigen::Index);
    void clear_contribution_diag();
    void set_contribution_targets(
        Eigen::Index,
        const std::vector<std::tuple<Eigen::Index, Eigen::Index, Eigen::Index>> &);
    void clear_contribution_targets();
    bool contribution_target_enabled(Eigen::Index, Eigen::Index, Eigen::Index) const;
    void record_contribution(Eigen::Index, Eigen::Index, Eigen::Index, double,
                             double, int, int, int);
    void record_contribution(Eigen::Index, Eigen::Index, Eigen::Index, double,
                             double, double, int, int, int);
    void calculate_stokes(std::vector<Eigen::MatrixXd>&, const Eigen::MatrixXd&,
                          Eigen::Index, Eigen::Index, int, int);
    void calculate_stokes(std::vector<Eigen::Tensor<double,3>>&, const Eigen::MatrixXd&,
                          Eigen::Index, Eigen::Index, int, int);
    void process_maps_for_pixel(Eigen::Index, Eigen::Index, int, int, const Eigen::MatrixXd&);
    void zero_out_maps(Eigen::Index, Eigen::Index, int, int);
    // normalize polarized maps
    void normalize_polarized_maps();

    // calculate map coverage region
    std::tuple<double, Eigen::MatrixXd, Eigen::Index, Eigen::Index> calc_cov_region(Eigen::Index);

    // calculate map psds
    void calc_map_psd();
    // calculate map histograms
    void calc_map_hist();

    // calculate mean square error of weight maps
    void calc_median_err();
    // calculate average rms of noise maps
    void calc_median_rms();
    // calculate empirical noise products from jackknife noise maps
    void calc_noise_products(bool, bool = true);
    void calc_noise_products(Eigen::Index, bool, bool = true);
    void clear_noise_products();
    // Apply one declared fixed linear aperture/background/template projection
    // to every compatible realization, then return S_R/R for the projected
    // scalars. This is a conditional finite-stack diagnostic, not an aperture
    // uncertainty or physical-noise calibration.
    double calc_fixed_projection_stack_scatter(
        Eigen::Index, const Eigen::MatrixXd &, double = 1.0) const;
    // calculate mean rms of signal maps within an annulus
    void calc_median_rms_annulus(double, double);
    // find sources in maps
    bool find_sources(Eigen::Index);
};

} // namespace mapmaking
