#pragma once

// Included by mapdiag_stats.h inside namespace citlali::pipeline.

struct MapdiagTailStats {
    double frac_abs3 = std::numeric_limits<double>::quiet_NaN();
    double frac_pos3 = std::numeric_limits<double>::quiet_NaN();
    double frac_neg3 = std::numeric_limits<double>::quiet_NaN();
    double excess_abs3 = std::numeric_limits<double>::quiet_NaN();
    double excess_pos3 = std::numeric_limits<double>::quiet_NaN();
    double excess_neg3 = std::numeric_limits<double>::quiet_NaN();
    double skew = std::numeric_limits<double>::quiet_NaN();
};

struct MapdiagStatsContext {
    double fill_value;

    double median(const std::vector<double> &values) const;
    double quantile(std::vector<double> values, double q) const;
    std::vector<double> collect_masked_values(
        const Eigen::MatrixXd &matrix, const Eigen::ArrayXXd &mask) const;
    MapdiagTailStats tail_stats(const std::vector<double> &values) const;
};

inline void mapdiag_append_finite(std::vector<double> &values, double value);

struct MapdiagNoiseTailSamples {
    std::vector<double> rms;
    std::vector<double> tail_abs;
    std::vector<double> tail_pos;
    std::vector<double> tail_neg;
    std::vector<double> excess_abs;
    std::vector<double> excess_pos;
    std::vector<double> excess_neg;
    std::vector<double> skew;

    void reserve(std::size_t n_noise);
    void add_tail_stats(const MapdiagTailStats &stats);
};

struct MapdiagNoiseTailSummary {
    double rms_p16;
    double rms_p84;
    double tail_abs;
    double tail_pos;
    double tail_neg;
    double excess_abs;
    double excess_pos;
    double excess_neg;
    double skew;
};

struct MapdiagNoiseProductStats {
    double weight_median_ratio;
    double weight_scale;
    double s2n_sigma;
    double valid_pixels;
};

struct MapdiagFormalNoiseStats {
    double median_err;
    double median_rms;
    double empirical_to_formal_ratio;
};

struct MapdiagSourceDistanceContext {
    double center_row;
    double center_col;
    double pixel_size_arcsec;
    double fill_value;
};

struct MapdiagRobustCenterStats {
    double center;
    double robust_sigma;
};

struct MapdiagMapPixelCandidate {
    int row;
    int col;
    int uid;
    int scan;
    long long sample;
    double value;
    double weight;
    double n_eff;
    double robust_z;
    double leave_one_out_z;
    double source_distance_arcsec;
    bool source_protected;
    bool has_contributor;
};

struct MapdiagDetectorDominance {
    int uid;
    int scan;
    int count;
    double max_abs_value;
    double max_abs_leave_one_out_z;
};

struct MapdiagWeightStats {
    int n_valid_pixels;
    int n_core_pixels;
    double weight_sum;
    double core_weight_sum;
};

struct MapdiagCoverageStats {
    double sum;
    double max;
    double median_core;
};

struct MapdiagPeakStats {
    double peak_abs_sig2noise;
    int peak_row;
    int peak_col;
    double core_peak_abs_sig2noise;
};

struct MapdiagPeakRefs {
    std::vector<double> &peak_abs_sig2noise;
    std::vector<double> &core_peak_abs_sig2noise;
    std::vector<int> &peak_row;
    std::vector<int> &peak_col;
};

struct MapdiagWeightRefs {
    std::vector<double> &weight_sum;
    std::vector<double> &core_weight_sum;
    std::vector<int> &n_valid_pixels;
    std::vector<int> &n_core_pixels;
};

struct MapdiagCoverageRefs {
    std::vector<double> &coverage_sum;
    std::vector<double> &coverage_max;
    std::vector<double> &coverage_median_core;
};

struct MapdiagNoiseProductRefs {
    std::vector<double> &weight_median_ratio;
    std::vector<double> &weight_scale;
    std::vector<double> &s2n_sigma;
    std::vector<double> &valid_pixels;
};

struct MapdiagFormalNoiseRefs {
    std::vector<double> &median_err;
    std::vector<double> &median_rms;
    std::vector<double> &empirical_to_formal_ratio;
};

struct MapdiagCoreTailRefs {
    std::vector<double> &frac_abs3;
    std::vector<double> &frac_pos3;
    std::vector<double> &frac_neg3;
    std::vector<double> &excess_abs3;
    std::vector<double> &excess_pos3;
    std::vector<double> &excess_neg3;
    std::vector<double> &skew;
};

struct MapdiagNoiseTailRefs {
    std::vector<double> &rms_p16;
    std::vector<double> &rms_p84;
    std::vector<double> &frac_abs3;
    std::vector<double> &frac_pos3;
    std::vector<double> &frac_neg3;
    std::vector<double> &excess_abs3;
    std::vector<double> &excess_pos3;
    std::vector<double> &excess_neg3;
    std::vector<double> &skew;
};

using MapdiagNoiseMatrix =
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>;

