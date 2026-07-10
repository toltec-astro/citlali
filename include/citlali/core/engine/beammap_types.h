#pragma once

#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace citlali::engine_detail::beammap {

struct SoftPriorSlot {
    double x_arcsec = 0.0;
    double y_arcsec = 0.0;
    double sx_arcsec = 8.0;
    double sy_arcsec = 8.0;
    int slot_index = -1;
};

struct PriorArrayAlignment {
    bool valid = false;
    double cos_theta = 1.0;
    double sin_theta = 0.0;
    double theta_rad = 0.0;
    double dx_arcsec = 0.0;
    double dy_arcsec = 0.0;
    Eigen::Index n_matches = 0;
    double rms_arcsec = std::numeric_limits<double>::quiet_NaN();
};

struct BeammapPriorFrameCenterSamples {
    std::map<int, std::vector<double>> x_by_array;
    std::map<int, std::vector<double>> y_by_array;
    std::set<int> arrays_missing;
    Eigen::Index n_previous = 0;
    Eigen::Index n_blind = 0;
};

struct BeammapPriorAlignmentPair {
    double obs_x = 0.0;
    double obs_y = 0.0;
    double slot_x = 0.0;
    double slot_y = 0.0;
};

struct BeammapPriorAlignmentSamples {
    std::map<int, std::vector<BeammapPriorAlignmentPair>> pairs_by_array;
    std::vector<BeammapPriorAlignmentPair> all_pairs;
    std::set<int> arrays_with_alignment_pairs;
    Eigen::Index n_matches = 0;
};

struct BeammapPriorAlignmentOverlapBox {
    bool valid = false;
    double x_low = -std::numeric_limits<double>::infinity();
    double x_high = std::numeric_limits<double>::infinity();
    double y_low = -std::numeric_limits<double>::infinity();
    double y_high = std::numeric_limits<double>::infinity();
};

enum class BeammapFitInitMode { Blind, Previous, Prior };

struct BeammapFitAttemptFlags {
    bool init_amp_zero = false;
    bool amp_bounds_zero = false;
};

struct BeammapPreviousFitInit {
    bool valid = false;
    bool rejected_by_peak = false;
    double row = -99.0;
    double col = -99.0;
};

struct BeammapFitInitSelection {
    bool skip_fit = false;
    bool from_previous = false;
    bool from_prior = false;
    BeammapFitInitMode mode = BeammapFitInitMode::Blind;
    double row = -99.0;
    double col = -99.0;
};

struct BeammapFitIterationStats {
    Eigen::VectorXi bound_low;
    Eigen::VectorXi bound_high;
    Eigen::Index bound_any = 0;
    Eigen::Index init_prev = 0;
    Eigen::Index init_prior = 0;
    Eigen::Index init_blind = 0;
    Eigen::Index init_skip = 0;
    Eigen::Index attempt_prev = 0;
    Eigen::Index attempt_prior = 0;
    Eigen::Index attempt_blind = 0;
    Eigen::Index fail_prev = 0;
    Eigen::Index fail_prior = 0;
    Eigen::Index fail_blind = 0;
    Eigen::Index prev_rejected_by_peak = 0;
    Eigen::Index init_amp_zero_prev = 0;
    Eigen::Index init_amp_zero_prior = 0;
    Eigen::Index init_amp_zero_blind = 0;
    Eigen::Index amp_bounds_zero_prev = 0;
    Eigen::Index amp_bounds_zero_prior = 0;
    Eigen::Index amp_bounds_zero_blind = 0;

    explicit BeammapFitIterationStats(Eigen::Index n_params)
        : bound_low(Eigen::VectorXi::Zero(n_params)),
          bound_high(Eigen::VectorXi::Zero(n_params)) {}
};

struct BeammapDetectorSourceCenterStats {
    Eigen::Index n_valid = 0;
    Eigen::Index n_valid_fwhm = 0;
};

struct BeammapDetectorTodPreflight {
    bool write_output = false;
    Eigen::Index n_scans = 0;
    int n_uniform = 0;
    int n_dense = 0;
    Eigen::Index n_slots = 0;
    Eigen::Index n_samples_max = 0;
};

struct BeammapDetectorTodPointingSamples {
    bool valid = false;
    Eigen::Index n_sampled = 0;
    std::vector<Eigen::Index> sampled_indices;
    std::vector<Eigen::Index> sampled_scan;
    std::map<std::string, Eigen::VectorXd> sampled_tel_data;
    std::map<std::string, Eigen::VectorXd> pointing_offsets;
};

struct BeammapDetectorTodSelections {
    int fill_int = -2147483647;
    double fill_double = std::numeric_limits<double>::quiet_NaN();
    float fill_float = std::numeric_limits<float>::quiet_NaN();
    signed char fill_flag = static_cast<signed char>(-1);
    std::vector<int> slot_scan_index;
    std::vector<int> slot_kind;
    std::vector<int> slot_n_samples;
    std::vector<int> slot_inner_start;
    std::vector<int> slot_inner_end;
    std::vector<int> slot_outer_start;
    std::vector<int> slot_outer_end;
    std::vector<double> slot_source_distance_arcsec;
    std::vector<int> det_center_scan_index;
    std::vector<double> det_center_distance_arcsec;
    std::vector<double> det_fit_x_arcsec;
    std::vector<double> det_fit_y_arcsec;
    std::vector<int> det_fit_good;
    Eigen::Index n_det_fit_positions = 0;
    Eigen::Index n_det_fallback_positions = 0;
    std::string center_scan_summary;
    double median_center_distance_arcsec =
        std::numeric_limits<double>::quiet_NaN();
};

struct BeammapEmpiricalTemplateGeometry {
    Eigen::Index template_radius_pix = 0;
    Eigen::Index match_radius_pix = 0;
    Eigen::Index peak_radius_pix = 0;
    Eigen::Index template_peak_radius_pix = 0;
    Eigen::Index side = 0;
    Eigen::Index center = 0;
};

struct BeammapEmpiricalTemplateCandidate {
    Eigen::Index map_index = -1;
    double shape_score = std::numeric_limits<double>::infinity();
    double snr = 0.0;
};

struct BeammapEmpiricalTemplateShapeMedians {
    bool valid = false;
    double a_fwhm = std::numeric_limits<double>::quiet_NaN();
    double b_fwhm = std::numeric_limits<double>::quiet_NaN();
};

struct BeammapArrayTemplate {
    bool valid = false;
    Eigen::MatrixXd shape;
    Eigen::Index n_detectors = 0;
};

struct BeammapTemplateFitResult {
    bool valid = false;
    double amp = std::numeric_limits<double>::quiet_NaN();
    double offset = std::numeric_limits<double>::quiet_NaN();
    double resid_rms = std::numeric_limits<double>::quiet_NaN();
    Eigen::Index npix = 0;
};

struct BeammapTemplateFitSamples {
    std::vector<double> y;
    std::vector<double> t;
    std::vector<double> w;
};

struct BeammapArrayPositionMedians {
    std::map<std::string, double> x_t;
    std::map<std::string, double> y_t;
};

struct BeammapPriorDistanceFrame {
    bool apply_derot = false;
    double cos_rot = 1.0;
    double sin_rot = 0.0;
};

struct BeammapSplitMapOutputFiles {
    std::vector<std::string> base_filepaths;
    std::vector<std::string> base_noise_filepaths;
};

struct RFIMaskScanSummary {
    Eigen::Index n_det_candidates = 0;
    Eigen::Index n_det_flagged = 0;
    Eigen::Index n_samples_flagged = 0;
    Eigen::Index n_det_rejected = 0;
};

struct ScanBandMaskSummary {
    Eigen::Index n_det_flagged = 0;
    Eigen::Index n_rows_flagged = 0;
    Eigen::Index n_samples_flagged = 0;
    Eigen::Index n_det_rejected = 0;
};

struct ScanBandRowStats {
    std::vector<double> medians;
    std::vector<double> sigmas;
    std::vector<Eigen::Index> counts;
    std::vector<double> interior_medians;
    std::vector<double> interior_sigmas;
};

struct ScanBandEdgeRows {
    std::vector<Eigen::Index> top;
    std::vector<Eigen::Index> bottom;
};

struct ScanBandProposedFlags {
    std::vector<std::pair<Eigen::Index, Eigen::Index>> samples;
    Eigen::Index n_good_samples = 0;
};

} // namespace citlali::engine_detail::beammap
