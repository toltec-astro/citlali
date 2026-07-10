#pragma once

#include <map>
#include <vector>
#include <string>
#include <cstdlib>
#include <limits>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <memory>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <condition_variable>
#include <queue>
#include <set>

#include <citlali/core/engine/engine.h>
#include <citlali/core/engine/beammap_types.h>
#include <citlali/core/utils/ecsv_io.h>

using timestream::TCData;
using timestream::RTCProc;
using timestream::PTCProc;

// selects the type of TCData
using timestream::TCDataKind;

namespace citlali::pipeline {
struct BeammapArrayFlaggingLimits;
}

namespace citlali::config {
struct BeammapPriorsConfig;
struct BeammapSplitFitsByFlagConfig;
}

class Beammap: public Engine {
public:
    using SoftPriorSlot = citlali::engine_detail::beammap::SoftPriorSlot;
    using PriorArrayAlignment =
        citlali::engine_detail::beammap::PriorArrayAlignment;
    using BeammapPriorFrameCenterSamples =
        citlali::engine_detail::beammap::BeammapPriorFrameCenterSamples;
    using BeammapPriorAlignmentPair =
        citlali::engine_detail::beammap::BeammapPriorAlignmentPair;
    using BeammapPriorAlignmentSamples =
        citlali::engine_detail::beammap::BeammapPriorAlignmentSamples;
    using BeammapPriorAlignmentOverlapBox =
        citlali::engine_detail::beammap::BeammapPriorAlignmentOverlapBox;
    using BeammapFitInitMode =
        citlali::engine_detail::beammap::BeammapFitInitMode;
    using BeammapFitAttemptFlags =
        citlali::engine_detail::beammap::BeammapFitAttemptFlags;
    using BeammapPreviousFitInit =
        citlali::engine_detail::beammap::BeammapPreviousFitInit;
    using BeammapFitInitSelection =
        citlali::engine_detail::beammap::BeammapFitInitSelection;
    using BeammapFitIterationStats =
        citlali::engine_detail::beammap::BeammapFitIterationStats;
    using BeammapDetectorTodPreflight =
        citlali::engine_detail::beammap::BeammapDetectorTodPreflight;
    using BeammapDetectorTodPointingSamples =
        citlali::engine_detail::beammap::BeammapDetectorTodPointingSamples;
    using BeammapDetectorTodSelections =
        citlali::engine_detail::beammap::BeammapDetectorTodSelections;
    using BeammapEmpiricalTemplateGeometry =
        citlali::engine_detail::beammap::BeammapEmpiricalTemplateGeometry;
    using BeammapEmpiricalTemplateCandidate =
        citlali::engine_detail::beammap::BeammapEmpiricalTemplateCandidate;
    using BeammapEmpiricalTemplateShapeMedians =
        citlali::engine_detail::beammap::BeammapEmpiricalTemplateShapeMedians;
    using BeammapArrayTemplate =
        citlali::engine_detail::beammap::BeammapArrayTemplate;
    using BeammapTemplateFitResult =
        citlali::engine_detail::beammap::BeammapTemplateFitResult;
    using BeammapTemplateFitSamples =
        citlali::engine_detail::beammap::BeammapTemplateFitSamples;
    using BeammapArrayPositionMedians =
        citlali::engine_detail::beammap::BeammapArrayPositionMedians;
    using BeammapPriorDistanceFrame =
        citlali::engine_detail::beammap::BeammapPriorDistanceFrame;
    using BeammapSplitMapOutputFiles =
        citlali::engine_detail::beammap::BeammapSplitMapOutputFiles;
    using RFIMaskScanSummary =
        citlali::engine_detail::beammap::RFIMaskScanSummary;
    using ScanBandMaskSummary =
        citlali::engine_detail::beammap::ScanBandMaskSummary;
    using ScanBandRowStats =
        citlali::engine_detail::beammap::ScanBandRowStats;
    using ScanBandEdgeRows =
        citlali::engine_detail::beammap::ScanBandEdgeRows;
    using ScanBandProposedFlags =
        citlali::engine_detail::beammap::ScanBandProposedFlags;

    bool beammap_soft_priors_loaded = false;
    bool beammap_soft_priors_are_centered = false;
    bool beammap_soft_priors_are_derotated = false;
    std::map<std::pair<int, int>, std::vector<SoftPriorSlot>> beammap_soft_prior_slots;
    std::map<int, double> beammap_prior_array_center_x_arcsec;
    std::map<int, double> beammap_prior_array_center_y_arcsec;
    std::map<int, PriorArrayAlignment> beammap_prior_array_alignment;

    // parallel policies for each section
    std::string  map_parallel_policy;

    // vector to store each scan's PTCData
    std::vector<TCData<TCDataKind::PTC,Eigen::MatrixXd>> ptcs0, ptcs;

    // copy of obs map buffer for map iteration
    mapmaking::MapBuffer omb_copy{"omb"};

    // vector to store each scan's calib class
    std::vector<engine::Calib> calib_scans0, calib_scans;

    // beammap iteration parameters
    Eigen::Index current_iter;

    // vector for convergence check
    Eigen::Matrix<bool, Eigen::Dynamic, 1> converged;

    // vector to record convergence iteration
    Eigen::Vector<int, Eigen::Dynamic> converge_iter;

    // previous iteration fit parameters
    Eigen::MatrixXd p0, perror0;

    // current iteration fit parameters
    Eigen::MatrixXd params, perrors;

    // reference detector
    Eigen::Index beammap_reference_det_found = -99;

    // bitwise flags
    enum AptFlags {
        Good         = 0,
        BadFit       = 1 << 0,
        AzFWHM       = 1 << 1,
        ElFWHM       = 1 << 2,
        Sig2Noise    = 1 << 3,
        Sens         = 1 << 4,
        Position     = 1 << 5,
        PriorDist    = 1 << 6,
        NetworkPos   = 1 << 7,
        };

    // holds bitwise flags
    Eigen::Matrix<uint16_t,Eigen::Dynamic,1> flag2;

    // good fits
    Eigen::Matrix<bool, Eigen::Dynamic, 1> good_fits;

    // per-map fit-bound diagnostics (for final QC outputs)
    Eigen::MatrixXd fit_diag_init_params;
    Eigen::MatrixXd fit_diag_lower_limits;
    Eigen::MatrixXd fit_diag_upper_limits;
    Eigen::MatrixXi fit_diag_hit_lower;
    Eigen::MatrixXi fit_diag_hit_upper;
    Eigen::VectorXi fit_diag_bound_code;
    Eigen::VectorXi fit_diag_bound_nhit;

    enum PriorDiagColumn {
        prior_init_mode_col = 0,
        prior_used_col,
        prior_fallback_blind_col,
        prior_no_candidate_reason_col,
        prior_slot_index_col,
        prior_match_d2_col,
        prior_match_score_col,
        prior_candidate_snr_col,
        prior_n_candidates_col,
        prior_n_candidates_keep_col,
        prior_n_candidates_gate_col,
        prior_candidate_x_raw_col,
        prior_candidate_y_raw_col,
        prior_candidate_x_prior_col,
        prior_candidate_y_prior_col,
        prior_center_x_col,
        prior_center_y_col,
        prior_derot_elev_col,
        prior_slot_x_col,
        prior_slot_y_col,
        prior_slot_sx_col,
        prior_slot_sy_col,
        n_prior_diag_cols
    };

    // per-map prior-init diagnostics (for final QC outputs)
    Eigen::MatrixXd prior_diag_values;

    // diagnostics for sample-level beammap RFI masking
    Eigen::VectorXi rfi_mask_samples_flagged;
    Eigen::VectorXi rfi_mask_scans_flagged;
    Eigen::VectorXi scan_band_mask_samples_flagged;
    Eigen::VectorXi scan_band_mask_rows_flagged;
    Eigen::VectorXi scan_band_mask_edge_code;
    Eigen::VectorXi scan_band_mask_rejected;
    Eigen::VectorXd final_prior_d2_diag;
    Eigen::VectorXi final_prior_slot_index_diag;
    std::shared_ptr<std::mutex> rfi_mask_diag_mutex = std::make_shared<std::mutex>();

    // placeholder vectors for grppi maps
    std::vector<int> scan_in_vec, scan_out_vec;
    std::vector<int> det_in_vec, det_out_vec;

    // initial setup for each obs
    void assign_beammap_kids_tone_indices();
    void register_beammap_kids_tone_column();
    void setup_beammap_kids_tone_column();
    void resize_beammap_scan_buffers();
    void reset_beammap_fit_buffers();
    void reset_beammap_mask_diagnostics();
    void reset_beammap_convergence_state();
    void resize_beammap_state_buffers();
    void populate_beammap_identity_metadata();
    void populate_beammap_phase_metadata();
    void populate_beammap_flux_metadata();
    void populate_beammap_time_and_frame_metadata();
    void populate_beammap_tau_metadata();
    void populate_beammap_header_metadata();
    void populate_beammap_reference_metadata();
    void populate_beammap_masking_metadata();
    void populate_beammap_weighting_and_fit_metadata();
    void populate_beammap_setup_metadata();
    void init_beammap_diagnostic_apt_column(
        const std::string &name,
        double fill_value,
        const std::string &unit,
        const std::string &description);
    void init_beammap_diagnostic_apt_columns();
    void init_beammap_flag_metadata();
    void reset_beammap_soft_prior_setup_state();
    void load_or_disable_beammap_soft_priors();
    void populate_beammap_soft_prior_metadata();
    void configure_beammap_soft_prior_setup();
    void setup();

    // timestream grppi pipeline
    template <class KidsProc, class RawObs>
    void timestream_pipeline(KidsProc &, RawObs &, bool write_outputs = true);

    // run the raw time chunk processing
    template <class KidsProc>
    auto run_timestream(KidsProc &, bool write_outputs = true);

    // run the loop pipeline
    template <class KidsProc, class RawObs>
    void loop_pipeline(KidsProc &, RawObs &);
    void calculate_beammap_detector_sensitivities(
        const std::string &map_parallel_policy);
    void populate_beammap_detector_fit_apt_columns();
    void populate_beammap_mask_diagnostic_apt_columns();
    void log_beammap_final_bound_summary();
    void write_beammap_final_prior_diagnostics_to_apt();
    void refresh_beammap_final_calibration_products();
    void update_beammap_final_tod_pointing(
        const std::string &map_parallel_policy,
        citlali::config::MapGrouping mapmaking_grouping);
    void finalize_beammap_detector_grouping_outputs(
        const std::string &map_parallel_policy,
        citlali::config::MapGrouping mapmaking_grouping);
    void finalize_beammap_non_detector_grouping_outputs();

    // run the iterative stage
    template <class KidsProc, class RawObs>
    void run_loop(KidsProc &, RawObs &);
    template <class RandomBits, class Generator>
    void run_beammap_mapmaking_pass(bool update_progress,
                                    RandomBits &rands,
                                    Generator &eng);
    template <class RandomBits, class Generator>
    void run_beammap_mapmaking_stage(bool locator_iter,
                                     bool measurement_iter,
                                     bool detector_grouping,
                                     RandomBits &rands,
                                     Generator &eng);
    template <class KidsProc, class RawObs>
    bool maybe_run_beammap_source_aware_rtc(KidsProc &, RawObs &,
                                            bool first_measurement_iter,
                                            bool detector_grouping);
    void fit_beammap_maps(bool detector_grouping, bool measurement_iter);
    void require_beammap_fit_map_geometry(Eigen::Index map_index) const;
    void log_beammap_fit_map_stats(Eigen::Index map_index) const;
    bool prepare_beammap_fit_map(Eigen::Index map_index);
    double beammap_init_fwhm_pix(Eigen::Index map_index);
    void restore_converged_beammap_fit_result(Eigen::Index map_index);
    void reset_beammap_fit_diagnostics(Eigen::Index map_index);
    void clear_beammap_fit_result(Eigen::Index map_index);
    bool has_beammap_prior_diagnostics() const;
    void reset_beammap_prior_diagnostics(Eigen::Index map_index);
    bool beammap_prior_position_compatible(
        Eigen::Index map_index, double row, double col,
        double derot_elev_rad, double prior_max_d2,
        double &d2_out);
    bool beammap_prior_allows_peak_switch(Eigen::Index map_index,
                                          double prev_row, double prev_col,
                                          Eigen::Index peak_row,
                                          Eigen::Index peak_col);
    bool has_previous_beammap_fit_init_candidate(
        Eigen::Index map_index, bool measurement_iter) const;
    bool read_previous_beammap_fit_seed(
        Eigen::Index map_index, double prev_row, double prev_col,
        double &seed_signal, double &seed_weight) const;
    bool should_reject_previous_beammap_fit_for_peak(
        Eigen::Index map_index, double prev_row, double prev_col,
        double seed_signal, double seed_weight, bool can_try_prior,
        double init_fwhm);
    BeammapPreviousFitInit choose_previous_beammap_fit_init(
        Eigen::Index map_index, bool measurement_iter, bool can_try_prior,
        double init_fwhm);
    void record_beammap_prior_init_mode(
        Eigen::Index map_index, const BeammapFitInitSelection &init_selection);
    bool try_beammap_prior_fit_init(
        Eigen::Index map_index,
        BeammapFitInitSelection &selection,
        BeammapFitIterationStats &fit_stats);
    BeammapFitInitSelection choose_beammap_fit_init(
        Eigen::Index map_index, bool measurement_iter, bool can_try_prior,
        double init_fwhm, BeammapFitIterationStats &fit_stats);
    const char *beammap_fit_init_mode_name(BeammapFitInitMode init_mode) const;
    BeammapFitAttemptFlags beammap_fit_attempt_flags(
        const engine_utils::mapFitter::FitDiagnostics &fit_diag) const;
    void record_beammap_fit_attempt_stats(
        BeammapFitIterationStats &fit_stats, BeammapFitInitMode init_mode,
        bool good_fit, bool init_amp_zero, bool amp_bounds_zero);
    bool has_complete_beammap_fit_diagnostics(
        const engine_utils::mapFitter::FitDiagnostics &fit_diag) const;
    void record_beammap_fit_diagnostics(
        Eigen::Index map_index,
        const engine_utils::mapFitter::FitDiagnostics &fit_diag,
        BeammapFitIterationStats &fit_stats);
    void log_beammap_fit_iteration_stats(
        const BeammapFitIterationStats &fit_stats);
    bool update_beammap_convergence_state();
    bool advance_beammap_iteration_state();
    void write_or_clear_beammap_ptc_products_for_iter(int completed_iter,
                                                      bool keep_going);
    void process_beammap_ptc_scan(
        int scan_index, bool locator_iter, bool measurement_iter,
        bool detector_grouping,
        const std::shared_ptr<std::mutex> &ptc_line_audit_mutex);
    bool subtract_beammap_model_for_ptc_scan(int scan_index, bool measurement_iter);
    void restore_beammap_model_for_ptc_scan(int scan_index, bool measurement_iter);
    void remove_bad_beammap_dets_for_scan(int scan_index, bool locator_iter,
                                          bool detector_grouping);
    void apply_beammap_ptc_scan_weights(int scan_index, bool measurement_iter,
                                        bool detector_grouping);
    void run_beammap_ptc_cleaning_pass(bool locator_iter,
                                       bool measurement_iter,
                                       bool detector_grouping);
    void populate_beammap_maps(
        citlali::config::MapGrouping mapmaking_grouping,
        citlali::config::MapMethod mapmaking_method,
        const Eigen::Matrix<bool, Eigen::Dynamic, 1> *active_maps,
        bool update_progress);
    void normalize_beammap_maps_after_pass(
        const Eigen::Matrix<bool, Eigen::Dynamic, 1> *active_maps,
        const std::string &profile_context);
    void prepare_beammap_iteration_state(bool rerun_source_aware_rtc,
                                         bool measurement_iter,
                                         bool first_measurement_iter,
                                         bool detector_grouping);

    // robust sample-level masking for short RFI bursts in detector beammaps
    RFIMaskScanSummary apply_rfi_sample_mask(TCData<TCDataKind::PTC,Eigen::MatrixXd> &);

    // detector-map edge-band masking for coherent bad scan legs
    ScanBandMaskSummary apply_scan_band_mask(mapmaking::MapBuffer &);
    ScanBandRowStats calculate_scan_band_row_stats(
        const Eigen::MatrixXd &signal,
        const Eigen::MatrixXd &weight,
        Eigen::Index n_rows,
        Eigen::Index n_cols,
        Eigen::Index search_rows,
        Eigen::Index min_row_pixels);
    std::vector<Eigen::Index> collect_scan_band_edge_rows(
        const ScanBandRowStats &row_stats,
        bool from_top,
        Eigen::Index n_rows,
        Eigen::Index search_rows,
        Eigen::Index min_row_pixels,
        Eigen::Index min_contiguous_rows,
        double interior_median,
        double interior_median_sigma,
        double interior_row_sigma_median,
        double median_sigma_threshold,
        double sigma_ratio_threshold,
        double eps);
    ScanBandEdgeRows select_scan_band_edge_rows(
        const ScanBandRowStats &row_stats,
        Eigen::Index n_rows,
        Eigen::Index search_rows,
        Eigen::Index min_row_pixels,
        Eigen::Index min_contiguous_rows,
        double median_sigma_threshold,
        double sigma_ratio_threshold,
        double eps);
    std::vector<unsigned char> make_scan_band_bad_row_mask(
        const ScanBandEdgeRows &edge_rows,
        Eigen::Index n_rows,
        Eigen::Index &n_bad_rows);
    ScanBandProposedFlags collect_scan_band_proposed_flags(
        Eigen::Index det,
        const mapmaking::MapBuffer &map_buffer,
        const std::vector<unsigned char> &bad_row_mask,
        double row0);
    bool reject_scan_band_mask_candidate(
        Eigen::Index det,
        Eigen::Index n_bad_rows,
        std::size_t n_proposed_flags,
        double flagged_fraction,
        double max_flagged_fraction,
        ScanBandMaskSummary &summary);
    void apply_scan_band_mask_flags(
        Eigen::Index det,
        const std::vector<std::pair<Eigen::Index, Eigen::Index>> &proposed_flags);
    void record_scan_band_mask_success(
        Eigen::Index det,
        Eigen::Index n_bad_rows,
        const std::vector<std::pair<Eigen::Index, Eigen::Index>> &proposed_flags,
        const std::vector<Eigen::Index> &top_rows,
        const std::vector<Eigen::Index> &bottom_rows,
        double flagged_fraction,
        ScanBandMaskSummary &summary);
    void log_beammap_masking_config();

    // optional prior-assisted peak initialization
    bool is_beammap_locator_iter(Eigen::Index iter) const;
    bool is_beammap_measurement_iter(Eigen::Index iter) const;
    bool is_beammap_first_measurement_iter(Eigen::Index iter) const;
    bool has_completed_beammap_measurement_iter(Eigen::Index iter) const;
    std::string beammap_iter_phase_name(Eigen::Index iter) const;
    std::filesystem::path resolve_soft_priors_filepath() const;
    bool load_soft_priors();
    bool find_map_weighted_peak(Eigen::Index map_index, Eigen::Index &best_row,
                                Eigen::Index &best_col, double &best_snr) const;
    double get_prior_derot_elev_rad() const;
    double effective_prior_max_d2() const;
    double effective_prior_score_lambda() const;
    bool observed_to_prior_frame(int array, double x_raw_arcsec, double y_raw_arcsec,
                                 double derot_elev_rad, double &x_prior_arcsec,
                                 double &y_prior_arcsec, double *center_x_arcsec = nullptr,
                                 double *center_y_arcsec = nullptr,
                                 bool apply_empirical_alignment = true) const;
    bool match_prior_slot(int array, int nw, double x_prior_arcsec, double y_prior_arcsec,
                          double &best_d2, int &best_slot, double *slot_x_arcsec = nullptr,
                          double *slot_y_arcsec = nullptr, double *slot_sx_arcsec = nullptr,
                          double *slot_sy_arcsec = nullptr) const;
    void reset_beammap_prior_frame_estimates();
    BeammapPriorFrameCenterSamples collect_beammap_prior_frame_center_samples();
    void apply_beammap_prior_frame_center_samples(
        const BeammapPriorFrameCenterSamples &center_samples);
    bool should_update_beammap_prior_alignment(
        const citlali::config::BeammapPriorsConfig &priors_config) const;
    bool is_beammap_prior_alignment_sample_candidate(
        Eigen::Index map_index);
    bool make_beammap_prior_alignment_pair(
        Eigen::Index map_index,
        const citlali::config::BeammapPriorsConfig &priors_config,
        double derot_elev_rad,
        int &array,
        BeammapPriorAlignmentPair &pair);
    BeammapPriorAlignmentSamples collect_beammap_prior_alignment_samples(
        const citlali::config::BeammapPriorsConfig &priors_config);
    std::pair<double, double> median_beammap_prior_alignment_translation(
        const std::vector<BeammapPriorAlignmentPair> &pairs) const;
    double fit_beammap_prior_alignment_rotation(
        const std::vector<BeammapPriorAlignmentPair> &pairs,
        const citlali::config::BeammapPriorsConfig &priors_config,
        const std::string &label,
        double tx,
        double ty) const;
    std::pair<double, double> median_beammap_prior_alignment_translation_after_rotation(
        const std::vector<BeammapPriorAlignmentPair> &pairs,
        double cos_theta,
        double sin_theta) const;
    double beammap_prior_alignment_rms(
        const std::vector<BeammapPriorAlignmentPair> &pairs,
        double cos_theta,
        double sin_theta,
        double tx,
        double ty) const;
    bool fit_beammap_prior_alignment(
        const std::vector<BeammapPriorAlignmentPair> &pairs,
        const citlali::config::BeammapPriorsConfig &priors_config,
        const std::string &label,
        PriorArrayAlignment &alignment);
    BeammapPriorAlignmentOverlapBox beammap_prior_alignment_overlap_box(
        const BeammapPriorAlignmentSamples &alignment_samples,
        const citlali::config::BeammapPriorsConfig &priors_config) const;
    std::vector<BeammapPriorAlignmentPair> filter_beammap_prior_alignment_pairs_to_overlap_box(
        const BeammapPriorAlignmentSamples &alignment_samples,
        const BeammapPriorAlignmentOverlapBox &overlap_box) const;
    std::vector<BeammapPriorAlignmentPair> select_common_beammap_prior_alignment_pairs(
        const BeammapPriorAlignmentSamples &alignment_samples,
        const citlali::config::BeammapPriorsConfig &priors_config);
    void apply_beammap_prior_alignment_samples(
        const BeammapPriorAlignmentSamples &alignment_samples,
        const citlali::config::BeammapPriorsConfig &priors_config);
    Eigen::Index update_beammap_prior_alignment_estimates(
        const citlali::config::BeammapPriorsConfig &priors_config);
    void log_beammap_prior_frame_estimate_summary(
        const BeammapPriorFrameCenterSamples &center_samples,
        Eigen::Index n_alignment_matches) const;
    void update_prior_frame_estimates();
    bool choose_prior_guided_init(Eigen::Index map_index, double &init_row, double &init_col);
    void clear_beammap_detector_source_centers();
    bool has_complete_beammap_detector_source_center_state() const;
    void configure_detector_source_centers_from_previous_fit();
    double calc_map_support_stddev(Eigen::Index map_index, bool exclude_fit_core = false) const;
    double calc_beammap_convergence_delta(Eigen::Index map_index) const;
    void init_empirical_template_calibration_columns();
    void reset_empirical_template_calibration_columns();
    void seed_empirical_template_gaussian_fallback(Eigen::Index n_fallback);
    BeammapEmpiricalTemplateGeometry make_empirical_template_geometry(
        double pix_to_arcsec) const;
    bool empirical_template_inputs_available() const;
    bool extract_empirical_template_normalized_cut(
        Eigen::Index map_index,
        const BeammapEmpiricalTemplateGeometry &geometry,
        Eigen::MatrixXd &cut,
        double &peak_amp);
    bool is_empirical_template_library_detector(
        Eigen::Index map_index,
        int array);
    BeammapEmpiricalTemplateShapeMedians empirical_template_shape_medians(
        int array);
    std::vector<BeammapEmpiricalTemplateCandidate> collect_empirical_template_candidates(
        int array,
        const BeammapEmpiricalTemplateShapeMedians &shape_medians);
    std::vector<Eigen::MatrixXd> collect_empirical_template_cuts(
        const std::vector<BeammapEmpiricalTemplateCandidate> &candidates,
        const BeammapEmpiricalTemplateGeometry &geometry);
    Eigen::MatrixXd median_empirical_template_shape(
        const std::vector<Eigen::MatrixXd> &cuts,
        const BeammapEmpiricalTemplateGeometry &geometry) const;
    double empirical_template_peak_value(
        const Eigen::MatrixXd &templ,
        const BeammapEmpiricalTemplateGeometry &geometry) const;
    std::map<int, BeammapArrayTemplate> build_empirical_template_library(
        const BeammapEmpiricalTemplateGeometry &geometry);
    void record_empirical_template_peak(
        Eigen::Index map_index,
        double row0,
        double col0,
        double baseline,
        const BeammapEmpiricalTemplateGeometry &geometry);
    BeammapTemplateFitSamples collect_empirical_template_fit_samples(
        Eigen::Index map_index,
        const Eigen::MatrixXd &templ,
        double row0,
        double col0,
        double baseline,
        const BeammapEmpiricalTemplateGeometry &geometry);
    double empirical_template_weight_cap(
        const std::vector<double> &weights) const;
    bool solve_empirical_template_linear_fit(
        const BeammapTemplateFitSamples &samples,
        double weight_cap,
        BeammapTemplateFitResult &fit_result) const;
    double empirical_template_resid_rms(
        const BeammapTemplateFitSamples &samples,
        double weight_cap,
        const BeammapTemplateFitResult &fit_result) const;
    bool solve_empirical_template(
        Eigen::Index map_index,
        const Eigen::MatrixXd &templ,
        const BeammapEmpiricalTemplateGeometry &geometry,
        BeammapTemplateFitResult &fit_result);
    double seed_empirical_template_detector_calibration(
        Eigen::Index map_index);
    void record_empirical_template_fit_result(
        Eigen::Index map_index,
        double fit_amp,
        const BeammapTemplateFitResult &fit_result);
    bool apply_empirical_template_detector_calibration(
        Eigen::Index map_index,
        const std::map<int, BeammapArrayTemplate> &templates,
        const BeammapEmpiricalTemplateGeometry &geometry);
    void apply_empirical_template_calibration(
        const std::map<int, BeammapArrayTemplate> &templates,
        const BeammapEmpiricalTemplateGeometry &geometry);
    void calc_empirical_template_calibration();
    void mark_beammap_detector_flagged(
        Eigen::Index detector_index,
        AptFlags flag,
        std::atomic<int> &n_flagged_dets);
    bool beammap_fit_quality_values_valid(
        Eigen::Index detector_index,
        double map_std_dev);
    void update_beammap_fit_sig2noise(Eigen::Index detector_index);
    bool beammap_az_fwhm_outlier(
        Eigen::Index detector_index,
        const citlali::pipeline::BeammapArrayFlaggingLimits &flag_limits,
        const std::string &array_name);
    bool beammap_el_fwhm_outlier(
        Eigen::Index detector_index,
        const citlali::pipeline::BeammapArrayFlaggingLimits &flag_limits,
        const std::string &array_name);
    bool beammap_map_sig2noise_outlier(
        double map_sig2noise,
        const citlali::pipeline::BeammapArrayFlaggingLimits &flag_limits,
        const std::string &array_name);
    void flag_beammap_fit_quality_detector(
        Eigen::Index detector_index,
        const citlali::pipeline::BeammapArrayFlaggingLimits &flag_limits,
        std::atomic<int> &n_flagged_dets);
    void flag_beammap_fit_quality_outliers(
        const citlali::pipeline::BeammapArrayFlaggingLimits &flag_limits,
        const std::string &runtime_parallel_policy,
        std::atomic<int> &n_flagged_dets);
    std::map<Eigen::Index, double> beammap_network_median_sensitivities();
    void flag_beammap_sensitivity_outliers(
        std::map<Eigen::Index, double> &nw_median_sens,
        double lower_sens_factor,
        double upper_sens_factor,
        const std::string &runtime_parallel_policy,
        std::atomic<int> &n_flagged_dets);
    BeammapArrayPositionMedians beammap_array_position_medians();
    void flag_beammap_position_outliers(
        const citlali::pipeline::BeammapArrayFlaggingLimits &flag_limits,
        BeammapArrayPositionMedians &array_position_medians,
        const std::string &runtime_parallel_policy,
        std::atomic<int> &n_flagged_dets);
    BeammapPriorDistanceFrame beammap_prior_distance_frame();
    bool beammap_soft_prior_slot_valid(const SoftPriorSlot &slot) const;
    double beammap_detector_prior_distance2(
        Eigen::Index detector_index,
        const BeammapArrayPositionMedians &array_position_medians,
        const BeammapPriorDistanceFrame &frame);
    void flag_beammap_prior_distance_detector(
        Eigen::Index detector_index,
        double max_prior_d2,
        const BeammapArrayPositionMedians &array_position_medians,
        const BeammapPriorDistanceFrame &frame,
        std::atomic<int> &n_prior_dist_hits,
        std::atomic<int> &n_flagged_dets);
    void flag_beammap_prior_distance_outliers(
        double max_prior_d2,
        const BeammapArrayPositionMedians &array_position_medians,
        const std::string &runtime_parallel_policy,
        std::atomic<int> &n_flagged_dets);
    double beammap_detector_flux_calibration_amp(Eigen::Index detector_index);
    void clear_beammap_detector_flux_conversion(Eigen::Index detector_index);
    void reject_beammap_detector_flux_conversion(Eigen::Index detector_index);
    void calculate_beammap_detector_flux_conversion(Eigen::Index detector_index);
    void update_beammap_array_source_flux_density();
    void calculate_beammap_flux_conversion_factors(
        const std::string &runtime_parallel_policy);

    // flag detectors
    void set_apt_flags();

    // derotate apt and subtract reference detector
    void process_apt();
    bool gather_beammap_reference_candidates(
        const std::vector<Eigen::Index> &ref_nws,
        Eigen::VectorXd &x_t,
        Eigen::VectorXd &y_t,
        Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1> &det_indices);
    bool gather_all_unflagged_beammap_reference_candidates(
        Eigen::VectorXd &x_t,
        Eigen::VectorXd &y_t,
        Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1> &det_indices);
    void resolve_automatic_beammap_reference_detector(
        double &ref_det_x_t, double &ref_det_y_t);
    void select_beammap_reference_detector(
        double &ref_det_x_t, double &ref_det_y_t);
    void record_beammap_reference_metadata(
        double ref_det_x_t, double ref_det_y_t);
    void preserve_beammap_raw_detector_offsets();
    void populate_beammap_derotation_elevation();
    void apply_beammap_reference_offsets(
        double ref_det_x_t, double ref_det_y_t);
    void apply_beammap_derotation(bool derotate);
    void apply_final_network_position_flags();
    void update_final_prior_match_diagnostics();
    void log_final_network_qc_summary();
    void clear_beammap_ptc_diagnostics();
    void write_beammap_ptc_products(int output_iter);
    void write_beammap_ptc_chunk_summaries(int output_iter);
    void write_beammap_ptc_diag_sidecar(int output_iter);
    void write_beammap_processed_ptc_tod(int output_iter);
    void write_beammap_detector_ptc_tod_stage(int output_iter);
    BeammapDetectorTodPreflight prepare_detector_specific_ptc_tod_output();
    BeammapDetectorTodPointingSamples sample_detector_tod_pointing(
        Eigen::Index n_scans);
    BeammapDetectorTodSelections make_detector_tod_selections(
        const BeammapDetectorTodPreflight &preflight,
        BeammapDetectorTodPointingSamples &pointing_samples,
        const std::vector<Eigen::Index> &uniform_scans);
    void write_detector_specific_ptc_tod_file(
        const std::string &filename,
        int output_iter,
        const BeammapDetectorTodPreflight &preflight,
        const BeammapDetectorTodSelections &selections);
    void write_detector_specific_ptc_tod(int output_iter);
    void write_detector_table_outputs();
    void write_beammap_fit_qc_table(const std::string &apt_filename);
    // main pipeline process
    template <class KidsProc, class RawObs>
    void pipeline(KidsProc &, RawObs &);

    // output files
    std::string write_beammap_apt_table();
    template <mapmaking::MapType map_type>
    void output();
    void add_beammap_detector_map_header(
        std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *f_io,
        Eigen::Index detector_index,
        Eigen::Index signal_hdu_index,
        const char *breadcrumb,
        int flag_value = -1);
    template <mapmaking::MapType map_type>
    void maybe_add_beammap_detector_map_header(
        std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *f_io,
        Eigen::Index detector_index,
        Eigen::Index signal_hdu_index,
        bool detector_grouping,
        const char *breadcrumb,
        int flag_value = -1);
    void add_beammap_map_primary_headers(
        mapmaking::MapBuffer *mb,
        std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *f_io,
        std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *n_io,
        const std::string &profile_stage_name,
        const std::string &profile_context,
        int flag_value = -1);
    BeammapSplitMapOutputFiles prepare_split_beammap_map_output_files(
        std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *f_io,
        std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *n_io);
    template <mapmaking::MapType map_type>
    void write_standard_beammap_map_entries(
        mapmaking::MapBuffer *mb,
        std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *f_io,
        std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *n_io,
        const std::string &dir_name,
        bool detector_grouping);
    template <mapmaking::MapType map_type>
    void write_split_beammap_flag_maps(
        mapmaking::MapBuffer *mb,
        std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *split_f_io,
        std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *split_n_io,
        const std::string &dir_name,
        bool detector_grouping,
        int flag_value,
        Eigen::Index n_flag_maps);
    template <mapmaking::MapType map_type>
    bool should_split_beammap_maps_by_flag(
        bool detector_grouping,
        const citlali::config::BeammapSplitFitsByFlagConfig &split_config);
    template <mapmaking::MapType map_type>
    void write_beammap_non_detector_map_diagnostics(
        mapmaking::MapBuffer *mb,
        const std::string &dir_name,
        bool detector_grouping);
    template <mapmaking::MapType map_type>
    void write_standard_beammap_map_products(
        mapmaking::MapBuffer *mb,
        std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *f_io,
        std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *n_io,
        const std::string &dir_name,
        bool detector_grouping);
    template <mapmaking::MapType map_type>
    void write_split_beammap_map_products(
        mapmaking::MapBuffer *mb,
        std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *f_io,
        std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *n_io,
        const std::string &dir_name,
        bool detector_grouping,
        const std::vector<int> &flag_values);
    template <mapmaking::MapType map_type>
    void write_beammap_map_products(
        mapmaking::MapBuffer *mb,
        std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *f_io,
        std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *n_io,
        const std::string &dir_name);
};

#include <citlali/core/engine/detail/beammap_detector_tod_output_impl.h>
#include <citlali/core/engine/detail/beammap_ptc_product_output_impl.h>
#include <citlali/core/engine/detail/beammap_apt_table_output_impl.h>
#include <citlali/core/engine/detail/beammap_detector_table_output_impl.h>
#include <citlali/core/engine/detail/beammap_map_product_output_impl.h>
#include <citlali/core/engine/detail/beammap_empirical_template_calibration_impl.h>
#include <citlali/core/engine/detail/beammap_masking_impl.h>
#include <citlali/core/engine/detail/beammap_pipeline_impl.h>
#include <citlali/core/engine/detail/beammap_prior_impl.h>
#include <citlali/core/engine/detail/beammap_run_loop_impl.h>
#include <citlali/core/engine/detail/beammap_finalization_impl.h>
#include <citlali/core/engine/detail/beammap_convergence_impl.h>
#include <citlali/core/engine/detail/beammap_output_impl.h>
