#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <complex>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <unsupported/Eigen/FFT>

#include <tula/logging.h>
#include <tula/nc.h>
#include <tula/algorithm/ei_stats.h>

#include <citlali/core/engine/io.h>
#include <citlali/core/utils/utils.h>
#include <citlali/core/utils/pointing.h>

#include <citlali/core/timestream/timestream.h>
#include <citlali/core/timestream/ptc/clean.h>
#include <citlali/core/timestream/rtc/despike.h>

#include <citlali/core/utils/toltec_io.h>

namespace timestream {

using timestream::TCData;

class PTCProc: public TCProc {
public:
    // controls for timestream reduction
    bool run_clean;
    // median weight factor
    double med_weight_factor;
    // upper and lower weight limits for outliers
    double lower_weight_factor, upper_weight_factor;
    // weight type (full, approximate, const)
    std::string weighting_type;

    // ptc tod proc
    timestream::Cleaner cleaner;

    struct CorrNWDiagSummary {
        Eigen::Index nw = -1;
        Eigen::Index n_det_input = 0;
        Eigen::Index n_det_candidates = 0;
        Eigen::Index n_det_used = 0;
        Eigen::Index n_det_grouped = 0;
        Eigen::Index n_det_ungrouped = 0;
        Eigen::Index n_groups_raw = 0;
        Eigen::Index n_groups_final = 0;
        Eigen::Index sample_step = 1;
    };

    struct WeightCorrPenaltyTermOptions {
        bool enabled = true;
        double ref = 0.05;
        double span = 0.15;
        double weight = 1.0;
    };

    struct WeightCorrPenaltyBandOptions {
        bool enabled = false;
        double ref = 0.6;
        double span = 2.0;
        double weight = 0.5;
        double low_min_Hz = 0.05;
        double low_max_Hz = 0.5;
        double mid_min_Hz = 0.5;
        double mid_max_Hz = 2.0;
    };

    struct WeightCorrPenaltyOptions {
        bool enabled = false;
        double min_good_frac = 0.7;
        int min_overlap = 200;
        int max_samples = 20000;
        int max_pairs = 4000;
        std::uint32_t seed = 12345;
        double floor = 0.05;
        double exponent = 2.0;
        WeightCorrPenaltyTermOptions pair_corr;
        WeightCorrPenaltyTermOptions cm_el_corr{false, 0.05, 0.25, 0.5};
        WeightCorrPenaltyBandOptions cm_low_mid_ratio;
    };

    struct WeightCorrPenaltyDiagSummary {
        Eigen::Index nw = -1;
        Eigen::Index n_det_input = 0;
        Eigen::Index n_det_candidates = 0;
        Eigen::Index n_det_used = 0;
        Eigen::Index n_det_weighted = 0;
        Eigen::Index sample_step = 1;
        double pair_med_abs_corr = std::numeric_limits<double>::quiet_NaN();
        double cm_el_abs_corr = std::numeric_limits<double>::quiet_NaN();
        double cm_low_mid_ratio = std::numeric_limits<double>::quiet_NaN();
        double severity = 0.0;
        double penalty_factor = 1.0;
    };

    struct BusyRowSuppressionOptions {
        bool enabled = false;
        bool require_busy_veto = true;
        int min_candidate_clusters = 5;
        double min_max_unflagged_residual_z = 25.0;
        double factor = 0.0;
    };

    struct BusyRowSuppressionDiagSummary {
        Eigen::Index nw = -1;
        Eigen::Index n_det_weighted = 0;
        Eigen::Index n_candidate_clusters = 0;
        bool busy_network_vetoed = false;
        bool applied = false;
        double max_unflagged_residual_z = std::numeric_limits<double>::quiet_NaN();
        double factor = 1.0;
    };

    WeightCorrPenaltyOptions weight_corr_penalty;
    BusyRowSuppressionOptions busy_row_suppression;
    std::map<Eigen::Index, Eigen::VectorXi> corr_nw_group_ids_by_scan;
    std::map<Eigen::Index, std::vector<CorrNWDiagSummary>> corr_nw_summary_by_scan;
    std::map<Eigen::Index, std::vector<WeightCorrPenaltyDiagSummary>> weight_corr_penalty_summary_by_scan;
    std::map<Eigen::Index, std::vector<BusyRowSuppressionDiagSummary>> busy_row_suppression_summary_by_scan;

    struct AdaptiveSelectorDiagSummary {
        Eigen::Index nw = -1;
        Eigen::Index n_det_input = 0;
        Eigen::Index n_det_used = 0;
        Eigen::Index n_time_used = 0;
        Eigen::Index sample_step = 1;
        Eigen::Index baseline_k = 0;
        Eigen::Index chosen_k = 0;
        Eigen::Index runnerup_k = -1;
        Eigen::Index n_candidates = 0;
        int selector_used = 0;
        int selector_fallback = 0;
        double chosen_score = std::numeric_limits<double>::quiet_NaN();
        double runnerup_score = std::numeric_limits<double>::quiet_NaN();
        double score_margin = std::numeric_limits<double>::quiet_NaN();
        double chosen_med_abs_corr = std::numeric_limits<double>::quiet_NaN();
        double chosen_cm_low_mid_ratio = std::numeric_limits<double>::quiet_NaN();
        double chosen_tail4_binom_z = std::numeric_limits<double>::quiet_NaN();
        double chosen_top_mode_frac = std::numeric_limits<double>::quiet_NaN();
        double eig_solve_msec = std::numeric_limits<double>::quiet_NaN();
        double candidate_eval_msec = std::numeric_limits<double>::quiet_NaN();
        double total_msec = std::numeric_limits<double>::quiet_NaN();
    };

    std::map<Eigen::Index, std::vector<AdaptiveSelectorDiagSummary>> adaptive_selector_summary_by_scan;

    struct SecondPassLocalOptions {
        bool enabled = false;
        double min_spike_sigma = 8.0;
        double min_good_frac = 0.5;
        double baseline_window_sec = 0.25;
        double sigma_scale = 0.75;
        double delta_sigma_scale = 0.75;
        double raw_candidate_rel_sigma_scale = 1.0;
        double raw_window_sec = 0.18;
        double raw_half_peak_frac = 0.5;
        double raw_max_width_sec = 0.18;
        double delta_window_sec = 0.12;
        double delta_half_peak_frac = 0.5;
        double delta_max_width_sec = 0.10;
        double max_step_shift_z = 3.0;
        double merge_within_detector_sec = 0.08;
        double cluster_events_sec = 0.08;
        int min_cluster_detectors = 3;
        double high_score_cluster_override = 9.0;
        int max_auto_flag_clusters_per_network = 3;
    };

    struct SecondPassDiagSummary {
        Eigen::Index nw = -1;
        Eigen::Index n_det = 0;
        Eigen::Index n_pts = 0;
        Eigen::Index n_merged_events_total = 0;
        Eigen::Index n_clusters_total = 0;
        Eigen::Index n_candidate_events = 0;
        Eigen::Index n_candidate_clusters = 0;
        Eigen::Index n_accepted_events = 0;
        Eigen::Index n_accepted_clusters = 0;
        Eigen::Index n_det_with_added_flags = 0;
        bool busy_network_vetoed = false;
        double existing_flagged_fraction = std::numeric_limits<double>::quiet_NaN();
        double proposed_flagged_fraction = std::numeric_limits<double>::quiet_NaN();
        double newly_flagged_fraction = std::numeric_limits<double>::quiet_NaN();
        double max_unflagged_residual_z = std::numeric_limits<double>::quiet_NaN();
        int max_unflagged_residual_uid = kTransientFillInt;
        double top_candidate_cluster_peak_score = std::numeric_limits<double>::quiet_NaN();
        Eigen::Index top_candidate_cluster_n_detectors = 0;
        Eigen::Index top_candidate_cluster_n_events = 0;
        int top_candidate_cluster_sample = kTransientFillInt;
        int top_event_uid = kTransientFillInt;
        TransientEvent top_event;
    };

    SecondPassLocalOptions second_pass_local;
    std::map<Eigen::Index, std::vector<SecondPassDiagSummary>> second_pass_summary_by_scan;
    std::map<Eigen::Index, Eigen::Matrix<signed char, Eigen::Dynamic, Eigen::Dynamic>> second_pass_added_flags_by_scan;

    // get config file
    template <typename config_t>
    void get_config(config_t &, std::vector<std::vector<std::string>> &,
                    std::vector<std::vector<std::string>> &);

    // subtract detector means
    void subtract_mean(TCData<TCDataKind::PTC, Eigen::MatrixXd> &,
                       const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> *flags_override = nullptr);

    // run main processing stage
    template <class calib_type>
    void run(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, TCData<TCDataKind::PTC, Eigen::MatrixXd> &,
             calib_type &, std::string, std::string);

    template <class calib_type>
    void apply_second_pass_local(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_type &);

    template <typename calib_t>
    void append_diag_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, std::string, calib_t &,
                               Eigen::Index scan_row_index = -1);

    void clear_cached_diagnostics(Eigen::Index scan_id);

    // calculate detector weights
    template <typename apt_type, class tel_type>
    void calc_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, apt_type &, tel_type &);

    // reset outlier weights to the median
    template <typename calib_t>
    auto reset_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_t &, std::string);

    // append time chunk to tod netcdf file
    template <typename calib_t, typename pointing_offset_t>
    void append_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, std::string, std::string, std::string &,
                          pointing_offset_t &, calib_t &, bool apply_det_offsets = false,
                          Eigen::Index scan_row_index = -1);
};

// get config file
template <typename config_t>
void PTCProc::get_config(config_t &config, std::vector<std::vector<std::string>> &missing_keys,
                         std::vector<std::vector<std::string>> &invalid_keys) {

    // weight type
    get_config_value(config, weighting_type, missing_keys, invalid_keys,
                     std::tuple{"timestream","processed_time_chunk","weighting","type"},{"full","approximate","const"});
    // median weight factor
    get_config_value(config, med_weight_factor, missing_keys, invalid_keys,
                     std::tuple{"timestream","processed_time_chunk","weighting","median_map_weight_factor"});
    // lower inv var factor
    get_config_value(config, lower_inv_var_factor, missing_keys, invalid_keys,
                     std::tuple{"timestream","processed_time_chunk","flagging","lower_tod_inv_var_factor"});
    // upper inv var factor
    get_config_value(config, upper_inv_var_factor, missing_keys, invalid_keys,
                     std::tuple{"timestream","processed_time_chunk","flagging","upper_tod_inv_var_factor"});

    // lower weight factor
    get_config_value(config, lower_weight_factor, missing_keys, invalid_keys,
                     std::tuple{"timestream","processed_time_chunk","weighting","lower_map_weight_factor"});
    // upper weight factor
    get_config_value(config, upper_weight_factor, missing_keys, invalid_keys,
                     std::tuple{"timestream","processed_time_chunk","weighting","upper_map_weight_factor"});

    second_pass_local = SecondPassLocalOptions{};
    if (config.template has_typed<bool>(
            std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","enabled"})) {
        get_config_value(config, second_pass_local.enabled, missing_keys, invalid_keys,
                         std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","enabled"});
    }
    if (second_pass_local.enabled) {
        if (config.template has_typed<double>(
                std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","min_spike_sigma"})) {
            get_config_value(config, second_pass_local.min_spike_sigma, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","min_spike_sigma"},
                             {}, {0.0});
        }
        if (config.template has_typed<double>(
                std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","min_good_frac"})) {
            get_config_value(config, second_pass_local.min_good_frac, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","min_good_frac"},
                             {}, {0.0}, {1.0});
        }
        if (config.template has_typed<double>(
                std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","baseline_window_sec"})) {
            get_config_value(config, second_pass_local.baseline_window_sec, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","baseline_window_sec"},
                             {}, {0.0});
        }
        if (config.template has_typed<double>(
                std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","sigma_scale"})) {
            get_config_value(config, second_pass_local.sigma_scale, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","sigma_scale"},
                             {}, {0.0});
        }
        if (config.template has_typed<double>(
                std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","delta_sigma_scale"})) {
            get_config_value(config, second_pass_local.delta_sigma_scale, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","delta_sigma_scale"},
                             {}, {0.0});
        }
        if (config.template has_typed<double>(
                std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","raw_candidate_rel_sigma_scale"})) {
            get_config_value(config, second_pass_local.raw_candidate_rel_sigma_scale, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","raw_candidate_rel_sigma_scale"},
                             {}, {0.0});
        }
        if (config.template has_typed<double>(
                std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","raw_window_sec"})) {
            get_config_value(config, second_pass_local.raw_window_sec, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","raw_window_sec"},
                             {}, {0.0});
        }
        if (config.template has_typed<double>(
                std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","raw_half_peak_frac"})) {
            get_config_value(config, second_pass_local.raw_half_peak_frac, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","raw_half_peak_frac"},
                             {}, {0.0});
        }
        if (config.template has_typed<double>(
                std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","raw_max_width_sec"})) {
            get_config_value(config, second_pass_local.raw_max_width_sec, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","raw_max_width_sec"},
                             {}, {0.0});
        }
        if (config.template has_typed<double>(
                std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","delta_window_sec"})) {
            get_config_value(config, second_pass_local.delta_window_sec, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","delta_window_sec"},
                             {}, {0.0});
        }
        if (config.template has_typed<double>(
                std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","delta_half_peak_frac"})) {
            get_config_value(config, second_pass_local.delta_half_peak_frac, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","delta_half_peak_frac"},
                             {}, {0.0});
        }
        if (config.template has_typed<double>(
                std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","delta_max_width_sec"})) {
            get_config_value(config, second_pass_local.delta_max_width_sec, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","delta_max_width_sec"},
                             {}, {0.0});
        }
        if (config.template has_typed<double>(
                std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","max_step_shift_z"})) {
            get_config_value(config, second_pass_local.max_step_shift_z, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","max_step_shift_z"},
                             {}, {0.0});
        }
        if (config.template has_typed<double>(
                std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","merge_within_detector_sec"})) {
            get_config_value(config, second_pass_local.merge_within_detector_sec, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","merge_within_detector_sec"},
                             {}, {0.0});
        }
        if (config.template has_typed<double>(
                std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","cluster_events_sec"})) {
            get_config_value(config, second_pass_local.cluster_events_sec, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","cluster_events_sec"},
                             {}, {0.0});
        }
        if (config.template has_typed<int>(
                std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","min_cluster_detectors"})) {
            get_config_value(config, second_pass_local.min_cluster_detectors, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","min_cluster_detectors"},
                             {}, {1});
        }
        if (config.template has_typed<double>(
                std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","high_score_cluster_override"})) {
            get_config_value(config, second_pass_local.high_score_cluster_override, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","high_score_cluster_override"},
                             {}, {0.0});
        }
        if (config.template has_typed<int>(
                std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","max_auto_flag_clusters_per_network"})) {
            get_config_value(config, second_pass_local.max_auto_flag_clusters_per_network, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","flagging","second_pass_local","max_auto_flag_clusters_per_network"},
                             {}, {1});
        }
    }

    // optional hard suppression of pathological busy scan/network rows using second-pass diagnostics
    busy_row_suppression = BusyRowSuppressionOptions{};
    if (config.template has_typed<bool>(std::tuple{"timestream","processed_time_chunk","weighting","busy_row_suppression","enabled"})) {
        get_config_value(config, busy_row_suppression.enabled, missing_keys, invalid_keys,
                         std::tuple{"timestream","processed_time_chunk","weighting","busy_row_suppression","enabled"});
    }
    if (busy_row_suppression.enabled) {
        if (config.template has_typed<bool>(std::tuple{"timestream","processed_time_chunk","weighting","busy_row_suppression","require_busy_veto"})) {
            get_config_value(config, busy_row_suppression.require_busy_veto, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","weighting","busy_row_suppression","require_busy_veto"});
        }
        if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","weighting","busy_row_suppression","min_candidate_clusters"})) {
            get_config_value(config, busy_row_suppression.min_candidate_clusters, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","weighting","busy_row_suppression","min_candidate_clusters"},
                             {}, {0});
        }
        if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","weighting","busy_row_suppression","min_max_unflagged_residual_z"})) {
            get_config_value(config, busy_row_suppression.min_max_unflagged_residual_z, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","weighting","busy_row_suppression","min_max_unflagged_residual_z"},
                             {}, {0.0});
        }
        if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","weighting","busy_row_suppression","factor"})) {
            get_config_value(config, busy_row_suppression.factor, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","weighting","busy_row_suppression","factor"},
                             {}, {0.0}, {1.0});
        }
        if (!second_pass_local.enabled) {
            logger->warn("weighting.busy_row_suppression requires flagging.second_pass_local.enabled; disabling busy-row suppression");
            busy_row_suppression.enabled = false;
        } else {
            logger->info(
                "weighting.busy_row_suppression enabled: require_busy_veto={} min_candidate_clusters={} "
                "min_max_unflagged_residual_z={} factor={}",
                busy_row_suppression.require_busy_veto,
                busy_row_suppression.min_candidate_clusters,
                busy_row_suppression.min_max_unflagged_residual_z,
                busy_row_suppression.factor);
        }
    }

    // optional per-network, per-scan correlation-based weight penalty
    weight_corr_penalty = WeightCorrPenaltyOptions{};
    if (config.template has_typed<bool>(std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","enabled"})) {
        get_config_value(config, weight_corr_penalty.enabled, missing_keys, invalid_keys,
                         std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","enabled"});
    }
    if (weight_corr_penalty.enabled) {
        if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","min_good_frac"})) {
            get_config_value(config, weight_corr_penalty.min_good_frac, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","min_good_frac"},
                             {}, {0.0}, {1.0});
        }
        if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","min_overlap"})) {
            get_config_value(config, weight_corr_penalty.min_overlap, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","min_overlap"},
                             {}, {2});
        }
        if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","max_samples"})) {
            get_config_value(config, weight_corr_penalty.max_samples, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","max_samples"},
                             {}, {0});
        }
        if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","max_pairs"})) {
            get_config_value(config, weight_corr_penalty.max_pairs, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","max_pairs"},
                             {}, {0});
        }
        int corr_seed = static_cast<int>(weight_corr_penalty.seed);
        if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","seed"})) {
            get_config_value(config, corr_seed, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","seed"},
                             {}, {0});
        }
        weight_corr_penalty.seed = static_cast<std::uint32_t>(corr_seed);
        if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","floor"})) {
            get_config_value(config, weight_corr_penalty.floor, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","floor"},
                             {}, {0.0}, {1.0});
        }
        if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","exponent"})) {
            get_config_value(config, weight_corr_penalty.exponent, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","exponent"},
                             {}, {0.0});
        }

        if (config.template has_typed<bool>(std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","pair_corr","enabled"})) {
            get_config_value(config, weight_corr_penalty.pair_corr.enabled, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","pair_corr","enabled"});
        }
        if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","pair_corr","ref"})) {
            get_config_value(config, weight_corr_penalty.pair_corr.ref, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","pair_corr","ref"});
        }
        if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","pair_corr","span"})) {
            get_config_value(config, weight_corr_penalty.pair_corr.span, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","pair_corr","span"},
                             {}, {1e-12});
        }
        if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","pair_corr","weight"})) {
            get_config_value(config, weight_corr_penalty.pair_corr.weight, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","pair_corr","weight"},
                             {}, {0.0});
        }

        if (config.template has_typed<bool>(std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","cm_el_corr","enabled"})) {
            get_config_value(config, weight_corr_penalty.cm_el_corr.enabled, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","cm_el_corr","enabled"});
        }
        if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","cm_el_corr","ref"})) {
            get_config_value(config, weight_corr_penalty.cm_el_corr.ref, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","cm_el_corr","ref"});
        }
        if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","cm_el_corr","span"})) {
            get_config_value(config, weight_corr_penalty.cm_el_corr.span, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","cm_el_corr","span"},
                             {}, {1e-12});
        }
        if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","cm_el_corr","weight"})) {
            get_config_value(config, weight_corr_penalty.cm_el_corr.weight, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","cm_el_corr","weight"},
                             {}, {0.0});
        }

        if (config.template has_typed<bool>(std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","cm_low_mid_ratio","enabled"})) {
            get_config_value(config, weight_corr_penalty.cm_low_mid_ratio.enabled, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","cm_low_mid_ratio","enabled"});
        }
        if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","cm_low_mid_ratio","ref"})) {
            get_config_value(config, weight_corr_penalty.cm_low_mid_ratio.ref, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","cm_low_mid_ratio","ref"});
        }
        if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","cm_low_mid_ratio","span"})) {
            get_config_value(config, weight_corr_penalty.cm_low_mid_ratio.span, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","cm_low_mid_ratio","span"},
                             {}, {1e-12});
        }
        if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","cm_low_mid_ratio","weight"})) {
            get_config_value(config, weight_corr_penalty.cm_low_mid_ratio.weight, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","cm_low_mid_ratio","weight"},
                             {}, {0.0});
        }
        if (config.template has_typed<std::vector<double>>(std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","cm_low_mid_ratio","low_band_Hz"})) {
            auto low_band = config.template get_typed<std::vector<double>>(
                std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","cm_low_mid_ratio","low_band_Hz"});
            if (low_band.size() == 2 && low_band[0] >= 0.0 && low_band[1] > low_band[0]) {
                weight_corr_penalty.cm_low_mid_ratio.low_min_Hz = low_band[0];
                weight_corr_penalty.cm_low_mid_ratio.low_max_Hz = low_band[1];
            } else {
                logger->warn("weighting.corr_penalty.cm_low_mid_ratio.low_band_Hz must be [fmin, fmax] with 0<=fmin<fmax");
            }
        }
        if (config.template has_typed<std::vector<double>>(std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","cm_low_mid_ratio","mid_band_Hz"})) {
            auto mid_band = config.template get_typed<std::vector<double>>(
                std::tuple{"timestream","processed_time_chunk","weighting","corr_penalty","cm_low_mid_ratio","mid_band_Hz"});
            if (mid_band.size() == 2 && mid_band[0] >= 0.0 && mid_band[1] > mid_band[0]) {
                weight_corr_penalty.cm_low_mid_ratio.mid_min_Hz = mid_band[0];
                weight_corr_penalty.cm_low_mid_ratio.mid_max_Hz = mid_band[1];
            } else {
                logger->warn("weighting.corr_penalty.cm_low_mid_ratio.mid_band_Hz must be [fmin, fmax] with 0<=fmin<fmax");
            }
        }
        logger->info(
            "weighting.corr_penalty enabled: min_good_frac={} min_overlap={} max_samples={} max_pairs={} floor={} exponent={} "
            "pair(enabled={}, ref={}, span={}, weight={}) cm_el(enabled={}, ref={}, span={}, weight={}) "
            "cm_low_mid(enabled={}, ref={}, span={}, weight={}, low=[{}, {}], mid=[{}, {}])",
            weight_corr_penalty.min_good_frac, weight_corr_penalty.min_overlap,
            weight_corr_penalty.max_samples, weight_corr_penalty.max_pairs,
            weight_corr_penalty.floor, weight_corr_penalty.exponent,
            weight_corr_penalty.pair_corr.enabled, weight_corr_penalty.pair_corr.ref,
            weight_corr_penalty.pair_corr.span, weight_corr_penalty.pair_corr.weight,
            weight_corr_penalty.cm_el_corr.enabled, weight_corr_penalty.cm_el_corr.ref,
            weight_corr_penalty.cm_el_corr.span, weight_corr_penalty.cm_el_corr.weight,
            weight_corr_penalty.cm_low_mid_ratio.enabled, weight_corr_penalty.cm_low_mid_ratio.ref,
            weight_corr_penalty.cm_low_mid_ratio.span, weight_corr_penalty.cm_low_mid_ratio.weight,
            weight_corr_penalty.cm_low_mid_ratio.low_min_Hz, weight_corr_penalty.cm_low_mid_ratio.low_max_Hz,
            weight_corr_penalty.cm_low_mid_ratio.mid_min_Hz, weight_corr_penalty.cm_low_mid_ratio.mid_max_Hz);
    }

    // run fruit loops?
    get_config_value(config, run_fruit_loops, missing_keys, invalid_keys,
                     std::tuple{"timestream","fruit_loops","enabled"});
    fruit_loops_recompute_weights_after_addback = false;

    if (run_fruit_loops) {
        // save all fruit loops iterations?
        get_config_value(config, save_all_iters, missing_keys, invalid_keys,
                         std::tuple{"timestream","fruit_loops","save_all_iters"});
        // fruit looops path
        get_config_value(config, fruit_loops_path, missing_keys, invalid_keys,
                         std::tuple{"timestream","fruit_loops","path"});
        // fruit looops type
        get_config_value(config, fruit_loops_type, missing_keys, invalid_keys,
                         std::tuple{"timestream","fruit_loops","type"});
	// fruit looops mode
        get_config_value(config, fruit_mode, missing_keys, invalid_keys,
                         std::tuple{"timestream","fruit_loops","mode"}, {"upper", "lower", "both"});
        // let user specify "coadd" or "coadded"
        if (fruit_loops_type == "coadded") {
            fruit_loops_type = "coadd";
        }
        // fruit loops signal-to-noise
        get_config_value(config, fruit_loops_sig2noise, missing_keys, invalid_keys,
                         std::tuple{"timestream","fruit_loops", "sig2noise_limit"});
        // fruit loops flux density limit
        auto fruit_loops_flux_vec = config.template get_typed<std::vector<double>>(std::tuple{"timestream","fruit_loops","array_flux_limit"});
        fruit_loops_flux = Eigen::Map<Eigen::VectorXd>(fruit_loops_flux_vec.data(), fruit_loops_flux_vec.size());
        if (config.template has_typed<double>(std::tuple{"timestream","fruit_loops","center_keep_radius_arcsec"})) {
            get_config_value(config, fruit_loops_center_keep_radius_arcsec, missing_keys, invalid_keys,
                             std::tuple{"timestream","fruit_loops","center_keep_radius_arcsec"}, {}, {0.0});
        }
        else {
            fruit_loops_center_keep_radius_arcsec = 0.0;
        }

        if (config.template has_typed<std::string>(std::tuple{"timestream","fruit_loops","interp_mode_override"})) {
            get_config_value(config, fruit_loops_interp_mode_override, missing_keys, invalid_keys,
                             std::tuple{"timestream","fruit_loops","interp_mode_override"},
                             {"auto", "nearest", "bilinear", "jinc", "trunc", "legacy_nearest"});
        }
        else {
            fruit_loops_interp_mode_override = "auto";
        }
        if (fruit_loops_interp_mode_override == "legacy_nearest") {
            fruit_loops_interp_mode_override = "trunc";
        }

        if (config.template has_typed<bool>(std::tuple{"timestream","fruit_loops","legacy_center"})) {
            get_config_value(config, fruit_loops_legacy_center, missing_keys, invalid_keys,
                             std::tuple{"timestream","fruit_loops","legacy_center"});
        }
        else {
            fruit_loops_legacy_center = false;
        }

        if (config.template has_typed<bool>(std::tuple{"timestream","fruit_loops","recompute_weights_after_addback"})) {
            get_config_value(config, fruit_loops_recompute_weights_after_addback, missing_keys, invalid_keys,
                             std::tuple{"timestream","fruit_loops","recompute_weights_after_addback"});
        }
        else {
            fruit_loops_recompute_weights_after_addback = false;
        }

        // maximum fruit loops iterations
        get_config_value(config, fruit_loops_iters, missing_keys, invalid_keys,
                         std::tuple{"timestream","fruit_loops","max_iters"});
    }

    // run clean?
    get_config_value(config, run_clean, missing_keys, invalid_keys,
                     std::tuple{"timestream","processed_time_chunk","clean", "enabled"});

    if (run_clean) {
        // get cleaning grouping vector
        cleaner.grouping = config.template get_typed<std::vector<std::string>>(std::tuple{"timestream","processed_time_chunk","clean","grouping"});
        const bool have_standard_pca_block =
            config.template has_typed<bool>(std::tuple{"timestream","processed_time_chunk","clean","standard_pca","enabled"});
        if (have_standard_pca_block) {
            get_config_value(config, cleaner.standard_pca.enabled, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","clean","standard_pca","enabled"});
        }
        // get cleaning number of eigenvalues vector
        for (auto const& [arr_index, arr_name] : toltec_io.array_name_map) {
            std::vector<Eigen::Index> n_eig_to_cut;
            if (config.template has_typed<std::vector<Eigen::Index>>(
                    std::tuple{"timestream","processed_time_chunk","clean","standard_pca","n_eig_to_cut",arr_name})) {
                n_eig_to_cut = config.template get_typed<std::vector<Eigen::Index>>(
                    std::tuple{"timestream","processed_time_chunk","clean","standard_pca","n_eig_to_cut",arr_name});
            }
            else {
                if (config.template has_typed<std::vector<Eigen::Index>>(
                        std::tuple{"timestream","processed_time_chunk","clean","n_eig_to_cut",arr_name})) {
                    n_eig_to_cut = config.template get_typed<std::vector<Eigen::Index>>(
                        std::tuple{"timestream","processed_time_chunk","clean","n_eig_to_cut",arr_name});
                }
            }
            if (n_eig_to_cut.empty()) {
                logger->warn("clean.n_eig_to_cut.{} is empty; defaulting to 0 for all {} grouping pass(es)",
                             arr_name, cleaner.grouping.size());
                n_eig_to_cut.assign(cleaner.grouping.size(), 0);
            }
            else if (n_eig_to_cut.size() < cleaner.grouping.size()) {
                logger->warn("clean.n_eig_to_cut.{} has {} value(s) but clean.grouping has {} pass(es); padding with last value {}",
                             arr_name, n_eig_to_cut.size(), cleaner.grouping.size(), n_eig_to_cut.back());
                n_eig_to_cut.resize(cleaner.grouping.size(), n_eig_to_cut.back());
            }
            // add eigenvalues to cleaner class
            cleaner.n_eig_to_cut[arr_index] = (Eigen::Map<Eigen::VectorXI>(n_eig_to_cut.data(),n_eig_to_cut.size()));
        }

        // stddev limit
        if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","clean","standard_pca","stddev_limit"})) {
            get_config_value(config, cleaner.stddev_limit, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","clean","standard_pca","stddev_limit"});
        }
        else if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","clean","stddev_limit"})) {
            get_config_value(config, cleaner.stddev_limit, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","clean","stddev_limit"});
        }
        // optional: number of eigenvalues to calculate (0 => full spectrum)
        if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","clean","standard_pca","n_calc"})) {
            get_config_value(config, cleaner.n_calc, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","clean","standard_pca","n_calc"},{},{0});
        }
        else if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","clean","n_calc"})) {
            get_config_value(config, cleaner.n_calc, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","clean","n_calc"},{},{0});
        }
        // optional brute-force null-model mode selection
        if (config.template has_typed<bool>(std::tuple{"timestream","processed_time_chunk","clean","null_model","enabled"})) {
            get_config_value(config, cleaner.null_model.enabled, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","clean","null_model","enabled"});
        }
        if (config.template has_typed<bool>(std::tuple{"timestream","processed_time_chunk","clean","marchenko_pastur","enabled"})) {
            get_config_value(config, cleaner.marchenko_pastur.enabled, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","clean","marchenko_pastur","enabled"});
        }
        if (config.template has_typed<bool>(std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","enabled"})) {
            get_config_value(config, cleaner.adaptive_selector.enabled, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","enabled"});
        }
        if (!have_standard_pca_block) {
            cleaner.standard_pca.enabled =
                !(cleaner.null_model.enabled || cleaner.marchenko_pastur.enabled || cleaner.adaptive_selector.enabled);
        }
        const int n_enabled_cleaners =
            static_cast<int>(cleaner.standard_pca.enabled) +
            static_cast<int>(cleaner.null_model.enabled) +
            static_cast<int>(cleaner.marchenko_pastur.enabled) +
            static_cast<int>(cleaner.adaptive_selector.enabled);
        if (n_enabled_cleaners != 1) {
            logger->error(
                "exactly one cleaner must be enabled when clean.enabled=true; got standard_pca={} null_model={} marchenko_pastur={} adaptive_selector={}",
                cleaner.standard_pca.enabled, cleaner.null_model.enabled,
                cleaner.marchenko_pastur.enabled, cleaner.adaptive_selector.enabled);
            std::exit(EXIT_FAILURE);
        }
        logger->info("clean.active={}", cleaner.active_cleaner_label());
        // optional correlation-defined grouping inside each network
        if (config.template has_typed<bool>(std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","enabled"})) {
            get_config_value(config, cleaner.corr_grouping.enabled, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","enabled"});
        }
        if (cleaner.corr_grouping.enabled) {
            if (config.template has_typed<std::string>(std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","metric"})) {
                get_config_value(config, cleaner.corr_grouping.metric, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","metric"},
                                 {"abs", "signed"});
            }
            if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","corr_min"})) {
                get_config_value(config, cleaner.corr_grouping.corr_min, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","corr_min"},
                                 {}, {0.0}, {1.0});
            }
            if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","min_overlap"})) {
                get_config_value(config, cleaner.corr_grouping.min_overlap, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","min_overlap"},
                                 {}, {1});
            }
            if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","min_good_frac"})) {
                get_config_value(config, cleaner.corr_grouping.min_good_frac, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","min_good_frac"},
                                 {}, {0.0}, {1.0});
            }
            if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","min_group_size"})) {
                get_config_value(config, cleaner.corr_grouping.min_group_size, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","min_group_size"},
                                 {}, {2});
            }
            if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","max_samples"})) {
                get_config_value(config, cleaner.corr_grouping.max_samples, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","max_samples"},
                                 {}, {0});
            }
            if (config.template has_typed<bool>(std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","clean_residual"})) {
                get_config_value(config, cleaner.corr_grouping.clean_residual, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","clean_residual"});
            }
            logger->info("clean.corr_grouping enabled: metric={} corr_min={} min_overlap={} min_good_frac={} min_group_size={} max_samples={} clean_residual={}",
                         cleaner.corr_grouping.metric, cleaner.corr_grouping.corr_min, cleaner.corr_grouping.min_overlap,
                         cleaner.corr_grouping.min_good_frac, cleaner.corr_grouping.min_group_size,
                         cleaner.corr_grouping.max_samples, cleaner.corr_grouping.clean_residual);
        }
        if (cleaner.null_model.enabled) {
            if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","clean","null_model","n_surrogates"})) {
                get_config_value(config, cleaner.null_model.n_surrogates, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","null_model","n_surrogates"},{},{4});
            }
            if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","clean","null_model","quantile"})) {
                get_config_value(config, cleaner.null_model.quantile, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","null_model","quantile"},{},{0.5},{0.999999});
            }
            if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","clean","null_model","min_good_frac"})) {
                get_config_value(config, cleaner.null_model.min_good_frac, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","null_model","min_good_frac"},{},{0.0},{1.0});
            }
            if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","clean","null_model","max_modes"})) {
                get_config_value(config, cleaner.null_model.max_modes, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","null_model","max_modes"},{},{0});
            }
            if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","clean","null_model","max_samples"})) {
                get_config_value(config, cleaner.null_model.max_samples, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","null_model","max_samples"},{},{0});
            }
            int null_seed = static_cast<int>(cleaner.null_model.seed);
            if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","clean","null_model","seed"})) {
                get_config_value(config, null_seed, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","null_model","seed"},{},{0});
            }
            cleaner.null_model.seed = static_cast<std::uint32_t>(null_seed);
            // optional: restrict null-model mode selection to a subset of clean.grouping passes
            cleaner.null_model.grouping.clear();
            if (config.template has_typed<std::vector<std::string>>(
                    std::tuple{"timestream","processed_time_chunk","clean","null_model","grouping"})) {
                auto null_grouping = config.template get_typed<std::vector<std::string>>(
                    std::tuple{"timestream","processed_time_chunk","clean","null_model","grouping"});
                std::unordered_set<std::string> seen;
                for (const auto &g_raw : null_grouping) {
                    auto g = cleaner.normalize_group_name(g_raw);
                    if (g != "all" && g != "array" && g != "nw" && g != "detector" && g != "fg" && g != "corr_nw") {
                        logger->warn("clean.null_model.grouping contains unsupported entry '{}'; ignoring", g_raw);
                        continue;
                    }
                    if (seen.insert(g).second) {
                        cleaner.null_model.grouping.push_back(g);
                    }
                }
            }
            logger->info("clean.null_model enabled: n_surrogates={} quantile={} min_good_frac={} max_modes={} max_samples={} seed={}",
                         cleaner.null_model.n_surrogates, cleaner.null_model.quantile,
                         cleaner.null_model.min_good_frac, cleaner.null_model.max_modes,
                         cleaner.null_model.max_samples, cleaner.null_model.seed);
            if (!cleaner.null_model.grouping.empty()) {
                std::string groups_joined;
                for (std::size_t i = 0; i < cleaner.null_model.grouping.size(); ++i) {
                    if (i > 0) {
                        groups_joined += ",";
                    }
                    groups_joined += cleaner.null_model.grouping[i];
                }
                logger->info("clean.null_model active for grouping(s): {}", groups_joined);
            }
        }
        if (cleaner.marchenko_pastur.enabled) {
            if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","clean","marchenko_pastur","min_good_frac"})) {
                get_config_value(config, cleaner.marchenko_pastur.min_good_frac, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","marchenko_pastur","min_good_frac"},
                                 {}, {0.0}, {1.0});
            }
            if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","clean","marchenko_pastur","max_modes"})) {
                get_config_value(config, cleaner.marchenko_pastur.max_modes, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","marchenko_pastur","max_modes"},
                                 {}, {0});
            }
            if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","clean","marchenko_pastur","max_samples"})) {
                get_config_value(config, cleaner.marchenko_pastur.max_samples, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","marchenko_pastur","max_samples"},
                                 {}, {0});
            }
            if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","clean","marchenko_pastur","band_low_Hz"})) {
                get_config_value(config, cleaner.marchenko_pastur.band_low_Hz, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","marchenko_pastur","band_low_Hz"},
                                 {}, {0.0});
            }
            if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","clean","marchenko_pastur","band_high_Hz"})) {
                get_config_value(config, cleaner.marchenko_pastur.band_high_Hz, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","marchenko_pastur","band_high_Hz"},
                                 {}, {0.0});
            }
            if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","clean","marchenko_pastur","clip_z"})) {
                get_config_value(config, cleaner.marchenko_pastur.clip_z, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","marchenko_pastur","clip_z"});
            }
            if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","clean","marchenko_pastur","bulk_keep_frac"})) {
                get_config_value(config, cleaner.marchenko_pastur.bulk_keep_frac, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","marchenko_pastur","bulk_keep_frac"},
                                 {}, {0.1}, {1.0});
            }
            if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","clean","marchenko_pastur","q_grid_size"})) {
                get_config_value(config, cleaner.marchenko_pastur.q_grid_size, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","marchenko_pastur","q_grid_size"},
                                 {}, {8});
            }
            cleaner.marchenko_pastur.grouping.clear();
            if (config.template has_typed<std::vector<std::string>>(
                    std::tuple{"timestream","processed_time_chunk","clean","marchenko_pastur","grouping"})) {
                auto mp_grouping = config.template get_typed<std::vector<std::string>>(
                    std::tuple{"timestream","processed_time_chunk","clean","marchenko_pastur","grouping"});
                std::unordered_set<std::string> seen;
                for (const auto &g_raw : mp_grouping) {
                    auto g = cleaner.normalize_group_name(g_raw);
                    if (g != "all" && g != "array" && g != "nw" && g != "detector" && g != "fg" && g != "corr_nw") {
                        logger->warn("clean.marchenko_pastur.grouping contains unsupported entry '{}'; ignoring", g_raw);
                        continue;
                    }
                    if (seen.insert(g).second) {
                        cleaner.marchenko_pastur.grouping.push_back(g);
                    }
                }
            }
            logger->info(
                "clean.marchenko_pastur enabled: min_good_frac={} max_modes={} max_samples={} band_low_Hz={} band_high_Hz={} clip_z={} bulk_keep_frac={} q_grid_size={}",
                cleaner.marchenko_pastur.min_good_frac, cleaner.marchenko_pastur.max_modes,
                cleaner.marchenko_pastur.max_samples, cleaner.marchenko_pastur.band_low_Hz,
                cleaner.marchenko_pastur.band_high_Hz, cleaner.marchenko_pastur.clip_z,
                cleaner.marchenko_pastur.bulk_keep_frac, cleaner.marchenko_pastur.q_grid_size);
            if (!cleaner.marchenko_pastur.grouping.empty()) {
                std::string groups_joined;
                for (std::size_t i = 0; i < cleaner.marchenko_pastur.grouping.size(); ++i) {
                    if (i > 0) {
                        groups_joined += ",";
                    }
                    groups_joined += cleaner.marchenko_pastur.grouping[i];
                }
                logger->info("clean.marchenko_pastur active for grouping(s): {}", groups_joined);
            }
        }
        if (cleaner.adaptive_selector.enabled) {
            if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","min_good_frac"})) {
                get_config_value(config, cleaner.adaptive_selector.min_good_frac, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","min_good_frac"},
                                 {}, {0.0}, {1.0});
            }
            if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","max_det"})) {
                get_config_value(config, cleaner.adaptive_selector.max_det, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","max_det"},
                                 {}, {0});
            }
            if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","max_samples"})) {
                get_config_value(config, cleaner.adaptive_selector.max_samples, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","max_samples"},
                                 {}, {0});
            }
            if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","max_pairs"})) {
                get_config_value(config, cleaner.adaptive_selector.max_pairs, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","max_pairs"},
                                 {}, {0});
            }
            int adaptive_seed = static_cast<int>(cleaner.adaptive_selector.seed);
            if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","seed"})) {
                get_config_value(config, adaptive_seed, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","seed"},
                                 {}, {0});
            }
            cleaner.adaptive_selector.seed = static_cast<std::uint32_t>(adaptive_seed);
            if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","clip_z"})) {
                get_config_value(config, cleaner.adaptive_selector.clip_z, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","clip_z"});
            }
            if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","low_weight"})) {
                get_config_value(config, cleaner.adaptive_selector.low_weight, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","low_weight"},
                                 {}, {0.0});
            }
            if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","tail_weight"})) {
                get_config_value(config, cleaner.adaptive_selector.tail_weight, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","tail_weight"},
                                 {}, {0.0});
            }
            if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","topmode_weight"})) {
                get_config_value(config, cleaner.adaptive_selector.topmode_weight, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","topmode_weight"},
                                 {}, {0.0});
            }
            if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","reg_weight"})) {
                get_config_value(config, cleaner.adaptive_selector.reg_weight, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","reg_weight"},
                                 {}, {0.0});
            }
            if (config.template has_typed<bool>(std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","log_candidates"})) {
                get_config_value(config, cleaner.adaptive_selector.log_candidates, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","log_candidates"});
            }
            if (config.template has_typed<std::vector<int>>(std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","candidate_offsets"})) {
                auto offsets = config.template get_typed<std::vector<int>>(
                    std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","candidate_offsets"});
                if (!offsets.empty()) {
                    cleaner.adaptive_selector.candidate_offsets = offsets;
                }
            }
            auto parse_band = [&](const std::vector<double> &band, std::array<double, 2> &dst,
                                  const std::string &name) {
                if (band.size() == 2 && band[0] >= 0.0 && band[1] > band[0]) {
                    dst[0] = band[0];
                    dst[1] = band[1];
                } else {
                    logger->warn("clean.adaptive_selector.{} must be [fmin, fmax] with 0<=fmin<fmax", name);
                }
            };
            if (config.template has_typed<std::vector<double>>(std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","low_band_Hz"})) {
                parse_band(
                    config.template get_typed<std::vector<double>>(
                        std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","low_band_Hz"}),
                    cleaner.adaptive_selector.low_band_Hz, "low_band_Hz");
            }
            if (config.template has_typed<std::vector<double>>(std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","mid_band_Hz"})) {
                parse_band(
                    config.template get_typed<std::vector<double>>(
                        std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","mid_band_Hz"}),
                    cleaner.adaptive_selector.mid_band_Hz, "mid_band_Hz");
            }
            cleaner.adaptive_selector.grouping.clear();
            if (config.template has_typed<std::vector<std::string>>(
                    std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","grouping"})) {
                auto selector_grouping = config.template get_typed<std::vector<std::string>>(
                    std::tuple{"timestream","processed_time_chunk","clean","adaptive_selector","grouping"});
                std::unordered_set<std::string> seen;
                for (const auto &g_raw : selector_grouping) {
                    auto g = cleaner.normalize_group_name(g_raw);
                    if (g != "all" && g != "array" && g != "nw" && g != "detector" && g != "fg" && g != "corr_nw") {
                        logger->warn("clean.adaptive_selector.grouping contains unsupported entry '{}'; ignoring", g_raw);
                        continue;
                    }
                    if (seen.insert(g).second) {
                        cleaner.adaptive_selector.grouping.push_back(g);
                    }
                }
            }
            std::string adaptive_offsets_joined;
            for (std::size_t i = 0; i < cleaner.adaptive_selector.candidate_offsets.size(); ++i) {
                if (i > 0) {
                    adaptive_offsets_joined += ",";
                }
                adaptive_offsets_joined += std::to_string(cleaner.adaptive_selector.candidate_offsets[i]);
            }
            logger->info(
                "clean.adaptive_selector enabled: min_good_frac={} max_det={} max_samples={} max_pairs={} clip_z={} low_weight={} tail_weight={} topmode_weight={} reg_weight={} low_band=[{}, {}] mid_band=[{}, {}] offsets={}",
                cleaner.adaptive_selector.min_good_frac, cleaner.adaptive_selector.max_det,
                cleaner.adaptive_selector.max_samples, cleaner.adaptive_selector.max_pairs,
                cleaner.adaptive_selector.clip_z, cleaner.adaptive_selector.low_weight,
                cleaner.adaptive_selector.tail_weight, cleaner.adaptive_selector.topmode_weight,
                cleaner.adaptive_selector.reg_weight, cleaner.adaptive_selector.low_band_Hz[0],
                cleaner.adaptive_selector.low_band_Hz[1], cleaner.adaptive_selector.mid_band_Hz[0],
                cleaner.adaptive_selector.mid_band_Hz[1],
                adaptive_offsets_joined);
            if (!cleaner.adaptive_selector.grouping.empty()) {
                std::string groups_joined;
                for (std::size_t i = 0; i < cleaner.adaptive_selector.grouping.size(); ++i) {
                    if (i > 0) {
                        groups_joined += ",";
                    }
                    groups_joined += cleaner.adaptive_selector.grouping[i];
                }
                logger->info("clean.adaptive_selector active for grouping(s): {}", groups_joined);
            }
        }
        // mask radius in arcseconds
        get_config_value(config, mask_radius_arcsec, missing_keys, invalid_keys,
                         std::tuple{"timestream","processed_time_chunk","clean","mask_radius_arcsec"});

        // upper weight factor
        get_config_value(config, cleaner.tau, missing_keys, invalid_keys,
                         std::tuple{"timestream","processed_time_chunk","clean","tau"});
    }

    if (second_pass_local.enabled) {
        logger->info(
            "processed_time_chunk.flagging.second_pass_local enabled: min_spike_sigma={} min_good_frac={} baseline_window_sec={} raw_window_sec={} delta_window_sec={} merge_within_detector_sec={} cluster_events_sec={} min_cluster_detectors={} high_score_cluster_override={} max_auto_flag_clusters_per_network={}",
            second_pass_local.min_spike_sigma, second_pass_local.min_good_frac,
            second_pass_local.baseline_window_sec, second_pass_local.raw_window_sec,
            second_pass_local.delta_window_sec, second_pass_local.merge_within_detector_sec,
            second_pass_local.cluster_events_sec, second_pass_local.min_cluster_detectors,
            second_pass_local.high_score_cluster_override,
            second_pass_local.max_auto_flag_clusters_per_network);
    }
}

void PTCProc::subtract_mean(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in,
                            const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> *flags_override) {
    const auto &flags_ref = flags_override ? *flags_override : in.flags.data;
    // cast flags to double and flip 1's and 0's so we can multiply by the data
    auto f = (flags_ref.derived().array().cast <double> ().array() - 1).abs();
    // mean of each detector
    Eigen::RowVectorXd col_mean = (in.scans.data.derived().array()*f).colwise().sum()/
                                   f.colwise().sum();

    // remove nans from completely flagged detectors
    Eigen::RowVectorXd dm = (col_mean).array().isNaN().select(0,col_mean);

    // subtract mean from data and copy into det matrix
    in.scans.data.noalias() = in.scans.data.derived().rowwise() - dm;

    // subtract kernel mean
    if (in.kernel.data.size()!=0) {
        Eigen::RowVectorXd col_mean = (in.kernel.data.derived().array()*f).colwise().sum()/
                                      f.colwise().sum();

        // remove nans from completely flagged detectors
        Eigen::RowVectorXd dm = (col_mean).array().isNaN().select(0,col_mean);

        // subtract mean from data and copy into det matrix
        in.kernel.data.noalias() = in.kernel.data.derived().rowwise() - dm;
    }
}

template <class calib_type>
void PTCProc::run(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, TCData<TCDataKind::PTC, Eigen::MatrixXd> &out,
                  calib_type &calib, std::string pixel_axes, std::string map_grouping) {

    Eigen::Index n_pts = in.scans.data.rows();
    Eigen::Index n_dets = in.scans.data.cols();

    // subtract mean from data and kernel, optionally masking the source region
    if (run_clean && mask_radius_arcsec > 0) {
        auto mean_flags = mask_region(in, calib, pixel_axes, map_grouping, n_pts, n_dets, 0);
        subtract_mean(in, &mean_flags);
    }
    else {
        subtract_mean(in);
    }

    if (run_clean) {
        logger->info("cleaning");
        // Use a local copy so per-pass state does not leak across concurrent run() calls.
        auto cleaner_local = cleaner;
        // number of samples
        Eigen::Index n_pts = in.scans.data.rows();
        // index for number of cleaning groups in vectors
        Eigen::Index indx = 0;
        std::vector<AdaptiveSelectorDiagSummary> adaptive_summary_scan;
        const bool want_eigs = (run_tod_output || write_evals);
        const bool store_eigs = want_eigs && (cleaner_local.n_calc > 0);
        bool warned_eigs = false;

        // loop through config groupings
        const bool null_model_enabled_global = cleaner_local.null_model.enabled;
        const bool marchenko_pastur_enabled_global = cleaner_local.marchenko_pastur.enabled;
        const bool adaptive_selector_enabled_global = cleaner_local.adaptive_selector.enabled;
        for (const auto & group: cleaner_local.grouping) {
            std::string effective_group = group;
            if (group == "corr_nw" && !cleaner_local.corr_grouping.enabled) {
                logger->warn("cleaning group 'corr_nw' requested but clean.corr_grouping.enabled=false; falling back to nw");
                effective_group = "nw";
            }
            // optional per-group null-model gating
            const bool null_model_for_group =
                null_model_enabled_global && cleaner_local.null_model_enabled_for_group(effective_group);
            if (null_model_enabled_global && !null_model_for_group) {
                logger->debug("null_model disabled for {} grouping", effective_group);
            }
            const bool marchenko_pastur_for_group =
                marchenko_pastur_enabled_global && cleaner_local.marchenko_pastur_enabled_for_group(effective_group);
            if (marchenko_pastur_enabled_global && !marchenko_pastur_for_group) {
                logger->debug("marchenko_pastur disabled for {} grouping", effective_group);
            }
            const bool adaptive_selector_for_group =
                adaptive_selector_enabled_global &&
                cleaner_local.adaptive_selector_enabled_for_group(effective_group) &&
                effective_group != "corr_nw";
            if (adaptive_selector_enabled_global && effective_group == "corr_nw" &&
                cleaner_local.adaptive_selector_enabled_for_group(effective_group)) {
                logger->warn("clean.adaptive_selector currently skips corr_nw sub-groups; using configured fixed cut instead");
            }
            if (adaptive_selector_enabled_global &&
                !cleaner_local.adaptive_selector_enabled_for_group(effective_group)) {
                logger->debug("adaptive_selector disabled for {} grouping", effective_group);
            }

            auto get_forced_limit_index_safe = [&](const auto &scans_view,
                                                   const auto &flags_view,
                                                   const auto &apt_flags_view,
                                                   const std::string &group_name_log,
                                                   const Eigen::Index group_key_log,
                                                   const Eigen::Index arr_index_log) {
                try {
                    if (null_model_for_group) {
                        return cleaner_local.get_null_model_index(scans_view, flags_view, apt_flags_view);
                    }
                    if (marchenko_pastur_for_group) {
                        return cleaner_local.get_marchenko_pastur_index(scans_view, flags_view, apt_flags_view);
                    }
                }
                catch (const std::exception &e) {
                    logger->warn(
                        "adaptive cleaner {} failed for grouping={} key={} array={} n_pts={} n_dets={}; "
                        "falling back to configured PCA cut: {}",
                        cleaner_local.active_cleaner_label(), group_name_log, group_key_log, arr_index_log,
                        scans_view.rows(), scans_view.cols(), e.what());
                }
                return Eigen::Index{-1};
            };

            logger->debug("cleaning with {} grouping", effective_group);

            if (store_eigs) {
                // add current group to eval/evec vectors
                out.evals.data.emplace_back();
                out.evecs.data.emplace_back();
            }
            else if (want_eigs && !warned_eigs) {
                logger->warn("n_calc=0; skipping eval/evec output");
                warned_eigs = true;
            }

            // map of tuples to hold detector limits
            std::map<Eigen::Index, std::tuple<Eigen::Index, Eigen::Index>> grp_limits;

            if (group == "corr_nw" && cleaner_local.corr_grouping.enabled) {
                    Eigen::VectorXi corr_group_ids_scan = Eigen::VectorXi::Constant(in.scans.data.cols(), -1);
                    std::vector<CorrNWDiagSummary> corr_summary_scan;
                    corr_summary_scan.reserve(static_cast<std::size_t>(calib.n_nws));
                    grp_limits = get_grouping("nw", calib, in.scans.data.cols());
                    for (auto const& [key, val] : grp_limits) {
                        const Eigen::Index nw_index = key;
                        const Eigen::Index arr_index = toltec_io.nw_to_array_map[key];
                        auto [start_index, n_dets] = std::make_tuple(std::get<0>(val), std::get<1>(val) - std::get<0>(val));

                        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> masked_flags;
                        if (mask_radius_arcsec > 0) {
                            masked_flags = mask_region(in, calib, pixel_axes, map_grouping, n_pts, n_dets, start_index);
                        }
                        else {
                            masked_flags = in.flags.data.block(0, start_index, n_pts, n_dets);
                        }

                        auto in_scans_block = in.scans.data.block(0, start_index, n_pts, n_dets);
                        auto out_scans_block = out.scans.data.block(0, start_index, n_pts, n_dets);
                        out_scans_block = in_scans_block;

                        auto apt_flags = calib.apt["flag"].segment(start_index, n_dets);

                        if (in.kernel.data.size()!=0) {
                            auto in_kernel_block = in.kernel.data.block(0, start_index, n_pts, n_dets);
                            auto out_kernel_block = out.kernel.data.block(0, start_index, n_pts, n_dets);
                            out_kernel_block = in_kernel_block;
                        }

                        auto corr_groups = cleaner_local.get_corr_groups(in_scans_block, masked_flags, apt_flags);
                        logger->info("cleaning corr_nw {} groups={} grouped={} ungrouped={} candidates={} used={} step={}",
                                     key, corr_groups.n_groups_final, corr_groups.n_det_grouped, corr_groups.n_det_ungrouped,
                                     corr_groups.n_det_candidates, corr_groups.n_det_used, corr_groups.sample_step);
                        corr_summary_scan.push_back(CorrNWDiagSummary{
                            .nw = nw_index,
                            .n_det_input = corr_groups.n_det_input,
                            .n_det_candidates = corr_groups.n_det_candidates,
                            .n_det_used = corr_groups.n_det_used,
                            .n_det_grouped = corr_groups.n_det_grouped,
                            .n_det_ungrouped = corr_groups.n_det_ungrouped,
                            .n_groups_raw = corr_groups.n_groups_raw,
                            .n_groups_final = corr_groups.n_groups_final,
                            .sample_step = corr_groups.sample_step,
                        });

                        auto extract_scans_cols = [&](const auto &m, const std::vector<Eigen::Index> &cols) {
                            Eigen::MatrixXd out_m(m.rows(), static_cast<Eigen::Index>(cols.size()));
                            for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(cols.size()); ++c) {
                                out_m.col(c) = m.col(cols[static_cast<std::size_t>(c)]);
                            }
                            return out_m;
                        };
                        auto extract_flag_cols = [&](const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> &m,
                                                     const std::vector<Eigen::Index> &cols) {
                            Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> out_m(
                                m.rows(), static_cast<Eigen::Index>(cols.size()));
                            for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(cols.size()); ++c) {
                                out_m.col(c) = m.col(cols[static_cast<std::size_t>(c)]);
                            }
                            return out_m;
                        };
                        auto extract_apt_cols = [&](const auto &v, const std::vector<Eigen::Index> &cols) {
                            Eigen::VectorXd out_v(static_cast<Eigen::Index>(cols.size()));
                            for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(cols.size()); ++c) {
                                out_v(c) = v(cols[static_cast<std::size_t>(c)]);
                            }
                            return out_v;
                        };
                        auto scatter_cols = [&](auto &dst, const Eigen::MatrixXd &src, const std::vector<Eigen::Index> &cols) {
                            for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(cols.size()); ++c) {
                                dst.col(cols[static_cast<std::size_t>(c)]) = src.col(c);
                            }
                        };

                        for (Eigen::Index gidx = 0; gidx < static_cast<Eigen::Index>(corr_groups.groups.size()); ++gidx) {
                            const auto &cols = corr_groups.groups[static_cast<std::size_t>(gidx)];
                            if (cols.size() < 2) {
                                continue;
                            }
                            for (const auto &local_col : cols) {
                                corr_group_ids_scan(start_index + local_col) = gidx;
                            }

                            auto in_scans_sub = extract_scans_cols(in_scans_block, cols);
                            auto out_scans_sub = in_scans_sub;
                            auto flags_sub = extract_flag_cols(masked_flags, cols);
                            auto apt_flags_sub = extract_apt_cols(apt_flags, cols);

                            if (!(apt_flags_sub.array() == 0).any()) {
                                continue;
                            }

                            auto [evals, evecs] = cleaner_local.calc_eig_values<timestream::Cleaner::SpectraBackend>(
                                in_scans_sub, flags_sub, apt_flags_sub, cleaner_local.n_eig_to_cut[arr_index](indx));
                            Eigen::Index forced_limit_index = get_forced_limit_index_safe(
                                in_scans_sub, flags_sub, apt_flags_sub, group, nw_index, arr_index);

                            if (store_eigs) {
                                Eigen::Index n_keep = std::min<Eigen::Index>(cleaner_local.n_calc, evals.size());
                                if (n_keep > 0) {
                                    Eigen::VectorXd ev = evals.head(n_keep);
                                    Eigen::MatrixXd evc = evecs.leftCols(n_keep);
                                    out.evals.data[indx].push_back(std::move(ev));
                                    out.evecs.data[indx].push_back(std::move(evc));
                                }
                            }

                            cleaner_local.remove_eig_values<timestream::Cleaner::SpectraBackend>(
                                in_scans_sub, flags_sub, evals, evecs, out_scans_sub,
                                cleaner_local.n_eig_to_cut[arr_index](indx), forced_limit_index,
                                group, nw_index, arr_index);
                            scatter_cols(out_scans_block, out_scans_sub, cols);

                            if (in.kernel.data.size()!=0) {
                                auto in_kernel_block = in.kernel.data.block(0, start_index, n_pts, n_dets);
                                auto out_kernel_block = out.kernel.data.block(0, start_index, n_pts, n_dets);
                                auto in_kernel_sub = extract_scans_cols(in_kernel_block, cols);
                                auto out_kernel_sub = in_kernel_sub;
                                cleaner_local.remove_eig_values<timestream::Cleaner::SpectraBackend>(
                                    in_kernel_sub, flags_sub, evals, evecs, out_kernel_sub,
                                    cleaner_local.n_eig_to_cut[arr_index](indx), forced_limit_index,
                                    group, nw_index, arr_index);
                                scatter_cols(out_kernel_block, out_kernel_sub, cols);
                            }
                        }
                    }
                    corr_nw_group_ids_by_scan[in.index.data] = std::move(corr_group_ids_scan);
                    corr_nw_summary_by_scan[in.index.data] = std::move(corr_summary_scan);
                    indx++;
                    out.status.cleaned = true;
                    continue;
            }

            // use all detectors for cleaning
            if (effective_group == "all") {
                grp_limits[0] = std::make_tuple(0,in.scans.data.cols());
            }
            else {
                // get group limits
                grp_limits = get_grouping(effective_group, calib, in.scans.data.cols());
            }
            // loop through cleaning groups
            for (auto const& [key, val] : grp_limits) {
                Eigen::Index arr_index;
                Eigen::Index nw_index = -1;
                // use all detectors
                if (effective_group=="all") {
                    arr_index = calib.arrays(0);
                }
                // use network grouping
                else if (effective_group=="nw" || effective_group=="network") {
                    nw_index = key;
                    arr_index = toltec_io.nw_to_array_map[key];
                }
                // use array grouping
                else if (effective_group=="array") {
                    arr_index = key;
                }

                // start index and number of detectors
                auto [start_index, n_dets] = std::make_tuple(std::get<0>(val), std::get<1>(val) - std::get<0>(val));

                // matrix for flags so we don't overwrite the raw flags
                Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> masked_flags;

                // mask region if radius is >0
                if (mask_radius_arcsec > 0) {
                    // samples that were masked will be flagged
                    masked_flags = mask_region(in, calib, pixel_axes, map_grouping, n_pts, n_dets, start_index);
                }
                // otherwise just use input flags
                else {
                    masked_flags = in.flags.data.block(0, start_index, n_pts, n_dets);
                }

                auto in_scans_block = in.scans.data.block(0, start_index, n_pts, n_dets);
                auto out_scans_block = out.scans.data.block(0, start_index, n_pts, n_dets);

                // get the block of out scans that corresponds to the current array
                auto apt_flags = calib.apt["flag"].segment(start_index, n_dets);

                // check if any good flags
                if ((apt_flags.array()==0).any()) {
                    logger->info("cleaning {} {}", effective_group, key);
                    const Eigen::Index baseline_k = cleaner_local.n_eig_to_cut[arr_index](indx);
                    Eigen::Index solve_n_eig = baseline_k;
                    if (adaptive_selector_for_group) {
                        auto candidate_ks = cleaner_local.adaptive_selector_candidate_cuts(
                            baseline_k, n_dets - 1);
                        if (!candidate_ks.empty()) {
                            solve_n_eig = candidate_ks.back();
                        }
                    }
                    // calculate eigenvalues and eigenvalues
                    const auto eig_t0 = std::chrono::steady_clock::now();
                    auto [evals, evecs] = cleaner_local.calc_eig_values<timestream::Cleaner::SpectraBackend>(
                        in_scans_block, masked_flags, apt_flags, solve_n_eig);
                    const auto eig_t1 = std::chrono::steady_clock::now();
                    const double eig_solve_msec =
                        std::chrono::duration<double, std::milli>(eig_t1 - eig_t0).count();
                    Eigen::Index forced_limit_index = get_forced_limit_index_safe(
                        in_scans_block, masked_flags, apt_flags, effective_group, key, arr_index);
                    timestream::Cleaner::AdaptiveSelectorResult adaptive_result;
                    if (adaptive_selector_for_group) {
                        adaptive_result = cleaner_local.select_adaptive_cut(
                            in_scans_block, masked_flags, apt_flags, evecs,
                            baseline_k, effective_group, key, arr_index);
                    }

                    if (store_eigs) {
                        // get first n_calc eigenvalues and eigenvectors
                        Eigen::Index n_keep = std::min<Eigen::Index>(cleaner_local.n_calc, evals.size());
                        if (n_keep > 0) {
                            Eigen::VectorXd ev = evals.head(n_keep);
                            Eigen::MatrixXd evc = evecs.leftCols(n_keep);

                            // avoid dumping full matrices in debug; can be huge and unstable
                            const Eigen::Index n_show = std::min<Eigen::Index>(n_keep, 8);
                            logger->debug("evals n={} head({})={}", n_keep, n_show, ev.head(n_show).transpose());
                            logger->debug("evecs shape={}x{} (values omitted)", evc.rows(), evc.cols());

                            // copy evals and evecs to ptcdata
                            out.evals.data[indx].push_back(std::move(ev));
                            out.evecs.data[indx].push_back(std::move(evc));
                        }
                    }

                    Eigen::Index k_to_apply = baseline_k;
                    if (adaptive_selector_for_group && adaptive_result.used &&
                        adaptive_result.chosen_cleaned_scans.rows() == out_scans_block.rows() &&
                        adaptive_result.chosen_cleaned_scans.cols() == out_scans_block.cols()) {
                        k_to_apply = adaptive_result.chosen_k;
                        out_scans_block = adaptive_result.chosen_cleaned_scans;
                    }
                    else if (adaptive_selector_for_group && adaptive_result.used) {
                        k_to_apply = adaptive_result.chosen_k;
                        cleaner_local.remove_eig_values<timestream::Cleaner::SpectraBackend>(
                            in_scans_block, masked_flags, evals, evecs, out_scans_block,
                            k_to_apply, forced_limit_index,
                            effective_group, nw_index, arr_index);
                    }
                    else {
                        cleaner_local.remove_eig_values<timestream::Cleaner::SpectraBackend>(
                            in_scans_block, masked_flags, evals, evecs, out_scans_block,
                            baseline_k, forced_limit_index,
                            effective_group, nw_index, arr_index);
                    }

                    if (adaptive_selector_for_group) {
                        const double total_selector_msec = eig_solve_msec +
                            (std::isfinite(adaptive_result.candidate_eval_msec)
                                 ? adaptive_result.candidate_eval_msec
                                 : 0.0);
                        logger->info(
                            "adaptive_selector timing grouping={} key={} nw={} baseline_k={} chosen_k={} eig_ms={} candidate_ms={} total_ms={} margin={}",
                            effective_group, key, nw_index, baseline_k, k_to_apply,
                            eig_solve_msec, adaptive_result.candidate_eval_msec,
                            total_selector_msec, adaptive_result.score_margin);
                        adaptive_summary_scan.push_back(AdaptiveSelectorDiagSummary{
                            .nw = nw_index,
                            .n_det_input = in_scans_block.cols(),
                            .n_det_used = adaptive_result.chosen_diag.n_det_used,
                            .n_time_used = adaptive_result.chosen_diag.n_time_used,
                            .sample_step = adaptive_result.chosen_diag.sample_step,
                            .baseline_k = baseline_k,
                            .chosen_k = k_to_apply,
                            .runnerup_k = adaptive_result.runnerup_k,
                            .n_candidates = adaptive_result.n_candidates,
                            .selector_used = adaptive_result.used ? 1 : 0,
                            .selector_fallback = adaptive_result.fallback ? 1 : 0,
                            .chosen_score = adaptive_result.chosen_score,
                            .runnerup_score = adaptive_result.runnerup_score,
                            .score_margin = adaptive_result.score_margin,
                            .chosen_med_abs_corr = adaptive_result.chosen_diag.med_abs_corr,
                            .chosen_cm_low_mid_ratio = adaptive_result.chosen_diag.cm_low_mid_ratio,
                            .chosen_tail4_binom_z = adaptive_result.chosen_diag.tail4_binom_z,
                            .chosen_top_mode_frac = adaptive_result.chosen_diag.top_mode_frac,
                            .eig_solve_msec = eig_solve_msec,
                            .candidate_eval_msec = adaptive_result.candidate_eval_msec,
                            .total_msec = total_selector_msec,
                        });
                    }

                    if (in.kernel.data.size()!=0) {
                        // check if any good flags
                            logger->debug("cleaning kernel");
                            auto in_kernel_block = in.kernel.data.block(0, start_index, n_pts, n_dets);
                            auto out_kernel_block = in.kernel.data.block(0, start_index, n_pts, n_dets);

                            // remove eigenvalues from the kernel and reconstruct the tod
                            cleaner_local.remove_eig_values<timestream::Cleaner::SpectraBackend>(
                                in_kernel_block, masked_flags, evals, evecs, out_kernel_block,
                                k_to_apply, forced_limit_index,
                                effective_group, nw_index, arr_index);
                    }
                }
                // otherwise just copy the data
                else {
                    logger->debug("no good detectors found. skipping clean.");
                    // copy scans
                    out.scans.data.block(0, start_index, n_pts, n_dets) = in.scans.data.block(0, start_index, n_pts, n_dets);
                    // copy kernel
                    if (in.kernel.data.size()!=0) {
                        out.kernel.data.block(0, start_index, n_pts, n_dets) = in.kernel.data.block(0, start_index, n_pts, n_dets);
                    }
                }
            }
            indx++;
            // set as cleaned
            out.status.cleaned = true;
        }
        if (!adaptive_summary_scan.empty()) {
            adaptive_selector_summary_by_scan[in.index.data] = std::move(adaptive_summary_scan);
        }
    }

    if (second_pass_local.enabled) {
        if (!run_clean) {
            logger->warn("processed_time_chunk.flagging.second_pass_local enabled but clean.enabled=false; skipping PTC second-pass residual flagging");
        }
        else {
            apply_second_pass_local(out, calib);
        }
    }
}

template <class calib_type>
void PTCProc::apply_second_pass_local(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, calib_type &calib) {

    struct DetectorEventRow {
        Eigen::Index nw = -1;
        Eigen::Index uid = -1;
        Eigen::Index det_index = -1;
        TransientEventKind kind = TransientEventKind::unknown;
        Eigen::Index sample = -1;
        double score = std::numeric_limits<double>::quiet_NaN();
        Eigen::Index start_sample = -1;
        Eigen::Index end_sample = -1;
        Eigen::Index width_samples = 0;
        double baseline_shift_z = std::numeric_limits<double>::quiet_NaN();
        double dt_sec = 1.0;
    };

    struct EventCluster {
        Eigen::Index sample = -1;
        Eigen::Index start_sample = -1;
        Eigen::Index end_sample = -1;
        double peak_score = std::numeric_limits<double>::quiet_NaN();
        Eigen::Index top_uid = -1;
        TransientEventKind top_kind = TransientEventKind::unknown;
        Eigen::Index n_detector_events = 0;
        Eigen::Index n_detectors = 0;
        std::vector<DetectorEventRow> rows;
    };

    const Eigen::Index n_pts = in.scans.data.rows();
    const Eigen::Index n_dets_total = in.scans.data.cols();
    if (n_pts < 3 || n_dets_total <= 0) {
        return;
    }

    const double fsmp = (cleaner.sample_rate_Hz > 0.0) ? cleaner.sample_rate_Hz : 1.0;
    const double dt_sec = 1.0 / fsmp;
    int smooth_window = static_cast<int>(std::lround(second_pass_local.baseline_window_sec * fsmp));
    smooth_window = std::max(3, smooth_window);
    if ((smooth_window % 2) == 0) {
        ++smooth_window;
    }
    const Eigen::Index raw_gate_half_window = std::max<Eigen::Index>(
        4, static_cast<Eigen::Index>(std::llround(second_pass_local.raw_window_sec * fsmp)));
    const Eigen::Index raw_max_width_samples = std::max<Eigen::Index>(
        1, static_cast<Eigen::Index>(std::llround(second_pass_local.raw_max_width_sec * fsmp)));
    const Eigen::Index delta_gate_half_window = std::max<Eigen::Index>(
        4, static_cast<Eigen::Index>(std::llround(second_pass_local.delta_window_sec * fsmp)));
    const Eigen::Index delta_max_width_samples = std::max<Eigen::Index>(
        1, static_cast<Eigen::Index>(std::llround(second_pass_local.delta_max_width_sec * fsmp)));
    const Eigen::Index merge_samples = std::max<Eigen::Index>(
        1, static_cast<Eigen::Index>(std::llround(second_pass_local.merge_within_detector_sec * fsmp)));
    const Eigen::Index cluster_samples = std::max<Eigen::Index>(
        1, static_cast<Eigen::Index>(std::llround(second_pass_local.cluster_events_sec * fsmp)));

    auto robust_center_scale = [&](const Eigen::VectorXd &x,
                                   const Eigen::Matrix<bool, Eigen::Dynamic, 1> &flag_mask) {
        std::vector<double> vals;
        vals.reserve(static_cast<std::size_t>(x.size()));
        for (Eigen::Index i = 0; i < x.size(); ++i) {
            if (!flag_mask(i) && std::isfinite(x(i))) {
                vals.push_back(x(i));
            }
        }
        if (vals.size() < 8) {
            vals.clear();
            vals.reserve(static_cast<std::size_t>(x.size()));
            for (Eigen::Index i = 0; i < x.size(); ++i) {
                if (std::isfinite(x(i))) {
                    vals.push_back(x(i));
                }
            }
        }
        if (vals.size() < 8) {
            return std::make_pair(std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN());
        }
        Eigen::Map<const Eigen::VectorXd> vals_map(vals.data(), static_cast<Eigen::Index>(vals.size()));
        const double med = tula::alg::median(vals_map);
        Eigen::VectorXd abs_dev = (vals_map.array() - med).abs();
        double sigma = 1.4826 * tula::alg::median(abs_dev);
        if (!std::isfinite(sigma) || sigma <= 0.0) {
            sigma = engine_utils::calc_std_dev(abs_dev);
        }
        if (!std::isfinite(sigma) || sigma <= 0.0) {
            return std::make_pair(med, std::numeric_limits<double>::quiet_NaN());
        }
        return std::make_pair(med, sigma);
    };

    auto characterize_event =
        [&](const Eigen::VectorXd &resid,
            const Eigen::VectorXd &metric_abs_z,
            const Eigen::Matrix<bool, Eigen::Dynamic, 1> &base_flags,
            Eigen::Index metric_peak_index,
            Eigen::Index peak_sample,
            Eigen::Index gate_half_window,
            Eigen::Index max_width_samples,
            double half_peak_frac,
            double resid_sigma,
            double max_step_shift_z,
            TransientEventKind kind,
            bool metric_is_delta) {
            TransientEvent event;
            event.kind = kind;
            event.sample = static_cast<int>(peak_sample);
            if (!(std::isfinite(resid_sigma) && resid_sigma > 0.0) ||
                metric_peak_index < 0 || metric_peak_index >= metric_abs_z.size() ||
                peak_sample < 0 || peak_sample >= resid.size()) {
                return event;
            }

            const double peak_z = metric_abs_z(metric_peak_index);
            if (!std::isfinite(peak_z) || peak_z <= 0.0) {
                return event;
            }

            event.score = peak_z;
            if (kind == TransientEventKind::raw_like) {
                event.peak_abs_z = peak_z;
            }
            else if (kind == TransientEventKind::delta_like) {
                event.peak_delta_abs_z = peak_z;
            }

            const Eigen::Index left_bound = std::max<Eigen::Index>(0, metric_peak_index - gate_half_window);
            const Eigen::Index right_bound =
                std::min<Eigen::Index>(metric_abs_z.size() - 1, metric_peak_index + gate_half_window);
            const double width_thresh =
                std::max(half_peak_frac * peak_z, std::min(peak_z, 1.5));

            Eigen::Index left = metric_peak_index;
            while (left - 1 >= left_bound &&
                   std::isfinite(metric_abs_z(left - 1)) &&
                   metric_abs_z(left - 1) >= width_thresh) {
                --left;
            }
            Eigen::Index right = metric_peak_index;
            while (right + 1 <= right_bound &&
                   std::isfinite(metric_abs_z(right + 1)) &&
                   metric_abs_z(right + 1) >= width_thresh) {
                ++right;
            }

            const Eigen::Index event_start = std::max<Eigen::Index>(0, left);
            const Eigen::Index event_end = metric_is_delta
                ? std::min<Eigen::Index>(resid.size() - 1, right + 1)
                : std::min<Eigen::Index>(resid.size() - 1, right);
            const Eigen::Index width_samples = std::max<Eigen::Index>(0, event_end - event_start + 1);
            event.start_sample = static_cast<int>(event_start);
            event.end_sample = static_cast<int>(event_end);
            event.width_samples = static_cast<double>(width_samples);

            const Eigen::Index pre_lo = std::max<Eigen::Index>(0, peak_sample - gate_half_window);
            const Eigen::Index pre_hi = std::max<Eigen::Index>(pre_lo, peak_sample - (metric_is_delta ? 2 : 1));
            const Eigen::Index post_lo = std::min<Eigen::Index>(resid.size(), peak_sample + 2);
            const Eigen::Index post_hi = std::min<Eigen::Index>(resid.size(), peak_sample + gate_half_window + 1);
            std::vector<double> pre_vals;
            std::vector<double> post_vals;
            for (Eigen::Index i = pre_lo; i < pre_hi; ++i) {
                if (!base_flags(i) && std::isfinite(resid(i))) {
                    pre_vals.push_back(resid(i));
                }
            }
            for (Eigen::Index i = post_lo; i < post_hi; ++i) {
                if (!base_flags(i) && std::isfinite(resid(i))) {
                    post_vals.push_back(resid(i));
                }
            }
            if (pre_vals.size() >= 4 && post_vals.size() >= 4) {
                Eigen::Map<const Eigen::VectorXd> pre_map(pre_vals.data(), static_cast<Eigen::Index>(pre_vals.size()));
                Eigen::Map<const Eigen::VectorXd> post_map(post_vals.data(), static_cast<Eigen::Index>(post_vals.size()));
                const double pre_med = tula::alg::median(pre_map);
                const double post_med = tula::alg::median(post_map);
                event.baseline_shift_z = std::abs(post_med - pre_med) / resid_sigma;
            }

            event.accepted =
                width_samples <= max_width_samples &&
                std::isfinite(event.baseline_shift_z) &&
                event.baseline_shift_z <= max_step_shift_z;
            return event;
        };

    auto cluster_runs = [](const std::vector<Eigen::Index> &indices) {
        std::vector<std::pair<Eigen::Index, Eigen::Index>> runs;
        if (indices.empty()) {
            return runs;
        }
        Eigen::Index lo = indices.front();
        Eigen::Index hi = indices.front();
        for (std::size_t i = 1; i < indices.size(); ++i) {
            const auto idx = indices[i];
            if (idx <= hi + 1) {
                hi = idx;
            }
            else {
                runs.emplace_back(lo, hi);
                lo = idx;
                hi = idx;
            }
        }
        runs.emplace_back(lo, hi);
        return runs;
    };

    auto median_sample = [](std::vector<Eigen::Index> samples) {
        if (samples.empty()) {
            return Eigen::Index{-1};
        }
        const auto mid = samples.begin() + static_cast<std::ptrdiff_t>(samples.size() / 2);
        std::nth_element(samples.begin(), mid, samples.end());
        return *mid;
    };

    auto merge_detector_rows = [&](std::vector<DetectorEventRow> rows) {
        std::vector<DetectorEventRow> merged;
        if (rows.empty()) {
            return merged;
        }
        std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) {
            if (a.uid != b.uid) {
                return a.uid < b.uid;
            }
            return a.sample < b.sample;
        });
        std::vector<DetectorEventRow> group{rows.front()};
        auto flush = [&](const std::vector<DetectorEventRow> &current) {
            auto best_it = std::max_element(current.begin(), current.end(), [](const auto &a, const auto &b) {
                return a.score < b.score;
            });
            DetectorEventRow out = *best_it;
            std::vector<Eigen::Index> samples;
            samples.reserve(current.size());
            Eigen::Index start_sample = current.front().start_sample;
            Eigen::Index end_sample = current.front().end_sample;
            for (const auto &row : current) {
                samples.push_back(row.sample);
                start_sample = std::min(start_sample, row.start_sample);
                end_sample = std::max(end_sample, row.end_sample);
            }
            out.start_sample = start_sample;
            out.end_sample = end_sample;
            out.sample = median_sample(samples);
            out.width_samples = out.end_sample - out.start_sample + 1;
            merged.push_back(out);
        };
        for (std::size_t i = 1; i < rows.size(); ++i) {
            if (rows[i].uid == group.back().uid && rows[i].sample <= group.back().sample + merge_samples) {
                group.push_back(rows[i]);
            }
            else {
                flush(group);
                group.assign(1, rows[i]);
            }
        }
        flush(group);
        return merged;
    };

    auto cluster_event_rows = [&](std::vector<DetectorEventRow> rows) {
        std::vector<EventCluster> clusters;
        if (rows.empty()) {
            return clusters;
        }
        std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) {
            return a.sample < b.sample;
        });
        std::vector<DetectorEventRow> group{rows.front()};
        auto flush = [&](const std::vector<DetectorEventRow> &current) {
            auto best_it = std::max_element(current.begin(), current.end(), [](const auto &a, const auto &b) {
                return a.score < b.score;
            });
            EventCluster cluster;
            cluster.peak_score = best_it->score;
            cluster.top_uid = best_it->uid;
            cluster.top_kind = best_it->kind;
            cluster.rows = current;
            cluster.n_detector_events = static_cast<Eigen::Index>(current.size());
            std::vector<Eigen::Index> samples;
            std::unordered_set<Eigen::Index> uids;
            cluster.start_sample = current.front().start_sample;
            cluster.end_sample = current.front().end_sample;
            samples.reserve(current.size());
            for (const auto &row : current) {
                samples.push_back(row.sample);
                uids.insert(row.uid);
                cluster.start_sample = std::min(cluster.start_sample, row.start_sample);
                cluster.end_sample = std::max(cluster.end_sample, row.end_sample);
            }
            cluster.sample = median_sample(samples);
            cluster.n_detectors = static_cast<Eigen::Index>(uids.size());
            clusters.push_back(cluster);
        };
        for (std::size_t i = 1; i < rows.size(); ++i) {
            Eigen::Index group_max_sample = group.front().sample;
            for (const auto &row : group) {
                group_max_sample = std::max(group_max_sample, row.sample);
            }
            if (rows[i].sample <= group_max_sample + cluster_samples) {
                group.push_back(rows[i]);
            }
            else {
                flush(group);
                group.assign(1, rows[i]);
            }
        }
        flush(group);
        std::sort(clusters.begin(), clusters.end(), [](const auto &a, const auto &b) {
            if (a.peak_score != b.peak_score) {
                return a.peak_score > b.peak_score;
            }
            if (a.sample != b.sample) {
                return a.sample < b.sample;
            }
            return a.top_uid < b.top_uid;
        });
        return clusters;
    };

    auto analyze_detector =
        [&](const Eigen::VectorXd &signal,
            const Eigen::Matrix<bool, Eigen::Dynamic, 1> &base_flags) {
            std::vector<TransientEvent> events;
            Eigen::Matrix<bool, Eigen::Dynamic, 1> final_flags =
                Eigen::Matrix<bool, Eigen::Dynamic, 1>::Zero(n_pts);
            Eigen::VectorXd resid_z = Eigen::VectorXd::Constant(
                n_pts, std::numeric_limits<double>::quiet_NaN());

            Eigen::Index n_good = 0;
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                if (!base_flags(i) && std::isfinite(signal(i))) {
                    ++n_good;
                }
            }
            const double good_frac = static_cast<double>(n_good) / static_cast<double>(n_pts);
            if (good_frac < second_pass_local.min_good_frac) {
                return std::make_tuple(events, final_flags, resid_z);
            }

            auto [med, sigma] = robust_center_scale(signal, base_flags);
            if (!std::isfinite(sigma) || sigma <= 0.0) {
                return std::make_tuple(events, final_flags, resid_z);
            }

            Eigen::VectorXd baseline_input = signal;
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                if (base_flags(i) || !std::isfinite(baseline_input(i))) {
                    baseline_input(i) = med;
                }
            }
            Eigen::VectorXd smooth = Eigen::VectorXd::Zero(n_pts);
            engine_utils::smooth<engine_utils::SmoothType::edge_truncate>(
                baseline_input, smooth, smooth_window);
            Eigen::VectorXd resid = signal - smooth;

            auto [resid_med, resid_sigma] = robust_center_scale(resid, base_flags);
            if (!std::isfinite(resid_sigma) || resid_sigma <= 0.0) {
                return std::make_tuple(events, final_flags, resid_z);
            }

            Eigen::VectorXd abs_dev = (resid.array() - resid_med).abs();
            Eigen::VectorXd local_abs_z = abs_dev / resid_sigma;
            resid_z = resid / resid_sigma;
            const double raw_candidate_z =
                second_pass_local.raw_candidate_rel_sigma_scale *
                second_pass_local.sigma_scale *
                second_pass_local.min_spike_sigma;

            Eigen::Matrix<bool, Eigen::Dynamic, 1> raw_flags =
                Eigen::Matrix<bool, Eigen::Dynamic, 1>::Zero(n_pts);
            std::vector<Eigen::Index> candidate_samples;
            candidate_samples.reserve(static_cast<std::size_t>(n_pts));
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                if (!base_flags(i) && std::isfinite(local_abs_z(i)) && local_abs_z(i) > raw_candidate_z) {
                    candidate_samples.push_back(i);
                }
            }
            for (const auto &[lo, hi] : cluster_runs(candidate_samples)) {
                Eigen::Index best_sample = lo;
                double best_z = -1.0;
                for (Eigen::Index sample = lo; sample <= hi; ++sample) {
                    if (std::isfinite(local_abs_z(sample)) && local_abs_z(sample) > best_z) {
                        best_z = local_abs_z(sample);
                        best_sample = sample;
                    }
                }
                auto event = characterize_event(
                    resid, local_abs_z, base_flags, best_sample, best_sample,
                    raw_gate_half_window, raw_max_width_samples,
                    second_pass_local.raw_half_peak_frac, resid_sigma,
                    second_pass_local.max_step_shift_z,
                    TransientEventKind::raw_like, false);
                if (event.accepted) {
                    raw_flags.segment(event.start_sample, event.end_sample - event.start_sample + 1).setOnes();
                    events.push_back(event);
                }
            }

            final_flags = raw_flags;
            std::vector<double> local_delta_vals;
            std::vector<Eigen::Index> local_delta_edges;
            local_delta_vals.reserve(static_cast<std::size_t>(std::max<Eigen::Index>(n_pts - 1, 0)));
            local_delta_edges.reserve(static_cast<std::size_t>(std::max<Eigen::Index>(n_pts - 1, 0)));
            for (Eigen::Index i = 0; i < n_pts - 1; ++i) {
                if (base_flags(i) || base_flags(i + 1) || raw_flags(i) || raw_flags(i + 1)) {
                    continue;
                }
                if (!std::isfinite(resid(i)) || !std::isfinite(resid(i + 1))) {
                    continue;
                }
                local_delta_vals.push_back(resid(i + 1) - resid(i));
                local_delta_edges.push_back(i);
            }

            if (local_delta_vals.size() >= 8) {
                Eigen::Map<const Eigen::VectorXd> delta_map(
                    local_delta_vals.data(), static_cast<Eigen::Index>(local_delta_vals.size()));
                const double delta_med = tula::alg::median(delta_map);
                Eigen::VectorXd delta_abs_dev = (delta_map.array() - delta_med).abs();
                double delta_sigma = 1.4826 * tula::alg::median(delta_abs_dev);
                if (!std::isfinite(delta_sigma) || delta_sigma <= 0.0) {
                    delta_sigma = engine_utils::calc_std_dev(delta_abs_dev);
                }
                if (std::isfinite(delta_sigma) && delta_sigma > 0.0) {
                    Eigen::VectorXd local_delta_abs_z =
                        Eigen::VectorXd::Constant(std::max<Eigen::Index>(n_pts - 1, 0),
                                                  std::numeric_limits<double>::quiet_NaN());
                    std::vector<Eigen::Index> candidate_edges;
                    const double local_delta_cutoff =
                        second_pass_local.delta_sigma_scale *
                        second_pass_local.min_spike_sigma * delta_sigma;
                    for (std::size_t i = 0; i < local_delta_edges.size(); ++i) {
                        const auto edge = local_delta_edges[i];
                        const double abs_delta = std::abs(local_delta_vals[i] - delta_med);
                        local_delta_abs_z(edge) = abs_delta / delta_sigma;
                        if (abs_delta > local_delta_cutoff) {
                            candidate_edges.push_back(edge);
                        }
                    }
                    for (const auto &[lo, hi] : cluster_runs(candidate_edges)) {
                        Eigen::Index best_edge = lo;
                        double best_z = -1.0;
                        for (Eigen::Index edge = lo; edge <= hi; ++edge) {
                            if (edge >= 0 && edge < local_delta_abs_z.size() &&
                                std::isfinite(local_delta_abs_z(edge)) &&
                                local_delta_abs_z(edge) > best_z) {
                                best_z = local_delta_abs_z(edge);
                                best_edge = edge;
                            }
                        }
                        auto event = characterize_event(
                            resid, local_delta_abs_z, base_flags, best_edge, best_edge + 1,
                            delta_gate_half_window, delta_max_width_samples,
                            second_pass_local.delta_half_peak_frac, resid_sigma,
                            second_pass_local.max_step_shift_z,
                            TransientEventKind::delta_like, true);
                        if (event.accepted) {
                            final_flags(best_edge) = true;
                            if (best_edge + 1 < n_pts) {
                                final_flags(best_edge + 1) = true;
                            }
                            events.push_back(event);
                        }
                    }
                }
            }

            return std::make_tuple(events, final_flags, resid_z);
        };

    auto group_limits = get_grouping("nw", calib, in.scans.data.cols());
    std::vector<SecondPassDiagSummary> summaries;
    summaries.reserve(group_limits.size());
    Eigen::Matrix<signed char, Eigen::Dynamic, Eigen::Dynamic> added_flags_out;
    if (run_tod_output) {
        added_flags_out = Eigen::Matrix<signed char, Eigen::Dynamic, Eigen::Dynamic>::Zero(n_pts, n_dets_total);
    }

    for (const auto &[key, val] : group_limits) {
        const Eigen::Index nw_index = key;
        const auto start_index = std::get<0>(val);
        const auto n_dets = std::get<1>(val) - std::get<0>(val);
        if (n_dets <= 0) {
            continue;
        }

        const auto apt_flags = calib.apt["flag"].segment(start_index, n_dets);
        auto flags_block = in.flags.data.block(0, start_index, n_pts, n_dets);
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> existing_flags_block = flags_block;
        std::unordered_map<Eigen::Index, Eigen::Index> local_det_lookup;
        local_det_lookup.reserve(static_cast<std::size_t>(n_dets));

        std::vector<DetectorEventRow> detector_rows;
        double residual_peak = std::numeric_limits<double>::quiet_NaN();
        int residual_peak_uid = kTransientFillInt;

        for (Eigen::Index local_j = 0; local_j < n_dets; ++local_j) {
            const Eigen::Index det_col = start_index + local_j;
            local_det_lookup[det_col] = local_j;
            if (apt_flags(local_j) != 0) {
                continue;
            }
            auto signal = in.scans.data.col(det_col);
            auto det_flags = in.flags.data.col(det_col);
            auto [events, det_prop_flags, det_resid_z] = analyze_detector(signal, det_flags);

            bool det_has_resid = false;
            double det_peak = std::numeric_limits<double>::quiet_NaN();
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                if (!det_flags(i) && std::isfinite(det_resid_z(i))) {
                    const double v = std::abs(det_resid_z(i));
                    if (!det_has_resid || v > det_peak) {
                        det_peak = v;
                        det_has_resid = true;
                    }
                }
            }
            if (det_has_resid && (!std::isfinite(residual_peak) || det_peak > residual_peak)) {
                residual_peak = det_peak;
                residual_peak_uid = static_cast<int>(calib.apt["uid"](det_col));
            }

            for (const auto &event : events) {
                detector_rows.push_back(DetectorEventRow{
                    .nw = nw_index,
                    .uid = static_cast<Eigen::Index>(calib.apt["uid"](det_col)),
                    .det_index = det_col,
                    .kind = event.kind,
                    .sample = event.sample,
                    .score = event.score,
                    .start_sample = event.start_sample,
                    .end_sample = event.end_sample,
                    .width_samples = std::max(0, event.end_sample - event.start_sample + 1),
                    .baseline_shift_z = event.baseline_shift_z,
                    .dt_sec = dt_sec,
                });
            }
        }

        auto merged_events = merge_detector_rows(detector_rows);
        auto clusters = cluster_event_rows(merged_events);
        std::vector<EventCluster> candidate_clusters;
        for (const auto &cluster : clusters) {
            if (cluster.n_detectors >= second_pass_local.min_cluster_detectors ||
                cluster.peak_score >= second_pass_local.high_score_cluster_override) {
                candidate_clusters.push_back(cluster);
            }
        }
        std::sort(candidate_clusters.begin(), candidate_clusters.end(), [](const auto &a, const auto &b) {
            if (a.peak_score != b.peak_score) {
                return a.peak_score > b.peak_score;
            }
            if (a.sample != b.sample) {
                return a.sample < b.sample;
            }
            return a.top_uid < b.top_uid;
        });

        const bool busy_network_vetoed =
            static_cast<int>(candidate_clusters.size()) > second_pass_local.max_auto_flag_clusters_per_network;
        std::vector<EventCluster> accepted_clusters = busy_network_vetoed
            ? std::vector<EventCluster>{}
            : candidate_clusters;

        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> accepted_flags_block =
            Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>::Zero(n_pts, n_dets);
        std::vector<DetectorEventRow> accepted_rows;
        accepted_rows.reserve(detector_rows.size());
        for (const auto &cluster : accepted_clusters) {
            for (const auto &row : cluster.rows) {
                accepted_rows.push_back(row);
                const auto it = local_det_lookup.find(row.det_index);
                if (it == local_det_lookup.end()) {
                    continue;
                }
                accepted_flags_block.block(
                    row.start_sample, it->second, row.end_sample - row.start_sample + 1, 1).setOnes();
            }
        }
        std::sort(accepted_rows.begin(), accepted_rows.end(), [](const auto &a, const auto &b) {
            return a.score > b.score;
        });

        flags_block = existing_flags_block.array() || accepted_flags_block.array();
        if (run_tod_output) {
            added_flags_out.block(0, start_index, n_pts, n_dets) =
                accepted_flags_block.cast<signed char>();
        }

        Eigen::Index n_det_with_added_flags = 0;
        for (Eigen::Index j = 0; j < n_dets; ++j) {
            bool any = false;
            for (Eigen::Index i = 0; i < n_pts; ++i) {
                if (accepted_flags_block(i, j)) {
                    any = true;
                    break;
                }
            }
            if (any) {
                ++n_det_with_added_flags;
            }
        }

        SecondPassDiagSummary summary;
        summary.nw = nw_index;
        summary.n_det = n_dets;
        summary.n_pts = n_pts;
        summary.n_merged_events_total = static_cast<Eigen::Index>(merged_events.size());
        summary.n_clusters_total = static_cast<Eigen::Index>(clusters.size());
        summary.n_candidate_clusters = static_cast<Eigen::Index>(candidate_clusters.size());
        summary.n_candidate_events = 0;
        for (const auto &cluster : candidate_clusters) {
            summary.n_candidate_events += cluster.n_detector_events;
        }
        summary.n_accepted_clusters = static_cast<Eigen::Index>(accepted_clusters.size());
        summary.n_accepted_events = static_cast<Eigen::Index>(accepted_rows.size());
        summary.n_det_with_added_flags = n_det_with_added_flags;
        summary.busy_network_vetoed = busy_network_vetoed;
        summary.existing_flagged_fraction = existing_flags_block.cast<double>().mean();
        summary.proposed_flagged_fraction = accepted_flags_block.cast<double>().mean();
        summary.newly_flagged_fraction =
            (accepted_flags_block.array() && !existing_flags_block.array())
                .template cast<double>().mean();
        summary.max_unflagged_residual_z = residual_peak;
        summary.max_unflagged_residual_uid = residual_peak_uid;
        if (!candidate_clusters.empty()) {
            summary.top_candidate_cluster_peak_score = candidate_clusters.front().peak_score;
            summary.top_candidate_cluster_n_detectors = candidate_clusters.front().n_detectors;
            summary.top_candidate_cluster_n_events = candidate_clusters.front().n_detector_events;
            summary.top_candidate_cluster_sample = static_cast<int>(candidate_clusters.front().sample);
        }
        if (!accepted_rows.empty()) {
            summary.top_event_uid = static_cast<int>(accepted_rows.front().uid);
            summary.top_event.kind = accepted_rows.front().kind;
            summary.top_event.sample = static_cast<int>(accepted_rows.front().sample);
            summary.top_event.start_sample = static_cast<int>(accepted_rows.front().start_sample);
            summary.top_event.end_sample = static_cast<int>(accepted_rows.front().end_sample);
            summary.top_event.score = accepted_rows.front().score;
            summary.top_event.width_samples = static_cast<double>(accepted_rows.front().width_samples);
            summary.top_event.baseline_shift_z = accepted_rows.front().baseline_shift_z;
            summary.top_event.accepted = true;
        }
        summaries.push_back(summary);

        if (!candidate_clusters.empty()) {
            logger->info(
                "PTC second pass scan {} nw {} candidate_clusters={} accepted_clusters={} busy_veto={} newly_flagged_fraction={} top_candidate_peak_score={} top_candidate_n_detectors={}",
                static_cast<long long>(in.index.data) + 1, static_cast<long long>(nw_index),
                static_cast<long long>(summary.n_candidate_clusters),
                static_cast<long long>(summary.n_accepted_clusters),
                summary.busy_network_vetoed ? 1 : 0,
                summary.newly_flagged_fraction,
                summary.top_candidate_cluster_peak_score,
                static_cast<long long>(summary.top_candidate_cluster_n_detectors));
        }
    }

    second_pass_summary_by_scan[in.index.data] = summaries;
    if (run_tod_output) {
        second_pass_added_flags_by_scan[in.index.data] = std::move(added_flags_out);
    }
}

template <typename apt_type, class tel_type>
void PTCProc::calc_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, apt_type &apt, tel_type &telescope) {
    // number of detectors
    Eigen::Index n_dets = in.scans.data.cols();
    const auto scan_index_1based = static_cast<long long>(in.index.data) + 1;

    // resize weights to number of detectors
    in.weights.data = Eigen::VectorXd::Zero(n_dets);

    // approximate weighting
    if (weighting_type == "approximate") {
        logger->debug("calculating weights using detector sensitivities");
        // unit conversion x flux calibration factor x 1/exp(-tau)
        double conversion_factor;

        // loop through detectors and calculate weights
        for (Eigen::Index i=0; i<n_dets; ++i) {
            // current detector index
            Eigen::Index det_index = i;
            if (apt["flag"](det_index)!=0) {
                in.weights.data(i) = 0;
                continue;
            }
            // if flux calibrated, get flux conversion factor
            if (in.status.calibrated) {
                conversion_factor = in.fcf.data(i);
            }
            // otherwise fcf is unity
            else {
                conversion_factor = 1;
            }
            // make sure flux conversion is not zero (otherwise weight=0)
            if (conversion_factor*apt["sens"](det_index)!=0) {
                // calculate weights while applying flux calibration
                in.weights.data(i) = pow(sqrt(telescope.d_fsmp)*apt["sens"](det_index)*conversion_factor,-2.0);
            }
            else {
                in.weights.data(i) = 0;
            }
        }
    }
    // use full weighting
    else if (weighting_type == "full"){
        logger->debug("calculating weights using timestream variance");
        const bool use_source_weight_mask =
            mask_radius_arcsec > 0.0 &&
            fruit_loops_source_valid.size() > 0 &&
            fruit_loops_source_lat.size() == fruit_loops_source_valid.size() &&
            fruit_loops_source_lon.size() == fruit_loops_source_valid.size();
        const double source_mask_radius_rad = mask_radius_arcsec * ASEC_TO_RAD;

        if (use_source_weight_mask) {
            logger->info("calculating full weights with source mask (radius {} arcsec) for scan {}",
                         mask_radius_arcsec, scan_index_1based);
        }

        // loop through detectors
        for (Eigen::Index i=0; i<n_dets; ++i) {
            // only calculate weights if detector is unflagged
            if (apt["flag"](i)==0) {
                // make Eigen::Maps for each detector's scan
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> scans(
                    in.scans.data.col(i).data(), in.scans.data.rows());
                Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> base_flags(
                    in.flags.data.col(i).data(), in.flags.data.rows());

                // unflagged detector stddev
                double det_std_dev = 0.0;
                if (use_source_weight_mask &&
                    i < in.map_indices.data.size()) {
                    const auto map_index = in.map_indices.data(i);
                    if (map_index >= 0 &&
                        map_index < fruit_loops_source_valid.size() &&
                        fruit_loops_source_valid(map_index)) {
                        Eigen::Matrix<bool, Eigen::Dynamic, 1> weight_flags = base_flags;
                        auto [lat, lon] = engine_utils::calc_det_pointing(
                            in.tel_data.data, apt["x_t"](i), apt["y_t"](i),
                            telescope.pixel_axes, in.pointing_offsets_arcsec.data,
                            active_map_grouping);
                        const double source_lat = fruit_loops_source_lat(map_index);
                        const double source_lon = fruit_loops_source_lon(map_index);
                        for (Eigen::Index j = 0; j < weight_flags.size(); ++j) {
                            const double dlat = lat(j) - source_lat;
                            const double dlon = lon(j) - source_lon;
                            if (std::sqrt(dlat * dlat + dlon * dlon) < source_mask_radius_rad) {
                                weight_flags(j) = 1;
                            }
                        }
                        det_std_dev = engine_utils::calc_std_dev(scans, weight_flags);
                    }
                    else {
                        det_std_dev = engine_utils::calc_std_dev(scans, base_flags);
                    }
                }
                else {
                    det_std_dev = engine_utils::calc_std_dev(scans, base_flags);
                }
                // if stddev is not zero
                if (det_std_dev !=0) {
                    // weight = 1/(stddev)^2
                    in.weights.data(i) = pow(det_std_dev,-2);
                }
                // otherwise weight = 0 (not included in maps)
                else {
                    in.weights.data(i) = 0;
                }
            }
            // otherwise weight = 0 (not included in maps)
            else {
                in.weights.data(i) = 0;
            }
        }
    }
    // constant weighting
    else if (weighting_type == "const") {
        for (Eigen::Index i=0; i<n_dets; ++i) {
            // only calculate weights if detector is unflagged
            if (apt["flag"](i)==0) {
                in.weights.data(i) = 1;
            }
            // otherwise set to zero
            else {
                in.weights.data(i) = 0;
            }
        }
    }

    auto finite_or_nan = [](double v) {
        if (std::isfinite(v)) {
            return v;
        }
        return std::numeric_limits<double>::quiet_NaN();
    };

    std::map<Eigen::Index, std::tuple<Eigen::Index, Eigen::Index>> nw_limits;
    if (weight_corr_penalty.enabled || busy_row_suppression.enabled) {
        if (n_dets > 0) {
            Eigen::Index nw_i = static_cast<Eigen::Index>(apt["nw"](0));
            nw_limits[nw_i] = std::tuple<Eigen::Index, Eigen::Index>{0, 1};
            std::unordered_set<Eigen::Index> seen;
            seen.insert(nw_i);
            for (Eigen::Index i = 1; i < n_dets; ++i) {
                auto nw_v = static_cast<Eigen::Index>(apt["nw"](i));
                if (nw_v == nw_i) {
                    std::get<1>(nw_limits[nw_i]) = i + 1;
                }
                else {
                    if (seen.find(nw_v) != seen.end()) {
                        logger->error("non-contiguous grouping detected for 'nw' value {}", nw_v);
                        std::exit(EXIT_FAILURE);
                    }
                    seen.insert(nw_v);
                    nw_i = nw_v;
                    nw_limits[nw_i] = std::tuple<Eigen::Index, Eigen::Index>{i, i + 1};
                }
            }
        }
    }

    if (weight_corr_penalty.enabled) {
        auto clamp01 = [](double v) {
            return std::clamp(v, 0.0, 1.0);
        };
        auto median_from_values = [](std::vector<double> values) {
            if (values.empty()) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const auto mid = values.size() / 2;
            std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
            double med = values[mid];
            if ((values.size() % 2) == 0) {
                auto max_it = std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid));
                med = 0.5 * (med + *max_it);
            }
            return med;
        };
        auto pearson_corr = [](const std::vector<double> &x, const std::vector<double> &y) {
            if (x.size() != y.size() || x.size() < 2) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            double sx = 0.0;
            double sy = 0.0;
            double sxx = 0.0;
            double syy = 0.0;
            double sxy = 0.0;
            for (std::size_t i = 0; i < x.size(); ++i) {
                const double xv = x[i];
                const double yv = y[i];
                sx += xv;
                sy += yv;
                sxx += xv * xv;
                syy += yv * yv;
                sxy += xv * yv;
            }
            const double n = static_cast<double>(x.size());
            const double vx = sxx - (sx * sx) / n;
            const double vy = syy - (sy * sy) / n;
            if (vx <= 0.0 || vy <= 0.0 || !std::isfinite(vx) || !std::isfinite(vy)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const double cov = sxy - (sx * sy) / n;
            const double corr = cov / std::sqrt(vx * vy);
            if (!std::isfinite(corr)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return std::clamp(corr, -1.0, 1.0);
        };
        auto score_metric = [&](double metric, const auto &term) {
            if (!term.enabled || term.weight <= 0.0 || !std::isfinite(metric)) {
                return std::pair<double, double>{0.0, 0.0};
            }
            const double span = std::max(term.span, 1e-12);
            const double score = clamp01((metric - term.ref) / span);
            return std::pair<double, double>{term.weight * score, term.weight};
        };

        const Eigen::Index n_pts_full = in.scans.data.rows();
        std::vector<WeightCorrPenaltyDiagSummary> penalty_summary;
        penalty_summary.reserve(static_cast<std::size_t>(nw_limits.size()));

        for (const auto &[nw, limits] : nw_limits) {
            const auto [start_index, end_index] = limits;
            const Eigen::Index n_det_group = end_index - start_index;

            Eigen::Index sample_step = 1;
            if (weight_corr_penalty.max_samples > 0 &&
                n_pts_full > static_cast<Eigen::Index>(weight_corr_penalty.max_samples)) {
                sample_step = static_cast<Eigen::Index>(std::ceil(
                    static_cast<double>(n_pts_full) / static_cast<double>(weight_corr_penalty.max_samples)));
            }
            sample_step = std::max<Eigen::Index>(sample_step, 1);
            const Eigen::Index n_pts = (n_pts_full + sample_step - 1) / sample_step;

            std::vector<Eigen::Index> det_keep;
            std::vector<double> det_mean;
            std::vector<double> det_std;
            det_keep.reserve(static_cast<std::size_t>(n_det_group));
            det_mean.reserve(static_cast<std::size_t>(n_det_group));
            det_std.reserve(static_cast<std::size_t>(n_det_group));

            Eigen::Index n_candidates = 0;
            for (Eigen::Index j = start_index; j < end_index; ++j) {
                if (apt["flag"](j) != 0) {
                    continue;
                }
                double sum = 0.0;
                double sum2 = 0.0;
                double count = 0.0;
                for (Eigen::Index is = 0; is < n_pts; ++is) {
                    const Eigen::Index i = is * sample_step;
                    if (i >= n_pts_full) {
                        break;
                    }
                    if (in.flags.data(i, j)) {
                        continue;
                    }
                    const double v = in.scans.data(i, j);
                    if (!std::isfinite(v)) {
                        continue;
                    }
                    sum += v;
                    sum2 += v * v;
                    count += 1.0;
                }
                if (count <= 1.0) {
                    continue;
                }
                const double frac = count / static_cast<double>(n_pts);
                if (frac < weight_corr_penalty.min_good_frac) {
                    continue;
                }
                n_candidates++;
                const double mean = sum / count;
                const double var_num = sum2 - (sum * sum) / count;
                const double var_den = count - 1.0;
                if (var_den <= 0.0) {
                    continue;
                }
                const double var = var_num / var_den;
                if (!(var > 0.0) || !std::isfinite(var)) {
                    continue;
                }
                const double std = std::sqrt(var);
                if (!(std > 0.0) || !std::isfinite(std)) {
                    continue;
                }
                det_keep.push_back(j);
                det_mean.push_back(mean);
                det_std.push_back(std);
            }
            const Eigen::Index n_used = static_cast<Eigen::Index>(det_keep.size());

            auto pair_corr_for = [&](Eigen::Index det_a, Eigen::Index det_b) {
                double sx = 0.0;
                double sy = 0.0;
                double sxx = 0.0;
                double syy = 0.0;
                double sxy = 0.0;
                Eigen::Index n_ov = 0;
                for (Eigen::Index is = 0; is < n_pts; ++is) {
                    const Eigen::Index i = is * sample_step;
                    if (i >= n_pts_full) {
                        break;
                    }
                    if (in.flags.data(i, det_a) || in.flags.data(i, det_b)) {
                        continue;
                    }
                    const double x = in.scans.data(i, det_a);
                    const double y = in.scans.data(i, det_b);
                    if (!std::isfinite(x) || !std::isfinite(y)) {
                        continue;
                    }
                    sx += x;
                    sy += y;
                    sxx += x * x;
                    syy += y * y;
                    sxy += x * y;
                    n_ov++;
                }
                const Eigen::Index min_overlap = std::max<Eigen::Index>(2, weight_corr_penalty.min_overlap);
                if (n_ov < min_overlap) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                const double n = static_cast<double>(n_ov);
                const double vx = sxx - (sx * sx) / n;
                const double vy = syy - (sy * sy) / n;
                if (!(vx > 0.0) || !(vy > 0.0) || !std::isfinite(vx) || !std::isfinite(vy)) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                const double cov = sxy - (sx * sy) / n;
                const double corr = cov / std::sqrt(vx * vy);
                if (!std::isfinite(corr)) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                return std::clamp(corr, -1.0, 1.0);
            };

            double pair_med_abs_corr = std::numeric_limits<double>::quiet_NaN();
            if (weight_corr_penalty.pair_corr.enabled && n_used >= 2) {
                const std::uint64_t n_pairs_total = static_cast<std::uint64_t>(n_used) *
                                                    static_cast<std::uint64_t>(n_used - 1) / 2ULL;
                std::uint64_t target_pairs = n_pairs_total;
                if (weight_corr_penalty.max_pairs > 0) {
                    target_pairs = std::min<std::uint64_t>(
                        n_pairs_total, static_cast<std::uint64_t>(weight_corr_penalty.max_pairs));
                }
                std::vector<double> abs_corrs;
                abs_corrs.reserve(static_cast<std::size_t>(target_pairs));

                if (target_pairs == n_pairs_total) {
                    for (Eigen::Index i = 0; i < n_used; ++i) {
                        for (Eigen::Index j = i + 1; j < n_used; ++j) {
                            const double c = pair_corr_for(
                                det_keep[static_cast<std::size_t>(i)],
                                det_keep[static_cast<std::size_t>(j)]);
                            if (std::isfinite(c)) {
                                abs_corrs.push_back(std::abs(c));
                            }
                        }
                    }
                }
                else if (target_pairs > 0) {
                    const std::uint64_t seed_mix =
                        static_cast<std::uint64_t>(weight_corr_penalty.seed) ^
                        (static_cast<std::uint64_t>(scan_index_1based + 1) * 1315423911ULL) ^
                        (static_cast<std::uint64_t>(nw + 1) * 2654435761ULL);
                    std::mt19937 rng_nw(static_cast<std::uint32_t>(seed_mix & 0xffffffffULL));
                    std::uniform_int_distribution<Eigen::Index> det_dist(0, n_used - 1);
                    std::unordered_set<std::uint64_t> seen_pairs;
                    seen_pairs.reserve(static_cast<std::size_t>(target_pairs * 2 + 1));
                    std::uint64_t tries = 0;
                    const std::uint64_t max_tries = std::max<std::uint64_t>(target_pairs * 32ULL, 1024ULL);
                    while (seen_pairs.size() < target_pairs && tries < max_tries) {
                        tries++;
                        Eigen::Index a = det_dist(rng_nw);
                        Eigen::Index b = det_dist(rng_nw);
                        if (a == b) {
                            continue;
                        }
                        if (a > b) {
                            std::swap(a, b);
                        }
                        const auto key = (static_cast<std::uint64_t>(a) << 32ULL) |
                                         static_cast<std::uint64_t>(b);
                        if (!seen_pairs.insert(key).second) {
                            continue;
                        }
                        const double c = pair_corr_for(
                            det_keep[static_cast<std::size_t>(a)],
                            det_keep[static_cast<std::size_t>(b)]);
                        if (std::isfinite(c)) {
                            abs_corrs.push_back(std::abs(c));
                        }
                    }
                }
                pair_med_abs_corr = median_from_values(std::move(abs_corrs));
            }

            Eigen::VectorXd cm = Eigen::VectorXd::Constant(n_pts, std::numeric_limits<double>::quiet_NaN());
            std::vector<double> cm_valid;
            std::vector<double> el_valid;
            double cm_el_abs_corr = std::numeric_limits<double>::quiet_NaN();
            double cm_low_mid_ratio = std::numeric_limits<double>::quiet_NaN();

            const bool need_cm = (weight_corr_penalty.cm_el_corr.enabled ||
                                  weight_corr_penalty.cm_low_mid_ratio.enabled) && (n_used > 0);
            if (need_cm) {
                for (Eigen::Index is = 0; is < n_pts; ++is) {
                    const Eigen::Index i = is * sample_step;
                    if (i >= n_pts_full) {
                        break;
                    }
                    double sum = 0.0;
                    Eigen::Index count = 0;
                    for (Eigen::Index k = 0; k < n_used; ++k) {
                        const Eigen::Index det = det_keep[static_cast<std::size_t>(k)];
                        if (in.flags.data(i, det)) {
                            continue;
                        }
                        const double v = in.scans.data(i, det);
                        if (!std::isfinite(v)) {
                            continue;
                        }
                        const double z = (v - det_mean[static_cast<std::size_t>(k)]) /
                                         det_std[static_cast<std::size_t>(k)];
                        if (!std::isfinite(z)) {
                            continue;
                        }
                        sum += z;
                        count++;
                    }
                    if (count >= 2) {
                        cm(is) = sum / static_cast<double>(count);
                    }
                }

                if (weight_corr_penalty.cm_el_corr.enabled) {
                    const auto el_it = in.tel_data.data.find("TelElAct");
                    if (el_it != in.tel_data.data.end()) {
                        const auto &tel_el = el_it->second;
                        cm_valid.reserve(static_cast<std::size_t>(n_pts));
                        el_valid.reserve(static_cast<std::size_t>(n_pts));
                        for (Eigen::Index is = 0; is < n_pts; ++is) {
                            const Eigen::Index i = is * sample_step;
                            if (i >= n_pts_full || i >= tel_el.size()) {
                                break;
                            }
                            const double c = cm(is);
                            const double e = tel_el(i);
                            if (!std::isfinite(c) || !std::isfinite(e)) {
                                continue;
                            }
                            cm_valid.push_back(c);
                            el_valid.push_back(e);
                        }
                        const double c = pearson_corr(cm_valid, el_valid);
                        if (std::isfinite(c)) {
                            cm_el_abs_corr = std::abs(c);
                        }
                    }
                }

                if (weight_corr_penalty.cm_low_mid_ratio.enabled) {
                    std::vector<double> cm_pts;
                    cm_pts.reserve(static_cast<std::size_t>(n_pts));
                    for (Eigen::Index is = 0; is < n_pts; ++is) {
                        const double c = cm(is);
                        if (std::isfinite(c)) {
                            cm_pts.push_back(c);
                        }
                    }
                    if (cm_pts.size() >= 8) {
                        const double cm_mean = std::accumulate(cm_pts.begin(), cm_pts.end(), 0.0) /
                                               static_cast<double>(cm_pts.size());
                        Eigen::VectorXd x = Eigen::VectorXd::Zero(n_pts);
                        for (Eigen::Index is = 0; is < n_pts; ++is) {
                            const double c = cm(is);
                            if (std::isfinite(c)) {
                                x(is) = c - cm_mean;
                            }
                        }
                        // mild taper to reduce leakage from scan edges
                        if (n_pts > 1) {
                            constexpr double two_pi = 6.283185307179586476925286766559;
                            for (Eigen::Index is = 0; is < n_pts; ++is) {
                                const double w = 0.5 * (1.0 - std::cos(
                                    two_pi * static_cast<double>(is) /
                                    static_cast<double>(n_pts - 1)));
                                x(is) *= w;
                            }
                        }

                        Eigen::FFT<double> fft;
                        fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
                        fft.SetFlag(Eigen::FFT<double>::Unscaled);
                        Eigen::VectorXcd freq;
                        fft.fwd(freq, x);

                        const double fs_eff = telescope.d_fsmp / static_cast<double>(sample_step);
                        if (fs_eff > 0.0 && freq.size() > 0) {
                            double p_low = 0.0;
                            double p_mid = 0.0;
                            const auto &band = weight_corr_penalty.cm_low_mid_ratio;
                            for (Eigen::Index k = 1; k < freq.size(); ++k) {
                                const double f = static_cast<double>(k) * fs_eff / static_cast<double>(n_pts);
                                const double p = std::norm(freq(k));
                                if (f >= band.low_min_Hz && f < band.low_max_Hz) {
                                    p_low += p;
                                }
                                if (f >= band.mid_min_Hz && f < band.mid_max_Hz) {
                                    p_mid += p;
                                }
                            }
                            if (p_mid > 0.0 && std::isfinite(p_low) && std::isfinite(p_mid)) {
                                cm_low_mid_ratio = p_low / p_mid;
                            }
                        }
                    }
                }
            }

            double score_num = 0.0;
            double score_den = 0.0;

            {
                const auto [n, d] = score_metric(pair_med_abs_corr, weight_corr_penalty.pair_corr);
                score_num += n;
                score_den += d;
            }
            {
                const auto [n, d] = score_metric(cm_el_abs_corr, weight_corr_penalty.cm_el_corr);
                score_num += n;
                score_den += d;
            }
            {
                const auto [n, d] = score_metric(cm_low_mid_ratio, weight_corr_penalty.cm_low_mid_ratio);
                score_num += n;
                score_den += d;
            }

            double severity = 0.0;
            if (score_den > 0.0 && std::isfinite(score_num)) {
                severity = clamp01(score_num / score_den);
            }

            const double floor = clamp01(weight_corr_penalty.floor);
            const double exponent = std::max(0.0, weight_corr_penalty.exponent);
            double penalty_factor = 1.0;
            if (score_den > 0.0) {
                penalty_factor = floor + (1.0 - floor) * std::pow(clamp01(1.0 - severity), exponent);
            }
            if (!std::isfinite(penalty_factor)) {
                penalty_factor = 1.0;
            }
            penalty_factor = std::clamp(penalty_factor, floor, 1.0);

            Eigen::Index n_weighted = 0;
            for (Eigen::Index j = start_index; j < end_index; ++j) {
                if (apt["flag"](j) != 0) {
                    continue;
                }
                if (!std::isfinite(in.weights.data(j)) || in.weights.data(j) <= 0.0) {
                    continue;
                }
                in.weights.data(j) *= penalty_factor;
                n_weighted++;
            }

            penalty_summary.push_back(WeightCorrPenaltyDiagSummary{
                .nw = nw,
                .n_det_input = n_det_group,
                .n_det_candidates = n_candidates,
                .n_det_used = n_used,
                .n_det_weighted = n_weighted,
                .sample_step = sample_step,
                .pair_med_abs_corr = finite_or_nan(pair_med_abs_corr),
                .cm_el_abs_corr = finite_or_nan(cm_el_abs_corr),
                .cm_low_mid_ratio = finite_or_nan(cm_low_mid_ratio),
                .severity = severity,
                .penalty_factor = penalty_factor,
            });

            logger->info(
                "weight corr_penalty scan={} nw={} dets_in={} candidates={} used={} weighted={} "
                "pair_med_abs_corr={} cm_el_abs_corr={} cm_low_mid_ratio={} severity={} factor={}",
                scan_index_1based, nw, n_det_group, n_candidates, n_used, n_weighted,
                finite_or_nan(pair_med_abs_corr), finite_or_nan(cm_el_abs_corr),
                finite_or_nan(cm_low_mid_ratio), severity, penalty_factor);
        }
        weight_corr_penalty_summary_by_scan[in.index.data] = std::move(penalty_summary);
    }

    if (busy_row_suppression.enabled) {
        std::unordered_map<Eigen::Index, const SecondPassDiagSummary *> second_pass_by_nw;
        const auto second_pass_it = second_pass_summary_by_scan.find(in.index.data);
        if (second_pass_it != second_pass_summary_by_scan.end()) {
            second_pass_by_nw.reserve(second_pass_it->second.size());
            for (const auto &row : second_pass_it->second) {
                second_pass_by_nw[row.nw] = &row;
            }
        } else {
            logger->warn(
                "weighting.busy_row_suppression enabled but no second-pass diagnostics were available for scan={}",
                scan_index_1based);
        }

        const double suppression_factor = std::clamp(busy_row_suppression.factor, 0.0, 1.0);
        std::vector<BusyRowSuppressionDiagSummary> suppression_summary;
        suppression_summary.reserve(static_cast<std::size_t>(nw_limits.size()));

        for (const auto &[nw, limits] : nw_limits) {
            const auto [start_index, end_index] = limits;
            BusyRowSuppressionDiagSummary summary;
            summary.nw = nw;

            const auto second_pass_nw_it = second_pass_by_nw.find(nw);
            if (second_pass_nw_it != second_pass_by_nw.end() && second_pass_nw_it->second != nullptr) {
                const auto &diag = *second_pass_nw_it->second;
                summary.busy_network_vetoed = diag.busy_network_vetoed;
                summary.n_candidate_clusters = diag.n_candidate_clusters;
                summary.max_unflagged_residual_z = finite_or_nan(diag.max_unflagged_residual_z);
            }

            const bool busy_ok = !busy_row_suppression.require_busy_veto || summary.busy_network_vetoed;
            const bool candidate_ok = summary.n_candidate_clusters >= busy_row_suppression.min_candidate_clusters;
            const bool residual_ok = std::isfinite(summary.max_unflagged_residual_z) &&
                                     summary.max_unflagged_residual_z >= busy_row_suppression.min_max_unflagged_residual_z;
            const bool should_suppress = busy_ok && candidate_ok && residual_ok && suppression_factor < 1.0;

            if (should_suppress) {
                for (Eigen::Index j = start_index; j < end_index; ++j) {
                    if (apt["flag"](j) != 0) {
                        continue;
                    }
                    if (!std::isfinite(in.weights.data(j)) || in.weights.data(j) <= 0.0) {
                        continue;
                    }
                    in.weights.data(j) *= suppression_factor;
                    summary.n_det_weighted++;
                }
            }
            summary.applied = should_suppress && summary.n_det_weighted > 0;
            summary.factor = summary.applied ? suppression_factor : 1.0;

            if (summary.applied) {
                logger->info(
                    "weight busy_row_suppression scan={} nw={} busy={} n_candidate_clusters={} "
                    "max_unflagged_residual_z={} factor={} weighted={}",
                    scan_index_1based, nw, summary.busy_network_vetoed, summary.n_candidate_clusters,
                    summary.max_unflagged_residual_z, summary.factor, summary.n_det_weighted);
            }

            suppression_summary.push_back(summary);
        }

        busy_row_suppression_summary_by_scan[in.index.data] = std::move(suppression_summary);
    }

    Eigen::Index n_apt_unflagged = 0;
    Eigen::Index n_nonfinite = 0;
    Eigen::Index n_positive = 0;
    Eigen::Index n_zero = 0;
    Eigen::Index n_negative = 0;
    for (Eigen::Index i = 0; i < n_dets; ++i) {
        if (apt["flag"](i) == 0) {
            n_apt_unflagged++;
        }
        const auto w = in.weights.data(i);
        if (!std::isfinite(w)) {
            n_nonfinite++;
        } else if (w > 0) {
            n_positive++;
        } else if (w == 0) {
            n_zero++;
        } else {
            n_negative++;
        }
    }
    logger->info(
        "weight calc summary scan={} type={} n_dets={} apt_unflagged={} "
        "positive={} zero={} negative={} nonfinite={}",
        scan_index_1based, weighting_type, n_dets, n_apt_unflagged, n_positive,
        n_zero, n_negative, n_nonfinite);
}

template <typename calib_t>
auto PTCProc::reset_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, calib_t &calib, std::string map_grouping) {

    // make a copy of the calib class for flagging
    calib_t calib_scan = calib;

    const auto scan_index_1based = static_cast<long long>(in.index.data) + 1;
    static std::atomic<long long> reset_weights_call_counter{0};
    const auto reset_call_id = ++reset_weights_call_counter;

    // only need to run if median weight factor >=1
    if (med_weight_factor >= 1 || lower_weight_factor > 0 || upper_weight_factor > 0) {
        // number of detectors
        Eigen::Index n_dets = in.scans.data.cols();

        // get group limits
        auto grp_limits = get_grouping("array", calib, n_dets);

        logger->info(
            "resetting weights call={} scan={} map_grouping={} n_dets={} "
            "med_weight_factor={} lower_weight_factor={} upper_weight_factor={}",
            reset_call_id, scan_index_1based, map_grouping, n_dets,
            med_weight_factor, lower_weight_factor, upper_weight_factor);

        // collect detectors that are un-flagged and have non-zero weights
        for (auto const& [key, val] : grp_limits) {
            // weights for current group
            auto grp_weights = in.weights.data(Eigen::seq(std::get<0>(grp_limits[key]),
                                                         std::get<1>(grp_limits[key])-1));
            const auto group_start = std::get<0>(grp_limits[key]);
            const auto group_end = std::get<1>(grp_limits[key]);
            const auto n_group_dets = group_end - group_start;
            // number of unflagged detectors, and unflagged with positive weights
            Eigen::Index n_unflagged = 0;
            Eigen::Index n_good_dets = 0;
            Eigen::Index n_nonfinite_weights = 0;
            Eigen::Index n_nonpositive_unflagged = 0;
            // start index of current group
            Eigen::Index j = group_start;

            // loop through detectors in current group
            for (Eigen::Index m=0; m<grp_weights.size(); ++m) {
                if (!std::isfinite(grp_weights(m))) {
                    n_nonfinite_weights++;
                }
                // count unflagged detectors
                if (calib.apt["flag"](j)==0) {
                    n_unflagged++;
                    if (grp_weights(m) > 0) {
                        n_good_dets++;
                    } else {
                        n_nonpositive_unflagged++;
                    }
                }
                j++;
            }

            // to hold good detectors
            Eigen::VectorXd good_wt;

            // if good detectors were found
            if (n_good_dets>0) {
                good_wt.resize(n_good_dets);

                // remove flagged dets
                j = std::get<0>(grp_limits[key]);
                Eigen::Index k = 0;
                for (Eigen::Index m=0; m<grp_weights.size(); ++m) {
                    if (calib.apt["flag"](j)==0 && grp_weights(m)>0) {
                        good_wt(k) = grp_weights(m);
                        k++;
                    }
                    j++;
                }
            }
            // otherwise just use all detectors
            else {
                good_wt = grp_weights;
            }

            // get median weight
            auto med_wt = tula::alg::median(good_wt);
            const auto lower_limit =
                lower_weight_factor != 0 ? lower_weight_factor * med_wt : 0.0;
            const auto upper_limit =
                upper_weight_factor != 0 ? upper_weight_factor * med_wt : 0.0;
            // store median weights
            in.median_weights.data.push_back(med_wt);

            int outliers = 0;
            int n_dets_low = 0;
            int n_dets_high = 0;

            // start index of current group
            j = group_start;
            // loop through detectors in current group
            for (Eigen::Index m=0; m<grp_weights.size(); ++m) {
                // if detector weight is med_weight_factor times larger than med_wt
                if (med_weight_factor >=1 && in.weights.data(j) > med_weight_factor*med_wt) {
                    // reset high weights to median
                    in.weights.data(j) = med_wt;
                    outliers++;
                }

                // only run if unflagged already
                if (calib.apt["flag"](j)==0) {
                    // flag those below limit
                    if ((in.weights.data(j) < (lower_weight_factor*med_wt)) && lower_weight_factor!=0) {
                        if (map_grouping!="detector") {
                            in.flags.data.col(j).setOnes();
                        }
                        else {
                            calib_scan.apt["flag"](j) = 1;
                        }
                        in.n_dets_low++;
                        n_dets_low++;
                    }

                    // flag those above limit
                    if ((in.weights.data(j) > (upper_weight_factor*med_wt)) && upper_weight_factor!=0) {
                        if (map_grouping!="detector") {
                            in.flags.data.col(j).setOnes();
                        }
                        else {
                            calib_scan.apt["flag"](j) = 1;
                        }
                        in.n_dets_high++;
                        n_dets_high++;
                    }
                }
                j++;
            }
            logger->info(
                "weight audit call={} scan={} array={} idx_range=[{}, {}) "
                "group_dets={} apt_unflagged={} apt_flagged={} "
                "positive_unflagged={} nonpositive_unflagged={} nonfinite_weights={} "
                "median_weight={} lower_limit={} upper_limit={}",
                reset_call_id, scan_index_1based, key, group_start, group_end,
                n_group_dets, n_unflagged, n_group_dets - n_unflagged, n_good_dets,
                n_nonpositive_unflagged, n_nonfinite_weights, med_wt, lower_limit,
                upper_limit);
            logger->info(
                "weight flags call={} scan={} array={} outlier_resets={} "
                "below_limit={}/{} above_limit={}/{}",
                reset_call_id, scan_index_1based, key, outliers, n_dets_low,
                n_unflagged, n_dets_high, n_unflagged);

            // sanity checks for impossible counter combinations
            if (n_unflagged < 0 || n_unflagged > n_group_dets ||
                n_good_dets < 0 || n_good_dets > n_unflagged ||
                n_dets_low < 0 || n_dets_low > n_unflagged ||
                n_dets_high < 0 || n_dets_high > n_unflagged) {
                logger->error(
                    "weight counter invariant failure call={} scan={} array={} "
                    "group_dets={} apt_unflagged={} positive_unflagged={} "
                    "below_count={} above_count={} outlier_count={}",
                    reset_call_id, scan_index_1based, key, n_group_dets,
                    n_unflagged, n_good_dets, n_dets_low, n_dets_high, outliers);
                const auto n_dump = std::min<Eigen::Index>(grp_weights.size(), 10);
                for (Eigen::Index m = 0; m < n_dump; ++m) {
                    const auto det_index = group_start + m;
                    logger->error(
                        "weight counter dump call={} scan={} array={} m={} det_index={} apt_flag={} weight={}",
                        reset_call_id, scan_index_1based, key, m, det_index,
                        calib.apt["flag"](det_index), in.weights.data(det_index));
                }
                std::exit(EXIT_FAILURE);
            }
        }

        // set up scan calib
        calib_scan.setup();
    }
    return std::move(calib_scan);
}

template <typename calib_t, typename pointing_offset_t>
void PTCProc::append_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, std::string filepath, std::string map_grouping,
                              std::string &pixel_axes, pointing_offset_t &pointing_offsets_arcsec, calib_t &calib,
                              bool apply_det_offsets, Eigen::Index scan_row_index) {

    using netCDF::NcDim;
    using netCDF::NcFile;
    using netCDF::NcType;
    using netCDF::NcVar;
    using namespace netCDF::exceptions;

    try {
        // open netcdf file
        predefs::suppress_hdf5_diagnostics_for_this_thread();
        std::lock_guard<std::mutex> lock(predefs::netcdf_io_mutex());
        NcFile fo(filepath, netCDF::NcFile::write);
        const auto n_pts_before_append = fo.getDim("n_pts").getSize();
        const auto n_dets_before_append = fo.getDim("n_dets").getSize();

        // append common time chunk variables
        append_base_to_netcdf(fo, in, map_grouping, pixel_axes, pointing_offsets_arcsec, calib, apply_det_offsets,
                              scan_row_index);

        // get dimensions
        NcDim n_dets_dim = fo.getDim("n_dets");

        // number of detectors currently in file
        unsigned long n_dets_exists = n_dets_dim.getSize();

        // append weights
        const auto scan_row = static_cast<unsigned long>((scan_row_index >= 0) ? scan_row_index : in.index.data);
        std::vector<std::size_t> start_index_weights = {scan_row, 0};
        std::vector<std::size_t> size_weights = {1, n_dets_exists};

        // get weight variable
        NcVar weights_v = fo.getVar("weights");

        // add weights to tod output
        weights_v.putVar(start_index_weights, size_weights, in.weights.data.data());

        const auto second_pass_summary_it = second_pass_summary_by_scan.find(in.index.data);
        const auto second_pass_added_it = second_pass_added_flags_by_scan.find(in.index.data);
        NcVar second_pass_added_v = fo.getVar("ptc_second_pass_added_flag");
        if (!second_pass_added_v.isNull() && second_pass_added_it != second_pass_added_flags_by_scan.end()) {
            std::vector<std::size_t> start_index = {n_pts_before_append, 0};
            std::vector<std::size_t> size = {1, n_dets_before_append};
            const auto &added = second_pass_added_it->second;
            const auto n_rows = std::min<unsigned long>(
                static_cast<unsigned long>(added.rows()),
                static_cast<unsigned long>(in.scans.data.rows()));
            for (unsigned long i = 0; i < n_rows; ++i) {
                start_index[0] = n_pts_before_append + i;
                Eigen::Matrix<signed char, 1, Eigen::Dynamic> row = added.row(static_cast<Eigen::Index>(i));
                second_pass_added_v.putVar(start_index, size, row.data());
            }
        }

        const auto corr_groups_it = corr_nw_group_ids_by_scan.find(in.index.data);
        const auto corr_summary_it = corr_nw_summary_by_scan.find(in.index.data);
        const auto weight_corr_penalty_it = weight_corr_penalty_summary_by_scan.find(in.index.data);
        const auto busy_row_suppression_it = busy_row_suppression_summary_by_scan.find(in.index.data);
        const auto adaptive_selector_it = adaptive_selector_summary_by_scan.find(in.index.data);
        const int corr_fill_value = -2147483647;

        // optional corr_nw diagnostics: detector group IDs per scan x detector
        NcVar corr_group_id_v = fo.getVar("corr_nw_group_id");
        if (!corr_group_id_v.isNull()) {
            std::vector<int> group_ids(static_cast<std::size_t>(n_dets_exists), corr_fill_value);
            if (corr_groups_it != corr_nw_group_ids_by_scan.end()) {
                const auto &gid = corr_groups_it->second;
                const auto n_copy = std::min<unsigned long>(n_dets_exists, static_cast<unsigned long>(gid.size()));
                for (unsigned long i = 0; i < n_copy; ++i) {
                    group_ids[static_cast<std::size_t>(i)] = static_cast<int>(gid(static_cast<Eigen::Index>(i)));
                }
            }
            corr_group_id_v.putVar(start_index_weights, size_weights, group_ids.data());
        }

        // optional corr_nw diagnostics: per-network summaries per scan
        NcVar corr_n_groups_v = fo.getVar("corr_nw_n_groups");
        if (!corr_n_groups_v.isNull()) {
            NcDim n_nws_dim = fo.getDim("n_nws_corr");
            if (!n_nws_dim.isNull()) {
                const auto n_nws = n_nws_dim.getSize();
                std::vector<int> v_n_groups(n_nws, corr_fill_value);
                std::vector<int> v_n_groups_raw(n_nws, corr_fill_value);
                std::vector<int> v_n_det_input(n_nws, corr_fill_value);
                std::vector<int> v_n_det_candidates(n_nws, corr_fill_value);
                std::vector<int> v_n_det_used(n_nws, corr_fill_value);
                std::vector<int> v_n_det_grouped(n_nws, corr_fill_value);
                std::vector<int> v_n_det_ungrouped(n_nws, corr_fill_value);
                std::vector<int> v_sample_step(n_nws, corr_fill_value);

                std::unordered_map<Eigen::Index, std::size_t> nw_to_index;
                nw_to_index.reserve(static_cast<std::size_t>(calib.nws.size()));
                for (Eigen::Index i = 0; i < calib.nws.size(); ++i) {
                    nw_to_index[calib.nws(i)] = static_cast<std::size_t>(i);
                }

                if (corr_summary_it != corr_nw_summary_by_scan.end()) {
                    for (const auto &row : corr_summary_it->second) {
                        const auto it = nw_to_index.find(row.nw);
                        if (it == nw_to_index.end() || it->second >= n_nws) {
                            continue;
                        }
                        const auto j = it->second;
                        v_n_groups[j] = static_cast<int>(row.n_groups_final);
                        v_n_groups_raw[j] = static_cast<int>(row.n_groups_raw);
                        v_n_det_input[j] = static_cast<int>(row.n_det_input);
                        v_n_det_candidates[j] = static_cast<int>(row.n_det_candidates);
                        v_n_det_used[j] = static_cast<int>(row.n_det_used);
                        v_n_det_grouped[j] = static_cast<int>(row.n_det_grouped);
                        v_n_det_ungrouped[j] = static_cast<int>(row.n_det_ungrouped);
                        v_sample_step[j] = static_cast<int>(row.sample_step);
                    }
                }

                std::vector<std::size_t> start_scan_nw = {scan_row, 0};
                std::vector<std::size_t> size_scan_nw = {1, n_nws};

                corr_n_groups_v.putVar(start_scan_nw, size_scan_nw, v_n_groups.data());
                fo.getVar("corr_nw_n_groups_raw").putVar(start_scan_nw, size_scan_nw, v_n_groups_raw.data());
                fo.getVar("corr_nw_n_det_input").putVar(start_scan_nw, size_scan_nw, v_n_det_input.data());
                fo.getVar("corr_nw_n_det_candidates").putVar(start_scan_nw, size_scan_nw, v_n_det_candidates.data());
                fo.getVar("corr_nw_n_det_used").putVar(start_scan_nw, size_scan_nw, v_n_det_used.data());
                fo.getVar("corr_nw_n_det_grouped").putVar(start_scan_nw, size_scan_nw, v_n_det_grouped.data());
                fo.getVar("corr_nw_n_det_ungrouped").putVar(start_scan_nw, size_scan_nw, v_n_det_ungrouped.data());
                fo.getVar("corr_nw_sample_step").putVar(start_scan_nw, size_scan_nw, v_sample_step.data());
            }
        }

        // optional diagnostics: per-network weight penalty summaries per scan
        NcVar wcorr_factor_v = fo.getVar("weight_corr_penalty_factor");
        if (!wcorr_factor_v.isNull()) {
            NcDim n_nws_dim = fo.getDim("n_nws_wcorr");
            if (!n_nws_dim.isNull()) {
                const auto n_nws = n_nws_dim.getSize();
                const double fill_double = std::numeric_limits<double>::quiet_NaN();
                std::vector<double> v_factor(n_nws, fill_double);
                std::vector<double> v_severity(n_nws, fill_double);
                std::vector<double> v_pair_corr(n_nws, fill_double);
                std::vector<double> v_cm_el_corr(n_nws, fill_double);
                std::vector<double> v_cm_low_mid(n_nws, fill_double);
                std::vector<int> v_n_det_input(n_nws, corr_fill_value);
                std::vector<int> v_n_det_candidates(n_nws, corr_fill_value);
                std::vector<int> v_n_det_used(n_nws, corr_fill_value);
                std::vector<int> v_n_det_weighted(n_nws, corr_fill_value);
                std::vector<int> v_sample_step(n_nws, corr_fill_value);

                std::unordered_map<Eigen::Index, std::size_t> nw_to_index;
                nw_to_index.reserve(static_cast<std::size_t>(calib.nws.size()));
                for (Eigen::Index i = 0; i < calib.nws.size(); ++i) {
                    nw_to_index[calib.nws(i)] = static_cast<std::size_t>(i);
                }

                if (weight_corr_penalty_it != weight_corr_penalty_summary_by_scan.end()) {
                    for (const auto &row : weight_corr_penalty_it->second) {
                        const auto it = nw_to_index.find(row.nw);
                        if (it == nw_to_index.end() || it->second >= n_nws) {
                            continue;
                        }
                        const auto j = it->second;
                        v_factor[j] = row.penalty_factor;
                        v_severity[j] = row.severity;
                        v_pair_corr[j] = row.pair_med_abs_corr;
                        v_cm_el_corr[j] = row.cm_el_abs_corr;
                        v_cm_low_mid[j] = row.cm_low_mid_ratio;
                        v_n_det_input[j] = static_cast<int>(row.n_det_input);
                        v_n_det_candidates[j] = static_cast<int>(row.n_det_candidates);
                        v_n_det_used[j] = static_cast<int>(row.n_det_used);
                        v_n_det_weighted[j] = static_cast<int>(row.n_det_weighted);
                        v_sample_step[j] = static_cast<int>(row.sample_step);
                    }
                }

                std::vector<std::size_t> start_scan_nw = {scan_row, 0};
                std::vector<std::size_t> size_scan_nw = {1, n_nws};

                wcorr_factor_v.putVar(start_scan_nw, size_scan_nw, v_factor.data());
                fo.getVar("weight_corr_penalty_severity").putVar(start_scan_nw, size_scan_nw, v_severity.data());
                fo.getVar("weight_corr_penalty_pair_med_abs_corr").putVar(start_scan_nw, size_scan_nw, v_pair_corr.data());
                fo.getVar("weight_corr_penalty_cm_el_abs_corr").putVar(start_scan_nw, size_scan_nw, v_cm_el_corr.data());
                fo.getVar("weight_corr_penalty_cm_low_mid_ratio").putVar(start_scan_nw, size_scan_nw, v_cm_low_mid.data());
                fo.getVar("weight_corr_penalty_n_det_input").putVar(start_scan_nw, size_scan_nw, v_n_det_input.data());
                fo.getVar("weight_corr_penalty_n_det_candidates").putVar(start_scan_nw, size_scan_nw, v_n_det_candidates.data());
                fo.getVar("weight_corr_penalty_n_det_used").putVar(start_scan_nw, size_scan_nw, v_n_det_used.data());
                fo.getVar("weight_corr_penalty_n_det_weighted").putVar(start_scan_nw, size_scan_nw, v_n_det_weighted.data());
                fo.getVar("weight_corr_penalty_sample_step").putVar(start_scan_nw, size_scan_nw, v_sample_step.data());
            }
        }

        NcVar wbusy_applied_v = fo.getVar("weight_busy_row_suppression_applied");
        if (!wbusy_applied_v.isNull()) {
            NcDim n_nws_dim = fo.getDim("n_nws_busy_row_suppression");
            if (!n_nws_dim.isNull()) {
                const auto n_nws = n_nws_dim.getSize();
                const double fill_double = std::numeric_limits<double>::quiet_NaN();
                std::vector<int> v_applied(n_nws, corr_fill_value);
                std::vector<int> v_busy(n_nws, corr_fill_value);
                std::vector<int> v_n_candidate_clusters(n_nws, corr_fill_value);
                std::vector<int> v_n_det_weighted(n_nws, corr_fill_value);
                std::vector<double> v_factor(n_nws, fill_double);
                std::vector<double> v_max_resid_z(n_nws, fill_double);

                std::unordered_map<Eigen::Index, std::size_t> nw_to_index;
                nw_to_index.reserve(static_cast<std::size_t>(calib.nws.size()));
                for (Eigen::Index i = 0; i < calib.nws.size(); ++i) {
                    nw_to_index[calib.nws(i)] = static_cast<std::size_t>(i);
                }

                if (busy_row_suppression_it != busy_row_suppression_summary_by_scan.end()) {
                    for (const auto &row : busy_row_suppression_it->second) {
                        const auto it = nw_to_index.find(row.nw);
                        if (it == nw_to_index.end() || it->second >= n_nws) {
                            continue;
                        }
                        const auto j = it->second;
                        v_applied[j] = row.applied ? 1 : 0;
                        v_busy[j] = row.busy_network_vetoed ? 1 : 0;
                        v_n_candidate_clusters[j] = static_cast<int>(row.n_candidate_clusters);
                        v_n_det_weighted[j] = static_cast<int>(row.n_det_weighted);
                        v_factor[j] = row.factor;
                        v_max_resid_z[j] = row.max_unflagged_residual_z;
                    }
                }

                std::vector<std::size_t> start_scan_nw = {scan_row, 0};
                std::vector<std::size_t> size_scan_nw = {1, n_nws};
                wbusy_applied_v.putVar(start_scan_nw, size_scan_nw, v_applied.data());
                fo.getVar("weight_busy_row_suppression_busy_network_vetoed").putVar(start_scan_nw, size_scan_nw, v_busy.data());
                fo.getVar("weight_busy_row_suppression_n_candidate_clusters").putVar(start_scan_nw, size_scan_nw, v_n_candidate_clusters.data());
                fo.getVar("weight_busy_row_suppression_n_det_weighted").putVar(start_scan_nw, size_scan_nw, v_n_det_weighted.data());
                fo.getVar("weight_busy_row_suppression_factor").putVar(start_scan_nw, size_scan_nw, v_factor.data());
                fo.getVar("weight_busy_row_suppression_max_unflagged_residual_z").putVar(start_scan_nw, size_scan_nw, v_max_resid_z.data());
            }
        }

        NcVar adaptive_chosen_k_v = fo.getVar("adaptive_pca_chosen_k");
        if (!adaptive_chosen_k_v.isNull()) {
            NcDim n_nws_dim = fo.getDim("n_nws_adaptive_pca");
            if (!n_nws_dim.isNull()) {
                const auto n_nws = n_nws_dim.getSize();
                const double fill_double = std::numeric_limits<double>::quiet_NaN();
                std::vector<int> v_selector_used(n_nws, corr_fill_value);
                std::vector<int> v_selector_fallback(n_nws, corr_fill_value);
                std::vector<int> v_baseline_k(n_nws, corr_fill_value);
                std::vector<int> v_chosen_k(n_nws, corr_fill_value);
                std::vector<int> v_runnerup_k(n_nws, corr_fill_value);
                std::vector<int> v_n_candidates(n_nws, corr_fill_value);
                std::vector<int> v_n_det_input(n_nws, corr_fill_value);
                std::vector<int> v_n_det_used(n_nws, corr_fill_value);
                std::vector<int> v_n_time_used(n_nws, corr_fill_value);
                std::vector<int> v_sample_step(n_nws, corr_fill_value);
                std::vector<double> v_chosen_score(n_nws, fill_double);
                std::vector<double> v_runnerup_score(n_nws, fill_double);
                std::vector<double> v_score_margin(n_nws, fill_double);
                std::vector<double> v_chosen_med_abs_corr(n_nws, fill_double);
                std::vector<double> v_chosen_cm_low_mid_ratio(n_nws, fill_double);
                std::vector<double> v_chosen_tail4_binom_z(n_nws, fill_double);
                std::vector<double> v_chosen_top_mode_frac(n_nws, fill_double);
                std::vector<double> v_eig_solve_msec(n_nws, fill_double);
                std::vector<double> v_candidate_eval_msec(n_nws, fill_double);
                std::vector<double> v_total_msec(n_nws, fill_double);

                std::unordered_map<Eigen::Index, std::size_t> nw_to_index;
                nw_to_index.reserve(static_cast<std::size_t>(calib.nws.size()));
                for (Eigen::Index i = 0; i < calib.nws.size(); ++i) {
                    nw_to_index[calib.nws(i)] = static_cast<std::size_t>(i);
                }

                if (adaptive_selector_it != adaptive_selector_summary_by_scan.end()) {
                    for (const auto &row : adaptive_selector_it->second) {
                        const auto it = nw_to_index.find(row.nw);
                        if (it == nw_to_index.end() || it->second >= n_nws) {
                            continue;
                        }
                        const auto j = it->second;
                        v_selector_used[j] = row.selector_used;
                        v_selector_fallback[j] = row.selector_fallback;
                        v_baseline_k[j] = static_cast<int>(row.baseline_k);
                        v_chosen_k[j] = static_cast<int>(row.chosen_k);
                        v_runnerup_k[j] = static_cast<int>(row.runnerup_k);
                        v_n_candidates[j] = static_cast<int>(row.n_candidates);
                        v_n_det_input[j] = static_cast<int>(row.n_det_input);
                        v_n_det_used[j] = static_cast<int>(row.n_det_used);
                        v_n_time_used[j] = static_cast<int>(row.n_time_used);
                        v_sample_step[j] = static_cast<int>(row.sample_step);
                        v_chosen_score[j] = row.chosen_score;
                        v_runnerup_score[j] = row.runnerup_score;
                        v_score_margin[j] = row.score_margin;
                        v_chosen_med_abs_corr[j] = row.chosen_med_abs_corr;
                        v_chosen_cm_low_mid_ratio[j] = row.chosen_cm_low_mid_ratio;
                        v_chosen_tail4_binom_z[j] = row.chosen_tail4_binom_z;
                        v_chosen_top_mode_frac[j] = row.chosen_top_mode_frac;
                        v_eig_solve_msec[j] = row.eig_solve_msec;
                        v_candidate_eval_msec[j] = row.candidate_eval_msec;
                        v_total_msec[j] = row.total_msec;
                    }
                }

                std::vector<std::size_t> start_scan_nw = {scan_row, 0};
                std::vector<std::size_t> size_scan_nw = {1, n_nws};
                fo.getVar("adaptive_pca_selector_used").putVar(start_scan_nw, size_scan_nw, v_selector_used.data());
                fo.getVar("adaptive_pca_selector_fallback").putVar(start_scan_nw, size_scan_nw, v_selector_fallback.data());
                fo.getVar("adaptive_pca_baseline_k").putVar(start_scan_nw, size_scan_nw, v_baseline_k.data());
                adaptive_chosen_k_v.putVar(start_scan_nw, size_scan_nw, v_chosen_k.data());
                fo.getVar("adaptive_pca_runnerup_k").putVar(start_scan_nw, size_scan_nw, v_runnerup_k.data());
                fo.getVar("adaptive_pca_n_candidates").putVar(start_scan_nw, size_scan_nw, v_n_candidates.data());
                fo.getVar("adaptive_pca_n_det_input").putVar(start_scan_nw, size_scan_nw, v_n_det_input.data());
                fo.getVar("adaptive_pca_n_det_used").putVar(start_scan_nw, size_scan_nw, v_n_det_used.data());
                fo.getVar("adaptive_pca_n_time_used").putVar(start_scan_nw, size_scan_nw, v_n_time_used.data());
                fo.getVar("adaptive_pca_sample_step").putVar(start_scan_nw, size_scan_nw, v_sample_step.data());
                fo.getVar("adaptive_pca_chosen_score").putVar(start_scan_nw, size_scan_nw, v_chosen_score.data());
                fo.getVar("adaptive_pca_runnerup_score").putVar(start_scan_nw, size_scan_nw, v_runnerup_score.data());
                fo.getVar("adaptive_pca_score_margin").putVar(start_scan_nw, size_scan_nw, v_score_margin.data());
                fo.getVar("adaptive_pca_chosen_med_abs_corr").putVar(start_scan_nw, size_scan_nw, v_chosen_med_abs_corr.data());
                fo.getVar("adaptive_pca_chosen_cm_low_mid_ratio").putVar(start_scan_nw, size_scan_nw, v_chosen_cm_low_mid_ratio.data());
                fo.getVar("adaptive_pca_chosen_tail4_binom_z").putVar(start_scan_nw, size_scan_nw, v_chosen_tail4_binom_z.data());
                fo.getVar("adaptive_pca_chosen_top_mode_frac").putVar(start_scan_nw, size_scan_nw, v_chosen_top_mode_frac.data());
                fo.getVar("adaptive_pca_eig_solve_msec").putVar(start_scan_nw, size_scan_nw, v_eig_solve_msec.data());
                fo.getVar("adaptive_pca_candidate_eval_msec").putVar(start_scan_nw, size_scan_nw, v_candidate_eval_msec.data());
                fo.getVar("adaptive_pca_total_msec").putVar(start_scan_nw, size_scan_nw, v_total_msec.data());
            }
        }

        NcVar second_pass_busy_v = fo.getVar("ptc_second_pass_busy_network_vetoed");
        if (!second_pass_busy_v.isNull()) {
            NcDim n_nws_dim = fo.getDim("n_nws_ptc_second_pass");
            if (!n_nws_dim.isNull()) {
                const auto n_nws = n_nws_dim.getSize();
                const double fill_double = std::numeric_limits<double>::quiet_NaN();
                std::vector<int> v_busy(n_nws, corr_fill_value);
                std::vector<int> v_n_candidate_clusters(n_nws, corr_fill_value);
                std::vector<int> v_n_candidate_events(n_nws, corr_fill_value);
                std::vector<int> v_n_accepted_clusters(n_nws, corr_fill_value);
                std::vector<int> v_n_accepted_events(n_nws, corr_fill_value);
                std::vector<int> v_n_det_with_added_flags(n_nws, corr_fill_value);
                std::vector<int> v_max_resid_uid(n_nws, corr_fill_value);
                std::vector<int> v_top_cluster_sample(n_nws, corr_fill_value);
                std::vector<int> v_top_cluster_n_detectors(n_nws, corr_fill_value);
                std::vector<int> v_top_cluster_n_events(n_nws, corr_fill_value);
                std::vector<int> v_top_event_kind(n_nws, corr_fill_value);
                std::vector<int> v_top_event_uid(n_nws, corr_fill_value);
                std::vector<int> v_top_event_sample(n_nws, corr_fill_value);
                std::vector<double> v_existing_frac(n_nws, fill_double);
                std::vector<double> v_proposed_frac(n_nws, fill_double);
                std::vector<double> v_new_frac(n_nws, fill_double);
                std::vector<double> v_max_resid_z(n_nws, fill_double);
                std::vector<double> v_top_cluster_peak(n_nws, fill_double);
                std::vector<double> v_top_event_score(n_nws, fill_double);

                std::unordered_map<Eigen::Index, std::size_t> nw_to_index;
                nw_to_index.reserve(static_cast<std::size_t>(calib.nws.size()));
                for (Eigen::Index i = 0; i < calib.nws.size(); ++i) {
                    nw_to_index[calib.nws(i)] = static_cast<std::size_t>(i);
                }
                if (second_pass_summary_it != second_pass_summary_by_scan.end()) {
                    for (const auto &row : second_pass_summary_it->second) {
                        const auto it = nw_to_index.find(row.nw);
                        if (it == nw_to_index.end() || it->second >= n_nws) {
                            continue;
                        }
                        const auto j = it->second;
                        v_busy[j] = row.busy_network_vetoed ? 1 : 0;
                        v_n_candidate_clusters[j] = static_cast<int>(row.n_candidate_clusters);
                        v_n_candidate_events[j] = static_cast<int>(row.n_candidate_events);
                        v_n_accepted_clusters[j] = static_cast<int>(row.n_accepted_clusters);
                        v_n_accepted_events[j] = static_cast<int>(row.n_accepted_events);
                        v_n_det_with_added_flags[j] = static_cast<int>(row.n_det_with_added_flags);
                        v_max_resid_uid[j] = row.max_unflagged_residual_uid;
                        v_top_cluster_sample[j] = row.top_candidate_cluster_sample;
                        v_top_cluster_n_detectors[j] = static_cast<int>(row.top_candidate_cluster_n_detectors);
                        v_top_cluster_n_events[j] = static_cast<int>(row.top_candidate_cluster_n_events);
                        v_top_event_kind[j] = row.top_event.kind_code();
                        v_top_event_uid[j] = row.top_event_uid;
                        v_top_event_sample[j] = row.top_event.sample;
                        v_existing_frac[j] = row.existing_flagged_fraction;
                        v_proposed_frac[j] = row.proposed_flagged_fraction;
                        v_new_frac[j] = row.newly_flagged_fraction;
                        v_max_resid_z[j] = row.max_unflagged_residual_z;
                        v_top_cluster_peak[j] = row.top_candidate_cluster_peak_score;
                        v_top_event_score[j] = row.top_event.score;
                    }
                }

                std::vector<std::size_t> start_scan_nw = {scan_row, 0};
                std::vector<std::size_t> size_scan_nw = {1, n_nws};
                second_pass_busy_v.putVar(start_scan_nw, size_scan_nw, v_busy.data());
                fo.getVar("ptc_second_pass_n_candidate_clusters").putVar(start_scan_nw, size_scan_nw, v_n_candidate_clusters.data());
                fo.getVar("ptc_second_pass_n_candidate_events").putVar(start_scan_nw, size_scan_nw, v_n_candidate_events.data());
                fo.getVar("ptc_second_pass_n_accepted_clusters").putVar(start_scan_nw, size_scan_nw, v_n_accepted_clusters.data());
                fo.getVar("ptc_second_pass_n_accepted_events").putVar(start_scan_nw, size_scan_nw, v_n_accepted_events.data());
                fo.getVar("ptc_second_pass_n_det_with_added_flags").putVar(start_scan_nw, size_scan_nw, v_n_det_with_added_flags.data());
                fo.getVar("ptc_second_pass_max_unflagged_residual_uid").putVar(start_scan_nw, size_scan_nw, v_max_resid_uid.data());
                fo.getVar("ptc_second_pass_top_candidate_cluster_sample").putVar(start_scan_nw, size_scan_nw, v_top_cluster_sample.data());
                fo.getVar("ptc_second_pass_top_candidate_cluster_n_detectors").putVar(start_scan_nw, size_scan_nw, v_top_cluster_n_detectors.data());
                fo.getVar("ptc_second_pass_top_candidate_cluster_n_events").putVar(start_scan_nw, size_scan_nw, v_top_cluster_n_events.data());
                fo.getVar("ptc_second_pass_top_event_kind").putVar(start_scan_nw, size_scan_nw, v_top_event_kind.data());
                fo.getVar("ptc_second_pass_top_event_uid").putVar(start_scan_nw, size_scan_nw, v_top_event_uid.data());
                fo.getVar("ptc_second_pass_top_event_sample").putVar(start_scan_nw, size_scan_nw, v_top_event_sample.data());
                fo.getVar("ptc_second_pass_existing_flagged_fraction").putVar(start_scan_nw, size_scan_nw, v_existing_frac.data());
                fo.getVar("ptc_second_pass_proposed_flagged_fraction").putVar(start_scan_nw, size_scan_nw, v_proposed_frac.data());
                fo.getVar("ptc_second_pass_newly_flagged_fraction").putVar(start_scan_nw, size_scan_nw, v_new_frac.data());
                fo.getVar("ptc_second_pass_max_unflagged_residual_z").putVar(start_scan_nw, size_scan_nw, v_max_resid_z.data());
                fo.getVar("ptc_second_pass_top_candidate_cluster_peak_score").putVar(start_scan_nw, size_scan_nw, v_top_cluster_peak.data());
                fo.getVar("ptc_second_pass_top_event_score").putVar(start_scan_nw, size_scan_nw, v_top_event_score.data());
            }
        }

        // drop per-scan diagnostics once persisted to netCDF
        if (corr_groups_it != corr_nw_group_ids_by_scan.end()) {
            corr_nw_group_ids_by_scan.erase(corr_groups_it);
        }
        if (corr_summary_it != corr_nw_summary_by_scan.end()) {
            corr_nw_summary_by_scan.erase(corr_summary_it);
        }
        if (weight_corr_penalty_it != weight_corr_penalty_summary_by_scan.end()) {
            weight_corr_penalty_summary_by_scan.erase(weight_corr_penalty_it);
        }
        if (busy_row_suppression_it != busy_row_suppression_summary_by_scan.end()) {
            busy_row_suppression_summary_by_scan.erase(busy_row_suppression_it);
        }
        if (adaptive_selector_it != adaptive_selector_summary_by_scan.end()) {
            adaptive_selector_summary_by_scan.erase(adaptive_selector_it);
        }
        if (second_pass_summary_it != second_pass_summary_by_scan.end()) {
            second_pass_summary_by_scan.erase(second_pass_summary_it);
        }
        if (second_pass_added_it != second_pass_added_flags_by_scan.end()) {
            second_pass_added_flags_by_scan.erase(second_pass_added_it);
        }

        if (write_evals) {
            if (cleaner.n_calc <= 0 || in.evals.data.empty()) {
                logger->warn("n_calc=0 or evals empty; skipping eval/evec output");
                // sync file to make sure it gets updated
                fo.sync();
                // close file
                fo.close();
                logger->info("tod chunk written to {}", filepath);
                return;
            }
            // get number of eigenvalues to save
            NcDim n_eigs_dim = fo.getDim("n_eigs");
            netCDF::NcDim n_eig_grp_dim = fo.getDim("n_eig_grp");

            // if eigenvalue dimension is null, add it
            if (n_eig_grp_dim.isNull()) {
                n_eig_grp_dim = fo.addDim("n_eig_grp",in.evals.data[0].size());
            }

            // dimensions for eigenvalue data
            std::vector<netCDF::NcDim> eval_dims = {n_eig_grp_dim, n_eigs_dim};

            // loop through cleaner gropuing
            for (Eigen::Index i=0; i<in.evals.data.size(); ++i) {
                NcVar eval_v = fo.addVar("evals_" + cleaner.grouping[i] + "_" + std::to_string(i) +
                                             "_chunk_" + std::to_string(in.index.data), netCDF::ncDouble,eval_dims);
                std::vector<std::size_t> start_eig_index = {0, 0};
                std::vector<std::size_t> size = {1, TULA_SIZET(cleaner.n_calc)};

                // loop through eigenvalues in current group
                for (const auto &evals: in.evals.data[i]) {
                    eval_v.putVar(start_eig_index,size,evals.data());
                    start_eig_index[0] += 1;
                }
            }

            // number of dimensions for eigenvectors
            std::vector<netCDF::NcDim> eig_dims = {n_dets_dim, n_eigs_dim};

            // loop through cleaner gropuing
            for (Eigen::Index i=0; i<in.evecs.data.size(); ++i) {
                // start at first row and col
                std::vector<std::size_t> start_eig_index = {0, 0};

                NcVar evec_v = fo.addVar("evecs_" + cleaner.grouping[i] + "_" + std::to_string(i) + "_chunk_" +
                                             std::to_string(in.index.data),netCDF::ncDouble,eig_dims);

                // loop through eigenvectors in current group
                for (const auto &evecs: in.evecs.data[i]) {
                    std::vector<std::size_t> size = {TULA_SIZET(evecs.rows()), TULA_SIZET(cleaner.n_calc)};

                    // transpose eigenvectors
                    Eigen::MatrixXd ev = evecs.transpose();
                    evec_v.putVar(start_eig_index, size, ev.data());

                    // increment start
                    start_eig_index[0] += TULA_SIZET(evecs.rows());
                }
            }
        }

        // sync file to make sure it gets updated
        fo.sync();
        // close file
        fo.close();
        logger->info("tod chunk written to {}", filepath);

    } catch (NcException &e) {
        logger->error("{}", e.what());
    }
}

template <typename calib_t>
void PTCProc::append_diag_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, std::string filepath,
                                    calib_t &calib, Eigen::Index scan_row_index) {
    using netCDF::NcDim;
    using netCDF::NcFile;
    using netCDF::NcVar;
    using namespace netCDF::exceptions;

    try {
        predefs::suppress_hdf5_diagnostics_for_this_thread();
        std::lock_guard<std::mutex> lock(predefs::netcdf_io_mutex());
        NcFile fo(filepath, netCDF::NcFile::write);
        const auto scan_row = static_cast<unsigned long>((scan_row_index >= 0) ? scan_row_index : in.index.data);
        const auto n_dets = fo.getDim("n_dets").getSize();
        std::vector<std::size_t> start_index_det = {scan_row, 0};
        std::vector<std::size_t> size_det = {1, n_dets};

        std::vector<double> weights(static_cast<std::size_t>(n_dets), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> rms(static_cast<std::size_t>(n_dets), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> stddev(static_cast<std::size_t>(n_dets), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> median(static_cast<std::size_t>(n_dets), std::numeric_limits<double>::quiet_NaN());
        std::vector<double> flagged_frac(static_cast<std::size_t>(n_dets), std::numeric_limits<double>::quiet_NaN());
        auto window_diag_it = remove_bad_dets_window_summary_by_scan.find(in.index.data);
        const double n_pts = static_cast<double>(in.scans.data.rows());
        const auto n_copy = std::min<unsigned long>(n_dets, static_cast<unsigned long>(in.scans.data.cols()));
        for (unsigned long i = 0; i < n_copy; ++i) {
            const auto det = static_cast<Eigen::Index>(i);
            Eigen::VectorXd scans = in.scans.data.col(det);
            Eigen::Matrix<bool, Eigen::Dynamic, 1> flags = in.flags.data.col(det);
            weights[static_cast<std::size_t>(i)] = (det < in.weights.data.size()) ? in.weights.data(det) : std::numeric_limits<double>::quiet_NaN();
            rms[static_cast<std::size_t>(i)] = engine_utils::calc_rms(scans);
            stddev[static_cast<std::size_t>(i)] = engine_utils::calc_std_dev(scans);
            median[static_cast<std::size_t>(i)] = tula::alg::median(scans);
            flagged_frac[static_cast<std::size_t>(i)] =
                (n_pts > 0.0) ? flags.cast<double>().sum() / n_pts : std::numeric_limits<double>::quiet_NaN();
        }

        if (window_diag_it == remove_bad_dets_window_summary_by_scan.end()) {
            auto infer_dt_sec = [&]() {
                auto it = in.tel_data.data.find("TelTime");
                if (it == in.tel_data.data.end() || it->second.size() < 2) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                std::vector<double> dt;
                dt.reserve(static_cast<std::size_t>(it->second.size() - 1));
                for (Eigen::Index i = 1; i < it->second.size(); ++i) {
                    const double delta = it->second(i) - it->second(i - 1);
                    if (std::isfinite(delta) && delta > 0.0) {
                        dt.push_back(delta);
                    }
                }
                if (dt.empty()) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                return tula::alg::median(Eigen::Map<Eigen::VectorXd>(dt.data(), dt.size()));
            };
            auto vector_quantile = [](std::vector<double> values, double q) {
                if (values.empty()) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                std::sort(values.begin(), values.end());
                q = std::clamp(q, 0.0, 1.0);
                const double pos = q * static_cast<double>(values.size() - 1);
                const auto lo = static_cast<std::size_t>(std::floor(pos));
                const auto hi = static_cast<std::size_t>(std::ceil(pos));
                if (lo == hi) {
                    return values[lo];
                }
                const double frac = pos - static_cast<double>(lo);
                return values[lo] * (1.0 - frac) + values[hi] * frac;
            };
            const double dt_sec = infer_dt_sec();
            Eigen::Index window_samples = in.scans.data.rows();
            if (std::isfinite(dt_sec) && dt_sec > 0.0 && remove_bad_dets_window_sec > 0.0) {
                window_samples = std::max<Eigen::Index>(
                    8, static_cast<Eigen::Index>(std::llround(remove_bad_dets_window_sec / dt_sec)));
            }
            window_samples = std::min<Eigen::Index>(window_samples, in.scans.data.rows());
            window_samples = std::max<Eigen::Index>(1, window_samples);

            auto summarize_windows = [&](Eigen::Index det_index) {
                RemoveBadDetsWindowDiagSummary summary;
                if (det_index < 0 || det_index >= in.scans.data.cols()) {
                    return summary;
                }
                Eigen::VectorXd scans = in.scans.data.col(det_index);
                Eigen::Matrix<bool, Eigen::Dynamic, 1> flags = in.flags.data.col(det_index);
                if (active_map_grouping == "detector" && mask_radius_arcsec > 0.0) {
                    Eigen::Matrix<bool, Eigen::Dynamic, 1> masked_flags = flags;
                    double az_off = calib.apt["x_t"](det_index);
                    double el_off = calib.apt["y_t"](det_index);
                    auto [lat, lon] = engine_utils::calc_det_pointing(
                        in.tel_data.data,
                        az_off,
                        el_off,
                        std::string{"altaz"},
                        in.pointing_offsets_arcsec.data,
                        active_map_grouping);
                    double source_lat = 0.0;
                    double source_lon = 0.0;
                    resolve_mask_center_rad(in, calib, active_map_grouping, det_index,
                                            source_lat, source_lon);
                    const double radius_rad = mask_radius_arcsec * ASEC_TO_RAD;
                    for (Eigen::Index sample = 0; sample < masked_flags.size(); ++sample) {
                        const double dlat = lat(sample) - source_lat;
                        const double dlon = lon(sample) - source_lon;
                        if (std::sqrt(dlat * dlat + dlon * dlon) < radius_rad) {
                            masked_flags(sample) = true;
                        }
                    }
                    flags = masked_flags;
                }

                summary.n_total_windows = static_cast<int>((scans.size() + window_samples - 1) / window_samples);
                std::vector<double> inv_vars;
                std::vector<double> flagged_fracs;
                inv_vars.reserve(static_cast<std::size_t>(summary.n_total_windows));
                flagged_fracs.reserve(static_cast<std::size_t>(summary.n_total_windows));

                for (Eigen::Index start = 0; start < scans.size(); start += window_samples) {
                    const Eigen::Index stop = std::min<Eigen::Index>(scans.size(), start + window_samples);
                    const Eigen::Index len = stop - start;
                    if (len <= 0) {
                        continue;
                    }
                    int n_flagged = 0;
                    for (Eigen::Index i = start; i < stop; ++i) {
                        if (flags(i)) {
                            ++n_flagged;
                        }
                    }
                    const double flagged_window_frac =
                        static_cast<double>(n_flagged) / static_cast<double>(len);
                    flagged_fracs.push_back(flagged_window_frac);

                    Eigen::VectorXd scan_window = scans.segment(start, len);
                    Eigen::Matrix<bool, Eigen::Dynamic, 1> flag_window = flags.segment(start, len);
                    const double sigma = engine_utils::calc_std_dev(scan_window, flag_window);
                    if (std::isfinite(sigma) && sigma > 0.0) {
                        inv_vars.push_back(std::pow(sigma, -2));
                    }
                }

                summary.n_valid_windows = static_cast<int>(inv_vars.size());
                if (summary.n_total_windows > 0) {
                    summary.valid_window_fraction =
                        static_cast<double>(summary.n_valid_windows) /
                        static_cast<double>(summary.n_total_windows);
                }
                if (!inv_vars.empty()) {
                    summary.inv_var_median = vector_quantile(inv_vars, 0.5);
                    summary.inv_var_q10 = vector_quantile(inv_vars, 0.1);
                    summary.inv_var_q90 = vector_quantile(inv_vars, 0.9);
                }
                if (!flagged_fracs.empty()) {
                    summary.flagged_frac_median = vector_quantile(flagged_fracs, 0.5);
                    summary.flagged_frac_max = *std::max_element(flagged_fracs.begin(), flagged_fracs.end());
                    const auto n_heavy = std::count_if(
                        flagged_fracs.begin(), flagged_fracs.end(),
                        [](double v) { return std::isfinite(v) && v >= 0.5; });
                    summary.heavily_flagged_window_fraction =
                        static_cast<double>(n_heavy) /
                        static_cast<double>(flagged_fracs.size());
                }
                return summary;
            };

            auto &window_diag = remove_bad_dets_window_summary_by_scan[in.index.data];
            window_diag.assign(static_cast<std::size_t>(in.scans.data.cols()),
                               RemoveBadDetsWindowDiagSummary{});
            for (Eigen::Index det = 0; det < in.scans.data.cols(); ++det) {
                window_diag[static_cast<std::size_t>(det)] = summarize_windows(det);
            }
            window_diag_it = remove_bad_dets_window_summary_by_scan.find(in.index.data);
        }

        fo.getVar("ptc_detector_weight").putVar(start_index_det, size_det, weights.data());
        fo.getVar("ptc_detector_rms").putVar(start_index_det, size_det, rms.data());
        fo.getVar("ptc_detector_stddev").putVar(start_index_det, size_det, stddev.data());
        fo.getVar("ptc_detector_median").putVar(start_index_det, size_det, median.data());
        fo.getVar("ptc_detector_flagged_fraction").putVar(start_index_det, size_det, flagged_frac.data());

        auto window_double_values = [&](auto getter) {
            std::vector<double> values(static_cast<std::size_t>(n_dets), std::numeric_limits<double>::quiet_NaN());
            if (window_diag_it != remove_bad_dets_window_summary_by_scan.end()) {
                const auto n_copy_diag = std::min<std::size_t>(
                    static_cast<std::size_t>(n_dets), window_diag_it->second.size());
                for (std::size_t i = 0; i < n_copy_diag; ++i) {
                    values[i] = getter(window_diag_it->second[i]);
                }
            }
            return values;
        };
        auto window_int_values = [&](auto getter) {
            std::vector<int> values(static_cast<std::size_t>(n_dets), -2147483647);
            if (window_diag_it != remove_bad_dets_window_summary_by_scan.end()) {
                const auto n_copy_diag = std::min<std::size_t>(
                    static_cast<std::size_t>(n_dets), window_diag_it->second.size());
                for (std::size_t i = 0; i < n_copy_diag; ++i) {
                    values[i] = getter(window_diag_it->second[i]);
                }
            }
            return values;
        };
        auto write_window_double = [&](const std::string &name, auto getter) {
            NcVar v = fo.getVar(name);
            if (!v.isNull()) {
                auto values = window_double_values(getter);
                v.putVar(start_index_det, size_det, values.data());
            }
        };
        auto write_window_int = [&](const std::string &name, auto getter) {
            NcVar v = fo.getVar(name);
            if (!v.isNull()) {
                auto values = window_int_values(getter);
                v.putVar(start_index_det, size_det, values.data());
            }
        };

        write_window_int("ptc_invvar_window_n_total",
                         [](const auto &row) { return row.n_total_windows; });
        write_window_int("ptc_invvar_window_n_valid",
                         [](const auto &row) { return row.n_valid_windows; });
        write_window_double("ptc_invvar_window_valid_fraction",
                            [](const auto &row) { return row.valid_window_fraction; });
        write_window_double("ptc_invvar_window_median",
                            [](const auto &row) { return row.inv_var_median; });
        write_window_double("ptc_invvar_window_q10",
                            [](const auto &row) { return row.inv_var_q10; });
        write_window_double("ptc_invvar_window_q90",
                            [](const auto &row) { return row.inv_var_q90; });
        write_window_double("ptc_invvar_window_flagged_frac_median",
                            [](const auto &row) { return row.flagged_frac_median; });
        write_window_double("ptc_invvar_window_flagged_frac_max",
                            [](const auto &row) { return row.flagged_frac_max; });
        write_window_double("ptc_invvar_window_heavy_flagged_fraction",
                            [](const auto &row) { return row.heavily_flagged_window_fraction; });

        const auto second_pass_summary_it = second_pass_summary_by_scan.find(in.index.data);
        const auto corr_summary_it = corr_nw_summary_by_scan.find(in.index.data);
        const auto weight_corr_penalty_it = weight_corr_penalty_summary_by_scan.find(in.index.data);
        const auto busy_row_suppression_it = busy_row_suppression_summary_by_scan.find(in.index.data);
        const auto adaptive_selector_it = adaptive_selector_summary_by_scan.find(in.index.data);
        const int fill_int = -2147483647;
        const double fill_double = std::numeric_limits<double>::quiet_NaN();

        auto build_nw_index = [&]() {
            std::unordered_map<Eigen::Index, std::size_t> nw_to_index;
            nw_to_index.reserve(static_cast<std::size_t>(calib.nws.size()));
            for (Eigen::Index i = 0; i < calib.nws.size(); ++i) {
                nw_to_index[calib.nws(i)] = static_cast<std::size_t>(i);
            }
            return nw_to_index;
        };

        const auto nw_to_index = build_nw_index();

        auto put_corr_nw = [&]() {
            NcVar corr_n_groups_v = fo.getVar("corr_nw_n_groups");
            if (corr_n_groups_v.isNull()) {
                return;
            }
            NcDim n_nws_dim = fo.getDim("n_nws_corr");
            if (n_nws_dim.isNull()) {
                return;
            }
            const auto n_nws = n_nws_dim.getSize();
            std::vector<int> v_n_groups(n_nws, fill_int);
            std::vector<int> v_n_groups_raw(n_nws, fill_int);
            std::vector<int> v_n_det_input(n_nws, fill_int);
            std::vector<int> v_n_det_candidates(n_nws, fill_int);
            std::vector<int> v_n_det_used(n_nws, fill_int);
            std::vector<int> v_n_det_grouped(n_nws, fill_int);
            std::vector<int> v_n_det_ungrouped(n_nws, fill_int);
            std::vector<int> v_sample_step(n_nws, fill_int);
            if (corr_summary_it != corr_nw_summary_by_scan.end()) {
                for (const auto &row : corr_summary_it->second) {
                    const auto it = nw_to_index.find(row.nw);
                    if (it == nw_to_index.end() || it->second >= n_nws) {
                        continue;
                    }
                    const auto j = it->second;
                    v_n_groups[j] = static_cast<int>(row.n_groups_final);
                    v_n_groups_raw[j] = static_cast<int>(row.n_groups_raw);
                    v_n_det_input[j] = static_cast<int>(row.n_det_input);
                    v_n_det_candidates[j] = static_cast<int>(row.n_det_candidates);
                    v_n_det_used[j] = static_cast<int>(row.n_det_used);
                    v_n_det_grouped[j] = static_cast<int>(row.n_det_grouped);
                    v_n_det_ungrouped[j] = static_cast<int>(row.n_det_ungrouped);
                    v_sample_step[j] = static_cast<int>(row.sample_step);
                }
            }
            std::vector<std::size_t> start_scan_nw = {scan_row, 0};
            std::vector<std::size_t> size_scan_nw = {1, n_nws};
            corr_n_groups_v.putVar(start_scan_nw, size_scan_nw, v_n_groups.data());
            fo.getVar("corr_nw_n_groups_raw").putVar(start_scan_nw, size_scan_nw, v_n_groups_raw.data());
            fo.getVar("corr_nw_n_det_input").putVar(start_scan_nw, size_scan_nw, v_n_det_input.data());
            fo.getVar("corr_nw_n_det_candidates").putVar(start_scan_nw, size_scan_nw, v_n_det_candidates.data());
            fo.getVar("corr_nw_n_det_used").putVar(start_scan_nw, size_scan_nw, v_n_det_used.data());
            fo.getVar("corr_nw_n_det_grouped").putVar(start_scan_nw, size_scan_nw, v_n_det_grouped.data());
            fo.getVar("corr_nw_n_det_ungrouped").putVar(start_scan_nw, size_scan_nw, v_n_det_ungrouped.data());
            fo.getVar("corr_nw_sample_step").putVar(start_scan_nw, size_scan_nw, v_sample_step.data());
        };

        auto put_weight_corr = [&]() {
            NcVar wcorr_factor_v = fo.getVar("weight_corr_penalty_factor");
            if (wcorr_factor_v.isNull()) {
                return;
            }
            NcDim n_nws_dim = fo.getDim("n_nws_wcorr");
            if (n_nws_dim.isNull()) {
                return;
            }
            const auto n_nws = n_nws_dim.getSize();
            std::vector<double> v_factor(n_nws, fill_double);
            std::vector<double> v_severity(n_nws, fill_double);
            std::vector<double> v_pair_corr(n_nws, fill_double);
            std::vector<double> v_cm_el_corr(n_nws, fill_double);
            std::vector<double> v_cm_low_mid(n_nws, fill_double);
            std::vector<int> v_n_det_input(n_nws, fill_int);
            std::vector<int> v_n_det_candidates(n_nws, fill_int);
            std::vector<int> v_n_det_used(n_nws, fill_int);
            std::vector<int> v_n_det_weighted(n_nws, fill_int);
            std::vector<int> v_sample_step(n_nws, fill_int);
            if (weight_corr_penalty_it != weight_corr_penalty_summary_by_scan.end()) {
                for (const auto &row : weight_corr_penalty_it->second) {
                    const auto it = nw_to_index.find(row.nw);
                    if (it == nw_to_index.end() || it->second >= n_nws) {
                        continue;
                    }
                    const auto j = it->second;
                    v_factor[j] = row.penalty_factor;
                    v_severity[j] = row.severity;
                    v_pair_corr[j] = row.pair_med_abs_corr;
                    v_cm_el_corr[j] = row.cm_el_abs_corr;
                    v_cm_low_mid[j] = row.cm_low_mid_ratio;
                    v_n_det_input[j] = static_cast<int>(row.n_det_input);
                    v_n_det_candidates[j] = static_cast<int>(row.n_det_candidates);
                    v_n_det_used[j] = static_cast<int>(row.n_det_used);
                    v_n_det_weighted[j] = static_cast<int>(row.n_det_weighted);
                    v_sample_step[j] = static_cast<int>(row.sample_step);
                }
            }
            std::vector<std::size_t> start_scan_nw = {scan_row, 0};
            std::vector<std::size_t> size_scan_nw = {1, n_nws};
            wcorr_factor_v.putVar(start_scan_nw, size_scan_nw, v_factor.data());
            fo.getVar("weight_corr_penalty_severity").putVar(start_scan_nw, size_scan_nw, v_severity.data());
            fo.getVar("weight_corr_penalty_pair_med_abs_corr").putVar(start_scan_nw, size_scan_nw, v_pair_corr.data());
            fo.getVar("weight_corr_penalty_cm_el_abs_corr").putVar(start_scan_nw, size_scan_nw, v_cm_el_corr.data());
            fo.getVar("weight_corr_penalty_cm_low_mid_ratio").putVar(start_scan_nw, size_scan_nw, v_cm_low_mid.data());
            fo.getVar("weight_corr_penalty_n_det_input").putVar(start_scan_nw, size_scan_nw, v_n_det_input.data());
            fo.getVar("weight_corr_penalty_n_det_candidates").putVar(start_scan_nw, size_scan_nw, v_n_det_candidates.data());
            fo.getVar("weight_corr_penalty_n_det_used").putVar(start_scan_nw, size_scan_nw, v_n_det_used.data());
            fo.getVar("weight_corr_penalty_n_det_weighted").putVar(start_scan_nw, size_scan_nw, v_n_det_weighted.data());
            fo.getVar("weight_corr_penalty_sample_step").putVar(start_scan_nw, size_scan_nw, v_sample_step.data());
        };

        auto put_busy_row = [&]() {
            NcVar wbusy_applied_v = fo.getVar("weight_busy_row_suppression_applied");
            if (wbusy_applied_v.isNull()) {
                return;
            }
            NcDim n_nws_dim = fo.getDim("n_nws_busy_row_suppression");
            if (n_nws_dim.isNull()) {
                return;
            }
            const auto n_nws = n_nws_dim.getSize();
            std::vector<int> v_applied(n_nws, fill_int);
            std::vector<int> v_busy(n_nws, fill_int);
            std::vector<int> v_n_candidate_clusters(n_nws, fill_int);
            std::vector<int> v_n_det_weighted(n_nws, fill_int);
            std::vector<double> v_factor(n_nws, fill_double);
            std::vector<double> v_max_resid_z(n_nws, fill_double);
            if (busy_row_suppression_it != busy_row_suppression_summary_by_scan.end()) {
                for (const auto &row : busy_row_suppression_it->second) {
                    const auto it = nw_to_index.find(row.nw);
                    if (it == nw_to_index.end() || it->second >= n_nws) {
                        continue;
                    }
                    const auto j = it->second;
                    v_applied[j] = row.applied ? 1 : 0;
                    v_busy[j] = row.busy_network_vetoed ? 1 : 0;
                    v_n_candidate_clusters[j] = static_cast<int>(row.n_candidate_clusters);
                    v_n_det_weighted[j] = static_cast<int>(row.n_det_weighted);
                    v_factor[j] = row.factor;
                    v_max_resid_z[j] = row.max_unflagged_residual_z;
                }
            }
            std::vector<std::size_t> start_scan_nw = {scan_row, 0};
            std::vector<std::size_t> size_scan_nw = {1, n_nws};
            wbusy_applied_v.putVar(start_scan_nw, size_scan_nw, v_applied.data());
            fo.getVar("weight_busy_row_suppression_busy_network_vetoed").putVar(start_scan_nw, size_scan_nw, v_busy.data());
            fo.getVar("weight_busy_row_suppression_n_candidate_clusters").putVar(start_scan_nw, size_scan_nw, v_n_candidate_clusters.data());
            fo.getVar("weight_busy_row_suppression_n_det_weighted").putVar(start_scan_nw, size_scan_nw, v_n_det_weighted.data());
            fo.getVar("weight_busy_row_suppression_factor").putVar(start_scan_nw, size_scan_nw, v_factor.data());
            fo.getVar("weight_busy_row_suppression_max_unflagged_residual_z").putVar(start_scan_nw, size_scan_nw, v_max_resid_z.data());
        };

        auto put_adaptive = [&]() {
            NcVar adaptive_chosen_k_v = fo.getVar("adaptive_pca_chosen_k");
            if (adaptive_chosen_k_v.isNull()) {
                return;
            }
            NcDim n_nws_dim = fo.getDim("n_nws_adaptive_pca");
            if (n_nws_dim.isNull()) {
                return;
            }
            const auto n_nws = n_nws_dim.getSize();
            std::vector<int> v_selector_used(n_nws, fill_int);
            std::vector<int> v_selector_fallback(n_nws, fill_int);
            std::vector<int> v_baseline_k(n_nws, fill_int);
            std::vector<int> v_chosen_k(n_nws, fill_int);
            std::vector<int> v_runnerup_k(n_nws, fill_int);
            std::vector<int> v_n_candidates(n_nws, fill_int);
            std::vector<int> v_n_det_input(n_nws, fill_int);
            std::vector<int> v_n_det_used(n_nws, fill_int);
            std::vector<int> v_n_time_used(n_nws, fill_int);
            std::vector<int> v_sample_step(n_nws, fill_int);
            std::vector<double> v_chosen_score(n_nws, fill_double);
            std::vector<double> v_runnerup_score(n_nws, fill_double);
            std::vector<double> v_score_margin(n_nws, fill_double);
            std::vector<double> v_chosen_med_abs_corr(n_nws, fill_double);
            std::vector<double> v_chosen_cm_low_mid_ratio(n_nws, fill_double);
            std::vector<double> v_chosen_tail4_binom_z(n_nws, fill_double);
            std::vector<double> v_chosen_top_mode_frac(n_nws, fill_double);
            std::vector<double> v_eig_solve_msec(n_nws, fill_double);
            std::vector<double> v_candidate_eval_msec(n_nws, fill_double);
            std::vector<double> v_total_msec(n_nws, fill_double);
            if (adaptive_selector_it != adaptive_selector_summary_by_scan.end()) {
                for (const auto &row : adaptive_selector_it->second) {
                    const auto it = nw_to_index.find(row.nw);
                    if (it == nw_to_index.end() || it->second >= n_nws) {
                        continue;
                    }
                    const auto j = it->second;
                    v_selector_used[j] = row.selector_used;
                    v_selector_fallback[j] = row.selector_fallback;
                    v_baseline_k[j] = static_cast<int>(row.baseline_k);
                    v_chosen_k[j] = static_cast<int>(row.chosen_k);
                    v_runnerup_k[j] = static_cast<int>(row.runnerup_k);
                    v_n_candidates[j] = static_cast<int>(row.n_candidates);
                    v_n_det_input[j] = static_cast<int>(row.n_det_input);
                    v_n_det_used[j] = static_cast<int>(row.n_det_used);
                    v_n_time_used[j] = static_cast<int>(row.n_time_used);
                    v_sample_step[j] = static_cast<int>(row.sample_step);
                    v_chosen_score[j] = row.chosen_score;
                    v_runnerup_score[j] = row.runnerup_score;
                    v_score_margin[j] = row.score_margin;
                    v_chosen_med_abs_corr[j] = row.chosen_med_abs_corr;
                    v_chosen_cm_low_mid_ratio[j] = row.chosen_cm_low_mid_ratio;
                    v_chosen_tail4_binom_z[j] = row.chosen_tail4_binom_z;
                    v_chosen_top_mode_frac[j] = row.chosen_top_mode_frac;
                    v_eig_solve_msec[j] = row.eig_solve_msec;
                    v_candidate_eval_msec[j] = row.candidate_eval_msec;
                    v_total_msec[j] = row.total_msec;
                }
            }
            std::vector<std::size_t> start_scan_nw = {scan_row, 0};
            std::vector<std::size_t> size_scan_nw = {1, n_nws};
            fo.getVar("adaptive_pca_selector_used").putVar(start_scan_nw, size_scan_nw, v_selector_used.data());
            fo.getVar("adaptive_pca_selector_fallback").putVar(start_scan_nw, size_scan_nw, v_selector_fallback.data());
            fo.getVar("adaptive_pca_baseline_k").putVar(start_scan_nw, size_scan_nw, v_baseline_k.data());
            adaptive_chosen_k_v.putVar(start_scan_nw, size_scan_nw, v_chosen_k.data());
            fo.getVar("adaptive_pca_runnerup_k").putVar(start_scan_nw, size_scan_nw, v_runnerup_k.data());
            fo.getVar("adaptive_pca_n_candidates").putVar(start_scan_nw, size_scan_nw, v_n_candidates.data());
            fo.getVar("adaptive_pca_n_det_input").putVar(start_scan_nw, size_scan_nw, v_n_det_input.data());
            fo.getVar("adaptive_pca_n_det_used").putVar(start_scan_nw, size_scan_nw, v_n_det_used.data());
            fo.getVar("adaptive_pca_n_time_used").putVar(start_scan_nw, size_scan_nw, v_n_time_used.data());
            fo.getVar("adaptive_pca_sample_step").putVar(start_scan_nw, size_scan_nw, v_sample_step.data());
            fo.getVar("adaptive_pca_chosen_score").putVar(start_scan_nw, size_scan_nw, v_chosen_score.data());
            fo.getVar("adaptive_pca_runnerup_score").putVar(start_scan_nw, size_scan_nw, v_runnerup_score.data());
            fo.getVar("adaptive_pca_score_margin").putVar(start_scan_nw, size_scan_nw, v_score_margin.data());
            fo.getVar("adaptive_pca_chosen_med_abs_corr").putVar(start_scan_nw, size_scan_nw, v_chosen_med_abs_corr.data());
            fo.getVar("adaptive_pca_chosen_cm_low_mid_ratio").putVar(start_scan_nw, size_scan_nw, v_chosen_cm_low_mid_ratio.data());
            fo.getVar("adaptive_pca_chosen_tail4_binom_z").putVar(start_scan_nw, size_scan_nw, v_chosen_tail4_binom_z.data());
            fo.getVar("adaptive_pca_chosen_top_mode_frac").putVar(start_scan_nw, size_scan_nw, v_chosen_top_mode_frac.data());
            fo.getVar("adaptive_pca_eig_solve_msec").putVar(start_scan_nw, size_scan_nw, v_eig_solve_msec.data());
            fo.getVar("adaptive_pca_candidate_eval_msec").putVar(start_scan_nw, size_scan_nw, v_candidate_eval_msec.data());
            fo.getVar("adaptive_pca_total_msec").putVar(start_scan_nw, size_scan_nw, v_total_msec.data());
        };

        auto put_second_pass = [&]() {
            NcVar second_pass_busy_v = fo.getVar("ptc_second_pass_busy_network_vetoed");
            if (second_pass_busy_v.isNull()) {
                return;
            }
            NcDim n_nws_dim = fo.getDim("n_nws_ptc_second_pass");
            if (n_nws_dim.isNull()) {
                return;
            }
            const auto n_nws = n_nws_dim.getSize();
            std::vector<int> v_busy(n_nws, fill_int);
            std::vector<int> v_n_candidate_clusters(n_nws, fill_int);
            std::vector<int> v_n_candidate_events(n_nws, fill_int);
            std::vector<int> v_n_accepted_clusters(n_nws, fill_int);
            std::vector<int> v_n_accepted_events(n_nws, fill_int);
            std::vector<int> v_n_det_with_added_flags(n_nws, fill_int);
            std::vector<int> v_max_resid_uid(n_nws, fill_int);
            std::vector<int> v_top_cluster_sample(n_nws, fill_int);
            std::vector<int> v_top_cluster_n_detectors(n_nws, fill_int);
            std::vector<int> v_top_cluster_n_events(n_nws, fill_int);
            std::vector<int> v_top_event_kind(n_nws, fill_int);
            std::vector<int> v_top_event_uid(n_nws, fill_int);
            std::vector<int> v_top_event_sample(n_nws, fill_int);
            std::vector<double> v_existing_frac(n_nws, fill_double);
            std::vector<double> v_proposed_frac(n_nws, fill_double);
            std::vector<double> v_new_frac(n_nws, fill_double);
            std::vector<double> v_max_resid_z(n_nws, fill_double);
            std::vector<double> v_top_cluster_peak(n_nws, fill_double);
            std::vector<double> v_top_event_score(n_nws, fill_double);
            if (second_pass_summary_it != second_pass_summary_by_scan.end()) {
                for (const auto &row : second_pass_summary_it->second) {
                    const auto it = nw_to_index.find(row.nw);
                    if (it == nw_to_index.end() || it->second >= n_nws) {
                        continue;
                    }
                    const auto j = it->second;
                    v_busy[j] = row.busy_network_vetoed ? 1 : 0;
                    v_n_candidate_clusters[j] = static_cast<int>(row.n_candidate_clusters);
                    v_n_candidate_events[j] = static_cast<int>(row.n_candidate_events);
                    v_n_accepted_clusters[j] = static_cast<int>(row.n_accepted_clusters);
                    v_n_accepted_events[j] = static_cast<int>(row.n_accepted_events);
                    v_n_det_with_added_flags[j] = static_cast<int>(row.n_det_with_added_flags);
                    v_max_resid_uid[j] = row.max_unflagged_residual_uid;
                    v_top_cluster_sample[j] = row.top_candidate_cluster_sample;
                    v_top_cluster_n_detectors[j] = static_cast<int>(row.top_candidate_cluster_n_detectors);
                    v_top_cluster_n_events[j] = static_cast<int>(row.top_candidate_cluster_n_events);
                    v_top_event_kind[j] = row.top_event.kind_code();
                    v_top_event_uid[j] = row.top_event_uid;
                    v_top_event_sample[j] = row.top_event.sample;
                    v_existing_frac[j] = row.existing_flagged_fraction;
                    v_proposed_frac[j] = row.proposed_flagged_fraction;
                    v_new_frac[j] = row.newly_flagged_fraction;
                    v_max_resid_z[j] = row.max_unflagged_residual_z;
                    v_top_cluster_peak[j] = row.top_candidate_cluster_peak_score;
                    v_top_event_score[j] = row.top_event.score;
                }
            }
            std::vector<std::size_t> start_scan_nw = {scan_row, 0};
            std::vector<std::size_t> size_scan_nw = {1, n_nws};
            second_pass_busy_v.putVar(start_scan_nw, size_scan_nw, v_busy.data());
            fo.getVar("ptc_second_pass_n_candidate_clusters").putVar(start_scan_nw, size_scan_nw, v_n_candidate_clusters.data());
            fo.getVar("ptc_second_pass_n_candidate_events").putVar(start_scan_nw, size_scan_nw, v_n_candidate_events.data());
            fo.getVar("ptc_second_pass_n_accepted_clusters").putVar(start_scan_nw, size_scan_nw, v_n_accepted_clusters.data());
            fo.getVar("ptc_second_pass_n_accepted_events").putVar(start_scan_nw, size_scan_nw, v_n_accepted_events.data());
            fo.getVar("ptc_second_pass_n_det_with_added_flags").putVar(start_scan_nw, size_scan_nw, v_n_det_with_added_flags.data());
            fo.getVar("ptc_second_pass_max_unflagged_residual_uid").putVar(start_scan_nw, size_scan_nw, v_max_resid_uid.data());
            fo.getVar("ptc_second_pass_top_candidate_cluster_sample").putVar(start_scan_nw, size_scan_nw, v_top_cluster_sample.data());
            fo.getVar("ptc_second_pass_top_candidate_cluster_n_detectors").putVar(start_scan_nw, size_scan_nw, v_top_cluster_n_detectors.data());
            fo.getVar("ptc_second_pass_top_candidate_cluster_n_events").putVar(start_scan_nw, size_scan_nw, v_top_cluster_n_events.data());
            fo.getVar("ptc_second_pass_top_event_kind").putVar(start_scan_nw, size_scan_nw, v_top_event_kind.data());
            fo.getVar("ptc_second_pass_top_event_uid").putVar(start_scan_nw, size_scan_nw, v_top_event_uid.data());
            fo.getVar("ptc_second_pass_top_event_sample").putVar(start_scan_nw, size_scan_nw, v_top_event_sample.data());
            fo.getVar("ptc_second_pass_existing_flagged_fraction").putVar(start_scan_nw, size_scan_nw, v_existing_frac.data());
            fo.getVar("ptc_second_pass_proposed_flagged_fraction").putVar(start_scan_nw, size_scan_nw, v_proposed_frac.data());
            fo.getVar("ptc_second_pass_newly_flagged_fraction").putVar(start_scan_nw, size_scan_nw, v_new_frac.data());
            fo.getVar("ptc_second_pass_max_unflagged_residual_z").putVar(start_scan_nw, size_scan_nw, v_max_resid_z.data());
            fo.getVar("ptc_second_pass_top_candidate_cluster_peak_score").putVar(start_scan_nw, size_scan_nw, v_top_cluster_peak.data());
            fo.getVar("ptc_second_pass_top_event_score").putVar(start_scan_nw, size_scan_nw, v_top_event_score.data());
        };

        put_corr_nw();
        put_weight_corr();
        put_busy_row();
        put_adaptive();
        put_second_pass();

        fo.sync();
        fo.close();
        logger->info("ptc diagnostics sidecar chunk written to {}", filepath);
    } catch (NcException &e) {
        logger->error("{}", e.what());
    }
}

inline void PTCProc::clear_cached_diagnostics(Eigen::Index scan_id) {
    remove_bad_dets_window_summary_by_scan.erase(scan_id);
    corr_nw_group_ids_by_scan.erase(scan_id);
    corr_nw_summary_by_scan.erase(scan_id);
    weight_corr_penalty_summary_by_scan.erase(scan_id);
    busy_row_suppression_summary_by_scan.erase(scan_id);
    adaptive_selector_summary_by_scan.erase(scan_id);
    second_pass_summary_by_scan.erase(scan_id);
    second_pass_added_flags_by_scan.erase(scan_id);
}

} // namespace timestream
