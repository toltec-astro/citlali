#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
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

    WeightCorrPenaltyOptions weight_corr_penalty;
    std::map<Eigen::Index, Eigen::VectorXi> corr_nw_group_ids_by_scan;
    std::map<Eigen::Index, std::vector<CorrNWDiagSummary>> corr_nw_summary_by_scan;
    std::map<Eigen::Index, std::vector<WeightCorrPenaltyDiagSummary>> weight_corr_penalty_summary_by_scan;

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
        if (cleaner.null_model.enabled && cleaner.marchenko_pastur.enabled) {
            logger->error("clean.null_model.enabled and clean.marchenko_pastur.enabled are mutually exclusive");
            std::exit(EXIT_FAILURE);
        }
        if (!have_standard_pca_block) {
            cleaner.standard_pca.enabled = !(cleaner.null_model.enabled || cleaner.marchenko_pastur.enabled);
        }
        const int n_enabled_cleaners =
            static_cast<int>(cleaner.standard_pca.enabled) +
            static_cast<int>(cleaner.null_model.enabled) +
            static_cast<int>(cleaner.marchenko_pastur.enabled);
        if (n_enabled_cleaners != 1) {
            logger->error(
                "exactly one cleaner must be enabled when clean.enabled=true; got standard_pca={} null_model={} marchenko_pastur={}",
                cleaner.standard_pca.enabled, cleaner.null_model.enabled, cleaner.marchenko_pastur.enabled);
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
        // mask radius in arcseconds
        get_config_value(config, mask_radius_arcsec, missing_keys, invalid_keys,
                         std::tuple{"timestream","processed_time_chunk","clean","mask_radius_arcsec"});

        // upper weight factor
        get_config_value(config, cleaner.tau, missing_keys, invalid_keys,
                         std::tuple{"timestream","processed_time_chunk","clean","tau"});
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
        const bool want_eigs = (run_tod_output || write_evals);
        const bool store_eigs = want_eigs && (cleaner_local.n_calc > 0);
        bool warned_eigs = false;

        // loop through config groupings
        const bool null_model_enabled_global = cleaner_local.null_model.enabled;
        const bool marchenko_pastur_enabled_global = cleaner_local.marchenko_pastur.enabled;
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
                            Eigen::Index forced_limit_index = -1;
                            if (null_model_for_group) {
                                forced_limit_index = cleaner_local.get_null_model_index(in_scans_sub, flags_sub, apt_flags_sub);
                            }
                            else if (marchenko_pastur_for_group) {
                                forced_limit_index = cleaner_local.get_marchenko_pastur_index(in_scans_sub, flags_sub, apt_flags_sub);
                            }

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
                    // calculate eigenvalues and eigenvalues
                    auto [evals, evecs] = cleaner_local.calc_eig_values<timestream::Cleaner::SpectraBackend>(
                        in_scans_block, masked_flags, apt_flags, cleaner_local.n_eig_to_cut[arr_index](indx));
                    Eigen::Index forced_limit_index = -1;
                    if (null_model_for_group) {
                        forced_limit_index = cleaner_local.get_null_model_index(in_scans_block, masked_flags, apt_flags);
                    }
                    else if (marchenko_pastur_for_group) {
                        forced_limit_index = cleaner_local.get_marchenko_pastur_index(in_scans_block, masked_flags, apt_flags);
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

                    // remove eigenvalues from the data and reconstruct the tod
                    cleaner_local.remove_eig_values<timestream::Cleaner::SpectraBackend>(
                        in_scans_block, masked_flags, evals, evecs, out_scans_block,
                        cleaner_local.n_eig_to_cut[arr_index](indx), forced_limit_index,
                        effective_group, nw_index, arr_index);

                    if (in.kernel.data.size()!=0) {
                        // check if any good flags
                            logger->debug("cleaning kernel");
                            auto in_kernel_block = in.kernel.data.block(0, start_index, n_pts, n_dets);
                            auto out_kernel_block = in.kernel.data.block(0, start_index, n_pts, n_dets);

                            // remove eigenvalues from the kernel and reconstruct the tod
                            cleaner_local.remove_eig_values<timestream::Cleaner::SpectraBackend>(
                                in_kernel_block, masked_flags, evals, evecs, out_kernel_block,
                                cleaner_local.n_eig_to_cut[arr_index](indx), forced_limit_index,
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
            !tod_mb.signal.empty() &&
            fruit_loops_source_valid.size() == static_cast<Eigen::Index>(tod_mb.signal.size());
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

    if (weight_corr_penalty.enabled) {
        auto finite_or_nan = [](double v) {
            if (std::isfinite(v)) {
                return v;
            }
            return std::numeric_limits<double>::quiet_NaN();
        };
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

        std::map<Eigen::Index, std::tuple<Eigen::Index, Eigen::Index>> nw_limits;
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

        const auto corr_groups_it = corr_nw_group_ids_by_scan.find(in.index.data);
        const auto corr_summary_it = corr_nw_summary_by_scan.find(in.index.data);
        const auto weight_corr_penalty_it = weight_corr_penalty_summary_by_scan.find(in.index.data);
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

} // namespace timestream
