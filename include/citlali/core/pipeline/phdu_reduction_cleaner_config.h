#pragma once

// Included by phdu_reduction_config.h inside namespace citlali::pipeline.

template <class FitsEntry, class WeightCorrPenalty, class Logger>
void add_phdu_weight_corr_penalty_config(
    FitsEntry &fits_entry, const std::string &array_name,
    const Logger &logger, const WeightCorrPenalty &penalty) {
    auto &hdu = fits_entry.pfits->pHDU();
    auto add_double_key = [&](const std::string &key, double value,
                              const std::string &comment,
                              double fallback = 0.0) {
        add_phdu_double_key(fits_entry, array_name, logger, key, value,
                            comment, fallback);
    };

    hdu.addKey("CONFIG.WEIGHT.CORR_PENALTY.ENABLED", penalty.enabled,
               "Enable per-network corr-based weight penalties");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.MIN_GOOD_FRAC",
                   penalty.min_good_frac,
                   "Minimum unflagged sample fraction per detector");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.MIN_OVERLAP",
                   penalty.min_overlap,
                   "Minimum overlap for pairwise corr metric");
    hdu.addKey("CONFIG.WEIGHT.CORR_PENALTY.MAX_SAMPLES",
               penalty.max_samples,
               "Max sampled timestream points for penalty metrics");
    hdu.addKey("CONFIG.WEIGHT.CORR_PENALTY.MAX_PAIRS", penalty.max_pairs,
               "Max sampled detector pairs for corr metric");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.FLOOR", penalty.floor,
                   "Minimum per-network multiplicative weight factor");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.EXPONENT", penalty.exponent,
                   "Exponent shaping corr penalty response");
    hdu.addKey("CONFIG.WEIGHT.CORR_PENALTY.PAIR.ENABLED",
               penalty.pair_corr.enabled,
               "Enable pairwise corr penalty term");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.PAIR.REF",
                   penalty.pair_corr.ref,
                   "Pairwise corr reference value");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.PAIR.SPAN",
                   penalty.pair_corr.span,
                   "Pairwise corr scale span");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.PAIR.WEIGHT",
                   penalty.pair_corr.weight,
                   "Pairwise corr term weight");
    hdu.addKey("CONFIG.WEIGHT.CORR_PENALTY.CM_EL.ENABLED",
               penalty.cm_el_corr.enabled,
               "Enable common-mode elevation corr penalty term");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.CM_EL.REF",
                   penalty.cm_el_corr.ref,
                   "Common-mode elevation corr reference");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.CM_EL.SPAN",
                   penalty.cm_el_corr.span,
                   "Common-mode elevation corr scale span");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.CM_EL.WEIGHT",
                   penalty.cm_el_corr.weight,
                   "Common-mode elevation corr term weight");
    hdu.addKey("CONFIG.WEIGHT.CORR_PENALTY.LOWMID.ENABLED",
               penalty.cm_low_mid_ratio.enabled,
               "Enable common-mode low/mid ratio penalty term");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.LOWMID.REF",
                   penalty.cm_low_mid_ratio.ref,
                   "Common-mode low/mid ratio reference");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.LOWMID.SPAN",
                   penalty.cm_low_mid_ratio.span,
                   "Common-mode low/mid ratio scale span");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.LOWMID.WEIGHT",
                   penalty.cm_low_mid_ratio.weight,
                   "Common-mode low/mid ratio term weight");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.LOWMID.LOWMIN_HZ",
                   penalty.cm_low_mid_ratio.low_band_Hz[0],
                   "Low-band minimum frequency for low/mid ratio");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.LOWMID.LOWMAX_HZ",
                   penalty.cm_low_mid_ratio.low_band_Hz[1],
                   "Low-band maximum frequency for low/mid ratio");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.LOWMID.MIDMIN_HZ",
                   penalty.cm_low_mid_ratio.mid_band_Hz[0],
                   "Mid-band minimum frequency for low/mid ratio");
    add_double_key("CONFIG.WEIGHT.CORR_PENALTY.LOWMID.MIDMAX_HZ",
                   penalty.cm_low_mid_ratio.mid_band_Hz[1],
                   "Mid-band maximum frequency for low/mid ratio");
}

template <class FitsEntry, class BusyRowSuppression, class Logger>
void add_phdu_busy_row_suppression_config(
    FitsEntry &fits_entry, const std::string &array_name,
    const Logger &logger, const BusyRowSuppression &suppression) {
    auto &hdu = fits_entry.pfits->pHDU();
    auto add_double_key = [&](const std::string &key, double value,
                              const std::string &comment,
                              double fallback = 0.0) {
        add_phdu_double_key(fits_entry, array_name, logger, key, value,
                            comment, fallback);
    };

    hdu.addKey("CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.ENABLED",
               suppression.enabled,
               "Enable busy scan/network row weight suppression");
    hdu.addKey("CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.REQUIRE_BUSY_VETO",
               suppression.require_busy_veto,
               "Require second-pass busy-network veto before suppression");
    hdu.addKey("CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.MIN_CAND_CLUSTERS",
               suppression.min_candidate_clusters,
               "Minimum candidate residual clusters for suppression");
    add_double_key("CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.MIN_MAX_RESID_Z",
                   suppression.min_max_unflagged_residual_z,
                   "Minimum max unflagged residual z for suppression");
    add_double_key("CONFIG.WEIGHT.BUSY_ROW_SUPPRESS.FACTOR",
                   suppression.factor,
                   "Busy-row multiplicative weight suppression factor");
}

template <class FitsEntry, class NEigRemoved, class Logger>
void add_phdu_cleaner_config(FitsEntry &fits_entry,
                             const std::string &array_name,
                             const Logger &logger,
                             const citlali::config::ProcessedTimeChunkCleanConfig
                                 &clean,
                             NEigRemoved n_eig_removed) {
    auto &hdu = fits_entry.pfits->pHDU();
    auto add_double_key = [&](const std::string &key, double value,
                              const std::string &comment,
                              double fallback = 0.0) {
        add_phdu_double_key(fits_entry, array_name, logger, key, value,
                            comment, fallback);
    };
    const auto adaptive_offsets_joined =
        join_numeric_values(clean.adaptive_selector.candidate_offsets);
    const auto adaptive_grouping_joined =
        join_string_values(clean.adaptive_selector.grouping);

    hdu.addKey("CONFIG.CLEANED", clean.enabled, "Cleaned");
    hdu.addKey("CONFIG.CLEANED.MODESEL",
               std::string{citlali::config::to_string(clean.active)},
               "PTC cleaner method");
    hdu.addKey("CONFIG.CLEANED.MP.ENABLED",
               clean.marchenko_pastur.enabled,
               "Marchenko-Pastur mode selection enabled");
    add_double_key("CONFIG.CLEANED.MP.BANDLOW_HZ",
                   clean.marchenko_pastur.band_low_Hz,
                   "MP covariance low-band edge (Hz)");
    add_double_key("CONFIG.CLEANED.MP.BANDHIGH_HZ",
                   clean.marchenko_pastur.band_high_Hz,
                   "MP covariance high-band edge (Hz)");
    hdu.addKey("CONFIG.CLEANED.MP.MAXMODES",
               clean.marchenko_pastur.max_modes,
               "MP max modes considered");
    hdu.addKey("CONFIG.CLEANED.ADAPT.ENABLED",
               clean.adaptive_selector.enabled,
               "Bounded adaptive PCA selector enabled");
    add_double_key("CONFIG.CLEANED.ADAPT.MIN_GOOD_FRAC",
                   clean.adaptive_selector.min_good_frac,
                   "Adaptive PCA minimum unflagged detector fraction");
    hdu.addKey("CONFIG.CLEANED.ADAPT.MAX_DET",
               clean.adaptive_selector.max_det,
               "Adaptive PCA max detectors used for scoring");
    hdu.addKey("CONFIG.CLEANED.ADAPT.MAX_SAMPLES",
               clean.adaptive_selector.max_samples,
               "Adaptive PCA max time samples used for scoring");
    hdu.addKey("CONFIG.CLEANED.ADAPT.MAX_PAIRS",
               clean.adaptive_selector.max_pairs,
               "Adaptive PCA max detector pairs used for scoring");
    add_double_key("CONFIG.CLEANED.ADAPT.CLIP_Z",
                   clean.adaptive_selector.clip_z,
                   "Adaptive PCA residual clip threshold");
    add_double_key("CONFIG.CLEANED.ADAPT.LOW_WEIGHT",
                   clean.adaptive_selector.low_weight,
                   "Adaptive PCA low-band selector weight");
    add_double_key("CONFIG.CLEANED.ADAPT.TAIL_WEIGHT",
                   clean.adaptive_selector.tail_weight,
                   "Adaptive PCA tail selector weight");
    add_double_key("CONFIG.CLEANED.ADAPT.TOPMODE_WEIGHT",
                   clean.adaptive_selector.topmode_weight,
                   "Adaptive PCA top-mode selector weight");
    add_double_key("CONFIG.CLEANED.ADAPT.REG_WEIGHT",
                   clean.adaptive_selector.reg_weight,
                   "Adaptive PCA regularization-to-baseline weight");
    add_double_key("CONFIG.CLEANED.ADAPT.LOWMIN_HZ",
                   clean.adaptive_selector.low_band_Hz[0],
                   "Adaptive PCA low-band minimum frequency");
    add_double_key("CONFIG.CLEANED.ADAPT.LOWMAX_HZ",
                   clean.adaptive_selector.low_band_Hz[1],
                   "Adaptive PCA low-band maximum frequency");
    add_double_key("CONFIG.CLEANED.ADAPT.MIDMIN_HZ",
                   clean.adaptive_selector.mid_band_Hz[0],
                   "Adaptive PCA mid-band minimum frequency");
    add_double_key("CONFIG.CLEANED.ADAPT.MIDMAX_HZ",
                   clean.adaptive_selector.mid_band_Hz[1],
                   "Adaptive PCA mid-band maximum frequency");
    hdu.addKey("CONFIG.CLEANED.ADAPT.OFFSETS", adaptive_offsets_joined,
               "Adaptive PCA candidate cut offsets");
    hdu.addKey("CONFIG.CLEANED.ADAPT.GROUPING", adaptive_grouping_joined,
               "Grouping subset where adaptive PCA is active");
    hdu.addKey("CONFIG.CLEANED.ADAPT.LOGCAND",
               clean.adaptive_selector.log_candidates,
               "Adaptive PCA per-candidate logging enabled");
    hdu.addKey("CONFIG.CLEANED.NEIG", n_eig_removed,
               "Number of eigenvalues removed");
}
