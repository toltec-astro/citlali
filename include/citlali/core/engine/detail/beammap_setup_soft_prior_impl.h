#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

void Beammap::reset_beammap_soft_prior_setup_state() {
    beammap_soft_prior_slots.clear();
    beammap_soft_priors_loaded = false;
    beammap_soft_priors_are_centered = false;
    beammap_soft_priors_are_derotated = false;
    beammap_prior_array_center_x_arcsec.clear();
    beammap_prior_array_center_y_arcsec.clear();
    beammap_prior_array_alignment.clear();
}

void Beammap::load_or_disable_beammap_soft_priors() {
    auto &beammap_config = citlali::pipeline::beammap_config(*this);
    const auto &mapmaking_config = citlali::pipeline::mapmaking_config(*this);

    auto &priors_config = beammap_config.priors;
    if (priors_config.enabled) {
        if (mapmaking_config.grouping !=
            citlali::config::MapGrouping::detector) {
            logger->warn("beammap priors requested but map_grouping={} (requires detector); disabling priors",
                         citlali::pipeline::active_map_grouping_name(*this));
            priors_config.enabled = false;
        }
        else if (!load_soft_priors()) {
            logger->warn("beammap priors failed to load; disabling prior-guided initialization");
            priors_config.enabled = false;
        }
    }
}

void Beammap::populate_beammap_soft_prior_metadata() {
    const auto &priors_config =
        citlali::pipeline::beammap_config(*this).priors;

    calib.apt_meta["beammap_priors_enabled"] = priors_config.enabled;
    calib.apt_meta["beammap_priors_filepath"] = priors_config.filepath;
    calib.apt_meta["beammap_priors_candidate_top_n"] = priors_config.candidate_top_n;
    calib.apt_meta["beammap_priors_min_snr"] = priors_config.min_snr;
    calib.apt_meta["beammap_priors_max_d2"] = priors_config.max_d2;
    calib.apt_meta["beammap_priors_max_d2_iter0"] = priors_config.max_d2_iter0;
    calib.apt_meta["beammap_priors_max_d2_after_iter0"] = priors_config.max_d2_after_iter0;
    calib.apt_meta["beammap_priors_score_lambda"] = priors_config.score_lambda;
    calib.apt_meta["beammap_priors_score_lambda_iter0"] = priors_config.score_lambda_iter0;
    calib.apt_meta["beammap_priors_score_lambda_after_iter0"] = priors_config.score_lambda_after_iter0;
    calib.apt_meta["beammap_priors_fallback_blind"] = priors_config.fallback_blind;
    calib.apt_meta["beammap_priors_align_after_iter0"] = priors_config.align_after_iter0;
    calib.apt_meta["beammap_priors_alignment_scope"] =
        std::string(citlali::config::to_string(priors_config.alignment_scope));
    calib.apt_meta["beammap_priors_alignment_common_support"] =
        std::string(citlali::config::to_string(
            priors_config.alignment_common_support));
    calib.apt_meta["beammap_priors_alignment_common_support_quantile"] =
        priors_config.alignment_common_support_quantile;
    calib.apt_meta["beammap_priors_alignment_min_matches"] = priors_config.alignment_min_matches;
    calib.apt_meta["beammap_priors_alignment_max_d2"] = priors_config.alignment_max_d2;
    calib.apt_meta["beammap_priors_alignment_fit_rotation"] = priors_config.alignment_fit_rotation;
    calib.apt_meta["beammap_priors_alignment_max_rotation_deg"] = priors_config.alignment_max_rotation_deg;
}

void Beammap::configure_beammap_soft_prior_setup() {
    reset_beammap_soft_prior_setup_state();
    load_or_disable_beammap_soft_priors();
    populate_beammap_soft_prior_metadata();
}
