#pragma once

// Engine timestream config implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/timestream_config_mirror.h>

template<typename CT>
void Engine::get_ptc_config(CT &config) {
    logger->info("getting ptc config options");
    // get ptcproc config
    ptcproc.get_config(config, missing_keys, invalid_keys);
    citlali::pipeline::mirror_fruit_loops_config(
        typed_timestream_config.fruit_loops, ptcproc);
    citlali::pipeline::mirror_processed_clean_config(
        typed_timestream_config.processed_time_chunk.clean, ptcproc,
        toltec_io.array_name_map);
    auto &typed_weighting =
        typed_timestream_config.processed_time_chunk.weighting;
    auto &typed_flagging =
        typed_timestream_config.processed_time_chunk.flagging;
    citlali::pipeline::mirror_processed_weighting_config(
        typed_weighting, typed_flagging, ptcproc);
    const auto &weight_validation = ptcproc.weight_validation;
    citlali::pipeline::mirror_processed_weight_validation_config(
        typed_weighting.validation, weight_validation);

    const auto &weight_corr_penalty = ptcproc.weight_corr_penalty;
    auto &typed_corr_penalty = typed_weighting.corr_penalty;
    typed_corr_penalty.enabled = weight_corr_penalty.enabled;
    typed_corr_penalty.min_good_frac = weight_corr_penalty.min_good_frac;
    typed_corr_penalty.min_overlap = weight_corr_penalty.min_overlap;
    typed_corr_penalty.max_samples = weight_corr_penalty.max_samples;
    typed_corr_penalty.max_pairs = weight_corr_penalty.max_pairs;
    typed_corr_penalty.seed = static_cast<int>(weight_corr_penalty.seed);
    typed_corr_penalty.floor = weight_corr_penalty.floor;
    typed_corr_penalty.exponent = weight_corr_penalty.exponent;
    typed_corr_penalty.pair_corr.enabled =
        weight_corr_penalty.pair_corr.enabled;
    typed_corr_penalty.pair_corr.ref = weight_corr_penalty.pair_corr.ref;
    typed_corr_penalty.pair_corr.span = weight_corr_penalty.pair_corr.span;
    typed_corr_penalty.pair_corr.weight = weight_corr_penalty.pair_corr.weight;
    typed_corr_penalty.cm_el_corr.enabled =
        weight_corr_penalty.cm_el_corr.enabled;
    typed_corr_penalty.cm_el_corr.ref = weight_corr_penalty.cm_el_corr.ref;
    typed_corr_penalty.cm_el_corr.span = weight_corr_penalty.cm_el_corr.span;
    typed_corr_penalty.cm_el_corr.weight =
        weight_corr_penalty.cm_el_corr.weight;
    typed_corr_penalty.cm_low_mid_ratio.enabled =
        weight_corr_penalty.cm_low_mid_ratio.enabled;
    typed_corr_penalty.cm_low_mid_ratio.ref =
        weight_corr_penalty.cm_low_mid_ratio.ref;
    typed_corr_penalty.cm_low_mid_ratio.span =
        weight_corr_penalty.cm_low_mid_ratio.span;
    typed_corr_penalty.cm_low_mid_ratio.weight =
        weight_corr_penalty.cm_low_mid_ratio.weight;
    typed_corr_penalty.cm_low_mid_ratio.low_band_Hz = {
        weight_corr_penalty.cm_low_mid_ratio.low_min_Hz,
        weight_corr_penalty.cm_low_mid_ratio.low_max_Hz};
    typed_corr_penalty.cm_low_mid_ratio.mid_band_Hz = {
        weight_corr_penalty.cm_low_mid_ratio.mid_min_Hz,
        weight_corr_penalty.cm_low_mid_ratio.mid_max_Hz};

    auto &typed_second_pass =
        typed_timestream_config.processed_time_chunk.flagging.second_pass_local;
    typed_second_pass.enabled = ptcproc.second_pass_local.enabled;
    typed_second_pass.min_spike_sigma =
        ptcproc.second_pass_local.min_spike_sigma;
    typed_second_pass.min_good_frac = ptcproc.second_pass_local.min_good_frac;
    typed_second_pass.baseline_window_sec =
        ptcproc.second_pass_local.baseline_window_sec;
    typed_second_pass.sigma_scale = ptcproc.second_pass_local.sigma_scale;
    typed_second_pass.delta_sigma_scale =
        ptcproc.second_pass_local.delta_sigma_scale;
    typed_second_pass.raw_candidate_rel_sigma_scale =
        ptcproc.second_pass_local.raw_candidate_rel_sigma_scale;
    typed_second_pass.raw_window_sec =
        ptcproc.second_pass_local.raw_window_sec;
    typed_second_pass.raw_half_peak_frac =
        ptcproc.second_pass_local.raw_half_peak_frac;
    typed_second_pass.raw_max_width_sec =
        ptcproc.second_pass_local.raw_max_width_sec;
    typed_second_pass.delta_window_sec =
        ptcproc.second_pass_local.delta_window_sec;
    typed_second_pass.delta_half_peak_frac =
        ptcproc.second_pass_local.delta_half_peak_frac;
    typed_second_pass.delta_max_width_sec =
        ptcproc.second_pass_local.delta_max_width_sec;
    typed_second_pass.max_step_shift_z =
        ptcproc.second_pass_local.max_step_shift_z;
    typed_second_pass.high_score_event_override =
        ptcproc.second_pass_local.high_score_event_override;
    typed_second_pass.merge_within_detector_sec =
        ptcproc.second_pass_local.merge_within_detector_sec;
    typed_second_pass.cluster_events_sec =
        ptcproc.second_pass_local.cluster_events_sec;
    typed_second_pass.min_cluster_detectors =
        ptcproc.second_pass_local.min_cluster_detectors;
    typed_second_pass.high_score_cluster_override =
        ptcproc.second_pass_local.high_score_cluster_override;
    typed_second_pass.max_auto_flag_clusters_per_network =
        ptcproc.second_pass_local.max_auto_flag_clusters_per_network;
    typed_second_pass.selective_busy_network_acceptance_enabled =
        ptcproc.second_pass_local.selective_busy_network_acceptance_enabled;
    typed_second_pass.source_protection.enabled =
        ptcproc.second_pass_local.source_protection_config_enabled;
    typed_second_pass.source_protection.radius_arcsec =
        ptcproc.second_pass_local.source_protection_radius_arcsec;

    // copy tod output bool for eigenvalues
    ptcproc.run_tod_output = run_tod_output;
    ptcproc.write_evals = diagnostics.write_evals;
}
