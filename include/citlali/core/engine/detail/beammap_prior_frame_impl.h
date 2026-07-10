#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/engine/detail/beammap_prior_frame_center_impl.h>
#include <citlali/core/engine/detail/beammap_prior_alignment_impl.h>

void Beammap::reset_beammap_prior_frame_estimates() {
    beammap_prior_array_center_x_arcsec.clear();
    beammap_prior_array_center_y_arcsec.clear();
    beammap_prior_array_alignment.clear();
}

bool Beammap::should_update_beammap_prior_alignment(
    const citlali::config::BeammapPriorsConfig &priors_config) const {
    return priors_config.align_after_iter0 &&
           is_beammap_measurement_iter(current_iter) &&
           p0.rows() == map_indices.n_maps &&
           p0.cols() > 2;
}

Eigen::Index Beammap::update_beammap_prior_alignment_estimates(
    const citlali::config::BeammapPriorsConfig &priors_config) {
    if (!should_update_beammap_prior_alignment(priors_config)) {
        return 0;
    }

    auto alignment_samples =
        collect_beammap_prior_alignment_samples(priors_config);
    const Eigen::Index n_alignment_matches = alignment_samples.n_matches;
    apply_beammap_prior_alignment_samples(alignment_samples, priors_config);
    return n_alignment_matches;
}

void Beammap::log_beammap_prior_frame_estimate_summary(
    const Beammap::BeammapPriorFrameCenterSamples &center_samples,
    Eigen::Index n_alignment_matches) const {
    logger->info(
        "beammap priors frame estimate (iter {}): previous={} blind={} arrays={} alignment_matches={} aligned_arrays={}",
        current_iter, center_samples.n_previous, center_samples.n_blind,
        beammap_prior_array_center_x_arcsec.size(),
        n_alignment_matches, beammap_prior_array_alignment.size());
}

void Beammap::update_prior_frame_estimates() {
    reset_beammap_prior_frame_estimates();

    const auto center_samples = collect_beammap_prior_frame_center_samples();
    apply_beammap_prior_frame_center_samples(center_samples);

    const auto &priors_config =
        citlali::pipeline::beammap_config(*this).priors;
    const Eigen::Index n_alignment_matches =
        update_beammap_prior_alignment_estimates(priors_config);

    log_beammap_prior_frame_estimate_summary(center_samples, n_alignment_matches);
}
