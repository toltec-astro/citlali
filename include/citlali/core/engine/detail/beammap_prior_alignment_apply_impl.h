#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

void Beammap::apply_beammap_prior_alignment_samples(
    const Beammap::BeammapPriorAlignmentSamples &alignment_samples,
    const citlali::config::BeammapPriorsConfig &priors_config) {
    if (citlali::config::uses_common_prior_alignment(priors_config)) {
        auto common_pairs =
            select_common_beammap_prior_alignment_pairs(
                alignment_samples, priors_config);
        PriorArrayAlignment alignment;
        if (fit_beammap_prior_alignment(
                common_pairs, priors_config, "scope=common", alignment)) {
            for (int array : alignment_samples.arrays_with_alignment_pairs) {
                beammap_prior_array_alignment[array] = alignment;
            }
            logger->info(
                "beammap prior empirical alignment (iter {} scope=common): arrays={} matches={} dx={} dy={} rot_deg={} rms={}",
                current_iter, alignment_samples.arrays_with_alignment_pairs.size(), alignment.n_matches,
                alignment.dx_arcsec, alignment.dy_arcsec,
                alignment.theta_rad * RAD_TO_DEG, alignment.rms_arcsec);
        }
    }
    else {
        for (const auto &[array, pairs] : alignment_samples.pairs_by_array) {
            PriorArrayAlignment alignment;
            if (!fit_beammap_prior_alignment(
                    pairs, priors_config, fmt::format("array={}", array), alignment)) {
                continue;
            }
            beammap_prior_array_alignment[array] = alignment;

            logger->info(
                "beammap prior empirical alignment (iter {} array={}): matches={} dx={} dy={} rot_deg={} rms={}",
                current_iter, array, alignment.n_matches, alignment.dx_arcsec,
                alignment.dy_arcsec, alignment.theta_rad * RAD_TO_DEG, alignment.rms_arcsec);
        }
    }
}
