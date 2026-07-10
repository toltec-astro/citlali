#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

bool Beammap::is_beammap_prior_alignment_sample_candidate(Eigen::Index map_index) {
    if (map_index >= good_fits.size() || !good_fits(map_index)) {
        return false;
    }
    if (fit_diag_bound_nhit.size() == map_indices.n_maps && fit_diag_bound_nhit(map_index) > 0) {
        return false;
    }
    return std::isfinite(p0(map_index, 0)) && p0(map_index, 0) > 0.0 &&
           std::isfinite(p0(map_index, 1)) && std::isfinite(p0(map_index, 2));
}

bool Beammap::make_beammap_prior_alignment_pair(
    Eigen::Index map_index,
    const citlali::config::BeammapPriorsConfig &priors_config,
    double derot_elev_rad,
    int &array,
    Beammap::BeammapPriorAlignmentPair &pair) {
    if (!is_beammap_prior_alignment_sample_candidate(map_index)) {
        return false;
    }

    array = static_cast<int>(map_indices.maps_to_arrays(map_index));
    const int nw = static_cast<int>(std::lround(calib.apt["nw"](map_index)));
    const double x_raw =
        RAD_TO_ASEC * omb.pixel_size_rad * (p0(map_index, 1) - (omb.n_cols - 1) / 2.0);
    const double y_raw =
        RAD_TO_ASEC * omb.pixel_size_rad * (p0(map_index, 2) - (omb.n_rows - 1) / 2.0);

    double x_prior = std::numeric_limits<double>::quiet_NaN();
    double y_prior = std::numeric_limits<double>::quiet_NaN();
    if (!observed_to_prior_frame(array, x_raw, y_raw, derot_elev_rad,
                                 x_prior, y_prior, nullptr, nullptr, false)) {
        return false;
    }

    double d2 = std::numeric_limits<double>::infinity();
    int slot_index = -1;
    double slot_x = std::numeric_limits<double>::quiet_NaN();
    double slot_y = std::numeric_limits<double>::quiet_NaN();
    if (!match_prior_slot(array, nw, x_prior, y_prior, d2, slot_index, &slot_x, &slot_y)) {
        return false;
    }
    static_cast<void>(slot_index);
    if (priors_config.alignment_max_d2 > 0.0 &&
        d2 > priors_config.alignment_max_d2) {
        return false;
    }

    pair = BeammapPriorAlignmentPair{x_prior, y_prior, slot_x, slot_y};
    return true;
}

Beammap::BeammapPriorAlignmentSamples
Beammap::collect_beammap_prior_alignment_samples(
    const citlali::config::BeammapPriorsConfig &priors_config) {
    BeammapPriorAlignmentSamples alignment_samples;
    const double derot_elev_rad = get_prior_derot_elev_rad();

    for (Eigen::Index i = 0; i < map_indices.n_maps; ++i) {
        int array = 0;
        BeammapPriorAlignmentPair pair;
        if (!make_beammap_prior_alignment_pair(
                i, priors_config, derot_elev_rad, array, pair)) {
            continue;
        }
        alignment_samples.pairs_by_array[array].push_back(pair);
        alignment_samples.all_pairs.push_back(pair);
        alignment_samples.arrays_with_alignment_pairs.insert(array);
        alignment_samples.n_matches++;
    }

    return alignment_samples;
}
