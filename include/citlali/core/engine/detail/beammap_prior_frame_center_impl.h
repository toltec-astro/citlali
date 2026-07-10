#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

Beammap::BeammapPriorFrameCenterSamples
Beammap::collect_beammap_prior_frame_center_samples() {
    BeammapPriorFrameCenterSamples center_samples;
    for (Eigen::Index i = 0; i < map_indices.n_maps; ++i) {
        center_samples.arrays_missing.insert(
            static_cast<int>(map_indices.maps_to_arrays(i)));
    }

    if (is_beammap_measurement_iter(current_iter) && p0.rows() == map_indices.n_maps && p0.cols() > 2) {
        for (Eigen::Index i = 0; i < map_indices.n_maps; ++i) {
            if (i < good_fits.size() && !good_fits(i)) {
                continue;
            }
            if (fit_diag_bound_nhit.size() == map_indices.n_maps && fit_diag_bound_nhit(i) > 0) {
                continue;
            }
            if (!(std::isfinite(p0(i, 0)) && p0(i, 0) > 0.0 &&
                  std::isfinite(p0(i, 1)) && std::isfinite(p0(i, 2)))) {
                continue;
            }
            const int array = static_cast<int>(map_indices.maps_to_arrays(i));
            const double x_arcsec =
                RAD_TO_ASEC * omb.pixel_size_rad * (p0(i, 1) - (omb.n_cols - 1) / 2.0);
            const double y_arcsec =
                RAD_TO_ASEC * omb.pixel_size_rad * (p0(i, 2) - (omb.n_rows - 1) / 2.0);
            center_samples.x_by_array[array].push_back(x_arcsec);
            center_samples.y_by_array[array].push_back(y_arcsec);
            center_samples.arrays_missing.erase(array);
            center_samples.n_previous++;
        }
    }

    if (!center_samples.arrays_missing.empty()) {
        for (Eigen::Index i = 0; i < map_indices.n_maps; ++i) {
            const int array = static_cast<int>(map_indices.maps_to_arrays(i));
            if (!center_samples.arrays_missing.count(array)) {
                continue;
            }

            Eigen::Index peak_row = -1;
            Eigen::Index peak_col = -1;
            double peak_snr = -std::numeric_limits<double>::infinity();
            if (!find_map_weighted_peak(i, peak_row, peak_col, peak_snr)) {
                continue;
            }

            const double x_arcsec =
                RAD_TO_ASEC * omb.pixel_size_rad * (static_cast<double>(peak_col) - (omb.n_cols - 1) / 2.0);
            const double y_arcsec =
                RAD_TO_ASEC * omb.pixel_size_rad * (static_cast<double>(peak_row) - (omb.n_rows - 1) / 2.0);
            center_samples.x_by_array[array].push_back(x_arcsec);
            center_samples.y_by_array[array].push_back(y_arcsec);
            center_samples.n_blind++;
        }
    }

    return center_samples;
}

void Beammap::apply_beammap_prior_frame_center_samples(
    const Beammap::BeammapPriorFrameCenterSamples &center_samples) {
    for (const auto &[array, xs] : center_samples.x_by_array) {
        if (xs.empty()) {
            continue;
        }
        Eigen::Map<const Eigen::VectorXd> x_vec(xs.data(), static_cast<Eigen::Index>(xs.size()));
        auto y_it = center_samples.y_by_array.find(array);
        if (y_it == center_samples.y_by_array.end() || y_it->second.size() != xs.size()) {
            continue;
        }
        Eigen::Map<const Eigen::VectorXd> y_vec(y_it->second.data(), static_cast<Eigen::Index>(y_it->second.size()));
        beammap_prior_array_center_x_arcsec[array] = tula::alg::median(x_vec);
        beammap_prior_array_center_y_arcsec[array] = tula::alg::median(y_vec);
    }
}
