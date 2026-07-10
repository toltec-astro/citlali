#pragma once

// Beammap fit initialization implementation detail.
// Include this only after Beammap has been declared.

Beammap::BeammapPriorPositionCheck Beammap::beammap_prior_position_compatible(
    Eigen::Index map_index, double row, double col,
    double derot_elev_rad, double prior_max_d2) {
    BeammapPriorPositionCheck check;
    const int array_int = static_cast<int>(map_indices.maps_to_arrays(map_index));
    const int nw_int = static_cast<int>(std::lround(calib.apt["nw"](map_index)));
    const double pix_to_arcsec = RAD_TO_ASEC * omb.pixel_size_rad;
    const double col0 = static_cast<double>(omb.n_cols - 1) / 2.0;
    const double row0 = static_cast<double>(omb.n_rows - 1) / 2.0;
    const double x_raw = pix_to_arcsec * (col - col0);
    const double y_raw = pix_to_arcsec * (row - row0);
    double x_prior = std::numeric_limits<double>::quiet_NaN();
    double y_prior = std::numeric_limits<double>::quiet_NaN();
    int slot_index = -1;
    if (!observed_to_prior_frame(array_int, x_raw, y_raw, derot_elev_rad,
                                 x_prior, y_prior, nullptr, nullptr, true)) {
        return check;
    }
    if (!match_prior_slot(array_int, nw_int, x_prior, y_prior,
                          check.d2, slot_index)) {
        return check;
    }
    static_cast<void>(slot_index);
    check.compatible = prior_max_d2 <= 0.0 || check.d2 <= prior_max_d2;
    return check;
}

bool Beammap::beammap_prior_allows_peak_switch(Eigen::Index map_index,
                                               double prev_row, double prev_col,
                                               Eigen::Index peak_row,
                                               Eigen::Index peak_col) {
    const double derot_elev_rad = get_prior_derot_elev_rad();
    const double prior_max_d2 = effective_prior_max_d2();

    const auto prev_prior =
        beammap_prior_position_compatible(
            map_index, prev_row, prev_col, derot_elev_rad, prior_max_d2);
    const auto peak_prior = beammap_prior_position_compatible(
        map_index,
        static_cast<double>(peak_row), static_cast<double>(peak_col),
        derot_elev_rad, prior_max_d2);
    const bool prior_allows_switch =
        peak_prior.compatible || !prev_prior.compatible;
    if (!prior_allows_switch) {
        logger->debug(
            "beammap fit map={} kept previous init over stronger weighted peak because prior d2 prev={} peak={} max_d2={}",
            map_index, prev_prior.d2, peak_prior.d2, prior_max_d2);
    }
    return prior_allows_switch;
}
