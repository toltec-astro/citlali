#pragma once

// Beammap fit initialization implementation detail.
// Include this only after Beammap has been declared.

bool Beammap::has_previous_beammap_fit_init_candidate(
    Eigen::Index map_index, bool measurement_iter) const {
    return measurement_iter &&
           good_fits(map_index) &&
           p0.cols() > 2 &&
           std::isfinite(p0(map_index, 0)) && p0(map_index, 0) > 0.0 &&
           std::isfinite(p0(map_index, 1)) &&
           std::isfinite(p0(map_index, 2));
}

Beammap::BeammapPreviousFitSeed Beammap::read_previous_beammap_fit_seed(
    Eigen::Index map_index, double prev_row, double prev_col) const {
    BeammapPreviousFitSeed seed;
    Eigen::Index prev_row_i =
        static_cast<Eigen::Index>(std::llround(prev_row));
    Eigen::Index prev_col_i =
        static_cast<Eigen::Index>(std::llround(prev_col));
    if (prev_row_i < 0 || prev_row_i >= omb.signal[map_index].rows() ||
        prev_col_i < 0 || prev_col_i >= omb.signal[map_index].cols()) {
        return seed;
    }

    seed.weight = omb.weight[map_index](prev_row_i, prev_col_i);
    seed.signal = omb.signal[map_index](prev_row_i, prev_col_i);
    seed.valid = std::isfinite(seed.weight) && seed.weight > 0.0 &&
                 std::isfinite(seed.signal) && seed.signal > 0.0;
    return seed;
}

bool Beammap::should_reject_previous_beammap_fit_for_peak(
    Eigen::Index map_index, double prev_row, double prev_col,
    const Beammap::BeammapPreviousFitSeed &seed, bool can_try_prior,
    double init_fwhm) {
    Eigen::Index peak_row = -1;
    Eigen::Index peak_col = -1;
    double peak_snr = -std::numeric_limits<double>::infinity();
    if (!find_map_weighted_peak(map_index, peak_row, peak_col, peak_snr) ||
        peak_row < 0 || peak_col < 0 || !std::isfinite(peak_snr)) {
        return false;
    }

    const double prev_snr = seed.signal * std::sqrt(seed.weight);
    const double dr = static_cast<double>(peak_row) - prev_row;
    const double dc = static_cast<double>(peak_col) - prev_col;
    const double dist_pix = std::sqrt(dr * dr + dc * dc);
    const double min_switch_dist_pix = std::max(1.0, init_fwhm);
    constexpr double min_switch_snr_ratio = 1.25;
    const bool prior_allows_switch =
        !can_try_prior ||
        beammap_prior_allows_peak_switch(
            map_index, prev_row, prev_col, peak_row, peak_col);
    if (std::isfinite(prev_snr) &&
        peak_snr > min_switch_snr_ratio * prev_snr &&
        dist_pix > min_switch_dist_pix &&
        prior_allows_switch) {
        logger->debug(
            "beammap fit map={} rejected previous init: current weighted peak row={} col={} snr={} is {} pix from previous row={} col={} snr={}",
            map_index, peak_row, peak_col, peak_snr, dist_pix,
            prev_row, prev_col, prev_snr);
        return true;
    }
    return false;
}

Beammap::BeammapPreviousFitInit Beammap::choose_previous_beammap_fit_init(
    Eigen::Index map_index, bool measurement_iter, bool can_try_prior,
    double init_fwhm) {
    BeammapPreviousFitInit result;

    if (!has_previous_beammap_fit_init_candidate(map_index, measurement_iter)) {
        return result;
    }

    const double prev_col = p0(map_index, 1);
    const double prev_row = p0(map_index, 2);
    const auto seed =
        read_previous_beammap_fit_seed(map_index, prev_row, prev_col);
    bool use_previous_seed = seed.valid;
    if (use_previous_seed &&
        should_reject_previous_beammap_fit_for_peak(
            map_index, prev_row, prev_col, seed, can_try_prior, init_fwhm)) {
        use_previous_seed = false;
        result.rejected_by_peak = true;
    }

    if (use_previous_seed) {
        result.valid = true;
        result.col = prev_col;
        result.row = prev_row;
    }
    else {
        logger->debug(
            "beammap fit map={} rejected previous init at row={} col={} due to invalid/no-weight/non-positive seed pixel",
            map_index, prev_row, prev_col);
    }
    return result;
}
