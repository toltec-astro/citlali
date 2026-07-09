#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/pipeline/reduction_config_accessors.h>

bool Beammap::choose_prior_guided_init(Eigen::Index map_index, double &init_row, double &init_col) {
    init_row = -99.0;
    init_col = -99.0;

    auto set_prior_diag = [&](PriorDiagColumn col, double value) {
        if (map_index >= 0 && map_index < prior_diag_values.rows() &&
            col >= 0 && col < prior_diag_values.cols()) {
            prior_diag_values(map_index, col) = value;
        }
    };

    constexpr int prior_reason_none = 0;
    constexpr int prior_reason_no_slot_group = 1;
    constexpr int prior_reason_no_valid_weighted_pixels = 2;
    constexpr int prior_reason_invalid_sigma = 3;
    constexpr int prior_reason_below_min_snr = 4;
    constexpr int prior_reason_gate_rejected = 5;

    if (!beammap_soft_priors_loaded ||
        citlali::pipeline::mapmaking_config(*this).grouping !=
            citlali::config::MapGrouping::detector) {
        return false;
    }
    if (map_index < 0 || map_index >= map_indices.n_maps || map_index >= calib.n_dets) {
        return false;
    }
    if (map_index >= map_indices.maps_to_arrays.size() || map_index >= calib.apt["nw"].size()) {
        return false;
    }

    const int array = static_cast<int>(map_indices.maps_to_arrays(map_index));
    const int nw = static_cast<int>(std::lround(calib.apt["nw"](map_index)));
    auto slots_it = beammap_soft_prior_slots.find({array, nw});
    if (slots_it == beammap_soft_prior_slots.end() || slots_it->second.empty()) {
        set_prior_diag(prior_no_candidate_reason_col, prior_reason_no_slot_group);
        return false;
    }

    const auto &sig = omb.signal[map_index];
    const auto &wt = omb.weight[map_index];
    if (sig.rows() <= 0 || sig.cols() <= 0 || wt.rows() != sig.rows() || wt.cols() != sig.cols()) {
        set_prior_diag(prior_no_candidate_reason_col, prior_reason_no_valid_weighted_pixels);
        return false;
    }

    struct Candidate {
        double snr = 0.0;
        Eigen::Index row = 0;
        Eigen::Index col = 0;
    };

    std::vector<double> valid_signal;
    std::vector<double> valid_weight;
    valid_signal.reserve(static_cast<std::size_t>(sig.size()));
    valid_weight.reserve(static_cast<std::size_t>(sig.size()));
    for (Eigen::Index row = 0; row < sig.rows(); ++row) {
        for (Eigen::Index col = 0; col < sig.cols(); ++col) {
            const double s = sig(row, col);
            const double w = wt(row, col);
            if (!std::isfinite(s) || !std::isfinite(w) || w <= 0.0) {
                continue;
            }
            valid_signal.push_back(s);
            valid_weight.push_back(w);
        }
    }
    if (valid_signal.empty()) {
        set_prior_diag(prior_no_candidate_reason_col, prior_reason_no_valid_weighted_pixels);
        return false;
    }

    Eigen::Map<Eigen::VectorXd> sig_vec(valid_signal.data(), static_cast<Eigen::Index>(valid_signal.size()));
    const double sig_med = tula::alg::median(sig_vec);
    Eigen::VectorXd sig_abs_dev = (sig_vec.array() - sig_med).abs().matrix();
    double sig_sigma = 1.4826 * tula::alg::median(sig_abs_dev);
    if (!std::isfinite(sig_sigma) || sig_sigma <= std::numeric_limits<double>::epsilon()) {
        sig_sigma = engine_utils::calc_std_dev(sig_vec);
    }
    if (!std::isfinite(sig_sigma) || sig_sigma <= std::numeric_limits<double>::epsilon()) {
        set_prior_diag(prior_no_candidate_reason_col, prior_reason_invalid_sigma);
        return false;
    }

    Eigen::Map<Eigen::VectorXd> wt_vec(valid_weight.data(), static_cast<Eigen::Index>(valid_weight.size()));
    double wt_med = tula::alg::median(wt_vec);
    if (!std::isfinite(wt_med) || wt_med <= std::numeric_limits<double>::epsilon()) {
        wt_med = 1.0;
    }
    const auto &priors_config =
        citlali::pipeline::beammap_config(*this).priors;

    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>(sig.size()));
    for (Eigen::Index row = 0; row < sig.rows(); ++row) {
        for (Eigen::Index col = 0; col < sig.cols(); ++col) {
            const double s = sig(row, col);
            const double w = wt(row, col);
            if (!std::isfinite(s) || !std::isfinite(w) || w <= 0.0) {
                continue;
            }
            const double snr = ((s - sig_med) / sig_sigma) * std::sqrt(w / wt_med);
            if (!std::isfinite(snr) || snr < priors_config.min_snr) {
                continue;
            }
            candidates.push_back({snr, row, col});
        }
    }
    if (candidates.empty()) {
        logger->debug("beammap priors init map={} no candidates above min_snr={:.4g} (med={:.4g} sigma={:.4g} wt_med={:.4g})",
                      map_index, priors_config.min_snr, sig_med, sig_sigma, wt_med);
        set_prior_diag(prior_n_candidates_col, 0.0);
        set_prior_diag(prior_n_candidates_keep_col, 0.0);
        set_prior_diag(prior_n_candidates_gate_col, 0.0);
        set_prior_diag(prior_no_candidate_reason_col, prior_reason_below_min_snr);
        return false;
    }

    set_prior_diag(prior_n_candidates_col, static_cast<double>(candidates.size()));

    const std::size_t n_keep = std::min<std::size_t>(
        candidates.size(), static_cast<std::size_t>(std::max(1, priors_config.candidate_top_n)));
    set_prior_diag(prior_n_candidates_keep_col, static_cast<double>(n_keep));
    std::partial_sort(candidates.begin(), candidates.begin() + n_keep, candidates.end(),
                      [](const Candidate &a, const Candidate &b) { return a.snr > b.snr; });

    const double col0 = static_cast<double>(omb.n_cols - 1) / 2.0;
    const double row0 = static_cast<double>(omb.n_rows - 1) / 2.0;
    const double pix_to_arcsec = RAD_TO_ASEC * omb.pixel_size_rad;
    double derot_elev_rad = get_prior_derot_elev_rad();
    set_prior_diag(prior_derot_elev_col, derot_elev_rad);
    const double prior_max_d2 = effective_prior_max_d2();
    const double prior_score_lambda = effective_prior_score_lambda();

    bool found = false;
    double best_score = -std::numeric_limits<double>::infinity();
    double best_snr = -std::numeric_limits<double>::infinity();
    double best_d2 = std::numeric_limits<double>::infinity();
    Eigen::Index best_row = -1;
    Eigen::Index best_col = -1;
    int best_slot = -1;
    double best_x_raw = std::numeric_limits<double>::quiet_NaN();
    double best_y_raw = std::numeric_limits<double>::quiet_NaN();
    double best_x_prior = std::numeric_limits<double>::quiet_NaN();
    double best_y_prior = std::numeric_limits<double>::quiet_NaN();
    double best_slot_x = std::numeric_limits<double>::quiet_NaN();
    double best_slot_y = std::numeric_limits<double>::quiet_NaN();
    double best_slot_sx = std::numeric_limits<double>::quiet_NaN();
    double best_slot_sy = std::numeric_limits<double>::quiet_NaN();
    Eigen::Index n_gate = 0;

    for (std::size_t i = 0; i < n_keep; ++i) {
        const auto &cand = candidates[i];
        double x_arcsec_raw = pix_to_arcsec * (static_cast<double>(cand.col) - col0);
        double y_arcsec_raw = pix_to_arcsec * (static_cast<double>(cand.row) - row0);
        double center_x = std::numeric_limits<double>::quiet_NaN();
        double center_y = std::numeric_limits<double>::quiet_NaN();
        double x_arcsec = std::numeric_limits<double>::quiet_NaN();
        double y_arcsec = std::numeric_limits<double>::quiet_NaN();
        if (!observed_to_prior_frame(array, x_arcsec_raw, y_arcsec_raw, derot_elev_rad,
                                     x_arcsec, y_arcsec, &center_x, &center_y, true)) {
            continue;
        }

        double min_d2 = std::numeric_limits<double>::infinity();
        int min_slot = -1;
        double slot_x = std::numeric_limits<double>::quiet_NaN();
        double slot_y = std::numeric_limits<double>::quiet_NaN();
        double slot_sx = std::numeric_limits<double>::quiet_NaN();
        double slot_sy = std::numeric_limits<double>::quiet_NaN();
        if (!match_prior_slot(array, nw, x_arcsec, y_arcsec, min_d2, min_slot,
                              &slot_x, &slot_y, &slot_sx, &slot_sy)) {
            continue;
        }
        if (prior_max_d2 > 0.0 && min_d2 > prior_max_d2) {
            continue;
        }
        n_gate++;

        const double score = cand.snr - prior_score_lambda * min_d2;
        if (!found || score > best_score || (score == best_score && cand.snr > best_snr)) {
            found = true;
            best_score = score;
            best_snr = cand.snr;
            best_d2 = min_d2;
            best_row = cand.row;
            best_col = cand.col;
            best_slot = min_slot;
            best_x_raw = x_arcsec_raw;
            best_y_raw = y_arcsec_raw;
            best_x_prior = x_arcsec;
            best_y_prior = y_arcsec;
            best_slot_x = slot_x;
            best_slot_y = slot_y;
            best_slot_sx = slot_sx;
            best_slot_sy = slot_sy;
            if (std::isfinite(center_x) && std::isfinite(center_y)) {
                set_prior_diag(prior_center_x_col, center_x);
                set_prior_diag(prior_center_y_col, center_y);
            }
        }
    }

    set_prior_diag(prior_n_candidates_gate_col, static_cast<double>(n_gate));

    if (!found) {
        set_prior_diag(prior_no_candidate_reason_col, prior_reason_gate_rejected);
        return false;
    }

    init_row = static_cast<double>(best_row);
    init_col = static_cast<double>(best_col);
    set_prior_diag(prior_used_col, 1.0);
    set_prior_diag(prior_no_candidate_reason_col, prior_reason_none);
    set_prior_diag(prior_slot_index_col, static_cast<double>(best_slot));
    set_prior_diag(prior_match_d2_col, best_d2);
    set_prior_diag(prior_match_score_col, best_score);
    set_prior_diag(prior_candidate_snr_col, best_snr);
    set_prior_diag(prior_candidate_x_raw_col, best_x_raw);
    set_prior_diag(prior_candidate_y_raw_col, best_y_raw);
    set_prior_diag(prior_candidate_x_prior_col, best_x_prior);
    set_prior_diag(prior_candidate_y_prior_col, best_y_prior);
    set_prior_diag(prior_slot_x_col, best_slot_x);
    set_prior_diag(prior_slot_y_col, best_slot_y);
    set_prior_diag(prior_slot_sx_col, best_slot_sx);
    set_prior_diag(prior_slot_sy_col, best_slot_sy);
    logger->debug(
        "beammap priors init map={} det={} array={} nw={} row={} col={} snr={} d2={} slot={} lambda={} max_d2={}",
        map_index, map_index, array, nw, init_row, init_col, best_snr, best_d2,
        best_slot, prior_score_lambda, prior_max_d2);
    return true;
}
