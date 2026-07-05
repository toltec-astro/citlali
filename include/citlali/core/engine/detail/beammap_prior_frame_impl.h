#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

void Beammap::update_prior_frame_estimates() {
    beammap_prior_array_center_x_arcsec.clear();
    beammap_prior_array_center_y_arcsec.clear();
    beammap_prior_array_alignment.clear();

    std::map<int, std::vector<double>> x_by_array;
    std::map<int, std::vector<double>> y_by_array;
    std::set<int> arrays_missing;
    for (Eigen::Index i = 0; i < n_maps; ++i) {
        arrays_missing.insert(static_cast<int>(maps_to_arrays(i)));
    }

    Eigen::Index n_prev = 0;
    if (is_beammap_measurement_iter(current_iter) && p0.rows() == n_maps && p0.cols() > 2) {
        for (Eigen::Index i = 0; i < n_maps; ++i) {
            if (i < good_fits.size() && !good_fits(i)) {
                continue;
            }
            if (fit_diag_bound_nhit.size() == n_maps && fit_diag_bound_nhit(i) > 0) {
                continue;
            }
            if (!(std::isfinite(p0(i, 0)) && p0(i, 0) > 0.0 &&
                  std::isfinite(p0(i, 1)) && std::isfinite(p0(i, 2)))) {
                continue;
            }
            const int array = static_cast<int>(maps_to_arrays(i));
            const double x_arcsec =
                RAD_TO_ASEC * omb.pixel_size_rad * (p0(i, 1) - (omb.n_cols - 1) / 2.0);
            const double y_arcsec =
                RAD_TO_ASEC * omb.pixel_size_rad * (p0(i, 2) - (omb.n_rows - 1) / 2.0);
            x_by_array[array].push_back(x_arcsec);
            y_by_array[array].push_back(y_arcsec);
            arrays_missing.erase(array);
            n_prev++;
        }
    }

    Eigen::Index n_blind = 0;
    if (!arrays_missing.empty()) {
        for (Eigen::Index i = 0; i < n_maps; ++i) {
            const int array = static_cast<int>(maps_to_arrays(i));
            if (!arrays_missing.count(array)) {
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
            x_by_array[array].push_back(x_arcsec);
            y_by_array[array].push_back(y_arcsec);
            n_blind++;
        }
    }

    for (const auto &[array, xs] : x_by_array) {
        if (xs.empty()) {
            continue;
        }
        Eigen::Map<const Eigen::VectorXd> x_vec(xs.data(), static_cast<Eigen::Index>(xs.size()));
        auto y_it = y_by_array.find(array);
        if (y_it == y_by_array.end() || y_it->second.size() != xs.size()) {
            continue;
        }
        Eigen::Map<const Eigen::VectorXd> y_vec(y_it->second.data(), static_cast<Eigen::Index>(y_it->second.size()));
        beammap_prior_array_center_x_arcsec[array] = tula::alg::median(x_vec);
        beammap_prior_array_center_y_arcsec[array] = tula::alg::median(y_vec);
    }

    Eigen::Index n_alignment_matches = 0;
    if (beammap_priors_align_after_iter0 && is_beammap_measurement_iter(current_iter) &&
        p0.rows() == n_maps && p0.cols() > 2) {
        struct PriorPair {
            double obs_x = 0.0;
            double obs_y = 0.0;
            double slot_x = 0.0;
            double slot_y = 0.0;
        };
        std::map<int, std::vector<PriorPair>> pairs_by_array;
        std::vector<PriorPair> all_pairs;
        std::set<int> arrays_with_alignment_pairs;
        const double derot_elev_rad = get_prior_derot_elev_rad();

        for (Eigen::Index i = 0; i < n_maps; ++i) {
            if (i >= good_fits.size() || !good_fits(i)) {
                continue;
            }
            if (fit_diag_bound_nhit.size() == n_maps && fit_diag_bound_nhit(i) > 0) {
                continue;
            }
            if (!(std::isfinite(p0(i, 0)) && p0(i, 0) > 0.0 &&
                  std::isfinite(p0(i, 1)) && std::isfinite(p0(i, 2)))) {
                continue;
            }
            const int array = static_cast<int>(maps_to_arrays(i));
            const int nw = static_cast<int>(std::lround(calib.apt["nw"](i)));
            const double x_raw =
                RAD_TO_ASEC * omb.pixel_size_rad * (p0(i, 1) - (omb.n_cols - 1) / 2.0);
            const double y_raw =
                RAD_TO_ASEC * omb.pixel_size_rad * (p0(i, 2) - (omb.n_rows - 1) / 2.0);
            double x_prior = std::numeric_limits<double>::quiet_NaN();
            double y_prior = std::numeric_limits<double>::quiet_NaN();
            if (!observed_to_prior_frame(array, x_raw, y_raw, derot_elev_rad,
                                         x_prior, y_prior, nullptr, nullptr, false)) {
                continue;
            }
            double d2 = std::numeric_limits<double>::infinity();
            int slot_index = -1;
            double slot_x = std::numeric_limits<double>::quiet_NaN();
            double slot_y = std::numeric_limits<double>::quiet_NaN();
            if (!match_prior_slot(array, nw, x_prior, y_prior, d2, slot_index, &slot_x, &slot_y)) {
                continue;
            }
            static_cast<void>(slot_index);
            if (beammap_priors_alignment_max_d2 > 0.0 && d2 > beammap_priors_alignment_max_d2) {
                continue;
            }
            PriorPair pair{x_prior, y_prior, slot_x, slot_y};
            pairs_by_array[array].push_back(pair);
            all_pairs.push_back(pair);
            arrays_with_alignment_pairs.insert(array);
            n_alignment_matches++;
        }

        auto fit_prior_alignment = [&](const std::vector<PriorPair> &pairs,
                                       const std::string &label,
                                       PriorArrayAlignment &alignment) {
            if (pairs.size() < static_cast<std::size_t>(beammap_priors_alignment_min_matches)) {
                logger->debug("beammap prior alignment skipped {} matches={} min_matches={}",
                              label, pairs.size(), beammap_priors_alignment_min_matches);
                return false;
            }

            std::vector<double> dx_vals;
            std::vector<double> dy_vals;
            dx_vals.reserve(pairs.size());
            dy_vals.reserve(pairs.size());
            for (const auto &pair : pairs) {
                dx_vals.push_back(pair.slot_x - pair.obs_x);
                dy_vals.push_back(pair.slot_y - pair.obs_y);
            }
            Eigen::Map<Eigen::VectorXd> dx_vec(dx_vals.data(), static_cast<Eigen::Index>(dx_vals.size()));
            Eigen::Map<Eigen::VectorXd> dy_vec(dy_vals.data(), static_cast<Eigen::Index>(dy_vals.size()));
            double tx = tula::alg::median(dx_vec);
            double ty = tula::alg::median(dy_vec);

            double theta = 0.0;
            if (beammap_priors_alignment_fit_rotation) {
                double obs_mean_x = 0.0;
                double obs_mean_y = 0.0;
                double slot_mean_x = 0.0;
                double slot_mean_y = 0.0;
                for (const auto &pair : pairs) {
                    obs_mean_x += pair.obs_x + tx;
                    obs_mean_y += pair.obs_y + ty;
                    slot_mean_x += pair.slot_x;
                    slot_mean_y += pair.slot_y;
                }
                const double inv_n = 1.0 / static_cast<double>(pairs.size());
                obs_mean_x *= inv_n;
                obs_mean_y *= inv_n;
                slot_mean_x *= inv_n;
                slot_mean_y *= inv_n;

                double a = 0.0;
                double b = 0.0;
                for (const auto &pair : pairs) {
                    const double ox = pair.obs_x + tx - obs_mean_x;
                    const double oy = pair.obs_y + ty - obs_mean_y;
                    const double sx = pair.slot_x - slot_mean_x;
                    const double sy = pair.slot_y - slot_mean_y;
                    a += ox * sx + oy * sy;
                    b += ox * sy - oy * sx;
                }
                if (std::isfinite(a) && std::isfinite(b) &&
                    (std::abs(a) > 0.0 || std::abs(b) > 0.0)) {
                    theta = std::atan2(b, a);
                }
                const double max_theta = beammap_priors_alignment_max_rotation_deg * DEG_TO_RAD;
                if (!std::isfinite(theta) || std::abs(theta) > max_theta) {
                    logger->debug(
                        "beammap prior alignment {} rejected residual rotation {} deg (limit={} deg)",
                        label, theta * RAD_TO_DEG, beammap_priors_alignment_max_rotation_deg);
                    theta = 0.0;
                }
            }

            const double cos_theta = std::cos(theta);
            const double sin_theta = std::sin(theta);
            dx_vals.clear();
            dy_vals.clear();
            for (const auto &pair : pairs) {
                const double x_rot = cos_theta * pair.obs_x - sin_theta * pair.obs_y;
                const double y_rot = sin_theta * pair.obs_x + cos_theta * pair.obs_y;
                dx_vals.push_back(pair.slot_x - x_rot);
                dy_vals.push_back(pair.slot_y - y_rot);
            }
            Eigen::Map<Eigen::VectorXd> dx_vec_final(dx_vals.data(), static_cast<Eigen::Index>(dx_vals.size()));
            Eigen::Map<Eigen::VectorXd> dy_vec_final(dy_vals.data(), static_cast<Eigen::Index>(dy_vals.size()));
            tx = tula::alg::median(dx_vec_final);
            ty = tula::alg::median(dy_vec_final);

            double rss = 0.0;
            for (const auto &pair : pairs) {
                const double x_fit = cos_theta * pair.obs_x - sin_theta * pair.obs_y + tx;
                const double y_fit = sin_theta * pair.obs_x + cos_theta * pair.obs_y + ty;
                const double rx = x_fit - pair.slot_x;
                const double ry = y_fit - pair.slot_y;
                rss += rx * rx + ry * ry;
            }
            const double rms = std::sqrt(rss / static_cast<double>(pairs.size()));
            if (!(std::isfinite(tx) && std::isfinite(ty) && std::isfinite(rms))) {
                return false;
            }

            alignment.valid = true;
            alignment.cos_theta = cos_theta;
            alignment.sin_theta = sin_theta;
            alignment.theta_rad = theta;
            alignment.dx_arcsec = tx;
            alignment.dy_arcsec = ty;
            alignment.n_matches = static_cast<Eigen::Index>(pairs.size());
            alignment.rms_arcsec = rms;
            return true;
        };

        if (beammap_priors_alignment_scope == "common") {
            auto common_pairs = all_pairs;
            if (beammap_priors_alignment_common_support == "overlap_box" &&
                pairs_by_array.size() >= 2) {
                auto quantile = [](std::vector<double> values, double q) {
                    if (values.empty()) {
                        return std::numeric_limits<double>::quiet_NaN();
                    }
                    q = std::clamp(q, 0.0, 1.0);
                    std::sort(values.begin(), values.end());
                    const double pos = q * static_cast<double>(values.size() - 1);
                    const auto lo = static_cast<std::size_t>(std::floor(pos));
                    const auto hi = static_cast<std::size_t>(std::ceil(pos));
                    if (lo == hi) {
                        return values[lo];
                    }
                    const double frac = pos - static_cast<double>(lo);
                    return values[lo] * (1.0 - frac) + values[hi] * frac;
                };

                const double q_low = beammap_priors_alignment_common_support_quantile;
                const double q_high = 1.0 - beammap_priors_alignment_common_support_quantile;
                double overlap_x_low = -std::numeric_limits<double>::infinity();
                double overlap_x_high = std::numeric_limits<double>::infinity();
                double overlap_y_low = -std::numeric_limits<double>::infinity();
                double overlap_y_high = std::numeric_limits<double>::infinity();
                bool overlap_valid = true;

                for (const auto &[array, pairs] : pairs_by_array) {
                    static_cast<void>(array);
                    std::vector<double> xs;
                    std::vector<double> ys;
                    xs.reserve(pairs.size());
                    ys.reserve(pairs.size());
                    for (const auto &pair : pairs) {
                        if (std::isfinite(pair.slot_x) && std::isfinite(pair.slot_y)) {
                            xs.push_back(pair.slot_x);
                            ys.push_back(pair.slot_y);
                        }
                    }
                    const double x_low = quantile(xs, q_low);
                    const double x_high = quantile(xs, q_high);
                    const double y_low = quantile(ys, q_low);
                    const double y_high = quantile(ys, q_high);
                    if (!(std::isfinite(x_low) && std::isfinite(x_high) &&
                          std::isfinite(y_low) && std::isfinite(y_high))) {
                        overlap_valid = false;
                        break;
                    }
                    overlap_x_low = std::max(overlap_x_low, x_low);
                    overlap_x_high = std::min(overlap_x_high, x_high);
                    overlap_y_low = std::max(overlap_y_low, y_low);
                    overlap_y_high = std::min(overlap_y_high, y_high);
                }

                if (overlap_valid && overlap_x_low < overlap_x_high &&
                    overlap_y_low < overlap_y_high) {
                    std::vector<PriorPair> filtered_pairs;
                    filtered_pairs.reserve(all_pairs.size());
                    for (const auto &pair : all_pairs) {
                        if (pair.slot_x >= overlap_x_low && pair.slot_x <= overlap_x_high &&
                            pair.slot_y >= overlap_y_low && pair.slot_y <= overlap_y_high) {
                            filtered_pairs.push_back(pair);
                        }
                    }
                    if (filtered_pairs.size() >= static_cast<std::size_t>(beammap_priors_alignment_min_matches)) {
                        common_pairs.swap(filtered_pairs);
                    }
                    logger->info(
                        "beammap prior common alignment overlap_box (iter {}): q={} x=[{}, {}] y=[{}, {}] kept={}/{}",
                        current_iter, beammap_priors_alignment_common_support_quantile,
                        overlap_x_low, overlap_x_high, overlap_y_low, overlap_y_high,
                        common_pairs.size(), all_pairs.size());
                }
                else {
                    logger->debug(
                        "beammap prior common alignment overlap_box skipped: invalid overlap x=[{}, {}] y=[{}, {}]",
                        overlap_x_low, overlap_x_high, overlap_y_low, overlap_y_high);
                }
            }

            PriorArrayAlignment alignment;
            if (fit_prior_alignment(common_pairs, "scope=common", alignment)) {
                for (int array : arrays_with_alignment_pairs) {
                    beammap_prior_array_alignment[array] = alignment;
                }
                logger->info(
                    "beammap prior empirical alignment (iter {} scope=common): arrays={} matches={} dx={} dy={} rot_deg={} rms={}",
                    current_iter, arrays_with_alignment_pairs.size(), alignment.n_matches,
                    alignment.dx_arcsec, alignment.dy_arcsec,
                    alignment.theta_rad * RAD_TO_DEG, alignment.rms_arcsec);
            }
        }
        else {
            for (auto &[array, pairs] : pairs_by_array) {
                PriorArrayAlignment alignment;
                if (!fit_prior_alignment(pairs, fmt::format("array={}", array), alignment)) {
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

    logger->info(
        "beammap priors frame estimate (iter {}): previous={} blind={} arrays={} alignment_matches={} aligned_arrays={}",
        current_iter, n_prev, n_blind, beammap_prior_array_center_x_arcsec.size(),
        n_alignment_matches, beammap_prior_array_alignment.size());
}
