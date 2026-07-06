#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_prior_qc_stats.h>

void Beammap::apply_final_network_position_flags() {
    if (typed_config.mapmaking.grouping !=
        citlali::config::MapGrouping::detector) {
        return;
    }

    bool enabled = false;
    for (const auto &[arr_index, arr_name] : toltec_io.array_name_map) {
        auto it = network_robust_z.find(arr_name);
        if (it != network_robust_z.end() && it->second > 0.0) {
            enabled = true;
            break;
        }
    }
    if (!enabled) {
        return;
    }

    struct NetworkStats {
        bool valid = false;
        double median_x = 0.0;
        double median_y = 0.0;
        double sigma_x = 0.0;
        double sigma_y = 0.0;
        double threshold = 0.0;
    };

    std::map<std::pair<int, int>, NetworkStats> stats_by_network;
    constexpr Eigen::Index min_network_samples = 16;

    logger->debug("flagging final detector network positions");
    for (Eigen::Index i = 0; i < calib.n_arrays; ++i) {
        Eigen::Index array = calib.arrays(i);
        std::string array_name = toltec_io.array_name_map[array];
        const double threshold = network_robust_z[array_name];
        if (!(threshold > 0.0)) {
            continue;
        }

        for (Eigen::Index j = 0; j < calib.n_nws; ++j) {
            Eigen::Index nw = calib.nws(j);
            if (std::get<0>(calib.nw_limits[nw]) < 0 ||
                std::get<1>(calib.nw_limits[nw]) <= std::get<0>(calib.nw_limits[nw])) {
                continue;
            }
            if (static_cast<Eigen::Index>(calib.apt["array"](std::get<0>(calib.nw_limits[nw]))) != array) {
                continue;
            }

            std::vector<double> x_vals;
            std::vector<double> y_vals;
            x_vals.reserve(static_cast<std::size_t>(std::get<1>(calib.nw_limits[nw]) - std::get<0>(calib.nw_limits[nw])));
            y_vals.reserve(x_vals.capacity());

            for (Eigen::Index k = std::get<0>(calib.nw_limits[nw]); k < std::get<1>(calib.nw_limits[nw]); ++k) {
                if (calib.apt["flag"](k) != 0) {
                    continue;
                }
                const double x = calib.apt["x_t"](k);
                const double y = calib.apt["y_t"](k);
                if (!std::isfinite(x) || !std::isfinite(y)) {
                    continue;
                }
                x_vals.push_back(x);
                y_vals.push_back(y);
            }
            if (static_cast<Eigen::Index>(x_vals.size()) < min_network_samples) {
                continue;
            }

            Eigen::Map<Eigen::VectorXd> x_vec(x_vals.data(), static_cast<Eigen::Index>(x_vals.size()));
            Eigen::Map<Eigen::VectorXd> y_vec(y_vals.data(), static_cast<Eigen::Index>(y_vals.size()));
            const double median_x = tula::alg::median(x_vec);
            const double median_y = tula::alg::median(y_vec);
            Eigen::VectorXd x_abs_dev = (x_vec.array() - median_x).abs().matrix();
            Eigen::VectorXd y_abs_dev = (y_vec.array() - median_y).abs().matrix();
            double sigma_x = 1.4826 * tula::alg::median(x_abs_dev);
            double sigma_y = 1.4826 * tula::alg::median(y_abs_dev);
            if (!std::isfinite(sigma_x) || sigma_x <= std::numeric_limits<double>::epsilon()) {
                sigma_x = engine_utils::calc_std_dev(x_vec);
            }
            if (!std::isfinite(sigma_y) || sigma_y <= std::numeric_limits<double>::epsilon()) {
                sigma_y = engine_utils::calc_std_dev(y_vec);
            }
            if (!std::isfinite(sigma_x) || !std::isfinite(sigma_y) ||
                sigma_x <= std::numeric_limits<double>::epsilon() ||
                sigma_y <= std::numeric_limits<double>::epsilon()) {
                continue;
            }

            stats_by_network[{static_cast<int>(array), static_cast<int>(nw)}] =
                {true, median_x, median_y, sigma_x, sigma_y, threshold};
        }
    }

    std::atomic<int> n_flagged{0};
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        if (calib.apt["flag"](i) != 0) {
            return 0;
        }

        const int array_index = static_cast<int>(std::lround(calib.apt["array"](i)));
        const int nw_index = static_cast<int>(std::lround(calib.apt["nw"](i)));
        auto it = stats_by_network.find({array_index, nw_index});
        if (it == stats_by_network.end() || !it->second.valid) {
            return 0;
        }

        const double x = calib.apt["x_t"](i);
        const double y = calib.apt["y_t"](i);
        if (!std::isfinite(x) || !std::isfinite(y)) {
            return 0;
        }

        const double zx = (x - it->second.median_x) / it->second.sigma_x;
        const double zy = (y - it->second.median_y) / it->second.sigma_y;
        const double z = std::sqrt(zx * zx + zy * zy);
        if (!std::isfinite(z) || z <= it->second.threshold) {
            return 0;
        }

        calib.apt["flag"](i) = 1;
        calib.apt["flxscale"](i) = 0.0;
        calib.apt["sens"](i) = 0.0;
        flag2(i) |= AptFlags::NetworkPos;
        n_flagged++;
        return 0;
    });

    if (n_flagged.load() > 0) {
        std::string by_array;
        for (Eigen::Index i = 0; i < calib.n_arrays; ++i) {
            Eigen::Index array = calib.arrays(i);
            std::string array_name = toltec_io.array_name_map[array];
            Eigen::Index n_array_flagged = 0;
            if (calib.array_limits.count(array) > 0) {
                for (Eigen::Index k = std::get<0>(calib.array_limits[array]);
                     k < std::get<1>(calib.array_limits[array]); ++k) {
                    if ((flag2(k) & AptFlags::NetworkPos) != 0) {
                        n_array_flagged++;
                    }
                }
            }
            if (!by_array.empty()) {
                by_array += ", ";
            }
            by_array += array_name + "=" + std::to_string(n_array_flagged);
        }
        logger->info(
            "beammap final network-position flagging: {} detectors exceeded per-array robust-z limits ({})",
            n_flagged.load(), by_array);
    }
}

void Beammap::update_final_prior_match_diagnostics() {
    final_prior_d2_diag = Eigen::VectorXd::Constant(
        calib.n_dets, std::numeric_limits<double>::quiet_NaN());
    final_prior_slot_index_diag = Eigen::VectorXi::Constant(calib.n_dets, -1);

    if (typed_config.mapmaking.grouping !=
            citlali::config::MapGrouping::detector ||
        !beammap_soft_priors_loaded || beammap_soft_prior_slots.empty()) {
        return;
    }

    struct ArrayCenter {
        bool valid = false;
        double x = 0.0;
        double y = 0.0;
    };

    std::map<int, ArrayCenter> centers;
    for (Eigen::Index i = 0; i < calib.n_arrays; ++i) {
        const Eigen::Index array = calib.arrays(i);
        std::vector<double> x_vals;
        std::vector<double> y_vals;

        auto gather = [&](bool unflagged_only) {
            x_vals.clear();
            y_vals.clear();
            for (Eigen::Index k = 0; k < calib.n_dets; ++k) {
                if (static_cast<Eigen::Index>(std::lround(calib.apt["array"](k))) != array) {
                    continue;
                }
                if (unflagged_only && calib.apt["flag"](k) != 0) {
                    continue;
                }
                const double x = calib.apt["x_t_raw"](k);
                const double y = calib.apt["y_t_raw"](k);
                if (!std::isfinite(x) || !std::isfinite(y)) {
                    continue;
                }
                x_vals.push_back(x);
                y_vals.push_back(y);
            }
        };

        gather(true);
        if (x_vals.size() < 8) {
            gather(false);
        }
        if (x_vals.empty()) {
            continue;
        }

        double median_x = beammap_prior_qc_stats::median_or_nan(x_vals);
        double median_y = beammap_prior_qc_stats::median_or_nan(y_vals);
        if (!std::isfinite(median_x) || !std::isfinite(median_y)) {
            continue;
        }
        centers[static_cast<int>(array)] = {true, median_x, median_y};
    }

    for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
        const int array = static_cast<int>(std::lround(calib.apt["array"](i)));
        const int nw = static_cast<int>(std::lround(calib.apt["nw"](i)));
        auto slots_it = beammap_soft_prior_slots.find({array, nw});
        if (slots_it == beammap_soft_prior_slots.end() || slots_it->second.empty()) {
            continue;
        }

        double x_arcsec = calib.apt["x_t_raw"](i);
        double y_arcsec = calib.apt["y_t_raw"](i);
        if (!std::isfinite(x_arcsec) || !std::isfinite(y_arcsec)) {
            continue;
        }

        if (beammap_soft_priors_are_centered) {
            auto center_it = centers.find(array);
            if (center_it != centers.end() && center_it->second.valid) {
                x_arcsec -= center_it->second.x;
                y_arcsec -= center_it->second.y;
            }
        }

        if (beammap_soft_priors_are_derotated && telescope.pixel_axes == "altaz") {
            double derot_elev_rad = calib.apt["derot_elev"](i);
            if (!std::isfinite(derot_elev_rad)) {
                derot_elev_rad = telescope.tel_data["TelElAct"].mean();
            }
            if (!std::isfinite(derot_elev_rad)) {
                derot_elev_rad = 0.0;
            }
            if (std::abs(derot_elev_rad) > pi) {
                derot_elev_rad *= DEG_TO_RAD;
            }
            const double rot_az_off = std::cos(-derot_elev_rad) * x_arcsec -
                                      std::sin(-derot_elev_rad) * y_arcsec;
            const double rot_alt_off = std::sin(-derot_elev_rad) * x_arcsec +
                                       std::cos(-derot_elev_rad) * y_arcsec;
            x_arcsec = -rot_az_off;
            y_arcsec = -rot_alt_off;
        }

        double best_d2 = std::numeric_limits<double>::infinity();
        int best_slot = -1;
        for (const auto &slot : slots_it->second) {
            const double sx = std::max(slot.sx_arcsec, std::numeric_limits<double>::epsilon());
            const double sy = std::max(slot.sy_arcsec, std::numeric_limits<double>::epsilon());
            const double dx = (x_arcsec - slot.x_arcsec) / sx;
            const double dy = (y_arcsec - slot.y_arcsec) / sy;
            const double d2 = dx * dx + dy * dy;
            if (std::isfinite(d2) && d2 < best_d2) {
                best_d2 = d2;
                best_slot = slot.slot_index;
            }
        }
        if (std::isfinite(best_d2)) {
            final_prior_d2_diag(i) = best_d2;
            final_prior_slot_index_diag(i) = best_slot;
        }
    }
}

void Beammap::log_final_network_qc_summary() {
    if (typed_config.mapmaking.grouping !=
        citlali::config::MapGrouping::detector) {
        return;
    }

    logger->info("beammap final per-network qc summary follows");
    for (Eigen::Index i = 0; i < calib.n_arrays; ++i) {
        const Eigen::Index array = calib.arrays(i);
        const std::string array_name = toltec_io.array_name_map[array];

        for (Eigen::Index j = 0; j < calib.n_nws; ++j) {
            const Eigen::Index nw = calib.nws(j);
            if (calib.nw_limits.count(nw) == 0) {
                continue;
            }
            const auto [k0, k1] = calib.nw_limits[nw];
            if (k0 < 0 || k1 <= k0) {
                continue;
            }
            if (static_cast<Eigen::Index>(std::lround(calib.apt["array"](k0))) != array) {
                continue;
            }

            std::vector<double> a_vals;
            std::vector<double> b_vals;
            std::vector<double> snr_vals;
            std::vector<double> prior_d2_vals;
            Eigen::Index n_total = 0;
            Eigen::Index n_good = 0;
            for (Eigen::Index k = k0; k < k1; ++k) {
                n_total++;
                if (calib.apt["flag"](k) != 0) {
                    continue;
                }
                n_good++;
                if (std::isfinite(calib.apt["a_fwhm"](k))) {
                    a_vals.push_back(calib.apt["a_fwhm"](k));
                }
                if (std::isfinite(calib.apt["b_fwhm"](k))) {
                    b_vals.push_back(calib.apt["b_fwhm"](k));
                }
                if (std::isfinite(calib.apt["sig2noise"](k))) {
                    snr_vals.push_back(calib.apt["sig2noise"](k));
                }
                if (final_prior_d2_diag.size() == calib.n_dets &&
                    std::isfinite(final_prior_d2_diag(k))) {
                    prior_d2_vals.push_back(final_prior_d2_diag(k));
                }
            }

            const double good_frac =
                static_cast<double>(n_good) / static_cast<double>(std::max<Eigen::Index>(1, n_total));
            logger->info(
                "beammap network qc: array={} nw={} good={}/{} ({:.3f}) med_a_fwhm={} med_b_fwhm={} med_sig2noise={} med_final_prior_d2={}",
                array_name,
                static_cast<int>(nw),
                n_good,
                n_total,
                good_frac,
                beammap_prior_qc_stats::median_or_nan(a_vals),
                beammap_prior_qc_stats::median_or_nan(b_vals),
                beammap_prior_qc_stats::median_or_nan(snr_vals),
                beammap_prior_qc_stats::median_or_nan(prior_d2_vals));
        }
    }
}
