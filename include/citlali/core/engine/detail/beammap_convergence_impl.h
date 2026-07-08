#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

double Beammap::calc_map_support_stddev(Eigen::Index map_index, bool exclude_fit_core) const {
    if (map_index < 0 ||
        map_index >= static_cast<Eigen::Index>(omb.signal.size()) ||
        map_index >= static_cast<Eigen::Index>(omb.weight.size())) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const auto &sig = omb.signal[map_index];
    const auto &wt = omb.weight[map_index];
    if (sig.rows() <= 0 || sig.cols() <= 0 ||
        wt.rows() != sig.rows() || wt.cols() != sig.cols()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double core_row = std::numeric_limits<double>::quiet_NaN();
    double core_col = std::numeric_limits<double>::quiet_NaN();
    double core_radius_pix = 0.0;
    if (exclude_fit_core &&
        map_index < params.rows() &&
        params.cols() >= 5 &&
        std::isfinite(params(map_index, 1)) &&
        std::isfinite(params(map_index, 2)) &&
        std::isfinite(params(map_index, 3)) &&
        std::isfinite(params(map_index, 4)) &&
        params(map_index, 3) > 0.0 &&
        params(map_index, 4) > 0.0) {
        core_col = params(map_index, 1);
        core_row = params(map_index, 2);
        core_radius_pix = 2.0 * STD_TO_FWHM *
                          std::max(params(map_index, 3), params(map_index, 4));
    }

    auto collect = [&](bool exclude_core) {
        std::vector<double> values;
        values.reserve(static_cast<std::size_t>(sig.size()));
        const bool do_exclude =
            exclude_core &&
            std::isfinite(core_row) &&
            std::isfinite(core_col) &&
            core_radius_pix > 0.0;
        const double core_radius2 = core_radius_pix * core_radius_pix;
        for (Eigen::Index row = 0; row < sig.rows(); ++row) {
            for (Eigen::Index col = 0; col < sig.cols(); ++col) {
                const double s = sig(row, col);
                const double w = wt(row, col);
                if (!std::isfinite(s) || !std::isfinite(w) || w <= 0.0) {
                    continue;
                }
                if (do_exclude) {
                    const double dr = static_cast<double>(row) - core_row;
                    const double dc = static_cast<double>(col) - core_col;
                    if (dr * dr + dc * dc <= core_radius2) {
                        continue;
                    }
                }
                values.push_back(s);
            }
        }
        return values;
    };

    auto values = collect(exclude_fit_core);
    if (values.size() < static_cast<std::size_t>(std::max(16, map_fitter.n_params + 1))) {
        values = collect(false);
    }
    if (values.size() < 2) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    Eigen::Map<Eigen::VectorXd> vec(values.data(), static_cast<Eigen::Index>(values.size()));
    return engine_utils::calc_std_dev(vec);
}

double Beammap::calc_beammap_convergence_delta(Eigen::Index map_index) const {
    if (map_index < 0 ||
        map_index >= static_cast<Eigen::Index>(omb.signal.size()) ||
        map_index >= static_cast<Eigen::Index>(omb_copy.signal.size())) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double convergence_radius_arcsec =
        typed_config.beammap.iteration.convergence_radius_arcsec;
    if (convergence_radius_arcsec <= 0.0 || omb.pixel_size_rad <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const auto &prev_sig = omb_copy.signal[map_index];
    const auto &cur_sig = omb.signal[map_index];
    if (prev_sig.rows() <= 0 || prev_sig.cols() <= 0 ||
        cur_sig.rows() != prev_sig.rows() || cur_sig.cols() != prev_sig.cols()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const bool have_prev_wt =
        map_index < static_cast<Eigen::Index>(omb_copy.weight.size()) &&
        omb_copy.weight[map_index].rows() == prev_sig.rows() &&
        omb_copy.weight[map_index].cols() == prev_sig.cols();
    const bool have_cur_wt =
        map_index < static_cast<Eigen::Index>(omb.weight.size()) &&
        omb.weight[map_index].rows() == cur_sig.rows() &&
        omb.weight[map_index].cols() == cur_sig.cols();

    auto choose_center = [&](const Eigen::MatrixXd &fit_params,
                             double &center_row, double &center_col) {
        if (map_index >= fit_params.rows() || fit_params.cols() < 3) {
            return false;
        }
        const double col = fit_params(map_index, 1);
        const double row = fit_params(map_index, 2);
        if (!std::isfinite(row) || !std::isfinite(col)) {
            return false;
        }
        center_row = row;
        center_col = col;
        return true;
    };

    double center_row = std::numeric_limits<double>::quiet_NaN();
    double center_col = std::numeric_limits<double>::quiet_NaN();
    if (!choose_center(params, center_row, center_col) &&
        !choose_center(p0, center_row, center_col)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double radius_pix =
        convergence_radius_arcsec * ASEC_TO_RAD / omb.pixel_size_rad;
    const double radius2 = radius_pix * radius_pix;
    const Eigen::Index row_min = std::max<Eigen::Index>(
        0, static_cast<Eigen::Index>(std::floor(center_row - radius_pix)));
    const Eigen::Index row_max = std::min<Eigen::Index>(
        prev_sig.rows() - 1, static_cast<Eigen::Index>(std::ceil(center_row + radius_pix)));
    const Eigen::Index col_min = std::max<Eigen::Index>(
        0, static_cast<Eigen::Index>(std::floor(center_col - radius_pix)));
    const Eigen::Index col_max = std::min<Eigen::Index>(
        prev_sig.cols() - 1, static_cast<Eigen::Index>(std::ceil(center_col + radius_pix)));

    if (row_min > row_max || col_min > col_max) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double diff2_sum = 0.0;
    double prev2_sum = 0.0;
    Eigen::Index n_pix = 0;
    for (Eigen::Index row = row_min; row <= row_max; ++row) {
        for (Eigen::Index col = col_min; col <= col_max; ++col) {
            const double dr = static_cast<double>(row) - center_row;
            const double dc = static_cast<double>(col) - center_col;
            if (dr * dr + dc * dc > radius2) {
                continue;
            }
            const double prev = prev_sig(row, col);
            const double cur = cur_sig(row, col);
            if (!std::isfinite(prev) || !std::isfinite(cur)) {
                continue;
            }
            if (have_prev_wt) {
                const double wt = omb_copy.weight[map_index](row, col);
                if (!std::isfinite(wt) || wt <= 0.0) {
                    continue;
                }
            }
            if (have_cur_wt) {
                const double wt = omb.weight[map_index](row, col);
                if (!std::isfinite(wt) || wt <= 0.0) {
                    continue;
                }
            }
            const double diff = cur - prev;
            diff2_sum += diff * diff;
            prev2_sum += prev * prev;
            ++n_pix;
        }
    }

    if (n_pix < std::max<Eigen::Index>(8, map_fitter.n_params + 1)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double eps = std::numeric_limits<double>::epsilon();
    if (prev2_sum <= eps) {
        if (diff2_sum <= eps) {
            return 0.0;
        }
        return std::numeric_limits<double>::infinity();
    }
    return std::sqrt(diff2_sum / prev2_sum);
}
