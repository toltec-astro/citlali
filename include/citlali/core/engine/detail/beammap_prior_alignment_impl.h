#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_prior_qc_stats.h>

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

std::pair<double, double> Beammap::median_beammap_prior_alignment_translation(
    const std::vector<Beammap::BeammapPriorAlignmentPair> &pairs) const {
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
    return {tula::alg::median(dx_vec), tula::alg::median(dy_vec)};
}

double Beammap::fit_beammap_prior_alignment_rotation(
    const std::vector<Beammap::BeammapPriorAlignmentPair> &pairs,
    const citlali::config::BeammapPriorsConfig &priors_config,
    const std::string &label,
    double tx,
    double ty) const {
    double theta = 0.0;
    if (!priors_config.alignment_fit_rotation) {
        return theta;
    }

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
    const double max_theta =
        priors_config.alignment_max_rotation_deg * DEG_TO_RAD;
    if (!std::isfinite(theta) || std::abs(theta) > max_theta) {
        logger->debug(
            "beammap prior alignment {} rejected residual rotation {} deg (limit={} deg)",
            label, theta * RAD_TO_DEG,
            priors_config.alignment_max_rotation_deg);
        theta = 0.0;
    }

    return theta;
}

std::pair<double, double> Beammap::median_beammap_prior_alignment_translation_after_rotation(
    const std::vector<Beammap::BeammapPriorAlignmentPair> &pairs,
    double cos_theta,
    double sin_theta) const {
    std::vector<double> dx_vals;
    std::vector<double> dy_vals;
    dx_vals.reserve(pairs.size());
    dy_vals.reserve(pairs.size());
    for (const auto &pair : pairs) {
        const double x_rot = cos_theta * pair.obs_x - sin_theta * pair.obs_y;
        const double y_rot = sin_theta * pair.obs_x + cos_theta * pair.obs_y;
        dx_vals.push_back(pair.slot_x - x_rot);
        dy_vals.push_back(pair.slot_y - y_rot);
    }
    Eigen::Map<Eigen::VectorXd> dx_vec_final(dx_vals.data(), static_cast<Eigen::Index>(dx_vals.size()));
    Eigen::Map<Eigen::VectorXd> dy_vec_final(dy_vals.data(), static_cast<Eigen::Index>(dy_vals.size()));
    return {tula::alg::median(dx_vec_final), tula::alg::median(dy_vec_final)};
}

double Beammap::beammap_prior_alignment_rms(
    const std::vector<Beammap::BeammapPriorAlignmentPair> &pairs,
    double cos_theta,
    double sin_theta,
    double tx,
    double ty) const {
    double rss = 0.0;
    for (const auto &pair : pairs) {
        const double x_fit = cos_theta * pair.obs_x - sin_theta * pair.obs_y + tx;
        const double y_fit = sin_theta * pair.obs_x + cos_theta * pair.obs_y + ty;
        const double rx = x_fit - pair.slot_x;
        const double ry = y_fit - pair.slot_y;
        rss += rx * rx + ry * ry;
    }
    return std::sqrt(rss / static_cast<double>(pairs.size()));
}

bool Beammap::fit_beammap_prior_alignment(
    const std::vector<Beammap::BeammapPriorAlignmentPair> &pairs,
    const citlali::config::BeammapPriorsConfig &priors_config,
    const std::string &label,
    Beammap::PriorArrayAlignment &alignment) {
    if (pairs.size() < static_cast<std::size_t>(priors_config.alignment_min_matches)) {
        logger->debug("beammap prior alignment skipped {} matches={} min_matches={}",
                      label, pairs.size(), priors_config.alignment_min_matches);
        return false;
    }

    auto [tx, ty] = median_beammap_prior_alignment_translation(pairs);
    const double theta =
        fit_beammap_prior_alignment_rotation(pairs, priors_config, label, tx, ty);
    const double cos_theta = std::cos(theta);
    const double sin_theta = std::sin(theta);
    const auto final_translation =
        median_beammap_prior_alignment_translation_after_rotation(pairs, cos_theta, sin_theta);
    tx = final_translation.first;
    ty = final_translation.second;
    const double rms = beammap_prior_alignment_rms(pairs, cos_theta, sin_theta, tx, ty);
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
}

Beammap::BeammapPriorAlignmentOverlapBox Beammap::beammap_prior_alignment_overlap_box(
    const Beammap::BeammapPriorAlignmentSamples &alignment_samples,
    const citlali::config::BeammapPriorsConfig &priors_config) const {
    const double q_low =
        priors_config.alignment_common_support_quantile;
    const double q_high =
        1.0 - priors_config.alignment_common_support_quantile;
    BeammapPriorAlignmentOverlapBox overlap_box;
    bool overlap_valid = true;

    for (const auto &[array, pairs] : alignment_samples.pairs_by_array) {
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
        const double x_low = beammap_prior_qc_stats::quantile(xs, q_low);
        const double x_high = beammap_prior_qc_stats::quantile(xs, q_high);
        const double y_low = beammap_prior_qc_stats::quantile(ys, q_low);
        const double y_high = beammap_prior_qc_stats::quantile(ys, q_high);
        if (!(std::isfinite(x_low) && std::isfinite(x_high) &&
              std::isfinite(y_low) && std::isfinite(y_high))) {
            overlap_valid = false;
            break;
        }
        overlap_box.x_low = std::max(overlap_box.x_low, x_low);
        overlap_box.x_high = std::min(overlap_box.x_high, x_high);
        overlap_box.y_low = std::max(overlap_box.y_low, y_low);
        overlap_box.y_high = std::min(overlap_box.y_high, y_high);
    }

    overlap_box.valid = overlap_valid &&
                        overlap_box.x_low < overlap_box.x_high &&
                        overlap_box.y_low < overlap_box.y_high;
    return overlap_box;
}

std::vector<Beammap::BeammapPriorAlignmentPair>
Beammap::filter_beammap_prior_alignment_pairs_to_overlap_box(
    const Beammap::BeammapPriorAlignmentSamples &alignment_samples,
    const Beammap::BeammapPriorAlignmentOverlapBox &overlap_box) const {
    std::vector<BeammapPriorAlignmentPair> filtered_pairs;
    filtered_pairs.reserve(alignment_samples.all_pairs.size());
    for (const auto &pair : alignment_samples.all_pairs) {
        if (pair.slot_x >= overlap_box.x_low && pair.slot_x <= overlap_box.x_high &&
            pair.slot_y >= overlap_box.y_low && pair.slot_y <= overlap_box.y_high) {
            filtered_pairs.push_back(pair);
        }
    }
    return filtered_pairs;
}

std::vector<Beammap::BeammapPriorAlignmentPair>
Beammap::select_common_beammap_prior_alignment_pairs(
    const Beammap::BeammapPriorAlignmentSamples &alignment_samples,
    const citlali::config::BeammapPriorsConfig &priors_config) {
    auto common_pairs = alignment_samples.all_pairs;
    if (citlali::config::uses_overlap_box_prior_alignment_support(
            priors_config) &&
        alignment_samples.pairs_by_array.size() >= 2) {
        const auto overlap_box =
            beammap_prior_alignment_overlap_box(alignment_samples, priors_config);
        if (overlap_box.valid) {
            auto filtered_pairs =
                filter_beammap_prior_alignment_pairs_to_overlap_box(
                    alignment_samples, overlap_box);
            if (filtered_pairs.size() >=
                static_cast<std::size_t>(
                    priors_config.alignment_min_matches)) {
                common_pairs.swap(filtered_pairs);
            }
            logger->info(
                "beammap prior common alignment overlap_box (iter {}): q={} x=[{}, {}] y=[{}, {}] kept={}/{}",
                current_iter,
                priors_config.alignment_common_support_quantile,
                overlap_box.x_low, overlap_box.x_high, overlap_box.y_low, overlap_box.y_high,
                common_pairs.size(), alignment_samples.all_pairs.size());
        }
        else {
            logger->debug(
                "beammap prior common alignment overlap_box skipped: invalid overlap x=[{}, {}] y=[{}, {}]",
                overlap_box.x_low, overlap_box.x_high, overlap_box.y_low, overlap_box.y_high);
        }
    }

    return common_pairs;
}

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
