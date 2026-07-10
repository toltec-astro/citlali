#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

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
