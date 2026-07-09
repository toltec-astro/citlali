#pragma once

// Beammap prior frame matching implementation detail.
// Include this only after Beammap has been declared.

#include <cmath>
#include <limits>

double Beammap::get_prior_derot_elev_rad() const {
    double derot_elev_rad = 0.0;
    auto tel_el_it = telescope.tel_data.find("TelElAct");
    if (tel_el_it != telescope.tel_data.end() && tel_el_it->second.size() > 0) {
        derot_elev_rad = tel_el_it->second.mean();
    }
    if (!std::isfinite(derot_elev_rad)) {
        derot_elev_rad = 0.0;
    }
    if (std::abs(derot_elev_rad) > pi) {
        derot_elev_rad *= DEG_TO_RAD;
    }
    return derot_elev_rad;
}

double Beammap::effective_prior_max_d2() const {
    const auto &priors_config = typed_config.beammap.priors;
    return is_beammap_measurement_iter(current_iter)
               ? priors_config.max_d2_after_iter0
               : priors_config.max_d2_iter0;
}

double Beammap::effective_prior_score_lambda() const {
    const auto &priors_config = typed_config.beammap.priors;
    return is_beammap_measurement_iter(current_iter)
               ? priors_config.score_lambda_after_iter0
               : priors_config.score_lambda_iter0;
}

bool Beammap::observed_to_prior_frame(int array, double x_raw_arcsec, double y_raw_arcsec,
                                      double derot_elev_rad, double &x_prior_arcsec,
                                      double &y_prior_arcsec, double *center_x_arcsec,
                                      double *center_y_arcsec,
                                      bool apply_empirical_alignment) const {
    if (!std::isfinite(x_raw_arcsec) || !std::isfinite(y_raw_arcsec)) {
        return false;
    }

    double x = x_raw_arcsec;
    double y = y_raw_arcsec;
    double center_x = std::numeric_limits<double>::quiet_NaN();
    double center_y = std::numeric_limits<double>::quiet_NaN();

    if (beammap_soft_priors_are_centered) {
        auto x_it = beammap_prior_array_center_x_arcsec.find(array);
        auto y_it = beammap_prior_array_center_y_arcsec.find(array);
        if (x_it == beammap_prior_array_center_x_arcsec.end() ||
            y_it == beammap_prior_array_center_y_arcsec.end() ||
            !std::isfinite(x_it->second) || !std::isfinite(y_it->second)) {
            return false;
        }
        center_x = x_it->second;
        center_y = y_it->second;
        x -= center_x;
        y -= center_y;
    }

    if (center_x_arcsec != nullptr) {
        *center_x_arcsec = center_x;
    }
    if (center_y_arcsec != nullptr) {
        *center_y_arcsec = center_y;
    }

    if (beammap_soft_priors_are_derotated &&
        citlali::config::is_altaz_map_pixel_axes(telescope.pixel_axes)) {
        if (!std::isfinite(derot_elev_rad)) {
            derot_elev_rad = 0.0;
        }
        if (std::abs(derot_elev_rad) > pi) {
            derot_elev_rad *= DEG_TO_RAD;
        }
        const double cos_rot = std::cos(-derot_elev_rad);
        const double sin_rot = std::sin(-derot_elev_rad);
        const double rot_az_off = cos_rot * x - sin_rot * y;
        const double rot_alt_off = sin_rot * x + cos_rot * y;
        x = -rot_az_off;
        y = -rot_alt_off;
    }

    if (apply_empirical_alignment) {
        auto align_it = beammap_prior_array_alignment.find(array);
        if (align_it != beammap_prior_array_alignment.end() && align_it->second.valid) {
            const auto &align = align_it->second;
            const double x_rot = align.cos_theta * x - align.sin_theta * y;
            const double y_rot = align.sin_theta * x + align.cos_theta * y;
            x = x_rot + align.dx_arcsec;
            y = y_rot + align.dy_arcsec;
        }
    }

    x_prior_arcsec = x;
    y_prior_arcsec = y;
    return std::isfinite(x_prior_arcsec) && std::isfinite(y_prior_arcsec);
}

bool Beammap::match_prior_slot(int array, int nw, double x_prior_arcsec, double y_prior_arcsec,
                               double &best_d2, int &best_slot, double *slot_x_arcsec,
                               double *slot_y_arcsec, double *slot_sx_arcsec,
                               double *slot_sy_arcsec) const {
    best_d2 = std::numeric_limits<double>::infinity();
    best_slot = -1;
    auto slots_it = beammap_soft_prior_slots.find({array, nw});
    if (slots_it == beammap_soft_prior_slots.end() || slots_it->second.empty() ||
        !std::isfinite(x_prior_arcsec) || !std::isfinite(y_prior_arcsec)) {
        return false;
    }

    for (const auto &slot : slots_it->second) {
        if (!std::isfinite(slot.x_arcsec) || !std::isfinite(slot.y_arcsec) ||
            !std::isfinite(slot.sx_arcsec) || !std::isfinite(slot.sy_arcsec) ||
            slot.sx_arcsec <= 0.0 || slot.sy_arcsec <= 0.0) {
            continue;
        }
        const double dx = (x_prior_arcsec - slot.x_arcsec) / slot.sx_arcsec;
        const double dy = (y_prior_arcsec - slot.y_arcsec) / slot.sy_arcsec;
        const double d2 = dx * dx + dy * dy;
        if (std::isfinite(d2) && d2 < best_d2) {
            best_d2 = d2;
            best_slot = slot.slot_index;
            if (slot_x_arcsec != nullptr) {
                *slot_x_arcsec = slot.x_arcsec;
            }
            if (slot_y_arcsec != nullptr) {
                *slot_y_arcsec = slot.y_arcsec;
            }
            if (slot_sx_arcsec != nullptr) {
                *slot_sx_arcsec = slot.sx_arcsec;
            }
            if (slot_sy_arcsec != nullptr) {
                *slot_sy_arcsec = slot.sy_arcsec;
            }
        }
    }
    return std::isfinite(best_d2);
}
