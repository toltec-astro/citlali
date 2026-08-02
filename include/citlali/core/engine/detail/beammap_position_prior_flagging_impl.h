#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/pipeline/timestream_alignment_state.h>

Beammap::BeammapArrayPositionMedians Beammap::beammap_array_position_medians() {
    BeammapArrayPositionMedians medians;

    logger->debug("calculating array median positions");
    for (Eigen::Index i=0; i<calib.n_arrays; ++i) {
        Eigen::Index array = calib.arrays(i);
        std::string array_name = toltec_io.array_name_map[array];

        auto array_x_t = calib.apt["x_t"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                                     std::get<1>(calib.array_limits[array])-1));
        auto array_y_t = calib.apt["y_t"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                                     std::get<1>(calib.array_limits[array])-1));
        Eigen::Index n_good_det =
            (calib.apt["flag"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                          std::get<1>(calib.array_limits[array])-1)).array()==0).count();

        Eigen::VectorXd x_t, y_t;

        if (n_good_det>0) {
            x_t.resize(n_good_det);
            y_t.resize(n_good_det);

            Eigen::Index j = std::get<0>(calib.array_limits[array]);
            Eigen::Index k = 0;
            for (Eigen::Index m=0; m<array_x_t.size(); m++) {
                if (calib.apt["flag"](j)==0) {
                    x_t(k) = array_x_t(m);
                    y_t(k) = array_y_t(m);
                    k++;
                }
                j++;
            }
            medians.x_t[array_name] = tula::alg::median(x_t);
            medians.y_t[array_name] = tula::alg::median(y_t);
        }
        else {
            medians.x_t[array_name] = tula::alg::median(array_x_t);
            medians.y_t[array_name] = tula::alg::median(array_y_t);
        }
    }

    return medians;
}

void Beammap::flag_beammap_position_outliers(
    const citlali::pipeline::BeammapArrayFlaggingLimits &flag_limits,
    Beammap::BeammapArrayPositionMedians &array_position_medians,
    const std::string &runtime_parallel_policy,
    std::atomic<int> &n_flagged_dets) {
    logger->debug("flagging detector positions");
    grppi::map(tula::grppi_utils::dyn_ex(runtime_parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        auto array_index = calib.apt["array"](i);
        std::string array_name = toltec_io.array_name_map[array_index];

        double dist = sqrt(pow(calib.apt["x_t"](i) - array_position_medians.x_t[array_name],2) +
                           pow(calib.apt["y_t"](i) - array_position_medians.y_t[array_name],2));

        if (dist > flag_limits.max_dist_arcsec.at(array_name) &&
            flag_limits.max_dist_arcsec.at(array_name) > 0) {
            mark_beammap_detector_flagged(i, AptFlags::Position, n_flagged_dets);
        }

        return 0;
    });
}

Beammap::BeammapPriorDistanceFrame Beammap::beammap_prior_distance_frame() {
    double prior_derot_elev_rad =
        citlali::pipeline::governing_compatibility_mean(
            telescope.tel_data["TelElAct"], alignment);
    if (!std::isfinite(prior_derot_elev_rad)) {
        prior_derot_elev_rad = 0.0;
    }
    if (std::abs(prior_derot_elev_rad) > pi) {
        prior_derot_elev_rad *= DEG_TO_RAD;
    }

    BeammapPriorDistanceFrame frame;
    frame.apply_derot =
        beammap_soft_priors_are_derotated &&
        citlali::config::is_altaz_map_pixel_axes(telescope.pixel_axes);
    frame.cos_rot = std::cos(-prior_derot_elev_rad);
    frame.sin_rot = std::sin(-prior_derot_elev_rad);
    return frame;
}

bool Beammap::beammap_soft_prior_slot_valid(const SoftPriorSlot &slot) const {
    return std::isfinite(slot.x_arcsec) && std::isfinite(slot.y_arcsec) &&
           std::isfinite(slot.sx_arcsec) && std::isfinite(slot.sy_arcsec) &&
           slot.sx_arcsec > 0.0 && slot.sy_arcsec > 0.0;
}

double Beammap::beammap_detector_prior_distance2(
    Eigen::Index detector_index,
    const Beammap::BeammapArrayPositionMedians &array_position_medians,
    const Beammap::BeammapPriorDistanceFrame &frame) {
    const int array_index = static_cast<int>(std::lround(calib.apt["array"](detector_index)));
    const int nw_index = static_cast<int>(std::lround(calib.apt["nw"](detector_index)));
    std::string array_name = toltec_io.array_name_map[array_index];

    auto slots_it = beammap_soft_prior_slots.find({array_index, nw_index});
    if (slots_it == beammap_soft_prior_slots.end() || slots_it->second.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double x_arcsec = calib.apt["x_t"](detector_index);
    double y_arcsec = calib.apt["y_t"](detector_index);
    if (!std::isfinite(x_arcsec) || !std::isfinite(y_arcsec)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (beammap_soft_priors_are_centered) {
        auto x_it = array_position_medians.x_t.find(array_name);
        auto y_it = array_position_medians.y_t.find(array_name);
        const double median_x = (x_it != array_position_medians.x_t.end()) ? x_it->second : 0.0;
        const double median_y = (y_it != array_position_medians.y_t.end()) ? y_it->second : 0.0;
        x_arcsec -= median_x;
        y_arcsec -= median_y;
    }

    if (frame.apply_derot) {
        const double rot_az_off = frame.cos_rot * x_arcsec - frame.sin_rot * y_arcsec;
        const double rot_alt_off = frame.sin_rot * x_arcsec + frame.cos_rot * y_arcsec;
        x_arcsec = -rot_az_off;
        y_arcsec = -rot_alt_off;
    }

    double min_d2 = std::numeric_limits<double>::infinity();
    for (const auto &slot : slots_it->second) {
        if (!beammap_soft_prior_slot_valid(slot)) {
            continue;
        }
        const double dx = (x_arcsec - slot.x_arcsec) / slot.sx_arcsec;
        const double dy = (y_arcsec - slot.y_arcsec) / slot.sy_arcsec;
        const double d2 = dx * dx + dy * dy;
        if (std::isfinite(d2) && d2 < min_d2) {
            min_d2 = d2;
        }
    }
    return min_d2;
}

void Beammap::flag_beammap_prior_distance_detector(
    Eigen::Index detector_index,
    double max_prior_d2,
    const Beammap::BeammapArrayPositionMedians &array_position_medians,
    const Beammap::BeammapPriorDistanceFrame &frame,
    std::atomic<int> &n_prior_dist_hits,
    std::atomic<int> &n_flagged_dets) {
    const double min_d2 =
        beammap_detector_prior_distance2(detector_index, array_position_medians, frame);
    if (!std::isfinite(min_d2) || min_d2 <= max_prior_d2) {
        return;
    }

    n_prior_dist_hits++;
    mark_beammap_detector_flagged(detector_index, AptFlags::PriorDist, n_flagged_dets);
}

void Beammap::flag_beammap_prior_distance_outliers(
    double max_prior_d2,
    const Beammap::BeammapArrayPositionMedians &array_position_medians,
    const std::string &runtime_parallel_policy,
    std::atomic<int> &n_flagged_dets) {
    const bool prior_dist_flag_enabled =
        max_prior_d2 > 0.0 && beammap_soft_priors_loaded &&
        !beammap_soft_prior_slots.empty();
    if (max_prior_d2 <= 0.0) {
        return;
    }
    if (!prior_dist_flag_enabled) {
        logger->warn(
            "beammap.flagging.max_prior_d2={} requested but soft priors are unavailable; skipping prior-distance flagging",
            max_prior_d2);
        return;
    }

    const auto frame = beammap_prior_distance_frame();
    std::atomic<int> n_prior_dist_hits{0};

    logger->debug("flagging detector prior distances");
    grppi::map(tula::grppi_utils::dyn_ex(runtime_parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        flag_beammap_prior_distance_detector(
            i, max_prior_d2, array_position_medians, frame,
            n_prior_dist_hits, n_flagged_dets);
        return 0;
    });

    logger->info("beammap prior-distance flagging: {} detectors exceeded max_prior_d2={}",
                 n_prior_dist_hits.load(), max_prior_d2);
}
