#pragma once

// Beammap APT derotation implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/timestream_alignment_state.h>

void Beammap::record_beammap_reference_metadata(
    double ref_det_x_t, double ref_det_y_t) {
    // add reference detector to APT meta data
    calib.apt_meta["reference_x_t"] = ref_det_x_t;
    calib.apt_meta["reference_y_t"] = ref_det_y_t;
}

void Beammap::preserve_beammap_raw_detector_offsets() {
    // raw (not derotated or reference detector subtracted) detector x and y values
    calib.apt["x_t_raw"] = calib.apt["x_t"];
    calib.apt["y_t_raw"] = calib.apt["y_t"];
}

void Beammap::populate_beammap_derotation_elevation() {
    // per-detector derotation elevation for altaz beammaps
    calib.apt["derot_elev"].setConstant(
        citlali::pipeline::governing_compatibility_mean(
            telescope.tel_data["TelElAct"], alignment));
    if (citlali::config::is_altaz_map_pixel_axes(telescope.pixel_axes) &&
        citlali::config::is_detector_map_grouping(
            citlali::pipeline::mapmaking_config(*this).grouping) &&
        !ptcs.empty()) {
        Eigen::MatrixXd elev_best(omb.n_rows, omb.n_cols);
        Eigen::MatrixXd dist2_best(omb.n_rows, omb.n_cols);
        elev_best.setConstant(std::numeric_limits<double>::quiet_NaN());
        dist2_best.setConstant(std::numeric_limits<double>::infinity());

        for (const auto &ptc : ptcs) {
            const auto &alt = ptc.tel_data.data.at("alt_phys");
            const auto &az = ptc.tel_data.data.at("az_phys");
            const auto &el = ptc.tel_data.data.at("TelElAct");
            for (Eigen::Index k = 0; k < alt.size(); ++k) {
                double row =
                    alt(k) / omb.pixel_size_rad + (omb.n_rows - 1) / 2.0;
                double col =
                    az(k) / omb.pixel_size_rad + (omb.n_cols - 1) / 2.0;
                Eigen::Index ir = static_cast<Eigen::Index>(std::llround(row));
                Eigen::Index ic = static_cast<Eigen::Index>(std::llround(col));
                if ((ir >= 0) && (ir < omb.n_rows) && (ic >= 0) &&
                    (ic < omb.n_cols)) {
                    double lat_center =
                        (static_cast<double>(ir) - (omb.n_rows - 1) / 2.0) *
                        omb.pixel_size_rad;
                    double lon_center =
                        (static_cast<double>(ic) - (omb.n_cols - 1) / 2.0) *
                        omb.pixel_size_rad;
                    double dlat = alt(k) - lat_center;
                    double dlon = az(k) - lon_center;
                    double dist2 = dlat * dlat + dlon * dlon;
                    if (dist2 < dist2_best(ir, ic)) {
                        dist2_best(ir, ic) = dist2;
                        elev_best(ir, ic) = el(k);
                    }
                }
            }
        }

        for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
            double row =
                (calib.apt["y_t_raw"](i) * ASEC_TO_RAD) /
                    omb.pixel_size_rad +
                (omb.n_rows - 1) / 2.0;
            double col =
                (calib.apt["x_t_raw"](i) * ASEC_TO_RAD) /
                    omb.pixel_size_rad +
                (omb.n_cols - 1) / 2.0;
            Eigen::Index ir = static_cast<Eigen::Index>(std::llround(row));
            Eigen::Index ic = static_cast<Eigen::Index>(std::llround(col));
            if ((ir >= 0) && (ir < omb.n_rows) && (ic >= 0) &&
                (ic < omb.n_cols)) {
                double elev = elev_best(ir, ic);
                if (std::isfinite(elev)) {
                    calib.apt["derot_elev"](i) = elev;
                }
            }
        }
    }
}

void Beammap::apply_beammap_reference_offsets(
    double ref_det_x_t, double ref_det_y_t) {
    // align to reference detector if specified and subtract its position from x and y
    calib.apt["x_t"] = calib.apt["x_t"].array() - ref_det_x_t;
    calib.apt["y_t"] = calib.apt["y_t"].array() - ref_det_y_t;
}

void Beammap::apply_beammap_derotation(bool derotate) {
    // derotated detector x and y values
    calib.apt["x_t_derot"] = calib.apt["x_t"];
    calib.apt["y_t_derot"] = calib.apt["y_t"];

    // tolerate telescope streams that provide elevation in degrees.
    Eigen::VectorXd derot_elev_rad = calib.apt["derot_elev"];
    const double max_abs_elev = derot_elev_rad.array().abs().maxCoeff();
    if (std::isfinite(max_abs_elev) && max_abs_elev > 2.0 * pi + 0.1) {
        logger->warn("derot_elev appears to be in degrees (max |elev|={:.4g}); converting to radians",
                     max_abs_elev);
        derot_elev_rad *= DEG_TO_RAD;
    }

    // calculate derotated positions
    Eigen::VectorXd rot_az_off =
        cos(-derot_elev_rad.array()) * calib.apt["x_t_derot"].array() -
        sin(-derot_elev_rad.array()) * calib.apt["y_t_derot"].array();
    Eigen::VectorXd rot_alt_off =
        sin(-derot_elev_rad.array()) * calib.apt["x_t_derot"].array() +
        cos(-derot_elev_rad.array()) * calib.apt["y_t_derot"].array();

    // overwrite x_t and y_t
    calib.apt["x_t_derot"] = -rot_az_off;
    calib.apt["y_t_derot"] = -rot_alt_off;

    if (derotate) {
        logger->info("derotating apt");
        // if derotation requested set default positions to derotated positions
        calib.apt["x_t"] = calib.apt["x_t_derot"];
        calib.apt["y_t"] = calib.apt["y_t_derot"];
    }
}
