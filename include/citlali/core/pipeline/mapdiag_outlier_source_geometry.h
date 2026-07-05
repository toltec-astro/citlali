#pragma once

// Included by mapdiag_stats_outliers.h inside namespace citlali::pipeline.

inline double mapdiag_center_pixel_coordinate(Eigen::Index n_pixels) {
    return (static_cast<double>(n_pixels) - 1.0) / 2.0;
}

inline MapdiagSourceDistanceContext mapdiag_source_distance_context(
    Eigen::Index n_rows, Eigen::Index n_cols, double pixel_size_arcsec,
    double fill_value) {
    return {mapdiag_center_pixel_coordinate(n_rows),
            mapdiag_center_pixel_coordinate(n_cols),
            pixel_size_arcsec,
            fill_value};
}

template <class MapBuffer>
MapdiagSourceDistanceContext mapdiag_source_distance_context(
    const MapBuffer &mb, double rad_to_arcsec, double fill_value) {
    return mapdiag_source_distance_context(
        mapdiag_n_rows(mb), mapdiag_n_cols(mb),
        mb->pixel_size_rad * rad_to_arcsec, fill_value);
}

inline double mapdiag_source_distance_arcsec(
    Eigen::Index row, Eigen::Index col, double center_row,
    double center_col, double pixel_size_arcsec, double fill_value) {
    if (!std::isfinite(pixel_size_arcsec) || pixel_size_arcsec <= 0.0) {
        return fill_value;
    }
    const double drow =
        (static_cast<double>(row) - center_row) * pixel_size_arcsec;
    const double dcol =
        (static_cast<double>(col) - center_col) * pixel_size_arcsec;
    return std::hypot(drow, dcol);
}

inline double mapdiag_source_distance_arcsec(
    Eigen::Index row, Eigen::Index col,
    const MapdiagSourceDistanceContext &context) {
    return mapdiag_source_distance_arcsec(
        row, col, context.center_row, context.center_col,
        context.pixel_size_arcsec, context.fill_value);
}

inline bool mapdiag_is_source_protected(double distance_arcsec,
                                        double protect_radius_arcsec) {
    return protect_radius_arcsec > 0.0 && std::isfinite(distance_arcsec) &&
           distance_arcsec <= protect_radius_arcsec;
}

inline void apply_mapdiag_source_protection_mask(
    Eigen::ArrayXXd &mask, const MapdiagSourceDistanceContext &context,
    double protect_radius_arcsec) {
    for (Eigen::Index r = 0; r < mask.rows(); ++r) {
        for (Eigen::Index c = 0; c < mask.cols(); ++c) {
            const double dist_arcsec =
                mapdiag_source_distance_arcsec(r, c, context);
            if (mapdiag_is_source_protected(
                    dist_arcsec, protect_radius_arcsec)) {
                mask(r, c) = 0.0;
            }
        }
    }
}

inline Eigen::ArrayXXd mapdiag_off_source_core_mask(
    const Eigen::ArrayXXd &core_mask,
    const MapdiagSourceDistanceContext &context,
    double protect_radius_arcsec) {
    Eigen::ArrayXXd off_source_mask = core_mask;
    apply_mapdiag_source_protection_mask(
        off_source_mask, context, protect_radius_arcsec);
    return off_source_mask;
}

