#pragma once

// Included by mapdiag_stats_outliers.h inside namespace citlali::pipeline.

inline bool mapdiag_has_valid_contributor(int uid, int fill_int,
                                          double contribution_signal) {
    return uid != fill_int && std::isfinite(contribution_signal);
}

inline bool mapdiag_has_full_leave_one_out_inputs(
    double total_signal, double total_weight, double contribution_weight,
    double contribution_variance_weight, double total_variance_weight,
    double remaining_weight) {
    return std::isfinite(total_signal) && std::isfinite(total_weight) &&
           std::isfinite(contribution_weight) &&
           std::isfinite(contribution_variance_weight) &&
           std::isfinite(total_variance_weight) &&
           contribution_weight >= 0.0 && contribution_variance_weight >= 0.0 &&
           remaining_weight > std::numeric_limits<double>::epsilon() &&
           total_variance_weight > contribution_variance_weight;
}

inline double mapdiag_remaining_contribution_weight(double total_weight,
                                                    double contribution_weight) {
    return total_weight - contribution_weight;
}

inline double mapdiag_full_leave_one_out_value(double total_signal,
                                               double contribution_signal,
                                               double remaining_weight) {
    return (total_signal - contribution_signal) / remaining_weight;
}

inline void mapdiag_assign_leave_one_out_z(double value, double weight,
                                           double leave_one_out_value,
                                           double &leave_one_out_z) {
    const double residual = value - leave_one_out_value;
    if (std::isfinite(residual) && std::isfinite(weight) && weight > 0.0) {
        leave_one_out_z = residual * std::sqrt(weight);
    }
}

inline bool mapdiag_has_fallback_leave_one_out_inputs(
    double weight, double contribution_weight) {
    return std::isfinite(contribution_weight) && contribution_weight >= 0.0 &&
           weight > contribution_weight &&
           (weight - contribution_weight) >
               std::numeric_limits<double>::epsilon();
}

inline double mapdiag_raw_weighted_signal(double value, double weight) {
    return value * weight;
}

inline double mapdiag_fallback_leave_one_out_value(
    double raw_weighted_signal, double contribution_signal, double weight,
    double contribution_weight) {
    return (raw_weighted_signal - contribution_signal) /
           (weight - contribution_weight);
}

template <class NoiseList>
bool mapdiag_has_noise_realizations(
    const NoiseList &noise, Eigen::Index i, Eigen::Index n_noise) {
    return !noise.empty() && i >= 0 &&
           i < static_cast<Eigen::Index>(noise.size()) && n_noise > 0;
}

inline Eigen::Index mapdiag_noise_realization_size(Eigen::Index n_rows,
                                                   Eigen::Index n_cols) {
    return n_rows * n_cols;
}

inline Eigen::Index mapdiag_noise_realization_offset(
    Eigen::Index realization_index, Eigen::Index n_rows, Eigen::Index n_cols) {
    return realization_index * mapdiag_noise_realization_size(n_rows, n_cols);
}

inline MapdiagNoiseMatrix mapdiag_noise_matrix(
    double *noise_data, Eigen::Index realization_index, Eigen::Index n_rows,
    Eigen::Index n_cols) {
    return MapdiagNoiseMatrix(
        noise_data + mapdiag_noise_realization_offset(
                         realization_index, n_rows, n_cols),
        n_rows, n_cols);
}

template <class MapBuffer>
MapdiagNoiseMatrix mapdiag_noise_matrix(
    const MapBuffer &mb, Eigen::Index map_index,
    Eigen::Index realization_index) {
    return mapdiag_noise_matrix(
        mb->noise[map_index].data(), realization_index,
        mapdiag_n_rows(mb), mapdiag_n_cols(mb));
}

