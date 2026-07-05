#pragma once

// Included by mapdiag_stats_core.h inside namespace citlali::pipeline.

inline bool mapdiag_has_matrix_samples(const Eigen::MatrixXd &matrix) {
    return matrix.size() > 0;
}

inline bool mapdiag_has_signal_weight_samples(
    const Eigen::MatrixXd &signal, const Eigen::MatrixXd &weight) {
    return mapdiag_has_matrix_samples(signal) &&
           mapdiag_has_matrix_samples(weight);
}

template <class Matrix>
auto mapdiag_matrix_value(const Matrix &matrix, Eigen::Index row,
                          Eigen::Index col) {
    return matrix(row, col);
}

template <class Matrix>
double mapdiag_matrix_double_value(const Matrix &matrix, Eigen::Index row,
                                   Eigen::Index col) {
    return static_cast<double>(mapdiag_matrix_value(matrix, row, col));
}

inline Eigen::MatrixXd mapdiag_sig2noise_image(
    const Eigen::MatrixXd &signal, const Eigen::MatrixXd &weight) {
    return signal.array() * weight.array().max(0.0).sqrt();
}

inline double mapdiag_peak_signal_or_fill(const Eigen::MatrixXd &signal,
                                          double fill_value) {
    return mapdiag_has_matrix_samples(signal) ? signal.maxCoeff() : fill_value;
}

template <class SignalList>
void assign_mapdiag_peak_signal_or_fill(
    std::size_t idx, const SignalList &signals, Eigen::Index map_index,
    double fill_value, std::vector<double> &peak_signal) {
    peak_signal[idx] = mapdiag_peak_signal_or_fill(
        signals[map_index], fill_value);
}

inline double mapdiag_core_peak_abs_or_fill(const Eigen::MatrixXd &sig2noise,
                                            const Eigen::ArrayXXd &core_mask,
                                            int n_core_pixels,
                                            double fill_value) {
    if (n_core_pixels <= 0) {
        return fill_value;
    }
    const Eigen::MatrixXd core_sig2noise =
        (sig2noise.cwiseAbs().array() * core_mask).matrix();
    return core_sig2noise.maxCoeff();
}

inline MapdiagPeakStats mapdiag_peak_stats(
    const Eigen::MatrixXd &sig2noise, const Eigen::ArrayXXd &core_mask,
    int n_core_pixels, double fill_value) {
    Eigen::Index r_peak = 0;
    Eigen::Index c_peak = 0;
    const double peak_abs_sig2noise =
        sig2noise.cwiseAbs().maxCoeff(&r_peak, &c_peak);
    return {peak_abs_sig2noise,
            static_cast<int>(r_peak),
            static_cast<int>(c_peak),
            mapdiag_core_peak_abs_or_fill(
                sig2noise, core_mask, n_core_pixels, fill_value)};
}

