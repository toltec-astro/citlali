#pragma once

// Included by fits_image_metadata.h inside namespace citlali::pipeline.

inline double map_median_error_or_zero(double median_error_variance,
                                       bool is_beammap) {
    if (is_beammap) {
        return 0.0;
    }
    if (std::isfinite(median_error_variance) &&
        median_error_variance > std::numeric_limits<double>::epsilon()) {
        return std::sqrt(median_error_variance);
    }
    return 0.0;
}

inline bool has_negative_map_median_error(double median_error_variance,
                                          bool is_beammap) {
    return !is_beammap && std::isfinite(median_error_variance) &&
           median_error_variance < 0.0;
}

template <class Logger>
double map_median_error_or_zero_logged(double median_error_variance,
                                       bool is_beammap,
                                       const std::string &map_name,
                                       const std::string &filepath,
                                       const Logger &logger) {
    if (has_negative_map_median_error(median_error_variance, is_beammap)) {
        logger->warn("negative median_err for map {} in {}; using 0",
                     map_name, filepath);
    }
    return map_median_error_or_zero(median_error_variance, is_beammap);
}

template <class MedianRms>
double map_median_rms_or_zero(const MedianRms &median_rms, Eigen::Index i) {
    if (i < static_cast<Eigen::Index>(median_rms.size()) &&
        std::isfinite(median_rms(i))) {
        return median_rms(i);
    }
    return 0.0;
}

template <class MedianRms>
bool has_nonfinite_map_median_rms(const MedianRms &median_rms,
                                  Eigen::Index i) {
    return i < static_cast<Eigen::Index>(median_rms.size()) &&
           !std::isfinite(median_rms(i));
}

template <class MedianRms, class Logger>
double map_median_rms_or_zero_logged(const MedianRms &median_rms,
                                     Eigen::Index i,
                                     const std::string &map_name,
                                     const std::string &filepath,
                                     const Logger &logger) {
    if (has_nonfinite_map_median_rms(median_rms, i)) {
        logger->warn("non-finite median_rms for map {} in {}; using 0",
                     map_name, filepath);
    }
    return map_median_rms_or_zero(median_rms, i);
}

inline bool has_nonfinite_weight_threshold(double weight_threshold) {
    return !std::isfinite(weight_threshold);
}

inline double weight_threshold_or_zero(double weight_threshold) {
    return has_nonfinite_weight_threshold(weight_threshold) ? 0.0
                                                            : weight_threshold;
}

template <class Logger>
double weight_threshold_or_zero_logged(double weight_threshold,
                                       const std::string &map_name,
                                       const std::string &filepath,
                                       const Logger &logger) {
    if (has_nonfinite_weight_threshold(weight_threshold)) {
        logger->warn("non-finite weight threshold for map {} in {}; using 0",
                     map_name, filepath);
    }
    return weight_threshold_or_zero(weight_threshold);
}

template <class Matrix>
Eigen::MatrixXd coverage_mask_from_weight(const Matrix &weight,
                                          double weight_threshold) {
    Eigen::MatrixXd ones;
    Eigen::MatrixXd zeros;
    ones.setOnes(weight.rows(), weight.cols());
    zeros.setZero(weight.rows(), weight.cols());
    return (weight.array() < weight_threshold).select(zeros, ones);
}

template <class Matrix>
Eigen::MatrixXd standardized_signal_from_weight(const Matrix &signal,
                                                const Matrix &weight) {
    return signal.array() * weight.array().sqrt();
}

template <class NoiseList, class FitsIo>
bool should_write_noise_maps(const NoiseList &noise,
                             const FitsIo &noise_fits_io) {
    return !noise.empty() && !noise_fits_io->empty();
}

template <class SignalList, class WeightList, class Logger>
void require_primary_map_image_shapes(
    const SignalList &signal, const WeightList &weight, Eigen::Index map_i,
    Eigen::Index rows, Eigen::Index cols, const Logger &logger) {
    if (!has_map_image_slot(signal, map_i, rows, cols) ||
        !has_map_image_slot(weight, map_i, rows, cols)) {
        fail_required_output(logger, fmt::format(
            "write_maps primary image shape mismatch: map_i={} "
            "expected=({},{})",
            static_cast<long long>(map_i), static_cast<long long>(rows),
            static_cast<long long>(cols)));
    }
}

template <class Wcs, class Logger>
void require_map_wcs_cardinality(const Wcs &wcs,
                                 std::size_t minimum_axes,
                                 const Logger &logger) {
    const auto axes = wcs.ctype.size();
    if (axes < minimum_axes || wcs.cunit.size() != axes ||
        wcs.crval.size() != axes || wcs.cdelt.size() != axes ||
        wcs.crpix.size() != axes || wcs.naxis.size() < 2) {
        fail_required_output(logger, fmt::format(
            "write_maps inconsistent WCS cardinality: "
            "ctype={} cunit={} crval={} cdelt={} crpix={} naxis={} "
            "minimum_axes={}",
            axes, wcs.cunit.size(), wcs.crval.size(), wcs.cdelt.size(),
            wcs.crpix.size(), wcs.naxis.size(), minimum_axes));
    }
}

template <class FitsIo>
bool has_noise_fits_slot(const FitsIo &noise_fits_io,
                         Eigen::Index map_index) {
    return map_index >= 0 &&
           map_index < static_cast<Eigen::Index>(noise_fits_io->size());
}

template <class NoiseList>
bool has_noise_map_slot(const NoiseList &noise, Eigen::Index i) {
    return i >= 0 && i < static_cast<Eigen::Index>(noise.size());
}

template <class NoiseList, class FitsIo, class Logger>
void require_noise_map_write_slots(
    const NoiseList &noise, const FitsIo &noise_fits_io,
    Eigen::Index map_index, Eigen::Index map_i, const Logger &logger) {
    if (!has_noise_fits_slot(noise_fits_io, map_index)) {
        fail_required_output(logger, fmt::format(
            "write_maps noise file index out of range: map_index={} noise_fits_io_size={} map_i={}",
            static_cast<long long>(map_index),
            static_cast<long long>(noise_fits_io->size()),
            static_cast<long long>(map_i)));
    }
    if (!has_noise_map_slot(noise, map_i)) {
        fail_required_output(logger, fmt::format(
            "write_maps noise map index out of range: i={} noise_size={}",
            static_cast<long long>(map_i),
            static_cast<long long>(noise.size())));
    }
}

template <class NoiseList, class Logger>
void require_noise_map_tensor_shape(
    const NoiseList &noise, Eigen::Index map_i, Eigen::Index rows,
    Eigen::Index cols, Eigen::Index n_noise, const Logger &logger) {
    if (n_noise < 0) {
        fail_required_output(logger, fmt::format(
            "write_maps invalid noise realization count: n_noise={} map_i={}",
            static_cast<long long>(n_noise),
            static_cast<long long>(map_i)));
    }
    if (!has_noise_map_slot(noise, map_i)) {
        fail_required_output(logger, fmt::format(
            "write_maps noise map index out of range: i={} noise_size={}",
            static_cast<long long>(map_i),
            static_cast<long long>(noise.size())));
    }
    const auto &tensor = noise[map_i];
    if (tensor.dimension(0) != rows || tensor.dimension(1) != cols ||
        tensor.dimension(2) != n_noise) {
        fail_required_output(logger, fmt::format(
            "write_maps noise tensor shape mismatch: map_i={} "
            "actual=({},{},{}) expected=({},{},{})",
            static_cast<long long>(map_i),
            static_cast<long long>(tensor.dimension(0)),
            static_cast<long long>(tensor.dimension(1)),
            static_cast<long long>(tensor.dimension(2)),
            static_cast<long long>(rows), static_cast<long long>(cols),
            static_cast<long long>(n_noise)));
    }
}

template <class NoiseList, class FitsIo>
std::string noise_file_path_or_na(const NoiseList &noise,
                                  const FitsIo &noise_fits_io,
                                  Eigen::Index map_index) {
    if (!noise.empty() && has_noise_fits_slot(noise_fits_io, map_index)) {
        return noise_fits_io->at(map_index).filepath;
    }
    return std::string("N/A");
}

template <class NoiseList, class FitsIo>
std::string map_write_error_message(
    const std::string &map_name, Eigen::Index map_i,
    const std::string &filepath, const NoiseList &noise,
    const FitsIo &noise_fits_io, Eigen::Index map_index,
    const std::string &message) {
    return fmt::format(
        "failed to write map '{}' (map_i={} file={} noise_file={}): {}",
        map_name, static_cast<long long>(map_i), filepath,
        noise_file_path_or_na(noise, noise_fits_io, map_index), message);
}
