#pragma once

// Included by mapdiag_stats_core.h inside namespace citlali::pipeline.

inline double mapdiag_weight_threshold_or_zero(double weight_threshold) {
    if (std::isfinite(weight_threshold) && weight_threshold >= 0.0) {
        return weight_threshold;
    }
    return 0.0;
}

template <class MapBuffer>
double mapdiag_weight_threshold_for_map(const MapBuffer &mb,
                                        Eigen::Index map_index) {
    const auto cov_region = mb->calc_cov_region(map_index);
    return mapdiag_weight_threshold_or_zero(std::get<0>(cov_region));
}

inline Eigen::ArrayXXd mapdiag_valid_weight_mask(
    const Eigen::ArrayXXd &weight) {
    return (weight > 0.0).template cast<double>();
}

inline Eigen::ArrayXXd mapdiag_core_weight_mask(
    const Eigen::ArrayXXd &weight, double weight_threshold) {
    return ((weight >= weight_threshold) && (weight > 0.0))
        .template cast<double>();
}

inline Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>
mapdiag_positive_mask(const Eigen::ArrayXXd &mask) {
    return mask > 0.0;
}

inline Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>
mapdiag_valid_core_noise_mask(const Eigen::ArrayXXd &core_mask) {
    return mapdiag_positive_mask(core_mask);
}

template <class Mask>
double mapdiag_mask_count_as_double(const Mask &mask) {
    return static_cast<double>(mask.count());
}

template <class Mask>
double mapdiag_valid_core_noise_count(const Mask &valid_core_mask) {
    return mapdiag_mask_count_as_double(valid_core_mask);
}

inline bool mapdiag_has_positive_count(double count) {
    return count > 0.0;
}

template <class Mask>
int mapdiag_mask_sum_as_int(const Mask &mask) {
    return static_cast<int>(mask.sum());
}

template <class Values, class Mask>
double mapdiag_weighted_mask_sum(const Values &values, const Mask &mask) {
    return (values * mask).sum();
}

template <class Values, class ValidMask, class CoreMask>
MapdiagWeightStats mapdiag_weight_stats(const Values &weight,
                                        const ValidMask &valid_mask,
                                        const CoreMask &core_mask) {
    return {mapdiag_mask_sum_as_int(valid_mask),
            mapdiag_mask_sum_as_int(core_mask),
            mapdiag_weighted_mask_sum(weight, valid_mask),
            mapdiag_weighted_mask_sum(weight, core_mask)};
}

