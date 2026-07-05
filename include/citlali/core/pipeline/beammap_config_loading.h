#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

namespace citlali::pipeline {

template <class Config, class InvalidKeys>
std::vector<double> beammap_fixed_double_vector(
    Config &config, const std::vector<std::string> &path,
    std::size_t expected_size, InvalidKeys &invalid_keys) {
    std::vector<double> values;
    if (path.size() == 2) {
        values = config.template get_typed<std::vector<double>>(
            std::make_tuple(path[0], path[1]));
    }
    else {
        values = config.template get_typed<std::vector<double>>(
            std::make_tuple(path[0], path[1], path[2]));
    }
    if (values.size() != expected_size) {
        invalid_keys.push_back(path);
        values.resize(expected_size, 0.0);
    }
    return values;
}

template <class ArrayNameMap, class ValueMap>
void assign_beammap_array_flag_limits(
    const ArrayNameMap &array_name_map,
    const std::vector<double> &lower_fwhm_arcsec_vec,
    const std::vector<double> &upper_fwhm_arcsec_vec,
    const std::vector<double> &lower_sig2noise_vec,
    const std::vector<double> &upper_sig2noise_vec,
    const std::vector<double> &max_dist_arcsec_vec,
    const std::vector<double> &network_robust_z_vec,
    ValueMap &lower_fwhm_arcsec,
    ValueMap &upper_fwhm_arcsec,
    ValueMap &lower_sig2noise,
    ValueMap &upper_sig2noise,
    ValueMap &max_dist_arcsec,
    ValueMap &network_robust_z) {
    std::size_t i = 0;
    for (auto const& [arr_index, arr_name] : array_name_map) {
        (void)arr_index;
        lower_fwhm_arcsec[arr_name] = lower_fwhm_arcsec_vec[i];
        upper_fwhm_arcsec[arr_name] = upper_fwhm_arcsec_vec[i];
        lower_sig2noise[arr_name] = lower_sig2noise_vec[i];
        upper_sig2noise[arr_name] = upper_sig2noise_vec[i];
        max_dist_arcsec[arr_name] = max_dist_arcsec_vec[i];
        network_robust_z[arr_name] = network_robust_z_vec[i];
        ++i;
    }
}

}  // namespace citlali::pipeline
