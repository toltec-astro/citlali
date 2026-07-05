#pragma once

// Included by stats_netcdf.h inside namespace citlali::pipeline.

inline std::string stats_raw_directory(const std::string &obsnum_dir_name) {
    return obsnum_dir_name + "raw/";
}

inline bool stats_has_tod_output_subdir(
    const std::string &tod_output_subdir_name) {
    return tod_output_subdir_name != "null";
}

inline std::string stats_tod_output_subdir_path(
    const std::string &stats_dir,
    const std::string &tod_output_subdir_name) {
    return stats_dir + tod_output_subdir_name;
}

inline std::string stats_directory_from_subdir(
    const std::string &stats_subdir_path) {
    return stats_subdir_path + "/";
}

inline std::string stats_netcdf_filename(const std::string &stats_filename) {
    return stats_filename + ".nc";
}

template <auto DataType, auto ProductType, auto FilterType, class ToltecIo>
std::string stats_output_netcdf_filename(
    ToltecIo &toltec_io, const std::string &stats_dir,
    const std::string &reduction_type, const std::string &obsnum,
    bool simulated_observation) {
    const auto filename =
        toltec_io.template create_filename<DataType, ProductType, FilterType>(
            stats_dir, reduction_type, "", obsnum, simulated_observation);
    return stats_netcdf_filename(filename);
}

inline std::string stats_unit_or_empty(
    const std::map<std::string, std::string> &units,
    const std::string &stat) {
    const auto it = units.find(stat);
    return it == units.end() ? "" : it->second;
}

inline std::map<std::string, std::string>
detector_stats_units(const std::string &signal_unit) {
    return {
        {"rms", signal_unit},
        {"stddev", signal_unit},
        {"median", signal_unit},
        {"flagged_frac", "N/A"},
        {"weights", "1/(" + signal_unit + ")^2"}};
}

inline std::map<std::string, std::string>
group_stats_units(const std::string &signal_unit) {
    return {{"median_weights", "1/(" + signal_unit + ")^2"}};
}

