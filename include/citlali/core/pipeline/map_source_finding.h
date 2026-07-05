#pragma once

#include <cstddef>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <tula/grppi.h>
#include <yaml-cpp/yaml.h>

namespace citlali::pipeline {

struct SourceFitUnitScale {
    double pixel_to_arcsec;
    double source_fwhm_to_arcsec;
};

struct SourceFitUnitConstants {
    double rad_to_arcsec;
    double std_to_fwhm;
    double arcsec_to_rad;
    double rad_to_deg;
    double deg_to_rad;
    double arcsec_to_deg;
};

struct SourceInitialPosition {
    double row;
    double col;
};

inline double source_window_arcsec_to_rad(double source_window_arcsec,
                                          double arcsec_to_rad) {
    return source_window_arcsec * arcsec_to_rad;
}

inline double source_fitting_arcsec_to_pixels(double value_arcsec,
                                              double arcsec_to_rad,
                                              double pixel_size_rad) {
    return arcsec_to_rad * value_arcsec / pixel_size_rad;
}

template <class MapFitter>
void apply_positive_source_fit_limits(MapFitter &map_fitter) {
    if (map_fitter.flux_limits(0) > 0) {
        map_fitter.flux_low = map_fitter.flux_limits(0);
    }
    if (map_fitter.flux_limits(1) > 0) {
        map_fitter.flux_high = map_fitter.flux_limits(1);
    }
    if (map_fitter.fwhm_limits(0) > 0) {
        map_fitter.fwhm_low = map_fitter.fwhm_limits(0);
    }
    if (map_fitter.fwhm_limits(1) > 0) {
        map_fitter.fwhm_high = map_fitter.fwhm_limits(1);
    }
}

template <class ObservationMapBuffer, class CoaddMapBuffer>
void mirror_source_finding_config_to_coadd(
    const ObservationMapBuffer &omb, CoaddMapBuffer &cmb,
    bool run_coadd) {
    if (!run_coadd) {
        return;
    }
    cmb.source_sigma = omb.source_sigma;
    cmb.source_window_rad = omb.source_window_rad;
    cmb.source_finder_mode = omb.source_finder_mode;
}

template <class MapsToArrays, class InitFwhmForArray, class FitMapSources>
struct SourceFitCallbacks {
    MapsToArrays maps_to_arrays;
    InitFwhmForArray init_fwhm_for_array;
    FitMapSources fit_map_sources;
};

template <class MapsToArrays, class InitFwhmForArray, class FitMapSources>
SourceFitCallbacks<MapsToArrays, InitFwhmForArray, FitMapSources>
make_source_fit_callbacks(const MapsToArrays &maps_to_arrays,
                          const InitFwhmForArray &init_fwhm_for_array,
                          const FitMapSources &fit_map_sources) {
    return {maps_to_arrays, init_fwhm_for_array, fit_map_sources};
}

template <class MapToArray, class CalcStdDev, class WriteSourceTable>
struct SourceTableCallbacks {
    MapToArray maps_to_arrays;
    CalcStdDev calc_std_dev;
    WriteSourceTable write_source_table;
};

template <class MapToArray, class CalcStdDev, class WriteSourceTable>
SourceTableCallbacks<MapToArray, CalcStdDev, WriteSourceTable>
make_source_table_callbacks(const MapToArray &maps_to_arrays,
                            const CalcStdDev &calc_std_dev,
                            const WriteSourceTable &write_source_table) {
    return {maps_to_arrays, calc_std_dev, write_source_table};
}

constexpr int missing_source_location() {
    return -99;
}

template <class SourceCounts, class SourceLocations>
void append_missing_source_location(SourceCounts &source_counts,
                                    SourceLocations &row_source_locations,
                                    SourceLocations &col_source_locations) {
    source_counts.push_back(0);
    row_source_locations.push_back(Eigen::VectorXi::Ones(1));
    col_source_locations.push_back(Eigen::VectorXi::Ones(1));

    row_source_locations.back() *= missing_source_location();
    col_source_locations.back() *= missing_source_location();
}

inline bool has_sources(Eigen::Index n_sources) {
    return n_sources > 0;
}

template <class SourceCounts, class Logger>
void log_source_detection_result(bool sources_found,
                                 const SourceCounts &source_counts,
                                 const Logger &logger) {
    if (sources_found) {
        logger->info("{} source(s) found", source_counts.back());
    }
    else {
        logger->info("no sources found");
    }
}

template <class SourceCounts>
Eigen::Index count_map_sources(const SourceCounts &source_counts) {
    Eigen::Index n_sources = 0;
    for (const auto &sources : source_counts) {
        n_sources += sources;
    }
    return n_sources;
}

template <class MapBuffer>
void clear_source_detection_vectors(MapBuffer &map_buffer) {
    map_buffer.n_sources.clear();
    map_buffer.row_source_locs.clear();
    map_buffer.col_source_locs.clear();
}

template <class MapBuffer>
void initialize_source_fit_tables(MapBuffer &map_buffer,
                                  Eigen::Index n_params) {
    const Eigen::Index n_sources =
        count_map_sources(map_buffer.n_sources);

    map_buffer.source_params.setZero(n_sources, n_params);
    map_buffer.source_perror.setZero(n_sources, n_params);
}

template <class MapBuffer, class MapCount, class Logger>
void detect_map_sources(MapBuffer &map_buffer, MapCount n_maps,
                        const Logger &logger) {
    clear_source_detection_vectors(map_buffer);

    for (Eigen::Index i = 0; i < n_maps; ++i) {
        append_missing_source_location(
            map_buffer.n_sources, map_buffer.row_source_locs,
            map_buffer.col_source_locs);

        const auto sources_found = map_buffer.find_sources(i);
        log_source_detection_result(
            sources_found, map_buffer.n_sources, logger);
    }
}

inline double source_fit_initial_fwhm_pixels(
    double array_fwhm_arcsec, double arcsec_to_rad, double pixel_size_rad) {
    return array_fwhm_arcsec * arcsec_to_rad / pixel_size_rad;
}

template <class ArrayFwhm, class ArrayIndex>
double source_fit_initial_fwhm_for_array(
    ArrayFwhm &array_fwhm_arcsec, ArrayIndex array_index,
    double arcsec_to_rad, double pixel_size_rad) {
    return source_fit_initial_fwhm_pixels(
        array_fwhm_arcsec[array_index], arcsec_to_rad, pixel_size_rad);
}

inline double source_fit_pixel_to_arcsec(double rad_to_arcsec,
                                         double pixel_size_rad) {
    return rad_to_arcsec * pixel_size_rad;
}

inline double source_fit_fwhm_to_arcsec(double rad_to_arcsec,
                                        double std_to_fwhm,
                                        double pixel_size_rad) {
    return rad_to_arcsec * std_to_fwhm * pixel_size_rad;
}

inline SourceFitUnitScale source_fit_unit_scale(double rad_to_arcsec,
                                                double std_to_fwhm,
                                                double pixel_size_rad) {
    return {
        source_fit_pixel_to_arcsec(rad_to_arcsec, pixel_size_rad),
        source_fit_fwhm_to_arcsec(rad_to_arcsec, std_to_fwhm,
                                  pixel_size_rad)};
}

inline SourceFitUnitScale source_fit_unit_scale(
    const SourceFitUnitConstants &constants, double pixel_size_rad) {
    return source_fit_unit_scale(
        constants.rad_to_arcsec, constants.std_to_fwhm, pixel_size_rad);
}

inline SourceFitUnitConstants source_fit_unit_constants(
    double rad_to_arcsec, double std_to_fwhm, double arcsec_to_rad,
    double rad_to_deg, double deg_to_rad, double arcsec_to_deg) {
    return {
        rad_to_arcsec,
        std_to_fwhm,
        arcsec_to_rad,
        rad_to_deg,
        deg_to_rad,
        arcsec_to_deg};
}

template <class Params, class PErrors>
void rescale_source_fit_pixel_units(Params &params, PErrors &perrors,
                                    Eigen::Index n_rows, Eigen::Index n_cols,
                                    double pixel_to_arcsec,
                                    double source_fwhm_to_arcsec) {
    params(1) = pixel_to_arcsec * (params(1) - (n_cols - 1) / 2.0);
    params(2) = pixel_to_arcsec * (params(2) - (n_rows - 1) / 2.0);
    params(3) = source_fwhm_to_arcsec * params(3);
    params(4) = source_fwhm_to_arcsec * params(4);

    perrors(1) = pixel_to_arcsec * perrors(1);
    perrors(2) = pixel_to_arcsec * perrors(2);
    perrors(3) = source_fwhm_to_arcsec * perrors(3);
    perrors(4) = source_fwhm_to_arcsec * perrors(4);
}

inline bool source_fit_uses_radec_projection(const std::string &pixel_axes) {
    return pixel_axes == "radec";
}

template <class Params, class PErrors>
void rescale_source_fit_radec_errors(Params &params, PErrors &perrors,
                                     double ra_deg, double dec_deg,
                                     double arcsec_to_deg) {
    params(1) = ra_deg;
    params(2) = dec_deg;

    perrors(1) = perrors(1) * arcsec_to_deg;
    perrors(2) = perrors(2) * arcsec_to_deg;
}

template <class Params, class PErrors, class Wcs, class TangentToAbs>
void rescale_source_fit_result(
    Params &params, PErrors &perrors, Eigen::Index n_rows,
    Eigen::Index n_cols, double pixel_size_rad,
    const std::string &pixel_axes, const Wcs &wcs,
    const SourceFitUnitConstants &constants,
    const TangentToAbs &tangent_to_abs) {
    const auto unit_scale =
        source_fit_unit_scale(constants, pixel_size_rad);
    rescale_source_fit_pixel_units(
        params, perrors, n_rows, n_cols, unit_scale.pixel_to_arcsec,
        unit_scale.source_fwhm_to_arcsec);

    if (!source_fit_uses_radec_projection(pixel_axes)) {
        return;
    }

    Eigen::VectorXd lat(1), lon(1);
    lat << params(2) * constants.arcsec_to_rad;
    lon << params(1) * constants.arcsec_to_rad;

    auto [adec, ara] = tangent_to_abs(
        lat, lon, wcs.crval[0] * constants.deg_to_rad,
        wcs.crval[1] * constants.deg_to_rad);

    rescale_source_fit_radec_errors(
        params, perrors, ara(0) * constants.rad_to_deg,
        adec(0) * constants.rad_to_deg, constants.arcsec_to_deg);
}

template <class MapBuffer, class MapIndex, class SourceIndex>
SourceInitialPosition source_initial_position(
    const MapBuffer &map_buffer, MapIndex map_index,
    SourceIndex source_index) {
    return {
        static_cast<double>(
            map_buffer.row_source_locs[map_index](source_index)),
        static_cast<double>(
            map_buffer.col_source_locs[map_index](source_index))};
}

template <class SourceRow, class SourceIndex>
auto source_fit_result_row(SourceRow source_row_start,
                           SourceIndex source_index) {
    return source_row_start + source_index;
}

template <class MapBuffer, class SourceRow, class SourceIndex,
          class Params, class PErrors>
void store_source_fit_result(MapBuffer &map_buffer,
                             SourceRow source_row_start,
                             SourceIndex source_index,
                             const Params &params,
                             const PErrors &perrors) {
    const auto source_row =
        source_fit_result_row(source_row_start, source_index);
    map_buffer.source_params.row(source_row) = params;
    map_buffer.source_perror.row(source_row) = perrors;
}

template <class MapBuffer, class SourceRow, class SourceIndex,
          class Params, class PErrors, class TangentToAbs>
void normalize_and_store_source_fit_result(
    MapBuffer &map_buffer, SourceRow source_row_start,
    SourceIndex source_index, Params &params, PErrors &perrors,
    const std::string &pixel_axes,
    const SourceFitUnitConstants &constants,
    const TangentToAbs &tangent_to_abs) {
    rescale_source_fit_result(
        params, perrors, map_buffer.n_rows, map_buffer.n_cols,
        map_buffer.pixel_size_rad, pixel_axes, map_buffer.wcs,
        constants, tangent_to_abs);
    store_source_fit_result(
        map_buffer, source_row_start, source_index, params, perrors);
}

template <class SourceRow, class SourceCount>
auto next_source_fit_row_start(SourceRow source_row_start,
                               SourceCount n_map_sources) {
    return source_row_start + n_map_sources;
}

template <class MapBuffer, class MapCount, class SourceFitCallbacks>
void fit_detected_map_sources(MapBuffer &map_buffer, MapCount n_maps,
                              const SourceFitCallbacks &callbacks) {
    Eigen::Index source_row_start = 0;

    for (Eigen::Index i = 0; i < n_maps; ++i) {
        const auto n_map_sources = map_buffer.n_sources[i];
        if (!has_sources(n_map_sources)) {
            continue;
        }

        const auto array = callbacks.maps_to_arrays(i);
        const auto init_fwhm = callbacks.init_fwhm_for_array(array);
        callbacks.fit_map_sources(
            i, n_map_sources, init_fwhm, source_row_start);
        source_row_start =
            next_source_fit_row_start(source_row_start, n_map_sources);
    }
}

inline std::vector<std::string> source_table_header() {
    return {
        "array",
        "amp",
        "amp_err",
        "x_t",
        "x_t_err",
        "y_t",
        "y_t_err",
        "a_fwhm",
        "a_fwhm_err",
        "b_fwhm",
        "b_fwhm_err",
        "angle",
        "angle_err",
        "sig2noise"};
}

inline std::string source_position_units(const std::string &pixel_axes) {
    return pixel_axes == "radec" ? "deg" : "arcsec";
}

inline std::string source_obsnum_meta_key(Eigen::Index obsnum_index) {
    return "obsnum" + std::to_string(obsnum_index);
}

inline std::string source_units_meta_entry(const std::string &unit) {
    return "units: " + unit;
}

inline std::map<std::string, std::string> source_table_units(
    const std::string &signal_unit, const std::string &position_units) {
    return {
        {"array", "N/A"},
        {"amp", signal_unit},
        {"amp_err", signal_unit},
        {"x_t", position_units},
        {"x_t_err", position_units},
        {"y_t", position_units},
        {"y_t_err", position_units},
        {"a_fwhm", "arcsec"},
        {"a_fwhm_err", "arcsec"},
        {"b_fwhm", "arcsec"},
        {"b_fwhm_err", "arcsec"},
        {"angle", "rad"},
        {"angle_err", "rad"},
        {"sig2noise", "N/A"}};
}

inline std::map<std::string, std::string> source_table_units_for_pixel_axes(
    const std::string &signal_unit, const std::string &pixel_axes) {
    return source_table_units(signal_unit, source_position_units(pixel_axes));
}

inline Eigen::Index source_table_column_count(Eigen::Index n_params) {
    return 2 * n_params + 2;
}

inline Eigen::Index source_table_sig2noise_column(Eigen::Index n_params) {
    return 2 * n_params + 1;
}

inline Eigen::Index source_table_param_column(Eigen::Index source_param_index) {
    return 1 + 2 * source_param_index;
}

inline Eigen::Index source_table_error_column(Eigen::Index source_param_index) {
    return source_table_param_column(source_param_index) + 1;
}

template <class Obsnums, class HeaderUnits, class HeaderDescriptions>
YAML::Node source_table_meta(
    const Obsnums &obsnums, const std::string &source_name,
    const std::string &creation_date, const std::string &observation_date,
    const HeaderUnits &source_header_units,
    HeaderDescriptions &apt_header_description) {
    YAML::Node source_meta;

    for (Eigen::Index i = 0; i < obsnums.size(); ++i) {
        const auto obsnum_key = source_obsnum_meta_key(i);
        source_meta[obsnum_key] = obsnums[i];
    }

    source_meta["Source"] = source_name;
    source_meta["creation_date"] = creation_date;
    source_meta["date"] = observation_date;

    for (const auto &[key, val] : source_header_units) {
        source_meta[key].push_back(source_units_meta_entry(val));
        source_meta[key].push_back(apt_header_description[key]);
    }

    return source_meta;
}

template <class Obsnums, class HeaderDescriptions>
YAML::Node source_table_meta_for_observation(
    const Obsnums &obsnums, const std::string &signal_unit,
    const std::string &pixel_axes, const std::string &source_name,
    const std::string &creation_date, const std::string &observation_date,
    HeaderDescriptions &apt_header_description) {
    const auto source_header_units =
        source_table_units_for_pixel_axes(signal_unit, pixel_axes);
    return source_table_meta(
        obsnums, source_name, creation_date, observation_date,
        source_header_units, apt_header_description);
}

inline float source_signal_to_noise(double source_amplitude,
                                    double map_std_dev) {
    return static_cast<float>(source_amplitude / map_std_dev);
}

template <class SourceTable, class MapBuffer, class MapToArray,
          class CalcStdDev>
void populate_source_table_map_columns(
    SourceTable &source_table, MapBuffer &map_buffer,
    Eigen::Index sig2noise_col, const MapToArray &maps_to_arrays,
    const CalcStdDev &calc_std_dev) {
    Eigen::Index source_row = 0;
    for (Eigen::Index i = 0; i < map_buffer.n_sources.size(); ++i) {
        if (!has_sources(map_buffer.n_sources[i])) {
            continue;
        }

        const double map_std_dev = calc_std_dev(map_buffer.signal[i]);
        for (Eigen::Index j = 0; j < map_buffer.n_sources[i]; ++j) {
            source_table(source_row, 0) = maps_to_arrays(i);
            source_table(source_row, sig2noise_col) =
                source_signal_to_noise(
                    map_buffer.source_params(source_row, 0), map_std_dev);
            ++source_row;
        }
    }
}

template <class SourceTable, class MapBuffer>
void populate_source_table_fit_columns(SourceTable &source_table,
                                       MapBuffer &map_buffer,
                                       Eigen::Index n_params) {
    Eigen::Index source_param_index = 0;
    while (source_param_index < n_params) {
        const auto param_col =
            source_table_param_column(source_param_index);
        const auto error_col =
            source_table_error_column(source_param_index);
        source_table.col(param_col) =
            map_buffer.source_params.col(source_param_index)
                .template cast<float>();
        source_table.col(error_col) =
            map_buffer.source_perror.col(source_param_index)
                .template cast<float>();
        ++source_param_index;
    }
}

template <class MapBuffer, class MapToArray, class CalcStdDev>
Eigen::MatrixXf build_source_table(MapBuffer &map_buffer,
                                   Eigen::Index n_params,
                                   const MapToArray &maps_to_arrays,
                                   const CalcStdDev &calc_std_dev) {
    const Eigen::Index n_sources =
        count_map_sources(map_buffer.n_sources);
    const auto source_table_cols = source_table_column_count(n_params);
    Eigen::MatrixXf source_table(n_sources, source_table_cols);
    const auto sig2noise_col = source_table_sig2noise_column(n_params);

    populate_source_table_map_columns(
        source_table, map_buffer, sig2noise_col, maps_to_arrays,
        calc_std_dev);
    populate_source_table_fit_columns(
        source_table, map_buffer, n_params);

    return source_table;
}

template <class MapBuffer, class HeaderDescriptions, class SourceTableCallbacks>
void write_source_table_output(
    const std::string &source_filename, MapBuffer &map_buffer,
    Eigen::Index n_params, const std::string &pixel_axes,
    const std::string &source_name, const std::string &creation_date,
    const std::string &observation_date,
    HeaderDescriptions &apt_header_description,
    const SourceTableCallbacks &callbacks) {
    auto source_header = source_table_header();
    YAML::Node source_meta =
        source_table_meta_for_observation(
            map_buffer.obsnums, map_buffer.sig_unit, pixel_axes,
            source_name, creation_date, observation_date,
            apt_header_description);
    Eigen::MatrixXf source_table =
        build_source_table(
            map_buffer, n_params, callbacks.maps_to_arrays,
            callbacks.calc_std_dev);

    callbacks.write_source_table(
        source_filename, source_table, source_header, source_meta);
}

template <class SourceCount>
std::vector<int> source_index_vector(SourceCount n_sources) {
    std::vector<int> source_indices(static_cast<std::size_t>(n_sources));
    std::iota(source_indices.begin(), source_indices.end(), 0);
    return source_indices;
}

template <class ParallelPolicy, class SourceCount, class FitSource>
void fit_source_candidates(ParallelPolicy &parallel_policy,
                           SourceCount n_map_sources,
                           const FitSource &fit_source) {
    const auto source_in_vec = source_index_vector(n_map_sources);
    std::vector<int> source_out_vec(source_in_vec.size());

    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy),
               source_in_vec, source_out_vec, [&](auto source_index) {
        fit_source(source_index);
        return 0;
    });
}

template <auto MapType, class Engine, class MapBuffer, class Logger>
void find_map_sources_with_log(Engine &engine, MapBuffer &map_buffer,
                               const Logger &logger,
                               const char *log_message) {
    logger->info("{}", log_message);
    engine.template find_sources<MapType>(map_buffer);
}

template <auto MapType, class Engine, class MapBuffer, class Logger>
void find_map_sources_if_needed(Engine &engine, MapBuffer &map_buffer,
                                const Logger &logger, bool should_find,
                                const char *log_message) {
    if (should_find) {
        find_map_sources_with_log<MapType>(
            engine, map_buffer, logger, log_message);
    }
}

}  // namespace citlali::pipeline
