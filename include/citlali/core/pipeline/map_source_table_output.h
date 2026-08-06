#pragma once

// Included by map_source_finding.h inside namespace citlali::pipeline.

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

inline std::string source_position_units(citlali::config::MapPixelAxes pixel_axes) {
    return citlali::config::is_radec_map_pixel_axes(pixel_axes)
               ? "deg"
               : "arcsec";
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
    const std::string &signal_unit, citlali::config::MapPixelAxes pixel_axes) {
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
        source_meta[key].push_back(
            key == "sig2noise"
                ? "fitted amplitude divided by full-map RMS "
                  "(legacy name; not significance)"
                : apt_header_description[key]);
    }

    // The legacy column name is retained for compatibility. Its package join
    // identifies the actual quicklook ratio without duplicating the package
    // sidecar's full semantic provenance.
    auto contract = source_meta["noise_product_contract"];
    contract["package_id"] = "citlali-noise-products";
    contract["provenance_id"] = "noise_products_provenance.yaml";
    contract["column"] = "sig2noise";
    contract["product_identity"] =
        "fitted_amplitude_over_full_map_rms_ratio";
    contract["product_version"] = "SCI-NOI-002-v1";
    contract["semantic_digest"] =
        "sha256:718feaeebe6004be1714d7c23b33df67a157c6760016f18d6c85a0e75172ae48";
    contract["digest_kind"] = "semantic_contract_sha256";
    contract["scope"] = "source_table_row";
    contract["validity"] =
        "finite_amplitude_and_finite_positive_full_map_rms";
    contract["restriction"] =
        "legacy_alias_deprecated_not_significance";

    return source_meta;
}

template <class Obsnums, class HeaderDescriptions>
YAML::Node source_table_meta_for_observation(
    const Obsnums &obsnums, const std::string &signal_unit,
    citlali::config::MapPixelAxes pixel_axes, const std::string &source_name,
    const std::string &creation_date, const std::string &observation_date,
    HeaderDescriptions &apt_header_description) {
    const auto source_header_units =
        source_table_units_for_pixel_axes(signal_unit, pixel_axes);
    return source_table_meta(
        obsnums, source_name, creation_date, observation_date,
        source_header_units, apt_header_description);
}

inline float fitted_amplitude_over_full_map_rms_ratio(
    double source_amplitude, double map_std_dev) {
    if (!std::isfinite(source_amplitude) || !std::isfinite(map_std_dev) ||
        map_std_dev <= 0.0) {
        return std::numeric_limits<float>::quiet_NaN();
    }
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
                fitted_amplitude_over_full_map_rms_ratio(
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
    Eigen::Index n_params, citlali::config::MapPixelAxes pixel_axes,
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
