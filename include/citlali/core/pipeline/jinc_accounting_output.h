#pragma once

#include <citlali/core/error/error.h>
#include <citlali/core/mapmaking/map.h>
#include <citlali/core/utils/netcdf_io.h>

#include <filesystem>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace citlali::pipeline {

inline std::string jinc_accounting_base_filename(std::string mapdiag_base) {
    constexpr const char *suffix = "_mapdiag";
    if (mapdiag_base.size() >= std::char_traits<char>::length(suffix) &&
        mapdiag_base.compare(
            mapdiag_base.size() - std::char_traits<char>::length(suffix),
            std::char_traits<char>::length(suffix), suffix) == 0) {
        mapdiag_base.erase(
            mapdiag_base.size() - std::char_traits<char>::length(suffix));
    }
    return mapdiag_base + "_jinc_accounting";
}

template <class Matrix>
auto jinc_accounting_row_major_values(const Matrix &matrix) {
    using Scalar = typename Matrix::Scalar;
    std::vector<Scalar> values;
    values.reserve(static_cast<std::size_t>(matrix.size()));
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
        for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
            values.push_back(matrix(row, col));
        }
    }
    return values;
}

inline void add_jinc_accounting_double_plane(
    netCDF::NcFile &file, const std::string &name,
    const Eigen::MatrixXd &matrix, const netCDF::NcDim &rows,
    const netCDF::NcDim &cols) {
    auto var = file.addVar(name, netCDF::ncDouble, {rows, cols});
    set_netcdf_chunking_and_compression(
        var, {std::min<std::size_t>(128, rows.getSize()),
              std::min<std::size_t>(128, cols.getSize())});
    const auto values = jinc_accounting_row_major_values(matrix);
    var.putVar(values.data());
}

inline void add_jinc_accounting_count_plane(
    netCDF::NcFile &file, const std::string &name,
    const mapmaking::JincAccountingState::CountMatrix &matrix,
    const netCDF::NcDim &rows, const netCDF::NcDim &cols) {
    auto var = file.addVar(name, netCDF::ncInt64, {rows, cols});
    set_netcdf_chunking_and_compression(
        var, {std::min<std::size_t>(128, rows.getSize()),
              std::min<std::size_t>(128, cols.getSize())});
    const auto values = jinc_accounting_row_major_values(matrix);
    var.putVar(values.data());
}

inline void write_jinc_accounting_sample_table_atomic(
    const std::string &filename,
    const std::vector<mapmaking::JincAccountingSample> &samples) {
    namespace fs = std::filesystem;
    const fs::path final_path(filename);
    const fs::path temp_path(filename + ".tmp");
    std::error_code ec;
    fs::remove(temp_path, ec);
    try {
        std::ofstream out(temp_path, std::ios::trunc);
        if (!out) {
            throw std::runtime_error("cannot open temporary ECSV file");
        }
        out << "# %ECSV 1.0\n# ---\n# datatype:\n"
            << "# - {name: scan_index, datatype: int32}\n"
            << "# - {name: sample_index, datatype: int64}\n"
            << "# - {name: array_id, datatype: int32}\n"
            << "# - {name: uid, datatype: int32}\n"
            << "# - {name: processed_signal, datatype: float64}\n"
            << "# - {name: analysis_coefficient, datatype: float64}\n"
            << "# - {name: final_flag, datatype: int32}\n"
            << "# - {name: admitted, datatype: int32}\n"
            << "# - {name: reason, datatype: string}\n"
            << "# - {name: continuous_row, datatype: float64}\n"
            << "# - {name: continuous_col, datatype: float64}\n"
            << "# - {name: rounded_row, datatype: int64}\n"
            << "# - {name: rounded_col, datatype: int64}\n"
            << "# - {name: row_phase, datatype: float64}\n"
            << "# - {name: col_phase, datatype: float64}\n"
            << "# - {name: subpixel_index, datatype: int32}\n"
            << "# - {name: center_in_map, datatype: int32}\n"
            << "# - {name: contributed_pixel_count, datatype: int64}\n"
            << "# meta: {diagnostic_only: true, not_science_product: true}\n"
            << "# schema: astropy-2.0\n"
            << "scan_index sample_index array_id uid processed_signal "
               "analysis_coefficient final_flag admitted reason continuous_row "
               "continuous_col rounded_row rounded_col row_phase col_phase "
               "subpixel_index center_in_map contributed_pixel_count\n";
        out << std::setprecision(std::numeric_limits<double>::max_digits10);
        for (const auto &sample : samples) {
            out << sample.scan_index << ' ' << sample.sample_index << ' '
                << sample.array_id << ' ' << sample.uid << ' '
                << sample.processed_signal << ' '
                << sample.analysis_coefficient << ' ' << sample.final_flag
                << ' ' << sample.admitted << ' ' << sample.reason << ' '
                << sample.continuous_row << ' ' << sample.continuous_col
                << ' ' << sample.rounded_row << ' ' << sample.rounded_col
                << ' ' << sample.row_phase << ' ' << sample.col_phase << ' '
                << sample.subpixel_index << ' ' << sample.center_in_map << ' '
                << sample.contributed_pixel_count << '\n';
        }
        out.close();
        if (!out) {
            throw std::runtime_error("failed while writing ECSV data");
        }
        fs::remove(final_path, ec);
        ec.clear();
        fs::rename(temp_path, final_path, ec);
        if (ec) {
            throw std::runtime_error("failed to publish ECSV: " + ec.message());
        }
    }
    catch (const std::exception &error) {
        fs::remove(temp_path, ec);
        throw citlali::error::output(
            "failed to write required JINC accounting sample table " +
            filename + ": " + error.what());
    }
}

template <class MapBuffer>
void write_jinc_accounting_receipt(MapBuffer &buffer,
                                   const std::string &mapdiag_base) {
    auto &state = buffer.jinc_accounting;
    if (!state.enabled()) {
        return;
    }
    const auto slot = state.map_index;
    if (slot < 0 || slot >= static_cast<Eigen::Index>(buffer.weight.size())) {
        throw std::runtime_error("JINC accounting final map slot is absent");
    }
    if (!buffer.weight_formal.empty()) {
        if (slot >= static_cast<Eigen::Index>(buffer.weight_formal.size()) ||
            buffer.weight_formal[slot].rows() != state.formal_coefficient.rows() ||
            buffer.weight_formal[slot].cols() != state.formal_coefficient.cols() ||
            !(buffer.weight_formal[slot].array() ==
              state.formal_coefficient.array()).all()) {
            throw std::runtime_error(
                "JINC accounting formal-coefficient snapshot changed before output");
        }
    }
    const auto science_selection =
        engine_utils::find_weight_threshold_selection(
            buffer.weight[slot], buffer.cov_cut);
    const auto finite_positive =
        buffer.weight[slot].array().isFinite() &&
        (buffer.weight[slot].array() > 0.0);
    const Eigen::ArrayXXd science_support =
        (finite_positive &&
         (buffer.weight[slot].array() >= science_selection.threshold))
            .template cast<double>();
    double empirical_scale = 1.0;
    if (slot < buffer.noise_weight_scale.size()) {
        empirical_scale = buffer.noise_weight_scale(slot);
    }
    state.capture_finalization(
        buffer.weight[slot], science_support, science_selection.threshold,
        empirical_scale);
    state.require_complete();

    const std::string base = jinc_accounting_base_filename(mapdiag_base);
    write_netcdf_atomic(base + ".nc", [&](netCDF::NcFile &file) {
        const auto rows = file.addDim(
            "map_row", static_cast<std::size_t>(state.total_n.rows()));
        const auto cols = file.addDim(
            "map_col", static_cast<std::size_t>(state.total_n.cols()));
        add_netcdf_var(file, "schema_identity",
                       std::string{mapmaking::jinc_accounting_schema});
        add_netcdf_var(file, "diagnostic_only", true);
        add_netcdf_var(file, "not_science_product", true);
        add_netcdf_var(file, "not_checkpoint_state", true);
        add_netcdf_var(file, "obsnum", state.obsnum);
        add_netcdf_var(file, "fruit_iteration", state.fruit_iteration);
        add_netcdf_var(file, "array_name", state.array_name);
        add_netcdf_var(file, "array_id", state.array_id);
        add_netcdf_var(file, "map_index", state.map_index);
        add_netcdf_var(file, "target_uid", state.uid);
        add_netcdf_var(file, "target_scan_index", state.scan_index);
        add_netcdf_var(file, "signal_unit", buffer.sig_unit);
        add_netcdf_var(file, "pixel_size_rad", buffer.pixel_size_rad);
        add_netcdf_var(file, "coverage_cut", buffer.cov_cut);
        add_netcdf_var(file, "jinc_r_max", state.r_max);
        add_netcdf_var(file, "jinc_subpixel_n", state.subpixel_n);
        add_netcdf_var(file, "normalization_denominator_abs_min", 1e-8);
        add_netcdf_var(file, "normalization_threshold",
                       state.normalization_threshold);
        add_netcdf_var(file, "science_policy_threshold",
                       state.science_policy_threshold);
        add_netcdf_var(file, "empirical_coefficient_scale",
                       state.empirical_scale);
        add_netcdf_var(
            file, "normalization_identity",
            std::string{"m=N/C; formal_coefficient=C^2/Q; finite |C|>1e-8 and Q>0; positive order-statistic support"});
        add_netcdf_var(
            file, "coefficient_identity",
            std::string{"formal coefficient C^2/Q; empirical coefficient is its positive global rescale"});
        add_netcdf_var(
            file, "target_subset_identity",
            std::string{"exact selected array, UID, zero-based scan, and ordinarily admitted final-PTC occurrences"});

        auto add_vector = [&](const std::string &name,
                              const auto &values) {
            const auto dim = file.addDim(
                name + "_dim", static_cast<std::size_t>(values.size()));
            auto var = file.addVar(name, netCDF::ncDouble, dim);
            var.putVar(values.data());
        };
        add_vector("wcs_cdelt", buffer.wcs.cdelt);
        add_vector("wcs_crpix", buffer.wcs.crpix);
        add_vector("wcs_crval", buffer.wcs.crval);
        add_vector("jinc_shape_params", state.kernel_shape_params);
        add_vector("row_tangent_coordinate_rad", buffer.rows_tan_vec);
        add_vector("col_tangent_coordinate_rad", buffer.cols_tan_vec);

        add_jinc_accounting_double_plane(file, "total_N", state.total_n,
                                         rows, cols);
        add_jinc_accounting_double_plane(file, "total_C", state.total_c,
                                         rows, cols);
        add_jinc_accounting_double_plane(file, "total_Q", state.total_q,
                                         rows, cols);
        add_jinc_accounting_double_plane(file, "target_N", state.target_n,
                                         rows, cols);
        add_jinc_accounting_double_plane(file, "target_C", state.target_c,
                                         rows, cols);
        add_jinc_accounting_double_plane(file, "target_Q", state.target_q,
                                         rows, cols);
        add_jinc_accounting_double_plane(
            file, "total_abs_N_terms", state.total_abs_n, rows, cols);
        add_jinc_accounting_double_plane(
            file, "total_abs_C_terms", state.total_abs_c, rows, cols);
        add_jinc_accounting_double_plane(
            file, "target_abs_N_terms", state.target_abs_n, rows, cols);
        add_jinc_accounting_double_plane(
            file, "target_abs_C_terms", state.target_abs_c, rows, cols);
        add_jinc_accounting_count_plane(
            file, "total_occurrence_pixel_count",
            state.total_occurrence_count, rows, cols);
        add_jinc_accounting_count_plane(
            file, "target_occurrence_pixel_count",
            state.target_occurrence_count, rows, cols);
        add_jinc_accounting_count_plane(
            file, "total_unique_detector_count",
            state.total_unique_detector_count, rows, cols);
        add_jinc_accounting_count_plane(
            file, "target_unique_detector_count",
            state.target_unique_detector_count, rows, cols);
        add_jinc_accounting_double_plane(
            file, "formal_coefficient", state.formal_coefficient, rows, cols);
        add_jinc_accounting_double_plane(
            file, "empirical_coefficient", state.empirical_coefficient,
            rows, cols);
        add_jinc_accounting_count_plane(
            file, "normalization_support", state.normalization_support,
            rows, cols);
        add_jinc_accounting_count_plane(
            file, "science_policy_support", state.science_policy_support,
            rows, cols);
    });
    write_jinc_accounting_sample_table_atomic(
        base + "_target_samples.ecsv", state.target_samples);
}

}  // namespace citlali::pipeline
