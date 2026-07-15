#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/config/runtime_config.h>
#include <citlali/core/config/timestream_config.h>
#include <citlali/core/error/error.h>
#include <citlali/core/pipeline/mapmaking_resolution.h>

#include <fmt/format.h>

#include <string>
#include <string_view>
#include <vector>

namespace citlali::pipeline {

inline constexpr std::string_view jinc_matrix_backend_mode() {
    return "matrix";
}

inline constexpr std::string_view jinc_splines_backend_mode() {
    return "splines";
}

inline bool is_jinc_matrix_backend_mode(std::string_view mode) {
    return mode == jinc_matrix_backend_mode();
}

inline bool is_jinc_splines_backend_mode(std::string_view mode) {
    return mode == jinc_splines_backend_mode();
}

inline std::vector<std::string> allowed_map_regimes() {
    std::vector<std::string> values;
    values.reserve(citlali::config::source_map_regime_names.size());
    for (const auto &entry : citlali::config::source_map_regime_names) {
        values.emplace_back(entry.name);
    }
    return values;
}

inline bool map_grouping_disallows_polarization(
    bool run_polarization, citlali::config::ReductionType reduction_type,
    citlali::config::MapGrouping map_grouping) {
    return run_polarization &&
           ((citlali::config::is_beammap_reduction_type(reduction_type) &&
             citlali::config::is_automatic_map_grouping(map_grouping)) ||
            citlali::config::is_detector_map_grouping(map_grouping));
}

void enforce_map_grouping_polarization_policy(
    bool run_polarization, citlali::config::ReductionType reduction_type,
    citlali::config::MapGrouping map_grouping) {
    if (!map_grouping_disallows_polarization(
            run_polarization, reduction_type, map_grouping)) {
        return;
    }
    throw citlali::error::invalid_config(
        "detector grouping reductions do not currently support polarimetry mode");
}

void enforce_beammap_pixel_axes_policy(
    citlali::config::ReductionType reduction_type,
    citlali::config::MapPixelAxes pixel_axes) {
    if (!citlali::config::is_beammap_reduction_type(reduction_type) ||
        citlali::config::is_altaz_map_pixel_axes(pixel_axes)) {
        return;
    }
    throw citlali::error::invalid_config(fmt::format(
        "beammap reductions require mapmaking.pixel_axes='altaz'; got '{}'",
        citlali::config::to_string(pixel_axes)));
}

template <class RtcProc, class PtcProc>
void sync_map_grouping_to_timestream_processors(
    citlali::config::MapGrouping map_grouping, RtcProc &rtcproc,
    PtcProc &ptcproc) {
    const std::string map_grouping_name{
        std::string(citlali::config::to_string(map_grouping))};
    rtcproc.kernel.map_grouping = map_grouping_name;
    ptcproc.active_map_grouping = map_grouping_name;
}

template <class Calib>
int base_map_count_for_grouping(citlali::config::MapGrouping grouping,
                                const Calib &calib) {
    if (citlali::config::is_detector_map_grouping(grouping)) {
        return calib.n_dets;
    }
    if (citlali::config::is_network_map_grouping(grouping)) {
        return calib.n_nws;
    }
    if (citlali::config::is_array_map_grouping(grouping)) {
        return calib.n_arrays;
    }
    if (citlali::config::is_frequency_group_map_grouping(grouping)) {
        return static_cast<int>(calib.fg.size()) * calib.n_arrays;
    }
    return 0;
}

template <class Polarization>
int apply_polarization_map_count(int base_count, bool run_polarization,
                                 const Polarization &polarization) {
    return run_polarization
               ? base_count *
                     static_cast<int>(polarization.stokes_params.size())
               : base_count;
}

template <class OutputMapBlock, class CoaddMapBlock>
void apply_uncalibrated_map_units(bool run_calibrate,
                                  citlali::config::TodType tod_type,
                                  OutputMapBlock &omb,
                                  CoaddMapBlock &cmb) {
    if (run_calibrate) {
        return;
    }
    const std::string tod_type_name{
        std::string(citlali::config::to_string(tod_type))};
    omb.sig_unit = tod_type_name;
    cmb.sig_unit = tod_type_name;
}

template <class OutputMapBlock, class CoaddMapBlock, class JincMapmaker,
          class ParallelPolicy>
void sync_mapmaking_parallel_policy(const ParallelPolicy &parallel_policy,
                                    OutputMapBlock &omb, CoaddMapBlock &cmb,
                                    JincMapmaker &jinc_mm) {
    omb.parallel_policy = parallel_policy;
    cmb.parallel_policy = parallel_policy;
    jinc_mm.parallel_policy = parallel_policy;
}

template <class JincMapmaker, class PtcProc>
void finalize_jinc_filter_config(JincMapmaker &jinc_mm, PtcProc &ptcproc,
                                 double pixel_size_rad) {
    mirror_jinc_mapmaker_config_to_fruit_loops(jinc_mm, ptcproc);
    if (is_jinc_matrix_backend_mode(jinc_mm.mode)) {
        jinc_mm.allocate_jinc_matrix(pixel_size_rad);
    }
    else if (is_jinc_splines_backend_mode(jinc_mm.mode)) {
        jinc_mm.calculate_jinc_splines();
    }
}

template <class NaiveMapmaker, class JincMapmaker>
void set_mapmaker_polarization(bool run_polarization,
                               NaiveMapmaker &naive_mm,
                               JincMapmaker &jinc_mm) {
    naive_mm.run_polarization = run_polarization;
    jinc_mm.run_polarization = run_polarization;
}

}  // namespace citlali::pipeline
