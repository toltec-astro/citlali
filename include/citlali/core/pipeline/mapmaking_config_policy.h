#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/config/runtime_config.h>
#include <citlali/core/config/timestream_config.h>

#include <cstdlib>
#include <string>
#include <vector>

namespace citlali::pipeline {

inline std::vector<std::string> allowed_map_regimes() {
    return {"source_dominant", "source_faint", "blank_field", "unknown"};
}

inline bool map_grouping_disallows_polarization(
    bool run_polarization, citlali::config::ReductionType reduction_type,
    citlali::config::MapGrouping map_grouping) {
    return run_polarization &&
           ((citlali::config::is_beammap_reduction_type(reduction_type) &&
             citlali::config::is_automatic_map_grouping(map_grouping)) ||
            citlali::config::is_detector_map_grouping(map_grouping));
}

inline citlali::config::MapGrouping automatic_map_grouping_for_reduction(
    citlali::config::ReductionType reduction_type) {
    return citlali::config::is_beammap_reduction_type(reduction_type)
               ? citlali::config::MapGrouping::detector
               : citlali::config::MapGrouping::array;
}

inline bool detector_map_grouping_disallowed(
    citlali::config::ReductionType reduction_type,
    citlali::config::MapGrouping map_grouping) {
    return citlali::config::is_detector_map_grouping(map_grouping) &&
           !citlali::config::is_beammap_reduction_type(reduction_type);
}

inline citlali::config::MapGrouping effective_map_grouping_for_reduction(
    citlali::config::ReductionType reduction_type,
    citlali::config::MapGrouping requested_grouping) {
    auto grouping = requested_grouping;
    if (citlali::config::is_automatic_map_grouping(grouping)) {
        grouping = automatic_map_grouping_for_reduction(reduction_type);
    }
    if (detector_map_grouping_disallowed(reduction_type, grouping)) {
        grouping = citlali::config::MapGrouping::array;
    }
    return grouping;
}

template <class Logger>
void enforce_map_grouping_polarization_policy(
    bool run_polarization, citlali::config::ReductionType reduction_type,
    citlali::config::MapGrouping map_grouping, const Logger &logger) {
    if (!map_grouping_disallows_polarization(
            run_polarization, reduction_type, map_grouping)) {
        return;
    }
    logger->error(
        "Detector grouping reductions do not currently support polarimetry mode");
    std::exit(EXIT_FAILURE);
}

template <class Logger>
void enforce_beammap_pixel_axes_policy(
                                       citlali::config::ReductionType reduction_type,
                                       citlali::config::MapPixelAxes pixel_axes,
                                       const Logger &logger) {
    if (!citlali::config::is_beammap_reduction_type(reduction_type) ||
        citlali::config::is_altaz_map_pixel_axes(pixel_axes)) {
        return;
    }
    logger->error(
        "beammap reductions require mapmaking.pixel_axes='altaz'; got '{}'",
        citlali::config::to_string(pixel_axes));
    std::exit(EXIT_FAILURE);
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

template <class MapmakingConfig, class OutputMapBlock,
          class PostProcessingConfig>
void mirror_output_map_block_config(MapmakingConfig &target,
                                    const OutputMapBlock &omb,
                                    double rad_to_arcsec,
                                    PostProcessingConfig &post_processing) {
    target.coverage_cut = omb.cov_cut;
    target.pixel_size_arcsec = omb.pixel_size_rad * rad_to_arcsec;
    target.unit = omb.sig_unit;
    if (omb.wcs.naxis.size() >= 2) {
        target.x_size_pix = static_cast<int>(omb.wcs.naxis[0]);
        target.y_size_pix = static_cast<int>(omb.wcs.naxis[1]);
    }
    if (omb.wcs.crpix.size() >= 2) {
        target.crpix1 = omb.wcs.crpix[0];
        target.crpix2 = omb.wcs.crpix[1];
    }
    if (omb.crval_config.size() >= 2) {
        target.crval1_j2000 = omb.crval_config[0];
        target.crval2_j2000 = omb.crval_config[1];
    }
    post_processing.map_histogram_n_bins = omb.hist_n_bins;
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
    if (jinc_mm.mode == "matrix") {
        jinc_mm.allocate_jinc_matrix(pixel_size_rad);
    }
    else if (jinc_mm.mode == "splines") {
        jinc_mm.calculate_jinc_splines();
    }
}

template <class OutputMapBlock, class CoaddMapBlock>
void mirror_noise_map_settings_to_coadd(const OutputMapBlock &omb,
                                        CoaddMapBlock &cmb) {
    cmb.n_noise = omb.n_noise;
    cmb.randomize_dets = omb.randomize_dets;
}

template <class OutputMapBlock, class CoaddMapBlock, class NoiseConfig>
void disable_noise_map_settings(OutputMapBlock &omb, CoaddMapBlock &cmb,
                                NoiseConfig &typed_noise_config) {
    omb.n_noise = 0;
    cmb.n_noise = 0;
    typed_noise_config.n_noise_maps = 0;
}

template <class NaiveMapmaker, class JincMapmaker>
void set_mapmaker_polarization(bool run_polarization,
                               NaiveMapmaker &naive_mm,
                               JincMapmaker &jinc_mm) {
    naive_mm.run_polarization = run_polarization;
    jinc_mm.run_polarization = run_polarization;
}

}  // namespace citlali::pipeline
