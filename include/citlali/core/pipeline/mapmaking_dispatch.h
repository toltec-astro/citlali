#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/pipeline/timestream_native_science_projection.h>

#include <stdexcept>

namespace citlali::pipeline {

template <class NaiveMapmaker, class JincMapmaker, class PtcData,
          class MapBuffer, class MapIndices, class PixelAxes, class Apt>
void populate_naive_or_jinc_maps(citlali::config::MapMethod method,
                                 NaiveMapmaker &naive_mm,
                                 JincMapmaker &jinc_mm, PtcData &ptcdata,
                                 MapBuffer &omb, MapBuffer &cmb,
                                 MapIndices &map_indices,
                                 PixelAxes &pixel_axes, Apt &apt,
                                 double d_fsmp, bool run_omb,
                                 bool run_noise) {
    if (citlali::config::is_naive_map_method(method)) {
        naive_mm.populate_maps_naive(
            ptcdata, omb, cmb, map_indices, pixel_axes, apt, d_fsmp, run_omb,
            run_noise);
    }
    else if (citlali::config::is_jinc_map_method(method)) {
        jinc_mm.populate_maps_jinc(
            ptcdata, omb, cmb, map_indices, pixel_axes, apt, d_fsmp, run_omb,
            run_noise);
    }
}

template <class NaiveMapmaker, class JincMapmaker, class PtcData,
          class MapBuffer, class MapIndices, class PixelAxes, class Apt>
void populate_naive_or_jinc_maps_native(
    citlali::config::MapMethod method, NaiveMapmaker &naive_mm,
    JincMapmaker &jinc_mm, PtcData &ptcdata, MapBuffer &omb,
    MapBuffer &cmb, MapIndices &map_indices, PixelAxes &pixel_axes,
    Apt &apt, double d_fsmp, bool run_omb, bool run_noise,
    const NativeScienceProjection &projection) {
    if (citlali::config::is_naive_map_method(method)) {
        naive_mm.populate_maps_naive_native(
            ptcdata, omb, cmb, map_indices, pixel_axes, apt, d_fsmp,
            run_omb, run_noise, projection);
        return;
    }
    if (citlali::config::is_jinc_map_method(method)) {
        jinc_mm.populate_maps_jinc_parallel_native(
            ptcdata, omb, cmb, map_indices, pixel_axes, apt, d_fsmp,
            run_omb, run_noise, projection);
        return;
    }
    throw std::logic_error(
        "native science projection supports only naive or JINC mapmaking");
}

template <class NaiveMapmaker, class JincMapmaker, class MlMapmaker,
          class PtcData, class MapBuffer, class MapIndices, class PixelAxes,
          class CalibScan>
void populate_lali_maps(citlali::config::MapMethod method,
                        NaiveMapmaker &naive_mm, JincMapmaker &jinc_mm,
                        MlMapmaker &ml_mm, PtcData &ptcdata, MapBuffer &omb,
                        MapBuffer &cmb, MapIndices &map_indices,
                        PixelAxes &pixel_axes, CalibScan &calib_scan,
                        double d_fsmp, bool run_omb, bool run_noise) {
    if (citlali::config::is_maximum_likelihood_map_method(method)) {
        ml_mm.populate_maps_ml(
            ptcdata, omb, cmb, map_indices, pixel_axes, calib_scan, d_fsmp,
            run_omb, run_noise);
        return;
    }
    populate_naive_or_jinc_maps(
        method, naive_mm, jinc_mm, ptcdata, omb, cmb, map_indices, pixel_axes,
        calib_scan.apt, d_fsmp, run_omb, run_noise);
}

inline bool should_populate_final_noise_maps(bool make_noise_maps,
                                             bool run_fruit_loops,
                                             bool has_fruit_loop_map) {
    return make_noise_maps && !(run_fruit_loops && has_fruit_loop_map);
}

}  // namespace citlali::pipeline
