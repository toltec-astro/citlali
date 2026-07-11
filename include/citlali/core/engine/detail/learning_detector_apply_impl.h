#pragma once

// Engine learned detector-exclusion application detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/reduction_config_accessors.h>

template <class rtc_t, class calib_t>
void Engine::apply_learned_rtc_sample_masks(rtc_t &rtcdata, calib_t &calib_scan) {
    apply_learned_detector_exclusions(
        rtcdata, calib_scan, "pre_rtc_detector_exclusion", true, false,
        true, true);
    apply_learned_sample_masks(
        rtcdata, calib_scan, true, "pre_rtc",
        citlali::pipeline::raw_time_chunk_config(*this)
            .despike.source_protection.active,
        citlali::pipeline::raw_time_chunk_config(*this)
            .despike.source_protection.radius_arcsec);
}

template <class ptc_t, class calib_t>
void Engine::apply_learned_ptc_sample_masks(ptc_t &ptcdata, calib_t &calib_scan) {
    apply_learned_sample_masks(
        ptcdata, calib_scan, false, "pre_ptc",
        ptcproc.second_pass_local.source_protection_enabled,
        ptcproc.second_pass_local.source_protection_radius_arcsec);
}

template <class ptc_t, class calib_t>
void Engine::apply_learned_ptc_detector_exclusions(ptc_t &ptcdata,
                                                   calib_t &calib_scan) {
    apply_learned_detector_exclusions(
        ptcdata, calib_scan, "pre_ptc_detector_exclusion", false, true,
        true, true);
}

template <class tc_t, class calib_t>
void Engine::apply_learned_mapmaking_detector_exclusions(tc_t &tcdata,
                                                         calib_t &calib_scan) {
    apply_learned_detector_exclusions(
        tcdata, calib_scan, "pre_mapmaking_detector_exclusion", false, false,
        false, true);
}
