#pragma once

#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <Eigen/Core>

#include <cmath>
#include <stdexcept>

namespace citlali::pipeline {

template <class Engine>
void require_supported_native_consumer_execution(const Engine &engine) {
    const auto &raw = raw_time_chunk_config(engine);
    const auto &processed = processed_time_chunk_config(engine);
    const auto &clean = processed.clean;
    const auto &weighting = processed.weighting;

    if (polarimetry_config(engine).enabled) {
        throw std::logic_error(
            "native consumer polarization requires a separately approved measured-channel contract");
    }
    if (mapmaking_enabled(engine)) {
        const auto method = mapmaking_config(engine).method;
        if (!citlali::config::is_naive_map_method(method) &&
            !citlali::config::is_jinc_map_method(method)) {
            throw std::logic_error(
                "native consumer mapmaking supports only naive or JINC occurrence lineage");
        }
    }
    if (tod_output_enabled(engine)) {
        throw std::logic_error(
            "native consumer RTC/PTC TOD output requires native product publication lineage");
    }
    if (clean.enabled && clean.grouping.size() != 1) {
        throw std::logic_error(
            "native consumer requires exactly one established PTC grouping");
    }
    if (clean.mask_radius_arcsec > 0.0) {
        throw std::logic_error(
            "native consumer PTC source masking requires a validity-preserving exclusion contract");
    }
    if (weighting.source_mask_radius_arcsec > 0.0) {
        throw std::logic_error(
            "native consumer variance source masking requires detector-specific native weight support");
    }
    if (fruit_loops_config(engine).enabled) {
        throw std::logic_error(
            "native consumer fruit-loop projection requires a separately approved native feedback contract");
    }
    if (weighting.validation.enabled) {
        throw std::logic_error(
            "native consumer learned weight validation requires native iteration accumulation");
    }
}

template <class Engine>
void require_supported_native_consumer_observation(const Engine &engine) {
    require_supported_native_consumer_execution(engine);
    const auto duplicate = engine.calib.apt.find("duplicate_tone");
    if (duplicate == engine.calib.apt.end() ||
        duplicate->second.size() != engine.calib.n_dets) {
        throw std::logic_error(
            "native consumer requires exact duplicate-tone detector state");
    }
    for (Eigen::Index detector = 0;
         detector < duplicate->second.size(); ++detector) {
        const auto value = duplicate->second(detector);
        if (!std::isfinite(value) || (value != 0.0 && value != 1.0)) {
            throw std::logic_error(
                "native consumer duplicate-tone detector state must be exact binary data");
        }
    }
    const auto &scans = engine.telescope.scan_indices;
    if (scans.rows() != 4) {
        throw std::logic_error(
            "native consumer requires four-row scan interval authority");
    }
    for (Eigen::Index scan = 0; scan < scans.cols(); ++scan) {
        if (scans(0, scan) < 0 || scans(1, scan) < scans(0, scan)) {
            throw std::logic_error(
                "native consumer scan has invalid relational bounds");
        }
        if (scans(2, scan) != scans(0, scan) ||
            scans(3, scan) != scans(1, scan)) {
            throw std::logic_error(
                "native consumer requires a separately approved outer-context run contract");
        }
    }
}

}  // namespace citlali::pipeline
