#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/mapmaking/science_map_contract.h>
#include <citlali/core/pipeline/raw_timestream_policy.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/utils/constants.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace citlali::pipeline {

template <class HeaderMap>
double science_map_source_epoch(const HeaderMap &header) {
    const auto it = header.find("Header.Source.Epoch");
    if (it != header.end() && it->second.size() > 0 &&
        std::isfinite(it->second(0))) {
        return it->second(0);
    }
    // This is the existing FITS WCS adapter default, now captured in the
    // full-precision identity rather than recovered from narrowed headers.
    return 2000.0;
}

template <class HeaderMap>
double require_science_map_header_scalar(const HeaderMap &header,
                                         const std::string &key) {
    const auto it = header.find(key);
    if (it == header.end() || it->second.size() == 0 ||
        !std::isfinite(it->second(0))) {
        throw std::runtime_error(
            "science-map bundle identity requires finite telescope header " +
            key);
    }
    return it->second(0);
}

template <class Kernel>
std::string science_map_response_identity(const Kernel &kernel,
                                          bool enabled) {
    mapmaking::ScienceMapCanonicalDigest digest;
    digest.add_string("citlali-map-response-contract-v1");
    digest.add_integer(enabled ? 1 : 0);
    if (!enabled) {
        digest.add_string("identity-response");
        return digest.finish();
    }

    digest.add_string(kernel.type);
    digest.add_string(kernel.filepath);
    digest.add_double(kernel.sigma_rad);
    digest.add_double(kernel.fwhm_rad);
    digest.add_double(kernel.sigma_limit);
    digest.add_string(kernel.map_grouping);
    digest.add_integer(kernel.img_ext_names.size());
    for (const auto &name : kernel.img_ext_names) {
        digest.add_string(name);
    }
    digest.add_integer(kernel.images.size());
    for (const auto &image : kernel.images) {
        mapmaking::science_map_hash_matrix(digest, image);
    }

    const auto hash_vector = [&](const auto &values) {
        digest.add_integer(values.size());
        for (Eigen::Index i = 0; i < values.size(); ++i) {
            digest.add_double(static_cast<double>(values(i)));
        }
    };
    hash_vector(kernel.source_lat);
    hash_vector(kernel.source_lon);
    hash_vector(kernel.source_a_fwhm_rad);
    hash_vector(kernel.source_b_fwhm_rad);
    hash_vector(kernel.source_valid);
    return digest.finish();
}

template <class Engine>
mapmaking::ScienceMapSlotIdentity science_map_slot_identity(
    const Engine &engine, Eigen::Index slot) {
    const auto &indices = engine.map_indices;
    const auto &calib = engine.calib;
    const auto &grouping = engine.omb.map_grouping;
    if (slot < 0 || slot >= indices.n_maps ||
        slot >= indices.maps_to_arrays.size() ||
        slot >= indices.maps_to_stokes.size()) {
        throw std::runtime_error(
            "science-map ordered slot is outside the typed map-index state");
    }

    mapmaking::ScienceMapSlotIdentity result;
    result.ordered_slot = static_cast<std::size_t>(slot);
    result.grouping = grouping;
    result.array_identity =
        static_cast<long long>(indices.maps_to_arrays(slot));
    result.stokes_identity =
        static_cast<long long>(indices.maps_to_stokes(slot));

    const auto frequency_it =
        engine.toltec_io.array_freq_map.find(indices.maps_to_arrays(slot));
    if (frequency_it == engine.toltec_io.array_freq_map.end() ||
        !std::isfinite(frequency_it->second)) {
        throw std::runtime_error(
            "science-map slot lacks a finite array-frequency identity");
    }
    result.frequency_hz = frequency_it->second;

    Eigen::Index base_slot = slot;
    if (engine.rtcproc.run_polarization) {
        const Eigen::Index stokes_count = static_cast<Eigen::Index>(
            engine.rtcproc.polarization.stokes_params.size());
        if (stokes_count <= 0 || indices.n_maps % stokes_count != 0) {
            throw std::runtime_error(
                "science-map polarization slot cardinality is inconsistent");
        }
        base_slot %= indices.n_maps / stokes_count;
    }

    if (citlali::config::is_array_map_grouping(grouping)) {
        result.group_identity =
            "array:" + std::to_string(result.array_identity);
    }
    else if (citlali::config::is_network_map_grouping(grouping)) {
        if (base_slot >= calib.nws.size()) {
            throw std::runtime_error(
                "science-map network slot lacks a network identity");
        }
        result.group_identity =
            "network:" + std::to_string(calib.nws(base_slot));
    }
    else if (citlali::config::is_detector_map_grouping(grouping)) {
        const auto uid_it = calib.apt.find("uid");
        if (uid_it == calib.apt.end() || base_slot >= uid_it->second.size() ||
            !std::isfinite(uid_it->second(base_slot))) {
            throw std::runtime_error(
                "science-map detector slot lacks a finite detector UID");
        }
        const double uid = uid_it->second(base_slot);
        if (std::trunc(uid) != uid ||
            static_cast<long double>(uid) <
                static_cast<long double>(
                    std::numeric_limits<long long>::lowest()) ||
            static_cast<long double>(uid) >
                static_cast<long double>(
                    std::numeric_limits<long long>::max())) {
            throw std::runtime_error(
                "science-map detector UID must be an exact signed integer");
        }
        result.group_identity = "detector_uid:" +
            std::to_string(static_cast<long long>(uid));
    }
    else if (citlali::config::is_frequency_group_map_grouping(grouping)) {
        if (calib.fg.size() <= 0) {
            throw std::runtime_error(
                "science-map frequency-group slot lacks group identities");
        }
        const Eigen::Index fg_slot = base_slot % calib.fg.size();
        result.group_identity =
            "array:" + std::to_string(result.array_identity) +
            "/frequency_group:" + std::to_string(calib.fg(fg_slot));
    }
    else {
        throw std::runtime_error(
            "science-map bundle identity requires a resolved map grouping");
    }
    return result;
}

template <class Engine>
void configure_observation_science_map_identity(Engine &engine) {
    auto &products = engine.omb.science_products;
    if (!products.initialized || products.geometric_hits.empty()) {
        return;
    }

    const auto &config = mapmaking_config(engine);
    if (!citlali::config::is_naive_map_method(config.method) ||
        engine.rtcproc.run_polarization) {
        return;
    }
    if (engine.omb.n_rows <= 0 || engine.omb.n_cols <= 0 ||
        !std::isfinite(engine.omb.pixel_size_rad) ||
        engine.omb.pixel_size_rad <= 0.0) {
        throw std::runtime_error(
            "science-map identity requires finite positive observation geometry");
    }

    mapmaking::ScienceMapBundleIdentity identity;
    identity.grouping = engine.omb.map_grouping;
    identity.signal_unit = engine.omb.sig_unit;
    identity.estimator_identity =
        "ordinary-naive-normalized-gridding-v1";
    identity.response_identity = science_map_response_identity(
        engine.rtcproc.kernel, raw_kernel_enabled(engine));
    if (raw_kernel_enabled(engine)) {
        identity.required_companions.push_back("kernel_I");
    }
    for (Eigen::Index realization = 0;
         realization < engine.omb.n_noise; ++realization) {
        if (!engine.omb.noise.empty()) {
            identity.required_companions.push_back(
                "noise_realization_" + std::to_string(realization) + "_I");
        }
    }

    identity.wcs.coordinate_frame = engine.telescope.pixel_axes;
    identity.wcs.projection =
        citlali::config::is_altaz_map_pixel_axes(engine.telescope.pixel_axes)
            ? "offset-plane"
            : "TAN";
    if (citlali::config::is_radec_map_pixel_axes(
            engine.telescope.pixel_axes)) {
        identity.wcs.axis_types = {"RA---TAN", "DEC--TAN"};
        identity.wcs.axis_units = {"deg", "deg"};
        identity.wcs.pixel_scale = {
            -engine.omb.pixel_size_rad * RAD_TO_DEG,
            engine.omb.pixel_size_rad * RAD_TO_DEG};
        identity.wcs.reference_world = {
            require_science_map_header_scalar(
                engine.telescope.tel_header, "Header.Source.Ra") *
                RAD_TO_DEG,
            require_science_map_header_scalar(
                engine.telescope.tel_header, "Header.Source.Dec") *
                RAD_TO_DEG};
    }
    else if (citlali::config::is_galactic_map_pixel_axes(
                 engine.telescope.pixel_axes)) {
        identity.wcs.axis_types = {"GLON-TAN", "GLAT-TAN"};
        identity.wcs.axis_units = {"deg", "deg"};
        identity.wcs.pixel_scale = {
            -engine.omb.pixel_size_rad * RAD_TO_DEG,
            engine.omb.pixel_size_rad * RAD_TO_DEG};
        identity.wcs.reference_world = {
            require_science_map_header_scalar(
                engine.telescope.tel_header, "Header.Source.L") *
                RAD_TO_DEG,
            require_science_map_header_scalar(
                engine.telescope.tel_header, "Header.Source.B") *
                RAD_TO_DEG};
    }
    else if (citlali::config::is_altaz_map_pixel_axes(
                 engine.telescope.pixel_axes)) {
        identity.wcs.axis_types = {"AZOFFSET", "ELOFFSET"};
        identity.wcs.axis_units = {
            engine.omb.wcs.cunit.at(0), engine.omb.wcs.cunit.at(1)};
        const double conversion = identity.wcs.axis_units.at(0) == "arcsec"
                                      ? RAD_TO_ASEC
                                      : RAD_TO_DEG;
        identity.wcs.pixel_scale = {
            -engine.omb.pixel_size_rad * conversion,
            engine.omb.pixel_size_rad * conversion};
        identity.wcs.reference_world = {0.0, 0.0};
    }
    else {
        throw std::runtime_error(
            "science-map identity requires a supported coordinate frame");
    }
    identity.wcs.reference_pixel = {
        static_cast<double>(engine.omb.n_cols - 1) / 2.0,
        static_cast<double>(engine.omb.n_rows - 1) / 2.0};
    identity.wcs.source_epoch =
        science_map_source_epoch(engine.telescope.tel_header);
    identity.wcs.orientation_rad = 0.0;
    identity.rows = engine.omb.n_rows;
    identity.cols = engine.omb.n_cols;
    identity.ordered_slots.reserve(
        static_cast<std::size_t>(engine.map_indices.n_maps));
    for (Eigen::Index slot = 0; slot < engine.map_indices.n_maps; ++slot) {
        identity.ordered_slots.push_back(
            science_map_slot_identity(engine, slot));
    }

    products.bundle_identity = std::move(identity);
    products.identity_admitted = true;
}

}  // namespace citlali::pipeline
