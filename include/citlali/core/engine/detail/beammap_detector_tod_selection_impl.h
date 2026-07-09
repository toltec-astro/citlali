#pragma once

// Beammap detector TOD selection implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/config/calibration_config.h>
#include <citlali/core/engine/detail/beammap_detector_tod_selection.h>
#include <citlali/core/engine/detail/beammap_detector_tod_output_helpers.h>

#include <cstdlib>
#include <map>
#include <vector>

Beammap::BeammapDetectorTodPreflight
Beammap::prepare_detector_specific_ptc_tod_output() {
    BeammapDetectorTodPreflight preflight;
    const auto &detector_tod_config =
        typed_config.beammap.detector_tod_output;
    if (!detector_tod_config.enabled) {
        return preflight;
    }
    preflight.n_scans = telescope.scan_indices.cols();
    if (preflight.n_scans <= 0) {
        logger->error("cannot write detector-specific PTC TOD: no scans");
        std::exit(EXIT_FAILURE);
    }
    if (typed_config.mapmaking.grouping !=
        citlali::config::MapGrouping::detector) {
        logger->warn(
            "beammap.detector_tod_output requires detector map grouping; skipping detector-specific PTC TOD");
        return preflight;
    }
    const auto output_counts = beammap_detector_tod_output_helpers::output_counts(
        detector_tod_config.n_uniform,
        detector_tod_config.n_source_dense);
    preflight.n_uniform = output_counts.n_uniform;
    preflight.n_dense = output_counts.n_dense;
    preflight.n_slots = output_counts.n_slots;
    if (preflight.n_slots <= 0) {
        logger->warn("beammap.detector_tod_output requested with no output slots; skipping");
        return preflight;
    }
    preflight.n_samples_max =
        beammap_detector_tod_output_helpers::max_ptc_samples(ptcs);
    if (preflight.n_samples_max <= 0) {
        logger->warn("beammap.detector_tod_output has no PTC samples to write; skipping");
        return preflight;
    }
    preflight.write_output = true;
    return preflight;
}

Beammap::BeammapDetectorTodPointingSamples
Beammap::sample_detector_tod_pointing(Eigen::Index n_scans) {
    BeammapDetectorTodPointingSamples samples;
    auto [sampled_indices, sampled_scan] =
        beammap_detector_tod_selection::sampled_scan_samples(
            telescope.scan_indices, telescope.tel_data, n_scans);
    samples.sampled_indices = std::move(sampled_indices);
    samples.sampled_scan = std::move(sampled_scan);
    samples.n_sampled =
        static_cast<Eigen::Index>(samples.sampled_indices.size());
    if (samples.n_sampled <= 0) {
        logger->warn("beammap.detector_tod_output cannot sample telescope pointing; skipping");
        return samples;
    }

    samples.sampled_tel_data =
        beammap_detector_tod_selection::sample_tel_data(
            telescope.tel_data, samples.sampled_indices);
    samples.pointing_offsets[citlali::config::pointing_axis_az()] =
        beammap_detector_tod_selection::sample_pointing_offset(
            pointing_offsets_arcsec, citlali::config::pointing_axis_az(),
            samples.sampled_indices);
    samples.pointing_offsets[citlali::config::pointing_axis_alt()] =
        beammap_detector_tod_selection::sample_pointing_offset(
            pointing_offsets_arcsec, citlali::config::pointing_axis_alt(),
            samples.sampled_indices);
    samples.valid = true;
    return samples;
}

Beammap::BeammapDetectorTodSelections
Beammap::make_detector_tod_selections(
    const BeammapDetectorTodPreflight &preflight,
    BeammapDetectorTodPointingSamples &pointing_samples,
    const std::vector<Eigen::Index> &uniform_scans) {
    BeammapDetectorTodSelections selections;
    const Eigen::Index n_scans = preflight.n_scans;
    const int n_dense = preflight.n_dense;
    const Eigen::Index n_slots = preflight.n_slots;
    const auto total_det_slots =
        static_cast<std::size_t>(calib.n_dets) *
        static_cast<std::size_t>(n_slots);
    selections.slot_scan_index.assign(total_det_slots, selections.fill_int);
    selections.slot_kind.assign(total_det_slots, selections.fill_int);
    selections.slot_n_samples.assign(total_det_slots, selections.fill_int);
    selections.slot_inner_start.assign(total_det_slots, selections.fill_int);
    selections.slot_inner_end.assign(total_det_slots, selections.fill_int);
    selections.slot_outer_start.assign(total_det_slots, selections.fill_int);
    selections.slot_outer_end.assign(total_det_slots, selections.fill_int);
    selections.slot_source_distance_arcsec.assign(
        total_det_slots, selections.fill_double);
    selections.det_center_scan_index.assign(
        static_cast<std::size_t>(calib.n_dets), selections.fill_int);
    selections.det_center_distance_arcsec.assign(
        static_cast<std::size_t>(calib.n_dets), selections.fill_double);
    selections.det_fit_x_arcsec.assign(
        static_cast<std::size_t>(calib.n_dets), selections.fill_double);
    selections.det_fit_y_arcsec.assign(
        static_cast<std::size_t>(calib.n_dets), selections.fill_double);
    selections.det_fit_good.assign(
        static_cast<std::size_t>(calib.n_dets), 0);

    std::map<Eigen::Index, Eigen::Index> center_scan_counts;
    std::vector<double> center_distances;
    center_distances.reserve(static_cast<std::size_t>(calib.n_dets));

    for (Eigen::Index det = 0; det < calib.n_dets; ++det) {
        bool used_fit = false;
        auto [x_arcsec, y_arcsec] =
            beammap_detector_tod_selection::detector_source_position(
                det, good_fits, params, calib.apt["x_t"], calib.apt["y_t"],
                omb.pixel_size_rad, omb.n_cols, omb.n_rows, used_fit);
        selections.det_fit_x_arcsec[static_cast<std::size_t>(det)] =
            x_arcsec;
        selections.det_fit_y_arcsec[static_cast<std::size_t>(det)] =
            y_arcsec;
        selections.det_fit_good[static_cast<std::size_t>(det)] =
            used_fit ? 1 : 0;
        if (used_fit) {
            selections.n_det_fit_positions++;
        }
        else if (std::isfinite(x_arcsec) && std::isfinite(y_arcsec)) {
            selections.n_det_fallback_positions++;
        }

        std::vector<double> distances_arcsec;
        const Eigen::Index center_scan =
            beammap_detector_tod_selection::scan_distances_for_detector_source(
                det, x_arcsec, y_arcsec, n_scans,
                pointing_samples.n_sampled, pointing_samples.sampled_scan,
                pointing_samples.sampled_tel_data,
                calib.apt["x_t"], calib.apt["y_t"],
                telescope.pixel_axes, pointing_samples.pointing_offsets,
                typed_config.mapmaking.grouping, distances_arcsec);
        selections.det_center_scan_index[static_cast<std::size_t>(det)] =
            static_cast<int>(center_scan + 1);
        center_scan_counts[center_scan]++;
        if (center_scan >= 0 && center_scan < n_scans &&
            std::isfinite(distances_arcsec[static_cast<std::size_t>(center_scan)])) {
            selections.det_center_distance_arcsec[static_cast<std::size_t>(det)] =
                distances_arcsec[static_cast<std::size_t>(center_scan)];
            center_distances.push_back(
                distances_arcsec[static_cast<std::size_t>(center_scan)]);
        }

        Eigen::Index slot = 0;
        for (const auto scan_index : uniform_scans) {
            beammap_detector_tod_selection::fill_slot_scan_metadata(
                det, slot, n_slots, scan_index, n_scans, 1,
                telescope.scan_indices, ptcs, distances_arcsec,
                selections.slot_scan_index, selections.slot_kind,
                selections.slot_n_samples, selections.slot_inner_start,
                selections.slot_inner_end, selections.slot_outer_start,
                selections.slot_outer_end,
                selections.slot_source_distance_arcsec);
            slot++;
        }
        for (const auto scan_index :
             beammap_detector_tod_selection::dense_scan_window(
                 center_scan, n_dense, n_scans)) {
            beammap_detector_tod_selection::fill_slot_scan_metadata(
                det, slot, n_slots, scan_index, n_scans, 2,
                telescope.scan_indices, ptcs, distances_arcsec,
                selections.slot_scan_index, selections.slot_kind,
                selections.slot_n_samples, selections.slot_inner_start,
                selections.slot_inner_end, selections.slot_outer_start,
                selections.slot_outer_end,
                selections.slot_source_distance_arcsec);
            slot++;
        }
    }

    selections.center_scan_summary =
        beammap_detector_tod_selection::format_center_scan_counts(
            center_scan_counts);

    if (!center_distances.empty()) {
        Eigen::Map<Eigen::VectorXd> dist_vec(
            center_distances.data(),
            static_cast<Eigen::Index>(center_distances.size()));
        selections.median_center_distance_arcsec = tula::alg::median(dist_vec);
    }

    return selections;
}
