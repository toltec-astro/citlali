#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_detector_tod_selection.h>
#include <citlali/core/engine/detail/beammap_detector_tod_netcdf_helpers.h>
#include <citlali/core/engine/detail/beammap_detector_tod_output_helpers.h>
#include <citlali/core/engine/detail/beammap_detector_tod_selection_impl.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

void Beammap::write_detector_specific_ptc_tod_file(
    const std::string &filename,
    int output_iter,
    const BeammapDetectorTodPreflight &preflight,
    const BeammapDetectorTodSelections &selections) {
    const int n_uniform = preflight.n_uniform;
    const int n_dense = preflight.n_dense;
    const Eigen::Index n_slots = preflight.n_slots;
    const Eigen::Index n_samples_max = preflight.n_samples_max;

    write_netcdf_atomic(filename, [&](netCDF::NcFile &fo) {
        namespace tod_nc = beammap_detector_tod_netcdf_helpers;

        tod_nc::put_output_metadata(
            fo, observation_identity.obsnum, telescope,
            citlali::pipeline::runtime_reduction_type(*this),
            citlali::pipeline::timestream_config(*this).type,
            processed_time_chunk_fs_hz(), output_iter, n_uniform, n_dense);

        netCDF::NcDim n_dets_dim = fo.addDim("n_dets", calib.n_dets);
        netCDF::NcDim n_slots_dim = fo.addDim("n_detector_tod_slots", n_slots);
        netCDF::NcDim n_samples_dim = fo.addDim("n_samples_max", n_samples_max);
        std::vector<netCDF::NcDim> det_dims = {n_dets_dim};
        std::vector<netCDF::NcDim> det_slot_dims = {n_dets_dim, n_slots_dim};
        std::vector<netCDF::NcDim> data_dims = {
            n_dets_dim, n_slots_dim, n_samples_dim};
        std::vector<std::size_t> det_slot_chunks = {
            1, static_cast<std::size_t>(n_slots)};
        std::vector<std::size_t> data_chunks = {
            1, 1, static_cast<std::size_t>(n_samples_max)};

        tod_nc::put_detector_int(
            fo, det_dims, "detector_tod_uid", "detector UID along n_dets",
            tod_nc::apt_int_values(
                calib.apt, "uid", calib.n_dets, selections.fill_int));
        tod_nc::put_detector_int(
            fo, det_dims, "detector_tod_array", "array index along n_dets",
            tod_nc::apt_int_values(
                calib.apt, "array", calib.n_dets, selections.fill_int));
        tod_nc::put_detector_int(
            fo, det_dims, "detector_tod_network", "network index along n_dets",
            tod_nc::apt_int_values(
                calib.apt, "nw", calib.n_dets, selections.fill_int));
        tod_nc::put_detector_int(
            fo, det_dims, "detector_tod_fit_good",
            "1 when the source-crossing scan was centered on a good fit, else 0",
            selections.det_fit_good);
        tod_nc::put_detector_int(
            fo, det_dims, "detector_tod_source_center_scan_index",
            "1-based full-observation scan with closest approach to the fitted detector source position",
            selections.det_center_scan_index);
        tod_nc::put_detector_double(
            fo, det_dims, "detector_tod_source_center_distance_arcsec",
            "arcsec",
            "closest sampled distance from source in detector source-center scan",
            selections.det_center_distance_arcsec);
        tod_nc::put_detector_double(
            fo, det_dims, "detector_tod_fit_x_t_arcsec", "arcsec",
            "fitted detector source x_t used for dense scan selection",
            selections.det_fit_x_arcsec);
        tod_nc::put_detector_double(
            fo, det_dims, "detector_tod_fit_y_t_arcsec", "arcsec",
            "fitted detector source y_t used for dense scan selection",
            selections.det_fit_y_arcsec);

        tod_nc::put_slot_int(
            fo, det_slot_dims, det_slot_chunks, "detector_tod_slot_kind",
            "slot kind: 1=uniform over full raster, 2=dense around detector source crossing",
            selections.slot_kind);
        tod_nc::put_slot_int(
            fo, det_slot_dims, det_slot_chunks, "detector_tod_scan_index",
            "1-based full-observation scan index selected for this detector/slot",
            selections.slot_scan_index);
        tod_nc::put_slot_int(
            fo, det_slot_dims, det_slot_chunks, "detector_tod_n_samples",
            "number of PTC samples populated for this detector/slot",
            selections.slot_n_samples);
        tod_nc::put_slot_int(
            fo, det_slot_dims, det_slot_chunks,
            "detector_tod_scan_inner_start_sample",
            "raw inner-scan start sample from telescope scan definition",
            selections.slot_inner_start);
        tod_nc::put_slot_int(
            fo, det_slot_dims, det_slot_chunks,
            "detector_tod_scan_inner_end_sample",
            "raw inner-scan end sample from telescope scan definition",
            selections.slot_inner_end);
        tod_nc::put_slot_int(
            fo, det_slot_dims, det_slot_chunks,
            "detector_tod_scan_outer_start_sample",
            "raw outer-scan start sample from telescope scan definition",
            selections.slot_outer_start);
        tod_nc::put_slot_int(
            fo, det_slot_dims, det_slot_chunks,
            "detector_tod_scan_outer_end_sample",
            "raw outer-scan end sample from telescope scan definition",
            selections.slot_outer_end);
        tod_nc::put_slot_double(
            fo, det_slot_dims, det_slot_chunks,
            "detector_tod_source_distance_arcsec", "arcsec",
            "closest sampled source distance for this detector on this selected scan",
            selections.slot_source_distance_arcsec);

        netCDF::NcVar signal_v = fo.addVar("signal", netCDF::ncFloat, data_dims);
        signal_v.putAtt("units", omb.sig_unit);
        signal_v.putAtt(
            "comment",
            "PTC signal for each detector-specific selected scan; unused samples are NaN");
        set_netcdf_chunking_and_compression(signal_v, data_chunks, 1);

        netCDF::NcVar flags_v = fo.addVar("flags", netCDF::ncByte, data_dims);
        flags_v.putAtt("units", "N/A");
        flags_v.putAtt("comment", "0=good, 1=flagged, -1=unused sample");
        set_netcdf_chunking_and_compression(flags_v, data_chunks, 1);

        tod_nc::put_detector_tod_signal_flags(
            signal_v, flags_v, ptcs, selections.slot_scan_index, calib.n_dets,
            n_slots, n_samples_max, selections.fill_float,
            selections.fill_flag);
    });
}

void Beammap::write_detector_specific_ptc_tod(int output_iter) {
    const auto preflight = prepare_detector_specific_ptc_tod_output();
    if (!preflight.write_output) {
        return;
    }
    const Eigen::Index n_scans = preflight.n_scans;
    const int n_uniform = preflight.n_uniform;
    const int n_dense = preflight.n_dense;
    const Eigen::Index n_slots = preflight.n_slots;

    std::vector<Eigen::Index> uniform_scans =
        beammap_detector_tod_selection::uniform_scan_indices(n_uniform, n_scans);

    auto pointing_samples = sample_detector_tod_pointing(n_scans);
    if (!pointing_samples.valid) {
        return;
    }

    const auto selections = make_detector_tod_selections(
        preflight, pointing_samples, uniform_scans);

    const auto detector_tod_paths =
        beammap_detector_tod_output_helpers::output_paths(
        output_paths.obsnum_dir_name,
        citlali::pipeline::beammap_config(*this)
            .detector_tod_output.subdir_name,
        telescope.sim_obs, citlali::pipeline::runtime_reduction_type(*this),
        observation_identity.obsnum);
    const std::string &filename = detector_tod_paths.filename;

    logger->info(
        "writing detector-specific PTC TOD iter={} file={} n_dets={} n_slots={} n_uniform={} n_source_dense={} fit_positions={} fallback_positions={} median_center_distance_arcsec={} top_center_scans={}",
        output_iter,
        filename,
        calib.n_dets,
        n_slots,
        n_uniform,
        n_dense,
        selections.n_det_fit_positions,
        selections.n_det_fallback_positions,
        selections.median_center_distance_arcsec,
        selections.center_scan_summary);

    write_detector_specific_ptc_tod_file(
        filename, output_iter, preflight, selections);
}
