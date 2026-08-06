#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/pipeline/beammap_direction_selection.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <filesystem>
#include <stdexcept>

void Beammap::prepare_beammap_direction_selection() {
    const auto mode =
        citlali::pipeline::beammap_config(*this).direction_mode;
    if (citlali::pipeline::beammap_direction_mode_is_standard(mode)) {
        return;
    }
    if (beammap_direction_selection_initialized) {
        return;
    }
    if (telescope.obs_pgm != "Map" || telescope.exec_mode) {
        throw std::runtime_error(
            "nonstandard beammap direction_mode requires a raster Map observation");
    }
    const auto angle_it = telescope.tel_header.find("Header.Map.ScanAngle");
    if (angle_it == telescope.tel_header.end() ||
        angle_it->second.size() != 1) {
        throw std::runtime_error(
            "beammap direction selection requires scalar Header.Map.ScanAngle");
    }

    beammap_direction_selection =
        citlali::pipeline::make_beammap_direction_selection_plan(
            mode, telescope.scan_indices, telescope.tel_data,
            telescope.map_coord, angle_it->second(0));

    const auto registry_path =
        std::filesystem::path(output_paths.obsnum_dir_name) / "raw" /
        ("beammap_direction_scan_registry" +
         citlali::pipeline::beammap_direction_registry_suffix(mode) +
         ".csv");
    citlali::pipeline::write_beammap_direction_scan_registry(
        registry_path, beammap_direction_selection);
    beammap_direction_selection_initialized = true;
    logger->info(
        "beammap direction selection mode={} left_scans={} right_scans={} selected_scans={} registry={}",
        citlali::config::to_string(mode),
        beammap_direction_selection.left_count,
        beammap_direction_selection.right_count,
        beammap_direction_selection.selected_count,
        registry_path.string());
}

citlali::pipeline::BeammapDirectionBufferSelection
Beammap::beammap_direction_buffer_selection(Eigen::Index scan_index) const {
    const auto mode =
        citlali::pipeline::beammap_config(*this).direction_mode;
    if (citlali::pipeline::beammap_direction_mode_is_standard(mode)) {
        return {true, false, false};
    }
    if (!beammap_direction_selection_initialized || scan_index < 0 ||
        scan_index >= static_cast<Eigen::Index>(
                          beammap_direction_selection.scans.size())) {
        throw std::logic_error(
            "beammap direction selection is unavailable for a mapmaking scan");
    }
    return citlali::pipeline::beammap_direction_buffer_selection(
        mode, beammap_direction_selection.scans[
                  static_cast<std::size_t>(scan_index)]
                  .direction);
}
