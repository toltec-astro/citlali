#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_masking_stats.h>
#include <citlali/core/engine/detail/beammap_rfi_mask_impl.h>
#include <citlali/core/engine/detail/beammap_scan_band_mask_impl.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

void Beammap::log_beammap_masking_config() {
    const bool detector_grouping =
        citlali::pipeline::mapmaking_config(*this).grouping ==
        citlali::config::MapGrouping::detector;
    const auto &beammap_config = citlali::pipeline::beammap_config(*this);
    const auto &rfi_config = beammap_config.rfi_mask;
    const auto &scan_band_config = beammap_config.scan_band_mask;

    if (rfi_config.enabled && detector_grouping) {
        logger->info("beammap rfi mask enabled: block_size={} min_good={} sigma_threshold={:.4g} sigma_floor={:.4g} dilate_blocks={} max_flagged_fraction={:.4f}",
                     rfi_config.block_size_samples,
                     rfi_config.min_good_samples,
                     rfi_config.sigma_threshold,
                     rfi_config.sigma_floor,
                     rfi_config.dilate_blocks,
                     rfi_config.max_flagged_fraction);
    }
    if (scan_band_config.enabled && detector_grouping) {
        logger->info(
            "beammap scan-band mask enabled: edge_rows={} min_row_pixels={} min_contiguous_rows={} row_median_sigma_threshold={:.4g} row_sigma_ratio_threshold={:.4g} max_flagged_fraction={:.4f}",
            scan_band_config.edge_rows,
            scan_band_config.min_row_pixels,
            scan_band_config.min_contiguous_rows,
            scan_band_config.row_median_sigma_threshold,
            scan_band_config.row_sigma_ratio_threshold,
            scan_band_config.max_flagged_fraction);
    }
}
