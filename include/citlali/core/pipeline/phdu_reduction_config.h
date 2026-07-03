#pragma once

#include <string>

#include <citlali/core/pipeline/phdu_telescope_values.h>

namespace citlali::pipeline {

template <class FitsEntry, class ReductionLearning, class Logger>
void add_phdu_reduction_learning_config(
    FitsEntry &fits_entry, const std::string &array_name,
    const Logger &logger, const ReductionLearning &reduction_learning) {
    auto &hdu = fits_entry.pfits->pHDU();
    auto add_double_key = [&](const std::string &key, double value,
                              const std::string &comment,
                              double fallback = 0.0) {
        add_phdu_double_key(fits_entry, array_name, logger, key, value,
                            comment, fallback);
    };
    const auto &options = reduction_learning.options;

    hdu.addKey("CONFIG.LEARNING.ENABLED", options.enabled,
               "Enable shared reduction learning state");
    hdu.addKey("CONFIG.LEARNING.DIAGNOSTICS", options.diagnostics_enabled,
               "Write shared reduction learning diagnostics");
    hdu.addKey("CONFIG.LEARNING.LEARN_ITERS", options.learn_iters,
               "Initial fruitloops iterations used for learning");
    hdu.addKey("CONFIG.LEARNING.APPLY_ITER", options.apply_start_iter,
               "Earliest fruitloops iter applying learned state");
    hdu.addKey("CONFIG.LEARNING.MAP_OUTLIER_DET_EXCL",
               options.map_pixel_outlier_detector_exclusion_enabled,
               "Enable map-outlier learned detector exclusions");
    hdu.addKey("CONFIG.LEARNING.MAP_OUTLIER_DET_MINPIX",
               options.map_pixel_outlier_detector_exclusion_min_pixels,
               "Outlier pixels needed for learned detector exclusion");
    hdu.addKey("CONFIG.LEARNING.BUSY_DET_EXCL",
               options.busy_detector_exclusion_enabled,
               "Enable PTC-busy learned detector exclusions");
    hdu.addKey("CONFIG.LEARNING.NET_PATH.ENABLED",
               options.scan_network_pathology_enabled,
               "Enable learned scan-network pathology exclusions");
    hdu.addKey("CONFIG.LEARNING.NET_PATH.PRE_RTC",
               options.scan_network_pathology_apply_pre_rtc,
               "Apply scan-network exclusions before RTC");
    hdu.addKey("CONFIG.LEARNING.NET_PATH.PRE_PTC",
               options.scan_network_pathology_apply_pre_ptc,
               "Apply scan-network exclusions before PTC");
    hdu.addKey("CONFIG.LEARNING.NET_PATH.PRE_MAP",
               options.scan_network_pathology_apply_pre_mapmaking,
               "Apply scan-network exclusions before mapmaking");
    hdu.addKey("CONFIG.LEARNING.NET_PATH.MIN_CLUST",
               options.scan_network_pathology_min_candidate_clusters,
               "Min clusters for scan-network pathology");
    hdu.addKey("CONFIG.LEARNING.NET_PATH.MIN_EV",
               options.scan_network_pathology_min_candidate_events,
               "Min events for scan-network pathology");
    add_double_key("CONFIG.LEARNING.NET_PATH.MIN_Z",
                   options.scan_network_pathology_min_max_residual_z,
                   "Min residual z for scan-network pathology");
    hdu.addKey("CONFIG.LEARNING.NET_PATH.SEV_EV",
               options.scan_network_pathology_severe_candidate_events,
               "Severe event count for scan-network pathology");
    add_double_key("CONFIG.LEARNING.NET_PATH.SEV_Z",
                   options.scan_network_pathology_severe_max_residual_z,
                   "Severe residual z for scan-network pathology");
    add_double_key("CONFIG.LEARNING.NET_PATH.MAX_FRAC",
                   options.scan_network_pathology_max_new_flagged_fraction,
                   "Max new flagged fraction for network exclusions");
    hdu.addKey("CONFIG.LEARNING.PHASE",
               reduction_learning.current_phase_name(),
               "Shared reduction learning phase");
}

}  // namespace citlali::pipeline
