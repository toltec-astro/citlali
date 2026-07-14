#pragma once

#include <citlali/core/config/config_error.h>
#include <citlali/core/config/post_processing_config.h>

namespace citlali::config {

inline void validate(const MapFilterEdgeGuardConfig &config,
                     ValidationReport &report) {
    check_minimum(config.hits_core_fraction, 0.0,
                  {"post_processing", "map_filtering", "edge_guard",
                   "hits_core_fraction"},
                  report);
    check_minimum(config.guard_radius_fwhm, 0.0,
                  {"post_processing", "map_filtering", "edge_guard",
                   "guard_radius_fwhm"},
                  report);
    check_minimum(config.taper_min_fraction, 0.0,
                  {"post_processing", "map_filtering", "edge_guard",
                   "taper_min_fraction"},
                  report);
    check_maximum(config.taper_min_fraction, 1.0,
                  {"post_processing", "map_filtering", "edge_guard",
                   "taper_min_fraction"},
                  report);
}

inline void validate(const MapFilterConfig &config, ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    validate(config.edge_guard, report);
    check_minimum(config.denom_rel_tol, 0.0,
                  {"wiener_filter", "denom_rel_tol"}, report);
    check_maximum(config.denom_rel_tol, 1.0,
                  {"wiener_filter", "denom_rel_tol"}, report);
    check_minimum(config.tail_frac_tol, 0.0,
                  {"wiener_filter", "tail_frac_tol"}, report);
    check_maximum(config.tail_frac_tol, 1.0,
                  {"wiener_filter", "tail_frac_tol"}, report);
    check_minimum(config.max_loops, 1, {"wiener_filter", "max_loops"}, report);
    check_minimum(config.denom_check_iters, 0,
                  {"wiener_filter", "denom_check_iters"}, report);
    check_minimum(config.max_denom_iters, 0,
                  {"wiener_filter", "max_denom_iters"}, report);
    for (const auto &[array_name, fwhm_arcsec] : config.template_fwhm_arcsec) {
        if (array_name.empty()) {
            report.add_error({"wiener_filter", "template_fwhm_arcsec"},
                             "array name must not be empty");
        }
        check_greater_than(fwhm_arcsec, 0.0,
                           {"wiener_filter", "template_fwhm_arcsec"},
                           report);
    }
}

inline void validate(const SourceFittingConfig &config, ValidationReport &report) {
    if (!config.active) {
        return;
    }
    check_minimum(config.bounding_box_arcsec, 0.0,
                  {"post_processing", "source_fitting", "bounding_box_arcsec"},
                  report);
}

inline void validate(const SourceFindingConfig &config, ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    check_minimum(config.source_sigma, 0.0,
                  {"post_processing", "source_finding", "source_sigma"},
                  report);
    check_minimum(config.source_window_arcsec, 0.0,
                  {"post_processing", "source_finding", "source_window_arcsec"},
                  report);
    if (config.mode.empty()) {
        report.add_error({"post_processing", "source_finding", "mode"},
                         "must not be empty");
    }
}

inline void validate(const PostProcessingConfig &config, ValidationReport &report) {
    validate(config.map_filtering, report);
    check_minimum(config.map_histogram_n_bins, 0,
                  {"post_processing", "map_histogram_n_bins"}, report);
    validate(config.source_finding, report);
    if (config.source_finding.enabled && !config.map_filtering.enabled) {
        report.add_error(
            {"post_processing", "source_finding", "enabled"},
            "requires post_processing.map_filtering.enabled=true");
    }
    validate(config.source_fitting, report);
}

}  // namespace citlali::config
