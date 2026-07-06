#pragma once

// Beammap empirical-template calibration APT column schema.

namespace beammap_empirical_template_schema {

struct CalibrationColumnSpec {
    const char *name;
    const char *unit;
    const char *description;
    double fill_value;
};

inline std::vector<CalibrationColumnSpec> calibration_columns() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {
        {"cal_amp", "xs",
         "amplitude used for beammap flux calibration; empirical template when valid, Gaussian fallback otherwise",
         nan},
        {"cal_amp_method", "N/A",
         "calibration amplitude method code (0 Gaussian fallback, 1 empirical template)",
         0.0},
        {"template_amp", "xs",
         "empirical array-template matched amplitude with fitted local offset",
         nan},
        {"template_offset", "xs",
         "local offset fitted with empirical array-template amplitude",
         nan},
        {"template_resid_rms", "xs",
         "weighted residual RMS for empirical-template amplitude fit",
         nan},
        {"template_npix", "pix",
         "number of pixels used for empirical-template amplitude fit",
         0.0},
        {"template_amp_over_fit_amp", "N/A",
         "ratio of empirical-template matched amplitude to Gaussian fit amplitude",
         nan},
        {"cal_amp_over_fit_amp", "N/A",
         "ratio of calibration amplitude to Gaussian fit amplitude",
         nan},
        {"map_peak_amp", "xs",
         "baseline-subtracted local map peak within 8 arcsec of Gaussian fit center",
         nan},
        {"map_peak_amp_over_fit_amp", "N/A",
         "ratio of local map peak amplitude to Gaussian fit amplitude",
         nan}
    };
}

inline void append_cal_amp_method_legend(YAML::Node &apt_meta) {
    apt_meta["cal_amp_method"].push_back("0: Gaussian fit amplitude fallback");
    apt_meta["cal_amp_method"].push_back("1: empirical array-template matched amplitude");
}

} // namespace beammap_empirical_template_schema
