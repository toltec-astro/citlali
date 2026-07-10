#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_empirical_template_schema.h>

void Beammap::init_empirical_template_calibration_columns() {
    auto add_column = [&](const std::string &name,
                          const std::string &unit,
                          const std::string &description,
                          double fill_value) {
        calib.apt[name] = Eigen::VectorXd::Constant(calib.n_dets, fill_value);
        calib.apt_header_units[name] = unit;
        if (std::find(calib.apt_header_keys.begin(), calib.apt_header_keys.end(), name) ==
            calib.apt_header_keys.end()) {
            calib.apt_header_keys.push_back(name);
        }
        calib.apt_header_description[name] = description;
        calib.apt_meta[name].push_back("units: " + unit);
        calib.apt_meta[name].push_back(description);
    };

    for (const auto &column : beammap_empirical_template_schema::calibration_columns()) {
        add_column(column.name, column.unit, column.description, column.fill_value);
    }
    beammap_empirical_template_schema::append_cal_amp_method_legend(calib.apt_meta);
}

void Beammap::reset_empirical_template_calibration_columns() {
    auto ensure_column = [&](const std::string &name, double fill_value) {
        if (calib.apt.find(name) == calib.apt.end() ||
            calib.apt[name].size() != calib.n_dets) {
            calib.apt[name] = Eigen::VectorXd::Constant(calib.n_dets, fill_value);
        }
        else {
            calib.apt[name].setConstant(fill_value);
        }
    };

    ensure_column("cal_amp", std::numeric_limits<double>::quiet_NaN());
    ensure_column("cal_amp_method", 0.0);
    ensure_column("template_amp", std::numeric_limits<double>::quiet_NaN());
    ensure_column("template_offset", std::numeric_limits<double>::quiet_NaN());
    ensure_column("template_resid_rms", std::numeric_limits<double>::quiet_NaN());
    ensure_column("template_npix", 0.0);
    ensure_column("template_amp_over_fit_amp", std::numeric_limits<double>::quiet_NaN());
    ensure_column("cal_amp_over_fit_amp", std::numeric_limits<double>::quiet_NaN());
    ensure_column("map_peak_amp", std::numeric_limits<double>::quiet_NaN());
    ensure_column("map_peak_amp_over_fit_amp", std::numeric_limits<double>::quiet_NaN());
}

void Beammap::seed_empirical_template_gaussian_fallback(Eigen::Index n_fallback) {
    if (params.cols() <= 0) {
        return;
    }

    for (Eigen::Index i = 0; i < n_fallback; ++i) {
        const double fit_amp = params(i, 0);
        calib.apt["cal_amp"](i) = fit_amp;
        calib.apt["cal_amp_method"](i) = 0.0;
        if (std::isfinite(fit_amp) && fit_amp > 0.0) {
            calib.apt["cal_amp_over_fit_amp"](i) = 1.0;
        }
    }
}
