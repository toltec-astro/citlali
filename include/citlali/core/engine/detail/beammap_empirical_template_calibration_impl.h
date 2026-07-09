#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_empirical_template_schema.h>
#include <citlali/core/engine/detail/beammap_empirical_template_utils.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

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

Beammap::BeammapEmpiricalTemplateGeometry
Beammap::make_empirical_template_geometry(double pix_to_arcsec) const {
    BeammapEmpiricalTemplateGeometry geometry;
    geometry.template_radius_pix =
        std::max<Eigen::Index>(8, static_cast<Eigen::Index>(std::ceil(40.0 / pix_to_arcsec)));
    geometry.match_radius_pix =
        std::max<Eigen::Index>(4, static_cast<Eigen::Index>(std::ceil(35.0 / pix_to_arcsec)));
    geometry.peak_radius_pix =
        std::max<Eigen::Index>(2, static_cast<Eigen::Index>(std::ceil(8.0 / pix_to_arcsec)));
    geometry.template_peak_radius_pix =
        std::max<Eigen::Index>(1, static_cast<Eigen::Index>(std::ceil(4.0 / pix_to_arcsec)));
    geometry.side = 2 * geometry.template_radius_pix + 1;
    geometry.center = geometry.template_radius_pix;
    return geometry;
}

bool Beammap::empirical_template_inputs_available() const {
    return citlali::pipeline::mapmaking_config(*this).grouping ==
               citlali::config::MapGrouping::detector &&
           static_cast<Eigen::Index>(omb.signal.size()) == map_indices.n_maps &&
           static_cast<Eigen::Index>(omb.weight.size()) == map_indices.n_maps &&
           omb.pixel_size_rad > 0.0;
}

bool Beammap::extract_empirical_template_normalized_cut(
    Eigen::Index map_index,
    const Beammap::BeammapEmpiricalTemplateGeometry &geometry,
    Eigen::MatrixXd &cut,
    double &peak_amp) {
    if (map_index < 0 || map_index >= map_indices.n_maps ||
        map_index >= params.rows() || params.cols() < 3) {
        return false;
    }
    const double amp = params(map_index, 0);
    const double col0 = params(map_index, 1);
    const double row0 = params(map_index, 2);
    if (!std::isfinite(amp) || amp <= 0.0 ||
        !std::isfinite(row0) || !std::isfinite(col0)) {
        return false;
    }
    return beammap_empirical_template_utils::extract_normalized_cut(
        omb.signal[map_index], row0, col0, geometry.template_radius_pix,
        geometry.peak_radius_pix, cut, peak_amp);
}

bool Beammap::is_empirical_template_library_detector(Eigen::Index map_index, int array) {
    if (map_index < 0 || map_index >= map_indices.n_maps ||
        map_index >= calib.n_dets || calib.apt["flag"](map_index) != 0 ||
        !good_fits(map_index)) {
        return false;
    }
    if (static_cast<int>(std::lround(calib.apt["array"](map_index))) != array) {
        return false;
    }

    const double a = calib.apt["a_fwhm"](map_index);
    const double b = calib.apt["b_fwhm"](map_index);
    return std::isfinite(a) && std::isfinite(b) && a > 0.0 && b > 0.0;
}

Beammap::BeammapEmpiricalTemplateShapeMedians
Beammap::empirical_template_shape_medians(int array) {
    std::vector<double> a_values;
    std::vector<double> b_values;
    for (Eigen::Index i = 0; i < map_indices.n_maps; ++i) {
        if (!is_empirical_template_library_detector(i, array)) {
            continue;
        }
        a_values.push_back(calib.apt["a_fwhm"](i));
        b_values.push_back(calib.apt["b_fwhm"](i));
    }

    BeammapEmpiricalTemplateShapeMedians medians;
    medians.a_fwhm = beammap_empirical_template_utils::median_finite(a_values);
    medians.b_fwhm = beammap_empirical_template_utils::median_finite(b_values);
    medians.valid = std::isfinite(medians.a_fwhm) && std::isfinite(medians.b_fwhm) &&
                    medians.a_fwhm > 0.0 && medians.b_fwhm > 0.0;
    return medians;
}

std::vector<Beammap::BeammapEmpiricalTemplateCandidate>
Beammap::collect_empirical_template_candidates(
    int array,
    const Beammap::BeammapEmpiricalTemplateShapeMedians &shape_medians) {
    constexpr double min_template_snr = 20.0;
    std::vector<BeammapEmpiricalTemplateCandidate> candidates;
    if (!shape_medians.valid) {
        return candidates;
    }

    for (Eigen::Index i = 0; i < map_indices.n_maps; ++i) {
        if (!is_empirical_template_library_detector(i, array)) {
            continue;
        }
        if (fit_diag_bound_nhit.size() == map_indices.n_maps && fit_diag_bound_nhit(i) > 0) {
            continue;
        }
        const double rms = calc_map_support_stddev(i, true);
        const double snr = (std::isfinite(rms) && rms > 0.0 && std::isfinite(params(i, 0)))
                               ? params(i, 0) / rms
                               : 0.0;
        if (!std::isfinite(snr) || snr < min_template_snr) {
            continue;
        }

        const double a = calib.apt["a_fwhm"](i);
        const double b = calib.apt["b_fwhm"](i);
        const double shape_score =
            std::abs(a - shape_medians.a_fwhm) / shape_medians.a_fwhm +
            std::abs(b - shape_medians.b_fwhm) / shape_medians.b_fwhm;
        candidates.push_back({i, shape_score, snr});
    }

    return candidates;
}

std::vector<Eigen::MatrixXd> Beammap::collect_empirical_template_cuts(
    const std::vector<Beammap::BeammapEmpiricalTemplateCandidate> &candidates,
    const Beammap::BeammapEmpiricalTemplateGeometry &geometry) {
    std::vector<Eigen::MatrixXd> cuts;
    cuts.reserve(candidates.size());
    for (const auto &candidate : candidates) {
        Eigen::MatrixXd cut;
        double peak_amp = std::numeric_limits<double>::quiet_NaN();
        if (extract_empirical_template_normalized_cut(
                candidate.map_index, geometry, cut, peak_amp)) {
            cuts.push_back(std::move(cut));
        }
    }
    return cuts;
}

Eigen::MatrixXd Beammap::median_empirical_template_shape(
    const std::vector<Eigen::MatrixXd> &cuts,
    const Beammap::BeammapEmpiricalTemplateGeometry &geometry) const {
    Eigen::MatrixXd templ(geometry.side, geometry.side);
    templ.setConstant(std::numeric_limits<double>::quiet_NaN());

    std::vector<double> values;
    values.reserve(cuts.size());
    for (Eigen::Index rr = 0; rr < geometry.side; ++rr) {
        for (Eigen::Index cc = 0; cc < geometry.side; ++cc) {
            values.clear();
            for (const auto &cut : cuts) {
                const double value = cut(rr, cc);
                if (std::isfinite(value)) {
                    values.push_back(value);
                }
            }
            if (!values.empty()) {
                templ(rr, cc) = beammap_empirical_template_utils::median_finite(values);
            }
        }
    }

    return templ;
}

double Beammap::empirical_template_peak_value(
    const Eigen::MatrixXd &templ,
    const Beammap::BeammapEmpiricalTemplateGeometry &geometry) const {
    double template_peak = -std::numeric_limits<double>::infinity();
    for (Eigen::Index rr = 0; rr < geometry.side; ++rr) {
        const Eigen::Index dr = rr - geometry.center;
        for (Eigen::Index cc = 0; cc < geometry.side; ++cc) {
            const Eigen::Index dc = cc - geometry.center;
            if (dr * dr + dc * dc >
                geometry.template_peak_radius_pix * geometry.template_peak_radius_pix) {
                continue;
            }
            const double value = templ(rr, cc);
            if (std::isfinite(value)) {
                template_peak = std::max(template_peak, value);
            }
        }
    }
    return template_peak;
}

std::map<int, Beammap::BeammapArrayTemplate>
Beammap::build_empirical_template_library(
    const Beammap::BeammapEmpiricalTemplateGeometry &geometry) {
    constexpr Eigen::Index min_template_detectors = 25;
    constexpr Eigen::Index max_template_detectors = 500;

    std::map<int, BeammapArrayTemplate> templates;

    for (Eigen::Index arr_i = 0; arr_i < calib.n_arrays; ++arr_i) {
        const int array = static_cast<int>(calib.arrays(arr_i));
        const auto shape_medians = empirical_template_shape_medians(array);
        if (!shape_medians.valid) {
            continue;
        }

        auto candidates = collect_empirical_template_candidates(array, shape_medians);
        std::sort(candidates.begin(), candidates.end(),
                  [](const BeammapEmpiricalTemplateCandidate &lhs,
                     const BeammapEmpiricalTemplateCandidate &rhs) {
                      if (lhs.shape_score == rhs.shape_score) {
                          return lhs.snr > rhs.snr;
                      }
                      return lhs.shape_score < rhs.shape_score;
                  });
        if (static_cast<Eigen::Index>(candidates.size()) > max_template_detectors) {
            candidates.resize(static_cast<std::size_t>(max_template_detectors));
        }

        const auto cuts = collect_empirical_template_cuts(candidates, geometry);

        if (static_cast<Eigen::Index>(cuts.size()) < min_template_detectors) {
            logger->warn(
                "beammap empirical template skipped for array={} candidates={} usable={} min_required={}",
                toltec_io.array_name_map[array], candidates.size(), cuts.size(), min_template_detectors);
            continue;
        }

        auto templ = median_empirical_template_shape(cuts, geometry);
        const double template_peak = empirical_template_peak_value(templ, geometry);
        if (!std::isfinite(template_peak) || template_peak <= 0.0) {
            logger->warn("beammap empirical template skipped for array={} due to invalid template peak={:.4g}",
                         toltec_io.array_name_map[array], template_peak);
            continue;
        }
        templ.array() /= template_peak;
        templates[array] = {true, std::move(templ), static_cast<Eigen::Index>(cuts.size())};
        logger->info("beammap empirical template built for array={} using {} detectors",
                     toltec_io.array_name_map[array], cuts.size());
    }

    return templates;
}

void Beammap::record_empirical_template_peak(
    Eigen::Index map_index,
    double row0,
    double col0,
    double baseline,
    const Beammap::BeammapEmpiricalTemplateGeometry &geometry) {
    const double peak_amp =
        beammap_empirical_template_utils::local_peak(
            omb.signal[map_index], row0, col0, baseline, geometry.peak_radius_pix);
    if (std::isfinite(peak_amp)) {
        calib.apt["map_peak_amp"](map_index) = peak_amp;
        if (std::isfinite(params(map_index, 0)) && params(map_index, 0) > 0.0) {
            calib.apt["map_peak_amp_over_fit_amp"](map_index) = peak_amp / params(map_index, 0);
        }
    }
}

Beammap::BeammapTemplateFitSamples Beammap::collect_empirical_template_fit_samples(
    Eigen::Index map_index,
    const Eigen::MatrixXd &templ,
    double row0,
    double col0,
    double baseline,
    const Beammap::BeammapEmpiricalTemplateGeometry &geometry) {
    constexpr double min_template_value = 0.015;
    BeammapTemplateFitSamples samples;
    samples.y.reserve(static_cast<std::size_t>(geometry.side * geometry.side));
    samples.t.reserve(static_cast<std::size_t>(geometry.side * geometry.side));
    samples.w.reserve(static_cast<std::size_t>(geometry.side * geometry.side));

    for (Eigen::Index rr = 0; rr < geometry.side; ++rr) {
        const Eigen::Index dr = rr - geometry.center;
        for (Eigen::Index cc = 0; cc < geometry.side; ++cc) {
            const Eigen::Index dc = cc - geometry.center;
            if (dr * dr + dc * dc > geometry.match_radius_pix * geometry.match_radius_pix) {
                continue;
            }
            const double template_value = templ(rr, cc);
            if (!std::isfinite(template_value) || std::abs(template_value) <= min_template_value) {
                continue;
            }
            const double row = row0 + static_cast<double>(dr);
            const double col = col0 + static_cast<double>(dc);
            const double signal_value = beammap_empirical_template_utils::bilinear_sample(
                omb.signal[map_index], row, col);
            const double weight_value = beammap_empirical_template_utils::bilinear_sample(
                omb.weight[map_index], row, col);
            if (!std::isfinite(signal_value) || !std::isfinite(weight_value) || weight_value <= 0.0) {
                continue;
            }
            samples.y.push_back(signal_value - baseline);
            samples.t.push_back(template_value);
            samples.w.push_back(weight_value);
        }
    }

    return samples;
}

double Beammap::empirical_template_weight_cap(const std::vector<double> &weights) const {
    std::vector<double> w_sorted = weights;
    std::sort(w_sorted.begin(), w_sorted.end());
    const std::size_t q95_index =
        std::min<std::size_t>(w_sorted.size() - 1,
                              static_cast<std::size_t>(std::floor(0.95 * (w_sorted.size() - 1))));
    return w_sorted[q95_index];
}

bool Beammap::solve_empirical_template_linear_fit(
    const Beammap::BeammapTemplateFitSamples &samples,
    double weight_cap,
    Beammap::BeammapTemplateFitResult &fit_result) const {
    double sw = 0.0;
    double st = 0.0;
    double sy = 0.0;
    double stt = 0.0;
    double sty = 0.0;
    for (std::size_t k = 0; k < samples.y.size(); ++k) {
        const double wk = (std::isfinite(weight_cap) && weight_cap > 0.0)
                              ? std::min(samples.w[k], weight_cap)
                              : samples.w[k];
        sw += wk;
        st += wk * samples.t[k];
        sy += wk * samples.y[k];
        stt += wk * samples.t[k] * samples.t[k];
        sty += wk * samples.t[k] * samples.y[k];
    }
    const double det = stt * sw - st * st;
    if (!std::isfinite(det) || std::abs(det) <= std::numeric_limits<double>::epsilon()) {
        return false;
    }
    fit_result.amp = (sty * sw - sy * st) / det;
    fit_result.offset = (stt * sy - st * sty) / det;
    return std::isfinite(fit_result.amp) && std::isfinite(fit_result.offset) &&
           fit_result.amp > 0.0;
}

double Beammap::empirical_template_resid_rms(
    const Beammap::BeammapTemplateFitSamples &samples,
    double weight_cap,
    const Beammap::BeammapTemplateFitResult &fit_result) const {
    double resid2_sum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t k = 0; k < samples.y.size(); ++k) {
        const double wk = (std::isfinite(weight_cap) && weight_cap > 0.0)
                              ? std::min(samples.w[k], weight_cap)
                              : samples.w[k];
        const double resid = samples.y[k] - (fit_result.amp * samples.t[k] + fit_result.offset);
        resid2_sum += wk * resid * resid;
        weight_sum += wk;
    }
    if (weight_sum > 0.0) {
        return std::sqrt(resid2_sum / weight_sum);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

bool Beammap::solve_empirical_template(
    Eigen::Index map_index,
    const Eigen::MatrixXd &templ,
    const Beammap::BeammapEmpiricalTemplateGeometry &geometry,
    Beammap::BeammapTemplateFitResult &fit_result) {
    fit_result = BeammapTemplateFitResult{};
    if (map_index < 0 || map_index >= map_indices.n_maps ||
        map_index >= params.rows() || params.cols() < 3) {
        return false;
    }
    const double col0 = params(map_index, 1);
    const double row0 = params(map_index, 2);
    if (!std::isfinite(row0) || !std::isfinite(col0)) {
        return false;
    }
    const double baseline =
        beammap_empirical_template_utils::edge_baseline(
            omb.signal[map_index], row0, col0, geometry.template_radius_pix);
    if (!std::isfinite(baseline)) {
        return false;
    }
    record_empirical_template_peak(map_index, row0, col0, baseline, geometry);

    const auto samples =
        collect_empirical_template_fit_samples(map_index, templ, row0, col0, baseline, geometry);
    fit_result.npix = static_cast<Eigen::Index>(samples.y.size());
    if (fit_result.npix < 30) {
        return false;
    }

    const double w_cap = empirical_template_weight_cap(samples.w);
    if (!solve_empirical_template_linear_fit(samples, w_cap, fit_result)) {
        return false;
    }

    fit_result.resid_rms = empirical_template_resid_rms(samples, w_cap, fit_result);
    fit_result.valid = std::isfinite(fit_result.resid_rms);
    return fit_result.valid;
}

double Beammap::seed_empirical_template_detector_calibration(Eigen::Index map_index) {
    const double fit_amp = params(map_index, 0);
    calib.apt["cal_amp"](map_index) = fit_amp;
    calib.apt["cal_amp_method"](map_index) = 0.0;
    if (std::isfinite(fit_amp) && fit_amp > 0.0) {
        calib.apt["cal_amp_over_fit_amp"](map_index) = 1.0;
    }
    return fit_amp;
}

void Beammap::record_empirical_template_fit_result(
    Eigen::Index map_index,
    double fit_amp,
    const Beammap::BeammapTemplateFitResult &fit_result) {
    calib.apt["template_amp"](map_index) = fit_result.amp;
    calib.apt["template_offset"](map_index) = fit_result.offset;
    calib.apt["template_resid_rms"](map_index) = fit_result.resid_rms;
    calib.apt["template_npix"](map_index) = static_cast<double>(fit_result.npix);
    if (std::isfinite(fit_amp) && fit_amp > 0.0) {
        calib.apt["template_amp_over_fit_amp"](map_index) = fit_result.amp / fit_amp;
    }

    calib.apt["cal_amp"](map_index) = fit_result.amp;
    calib.apt["cal_amp_method"](map_index) = 1.0;
    if (std::isfinite(fit_amp) && fit_amp > 0.0) {
        calib.apt["cal_amp_over_fit_amp"](map_index) = fit_result.amp / fit_amp;
    }
}

bool Beammap::apply_empirical_template_detector_calibration(
    Eigen::Index map_index,
    const std::map<int, Beammap::BeammapArrayTemplate> &templates,
    const Beammap::BeammapEmpiricalTemplateGeometry &geometry) {
    const int array = static_cast<int>(std::lround(calib.apt["array"](map_index)));
    const double fit_amp = seed_empirical_template_detector_calibration(map_index);

    auto templ_it = templates.find(array);
    if (templ_it == templates.end() || !templ_it->second.valid) {
        return false;
    }

    BeammapTemplateFitResult fit_result;
    if (!solve_empirical_template(map_index, templ_it->second.shape, geometry, fit_result)) {
        return false;
    }

    record_empirical_template_fit_result(map_index, fit_amp, fit_result);
    return true;
}

void Beammap::apply_empirical_template_calibration(
    const std::map<int, Beammap::BeammapArrayTemplate> &templates,
    const Beammap::BeammapEmpiricalTemplateGeometry &geometry) {
    Eigen::Index n_template_amp = 0;
    Eigen::Index n_template_fallback = 0;
    for (Eigen::Index i = 0; i < map_indices.n_maps; ++i) {
        if (i >= calib.n_dets) {
            continue;
        }
        if (!apply_empirical_template_detector_calibration(i, templates, geometry)) {
            n_template_fallback++;
            continue;
        }
        n_template_amp++;
    }

    logger->info("beammap empirical-template calibration amplitudes: template={} fallback={}",
                 n_template_amp, n_template_fallback);
}

void Beammap::calc_empirical_template_calibration() {
    reset_empirical_template_calibration_columns();

    const Eigen::Index n_fallback =
        std::min<Eigen::Index>(map_indices.n_maps, std::min(calib.n_dets, params.rows()));
    seed_empirical_template_gaussian_fallback(n_fallback);

    if (!empirical_template_inputs_available()) {
        return;
    }

    const double pix_to_arcsec = RAD_TO_ASEC * omb.pixel_size_rad;
    if (!std::isfinite(pix_to_arcsec) || pix_to_arcsec <= 0.0) {
        return;
    }

    const auto geometry = make_empirical_template_geometry(pix_to_arcsec);
    const auto templates = build_empirical_template_library(geometry);
    apply_empirical_template_calibration(templates, geometry);
}
