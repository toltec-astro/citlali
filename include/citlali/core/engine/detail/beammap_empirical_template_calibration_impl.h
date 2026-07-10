#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_empirical_template_columns_impl.h>
#include <citlali/core/engine/detail/beammap_empirical_template_library_impl.h>
#include <citlali/core/engine/detail/beammap_empirical_template_utils.h>

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
