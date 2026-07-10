#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_empirical_template_utils.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

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
