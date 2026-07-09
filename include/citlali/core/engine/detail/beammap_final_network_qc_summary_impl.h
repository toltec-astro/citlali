#pragma once

// Beammap final network QC summary implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_prior_qc_stats.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <citlali/core/pipeline/reduction_config_accessors.h>

void Beammap::log_final_network_qc_summary() {
    if (citlali::pipeline::mapmaking_config(*this).grouping !=
        citlali::config::MapGrouping::detector) {
        return;
    }

    logger->info("beammap final per-network qc summary follows");
    for (Eigen::Index i = 0; i < calib.n_arrays; ++i) {
        const Eigen::Index array = calib.arrays(i);
        const std::string array_name = toltec_io.array_name_map[array];

        for (Eigen::Index j = 0; j < calib.n_nws; ++j) {
            const Eigen::Index nw = calib.nws(j);
            if (calib.nw_limits.count(nw) == 0) {
                continue;
            }
            const auto [k0, k1] = calib.nw_limits[nw];
            if (k0 < 0 || k1 <= k0) {
                continue;
            }
            if (static_cast<Eigen::Index>(std::lround(calib.apt["array"](k0))) != array) {
                continue;
            }

            std::vector<double> a_vals;
            std::vector<double> b_vals;
            std::vector<double> snr_vals;
            std::vector<double> prior_d2_vals;
            Eigen::Index n_total = 0;
            Eigen::Index n_good = 0;
            for (Eigen::Index k = k0; k < k1; ++k) {
                n_total++;
                if (calib.apt["flag"](k) != 0) {
                    continue;
                }
                n_good++;
                if (std::isfinite(calib.apt["a_fwhm"](k))) {
                    a_vals.push_back(calib.apt["a_fwhm"](k));
                }
                if (std::isfinite(calib.apt["b_fwhm"](k))) {
                    b_vals.push_back(calib.apt["b_fwhm"](k));
                }
                if (std::isfinite(calib.apt["sig2noise"](k))) {
                    snr_vals.push_back(calib.apt["sig2noise"](k));
                }
                if (final_prior_d2_diag.size() == calib.n_dets &&
                    std::isfinite(final_prior_d2_diag(k))) {
                    prior_d2_vals.push_back(final_prior_d2_diag(k));
                }
            }

            const double good_frac =
                static_cast<double>(n_good) / static_cast<double>(std::max<Eigen::Index>(1, n_total));
            logger->info(
                "beammap network qc: array={} nw={} good={}/{} ({:.3f}) med_a_fwhm={} med_b_fwhm={} med_sig2noise={} med_final_prior_d2={}",
                array_name,
                static_cast<int>(nw),
                n_good,
                n_total,
                good_frac,
                beammap_prior_qc_stats::median_or_nan(a_vals),
                beammap_prior_qc_stats::median_or_nan(b_vals),
                beammap_prior_qc_stats::median_or_nan(snr_vals),
                beammap_prior_qc_stats::median_or_nan(prior_d2_vals));
        }
    }
}
