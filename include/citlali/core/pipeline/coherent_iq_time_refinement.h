#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace citlali::pipeline {

inline constexpr const char *coherent_iq_time_refinement_method =
    "template_projection_boxcar_derivative_v1";

struct CoherentIqTimeRefinement {
    std::string status = "unavailable";
    std::string method = coherent_iq_time_refinement_method;
    double seed_time_unix_sec = std::numeric_limits<double>::quiet_NaN();
    double refined_time_unix_sec =
        std::numeric_limits<double>::quiet_NaN();
    double displacement_from_seed_sec =
        std::numeric_limits<double>::quiet_NaN();
    std::string primary_mode_id;
    double peak_absolute_derivative_mrad_per_sec =
        std::numeric_limits<double>::quiet_NaN();
    double derivative_snr = std::numeric_limits<double>::quiet_NaN();
    double peak_to_second_ratio =
        std::numeric_limits<double>::quiet_NaN();
    int compatible_tone_count = 0;
    int template_tone_count = 0;
    std::string note;
};

struct CoherentIqSharedTimeRefinement {
    std::string status = "unavailable";
    std::string method = "median_network_consensus_v1";
    double seed_time_unix_sec = std::numeric_limits<double>::quiet_NaN();
    double refined_time_unix_sec =
        std::numeric_limits<double>::quiet_NaN();
    double displacement_from_seed_sec =
        std::numeric_limits<double>::quiet_NaN();
    int contributing_network_count = 0;
    std::string contributing_networks;
    double contributing_network_span_sec =
        std::numeric_limits<double>::quiet_NaN();
    std::string note;
};

inline double coherent_iq_median(std::vector<double> values) {
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 == 1
               ? values[middle]
               : 0.5 * (values[middle - 1] + values[middle]);
}

inline std::vector<double> coherent_iq_centered_boxcar(
    const std::vector<double> &time_unix_sec,
    const std::vector<double> &values, double window_sec) {
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> result(values.size(), nan);
    if (time_unix_sec.size() != values.size() || values.empty() ||
        !(std::isfinite(window_sec) && window_sec > 0.0)) {
        return result;
    }
    std::vector<double> prefix_sum(values.size() + 1, 0.0);
    std::vector<int> prefix_count(values.size() + 1, 0);
    for (std::size_t index = 0; index < values.size(); ++index) {
        prefix_sum[index + 1] = prefix_sum[index];
        prefix_count[index + 1] = prefix_count[index];
        if (std::isfinite(values[index])) {
            prefix_sum[index + 1] += values[index];
            ++prefix_count[index + 1];
        }
    }
    const double half_window = 0.5 * window_sec;
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto begin = std::lower_bound(
            time_unix_sec.begin(), time_unix_sec.end(),
            time_unix_sec[index] - half_window);
        const auto end = std::upper_bound(
            time_unix_sec.begin(), time_unix_sec.end(),
            time_unix_sec[index] + half_window);
        const auto first = static_cast<std::size_t>(
            begin - time_unix_sec.begin());
        const auto last = static_cast<std::size_t>(
            end - time_unix_sec.begin());
        const int count = prefix_count[last] - prefix_count[first];
        if (count >= 3) {
            result[index] =
                (prefix_sum[last] - prefix_sum[first]) /
                static_cast<double>(count);
        }
    }
    return result;
}

inline CoherentIqTimeRefinement refine_coherent_iq_projected_event_time(
    const std::vector<double> &time_unix_sec,
    const std::vector<std::string> &mode_ids,
    const std::vector<std::vector<double>> &projected_phase_mrad,
    double seed_time_unix_sec, double search_half_width_sec,
    double smoothing_window_sec, double minimum_derivative_snr,
    double minimum_peak_ratio, double peak_exclusion_sec) {
    CoherentIqTimeRefinement result;
    result.seed_time_unix_sec = seed_time_unix_sec;
    if (time_unix_sec.size() < 7 || mode_ids.empty() ||
        projected_phase_mrad.size() != mode_ids.size() ||
        !std::isfinite(seed_time_unix_sec)) {
        result.status = "incomplete_projection_window";
        result.note = "time or projected-mode input is incomplete";
        return result;
    }
    const bool invalid_time = std::any_of(
        time_unix_sec.begin(), time_unix_sec.end(),
        [](double value) { return !std::isfinite(value); });
    const bool nonincreasing =
        std::adjacent_find(
            time_unix_sec.begin(), time_unix_sec.end(),
            [](double left, double right) { return left >= right; }) !=
        time_unix_sec.end();
    if (invalid_time || nonincreasing ||
        !(std::isfinite(search_half_width_sec) &&
          search_half_width_sec > 0.0) ||
        !(std::isfinite(smoothing_window_sec) &&
          smoothing_window_sec > 0.0)) {
        result.status = "invalid_projection_coordinate";
        result.note = "time coordinate or refinement window is invalid";
        return result;
    }

    struct Peak {
        std::size_t mode = 0;
        std::size_t sample = 0;
        double derivative = std::numeric_limits<double>::quiet_NaN();
        double snr = -std::numeric_limits<double>::infinity();
        double ratio = std::numeric_limits<double>::quiet_NaN();
    };
    std::vector<Peak> peaks;
    const double search_begin = seed_time_unix_sec - search_half_width_sec;
    const double search_end = seed_time_unix_sec + search_half_width_sec;
    for (std::size_t mode = 0; mode < mode_ids.size(); ++mode) {
        if (projected_phase_mrad[mode].size() != time_unix_sec.size()) {
            result.status = "incomplete_projection_window";
            result.note = "projected-mode length differs from time";
            return result;
        }
        const auto smooth = coherent_iq_centered_boxcar(
            time_unix_sec, projected_phase_mrad[mode],
            smoothing_window_sec);
        std::vector<std::pair<std::size_t, double>> derivatives;
        for (std::size_t sample = 1; sample + 1 < smooth.size(); ++sample) {
            if (time_unix_sec[sample] < search_begin ||
                time_unix_sec[sample] > search_end ||
                !std::isfinite(smooth[sample - 1]) ||
                !std::isfinite(smooth[sample + 1])) {
                continue;
            }
            const double delta_time =
                time_unix_sec[sample + 1] - time_unix_sec[sample - 1];
            if (!(std::isfinite(delta_time) && delta_time > 0.0)) {
                continue;
            }
            derivatives.emplace_back(
                sample,
                (smooth[sample + 1] - smooth[sample - 1]) /
                    delta_time);
        }
        if (derivatives.size() < 5) {
            continue;
        }
        const auto strongest = std::max_element(
            derivatives.begin(), derivatives.end(),
            [](const auto &left, const auto &right) {
                return std::abs(left.second) < std::abs(right.second);
            });
        std::vector<double> derivative_values;
        derivative_values.reserve(derivatives.size());
        for (const auto &[sample, derivative] : derivatives) {
            (void)sample;
            derivative_values.push_back(derivative);
        }
        const double median = coherent_iq_median(derivative_values);
        std::vector<double> deviations;
        deviations.reserve(derivative_values.size());
        for (const auto derivative : derivative_values) {
            deviations.push_back(std::abs(derivative - median));
        }
        const double robust_sigma =
            1.4826 * coherent_iq_median(std::move(deviations));
        const double peak_deviation =
            std::abs(strongest->second - median);
        const double snr = robust_sigma > 0.0
                               ? peak_deviation / robust_sigma
                               : (peak_deviation > 0.0
                                      ? std::numeric_limits<double>::max()
                                      : 0.0);
        double second = 0.0;
        for (const auto &[sample, derivative] : derivatives) {
            if (std::abs(time_unix_sec[sample] -
                         time_unix_sec[strongest->first]) >=
                peak_exclusion_sec) {
                second = std::max(second, std::abs(derivative));
            }
        }
        const double peak_absolute = std::abs(strongest->second);
        const double ratio = second > 0.0
                                 ? peak_absolute / second
                                 : (peak_absolute > 0.0
                                        ? std::numeric_limits<double>::max()
                                        : 0.0);
        peaks.push_back(
            {mode, strongest->first, strongest->second, snr, ratio});
    }
    if (peaks.empty()) {
        result.status = "incomplete_projection_window";
        result.note = "no mode has five finite derivative samples";
        return result;
    }
    const auto best = std::max_element(
        peaks.begin(), peaks.end(), [](const auto &left, const auto &right) {
            return std::tie(left.snr, left.ratio) <
                   std::tie(right.snr, right.ratio);
        });
    result.refined_time_unix_sec = time_unix_sec[best->sample];
    result.displacement_from_seed_sec =
        result.refined_time_unix_sec - seed_time_unix_sec;
    result.primary_mode_id = mode_ids[best->mode];
    result.peak_absolute_derivative_mrad_per_sec =
        std::abs(best->derivative);
    result.derivative_snr = best->snr;
    result.peak_to_second_ratio = best->ratio;
    if (result.refined_time_unix_sec <=
            search_begin + 0.5 * smoothing_window_sec ||
        result.refined_time_unix_sec >=
            search_end - 0.5 * smoothing_window_sec) {
        result.status = "boundary_peak";
        result.note = "strongest derivative is at the refinement boundary";
    } else if (result.derivative_snr < minimum_derivative_snr) {
        result.status = "low_derivative_snr";
        result.note = "strongest derivative is below the configured S/N";
    } else if (result.peak_to_second_ratio < minimum_peak_ratio) {
        result.status = "ambiguous_multiple_peaks";
        result.note = "a separated derivative peak has comparable strength";
    } else {
        result.status = "refined";
    }
    return result;
}

inline CoherentIqSharedTimeRefinement
consolidate_coherent_iq_time_refinements(
    double seed_time_unix_sec,
    const std::vector<std::pair<int, CoherentIqTimeRefinement>> &network_results,
    int minimum_networks, double consensus_tolerance_sec) {
    CoherentIqSharedTimeRefinement result;
    result.seed_time_unix_sec = seed_time_unix_sec;
    std::vector<std::pair<int, double>> accepted;
    for (const auto &[network, refinement] : network_results) {
        if (refinement.status == "refined" &&
            std::isfinite(refinement.refined_time_unix_sec)) {
            accepted.emplace_back(network, refinement.refined_time_unix_sec);
        }
    }
    if (static_cast<int>(accepted.size()) < minimum_networks) {
        result.status = "insufficient_network_support";
        result.note = "too few networks have an unambiguous local transition";
        return result;
    }
    std::vector<double> times;
    for (const auto &[network, time] : accepted) {
        (void)network;
        times.push_back(time);
    }
    const double first_center = coherent_iq_median(times);
    std::vector<std::pair<int, double>> inliers;
    for (const auto &entry : accepted) {
        if (std::abs(entry.second - first_center) <=
            consensus_tolerance_sec) {
            inliers.push_back(entry);
        }
    }
    if (static_cast<int>(inliers.size()) < minimum_networks) {
        result.status = "inconsistent_network_times";
        result.note = "network transition times lack configured consensus";
        return result;
    }
    times.clear();
    for (const auto &[network, time] : inliers) {
        (void)network;
        times.push_back(time);
    }
    result.refined_time_unix_sec = coherent_iq_median(times);
    result.displacement_from_seed_sec =
        result.refined_time_unix_sec - seed_time_unix_sec;
    result.contributing_network_count = static_cast<int>(inliers.size());
    result.contributing_network_span_sec =
        *std::max_element(times.begin(), times.end()) -
        *std::min_element(times.begin(), times.end());
    std::sort(inliers.begin(), inliers.end());
    for (const auto &[network, time] : inliers) {
        (void)time;
        if (!result.contributing_networks.empty()) {
            result.contributing_networks += " ";
        }
        result.contributing_networks += std::to_string(network);
    }
    result.status = "refined";
    return result;
}

}  // namespace citlali::pipeline
