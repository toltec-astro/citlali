#pragma once

#include <citlali/core/config/beammap_config.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace citlali::pipeline {

enum class BeammapScanDirection { left, right };

inline const char *to_string(BeammapScanDirection direction) noexcept {
    return direction == BeammapScanDirection::left ? "left" : "right";
}

struct BeammapDirectionScanRecord {
    Eigen::Index scan_index = -1;
    Eigen::Index science_start = -1;
    Eigen::Index science_stop_exclusive = -1;
    Eigen::Index sample_count = 0;
    double start_time_sec = std::numeric_limits<double>::quiet_NaN();
    double stop_time_sec = std::numeric_limits<double>::quiet_NaN();
    double duration_sec = std::numeric_limits<double>::quiet_NaN();
    double fast_axis_displacement_rad =
        std::numeric_limits<double>::quiet_NaN();
    double signed_fast_axis_rate_rad_per_sec =
        std::numeric_limits<double>::quiet_NaN();
    double same_sign_step_fraction =
        std::numeric_limits<double>::quiet_NaN();
    BeammapScanDirection direction = BeammapScanDirection::left;
    bool selected = false;
};

struct BeammapDirectionSelectionPlan {
    citlali::config::BeammapDirectionMode mode =
        citlali::config::BeammapDirectionMode::standard;
    std::string coordinate_x_key;
    std::string coordinate_y_key;
    double scan_angle_rad = std::numeric_limits<double>::quiet_NaN();
    std::vector<BeammapDirectionScanRecord> scans;
    Eigen::Index left_count = 0;
    Eigen::Index right_count = 0;
    Eigen::Index selected_count = 0;
};

inline bool beammap_direction_mode_is_standard(
    citlali::config::BeammapDirectionMode mode) noexcept {
    return mode == citlali::config::BeammapDirectionMode::standard;
}

inline bool beammap_direction_mode_is_all(
    citlali::config::BeammapDirectionMode mode) noexcept {
    return mode == citlali::config::BeammapDirectionMode::all;
}

inline bool beammap_direction_selects(
    citlali::config::BeammapDirectionMode mode,
    BeammapScanDirection direction) noexcept {
    switch (mode) {
        case citlali::config::BeammapDirectionMode::standard:
        case citlali::config::BeammapDirectionMode::all:
            return true;
        case citlali::config::BeammapDirectionMode::left:
            return direction == BeammapScanDirection::left;
        case citlali::config::BeammapDirectionMode::right:
            return direction == BeammapScanDirection::right;
    }
    return false;
}

struct BeammapDirectionBufferSelection {
    bool standard = false;
    bool left = false;
    bool right = false;
};

inline BeammapDirectionBufferSelection beammap_direction_buffer_selection(
    citlali::config::BeammapDirectionMode mode,
    BeammapScanDirection direction) noexcept {
    switch (mode) {
        case citlali::config::BeammapDirectionMode::standard:
            return {true, false, false};
        case citlali::config::BeammapDirectionMode::left:
            return {direction == BeammapScanDirection::left, false, false};
        case citlali::config::BeammapDirectionMode::right:
            return {direction == BeammapScanDirection::right, false, false};
        case citlali::config::BeammapDirectionMode::all:
            return {true, direction == BeammapScanDirection::left,
                    direction == BeammapScanDirection::right};
    }
    return {};
}

inline std::pair<std::string, std::string>
beammap_direction_coordinate_keys(const std::string &map_coord) {
    std::string lower = map_coord;
    std::transform(
        lower.begin(), lower.end(), lower.begin(),
        [](unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
    if (lower == "ra" || lower == "dec") {
        return {"ra_phys", "dec_phys"};
    }
    if (lower == "az" || lower == "el" || lower == "alt") {
        return {"az_phys", "alt_phys"};
    }
    if (lower == "gal" || lower == "l" || lower == "b") {
        return {"l_phys", "b_phys"};
    }
    throw std::runtime_error(
        "beammap direction selection requires a supported raster map coordinate");
}

template <class TelData>
const auto &require_beammap_direction_series(
    const TelData &tel_data, const std::string &key,
    Eigen::Index required_size) {
    const auto found = tel_data.find(key);
    if (found == tel_data.end() || found->second.size() != required_size) {
        throw std::runtime_error(
            "beammap direction selection lacks complete telescope series '" +
            key + "'");
    }
    return found->second;
}

template <class ScanIndices, class TelData>
BeammapDirectionSelectionPlan make_beammap_direction_selection_plan(
    citlali::config::BeammapDirectionMode mode,
    const ScanIndices &scan_indices, const TelData &tel_data,
    const std::string &map_coord, double scan_angle_rad) {
    if (beammap_direction_mode_is_standard(mode)) {
        return {mode};
    }
    if (scan_indices.rows() < 2 || scan_indices.cols() <= 0) {
        throw std::runtime_error(
            "beammap direction selection requires nonempty science scan windows");
    }
    if (!std::isfinite(scan_angle_rad)) {
        throw std::runtime_error(
            "beammap direction selection requires finite Header.Map.ScanAngle");
    }

    const auto coordinate_keys =
        beammap_direction_coordinate_keys(map_coord);
    const auto time_it = tel_data.find("TelTime");
    if (time_it == tel_data.end() || time_it->second.size() <= 0) {
        throw std::runtime_error(
            "beammap direction selection lacks telescope series 'TelTime'");
    }
    const Eigen::Index total_samples = time_it->second.size();
    const auto &time = require_beammap_direction_series(
        tel_data, "TelTime", total_samples);
    const auto &coordinate_x = require_beammap_direction_series(
        tel_data, coordinate_keys.first, total_samples);
    const auto &coordinate_y = require_beammap_direction_series(
        tel_data, coordinate_keys.second, total_samples);

    BeammapDirectionSelectionPlan plan;
    plan.mode = mode;
    plan.coordinate_x_key = coordinate_keys.first;
    plan.coordinate_y_key = coordinate_keys.second;
    plan.scan_angle_rad = scan_angle_rad;
    plan.scans.reserve(static_cast<std::size_t>(scan_indices.cols()));

    const double axis_x = std::cos(scan_angle_rad);
    const double axis_y = std::sin(scan_angle_rad);
    for (Eigen::Index scan = 0; scan < scan_indices.cols(); ++scan) {
        const Eigen::Index start = scan_indices(0, scan);
        const Eigen::Index inclusive_stop = scan_indices(1, scan);
        if (start < 0 || inclusive_stop <= start ||
            inclusive_stop >= total_samples) {
            throw std::runtime_error(
                "beammap direction selection found an invalid science scan window");
        }
        const Eigen::Index count = inclusive_stop - start + 1;
        const double t0 = time(start);
        const double t1 = time(inclusive_stop);
        if (!std::isfinite(t0) || !std::isfinite(t1) || !(t1 > t0)) {
            throw std::runtime_error(
                "beammap direction selection found invalid scan time support");
        }

        double mean_t = 0.0;
        double mean_p = 0.0;
        double max_abs_p = 0.0;
        for (Eigen::Index index = start; index <= inclusive_stop; ++index) {
            const double t = time(index) - t0;
            const double x = coordinate_x(index);
            const double y = coordinate_y(index);
            if (!std::isfinite(t) || !std::isfinite(x) ||
                !std::isfinite(y)) {
                throw std::runtime_error(
                    "beammap direction selection found nonfinite telescope trajectory data");
            }
            const double p = axis_x * x + axis_y * y;
            mean_t += t;
            mean_p += p;
            max_abs_p = std::max(max_abs_p, std::abs(p));
        }
        mean_t /= static_cast<double>(count);
        mean_p /= static_cast<double>(count);

        double numerator = 0.0;
        double denominator = 0.0;
        Eigen::Index same_sign_steps = 0;
        Eigen::Index nonzero_steps = 0;
        double previous_p = axis_x * coordinate_x(start) +
                            axis_y * coordinate_y(start);
        for (Eigen::Index index = start; index <= inclusive_stop; ++index) {
            const double relative_t = time(index) - t0;
            const double p = axis_x * coordinate_x(index) +
                             axis_y * coordinate_y(index);
            numerator += (relative_t - mean_t) * (p - mean_p);
            denominator += (relative_t - mean_t) *
                           (relative_t - mean_t);
            if (index > start) {
                if (!(time(index) > time(index - 1))) {
                    throw std::runtime_error(
                        "beammap direction selection requires strictly increasing telescope time");
                }
                const double step = p - previous_p;
                if (step != 0.0) {
                    ++nonzero_steps;
                }
                previous_p = p;
            }
        }
        if (!(denominator > 0.0)) {
            throw std::runtime_error(
                "beammap direction selection found degenerate scan time support");
        }
        const double rate = numerator / denominator;
        const double first_p = axis_x * coordinate_x(start) +
                               axis_y * coordinate_y(start);
        const double last_p = axis_x * coordinate_x(inclusive_stop) +
                              axis_y * coordinate_y(inclusive_stop);
        const double displacement = last_p - first_p;
        const double coordinate_tolerance =
            64.0 * std::numeric_limits<double>::epsilon() *
            std::max(1.0, max_abs_p);
        const double rate_tolerance = coordinate_tolerance / (t1 - t0);
        if (!std::isfinite(rate) ||
            std::abs(displacement) <= coordinate_tolerance ||
            std::abs(rate) <= rate_tolerance ||
            std::signbit(displacement) != std::signbit(rate)) {
            throw std::runtime_error(
                "beammap direction selection found an ambiguous or low-speed scan leg");
        }

        for (Eigen::Index index = start + 1; index <= inclusive_stop;
             ++index) {
            const double prior = axis_x * coordinate_x(index - 1) +
                                 axis_y * coordinate_y(index - 1);
            const double current = axis_x * coordinate_x(index) +
                                   axis_y * coordinate_y(index);
            const double step = current - prior;
            if (step != 0.0 && std::signbit(step) == std::signbit(rate)) {
                ++same_sign_steps;
            }
        }

        BeammapDirectionScanRecord record;
        record.scan_index = scan;
        record.science_start = start;
        record.science_stop_exclusive = inclusive_stop + 1;
        record.sample_count = count;
        record.start_time_sec = t0;
        record.stop_time_sec = t1;
        record.duration_sec = t1 - t0;
        record.fast_axis_displacement_rad = displacement;
        record.signed_fast_axis_rate_rad_per_sec = rate;
        record.same_sign_step_fraction =
            nonzero_steps > 0
                ? static_cast<double>(same_sign_steps) /
                      static_cast<double>(nonzero_steps)
                : 0.0;
        record.direction = rate < 0.0 ? BeammapScanDirection::left
                                      : BeammapScanDirection::right;
        record.selected = beammap_direction_selects(mode, record.direction);
        if (record.direction == BeammapScanDirection::left) {
            ++plan.left_count;
        }
        else {
            ++plan.right_count;
        }
        if (record.selected) {
            ++plan.selected_count;
        }
        plan.scans.push_back(record);
    }

    if (plan.selected_count <= 0) {
        throw std::runtime_error(
            "beammap direction selection chose no scan legs for the requested mode");
    }
    return plan;
}

inline std::string beammap_direction_product_suffix(
    citlali::config::BeammapDirectionMode mode) {
    if (beammap_direction_mode_is_standard(mode)) {
        return {};
    }
    if (mode == citlali::config::BeammapDirectionMode::all) {
        throw std::logic_error(
            "beammap direction_mode 'all' has no single product suffix");
    }
    return "_" + std::string{citlali::config::to_string(mode)};
}

inline std::string beammap_direction_registry_suffix(
    citlali::config::BeammapDirectionMode mode) {
    if (beammap_direction_mode_is_standard(mode)) {
        return {};
    }
    return "_" + std::string{citlali::config::to_string(mode)};
}

inline citlali::config::BeammapDirectionMode
beammap_direction_realized_product_mode(
    citlali::config::BeammapDirectionMode requested_mode,
    std::string_view realized_mode) {
    if (!beammap_direction_mode_is_all(requested_mode)) {
        return requested_mode;
    }
    const auto parsed =
        citlali::config::parse_beammap_direction_mode(realized_mode);
    if (!parsed.has_value() ||
        parsed.value() == citlali::config::BeammapDirectionMode::all) {
        throw std::logic_error(
            "beammap all-product output lacks a realized standard/left/right product identity");
    }
    return parsed.value();
}

inline std::string beammap_direction_product_filename(
    std::string filename, citlali::config::BeammapDirectionMode mode) {
    filename += beammap_direction_product_suffix(mode);
    return filename;
}

inline void write_beammap_direction_scan_registry(
    const std::filesystem::path &path,
    const BeammapDirectionSelectionPlan &plan) {
    std::ofstream stream(path);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << "scan_index,science_start,science_stop_exclusive,sample_count,"
              "start_time_sec,stop_time_sec,duration_sec,coordinate_x_key,"
              "coordinate_y_key,scan_angle_rad,fast_axis_displacement_rad,"
              "signed_fast_axis_rate_rad_per_sec,same_sign_step_fraction,"
              "direction,selected,mode\n";
    stream << std::setprecision(17);
    for (const auto &scan : plan.scans) {
        stream << scan.scan_index << ',' << scan.science_start << ','
               << scan.science_stop_exclusive << ',' << scan.sample_count
               << ',' << scan.start_time_sec << ',' << scan.stop_time_sec
               << ',' << scan.duration_sec << ',' << plan.coordinate_x_key
               << ',' << plan.coordinate_y_key << ',' << plan.scan_angle_rad
               << ',' << scan.fast_axis_displacement_rad << ','
               << scan.signed_fast_axis_rate_rad_per_sec << ','
               << scan.same_sign_step_fraction << ','
               << to_string(scan.direction) << ','
               << (scan.selected ? "true" : "false") << ','
               << citlali::config::to_string(plan.mode) << '\n';
    }
}

}  // namespace citlali::pipeline
