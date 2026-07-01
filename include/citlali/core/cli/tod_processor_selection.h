#pragma once

#include <fmt/core.h>
#include <tula/formatter/container.h>

#include <ostream>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace citlali::cli {

enum class TodProcessorSelectionStatus {
    ok,
    missing_reduction_type,
    invalid_reduction_type
};

inline auto reduction_type_config_key() {
    return std::tuple{"runtime", "reduction_type"};
}

inline std::vector<std::string> reduction_type_config_key_path() {
    return {"runtime", "reduction_type"};
}

template <class Config>
bool has_reduction_type_config(const Config &config) {
    return config.has(reduction_type_config_key());
}

template <class Config>
std::optional<std::string> read_reduction_type_config(Config &config) {
    if (!has_reduction_type_config(config)) {
        return std::nullopt;
    }
    return config.get_str(reduction_type_config_key());
}

template <class TodProcVariant, class ScienceTodProc, class PointingTodProc,
          class BeammapTodProc, class Config, class Logger>
bool emplace_tod_processor_for_reduction_type(
    TodProcVariant &todproc, const std::string &reduction_type,
    Config &config, const Logger &logger) {
    if (reduction_type == "science") {
        logger->info("reducing in science mode");
        todproc.template emplace<ScienceTodProc>(
            ScienceTodProc::from_config(config));
        return true;
    }

    if (reduction_type == "pointing") {
        logger->info("reducing in pointing mode");
        todproc.template emplace<PointingTodProc>(
            PointingTodProc::from_config(config));
        return true;
    }

    if (reduction_type == "beammap") {
        logger->info("reducing in beammap mode");
        todproc.template emplace<BeammapTodProc>(
            BeammapTodProc::from_config(config));
        return true;
    }

    return false;
}

template <class TodProcVariant, class ScienceTodProc, class PointingTodProc,
          class BeammapTodProc, class Config, class Logger>
TodProcessorSelectionStatus select_tod_processor_from_config(
    TodProcVariant &todproc, Config &config, const Logger &logger) {
    if (!has_reduction_type_config(config)) {
        return TodProcessorSelectionStatus::missing_reduction_type;
    }

    try {
        auto reduction_type = *read_reduction_type_config(config);

        if (!emplace_tod_processor_for_reduction_type<
                TodProcVariant, ScienceTodProc, PointingTodProc,
                BeammapTodProc>(todproc, reduction_type, config, logger)) {
            return TodProcessorSelectionStatus::invalid_reduction_type;
        }

    // catch bad yaml type conversion and mark as invalid
    } catch (YAML::TypedBadConversion<std::string>) {
        return TodProcessorSelectionStatus::invalid_reduction_type;
    }

    return TodProcessorSelectionStatus::ok;
}

inline bool report_tod_processor_selection_failure(
    TodProcessorSelectionStatus status, std::ostream &os) {
    if (status == TodProcessorSelectionStatus::missing_reduction_type) {
        auto missing_keys = reduction_type_config_key_path();
        os << fmt::format("missing keys={}", missing_keys) << "\n";
        return true;
    }

    if (status == TodProcessorSelectionStatus::invalid_reduction_type) {
        auto invalid_keys = reduction_type_config_key_path();
        os << fmt::format("invalid keys={}", invalid_keys) << "\n";
        return true;
    }

    return false;
}

}  // namespace citlali::cli
