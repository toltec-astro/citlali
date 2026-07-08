#pragma once

#include <citlali/core/config/runtime_config.h>

#include <fmt/core.h>
#include <tula/formatter/container.h>

#include <ostream>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace citlali::cli {

enum class TodProcessorSelectionStatus {
    ok,
    missing_reduction_type,
    invalid_reduction_type
};

template <class TodProc>
inline constexpr bool is_empty_tod_processor_v =
    std::is_same_v<TodProc, std::monostate>;

template <class ScienceTodProc, class PointingTodProc, class BeammapTodProc>
using TodProcessorVariant =
    std::variant<std::monostate, ScienceTodProc, PointingTodProc,
                 BeammapTodProc>;

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
    TodProcVariant &todproc, citlali::config::ReductionType reduction_type,
    Config &config, const Logger &logger) {
    if (citlali::config::is_science_reduction_type(reduction_type)) {
        logger->info("reducing in science mode");
        todproc.template emplace<ScienceTodProc>(
            ScienceTodProc::from_config(config));
        return true;
    }

    if (citlali::config::is_pointing_reduction_type(reduction_type)) {
        logger->info("reducing in pointing mode");
        todproc.template emplace<PointingTodProc>(
            PointingTodProc::from_config(config));
        return true;
    }

    if (citlali::config::is_beammap_reduction_type(reduction_type)) {
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
        const auto reduction_type_name = *read_reduction_type_config(config);
        auto reduction_type =
            citlali::config::parse_reduction_type(reduction_type_name);
        if (!reduction_type) {
            return TodProcessorSelectionStatus::invalid_reduction_type;
        }

        if (!emplace_tod_processor_for_reduction_type<
                TodProcVariant, ScienceTodProc, PointingTodProc,
                BeammapTodProc>(todproc, *reduction_type, config, logger)) {
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

template <class TodProcVariant, class ScienceTodProc, class PointingTodProc,
          class BeammapTodProc, class Config, class Logger>
bool select_tod_processor_or_report_failure(
    TodProcVariant &todproc, Config &config, const Logger &logger,
    std::ostream &os) {
    auto selection_status =
        select_tod_processor_from_config<
            TodProcVariant, ScienceTodProc, PointingTodProc,
            BeammapTodProc>(todproc, config, logger);

    return !report_tod_processor_selection_failure(selection_status, os);
}

template <class ScienceTodProc, class PointingTodProc, class BeammapTodProc,
          class Config, class Logger>
bool select_tod_processor_variant_or_report_failure(
    TodProcessorVariant<ScienceTodProc, PointingTodProc, BeammapTodProc>
        &todproc,
    Config &config, const Logger &logger, std::ostream &os) {
    using todproc_var_t =
        TodProcessorVariant<ScienceTodProc, PointingTodProc, BeammapTodProc>;
    return select_tod_processor_or_report_failure<
        todproc_var_t, ScienceTodProc, PointingTodProc, BeammapTodProc>(
        todproc, config, logger, os);
}

}  // namespace citlali::cli
