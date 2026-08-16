#pragma once

#include <citlali/core/pipeline/array_properties_table_source.h>
#include <citlali/core/pipeline/raw_kids_data_access.h>

#include <type_traits>
#include <vector>

struct RawObs;

namespace citlali::pipeline {

template <class RawObsType>
inline constexpr bool is_production_raw_observation_v =
    std::is_same_v<
        std::remove_cv_t<std::remove_reference_t<RawObsType>>, ::RawObs>;

template <class Engine, class RawObs, class Logger>
void load_array_properties_table(Engine &engine, const RawObs &rawobs,
                                 const Logger &logger) {
    auto apt_path = array_properties_table_filepath(rawobs);
    log_array_properties_table_filepath(apt_path, logger);

    std::vector<std::string> raw_filenames = raw_kids_filepaths(rawobs);
    std::vector<std::string> interfaces = raw_kids_interfaces(rawobs);

    load_array_properties_table_file(
        engine, apt_path, raw_filenames, interfaces);
}

template <class Engine, class RawObsType, class Logger>
void load_science_array_properties_table(
    Engine &engine, const RawObsType &rawobs, const Logger &logger) {
    if constexpr (is_production_raw_observation_v<RawObsType>) {
        auto apt_path = array_properties_table_filepath(rawobs);
        log_array_properties_table_filepath(apt_path, logger);

        const std::vector<std::string> raw_filenames =
            raw_kids_filepaths(rawobs);
        const std::vector<std::string> interfaces =
            raw_kids_interfaces(rawobs);

        load_canonical_observation_array_properties_table_file(
            engine, apt_path, raw_filenames, interfaces);
    }
    else {
        // Generic non-production scaffolding predates canonical APT
        // admission. The concrete production ::RawObs branch above has no
        // legacy fallback.
        load_array_properties_table(engine, rawobs, logger);
    }
}

}  // namespace citlali::pipeline
