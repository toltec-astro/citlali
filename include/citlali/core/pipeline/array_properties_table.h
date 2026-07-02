#pragma once

#include <citlali/core/pipeline/rawobs_data_items.h>

#include <string>
#include <vector>

namespace citlali::pipeline {

template <class RawObs>
std::string array_properties_table_filepath(const RawObs &rawobs) {
    return rawobs.array_prop_table().filepath();
}

template <class RawObs>
std::vector<std::string> raw_kids_filepaths(const RawObs &rawobs) {
    std::vector<std::string> result;
    for (const auto &data_item : rawobs.kidsdata()) {
        const auto &item = detail::unwrap_reference_wrapper(data_item);
        result.push_back(item.filepath());
    }
    return result;
}

template <class RawObs>
std::vector<std::string> raw_kids_interfaces(const RawObs &rawobs) {
    std::vector<std::string> result;
    for (const auto &data_item : rawobs.kidsdata()) {
        const auto &item = detail::unwrap_reference_wrapper(data_item);
        result.push_back(item.interface());
    }
    return result;
}

template <class Engine, class RawObs, class Logger>
void load_array_properties_table(Engine &engine, const RawObs &rawobs,
                                 const Logger &logger) {
    auto apt_path = array_properties_table_filepath(rawobs);
    logger->info("getting array properties table {}", apt_path);

    std::vector<std::string> raw_filenames = raw_kids_filepaths(rawobs);
    std::vector<std::string> interfaces;
    for (const auto &data_item : rawobs.kidsdata()) {
        const auto &item = detail::unwrap_reference_wrapper(data_item);
        interfaces.push_back(item.interface());
    }

    engine.calib.get_apt(apt_path, raw_filenames, interfaces);
}

}  // namespace citlali::pipeline
