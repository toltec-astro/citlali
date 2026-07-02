#pragma once

#include <citlali/core/pipeline/raw_kids_data_access.h>

#include <string>
#include <vector>

namespace citlali::pipeline {

template <class RawObs>
std::string array_properties_table_filepath(const RawObs &rawobs) {
    return rawobs.array_prop_table().filepath();
}

template <class Logger>
void log_array_properties_table_filepath(const std::string &filepath,
                                         const Logger &logger) {
    logger->info("getting array properties table {}", filepath);
}

template <class Engine>
void load_array_properties_table_file(
    Engine &engine, const std::string &filepath,
    std::vector<std::string> &raw_filenames,
    std::vector<std::string> &interfaces) {
    engine.calib.get_apt(filepath, raw_filenames, interfaces);
}

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

}  // namespace citlali::pipeline
