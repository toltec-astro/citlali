#pragma once

#include <citlali/core/pipeline/canonical_apt_detector_relation_v2.h>

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
    std::vector<std::string> &interfaces,
    AptDetectorRelationRetention retention =
        AptDetectorRelationRetention::retain) {
    engine.calib.get_apt(filepath, raw_filenames, interfaces, retention);
}

}  // namespace citlali::pipeline
