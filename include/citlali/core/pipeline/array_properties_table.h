#pragma once

#include <citlali/core/pipeline/array_properties_table_source.h>
#include <citlali/core/pipeline/raw_kids_data_access.h>

#include <vector>

namespace citlali::pipeline {

template <class Engine, class RawObs, class Logger>
void load_array_properties_table(Engine &engine, const RawObs &rawobs,
                                 const Logger &logger,
                                 AptDetectorRelationRetention retention =
                                     AptDetectorRelationRetention::retain) {
    auto apt_path = array_properties_table_filepath(rawobs);
    log_array_properties_table_filepath(apt_path, logger);

    std::vector<std::string> raw_filenames = raw_kids_filepaths(rawobs);
    std::vector<std::string> interfaces = raw_kids_interfaces(rawobs);

    load_array_properties_table_file(
        engine, apt_path, raw_filenames, interfaces, retention);
}

}  // namespace citlali::pipeline
