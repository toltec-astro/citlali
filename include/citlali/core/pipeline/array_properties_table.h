#pragma once

#include <citlali/core/pipeline/rawobs_data_items.h>

#include <string>
#include <vector>

namespace citlali::pipeline {

template <class Engine, class RawObs, class Logger>
void load_array_properties_table(Engine &engine, const RawObs &rawobs,
                                 const Logger &logger) {
    auto apt_path = rawobs.array_prop_table().filepath();
    logger->info("getting array properties table {}", apt_path);

    std::vector<std::string> raw_filenames, interfaces;
    for (const auto &data_item : rawobs.kidsdata()) {
        const auto &item = detail::unwrap_reference_wrapper(data_item);
        raw_filenames.push_back(item.filepath());
        interfaces.push_back(item.interface());
    }

    engine.calib.get_apt(apt_path, raw_filenames, interfaces);
}

}  // namespace citlali::pipeline
