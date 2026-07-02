#pragma once

#include <citlali/core/pipeline/rawobs_data_items.h>

#include <string>
#include <vector>

namespace citlali::pipeline {

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

}  // namespace citlali::pipeline
