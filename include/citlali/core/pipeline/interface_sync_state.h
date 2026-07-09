#pragma once

#include <map>
#include <string>

namespace citlali::pipeline {

struct InterfaceSyncState {
    std::map<std::string, double> offsets;
};

}  // namespace citlali::pipeline
