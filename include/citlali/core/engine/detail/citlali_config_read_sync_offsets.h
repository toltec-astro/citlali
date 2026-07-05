#pragma once

// Included by citlali_config_read.h inside namespace citlali::engine_detail {

inline const std::vector<std::string> &interface_sync_offset_keys() {
    static const std::vector<std::string> keys = {
        "toltec0",  "toltec1", "toltec2", "toltec3", "toltec4",
        "toltec5",  "toltec6", "toltec7", "toltec8", "toltec9",
        "toltec10", "toltec11", "toltec12", "hwpr"};
    return keys;
}

template <class Config, class OffsetMap, class Logger>
void read_interface_sync_offsets(
    Config &config, OffsetMap &interface_sync_offset, const Logger &logger) {
    const auto &interface_keys = interface_sync_offset_keys();
    for (const auto &key : interface_keys) {
        interface_sync_offset[key] = 0.0;
    }

    if (!config.has(std::tuple{"interface_sync_offset"})) {
        return;
    }

    auto interface_node = config.get_node(std::tuple{"interface_sync_offset"});
    std::set<std::string> configured_keys;
    for (Eigen::Index i = 0; i < interface_node.size(); ++i) {
        bool found_key = false;
        for (const auto &key : interface_keys) {
            if (config.has(std::tuple{"interface_sync_offset", i, key})) {
                auto offset = config.template get_typed<double>(
                    std::tuple{"interface_sync_offset", i, key});
                if (configured_keys.find(key) != configured_keys.end()) {
                    logger->warn(
                        "interface_sync_offset for {} specified multiple times; "
                        "using last value",
                        key);
                }
                interface_sync_offset[key] = offset;
                configured_keys.insert(key);
                found_key = true;
            }
        }
        if (!found_key) {
            logger->warn(
                "interface_sync_offset entry {} does not contain a recognized "
                "interface key; ignoring entry",
                i);
        }
    }

    for (const auto &key : interface_keys) {
        if (configured_keys.find(key) == configured_keys.end()) {
            logger->warn("interface_sync_offset missing {}; using 0.0 s", key);
        }
    }
}

