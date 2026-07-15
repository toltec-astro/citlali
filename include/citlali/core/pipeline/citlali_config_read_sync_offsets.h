#pragma once

// Included by citlali_config_read.h inside namespace citlali::pipeline.

inline const std::array<std::string_view, 14> &
interface_sync_offset_keys() {
    static constexpr std::array<std::string_view, 14> keys = {
        "toltec0",  "toltec1", "toltec2", "toltec3", "toltec4",
        "toltec5",  "toltec6", "toltec7", "toltec8", "toltec9",
        "toltec10", "toltec11", "toltec12", "hwpr"};
    return keys;
}

inline void set_interface_sync_offset(
    citlali::config::InterfaceSyncOffsetConfig &config,
    std::size_t key_index, double offset_sec) {
    if (key_index < citlali::config::toltec_interface_count) {
        config.toltec_offset_sec[key_index] = offset_sec;
    } else {
        config.hwpr_offset_sec = offset_sec;
    }
}

template <class Diagnostics, class Index>
void add_invalid_interface_sync_entry(
    Diagnostics &diagnostics, Index index) {
    add_invalid_config_key(
        std::tuple{
            std::string{"interface_sync_offset"},
            std::to_string(index)},
        diagnostics.invalid_key_paths());
}

template <class Config, class Diagnostics, class Logger>
bool read_interface_sync_offsets(
    Config &config,
    citlali::config::InterfaceSyncOffsetConfig &request,
    Diagnostics &diagnostics, const Logger &logger) {
    const auto missing_before = diagnostics.missing_key_paths().size();
    const auto invalid_before = diagnostics.invalid_key_paths().size();
    citlali::config::InterfaceSyncOffsetConfig candidate;

    if (!config.has(std::tuple{"interface_sync_offset"})) {
        request = candidate;
        return true;
    }

    const auto interface_node =
        config.get_node(std::tuple{"interface_sync_offset"});
    if (!interface_node.IsSequence()) {
        add_invalid_config_key(
            std::tuple{std::string{"interface_sync_offset"}},
            diagnostics.invalid_key_paths());
        return false;
    }

    const auto &keys = interface_sync_offset_keys();
    std::set<std::string> configured_keys;
    for (std::size_t index = 0; index < interface_node.size(); ++index) {
        const auto entry = interface_node[index];
        if (!entry.IsMap() || entry.size() != 1) {
            add_invalid_interface_sync_entry(diagnostics, index);
            continue;
        }

        const auto key = entry.begin()->first.template as<std::string>();
        const auto key_it = std::find(keys.begin(), keys.end(), key);
        if (key_it == keys.end()) {
            add_invalid_interface_sync_entry(diagnostics, index);
            continue;
        }
        if (!configured_keys.insert(key).second) {
            add_invalid_interface_sync_entry(diagnostics, index);
            continue;
        }

        try {
            const double offset_sec =
                entry.begin()->second.template as<double>();
            if (!std::isfinite(offset_sec)) {
                add_invalid_interface_sync_entry(diagnostics, index);
                continue;
            }
            set_interface_sync_offset(
                candidate,
                static_cast<std::size_t>(key_it - keys.begin()),
                offset_sec);
        } catch (const YAML::Exception &) {
            add_invalid_interface_sync_entry(diagnostics, index);
        }
    }

    for (const auto key : keys) {
        if (configured_keys.find(std::string{key}) ==
            configured_keys.end()) {
            logger->warn(
                "interface_sync_offset missing {}; using 0.0 s", key);
        }
    }

    const bool clean = config_parse_clean(
        diagnostics.missing_key_paths(), diagnostics.invalid_key_paths(),
        missing_before, invalid_before);
    if (clean) {
        request = candidate;
    }
    return clean;
}
