#pragma once

#include <cstddef>

namespace citlali::engine_detail {

template <class KeyList>
bool config_parse_clean(
    const KeyList &missing_keys, const KeyList &invalid_keys,
    std::size_t missing_before, std::size_t invalid_before) {
    return missing_keys.size() == missing_before &&
           invalid_keys.size() == invalid_before;
}

template <class Target, class Source, class KeyList>
void mirror_if_config_parsed(
    Target &target, const Source &source, const KeyList &missing_keys,
    const KeyList &invalid_keys, std::size_t missing_before,
    std::size_t invalid_before) {
    if (config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before)) {
        target = source;
    }
}

}  // namespace citlali::engine_detail
