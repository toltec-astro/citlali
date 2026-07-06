#pragma once

#include <citlali/core/engine/config.h>

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

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

template <class Config, class Key, class Param, class Target,
          class MissingKeys, class InvalidKeys>
void read_mirrored_config_value(
    Config &config, const Key &key, Param &param, Target &target,
    MissingKeys &missing_keys, InvalidKeys &invalid_keys,
    std::vector<std::decay_t<Param>> accepted_values = {},
    std::vector<std::decay_t<Param>> min_values = {},
    std::vector<std::decay_t<Param>> max_values = {}) {
    const auto missing_before = missing_keys.size();
    const auto invalid_before = invalid_keys.size();
    ::get_config_value(config, param, missing_keys, invalid_keys, key,
                       std::move(accepted_values), std::move(min_values),
                       std::move(max_values));
    mirror_if_config_parsed(target, param, missing_keys, invalid_keys,
                            missing_before, invalid_before);
}

template <class Config, class Key, class Param, class Target,
          class MissingKeys, class InvalidKeys>
void read_optional_mirrored_config_value(
    Config &config, const Key &key, Param &param, Target &target,
    MissingKeys &missing_keys, InvalidKeys &invalid_keys,
    std::vector<std::decay_t<Param>> accepted_values = {},
    std::vector<std::decay_t<Param>> min_values = {},
    std::vector<std::decay_t<Param>> max_values = {}) {
    using value_type = std::decay_t<Param>;
    if (!config.template has_typed<value_type>(key)) {
        return;
    }
    read_mirrored_config_value(
        config, key, param, target, missing_keys, invalid_keys,
        std::move(accepted_values), std::move(min_values),
        std::move(max_values));
}

template <class Config, class Key, class Param, class Target, class Parser,
          class MissingKeys, class InvalidKeys>
void read_parsed_mirrored_config_value(
    Config &config, const Key &key, Param &param, Target &target,
    Parser parser, MissingKeys &missing_keys, InvalidKeys &invalid_keys,
    std::vector<std::decay_t<Param>> accepted_values = {},
    std::vector<std::decay_t<Param>> min_values = {},
    std::vector<std::decay_t<Param>> max_values = {}) {
    const auto missing_before = missing_keys.size();
    const auto invalid_before = invalid_keys.size();
    ::get_config_value(config, param, missing_keys, invalid_keys, key,
                       std::move(accepted_values), std::move(min_values),
                       std::move(max_values));
    if (config_parse_clean(
            missing_keys, invalid_keys, missing_before, invalid_before)) {
        if (auto parsed = parser(param)) {
            target = *parsed;
        }
    }
}

template <class Config, class Key, class Param, class Target, class Parser,
          class MissingKeys, class InvalidKeys>
void read_optional_parsed_mirrored_config_value(
    Config &config, const Key &key, Param &param, Target &target,
    Parser parser, MissingKeys &missing_keys, InvalidKeys &invalid_keys,
    std::vector<std::decay_t<Param>> accepted_values = {},
    std::vector<std::decay_t<Param>> min_values = {},
    std::vector<std::decay_t<Param>> max_values = {}) {
    using value_type = std::decay_t<Param>;
    if (!config.template has_typed<value_type>(key)) {
        return;
    }
    read_parsed_mirrored_config_value(
        config, key, param, target, parser, missing_keys, invalid_keys,
        std::move(accepted_values), std::move(min_values),
        std::move(max_values));
}

}  // namespace citlali::engine_detail
