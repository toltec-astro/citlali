#pragma once

#include <citlali/core/engine/config.h>

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace citlali::pipeline {

template <class Key, class InvalidKeys>
void add_invalid_config_key(const Key &key, InvalidKeys &invalid_keys) {
    typename InvalidKeys::value_type path;
    engine_utils::for_each_in_tuple(
        key, [&path](const auto &component) { path.push_back(component); });
    invalid_keys.push_back(std::move(path));
}

template <class KeyList>
bool config_parse_clean(
    const KeyList &missing_keys, const KeyList &invalid_keys,
    std::size_t missing_before, std::size_t invalid_before) {
    return missing_keys.size() == missing_before &&
           invalid_keys.size() == invalid_before;
}

template <class Config, class Param, class Diagnostics, class Key>
void read_config_value(
    Config &config, Param &param, Diagnostics &diagnostics, const Key &key,
    std::vector<std::decay_t<Param>> accepted_values = {},
    std::vector<std::decay_t<Param>> min_values = {},
    std::vector<std::decay_t<Param>> max_values = {}) {
    ::get_config_value(
        config, param, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths(), key,
        std::move(accepted_values), std::move(min_values),
        std::move(max_values));
}

template <class Processor, class Config, class Diagnostics>
void read_processor_config(
    Processor &processor, Config &config, Diagnostics &diagnostics) {
    processor.get_config(
        config, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths());
}

template <class Config, class Key, class Param, class Target,
          class MissingKeys, class InvalidKeys>
void read_config_value_if_clean(
    Config &config, const Key &key, Param &param, Target &&on_parsed,
    MissingKeys &missing_keys, InvalidKeys &invalid_keys,
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
        std::forward<Target>(on_parsed)(param);
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
    read_config_value_if_clean(
        config, key, param, [&target](const auto &value) { target = value; },
        missing_keys, invalid_keys, std::move(accepted_values),
        std::move(min_values), std::move(max_values));
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
    read_config_value_if_clean(
        config, key, param,
        [&target, &parser, &key, &invalid_keys](const auto &value) {
            if (auto parsed = parser(value)) {
                target = *parsed;
            } else {
                add_invalid_config_key(key, invalid_keys);
            }
        },
        missing_keys, invalid_keys, std::move(accepted_values),
        std::move(min_values), std::move(max_values));
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

template <class Config, class Key, class Param, class Target,
          class Diagnostics>
void read_config_value_if_clean(
    Config &config, const Key &key, Param &param, Target &&on_parsed,
    Diagnostics &diagnostics,
    std::vector<std::decay_t<Param>> accepted_values = {},
    std::vector<std::decay_t<Param>> min_values = {},
    std::vector<std::decay_t<Param>> max_values = {}) {
    read_config_value_if_clean(
        config, key, param, std::forward<Target>(on_parsed),
        diagnostics.missing_key_paths(), diagnostics.invalid_key_paths(),
        std::move(accepted_values), std::move(min_values),
        std::move(max_values));
}

template <class Config, class Key, class Param, class Target,
          class Diagnostics>
void read_mirrored_config_value(
    Config &config, const Key &key, Param &param, Target &target,
    Diagnostics &diagnostics,
    std::vector<std::decay_t<Param>> accepted_values = {},
    std::vector<std::decay_t<Param>> min_values = {},
    std::vector<std::decay_t<Param>> max_values = {}) {
    read_mirrored_config_value(
        config, key, param, target, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths(), std::move(accepted_values),
        std::move(min_values), std::move(max_values));
}

template <class Config, class Key, class Param, class Target,
          class Diagnostics>
void read_optional_mirrored_config_value(
    Config &config, const Key &key, Param &param, Target &target,
    Diagnostics &diagnostics,
    std::vector<std::decay_t<Param>> accepted_values = {},
    std::vector<std::decay_t<Param>> min_values = {},
    std::vector<std::decay_t<Param>> max_values = {}) {
    read_optional_mirrored_config_value(
        config, key, param, target, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths(), std::move(accepted_values),
        std::move(min_values), std::move(max_values));
}

template <class Config, class Key, class Param, class Target, class Parser,
          class Diagnostics>
void read_parsed_mirrored_config_value(
    Config &config, const Key &key, Param &param, Target &target,
    Parser parser, Diagnostics &diagnostics,
    std::vector<std::decay_t<Param>> accepted_values = {},
    std::vector<std::decay_t<Param>> min_values = {},
    std::vector<std::decay_t<Param>> max_values = {}) {
    read_parsed_mirrored_config_value(
        config, key, param, target, parser, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths(), std::move(accepted_values),
        std::move(min_values), std::move(max_values));
}

template <class Config, class Key, class Param, class Target, class Parser,
          class Diagnostics>
void read_optional_parsed_mirrored_config_value(
    Config &config, const Key &key, Param &param, Target &target,
    Parser parser, Diagnostics &diagnostics,
    std::vector<std::decay_t<Param>> accepted_values = {},
    std::vector<std::decay_t<Param>> min_values = {},
    std::vector<std::decay_t<Param>> max_values = {}) {
    read_optional_parsed_mirrored_config_value(
        config, key, param, target, parser, diagnostics.missing_key_paths(),
        diagnostics.invalid_key_paths(), std::move(accepted_values),
        std::move(min_values), std::move(max_values));
}

}  // namespace citlali::pipeline
