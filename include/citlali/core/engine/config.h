#pragma once

#include <citlali/core/utils/utils.h>

template<typename param_t, typename option_t, typename key_vec_t>
void check_allowed(param_t param, key_vec_t &missing_keys, key_vec_t &invalid_keys,
                   std::vector<param_t> allowed, option_t option) {
    // loop through allowed values and see if param is contained within it
    if (!std::any_of(allowed.begin(), allowed.end(), [&](const auto i){return i==param;})) {
        // temporary vector to hold current invalid param's keys
        std::vector<std::string> invalid_temp;
        // push back invalid keys into temp vector
        engine_utils::for_each_in_tuple(option, [&](const auto &x) {
            invalid_temp.push_back(x);
        });

        // push temp invalid keys vector into invalid keys vector
        invalid_keys.push_back(invalid_temp);
    }
}
template<typename param_t, typename option_t, typename key_vec_t>
void check_range(param_t param, key_vec_t &missing_keys, key_vec_t &invalid_keys,
                 std::vector<param_t> min_val,  std::vector<param_t> max_val,
                 option_t option) {

    bool invalid = false;

    // make sure param is larger than minimum
    if (!min_val.empty()) {
        if (param < min_val.at(0)) {
            invalid = true;
        }
    }

    // make sure param is smaller than maximum
    if (!max_val.empty()) {
        if (param > max_val.at(0)) {
            invalid = true;
        }
    }

    // if param is invalid
    if (invalid) {
        // temporary vector to hold current invalid param's keys
        std::vector<std::string> invalid_temp;
        // push back invalid keys into temp vector
        engine_utils::for_each_in_tuple(option, [&](const auto &x) {
            invalid_temp.push_back(x);
        });

        // push temp invalid keys vector into invalid keys vector
        invalid_keys.push_back(invalid_temp);
    }
}

template <typename config_t, typename param_t, typename option_t, typename key_vec_t>
void get_config_value(config_t config, param_t &param, key_vec_t &missing_keys,
               key_vec_t &invalid_keys, option_t option, std::vector<param_t> allowed={},
               std::vector<param_t> min_val={}, std::vector<param_t> max_val={}) {

    // check if config option exists
    try {
        if (config.template has_typed<param_t>(option_t(option))) {
            // get the parameter from config
            param = config.template get_typed<param_t>(option_t(option));

            // if allowed values is specified, check against them
            if (!allowed.empty()) {
                check_allowed(param, missing_keys, invalid_keys, allowed, option);
            }

            // if a range is specified, check against them
            if (!min_val.empty() || !max_val.empty()) {
                check_range(param, missing_keys, invalid_keys, min_val, max_val, option);
            }
        }
        // else mark as missing
        else {
            // temporary vector to hold current missing param's keys
            std::vector<std::string> missing_temp;
            // push back missing keys into temp vector
            engine_utils::for_each_in_tuple(option, [&](const auto &x) {
                missing_temp.push_back(x);
            });
            // push temp missing keys vector into invalid keys vector
            missing_keys.push_back(missing_temp);
        }
    }
    catch (YAML::TypedBadConversion<param_t>) {
        // temporary vector to hold current invalid param's keys
        std::vector<std::string> invalid_temp;
        // push back invalid keys into temp vector
        engine_utils::for_each_in_tuple(option, [&](const auto &x) {
            invalid_temp.push_back(x);
        });

        // push temp invalid keys vector into invalid keys vector
        invalid_keys.push_back(invalid_temp);
    }
    catch (YAML::InvalidNode) {
        // temporary vector to hold current invalid param's keys
        std::vector<std::string> invalid_temp;
        // push back invalid keys into temp vector
        engine_utils::for_each_in_tuple(option, [&](const auto &x) {
            invalid_temp.push_back(x);
        });

        // push temp invalid keys vector into invalid keys vector
        invalid_keys.push_back(invalid_temp);
    }
}
