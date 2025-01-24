#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <memory>
#include <variant>
#include <stdexcept>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>
#include <algorithm>

// iterate over tuple (https://stackoverflow.com/questions/26902633/how-to-iterate-over-a-stdtuple-in-c-11)
template<class F, class...Ts, std::size_t...Is>
void for_each_in_tuple(const std::tuple<Ts...> & tuple, F func, std::index_sequence<Is...>) {
    using expander = int[];
    (void)expander { 0, ((void)func(std::get<Is>(tuple)), 0)... };
}

template<class F, class...Ts>
void for_each_in_tuple(const std::tuple<Ts...> & tuple, F func) {
    for_each_in_tuple(tuple, func, std::make_index_sequence<sizeof...(Ts)>());
}

template<typename Derived>
std::vector<std::tuple<double, int>> sorter(Eigen::DenseBase<Derived> &vec) {
    std::vector<std::tuple<double, int>> vis;
    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(vec.size(),0,vec.size()-1);

    for(Eigen::Index i=0; i<vec.size(); ++i) {
        std::tuple<double, double> vec_and_val(vec(i), indices(i));
        vis.push_back(vec_and_val);
    }

    std::sort(vis.begin(), vis.end(),
              [&](const std::tuple<double, int> &a, const std::tuple<double, int> &b) -> bool {
                  return std::get<0>(a) < std::get<0>(b);
              });

    return vis;
}

class ConfigValidator {
public:
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    tula::config::YamlConfig& config;
    std::vector<std::vector<std::string>> missing_keys, invalid_keys;

    ConfigValidator(tula::config::YamlConfig& _c)
        : config(_c) {}

    template<typename T, typename OptionType>
    void get(T& param, OptionType&& input_tuple) {
        get_config_value(param, std::forward<OptionType>(input_tuple));
    }

    template<typename T, typename OptionType>
    void check_allowed(T param, const std::vector<T>& allowed, OptionType option) {
        if (!std::any_of(allowed.begin(), allowed.end(), [&](const T& i){ return i == param; })) {
            std::vector<std::string> invalid_temp;
            for_each_in_tuple(option, [&](const auto& x) {
                invalid_temp.push_back(x);
            });
            invalid_keys.push_back(invalid_temp);
        }
    }

    template<typename T, typename OptionType>
    void check_range(T param, std::optional<T> min_val, std::optional<T> max_val, OptionType option) {
        bool invalid = false;

        if (min_val.has_value() && param < min_val.value()) {
            invalid = true;
        }
        if (max_val.has_value() && param > max_val.value()) {
            invalid = true;
        }

        if (invalid) {
            std::vector<std::string> invalid_temp;
            for_each_in_tuple(option, [&](const auto& x) {
                invalid_temp.push_back(x);
            });
            invalid_keys.push_back(invalid_temp);
        }
    }

    // overload for scalar values
    template <typename T, typename OptionType>
    void get_config_value(T& param, OptionType option, std::vector<T> allowed = {}, std::optional<T> min_val = {},
             std::optional<T> max_val = {}) {
        try {
            if (config.template has_typed<T>(option)) {
                param = config.template get_typed<T>(option);

                if (!allowed.empty()) {
                    check_allowed(param, allowed, option);
                }
                if (min_val.has_value() || max_val.has_value()) {
                    check_range(param, min_val, max_val, option);
                }

                logger->debug("got {} from config", option);
            }
            else {
                // handle missing keys
                std::vector<std::string> missing_temp;
                for_each_in_tuple(option, [&](const auto& x) {
                    missing_temp.push_back(x);
                });
                missing_keys.push_back(missing_temp);
            }
        }
        catch (const YAML::TypedBadConversion<T>&) {
            std::vector<std::string> invalid_temp;
            for_each_in_tuple(option, [&](const auto& x) {
                invalid_temp.push_back(x);
            });
            invalid_keys.push_back(invalid_temp);
        }
        catch (const YAML::InvalidNode&) {
            std::vector<std::string> invalid_temp;
            for_each_in_tuple(option, [&](const auto& x) {
                invalid_temp.push_back(x);
            });
            invalid_keys.push_back(invalid_temp);
        }
    }

    // overload for vector values
    template <typename T, typename OptionType>
    void get_config_value(std::vector<T>& param, OptionType option, std::vector<T> allowed = {},
             std::optional<T> min_val = {}, std::optional<T> max_val = {}) {

        try {
            if (config.template has_typed<std::vector<T>>(option)) {
                param = config.template get_typed<std::vector<T>>(option);

                for (const auto& p : param) {
                    if (!allowed.empty()) {
                        check_allowed(p, allowed, option);
                    }
                    if (min_val.has_value() || max_val.has_value()) {
                        check_range(p, min_val, max_val, option);
                    }
                }

                logger->debug("got vector {} from config", option);
            }
            else {
                // handle missing keys
                std::vector<std::string> missing_temp;
                for_each_in_tuple(option, [&](const auto& x) {
                    missing_temp.push_back(x);
                });
                missing_keys.push_back(missing_temp);
            }
        }
        catch (const YAML::TypedBadConversion<std::vector<T>>&) {
            std::vector<std::string> invalid_temp;
            for_each_in_tuple(option, [&](const auto& x) {
                invalid_temp.push_back(x);
            });
            invalid_keys.push_back(invalid_temp);
        }
        catch (const YAML::InvalidNode&) {
            std::vector<std::string> invalid_temp;
            for_each_in_tuple(option, [&](const auto& x) {
                invalid_temp.push_back(x);
            });
            invalid_keys.push_back(invalid_temp);
        }
    }
};
