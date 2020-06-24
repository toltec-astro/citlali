#pragma once
#include "formatter/container.h"
#include "logging.h"
#include <iomanip>
#include <sstream>

namespace config {

/**
 * @class Config
 * @brief A simple configuration class
 * @ingroup config
 */
class Config {

public:
    using key_t = std::string;
    using value_t =
        std::variant<std::monostate, bool, int, double, std::string>;
    using storage_t = std::map<key_t, value_t>;

    Config() = default;
    explicit Config(storage_t config) : m_config(std::move(config)) {}
    Config(const std::initializer_list<storage_t::value_type> &config)
        : Config(storage_t{config}) {}

    inline bool has(const key_t &k) const {
        return this->m_config.find(k) != m_config.end();
    }
    inline bool is_set(const key_t &k) const {
        return has(k) && !std::holds_alternative<std::monostate>(at(k));
    }
    template <typename F>
    inline auto try_call_with(const key_t &k, F &&f) const {
        SPDLOG_TRACE("try call with config key={} value={}", k, at(k));
        bool called = false;
        auto fn = [&](auto &&v) {
            if constexpr (std::is_invocable_v<F, DECAY(v)>) {
                SPDLOG_TRACE("f({}={}) called", k, at(k));
                called = true;
                return std::invoke(FWD(f), FWD(v));
            }
        };
        using RT =
            std::invoke_result_t<decltype(std::visit<decltype(fn), value_t>),
                                 decltype(fn), value_t>;
        if constexpr (std::is_void_v<RT>) {
            std::visit(fn, at(k));
            if (!called) {
                SPDLOG_TRACE("f({}) not called", k);
            }
            return called;
        } else {
            auto result = std::optional(std::visit(fn, at(k)));
            if (called) {
                return result;
            } else {
                SPDLOG_TRACE("f({}) not called", k);
            }
            return std::nullopt;
        }
    }
    template <typename F>
    inline auto call_if(const key_t &k, F &&f) const {
        SPDLOG_TRACE("try call if config key={} value={}", k, at(k));
        using RT = std::invoke_result_t<F>;
        if constexpr (std::is_void_v<RT>) {
            if (!has(k) || !std::holds_alternative<bool>(at(k)) ||
                !get_typed<bool>(k)) {
                SPDLOG_TRACE("f({}) not called", k);
                return false;
            }
            SPDLOG_TRACE("f({}={}) called", k, at(k));
            FWD(f);
            return true;
        } else {
            if (!has(k) || !std::holds_alternative<bool>(at(k)) ||
                !get_typed<bool>(k)) {
                SPDLOG_TRACE("f({}) not called", k);
                return std::optional<RT>{};
            }
            SPDLOG_TRACE("f({}={}) called", k, at(k));
            return FWD(f)();
        }
    }

    template <typename T>
    inline T get_typed(const std::string &key) const {
        SPDLOG_TRACE("get config key={} value={}", key, at(key));
        if constexpr (meta::is_instance<T, std::optional>::value) {
            // no throw
            if (has(key)) {
                return get_typed<typename T::value_type>(key);
            }
            return std::nullopt;
        } else {
            // may throw
            return std::get<T>(at(key));
        }
    }
    template <typename T>
    inline T get_typed(const std::string &key, T &&defval) const {
        if (has(key)) {
            return get_typed<T>(key);
        } else {
            return FWD(defval);
        }
    }

    template <typename T> inline T get_lexical(const std::string &key) const {
        const auto &v = at(key);
        if (std::holds_alternative<std::monostate>(v)) {
            throw std::bad_variant_access();
        }
        std::stringstream ss;
        std::visit(
            [&ss](auto &&arg) {
                if constexpr (!std::is_same_v<std::monostate,
                                              std::decay_t<decltype(arg)>>) {
                    ss << arg;
                }
            },
            at(key));
        T out;
        ss >> out;
        SPDLOG_TRACE("get config key={} value={} got={}", key, at(key), out);
        return out;
    }
    template <typename... Args>
    inline std::string get_str(const std::string &key, Args &&... args) const {
        return get_lexical<std::string>(key, FWD(args)...);
    }

    template <typename T> inline void set(const key_t &key, T &&v) {
        using T_ = std::decay_t<T>;
        using VT = std::conditional_t<std::is_same_v<T_, const char *>,
                                      std::string, T_>;
        at_or_add(key) = value_t{std::in_place_type<VT>, FWD(v)};
    }

    std::string pformat() const {
        if (this->m_config.size() == 0) {
            return "{}";
        }
        // compute width
        std::size_t key_width = 0;
        auto it = std::max_element(this->m_config.begin(), this->m_config.end(),
                                   [](const auto &a, const auto &b) {
                                       return a.first.size() < b.first.size();
                                   });
        if (it != this->m_config.end()) {
            key_width = it->first.size();
        }
        std::stringstream ss;
        ss << "{";
        for (const auto &p : this->m_config) {
            ss << "\n   " << std::setw(meta::size_cast<int>(key_width))
               << std::right << p.first;
            ss << fmt::format(": {}", p.second);
        }
        ss << "\n}";
        return ss.str();
    }

    const value_t &at(const std::string &key) const {
        try {
            return this->m_config.at(key);
        } catch (const std::out_of_range &) {
            throw std::out_of_range(fmt::format(
                "invalid key: \"{}\" in config {}", key, pformat()));
        }
    }
    value_t &at(const std::string &key) {
        return const_cast<value_t &>(const_cast<const Config *>(this)->at(key));
    }

    template <typename value_t_in = value_t>
    value_t &at_or_add(const std::string &key) {
        if (!has(key)) {
            this->m_config[key] = value_t_in{};
            SPDLOG_TRACE("add config key={}", key);
        }
        return at(key);
    }

    Config &update(Config other) {
        other.m_config.merge(std::move(m_config));
        m_config = std::move(other.m_config);
        return *this;
    }

private:
    storage_t m_config{};
};

} // namespace config
