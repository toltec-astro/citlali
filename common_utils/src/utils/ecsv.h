#pragma once

#include <algorithm>
#include <cassert>
#include <complex>
#include <regex>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <variant>
#include <yaml-cpp/yaml.h>

namespace YAML {

template <typename... Ts> struct convert<std::variant<Ts...>> {

    using variant_t = std::variant<Ts...>;

    /// @brief Recursively try convert node to variant alternative *in order*.
    /// The order of types matters here, for example,
    /// variant<string, int> will capture all values as string because
    /// in YAML/Json everything is string.
    /// for variant<int, string>, all nodes convertable to int are converted.
    template <std::size_t i = 0>
    static auto _to_variant(const YAML::Node &n) -> std::optional<variant_t> {
        if constexpr (i < std::variant_size_v<variant_t>) {
            using vt = std::variant_alternative_t<i, variant_t>;
            std::optional<vt> result{};
            if constexpr (std::is_same_v<vt, std::monostate>) {
                // capture null with this
                if (n.IsNull()) {
                    result = std::monostate{};
                }
            } else {
                if (vt v{}; YAML::convert<vt>::decode(n, v)) {
                    result = v;
                }
            }
            return result.has_value() ? std::move(result.value())
                                      : _to_variant<i + 1>(n);
        }
        return std::nullopt;
    }

    static bool decode(const Node &node, variant_t &rhs) {
        // capture null with monostate
        if (node.IsNull()) {

            if constexpr (std::bool_constant<(
                              std::is_same_v<std::monostate, Ts> ||
                              ...)>::value) {
                rhs = std::monostate{};
                return true;
            } else {
                return false;
            }
        }
        if (auto opt_result = _to_variant(node); opt_result.has_value()) {
            rhs = std::move(opt_result.value());
            return true;
        }
        return false;
    }
};

template <typename T> struct convert<std::optional<T>> {
    using optional_t = std::optional<T>;

    /// @brief Decode YAML node to optional.
    /// Note that in YAML/Json everything is string.
    /// optional<string> will capture all node.
    static bool decode(const Node &node, optional_t &rhs) {
        if (node.IsNull()) {
            rhs = std::nullopt;
            return true;
        }
        if (T v{}; convert<T>::decode(node, v)) {
            rhs = std::move(v);
            return true;
        }
        return false;
    }
};

} // namespace YAML

namespace ecsv {

namespace internal {

/// @brief Returns true if \p v ends with \p ending.
template <typename T, typename U> bool startswith(const T &v, const U &prefix) {
    if (prefix.size() > v.size()) {
        return false;
    }
    return std::equal(prefix.begin(), prefix.end(), v.begin());
}

template <class T> struct always_false : std::false_type {};

template <typename T, class Func, T... Is,
          template <typename TT, TT...> typename S>
constexpr auto apply_const_sequence(Func &&f, S<T, Is...>) -> decltype(auto) {
    return std::forward<Func>(f)(std::integral_constant<T, Is>{}...);
}

// check if type is template instance
template <typename, template <typename...> typename = std::void_t,
          template <typename...> typename = std::void_t>
struct is_instance : public std::false_type {};

template <typename... Ts, template <typename...> typename T>
struct is_instance<T<Ts...>, T> : public std::true_type {};

template <typename... Ts, template <typename...> typename T,
          template <typename...> typename U, typename... Us>
struct is_instance<T<U<Us...>, Ts...>, T, U> : public std::true_type {};

} // namespace internal

namespace spec {

static constexpr std::string_view ECSV_VERSION = "0.9";
static constexpr char ECSV_DELIM_CHAR = ' ';
static constexpr std::string_view ECSV_HEADER_PREFIX = "# ";
static constexpr std::string_view ECSV_VERSION_LINE_REGEX = "^# %ECSV .+";
static constexpr std::string_view ECSV_VERSION_LINE_PREFIX = "%ECSV ";
static constexpr std::string_view ECSV_META_TAG = "tag:yaml.org,2002:omap";
static constexpr std::string_view k_datatype = "datatype";
static constexpr std::string_view k_meta = "meta";
static constexpr std::string_view k_name = "name";

template <typename OStream> void dump_yaml_preamble(OStream &os) {
    os << ECSV_VERSION_LINE_PREFIX << ECSV_VERSION << "\n---\n";
}

template <typename OStream>
void dump_yaml_header(OStream &os, YAML::Node node) {
    std::stringstream buf;
    dump_yaml_preamble(buf);
    // dump the node to tmp buff
    buf << node;
    // process each line to prepend the header prefix
    std::string ln;
    while (std::getline(buf, ln)) {
        os << spec::ECSV_HEADER_PREFIX << ln << "\n";
    }
}

} // namespace spec

/// @brief Throw when there is an error when parse ECSV.
struct ParseError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// @brief Throw when there is an error when dump ECSV.
struct DumpError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// @brief Read stream line by line and parse as ECSV header
/// @param is Input stream to process.
/// @param lines Optional output to capture the read lines.
template <typename IStream>
auto parse_header(IStream &is, std::vector<std::string> *lines = nullptr) {
    std::regex re_ecsv_version{std::string{spec::ECSV_VERSION_LINE_REGEX}};

    std::stringstream ss_header; // stream to hold all header info.

    std::size_t l = 0; // running line number
    std::string ln{};  // running line content;
    std::vector<std::string> csv_colnames{};
    while (std::getline(is, ln)) {
        if (lines) {
            lines->push_back(ln);
        }
        // ignore any leading spaces
        ln.erase(ln.begin(), std::find_if(ln.begin(), ln.end(), [](auto ch) {
                     return !std::isspace(ch);
                 }));
        // check first line.
        if (l == 0) {
            if (std::regex_match(ln, re_ecsv_version)) {
                //
            } else {
                throw ParseError("no ECSV version line found");
            }
            ++l;
            continue;
        }
        // l > 0
        if (ln == "#") {
            // this is an empty line, just ignore
            ++l;
            continue;
        }
        if (internal::startswith(ln, spec::ECSV_HEADER_PREFIX)) {
            // this line is part of the header yaml
            // put the actual yaml part to the stream
            for (auto it = ln.begin() + spec::ECSV_HEADER_PREFIX.size();
                 it != ln.end(); ++it) {
                ss_header << *it;
            }
            ss_header << "\n";
            ++l;
            continue;
        }
        // else
        // this line is not the yaml header but could be the csv header
        {
            // break the line to get colnames
            std::string colname{};
            for (auto it = ln.begin(); it != ln.end(); ++it) {
                if (spec::ECSV_DELIM_CHAR != *it) {
                    // not a delim, so append to colname
                    colname += *it;
                    continue;
                }
                // found delim
                if (colname.empty()) {
                    // keep finding if nothing in colname
                    continue;
                }
                // got something in colname
                csv_colnames.push_back(colname);
                colname.clear(); // reset for the next
            }
            // get anything left in colname
            if (!colname.empty()) {
                csv_colnames.push_back(colname);
            }
            ++l;
            break;
        }
    }
    assert(lines ? (l == lines->size()) : true); // sanity check
    // construct the yaml object and return the stream
    auto header = YAML::Load(ss_header);
    // check the required items
    if (!header["datatype"]) {
        throw ParseError("Missing datatype in header YAML");
    }
    // collect data types
    // get datatypes
    std::vector<std::string> names;
    std::vector<std::string> dtypes;
    for (const auto &n : header["datatype"]) {
        names.emplace_back(n["name"].as<std::string>());
        dtypes.emplace_back(n["datatype"].as<std::string>());
    }
    if (csv_colnames != names) {
        throw ParseError("csv colnames is not the same as in the yaml header.");
    }
    // meta header
    YAML::Node meta{};
    if (header["meta"]) {
        meta = header["meta"];
    }
    return std::tuple(std::move(names), std::move(dtypes), std::move(meta));
};

template <typename T> std::string dtype_str() {
    // bool, int8, int16, int32, int64
    // uint8, uint16, uint32, uint64
    // float16, float32, float64, float128
    // complex64, complex128, complex256
    if constexpr (std::is_same_v<T, bool>) {
        return "bool";
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return "int8";
    } else if constexpr (std::is_same_v<T, int16_t>) {
        return "int16";
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return "int32";
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return "int64";
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return "uint8";
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return "uint16";
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return "uint32";
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return "uint64";
    } else if constexpr (std::is_same_v<T, float>) {
        return "float32";
    } else if constexpr (std::is_same_v<T, double>) {
        return "float64";
    } else if constexpr (std::is_same_v<T, long double>) {
        return "float128";
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return "complex64";
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        return "complex128";
    } else if constexpr (std::is_same_v<T, std::complex<long double>>) {
        return "complex256";
    } else {
        static_assert(internal::always_false<T>::value,
                      "ECSV NOT IMPLEMENTED FOR THIS SCALAR TYPE");
    }
}

/// @brief Check if the declared data types are uniform across all columns and
/// is T.
/// @tparam T The desired data type.
template <typename T>
bool check_uniform_dtype(const std::vector<std::string> &dtypes) {
    if (std::adjacent_find(dtypes.begin(), dtypes.end(),
                           std::not_equal_to<>()) == dtypes.end()) {
        // all dtypes are the same
        const auto &dtype = dtypes.front();
        return dtype == dtype_str<T>();
    }
    return false;
}

/// @brief Return a YAML node representing a column.
template <typename T> auto make_column_node(const std::string &name) {
    YAML::Node n{};
    n.SetStyle(YAML::EmitterStyle::Flow);
    std::string k_name{spec::k_name};
    std::string k_datatype{spec::k_datatype};
    n[k_name] = name;
    n[k_datatype] = dtype_str<T>();
    return n;
}

template <typename OStream, typename T, typename... Ts>
void dump_header(OStream &os, std::vector<std::string> colnames,
                 std::tuple<T, Ts...> types = {}, YAML::Node meta = {}) {
    YAML::Node header;
    header.SetStyle(YAML::EmitterStyle::Block);
    constexpr auto ntypes = sizeof...(Ts) + 1; // number of types include T
    if (ntypes > 1 && (ntypes != colnames.size())) {
        throw DumpError("mismatch number of types with colnames.");
    }
    std::string k_datatype{spec::k_datatype};
    std::string k_meta{spec::k_meta};
    if constexpr (ntypes == 1) {
        // all types are T. uniform.
        for (const auto &c : colnames) {
            header[k_datatype].push_back(make_column_node<T>(c));
        }
    } else {
        // types are differnet
        internal::apply_const_sequence(
            [&](auto &&i_) {
                constexpr auto i = std::decay_t<decltype(i_)>::value;
                // data type
                using U =
                    std::tuple_element_t<i, std::decay_t<decltype(types)>>;
                header[k_datatype].push_back(
                    (make_column_node<U>(colnames[i])));
            },
            std::make_index_sequence<ntypes>{});
    }
    if (!meta.IsNull()) {
        header[k_meta] = std::move(meta);
    }
    spec::dump_yaml_header(os, header);
}

/// @brief Create meta node from map-like data structure.
template <typename Map> auto map_to_meta(Map &&map) {
    YAML::Node n;
    using key_t = typename std::decay_t<Map>::key_type;
    using value_t = typename std::decay_t<Map>::mapped_type;
    for (const auto &item : std::forward<decltype(map)>(map)) {
        // handle value types
        if constexpr (internal::is_instance<value_t, std::variant>::value) {
            std::visit(
                [&](const auto &v) {
                    using vt = std::decay_t<decltype(v)>;
                    if constexpr (std::is_same_v<vt, std::monostate>) {
                        n.push_back(YAML::Null);
                    } else {
                        n.push_back(std::map{std::pair{item.first, v}});
                    };
                },
                item.second);
        } else {
            n.push_back(std::map{std::pair{item.first, item.second}});
        }
    };
    n.SetTag(std::string{spec::ECSV_META_TAG});
    return n;
}

/// @brief Create map from meta, if viable.
/// @param meta The meta.
/// @param rest A node to capture any uncaptured entries in meta.
/// The map key type must be int-like or string, and the mapped type can be
/// int-like, floating point, string, std::optional, or std::variant
template <typename key_t, typename value_t>
auto meta_to_map(YAML::Node meta, YAML::Node *rest = nullptr)
    -> std::map<key_t, value_t> {

    using item_t = std::pair<key_t, value_t>;
    auto decode_item = [](const auto &n) -> std::optional<item_t> {
        item_t item{};
        if (!YAML::convert<key_t>::decode(n.first, item.first)) {
            return std::nullopt;
        }
        if (!YAML::convert<value_t>::decode(n.second, item.second)) {
            return std::nullopt;
        }
        return std::move(item);
    };
    std::map<key_t, value_t> result;
    YAML::Node rest_{}; // this is to collect any uncaptured item
    rest_.SetTag(meta.Tag());
    if (meta.IsSequence() && (meta.Tag() == spec::ECSV_META_TAG)) {
        // ordered map
        for (const auto &v : meta) {
            assert(v.IsMap() && v.size() == 1);
            const auto it = v.begin();
            if (auto opt_item = decode_item(*it); opt_item) {
                result[opt_item.value().first] = opt_item.value().second;
            } else if (rest != nullptr) {
                rest_.push_back(v);
            }
        }
    } else if (meta.IsMap()) {
        for (const auto &v : meta) {
            if (auto opt_item = decode_item(v); opt_item) {
                result[opt_item.value().first] = opt_item.value().second;
            } else if (rest != nullptr) {
                rest_.operator[](v.first) = v.second;
            }
        }
    } else {
        if (rest != nullptr) {
            rest_.reset(std::move(meta));
        }
    }
    if (rest != nullptr) {
        rest->reset(std::move(rest_));
    }
    return result;
}

} // namespace ecsv