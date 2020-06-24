#pragma once

#include "formatter/byte.h"
#include "formatter/matrix.h"
#include "logging.h"
#include "meta.h"
#include <netcdf>
#include <sstream>
#include <variant>

namespace nc_utils {

namespace internal {

constexpr auto nctypes = std::array{
    netCDF::NcType::ncType::nc_BYTE,  netCDF::NcType::ncType::nc_CHAR,
    netCDF::NcType::ncType::nc_SHORT, netCDF::NcType::ncType::nc_INT,
    netCDF::NcType::ncType::nc_FLOAT, netCDF::NcType::ncType::nc_DOUBLE};

template <auto nctype>
using type_t =
    meta::switch_t<nctype,
                   meta::case_t<netCDF::NcType::ncType::nc_BYTE, std::byte>,
                   meta::case_t<netCDF::NcType::ncType::nc_CHAR, char>,
                   meta::case_t<netCDF::NcType::ncType::nc_SHORT, int16_t>,
                   meta::case_t<netCDF::NcType::ncType::nc_INT, int32_t>,
                   meta::case_t<netCDF::NcType::ncType::nc_FLOAT, float>,
                   meta::case_t<netCDF::NcType::ncType::nc_DOUBLE, double>>;

template <typename T>
inline constexpr auto nctype_v = meta::select_t<
    meta::cond_t<std::is_same_v<std::byte, T>,
                 std::integral_constant<netCDF::NcType::ncType,
                                        netCDF::NcType::ncType::nc_BYTE>>,
    meta::cond_t<std::is_same_v<char, T>,
                 std::integral_constant<netCDF::NcType::ncType,
                                        netCDF::NcType::ncType::nc_CHAR>>,
    meta::cond_t<std::is_same_v<int16_t, T>,
                 std::integral_constant<netCDF::NcType::ncType,
                                        netCDF::NcType::ncType::nc_SHORT>>,
    meta::cond_t<std::is_same_v<int32_t, T>,
                 std::integral_constant<netCDF::NcType::ncType,
                                        netCDF::NcType::ncType::nc_INT>>,
    meta::cond_t<std::is_same_v<float, T>,
                 std::integral_constant<netCDF::NcType::ncType,
                                        netCDF::NcType::ncType::nc_FLOAT>>,
    meta::cond_t<std::is_same_v<double, T>,
                 std::integral_constant<netCDF::NcType::ncType,
                                        netCDF::NcType::ncType::nc_DOUBLE>>>::
    value;

template <std::size_t... Is>
std::variant<type_t<std::get<Is>(nctypes)>...>
type_variant_impl(std::index_sequence<Is...>);
using type_variant_t = decltype(type_variant_impl(
    std::declval<std::make_index_sequence<nctypes.size()>>()));

inline type_variant_t dispatch_as_variant(netCDF::NcType::ncType t) {
    type_variant_t var{};
    using T = netCDF::NcType::ncType;
    switch (t) {
    case T::nc_BYTE: {
        var = type_t<T::nc_BYTE>{};
        break;
    }
    case T::nc_CHAR: {
        var = type_t<T::nc_CHAR>{};
        break;
    }
    case T::nc_SHORT: {
        var = type_t<T::nc_SHORT>{};
        break;
    }
    case T::nc_INT: {
        var = type_t<T::nc_INT>{};
        break;
    }
    case T::nc_FLOAT: {
        var = type_t<T::nc_FLOAT>{};
        break;
    }
    case T::nc_DOUBLE: {
        var = type_t<T::nc_DOUBLE>{};
        break;
    }
    default: {
        throw std::runtime_error(
            fmt::format("dispatch of type {} not implemented",
                        netCDF::NcType{t}.getName()));
    }
    }
    return var;
}

} // namespace internal

/*
 * @brief Visit variable with strong type.
 * @param func Callable with signature func(var, T{})
 * @param var NC variable.
 */
template <typename F, typename T>
auto visit(F &&func, T &&var) -> decltype(auto) {
    return std::visit(
        [args = FWD_CAPTURE(func, var)](auto t) {
            auto &&[func, var] = args;
            return std::invoke(FWD(func), FWD(var), t);
        },
        internal::dispatch_as_variant(var.getType().getTypeClass()));
}

template <typename T, typename Var> void validate_type(const Var &var) {
    constexpr auto expected_type_class = internal::nctype_v<T>;
    constexpr std::string_view kind = []() {
        if constexpr (std::is_base_of_v<netCDF::NcAtt, Var>) {
            return "attribute";
        } else {
            return "variable";
        }
    }();
    const auto &type = var.getType();
    const auto &name = var.getName();
    if (expected_type_class != type.getTypeClass()) {
        throw std::runtime_error(fmt::format(
            "mismatch {} {} of type {} with buffer of type {}", kind, name,
            type.getName(), netCDF::NcType{expected_type_class}.getName()));
    }
}

template <typename Att, typename Buffer,
          REQUIRES_(std::is_base_of<netCDF::NcAtt, Att>)>
void getattr(const Att &att, Buffer &buf) {
    auto name = att.getName();
    auto type = att.getType();
    validate_type<typename Buffer::value_type>(att);
    if constexpr (std::is_same_v<Buffer, std::string>) {
        att.getValues(buf);
    } else {
        // check length
        auto alen = att.getAttLength();
        auto blen = buf.size();
        if (alen == blen) {
            att.getValues(buf.data());
        } else {
            throw std::runtime_error(
                fmt::format("cannot get attr {} of len {} to buffer of len {}",
                            name, alen, blen));
        }
    }
}

template <typename T, typename Att,
          REQUIRES_(std::is_base_of<netCDF::NcAtt, Att>)>
T getattr(const Att &att) {
    using namespace netCDF;
    constexpr bool is_scalar =
        meta::is_one_of<T, internal::type_variant_t>::value;
    using Buffer = std::conditional_t<is_scalar, std::vector<T>, T>;
    auto len = att.getAttLength();
    // check scalar
    if constexpr (is_scalar) {
        if (len > 1) {
            throw std::runtime_error(
                fmt::format("cannot get attr {} of len {} as a scalar",
                            att.getName(), len));
        }
    }
    // get var
    Buffer buf;
    buf.resize(len);
    getattr(att, buf); // type is validated here
    if constexpr (is_scalar) {
        return buf[0];
    } else {
        return buf;
    }
}

template <typename T, REQUIRES_(meta::is_one_of<T, internal::type_variant_t>)>
T getscalar(const netCDF::NcVar &var) {
    validate_type<T>(var);
    if (var.getDimCount() != 0) {
        throw std::runtime_error(
            fmt::format("variable {} is not a scalar", var.getName()));
    }
    using namespace netCDF;
    // get var
    T buf;
    var.getVar(&buf);
    return buf;
}

inline std::string getstr(const netCDF::NcVar &var) {
    validate_type<char>(var);
    if (var.getDimCount() != 1) {
        throw std::runtime_error(
            fmt::format("variable {} is not a string", var.getName()));
    }
    using namespace netCDF;
    std::vector<char> buf(var.getDim(0).getSize());
    var.getVar(buf.data());
    return std::string(buf.data());
}

inline std::vector<std::string> getstrs(const netCDF::NcVar &var) {
    validate_type<char>(var);
    if (var.getDimCount() != 2) {
        throw std::runtime_error(
            fmt::format("variable {} is not a string vector", var.getName()));
    }
    using namespace netCDF;
    auto nstrs = var.getDim(0).getSize();
    auto nchars = var.getDim(1).getSize();
    std::vector<char> buf(nchars * nstrs);
    var.getVar(buf.data());
    std::vector<std::string> result;
    result.reserve(nstrs);
    for (std::size_t i = 0; i < nstrs; ++i) {
        result.emplace_back(std::string(buf.data() + i * nchars));
    }
    return result;
}

template <typename var_t_>
struct pprint {
    using var_t = var_t_;
    const var_t &nc;
    pprint(const var_t &nc_) : nc(nc_) {}

    static auto format_ncvaratt(const netCDF::NcVarAtt &att) {
        std::stringstream os;
        os << fmt::format("{}", att.getName());
        visit(
            [&os](const auto &att, auto t) {
                using T = DECAY(t);
                // handle char with string
                if constexpr (std::is_same_v<T, char>) {
                    auto buf = getattr<std::string>(att);
                    std::size_t maxlen = 70;
                    os << ": \"" << buf.substr(0, maxlen)
                       << (buf.size() > maxlen ? " ..." : "")
                       << fmt::format("\" ({})", att.getType().getName());
                    return;
                } else {
                    auto buf = getattr<std::vector<T>>(att);
                    os << " "
                       << fmt::format("{} ({})", buf, att.getType().getName());
                    return;
                }
            },
            att);
        return os.str();
    }
    static auto format_ncdim(const netCDF::NcDim &dim) {
        return fmt::format("{}({})", dim.getName(), dim.getSize());
    }
    static auto format_ncvar(const netCDF::NcVar &var,
                             std::size_t key_width = 0) {
        std::stringstream os;
        std::string namefmt{"{}"};
        if (key_width > 0) {
            namefmt = fmt::format("{{:>{}s}}", key_width);
        }
        os << fmt::format(" {}: ({})", fmt::format(namefmt, var.getName()),
                          var.getType().getName());
        auto dims = var.getDims();
        if (dims.size() > 0) {
            os << " [";
            for (auto it = dims.begin(); it != dims.end(); ++it) {
                if (it != dims.begin()) {
                    os << ", ";
                }
                os << format_ncdim(*it);
            }
            os << "]";
        }
        const auto &atts = var.getAtts();
        if (atts.size() > 0) {
            // os << " {";
            for (const auto &att : var.getAtts()) {
                os << fmt::format("\n {} {}", std::string(key_width + 1, ' '),
                                  format_ncvaratt(att.second));
            }
            // os << "}";
        }
        return os.str();
    }

    static auto format_ncfile(const netCDF::NcFile &fo) {
        std::stringstream os;
        // print out some info
        os << "info:\nsummary:\n"
           << fmt::format("    n_vars: {}\n", fo.getVarCount())
           << fmt::format("    n_atts: {}\n", fo.getAttCount())
           << fmt::format("    n_dims: {}\n", fo.getDimCount())
           << fmt::format("    n_grps: {}\n", fo.getGroupCount())
           << fmt::format("    n_typs: {}\n", fo.getTypeCount())
           << "variables:";
        const auto &vars = fo.getVars();
        auto max_key_width = 0;
        std::for_each(
            vars.begin(), vars.end(), [&max_key_width](auto &it) mutable {
                if (auto size = it.first.size(); size > max_key_width) {
                    max_key_width = size;
                }
            });
        for (const auto &var : fo.getVars()) {
            os << "\n" << pprint::format_ncvar(var.second, max_key_width);
        }
        os << "\n"
           << "dims:";
        for (const auto &dim : fo.getDims()) {
            os << "\n    " << pprint::format_ncdim(dim.second);
        }
        return os.str();
    }

    template <typename OStream>
    friend OStream &operator<<(OStream &os, const pprint &pp) {
        using var_t = typename pprint::var_t;
        if constexpr (std::is_same_v<var_t, netCDF::NcFile>) {
            return os << pprint::format_ncfile(pp.nc);
        } else if constexpr (std::is_same_v<var_t, netCDF::NcVar>) {
            return os << pprint::format_ncvar(pp.nc);
        } else if constexpr (std::is_same_v<var_t, netCDF::NcDim>) {
            return os << pprint::format_ncdim(pp.nc);
        } else if constexpr (std::is_same_v<var_t, netCDF::NcVarAtt>) {
            return os << pprint::format_ncvaratt(pp.nc);
        } else {
            static_assert(meta::always_false<var_t>::value,
                          "UNABLE TO FORMAT TYPE");
        }
    }
};

struct NcNodeMapper {
    using NcFile = netCDF::NcFile;
    using NcVar = netCDF::NcVar;
    using NcDim = netCDF::NcDim;
    using NcGroupAtt = netCDF::NcGroupAtt;
    using key_t = std::string_view;
    using node_t = std::variant<std::monostate, netCDF::NcFile, netCDF::NcVar,
                                netCDF::NcDim, netCDF::NcGroupAtt>;
    using keymap_t = std::unordered_map<key_t, std::string>;
    using nodemap_t = std::unordered_map<key_t, node_t>;

    NcNodeMapper(const NcFile &ncfile_, keymap_t keymap)
        : _(std ::move(keymap)), ncfile(ncfile_) {}

    template <typename T = node_t, typename... Args>
    bool has_node(key_t key, Args &&... keys) {
        if constexpr (std::is_same_v<T, node_t>) {
            for (const auto &k : {key, keys...}) {
                if (!(has_node<NcVar>(k) || has_node<NcDim>(k) ||
                      has_node<NcGroupAtt>(k))) {
                    return false;
                }
            }
            return true;
        } else {
            if constexpr (sizeof...(keys) == 0) {
                nodes.emplace(std::piecewise_construct, std::make_tuple(key),
                              std::make_tuple());
                auto getnode = [&]() {
                    if constexpr (std::is_same_v<T, NcVar>) {
                        return ncfile.getVar(_[key]);
                    } else if constexpr (std::is_same_v<T, NcDim>) {
                        return ncfile.getDim(_[key]);
                    } else if constexpr (std::is_same_v<T, NcGroupAtt>) {
                        return ncfile.getAtt(_[key]);
                    }
                };
                constexpr auto gettype = []() {
                    if constexpr (std::is_same_v<T, NcVar>) {
                        return "var";
                    } else if constexpr (std::is_same_v<T, NcDim>) {
                        return "dim";
                    } else if constexpr (std::is_same_v<T, NcGroupAtt>) {
                        return "att";
                    }
                };
                if (const auto &v = getnode(); !v.isNull()) {
                    SPDLOG_TRACE("{} key={} value={} ", gettype(), key,
                                 nc_utils::pprint{v});
                    nodes[key] = v;
                    return true;
                }
                SPDLOG_TRACE("{} key={} ({}) not found", gettype(), key,
                             _[key]);
                nodes.erase(key);
                return false;
            } else {
                for (const auto &k : {key, keys...}) {
                    if (!has_node<T>(k)) {
                        return false;
                    }
                }
                return true;
            }
        }
    }
    template <typename... Args> bool has_var(Args &&... keys) {
        return has_node<NcVar>(key_t{keys}...);
    }
    template <typename... Args> bool has_dim(Args &&... keys) {
        return has_node<NcDim>(key_t{keys}...);
    }
    template <typename... Args> bool has_att(Args &&... keys) {
        return has_node<NcGroupAtt>(key_t{keys}...);
    }
    template <typename T = node_t> const T &node(key_t key) {
        if constexpr (std::is_same_v<T, node_t>) {
            return nodes.at(key);
        } else {
            return std::get<T>(nodes.at(key));
        }
    }
    const auto &var(key_t key) { return node<NcVar>(key); }
    const auto &dim(key_t key) { return node<NcDim>(key); }
    const auto &att(key_t key) { return node<NcGroupAtt>(key); }
    keymap_t _;

private:
    nodemap_t nodes;
    const netCDF::NcFile &ncfile;
};

} // namespace  nc_utils
