#pragma once

#include <fmt/ostream.h>
#include <netcdf>
#include <regex>

template <
    typename R, typename T,
    typename = std::enable_if_t<std::is_base_of_v<netCDF::NcAtt, T>>,
    typename Buffer = std::enable_if_t<
        std::is_arithmetic_v<R> || meta::is_instance<R, std::vector>::value,
        std::conditional_t<std::is_arithmetic_v<R>, std::vector<R>, R>>>
R nc_getatt(const T &att) {
    using namespace netCDF;
    auto tid = att.getType().getId();
    constexpr bool scalar = std::is_arithmetic_v<R>;
    // check length
    auto len = att.getAttLength();
    if constexpr (scalar) {
        if (len > 1) {
            throw std::runtime_error("attribute is not a scalar");
        }
    }
    // buffer
    Buffer buf(len);
    auto get_typed = [&] (const NcType& t) {
        if (t.getId() == tid) {
            att.getValues(buf.data());
            if constexpr (scalar) {
                return buf[0];
            } else {
                return buf;
            }
        }
            throw std::runtime_error(fmt::format(
                             "attribute type mismatch: actual={} requested={}",
                                             att.getType().getName(),
                                             t.getName()
                                             ));
        };
    if constexpr (std::is_same_v<R, char>) {
        return get_typed(NcType::ncType::nc_CHAR);
    } else if constexpr(std::is_same_v<R, int16_t>) {
        return get_typed(NcType::ncType::nc_SHORT);
    } else if constexpr(std::is_same_v<R, int32_t>) {
        return get_typed(NcType::ncType::nc_INT);
    } else if constexpr(std::is_same_v<R, float>) {
        return get_typed(NcType::ncType::nc_FLOAT);
    } else if constexpr(std::is_same_v<R, double>) {
        return get_typed(NcType::ncType::nc_DOUBLE);
    } else {
        static_assert(meta::always_false<R>::value, "UNABLE TO HANDLE TYPE");
    }
}

template<typename T>
struct nc_pprint {
    using var_t = typename std::decay<T>::type;
    const T & nc;
    nc_pprint(const T& nc_): nc(nc_) {}
    static decltype(auto) format_ncvaratt(const netCDF::NcVarAtt& att) {
        std::stringstream os;
        os << fmt::format("{}[{}]", att.getName(), att.getType().getName());
        auto len = att.getAttLength();
        switch (att.getType().getId()) {
            case netCDF::NcType::ncType::nc_CHAR: {
                std::string buf;
                att.getValues(buf);
                auto maxlen = 20;
                os << " = \"" << buf.substr(0, maxlen) << (len > maxlen?" ...":"") << "\"";
                break;
            }
            case netCDF::NcType::ncType::nc_SHORT: {
                std::vector<int16_t> buf(len);
                att.getValues(buf.data());
                os << " = " << buf;
                break;
            }
            case netCDF::NcType::ncType::nc_INT: {
                std::vector<int32_t> buf(len);
                att.getValues(buf.data());
                os << "=" << buf;
                break;
            }
            case netCDF::NcType::ncType::nc_FLOAT: {
                std::vector<float> buf(len);
                att.getValues(buf.data());
                os << "=" << buf;
                break;
            }
            case netCDF::NcType::ncType::nc_DOUBLE: {
                std::vector<double> buf(len);
                att.getValues(buf.data());
                os << "=" << buf;
                break;
            }
            default: {
                throw std::runtime_error(fmt::format("unknown attribute type {}", att.getType().getName()));
            }
        }
        return os.str();
    }
    static decltype(auto) format_ncvar(const netCDF::NcVar& var) {
        std::stringstream os;
        os << fmt::format("var({}, {})[", var.getName(), var.getType().getName());
        auto dims = var.getDims();
        for (auto it = dims.begin(); it != dims.end(); ++it) {
            if (it != dims.begin()) os << ", ";
            os << fmt::format("{}({})", it->getName(), it->getSize());
        }
        os << "]{";
        for (const auto& att: var.getAtts()) {
            os << fmt::format("\n   {}", format_ncvaratt(att.second));
        }
        os << "}";
        return os.str();
    }
    static decltype(auto) format_ncfile(const netCDF::NcFile& fo) {
        std::stringstream os;
        // print out some info
        os << "------------------\n"
           << fmt::format("n_vars = {}\n", fo.getVarCount())
           << fmt::format("n_atts = {}\n", fo.getAttCount())
           << fmt::format("n_dims = {}\n", fo.getDimCount())
           << fmt::format("n_grps = {}\n", fo.getGroupCount())
           << fmt::format("n_typs = {}\n", fo.getTypeCount())
           << "------------------\nvars:";
        for (const auto& var: fo.getVars()) {
            os << "\n" << nc_pprint::format_ncvar(var.second);
        }
        return os.str();
    }
    template<typename OStream>
    friend OStream& operator << (OStream& os, const nc_pprint& pp) {
        using var_t = typename nc_pprint::var_t;
        if constexpr (std::is_same_v<var_t, netCDF::NcFile>) {
            return os << nc_pprint::format_ncfile(pp.nc);
        }  else if constexpr (std::is_same_v<var_t, netCDF::NcVar>) {
            return os << nc_pprint::format_ncvar(pp.nc);
        } else if constexpr (std::is_same_v<var_t, netCDF::NcVarAtt>) {
            return os << nc_pprint::format_ncvaratt(pp.nc);
        } else {
            static_assert (meta::always_false<var_t>::value, "UNABLE TO FORMAT TYPE");
        }
    }
};

namespace lali {

template<typename Numeric>
bool is_number(const std::string& s)
{
    Numeric n;
    return((std::istringstream(s) >> n >> std::ws).eof());
}

struct DataIOError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};
//using Metadata = lali::FlatConfig;

struct TelData {

    std::map<std::string, Eigen::Matrix<double,Eigen::Dynamic,1>> telMetaData;
    std::map <std::string, Eigen::VectorXd> srcCenter;

    static TelData fromNcFile(const std::string &filepath) {
        using namespace netCDF;
        using namespace netCDF::exceptions;


        try {
            SPDLOG_INFO("Read Telescope netCDF file {}", filepath);
            NcFile fo(filepath, NcFile::read, NcFile::classic);
            auto vars = fo.getVars();

            TelData data;

            Eigen::Index TelTime_npts = vars.find("Data.TelescopeBackend.TelTime")->second.getDim(0).getSize();
            data.telMetaData["TelTime"].resize(TelTime_npts);
            vars.find("Data.TelescopeBackend.TelTime")->second.getVar(data.telMetaData["TelTime"].data());
            // SPDLOG_INFO("TelTime {}", data.telMetaData["TelTime"]);

            Eigen::Index TelSourceRaAct_npts = vars.find("Data.TelescopeBackend.TelSourceRaAct")->second.getDim(0).getSize();
            data.telMetaData["TelRa"].resize(TelSourceRaAct_npts);
            vars.find("Data.TelescopeBackend.TelSourceRaAct")->second.getVar(data.telMetaData["TelRa"].data());
            data.telMetaData["TelRa"] = data.telMetaData["TelRa"]*DEG_TO_RAD;
             SPDLOG_INFO("TelRa {}", data.telMetaData["TelRa"]);

            Eigen::Index TelSourceDecAct_npts = vars.find("Data.TelescopeBackend.TelSourceDecAct")->second.getDim(0).getSize();
            data.telMetaData["TelDec"].resize(TelSourceDecAct_npts);
            vars.find("Data.TelescopeBackend.TelSourceDecAct")->second.getVar(data.telMetaData["TelDec"].data());
            data.telMetaData["TelDec"] = data.telMetaData["TelDec"]*DEG_TO_RAD;
             SPDLOG_INFO("TelDec {}", data.telMetaData["TelDec"]);

            Eigen::Index TelAzAct_npts = vars.find("Data.TelescopeBackend.TelAzAct")->second.getDim(0).getSize();
            data.telMetaData["TelAzAct"].resize(TelAzAct_npts);
            vars.find("Data.TelescopeBackend.TelAzAct")->second.getVar(data.telMetaData["TelAzAct"].data());
            data.telMetaData["TelAzAct"] = data.telMetaData["TelAzAct"]*DEG_TO_RAD;
             SPDLOG_INFO("TelAzAct {}", data.telMetaData["TelAzAct"]);

            Eigen::Index TelElAct_npts = vars.find("Data.TelescopeBackend.TelElAct")->second.getDim(0).getSize();
            data.telMetaData["TelElAct"].resize(TelElAct_npts);
            vars.find("Data.TelescopeBackend.TelElAct")->second.getVar(data.telMetaData["TelElAct"].data());
            data.telMetaData["TelElAct"] = data.telMetaData["TelElAct"]*DEG_TO_RAD;
             SPDLOG_INFO("TelElAct {}", data.telMetaData["TelElAct"]);

            Eigen::Index ActParAng_npts = vars.find("Data.TelescopeBackend.ActParAng")->second.getDim(0).getSize();
            data.telMetaData["ParAng"].resize(ActParAng_npts);
            vars.find("Data.TelescopeBackend.ActParAng")->second.getVar(data.telMetaData["ParAng"].data());
            data.telMetaData["ParAng"] = data.telMetaData["ParAng"]*DEG_TO_RAD;
            data.telMetaData["ParAng"] = pi-data.telMetaData["ParAng"].array();

             SPDLOG_INFO("ActParAng {}", data.telMetaData["ParAng"]);

            Eigen::Index hold_npts = vars.find("Data.TelescopeBackend.Hold")->second.getDim(0).getSize();
            data.telMetaData["Hold"].resize(hold_npts);
            vars.find("Data.TelescopeBackend.Hold")->second.getVar(data.telMetaData["Hold"].data());
             SPDLOG_INFO("hold {}", data.telMetaData["Hold"]);

            data.srcCenter["centerRa"].resize(2);
            data.srcCenter["centerRa"].setZero();
            data.srcCenter["centerRa"](0) = 92.0*DEG_TO_RAD;
            // vars.find("Header.Source.Ra")->second.getVar(data.srcCenter["centerRa"].data());

            data.srcCenter["centerDec"].resize(2);
            data.srcCenter["centerDec"].setZero();
            data.srcCenter["centerDec"](0) = -7.0*DEG_TO_RAD;

            // vars.find("Header.Source.Dec")->second.getVar(data.srcCenter["centerDec"].data());

            /* TEMP */
            data.telMetaData["TelAzCor"].setZero(TelAzAct_npts);
            data.telMetaData["TelElCor"].setZero(TelElAct_npts);

            data.telMetaData["TelAzDes"] = data.telMetaData["TelAzAct"];
            data.telMetaData["TelElDes"] = -data.telMetaData["TelElAct"];

            data.telMetaData["SourceAz"] = data.telMetaData["TelAzAct"];
            data.telMetaData["SourceEl"] = data.telMetaData["TelElAct"];

            return std::move(data);
        } catch (NcException &e) {
            SPDLOG_ERROR("{}", e.what());
            throw DataIOError{fmt::format(
                "failed to load data from netCDF file {}", filepath)};
        }
    }
};

} // namespace lali
