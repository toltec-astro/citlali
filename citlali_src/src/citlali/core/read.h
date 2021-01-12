#pragma once

#include "../core/observation.h"
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

namespace aztec {

template<typename Numeric>
bool is_number(const std::string& s)
{
    Numeric n;
    return((std::istringstream(s) >> n >> std::ws).eof());
}

struct DataIOError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};
using Metadata = lali::FlatConfig;

struct BeammapData {

    Eigen::MatrixXd scans;
    Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic> scanindex;

    ///std:map to hold telescope pointing and time matrices.
    std::map<std::string, Eigen::Matrix<double,Eigen::Dynamic,1>> telMetaData;

    Metadata meta;

    /*
    BeammapData() = default;
    ~BeammapData() = default;
    BeammapData(BeammapData&&) = default;
    BeammapData& operator=(BeammapData &&) = default;
    BeammapData(const BeammapData&) = delete;
    BeammapData& operator=(const BeammapData&) = delete;
    */

    static BeammapData fromNcFile(const std::string &filepath) {
        using namespace netCDF;
        using namespace netCDF::exceptions;

        std::vector<std::string> gbs(113);

        gbs = {"Data.AztecBackend.h1b1",
               "Data.AztecBackend.h1b10",
               "Data.AztecBackend.h1b11",
               "Data.AztecBackend.h1b15",
               "Data.AztecBackend.h1b17",
               "Data.AztecBackend.h1b18",
               "Data.AztecBackend.h1b19",
               "Data.AztecBackend.h1b2",
               "Data.AztecBackend.h1b20",
               "Data.AztecBackend.h1b22",
               "Data.AztecBackend.h1b24",
               "Data.AztecBackend.h1b4",
               "Data.AztecBackend.h1b5",
               "Data.AztecBackend.h1b6",
               "Data.AztecBackend.h1b7",
               "Data.AztecBackend.h1b9",
               "Data.AztecBackend.h2b10",
               "Data.AztecBackend.h2b11",
               "Data.AztecBackend.h2b13",
               "Data.AztecBackend.h2b15",
               "Data.AztecBackend.h2b16",
               "Data.AztecBackend.h2b17",
               "Data.AztecBackend.h2b18",
               "Data.AztecBackend.h2b19",
               "Data.AztecBackend.h2b2",
               "Data.AztecBackend.h2b20",
               "Data.AztecBackend.h2b22",
               "Data.AztecBackend.h2b23",
               "Data.AztecBackend.h2b24",
               "Data.AztecBackend.h2b3",
               "Data.AztecBackend.h2b4",
               "Data.AztecBackend.h2b5",
               "Data.AztecBackend.h2b6",
               "Data.AztecBackend.h2b8",
               "Data.AztecBackend.h2b9",
               "Data.AztecBackend.h3b1",
               "Data.AztecBackend.h3b10",
               "Data.AztecBackend.h3b11",
               "Data.AztecBackend.h3b12",
               "Data.AztecBackend.h3b13",
               "Data.AztecBackend.h3b14",
               "Data.AztecBackend.h3b15",
               "Data.AztecBackend.h3b16",
               "Data.AztecBackend.h3b17",
               "Data.AztecBackend.h3b18",
               "Data.AztecBackend.h3b19",
               "Data.AztecBackend.h3b2",
               "Data.AztecBackend.h3b20",
               "Data.AztecBackend.h3b23",
               "Data.AztecBackend.h3b24",
               "Data.AztecBackend.h3b3",
               "Data.AztecBackend.h3b4",
               "Data.AztecBackend.h3b5",
               "Data.AztecBackend.h3b6",
               "Data.AztecBackend.h3b7",
               "Data.AztecBackend.h3b9",
               "Data.AztecBackend.h4b1",
               "Data.AztecBackend.h4b10",
               "Data.AztecBackend.h4b11",
               "Data.AztecBackend.h4b12",
               "Data.AztecBackend.h4b13",
               "Data.AztecBackend.h4b14",
               "Data.AztecBackend.h4b15",
               "Data.AztecBackend.h4b17",
               "Data.AztecBackend.h4b18",
               "Data.AztecBackend.h4b19",
               "Data.AztecBackend.h4b2",
               "Data.AztecBackend.h4b20",
               "Data.AztecBackend.h4b21",
               "Data.AztecBackend.h4b22",
               "Data.AztecBackend.h4b23",
               "Data.AztecBackend.h4b24",
               "Data.AztecBackend.h4b3",
               "Data.AztecBackend.h4b4",
               "Data.AztecBackend.h4b5",
               "Data.AztecBackend.h4b6",
               "Data.AztecBackend.h4b7",
               "Data.AztecBackend.h4b8",
               "Data.AztecBackend.h4b9",
               "Data.AztecBackend.h5b1",
               "Data.AztecBackend.h5b10",
               "Data.AztecBackend.h5b11",
               "Data.AztecBackend.h5b12",
               "Data.AztecBackend.h5b13",
               "Data.AztecBackend.h5b14",
               "Data.AztecBackend.h5b15",
               "Data.AztecBackend.h5b17",
               "Data.AztecBackend.h5b20",
               "Data.AztecBackend.h5b21",
               "Data.AztecBackend.h5b22",
               "Data.AztecBackend.h5b23",
               "Data.AztecBackend.h5b24",
               "Data.AztecBackend.h5b4",
               "Data.AztecBackend.h5b5",
               "Data.AztecBackend.h5b6",
               "Data.AztecBackend.h5b7",
               "Data.AztecBackend.h5b8",
               "Data.AztecBackend.h5b9",
               "Data.AztecBackend.h6b1",
               "Data.AztecBackend.h6b10",
               "Data.AztecBackend.h6b12",
               "Data.AztecBackend.h6b15",
               "Data.AztecBackend.h6b17",
               "Data.AztecBackend.h6b18",
               "Data.AztecBackend.h6b19",
               "Data.AztecBackend.h6b2",
               "Data.AztecBackend.h6b20",
               "Data.AztecBackend.h6b21",
               "Data.AztecBackend.h6b22",
               "Data.AztecBackend.h6b23",
               "Data.AztecBackend.h6b4",
               "Data.AztecBackend.h6b6",
               "Data.AztecBackend.h6b7"};


        try {
            SPDLOG_INFO("read aztec beammap from netCDF file {}", filepath);
            NcFile fo(filepath, NcFile::read);
            //SPDLOG_INFO("{}", nc_pprint(fo));
            // go over the vars, get the detector variables
            auto vars = fo.getVars();
            std::set<std::string> dv_names;
            auto matched = 0;
            int k=0;
            int gi = 0;
            for (const auto& var: vars) {
                if (std::regex_match(
                            var.first,
                            std::regex("Data\\.AztecBackend\\.h\\d+b\\d+"))
                        ) {
                    ++matched;
                    // check good flag
                    //if (nc_getatt<int16_t>(var.second.getAtt("goodflag")) > 0) {
                    if(std::find(gbs.begin(), gbs.end(), var.first) != gbs.end()) {
                        dv_names.insert(var.first);
                        gi++;
                    }
                    k++;
                }
            }
            int ndetectors = dv_names.size();
            SPDLOG_INFO("number of good detector: {} out of {}", ndetectors, matched);
            // get number of scans per detector, and check if they are the same
            auto npts = -1;  //
            for (const auto& name: dv_names) {
                auto vs = vars.equal_range(name);
                if (auto n = std::distance(vs.first, vs.second); n != 1) {
                    throw std::runtime_error(fmt::format("detector {} has {} entries", name, n < 1?"no": "duplicated"));
                }
                const auto& var = vs.first->second;
                auto tmp = var.getDim(0).getSize() * var.getDim(1).getSize();
                if (npts < 0) {
                    npts = tmp;
                } else if (npts != tmp) {
                    throw std::runtime_error(
                        fmt::format("detector {} has mismatch npts {} with others {}", name, tmp, npts));
                }
            }
            SPDLOG_INFO("number of pts: {}", npts);
            // create data container
            BeammapData data;
            data.scans.resize(npts, ndetectors);
            // fill the container
            auto i = 0;
            for (const auto& name: dv_names) {
                const auto& var = vars.find(name)->second;
                var.getVar(data.scans.col(i).data());
                ++i;
            }

            data.telMetaData["Hold"].resize(npts);
            vars.find("Data.AztecBackend.Hold")->second.getVar(data.telMetaData["Hold"].data());
            //SPDLOG_INFO("hold {}", data.telMetaData["Hold"]);

            data.telMetaData["TelUtc"].resize(npts);
            vars.find("Data.AztecBackend.TelUtc")->second.getVar(data.telMetaData["TelUtc"].data());
            //SPDLOG_INFO("TelUtc {}", data.telMetaData["TelUtc"]);

            data.telMetaData["AztecUtc"].resize(npts);
            vars.find("Data.AztecBackend.AztecUtc")->second.getVar(data.telMetaData["AztecUtc"].data());
            //SPDLOG_INFO("AztecUtc {}", data.telMetaData["AztecUtc"]);

            data.telMetaData["TelAzAct"].resize(npts);
            vars.find("Data.AztecBackend.TelAzAct")->second.getVar(data.telMetaData["TelAzAct"].data());
            //SPDLOG_INFO("TelAzAct{}", data.telMetaData["TelAzAct"]);

            data.telMetaData["TelElAct"].resize(npts);
            vars.find("Data.AztecBackend.TelElAct")->second.getVar(data.telMetaData["TelElAct"].data());
            //SPDLOG_INFO("TelElAct{}", data.telMetaData["TelElAct"]);

            data.telMetaData["TelAzDes"].resize(npts);
            vars.find("Data.AztecBackend.TelAzDes")->second.getVar(data.telMetaData["TelAzDes"].data());
            //SPDLOG_INFO("TelAzDes{}", data.telMetaData["TelAzDes"]);

            data.telMetaData["TelElDes"].resize(npts);
            vars.find("Data.AztecBackend.TelElDes")->second.getVar(data.telMetaData["TelElDes"].data());
            //SPDLOG_INFO("TelElDes{}", data.telMetaData["TelElDes"]);

            data.telMetaData["TelAzCor"].resize(npts);
            vars.find("Data.AztecBackend.TelAzCor")->second.getVar(data.telMetaData["TelAzCor"].data());
            //SPDLOG_INFO("TelAzCor{}", data.telMetaData["TelAzCor"]);

            data.telMetaData["TelElCor"].resize(npts);
            vars.find("Data.AztecBackend.TelElCor")->second.getVar(data.telMetaData["TelElCor"].data());
            //SPDLOG_INFO("TelElCor{}", data.telMetaData["TelElCor"]);

            data.telMetaData["SourceAz"].resize(npts);
            vars.find("Data.AztecBackend.SourceAz")->second.getVar(data.telMetaData["SourceAz"].data());
            //SPDLOG_INFO("SourceAz{}", data.telMetaData["SourceAz"]);

            data.telMetaData["SourceEl"].resize(npts);
            vars.find("Data.AztecBackend.SourceEl")->second.getVar(data.telMetaData["SourceEl"].data());
            //SPDLOG_INFO("SourceEl{}", data.telMetaData["SourceEl"]);

            data.telMetaData["TelRa"].resize(npts);
            vars.find("Data.AztecBackend.SourceRaAct")->second.getVar(data.telMetaData["TelRa"].data());
            //SPDLOG_INFO("TelRa{}", data.telMetaData["TelRa"]);

            data.telMetaData["TelDec"].resize(npts);
            vars.find("Data.AztecBackend.SourceDecAct")->second.getVar(data.telMetaData["TelDec"].data());
            //SPDLOG_INFO("TelDec{}", data.telMetaData["TelDec"]);

            data.telMetaData["ParAng"].resize(npts);
            vars.find("Data.AztecBackend.ParAng")->second.getVar(data.telMetaData["ParAng"].data());
            //SPDLOG_INFO("ParAng{}", data.telMetaData["ParAng"]);

            data.telMetaData["ParAng"] = pi-data.telMetaData["ParAng"].array();

            observation::obs(data.scanindex,data.telMetaData,40,64,0.125);

            // meta data
            int nscans = data.scanindex.cols();
            data.meta.set("source", filepath);
            data.meta.set("npts", npts);
            data.meta.set("ndetectors", ndetectors);
            data.meta.set("nscans", nscans);
            SPDLOG_INFO("scans{}", data.scans);
            SPDLOG_INFO("scanindex{}", data.scanindex);
            SPDLOG_INFO("nscans {}", nscans);


            return std::move(data);
        } catch (NcException &e) {
            SPDLOG_ERROR("{}", e.what());
            throw DataIOError{fmt::format(
                "failed to load data from netCDF file {}", filepath)};
        }
    }
};

} // namespace aztec
