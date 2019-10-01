#pragma once

#include <Eigen/Core>
#include <netcdf>
#include <fmt/ostream.h>
#include <regex>

#include "../core/timestream/observation.h"

template<typename R, typename T, typename =std::enable_if_t<
             std::is_base_of_v<netCDF::NcAtt, T>>, typename Buffer=
         std::enable_if_t<
             std::is_arithmetic_v<R> || meta::is_instance<R, std::vector>::value,
             std::conditional_t<std::is_arithmetic_v<R>, std::vector<R>,
                R>
             >
         >
R nc_getatt (const T& att) {
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
                os << " = " << logging::pprint(buf);
                break;
            }
            case netCDF::NcType::ncType::nc_INT: {
                std::vector<int32_t> buf(len);
                att.getValues(buf.data());
                os << "=" << logging::pprint(buf);
                break;
            }
            case netCDF::NcType::ncType::nc_FLOAT: {
                std::vector<float> buf(len);
                att.getValues(buf.data());
                os << "=" << logging::pprint(buf);
                break;
            }
            case netCDF::NcType::ncType::nc_DOUBLE: {
                std::vector<double> buf(len);
                att.getValues(buf.data());
                os << "=" << logging::pprint(buf);
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
using Metadata = config::Config;

struct BeammapData {

    Eigen::MatrixXd scans;
    Eigen::MatrixXd scans_temp;

    Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic> scanindex;

    ///std:map to hold telescope pointing and time matrices.
    std::map<std::string, Eigen::Matrix<double,Eigen::Dynamic,1>> telescope_data;

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

        std::vector<int> gbs;

        gbs = { 0,   1,   3,   4,   5,   6,   8,   9,  10,  14,  16,  17,  18,
             19,  21,  23,  25,  26,  27,  28,  29,  31,  32,  33,  34,  36,
             38,  39,  40,  41,  42,  43,  45,  46,  47,  48,  49,  50,  51,
             52,  53,  54,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
             66,  67,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,
             81,  82,  83,  84,  85,  86,  88,  89,  90,  91,  92,  93,  94,
             95,  96,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
            110, 112, 115, 116, 117, 118, 119, 120, 121, 123, 125, 126, 129,
            131, 134, 136, 137, 138, 139, 140, 141, 142};


        try {
            SPDLOG_INFO("read aztec beammap from netCDF file {}", filepath);
            NcFile fo(filepath, NcFile::read);
            SPDLOG_INFO("{}", nc_pprint(fo));
            // go over the vars, get the detector variables
            auto vars = fo.getVars();
            std::set<std::string> dv_names;
            auto matched = 0;
            int k=0;
            for (const auto& var: vars) {
                if (std::regex_match(
                            var.first,
                            std::regex("Data\\.AztecBackend\\.h\\d+b\\d+"))
                        ) {
                    ++matched;
                    // check good flag
                    if (nc_getatt<int16_t>(var.second.getAtt("goodflag")) > 0) {
                    //if(std::find(gbs.begin(), gbs.end(), k) != gbs.end()) {

                      //   cerr << "k " << k << endl;
                        dv_names.insert(var.first);
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
            data.scans_temp.resize(npts, ndetectors);
            // fill the container
            auto i = 0;
            for (const auto& name: dv_names) {
                const auto& var = vars.find(name)->second;
                var.getVar(data.scans_temp.col(i).data());
                ++i;
            }

            data.telescope_data["Hold"].resize(npts);
            vars.find("Data.AztecBackend.Hold")->second.getVar(data.telescope_data["Hold"].data());
            SPDLOG_INFO("hold {}", logging::pprint(data.telescope_data["Hold"]));

            data.telescope_data["TelUtc"].resize(npts);
            vars.find("Data.AztecBackend.TelUtc")->second.getVar(data.telescope_data["TelUtc"].data());
            SPDLOG_INFO("TelUtc {}", logging::pprint(data.telescope_data["TelUtc"]));

            data.telescope_data["AztecUtc"].resize(npts);
            vars.find("Data.AztecBackend.AztecUtc")->second.getVar(data.telescope_data["AztecUtc"].data());
            SPDLOG_INFO("AztecUtc {}", logging::pprint(data.telescope_data["AztecUtc"]));

            /*data.telescope_data["SourceRaAct"].resize(npts);
            vars.find("Data.AztecBackend.SourceRaAct")->second.getVar(data.telescope_data["SourceRaAct"].data());
            SPDLOG_INFO("TelRa{}", logging::pprint(data.telescope_data["SourceRaAct"]));

            data.telescope_data["SourceDecAct"].resize(npts);
            vars.find("Data.AztecBackend.SourceDecAct")->second.getVar(data.telescope_data["SourceDecAct"].data());
            SPDLOG_INFO("TelDec{}", logging::pprint(data.telescope_data["SourceDecAct"]));*/

            data.telescope_data["TelAzAct"].resize(npts);
            vars.find("Data.AztecBackend.TelAzAct")->second.getVar(data.telescope_data["TelAzAct"].data());
            SPDLOG_INFO("TelAzAct{}", logging::pprint(data.telescope_data["TelAzAct"]));

            data.telescope_data["TelElAct"].resize(npts);
            vars.find("Data.AztecBackend.TelElAct")->second.getVar(data.telescope_data["TelElAct"].data());
            SPDLOG_INFO("TelElAct{}", logging::pprint(data.telescope_data["TelElAct"]));

            data.telescope_data["TelAzDes"].resize(npts);
            vars.find("Data.AztecBackend.TelAzDes")->second.getVar(data.telescope_data["TelAzDes"].data());
            SPDLOG_INFO("TelAzDes{}", logging::pprint(data.telescope_data["TelAzDes"]));

            data.telescope_data["TelElDes"].resize(npts);
            vars.find("Data.AztecBackend.TelElDes")->second.getVar(data.telescope_data["TelElDes"].data());
            SPDLOG_INFO("TelElDes{}", logging::pprint(data.telescope_data["TelElDes"]));

            data.telescope_data["TelAzCor"].resize(npts);
            vars.find("Data.AztecBackend.TelAzCor")->second.getVar(data.telescope_data["TelAzCor"].data());
            SPDLOG_INFO("TelAzCor{}", logging::pprint(data.telescope_data["TelAzCor"]));

            data.telescope_data["TelElCor"].resize(npts);
            vars.find("Data.AztecBackend.TelElCor")->second.getVar(data.telescope_data["TelElCor"].data());
            SPDLOG_INFO("TelElCor{}", logging::pprint(data.telescope_data["TelElCor"]));

            data.telescope_data["SourceAz"].resize(npts);
            vars.find("Data.AztecBackend.SourceAz")->second.getVar(data.telescope_data["SourceAz"].data());
            SPDLOG_INFO("SourceAz{}", logging::pprint(data.telescope_data["SourceAz"]));

            data.telescope_data["SourceEl"].resize(npts);
            vars.find("Data.AztecBackend.SourceEl")->second.getVar(data.telescope_data["SourceEl"].data());
            SPDLOG_INFO("SourceEl{}", logging::pprint(data.telescope_data["SourceEl"]));

            data.telescope_data["ParAng"].resize(npts);
            vars.find("Data.AztecBackend.ParAng")->second.getVar(data.telescope_data["ParAng"].data());
            SPDLOG_INFO("ParAng{}", logging::pprint(data.telescope_data["ParAng"]));

            data.telescope_data["ParAng"] = pi-data.telescope_data["ParAng"].array();

            observation::obs(data.scanindex,data.telescope_data,0,64,0.125);

            //Toggle commenting here to generate 4000 detectors (40 copies of first 100 detectors)
            Eigen::Index nmulti = 40;
            Eigen::Index j = 0;
            data.scans.resize(npts,4000);

            for (Eigen::Index i = 0;i <nmulti;i++) {
                data.scans.block(0,j,npts,100) = data.scans_temp.block(0,0,npts,100);
                j = j + 100;
            }

            //ndetectors = 4000;
            //end commenting for 4000 detectors

            data.scans = data.scans_temp; //temporary,  uncomment this and comment above block for normal reading of file
            data.scans_temp.resize(0,0);

            // meta data
            int nscans = data.scanindex.cols();
            data.meta.set("source", filepath);
            data.meta.set("npts", npts);
            data.meta.set("ndetectors", ndetectors);
            data.meta.set("nscans", nscans);
            SPDLOG_INFO("scans{}", logging::pprint(data.scans));
            SPDLOG_INFO("scanindex{}", logging::pprint(data.scanindex));
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
