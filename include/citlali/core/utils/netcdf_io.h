#pragma once

#include <fmt/ostream.h>
#include <netcdf>
#include <regex>

#include <Eigen/Core>

#include <tula/logging.h>
#include <tula/nc.h>

#include <citlali/core/utils/constants.h>

struct DataIOError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD, typename DerivedE, typename C, typename T>
void append_to_netcdf(std::string filepath, Eigen::DenseBase<DerivedA> &data, Eigen::DenseBase<DerivedB> &flag,
                  Eigen::DenseBase<DerivedE> &weights,
                  Eigen::DenseBase<DerivedC> &lat, Eigen::DenseBase<DerivedC> &lon, 
                  Eigen::DenseBase<DerivedD> &det_index_vector, 
                  C &calib_data,
                  T &tel_meta_data,
                  unsigned long dsize, unsigned long dstart=0, unsigned long offset=0) {

    using Eigen::Index;

    using netCDF::NcDim;
    using netCDF::NcFile;
    using netCDF::NcType;
    using netCDF::NcVar;
    using namespace netCDF::exceptions;

    netCDF::NcFile fo(filepath,netCDF::NcFile::write);
    auto vars = fo.getVars();

    try {
        // define the dimensions.
        NcDim d_nsmp = fo.getDim("nsamples");
        NcDim d_ndet = fo.getDim("ndetectors");
        NcDim d_wt = fo.getDim("wt_rows");
        unsigned long nsmp_exists = d_nsmp.getSize() - offset;
        unsigned long nwt_exists = d_wt.getSize();
        unsigned long nwt = weights.size();

        std::vector<std::size_t> i0{nsmp_exists, dstart};
        std::vector<std::size_t> s_d{1, dsize};
        std::vector<std::size_t> i02{nsmp_exists};
        std::vector<std::size_t> i03{dstart};
        std::vector<std::size_t> i04{nwt_exists,0};
        std::vector<std::size_t> s_d2{1};
        std::vector<std::size_t> s_d3{1,nwt};

        NcVar data_v = fo.getVar("DATA");
        NcVar flag_v = fo.getVar("FLAG");
        NcVar lat_v = fo.getVar("DY");
        NcVar lon_v = fo.getVar("DX");

        std::map<std::string, NcVar> tel_meta_vars;

        std::map<std::string, std::string> nc_vars = {
            {"TelTime", "seconds"},
            {"TelElDes", "radians"},
            {"ParAng", "radians"},
            {"TelLatPhys", "radians"},
            {"TelLonPhys", "radians"},
            {"SourceEl", "radians"},
            {"TelAzMap", "radians"},
            {"TelElMap", "radians"}
        };

        for (const auto&var: nc_vars) {
            tel_meta_vars[var.first] = fo.getVar(var.first);
        }

        NcVar p_v = fo.getVar("PIXID");

        NcVar eloff_v = fo.getVar("ELOFF");
        NcVar azoff_v = fo.getVar("AZOFF");
        NcVar afwhm_v = fo.getVar("AFWHM");
        NcVar bfwhm_v = fo.getVar("BFWHM");
        NcVar arrayid_v = fo.getVar("ARRAYID");
        NcVar nwid_v = fo.getVar("NETWORKID");

        NcVar wt_v = fo.getVar("WEIGHTS");
        SPDLOG_INFO("weights {}", weights);

        wt_v.putVar(i04,s_d3,weights.derived().data());

        for (std::size_t ii = 0; ii < TULA_SIZET(data.rows()); ++ii) {
            i0[0] = nsmp_exists + ii;
            i02[0] = nsmp_exists + ii;

            Eigen::VectorXd data_vec = data.row(ii);
            Eigen::Matrix<bool, Eigen::Dynamic,1> flag_vec = flag.row(ii);

            Eigen::Matrix<int,Eigen::Dynamic,1> fvi = flag_vec.cast<int> ();

            Eigen::VectorXd lat_vec = lat.row(ii);
            Eigen::VectorXd lon_vec = lon.row(ii);

            data_v.putVar(i0, s_d, data_vec.data());
            flag_v.putVar(i0, s_d, fvi.data());
            lat_v.putVar(i0, s_d, lat_vec.data());
            lon_v.putVar(i0, s_d, lon_vec.data());

            //e_v.putVar(i02, &elev(ii));
            //a_v.putVar(i02, &az(ii));
            //t_v.putVar(i02, &time(ii));

            for (const auto&var: nc_vars) {
                tel_meta_vars[var.first].putVar(i02,&tel_meta_data[var.first](ii));
            }  
        }

        for (std::size_t ii = 0; ii < TULA_SIZET(data.cols()); ++ii) {
            int di = det_index_vector[ii];

            i03[0] = dstart + ii;

            p_v.putVar(i03, s_d2, &di);

            auto eloff_i = ASEC_TO_RAD*calib_data["x_t"][di];
            auto azoff_i = ASEC_TO_RAD*calib_data["y_t"][di];

            auto a_fwhm_i = ASEC_TO_RAD*calib_data["a_fwhm"][di];
            auto b_fwhm_i = ASEC_TO_RAD*calib_data["b_fwhm"][di];

            arrayid_v.putVar(i03, s_d2, &calib_data["array"][di]);
            nwid_v.putVar(i03, s_d2, &calib_data["nw"][di]);
            eloff_v.putVar(i03, s_d2, &eloff_i);
            azoff_v.putVar(i03, s_d2, &azoff_i);
            afwhm_v.putVar(i03, s_d2, &a_fwhm_i);
            bfwhm_v.putVar(i03, s_d2, &b_fwhm_i);
            bfwhm_v.putVar(i03, s_d2, &b_fwhm_i);
        }

        fo.sync();
        fo.close();

    } catch (NcException &e) {
        SPDLOG_ERROR("{}", e.what());
        throw std::runtime_error(fmt::format("failed to append data to file {}",
                                             tula::nc_utils::pprint(fo)));
    }
}

/*
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
*/
