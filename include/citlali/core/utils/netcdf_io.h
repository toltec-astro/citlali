#pragma once

#include <fmt/ostream.h>
#include <netcdf>
#include <regex>

#include <Eigen/Core>

#include <tula/logging.h>

struct DataIOError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
void append_to_nc(std::string filepath, Eigen::DenseBase<DerivedA> &data, Eigen::DenseBase<DerivedC> &flag,
                  Eigen::DenseBase<DerivedA> &lat, Eigen::DenseBase<DerivedA> &lon, Eigen::DenseBase<DerivedB> &elev,
                  Eigen::DenseBase<DerivedD> &time, size_t start_row) {

    using Eigen::Index;

    using netCDF::NcDim;
    using netCDF::NcFile;
    using netCDF::NcType;
    using netCDF::NcVar;
    using namespace netCDF::exceptions;

    //auto &fo = io.file_obj();

    netCDF::NcFile fo(filepath,netCDF::NcFile::write);
    auto vars = fo.getVars();

    try {
        // define the dimensions.
        // NcDim d_ntimes = fo.addDim(_["ntimes"], SIZET(ntimes));
        NcDim d_nsmp = fo.getDim("nsamples"); // unlimited
        NcDim d_ndet = fo.getDim("ndetectors");
        unsigned long nsmp_exists = d_nsmp.getSize();
        //NcDim d_ntimecols = fo.getDim(_["ntimecols"]);
        /*if (!d_nsmp.isNull()){// || !d_ndet.isNull()) {
            //save_to_nc(fo, true);
            d_nsmp = fo.getDim("nsamples");
            d_ndet = fo.getDim("ndetectors");
            nsmp_exists = d_nsmp.getSize();
        }

        else {
            d_nsmp = fo.addDim("nsamples");
            d_ndet = fo.addDim("ndetectors",data.cols());
        }*/

        std::vector<netCDF::NcDim> dims;
        dims.push_back(d_nsmp);
        dims.push_back(d_ndet);

        std::vector<std::size_t> i0{nsmp_exists, 0};
        std::vector<std::size_t> s_d{1, d_ndet.getSize()};
        std::vector<std::size_t> i02{nsmp_exists};
        std::vector<std::size_t> i03{0};
        std::vector<std::size_t> s_d2{1};

           // std::vector<std::size_t> s{SIZET(ntimes), SIZET(ntones)};

        //if (!d_nsmp.isNull()) {
/*
        NcVar data_v = fo.addVar("data", netCDF::ncDouble, dims);
        NcVar lat_v = fo.addVar("lat", netCDF::ncDouble, dims);
        NcVar lon_v = fo.addVar("lon", netCDF::ncDouble, dims);
*/
            NcVar data_v = fo.getVar("DATA");//, netCDF::ncDouble, dims);
            NcVar flag_v = fo.getVar("FLAG");//, netCDF::ncDouble, dims);
            NcVar lat_v = fo.getVar("DY");//, netCDF::ncDouble, dims);
            NcVar lon_v = fo.getVar("DX");//, netCDF::ncDouble, dims);

            NcVar e_v = fo.getVar("ELEV");//, netCDF::ncDouble, dims);
            NcVar t_v = fo.getVar("TIME");//, netCDF::ncDouble, dims);

            NcVar p_v = fo.getVar("PIXID");//, netCDF::ncDouble, dims);

            auto put_data = [&](auto ii) mutable {
                i0[0] = nsmp_exists + ii;
                i02[0] = nsmp_exists + ii;

                data_v.putVar(i0, s_d, data.row(ii).data());
                flag_v.putVar(i0, s_d, flag.row(ii).data());
                lat_v.putVar(i0, s_d, lat.row(ii).data());
                lon_v.putVar(i0, s_d, lon.row(ii).data());

                e_v.putVar(i02, s_d2, &elev(ii));
                t_v.putVar(i02, s_d2, &time(ii));
            };

            for (std::size_t ii = 0; ii < TULA_SIZET(data.rows()); ++ii) {
                put_data(ii);
                //pb0.count(ntimes, ntimes / 10);
            }

            for (std::size_t ii = 0; ii < TULA_SIZET(data.cols()); ++ii) {
                int i = ii;
                std::vector<std::size_t> index{ii};
                p_v.putVar(index, s_d2, &i);
                //pb0.count(ntimes, ntimes / 10);
            }

            //fo.sync();
            fo.close();
        //}

        /*else {

           NcVar data_v = fo.addVar("data", netCDF::ncDouble, dims);
           NcVar lat_v = fo.addVar("lat", netCDF::ncDouble, dims);
           NcVar lon_v = fo.addVar("lon", netCDF::ncDouble, dims);

           auto put_data = [&](auto ii) mutable {
               i0[0] = nsmp_exists + ii;
               data_v.putVar(i0, s_d, data.row(ii).data());
               lat_v.putVar(i0, s_d, lat.row(ii).data());
               lon_v.putVar(i0, s_d, lon.row(ii).data());
           };

           for (std::size_t ii = 0; ii < TULA_SIZET(data.rows()); ++ii) {
               put_data(ii);
               //pb0.count(ntimes, ntimes / 10);
           }

        }*/

        //std::vector<std::size_t> s_t{1, d_ntimecols.getSize()};

        //const auto &tdata = data.wcs.time_axis.data;
        //assert(d_ntimecols.getSize() == TULA_SIZET(tdata.cols()));
        //assert(d_ntones.getSize() == TULA_SIZET(data_out.xs.data.cols()));
        //auto ntimes = data_out.xs.data.rows();

    } catch (NcException &e) {
        SPDLOG_ERROR("{}", e.what());
        //throw std::runtime_error(fmt::format("failed to append data to file {}",
                                             //tula::nc_utils::pprint(fo)));
    }
    //return io;
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
