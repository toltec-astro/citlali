#pragma once

#include <Eigen/Core>
#include <map>
#include <netcdf>
#include <fmt/ostream.h>

#include <tula/logging.h>

#include <citlali/core/utils/netcdf_io.h>
#include <citlali/core/utils/constants.h>

struct ToltecTelescope {
    std::map<std::string, std::string> backend_keys {
        {"Data.TelescopeBackend.TelTime", "TelTime"},
        {"Data.TelescopeBackend.SourceRaAct", "TelRa"},
        {"Data.TelescopeBackend.SourceDecAct", "TelDec"},
        {"Data.TelescopeBackend.TelAzAct", "TelAzAct"},
        {"Data.TelescopeBackend.TelElAct", "TelElAct"},
        {"Data.TelescopeBackend.SourceAz", "SourceAz"},
        {"Data.TelescopeBackend.SourceEl", "SourceEl"},
        {"Data.TelescopeBackend.ActParAng", "ActParAng"},
        {"Data.TelescopeBackend.ParAng", "ParAng"},
        {"Data.TelescopeBackend.Hold", "Hold"},
        {"Data.TelescopeBackend.TelAzCor", "TelAzCor"},
        {"Data.TelescopeBackend.TelElCor", "TelElCor"},
        {"Data.TelescopeBackend.TelAzDes", "TelAzDes"},
        {"Data.TelescopeBackend.TelElDes", "TelElDes"},
        {"Data.TelescopeBackend.PpsTime","PpsTime"},
        {"Data.TelescopeBackend.TelAzMap", "TelAzMap"},
        {"Data.TelescopeBackend.TelElMap", "TelElMap"}
    };

    std::map<std::string, std::string> src_keys {
        {"Header.Source.Ra", "Ra"},
        {"Header.Source.Dec", "Dec"}
    };

    std::map<std::string, std::string> header_keys {
        {"Header.M2.XAct","Header.M2.XAct"},
        {"Header.M2.YAct","Header.M2.YAct"},
        {"Header.M2.ZAct","Header.M2.ZAct"},
        {"Header.M2.TipAct","Header.M2.TipAct"},
        {"Header.M2.TiltAct","Header.M2.TiltAct"},
        {"Header.M2.XReq","Header.M2.XReq"},
        {"Header.M2.YReq","Header.M2.YReq"},
        {"Header.M2.ZReq","Header.M2.ZReq"},
        {"Header.M2.TipReq","Header.M2.TipReq"},
        {"Header.M2.TiltReq","Header.M2.TiltReq"},
        {"Header.M2.XDes","Header.M2.XDes"},
        {"Header.M2.YDes","Header.M2.YDes"},
        {"Header.M2.ZDes","Header.M2.ZDes"},
        {"Header.M2.TipDes","Header.M2.TipDes"},
        {"Header.M2.TiltDes","Header.M2.TiltDes"},
        {"Header.M2.XPcor","Header.M2.XPcor"},
        {"Header.M2.YPcor","Header.M2.YPcor"},
        {"Header.M2.ZPcor","Header.M2.ZPcor"},
        {"Header.M2.TipPcor","Header.M2.TipPcor"},
        {"Header.M2.TiltPcor","Header.M2.TiltPcor"},
        {"Header.M2.XCmd","Header.M2.XCmd"},
        {"Header.M2.YCmd","Header.M2.YCmd"},
        {"Header.M2.ZCmd","Header.M2.ZCmd"},
        {"Header.M2.TipCmd","Header.M2.TipCmd"},
        {"Header.M2.TiltCmd","Header.M2.TiltCmd"},
        {"Header.M1.ZernikeC","Header.M1.ZernikeC"}
    };
};

class Telescope: ToltecTelescope {
public:
    // map pattern variable (Map or lissajous)
    std::string map_pattern_type, source_name;
    char map_pattern_type_c [128];
    // source name
    char source_name_c [128];
    // telescope pointing vectors
    std::map<std::string, Eigen::VectorXd> tel_meta_data;
    std::map<std::string, Eigen::VectorXd> tel_header_data;
    // source center from telescope file
    std::map<std::string, Eigen::VectorXd> source_center;

    // coadded t_exp
    double t_exp = 0;
    double c_t_exp = 0;

    // map absolute center value
    double crval1_J2000, crval2_J2000;

    void get_telescope(const std::string &filepath) {
        using namespace netCDF;
        using namespace netCDF::exceptions;

        try {
            // get telescope file
            NcFile fo(filepath, NcFile::read, NcFile::classic);
            SPDLOG_INFO("read in telescope netCDF file {}", filepath);
            auto vars = fo.getVars();

            // get mapping pattern
            vars.find("Header.Dcs.ObsPgm")->second.getVar(&map_pattern_type_c);

            map_pattern_type = std::string(map_pattern_type_c);
            map_pattern_type = map_pattern_type.substr(0,50);

            SPDLOG_INFO("map_pattern_type {}", map_pattern_type);

            // get source name
            vars.find("Header.Source.SourceName")->second.getVar(&source_name_c);

            source_name = std::string(source_name_c);
            source_name = source_name.substr(0,50);
            SPDLOG_INFO("source_name {}", source_name);

            // loop through and get telescope backend vectors
            for (auto const& pair : backend_keys) {
                SPDLOG_INFO("PAIR.FIRST {}",pair.first);
                Eigen::Index npts = vars.find(pair.first)->second.getDim(0).getSize();
                tel_meta_data[pair.second].resize(npts);
                vars.find(pair.first)->second.getVar(tel_meta_data[pair.second].data());
            }

            // adjust parallactic angle parameters
            tel_meta_data["ParAng"] = pi - tel_meta_data["ParAng"].array();
            tel_meta_data["ActParAng"] = pi - tel_meta_data["ActParAng"].array();

            // loop through and get telescope header vectors
            for (auto const& pair : src_keys) {
                Eigen::Index npts = vars.find(pair.first)->second.getDim(0).getSize();
                    SPDLOG_INFO("src.FIRST {}",pair.first);
                source_center[pair.second].resize(npts);
                vars.find(pair.first)->second.getVar(source_center[pair.second].data());
            }


            // loop through and get telescope header vectors
            for (auto const& pair : header_keys) {
                Eigen::Index npts = 1;
                /*try {
                    Eigen::Index npts = vars.find(pair.first)->second.getDim(0).getSize();
                } catch (...) {
                    Eigen::Index npts = 1;
                }*/

                if (pair.first == "Header.M1.ZernikeC") {
                    npts = vars.find(pair.first)->second.getDim(0).getSize();
                }

                SPDLOG_INFO("{}",pair.second);

                tel_header_data[pair.second].resize(npts);
                vars.find(pair.first)->second.getVar(tel_header_data[pair.second].data());
            }

            // override center Ra if crval1_J2000 is non-zero
            if (crval1_J2000 !=0) {
                Eigen::VectorXd center_ra(source_center["Ra"].size());
                source_center["Ra"] = center_ra.setConstant(crval1_J2000*DEG_TO_RAD);
            }

            // override center Dec if crval2_J2000 is non-zero
            if (crval2_J2000 !=0) {
                Eigen::VectorXd center_dec(source_center["Dec"].size());
                source_center["Dec"] = center_dec.setConstant(crval2_J2000*DEG_TO_RAD);
            }

            Eigen::Index npts = tel_meta_data["TelTime"].size();

            t_exp = tel_meta_data["TelTime"](npts - 1) - tel_meta_data["TelTime"](0);
            fo.close();

        } catch (NcException &e) {
            SPDLOG_ERROR("{}", e.what());
            throw DataIOError{fmt::format(
                "failed to load data from netCDF file {}", filepath)};
        }
    }

    //template  <typename hdu_t>
    void copy_header_to_fits(const std::string &filepath) {//, hdu_t *hdu) {
        using namespace netCDF;
        using namespace netCDF::exceptions;

        try {
            // get telescope file
            NcFile fo(filepath, NcFile::read, NcFile::classic);
            auto vars = fo.getVars();

            for (const auto& var: vars) {
                std::size_t found = var.first.find("Header");
                if (found!=std::string::npos) {
                    SPDLOG_INFO("Header Key {}", var.first);
                }
            }

            fo.close();

        } catch (NcException &e) {
            SPDLOG_ERROR("{}", e.what());
            throw DataIOError{fmt::format(
                "failed to load data from netCDF file {}", filepath)};
        }

    }
};
