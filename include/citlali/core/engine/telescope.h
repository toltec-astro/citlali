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
        {"Data.TelescopeBackend.TelRaAct", "TelRa"},
        {"Data.TelescopeBackend.TelDecAct", "TelDec"},
        {"Data.TelescopeBackend.TelAzAct", "TelAzAct"},
        {"Data.TelescopeBackend.TelElAct", "TelElAct"},
        {"Data.TelescopeBackend.SourceAz", "SourceAz"},
        {"Data.TelescopeBackend.SourceEl", "SourceEl"},
        {"Data.TelescopeBackend.ActParAng", "ActParAng"},
        {"Data.TelescopeBackend.ParAng", "ParAng"},
        {"Data.TelescopeBackend.Hold", "Hold"}
    };

    std::map<std::string, std::string> header_keys {
        {"Header.Source.Ra", "Ra"},
        {"Header.Source.Dec", "Dec"}
    };
};

class Telescope: ToltecTelescope {
public:
    // map pattern variable (Map or lissajous)
    char map_pattern_type [128];
    // telescope pointing vectors
    std::map<std::string, Eigen::VectorXd> tel_meta_data;
    // source center from telescope file
    std::map<std::string, Eigen::VectorXd> source_center;

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
                vars.find("Header.Dcs.ObsPgm")->second.getVar(&map_pattern_type);

                // loop through and get telescope backend vectors
                for (auto const& pair : backend_keys) {
                    Eigen::Index npts = vars.find(pair.first)->second.getDim(0).getSize();
                    tel_meta_data[pair.second].resize(npts);
                    vars.find(pair.first)->second.getVar(tel_meta_data[pair.second].data());
                }

                // adjust parallactic angle parameters
                tel_meta_data["ParAng"] = pi - tel_meta_data["ParAng"].array();
                tel_meta_data["ActParAng"] = pi - tel_meta_data["ActParAng"].array();

                // loop through and get telescope header vectors
                for (auto const& pair : header_keys) {
                    Eigen::Index npts = vars.find(pair.first)->second.getDim(0).getSize();
                    source_center[pair.second].resize(npts);
                    vars.find(pair.first)->second.getVar(source_center[pair.second].data());
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

                /* TEMP */
                tel_meta_data["TelAzCor"].setZero(tel_meta_data["TelAzAct"].size());
                tel_meta_data["TelElCor"].setZero(tel_meta_data["TelElAct"].size());

                tel_meta_data["TelAzDes"] = tel_meta_data["TelAzAct"];
                tel_meta_data["TelElDes"] = tel_meta_data["TelElAct"];

                fo.close();

            } catch (NcException &e) {
                SPDLOG_ERROR("{}", e.what());
                throw DataIOError{fmt::format(
                    "failed to load data from netCDF file {}", filepath)};
            }
    }
};
