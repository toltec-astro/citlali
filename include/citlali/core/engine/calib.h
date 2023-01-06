#pragma once

#include <Eigen/Core>
#include <string>
#include <vector>
#include <map>
#include <netcdf>

#include <citlali/core/utils/ecsv_io.h>
#include <citlali/core/utils/netcdf_io.h>

namespace engine {

class Calib {
public:

    // apt filepath
    std::string apt_filepath;
    // apt table
    std::map<std::string, Eigen::VectorXd> apt;
    Eigen::VectorXd hwp_angle;

    // apt header
    YAML::Node apt_meta;

    // run with hwp
    bool run_hwp;

    // network and array names
    Eigen::VectorXI nws, arrays;

    // number of detectors
    Eigen::Index n_dets;

    // number of networks
    Eigen::Index n_nws;

    // number of arrays
    Eigen::Index n_arrays;

    // vector of upper and lower indices of each nw/array
    std::map<Eigen::Index, std::tuple<Eigen::Index, Eigen::Index>> nw_limits, array_limits;

    // average fwhms
    std::map<Eigen::Index, std::tuple<double, double>> nw_fwhms, array_fwhms;

    // average beam areas
    std::map<Eigen::Index, double> nw_beam_areas, array_beam_areas;

    // flux conversion
    Eigen::VectorXd flux_conversion_factor;
    std::map<std::string, double> mean_flux_conversion_factor;

    // map from array index to name
    std::map<Eigen::Index, std::string> array_name_map = {
        {0,"a1100"},
        {1,"a1400"},
        {2,"a2000"}
    };

    // keys for apt header
    std::vector<std::string> apt_header_keys = {
        {"uid"},
        {"array"},
        {"nw"},
        {"fg"},
        {"pg"},
        {"ori"},
        {"responsivity"},
        {"flxscale"},
        {"sens"},
        {"derot_elev"},
        {"amp"},
        {"amp_err"},
        {"x_t"},
        {"x_t_err"},
        {"y_t"},
        {"y_t_err"},
        {"a_fwhm"},
        {"a_fwhm_err"},
        {"b_fwhm"},
        {"b_fwhm_err"},
        {"angle"},
        {"angle_err"},
        {"converge_iter"},
        {"flag"},
        {"sig2noise"},
        //{"x_t_raw"},
        //{"y_t_raw"},
        //{"x_t_derot"},
        //{"y_t_derot"},
    };

    std::map<std::string,std::string> apt_header_units = {
        {"uid","N/A"},
        {"array","N/A"},
        {"nw","N/A"},
        {"fg","N/A"},
        {"pg","N/A"},
        {"ori","N/A"},
        {"responsivity","N/A"},
        {"flxscale","mJy/beam/xs"},
        {"sens","N/A"},
        {"derot_elev","rad"},
        {"amp","xs"},
        {"amp_err","xs"},
        {"x_t","arcsec"},
        {"x_t_err","arcsec"},
        {"y_t","arcsec"},
        {"y_t_err","arcsec"},
        {"a_fwhm","arcsec"},
        {"a_fwhm_err","arcsec"},
        {"b_fwhm","arcsec"},
        {"b_fwhm_err","arcsec"},
        {"angle","rad"},
        {"angle_err","rad"},
        {"converge_iter","N/A"},
        {"flag","N/A"},
        {"sig2noise","N/A"},
        {"x_t_raw","arcsec"},
        {"y_t_raw","arcsec"},
        {"x_t_derot","arcsec"},
        {"y_t_derot","arcsec"},
        };

    void setup();
    void get_apt(const std::string &, std::vector<std::string> &, std::vector<std::string> &);
    void get_hwp(const std::string &);
    void calc_flux_calibration(std::string);
};

} // namespace engine
