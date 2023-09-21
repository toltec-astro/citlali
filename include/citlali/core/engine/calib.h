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
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // apt filepath
    std::string apt_filepath;
    // apt table
    std::map<std::string, Eigen::VectorXd> apt;
    Eigen::VectorXd hwpr_angle, hwp_recvt;
    Eigen::MatrixXd hwp_ts;

    // detector frequency groups
    Eigen::VectorXI fg;

    // ignore the hwpr (set to true if hwpr is not installed)
    std::string ignore_hwpr;

    // fpga frequency for hwpr
    double hwpr_fpga_freq;

    // apt header
    YAML::Node apt_meta;

    // run with hwpr?
    bool run_hwpr;

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

    // mean flux conversion factor
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
        {"tone_freq"},
        {"array"},
        {"nw"},
        {"fg"},
        {"pg"},
        {"ori"},
        {"loc"},
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
        {"x_t_raw"},
        {"y_t_raw"},
        {"x_t_derot"},
        {"y_t_derot"},
    };

    std::map<std::string,std::string> apt_header_units = {
        {"uid","N/A"},
        {"tone_freq", "Hz"},
        {"array","N/A"},
        {"nw","N/A"},
        {"fg","N/A"},
        {"pg","N/A"},
        {"ori","N/A"},
        {"loc","N/A"},
        {"responsivity","N/A"},
        {"flxscale","mJy/beam/xs"},
        {"sens","mJy/beam x s^0.5"},
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

    std::map<std::string,std::string> apt_header_description = {
        {"uid","unique id"},
        {"tone_freq", "tone frequency"},
        {"array","array index"},
        {"nw","network index"},
        {"fg","frequency group"},
        {"pg","polarization group"},
        {"ori","orientation"},
        {"loc","location"},
        {"responsivity","responsivity"},
        {"flxscale","flux conversion scale"},
        {"sens","sensitivity"},
        {"derot_elev","derotation elevation angle"},
        {"amp","fitted amplitude"},
        {"amp_err","fitted amplitude error"},
        {"x_t","fitted azimuthal offset"},
        {"x_t_err","fitted azimuthal offset error"},
        {"y_t","fitted altitude offset"},
        {"y_t_err","fitted altitude offset error"},
        {"a_fwhm","fitted azimuthal FWHM"},
        {"a_fwhm_err","fitted azimuthal FWHM error"},
        {"b_fwhm","fitted altitude FWHM"},
        {"b_fwhm_err","fitted altitude FWHM error"},
        {"angle","fitted rotation angle"},
        {"angle_err","fitted rotation angle error"},
        {"converge_iter","beammap convergence iteration"},
        {"flag","bad detector"},
        {"sig2noise","signal to noise"},
        {"x_t_raw","raw azimuthal offset"},
        {"y_t_raw","raw altitude offset"},
        {"x_t_derot","derot azimuthal offset"},
        {"y_t_derot","derot altitude offset"},
        };

    void setup();
    void get_apt(const std::string &, std::vector<std::string> &, std::vector<std::string> &);
    void get_hwpr(const std::string &, bool);
    void calc_flux_calibration(std::string);
};

} // namespace engine
