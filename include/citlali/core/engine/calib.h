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

    // apt table
    std::map<std::string, Eigen::VectorXd> apt;
    Eigen::VectorXd hwp_angle;

    // apt header
    YAML::Node apt_header;

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

    // keys for apt header
    std::vector<std::string> apt_header_keys = {
        {"uid"},
        {"nw"},
        {"fg"},
        {"pg"},
        {"ori"},
        {"array"},
        {"flxscale"},
        {"x_t"},
        {"y_t"},
        {"a_fwhm"},
        {"b_fwhm"},
        {"angle"},
        {"responsivity"},
        {"flag"},
        {"sens"},
        {"sig2noise"}
    };

    void setup();
    void get_apt(const std::string &, std::vector<std::string> &);
    void get_hwp(const std::string &);
    void calc_flux_calibration(std::string);

};

void Calib::get_apt(const std::string &filepath, std::vector<std::string> &raw_filenames) {
    // read in the apt table
    auto [table, header] = to_matrix_from_ecsv(filepath);

    // apt header
    apt_header = header;

    // set the number of detectors
    Eigen::Index ndet = table.rows();

    // loop through the apt table header keys and populate calib_data
    for (auto const& value: apt_header_keys) {
        auto it = find(header.begin(), header.end(), value);
        if (it != header.end()) {
            int index = it - header.begin();
            apt[value] = table.col(index);
        }
    }

    setup();

    std::vector<Eigen::Index> roach_indices, missing;
    Eigen::Index n_dets_temp = apt["nw"].size();

    for (Eigen::Index i=0; i<raw_filenames.size(); i++) {
        netCDF::NcFile fo(raw_filenames[i], netCDF::NcFile::read);
        auto vars = fo.getVars();
        // get roach index
        int roach_index;
        vars.find("Header.Toltec.RoachIndex")->second.getVar(&roach_index);
        roach_indices.push_back(roach_index);
        fo.close();
    }

    auto roach_vec = Eigen::Map<Eigen::VectorXI>(roach_indices.data(), roach_indices.size());

    for (Eigen::Index i=0; i<nws.size(); i++) {
        if (!(roach_vec.array() == nws(i)).any()) {
            missing.push_back(nws(i));
            n_dets_temp = n_dets_temp - (apt["nw"].array() == nws(i)).count();
        }
    }

    auto missing_vec = Eigen::Map<Eigen::VectorXI>(missing.data(), missing.size());
    std::map<std::string, Eigen::VectorXd> apt_temp;

    for (auto const& value: apt_header_keys) {
        apt_temp[value].setZero(n_dets_temp);
        Eigen::Index i = 0;
        for (Eigen::Index j=0; j<apt["nw"].size(); j++) {
            if ((apt["nw"](j) != missing_vec.array()).all()) {
                apt_temp[value](i) = apt[value](j);
                i++;
            }
        }
    }

    apt.clear();
    for (auto const& value: apt_header_keys) {
        apt[value].setZero(n_dets_temp);
        apt[value] = apt_temp[value];
    }

    setup();
}

void Calib::get_hwp(const std::string &filepath) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    try {
        // get hwp file
        NcFile fo(filepath, NcFile::read, NcFile::classic);
        auto vars = fo.getVars();

        // check if hwp is enabled
        vars.find("Header.Hwp.RotatorEnabled")->second.getVar(&run_hwp);

        // get hwp signal
        Eigen::Index npts = vars.find("Data.Hwp.")->second.getDim(0).getSize();
        hwp_angle.resize(npts);

        vars.find("Data.Hwp.")->second.getVar(hwp_angle.data());

        fo.close();

    } catch (NcException &e) {
        SPDLOG_ERROR("{}", e.what());
        throw DataIOError{fmt::format(
            "failed to load data from netCDF file {}", filepath)};
    }
}

void Calib::calc_flux_calibration(std::string units) {
    // flux conversion is per detector
    flux_conversion_factor.resize(n_dets);

    // default is MJy/sr
    if (units == "MJy/sr") {
        flux_conversion_factor.setOnes();
    }

    // convert to mJy/beam
    else if (units == "mJy/beam") {
        for (Eigen::Index i=0; i<n_dets; i++) {
            auto det_fwhm = RAD_TO_ASEC*(apt["a_fwhm"](i) + apt["b_fwhm"](i))/2;
            auto beam_area = 2.*pi*pow(det_fwhm/STD_TO_FWHM,2);
            flux_conversion_factor(i) = beam_area*MJY_SR_TO_mJY_ASEC;
        }
    }

    else if (units == "uK/arcmin") {
        for (Eigen::Index i=0; i<n_dets; i++) {
            //auto det_fwhm = (calib.apt["a_fwhm"](i) + calib.apt["b_fwhm"](i))/2;
            //auto beam_area = 2.*pi*pow(det_fwhm/STD_TO_FWHM,2);
            //flux_conversion_factor(i) = beam_area*MJY_SR_TO_mJY_ASEC;
        }
    }
}

void Calib::setup() {
    // get number of detectors
    n_dets = apt["uid"].size();

    // get number of networks
    n_nws = ((apt["nw"].tail(n_dets - 1) - apt["nw"].head(n_dets - 1)).array() == 1).count() + 1;
    // get number of arrays
    n_arrays = ((apt["array"].tail(n_dets - 1) - apt["array"].head(n_dets - 1)).array() == 1).count() + 1;

    arrays.setZero(n_arrays);
    nws.setZero(n_nws);

    // set up network values
    nw_limits.clear();
    nw_fwhms.clear();
    nw_beam_areas.clear();

    Eigen::Index j = 0;
    Eigen::Index nw_i = apt["nw"](0);
    nw_limits[nw_i] = std::tuple<Eigen::Index, Eigen::Index>{0, 0};

    // loop through apt table networks, get highest index for current networks
    for (Eigen::Index i=0; i<apt["nw"].size(); i++) {
        if (apt["nw"](i) == nw_i) {
            std::get<1>(nw_limits[nw_i]) = i+1;
        }
        else {
            nw_i = apt["nw"](i);
            j += 1;
            nw_limits[nw_i] = std::tuple<Eigen::Index, Eigen::Index>{i, 0};
        }
    }

    // get average fwhms for networks
    j = 0;
    for (auto const& [key, val] : nw_limits) {
        nws(j) = key;
        j++;
        nw_fwhms[key] = std::tuple<double,double>{0, 0};
        std::get<0>(nw_fwhms[key]) = apt["a_fwhm"](Eigen::seq(std::get<0>(nw_limits[key]),
                                                              std::get<1>(nw_limits[key])-1)).mean();
        std::get<1>(nw_fwhms[key]) = apt["b_fwhm"](Eigen::seq(std::get<0>(nw_limits[key]),
                                                              std::get<1>(nw_limits[key])-1)).mean();

        auto avg_nw_fwhm = ((apt["a_fwhm"](Eigen::seq(std::get<0>(nw_limits[key]), std::get<1>(nw_limits[key])-1)) +
                          apt["b_fwhm"](Eigen::seq(std::get<0>(nw_limits[key]), std::get<1>(nw_limits[key])-1)))/2).mean();

        nw_beam_areas[key] = 2.*pi*pow(avg_nw_fwhm/STD_TO_FWHM,2);
    }

    // set up array values
    array_limits.clear();
    array_fwhms.clear();
    array_beam_areas.clear();

    Eigen::Index arr_i = apt["array"](0);
    array_limits[arr_i] = std::tuple<Eigen::Index, Eigen::Index>{0, 0};

    j = 0;

    // loop through apt table arrays, get highest index for current array
    for (Eigen::Index i=0; i<apt["array"].size(); i++) {
        if (apt["array"](i) == arr_i) {
            std::get<1>(array_limits[arr_i]) = i+1;
        }
        else {
            arr_i = apt["array"](i);
            j += 1;
            array_limits[arr_i] = std::tuple<Eigen::Index, Eigen::Index>{i, 0};
        }
    }

    // get average fwhms for networks
    j = 0;
    for (auto const& [key, val] : array_limits) {
        arrays(j) = key;
        j++;
        array_fwhms[key] = std::tuple<double,double>{0, 0};
        std::get<0>(array_fwhms[key]) = apt["a_fwhm"](Eigen::seq(std::get<0>(array_limits[key]),
                                                                 std::get<1>(array_limits[key])-1)).mean();
        std::get<1>(array_fwhms[key]) = apt["b_fwhm"](Eigen::seq(std::get<0>(array_limits[key]),
                                                                 std::get<1>(array_limits[key])-1)).mean();

        auto avg_array_fwhm = ((apt["a_fwhm"](Eigen::seq(std::get<0>(array_limits[key]), std::get<1>(array_limits[key])-1)) +
                             apt["b_fwhm"](Eigen::seq(std::get<0>(array_limits[key]), std::get<1>(array_limits[key])-1)))/2).mean();

        array_beam_areas[key] = 2.*pi*pow(avg_array_fwhm/STD_TO_FWHM,2);
    }

    SPDLOG_INFO("array_fwhms {}", array_fwhms);
    SPDLOG_INFO("nw_fwhms {}", nw_fwhms);
    SPDLOG_INFO("nws {} arrays {}", nws, arrays);
    SPDLOG_INFO("nw_beam_areas {} array_beam_areas {}", nw_beam_areas, array_beam_areas);
}

} // namespace engine
