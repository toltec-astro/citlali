#include <citlali/core/utils/constants.h>
#include <citlali/core/engine/calib.h>

namespace engine {
void Calib::get_apt(const std::string &filepath, std::vector<std::string> &raw_filenames, std::vector<std::string> &interfaces) {
    // store apt filepath
    apt_filepath = filepath;
    // read in the apt table
    auto [apt_temp, header, map_with_strs] = to_map_from_ecsv_mixted_type(filepath);

    // vector to hold any missing header keys
    std::vector<std::string> missing_header_keys, empty_header_keys;

    // look for missing header keys by comparing to required keys
    for (auto &apt_header_key: apt_header_keys) {
        bool found = std::find(header.begin(), header.end(), apt_header_key) != header.end();
        if (!found) {
            missing_header_keys.push_back(apt_header_key);
        }

        else if (apt_temp[apt_header_key].size()==0) {
            empty_header_keys.push_back(apt_header_key);
        }
    }

    // exit if any keys are missing
    if (!missing_header_keys.empty()) {
        SPDLOG_ERROR("apt table is missing required columns {}", missing_header_keys);
        std::exit(EXIT_FAILURE);
    }

    // exit if any keys are empty
    if (!empty_header_keys.empty()) {
        SPDLOG_ERROR("apt table columns are empty {}", empty_header_keys);
        std::exit(EXIT_FAILURE);
    }

    if (map_with_strs["Radesys"]!="altaz") {
        SPDLOG_ERROR("apt table is not in altaz reference frame");
        std::exit(EXIT_FAILURE);
    }

    // set apt table
    apt = apt_temp;

    // run setup on apt table
    setup();

    // vectors to hold roach indices and missing roaches
    std::vector<Eigen::Index> roach_indices, missing;
    Eigen::Index n_dets_temp = 0;

    // get roach indices from raw data files
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

    Eigen::VectorXi interfaces_vec(interfaces.size());

    // get network interfaces
    for (Eigen::Index i=0; i<interfaces.size(); i++) {
        interfaces_vec(i) = std::stoi(interfaces[i].substr(6));
    }

    // count up number of detectors
    for (Eigen::Index i=0; i<interfaces.size(); i++) {
        n_dets_temp = n_dets_temp + (apt["nw"].array() == interfaces_vec(i)).count();
    }

    // populate apt temp
    apt_temp.clear();
    for (auto const& value: apt_header_keys) {
        apt_temp[value].setZero(n_dets_temp);
        Eigen::Index i = 0;
        for (Eigen::Index j=0; j<apt["nw"].size(); j++) {
            if ((apt["nw"](j) == interfaces_vec.array()).any()) {
                apt_temp[value](i) = apt[value](j);
                i++;
            }
        }
    }

    // populate apt table
    apt.clear();
    for (auto const& value: apt_header_keys) {
        apt[value].setZero(n_dets_temp);
        apt[value] = apt_temp[value];
    }

    apt_temp.clear();

    // run setup on new apt table
    setup();
}

void Calib::get_hwp(const std::string &filepath) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    run_hwp = false;

    /*try {
        // get hwp file
        NcFile fo(filepath, NcFile::read, NcFile::classic);
        auto vars = fo.getVars();

        // check if hwp is enabled
        vars.find("Header.Hwp.RotatorInstalled")->second.getVar(&run_hwp);

        // get hwp signal
        Eigen::Index n_pts = vars.find("Data.Hwp.")->second.getDim(0).getSize();
        hwp_angle.resize(n_pts);

        vars.find("Data.Hwp.")->second.getVar(hwp_angle.data());

        // get hwp time
        hwp_ts.resize(n_pts);

        vars.find("Data.Hwp.Ts")->second.getVar(hwp_ts.data());

        fo.close();

    } catch (NcException &e) {
        SPDLOG_ERROR("{}", e.what());
        throw DataIOError{fmt::format(
            "failed to load data from netCDF file {}", filepath)};
    }
    */
}

void Calib::calc_flux_calibration(std::string units) {
    // flux conversion is per detector
    flux_conversion_factor.setOnes(n_dets);

    // default is mJy/beam
    if (units == "mJy/beam") {
        flux_conversion_factor.setOnes();
    }

    // convert to MJy/sr
    else if (units == "MJy/sr") {
        for (Eigen::Index i=0; i<n_dets; i++) {
            auto array = apt["array"](i);
            auto det_fwhm = (std::get<0>(array_fwhms[array]) + std::get<1>(array_fwhms[array]))/2;
            auto beam_area = 2.*pi*pow(det_fwhm*FWHM_TO_STD,2);
            flux_conversion_factor(i) = mJY_ASEC_to_MJY_SR/beam_area;
        }
    }

    else if (units == "uK/arcmin") {
    }

    // get mean flux conversion factor from all unflagged detectors
    for (Eigen::Index i=0; i<n_arrays; i++) {
        auto array = arrays[i];

        Eigen::Index start = std::get<0>(array_limits[array]);
        Eigen::Index end = std::get<1>(array_limits[array]);

        Eigen::Index n_good_dets = 0;

        std::string name = array_name_map[array];

        for (Eigen::Index j=start; j<end; j++) {
            if (apt["flag"](j)) {
                mean_flux_conversion_factor[name] += flux_conversion_factor(j);
                n_good_dets++;
            }
        }
        mean_flux_conversion_factor[name] = mean_flux_conversion_factor[name]/n_good_dets;
    }
}

void Calib::setup() {
    // get number of detectors
    n_dets = apt["uid"].size();

    // get number of networks
    n_nws = ((apt["nw"].tail(n_dets - 1) - apt["nw"].head(n_dets - 1)).array() > 0).count() + 1;

    // get number of arrays
    n_arrays = ((apt["array"].tail(n_dets - 1) - apt["array"].head(n_dets - 1)).array() > 0).count() + 1;

    nws.setZero(n_nws);
    arrays.setZero(n_arrays);

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
            std::get<1>(nw_limits[nw_i]) = i + 1;
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

        auto nw_a_fwhm = apt["a_fwhm"](Eigen::seq(std::get<0>(nw_limits[key]),
                                                std::get<1>(nw_limits[key])-1));

        auto nw_b_fwhm = apt["b_fwhm"](Eigen::seq(std::get<0>(nw_limits[key]),
                                                  std::get<1>(nw_limits[key])-1));

        Eigen::Index n_good_det = apt["flag"](Eigen::seq(std::get<0>(nw_limits[key]),
                                                         std::get<1>(nw_limits[key])-1)).sum();

        // remove flagged dets
        Eigen::Index k = std::get<0>(array_limits[key]);
        for (Eigen::Index i=0; i<nw_a_fwhm.size(); i++) {
            if (apt["flag"](k)) {
                std::get<0>(nw_fwhms[key]) = std::get<0>(nw_fwhms[key]) + nw_a_fwhm(i);
                std::get<1>(nw_fwhms[key]) = std::get<1>(nw_fwhms[key]) + nw_b_fwhm(i);
            }
            k++;
        }

        std::get<0>(nw_fwhms[key]) = std::get<0>(nw_fwhms[key])/n_good_det;
        std::get<1>(nw_fwhms[key]) = std::get<1>(nw_fwhms[key])/n_good_det;

        double avg_nw_fwhm = (std::get<0>(nw_fwhms[key]) + std::get<1>(nw_fwhms[key]))/2;

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

    // get average fwhms for arrays
    j = 0;
    for (auto const& [key, val] : array_limits) {
        arrays(j) = key;
        j++;
        array_fwhms[key] = std::tuple<double,double>{0, 0};

        auto array_a_fwhm = apt["a_fwhm"](Eigen::seq(std::get<0>(array_limits[key]),
                                                  std::get<1>(array_limits[key])-1));

        auto array_b_fwhm = apt["b_fwhm"](Eigen::seq(std::get<0>(array_limits[key]),
                                                  std::get<1>(array_limits[key])-1));

        Eigen::Index n_good_det = apt["flag"](Eigen::seq(std::get<0>(array_limits[key]),
                                                         std::get<1>(array_limits[key])-1)).sum();

        // remove flagged dets
        Eigen::Index k = std::get<0>(array_limits[key]);
        for (Eigen::Index i=0; i<array_a_fwhm.size(); i++) {
            if (apt["flag"](k)) {
                std::get<0>(array_fwhms[key]) = std::get<0>(array_fwhms[key]) + array_a_fwhm(i);
                std::get<1>(array_fwhms[key]) = std::get<1>(array_fwhms[key]) + array_b_fwhm(i);
            }
            k++;
        }

        std::get<0>(array_fwhms[key]) = std::get<0>(array_fwhms[key])/n_good_det;
        std::get<1>(array_fwhms[key]) = std::get<1>(array_fwhms[key])/n_good_det;

        double avg_array_fwhm = (std::get<0>(array_fwhms[key]) + std::get<1>(array_fwhms[key]))/2;

        array_beam_areas[key] = 2.*pi*pow(avg_array_fwhm/STD_TO_FWHM,2);
    }
}

} // namespace engine_utils
