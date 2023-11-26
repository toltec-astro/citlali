#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/utils.h>
#include <citlali/core/engine/calib.h>
#include <citlali/core/utils/toltec_io.h>

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
        // look for empty headers
        else if (apt_temp[apt_header_key].size()==0) {
            empty_header_keys.push_back(apt_header_key);
        }
    }

    // exit if any keys are missing
    if (!missing_header_keys.empty()) {
        logger->error("apt table is missing required columns {}", missing_header_keys);
        std::exit(EXIT_FAILURE);
    }

    // exit if any keys are empty
    if (!empty_header_keys.empty()) {
        logger->error("apt table columns are empty {}", empty_header_keys);
        std::exit(EXIT_FAILURE);
    }

    // exit if apt reference frame is not altaz (required for pointing calculations)
    if (map_with_strs["Radesys"]!="altaz") {
        logger->error("apt table is not in altaz reference frame");
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

    // vector to hold interface number
    Eigen::VectorXi interfaces_vec(interfaces.size());

    // get network interfaces
    for (Eigen::Index i=0; i<interfaces.size(); i++) {
        interfaces_vec(i) = std::stoi(interfaces[i].substr(6));
    }

    // count up number of detectors
    for (Eigen::Index i=0; i<interfaces.size(); i++) {
        n_dets_temp = n_dets_temp + (apt["nw"].array() == interfaces_vec(i)).count();
    }

    // clear apt
    apt_temp.clear();
    // populate apt temp
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

    // clear apt
    apt.clear();
    // populate apt table
    for (auto const& value: apt_header_keys) {
        apt[value].setZero(n_dets_temp);
        apt[value] = apt_temp[value];
    }

    // clear temporary apt
    apt_temp.clear();

    // run setup on new apt table
    setup();
}

void Calib::get_hwpr(const std::string &filepath, bool sim_obs) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    try {
        // get hwp file
        NcFile fo(filepath, NcFile::read, NcFile::classic);
        auto vars = fo.getVars();

        // variable for whether or not hwpr is installed
        std::string hwpr_install_v;

        // get hwp install vector for sim or real obs
        if (!sim_obs) {
            hwpr_install_v = "Header.Toltec.HwpInstalled";
        }
        else {
            hwpr_install_v = "Header.Hwp.Installed";
        }

        // check if hwpr is enabled
        vars.find(hwpr_install_v)->second.getVar(&run_hwpr);

        // if not enabled or running
        if (run_hwpr) {
            // get hwpr signal
            Eigen::Index n_pts = vars.find("Data.Hwp.")->second.getDim(0).getSize();
            hwpr_angle.resize(n_pts);
            // hwpr signal
            vars.find("Data.Hwp.")->second.getVar(hwpr_angle.data());

            // if real data
            if (!sim_obs) {
                // get hwpr time for interpolation
                hwpr_ts.resize(n_pts,6);

                // timing for hwpr (temporary)
                vars.find("Data.Hwp.Ts")->second.getVar(hwpr_ts.data());
                hwpr_ts.transposeInPlace();

                // UT time for hwpr
                Eigen::Index recvt_n_pts = vars.find("Data.Hwp.Uts")->second.getDim(0).getSize();
                hwpr_recvt.resize(recvt_n_pts);
                vars.find("Data.Hwp.Uts")->second.getVar(hwpr_recvt.data());

                // fpga frequency
                vars.find("Header.Toltec.FpgaFreq")->second.getVar(&hwpr_fpga_freq);
            }
        }

        fo.close();

    } catch (NcException &e) {
        logger->error("{}", e.what());
        throw DataIOError{fmt::format(
            "failed to load data from netCDF file {}", filepath)};
    }
}

void Calib::calc_flux_calibration(std::string units, double pixel_size_rad) {
    // flux conversion is per detector
    flux_conversion_factor.setOnes(n_dets);

    // default is mJy/beam (apt should always be in mJy/beam)
    if (units == "mJy/beam") {
        flux_conversion_factor.setOnes();
    }

    // convert to MJy/sr
    else if (units == "MJy/sr") {
        for (Eigen::Index i=0; i<n_dets; i++) {
            // current detector's array
            auto array = apt["array"](i);
            // det fwhm
            auto det_fwhm = (std::get<0>(array_fwhms[array]) + std::get<1>(array_fwhms[array]))/2;
            // beam area
            auto beam_area = 2.*pi*pow(det_fwhm*FWHM_TO_STD,2);
            // get MJy/Sr
            flux_conversion_factor(i) = mJY_ASEC_to_MJY_SR/beam_area;
        }
    }

    // convert to uK/beam
    else if (units == "uK") {
        engine_utils::toltecIO toltec_io;
        for (Eigen::Index i=0; i<n_dets; i++) {
            // current detector's array
            auto array = apt["array"](i);
            // array frequency
            auto freq_Hz = toltec_io.array_freq_map[array];
            // det fwhm
            auto det_fwhm = (std::get<0>(array_fwhms[array]) + std::get<1>(array_fwhms[array]))/2;
            // get uK
            flux_conversion_factor(i) = engine_utils::mJy_beam_to_uK(1, freq_Hz, det_fwhm);
        }
    }

    // convert to Jy/pixel
    else if (units == "Jy/pixel") {
        for (Eigen::Index i=0; i<n_dets; i++) {
            // current detector's array
            auto array = apt["array"](i);
            // det fwhm
            auto det_fwhm = (std::get<0>(array_fwhms[array]) + std::get<1>(array_fwhms[array]))/2;
            // beam area in steradians
            auto beam_area_rad = 2.*pi*pow(det_fwhm*FWHM_TO_STD*ASEC_TO_RAD,2);
            // get Jy/pixel
            flux_conversion_factor(i) = 1e-3/beam_area_rad*pow(pixel_size_rad,2);
        }
    }

    // get mean flux conversion factor from all unflagged detectors
    for (Eigen::Index i=0; i<n_arrays; i++) {
        auto array = arrays[i];
        // start indices for current array
        Eigen::Index start = std::get<0>(array_limits[array]);
        // end indices for current array
        Eigen::Index end = std::get<1>(array_limits[array]);
        // number of good detectors
        Eigen::Index n_good_dets = 0;
        // name of array
        std::string name = array_name_map[array];
        // loop through detectors in current array
        for (Eigen::Index j=start; j<end; j++) {
            // if good
            if (apt["flag"](j)!=1) {
                mean_flux_conversion_factor[name] += flux_conversion_factor(j);
                n_good_dets++;
            }
        }
        // calculate mean flux conversion factor
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

    // stores nw number
    nws.setZero(n_nws);
    // stores array number
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

        // nw a fwhm
        auto nw_a_fwhm = apt["a_fwhm"](Eigen::seq(std::get<0>(nw_limits[key]),
                                                std::get<1>(nw_limits[key])-1));
        // nw b fwhm
        auto nw_b_fwhm = apt["b_fwhm"](Eigen::seq(std::get<0>(nw_limits[key]),
                                                  std::get<1>(nw_limits[key])-1));
        // number of good detectors
        Eigen::Index n_good_det = (apt["flag"](Eigen::seq(std::get<0>(nw_limits[key]),
                                                         std::get<1>(nw_limits[key])-1)).array()==0).count();

        // remove flagged dets
        Eigen::Index k = std::get<0>(array_limits[key]);
        for (Eigen::Index i=0; i<nw_a_fwhm.size(); i++) {
            if (apt["flag"](k)!=1) {
                std::get<0>(nw_fwhms[key]) = std::get<0>(nw_fwhms[key]) + nw_a_fwhm(i);
                std::get<1>(nw_fwhms[key]) = std::get<1>(nw_fwhms[key]) + nw_b_fwhm(i);
            }
            k++;
        }

        std::get<0>(nw_fwhms[key]) = std::get<0>(nw_fwhms[key])/n_good_det;
        std::get<1>(nw_fwhms[key]) = std::get<1>(nw_fwhms[key])/n_good_det;

        // average of nw fwhms in both axes
        double avg_nw_fwhm = (std::get<0>(nw_fwhms[key]) + std::get<1>(nw_fwhms[key]))/2;
        // average nw beam area
        nw_beam_areas[key] = 2.*pi*pow(avg_nw_fwhm/STD_TO_FWHM,2);
    }

    // set up array values
    array_limits.clear();
    array_fwhms.clear();
    array_pas.clear();
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
    // loop through arrays
    for (auto const& [key, val] : array_limits) {
        arrays(j) = key;
        j++;
        array_fwhms[key] = std::tuple<double,double>{0, 0};

        // array a fwhm
        auto array_a_fwhm = apt["a_fwhm"](Eigen::seq(std::get<0>(array_limits[key]),
                                                  std::get<1>(array_limits[key])-1));
        // array b fwhm
        auto array_b_fwhm = apt["b_fwhm"](Eigen::seq(std::get<0>(array_limits[key]),
                                                  std::get<1>(array_limits[key])-1));
        // array rotation/position angle
        auto array_pa = apt["angle"](Eigen::seq(std::get<0>(array_limits[key]),
                                                     std::get<1>(array_limits[key])-1));
        // number of good detectors
        Eigen::Index n_good_det = (apt["flag"](Eigen::seq(std::get<0>(array_limits[key]),
                                                         std::get<1>(array_limits[key])-1)).array()==0).count();

        // remove flagged dets
        Eigen::Index k = std::get<0>(array_limits[key]);
        for (Eigen::Index i=0; i<array_a_fwhm.size(); i++) {
            if (apt["flag"](k)!=1) {
                std::get<0>(array_fwhms[key]) = std::get<0>(array_fwhms[key]) + array_a_fwhm(i);
                std::get<1>(array_fwhms[key]) = std::get<1>(array_fwhms[key]) + array_b_fwhm(i);
                array_pas[key] = array_pas[key] + array_pa(i);
            }
            k++;
        }

        // average fwhms and PA
        std::get<0>(array_fwhms[key]) = std::get<0>(array_fwhms[key])/n_good_det;
        std::get<1>(array_fwhms[key]) = std::get<1>(array_fwhms[key])/n_good_det;
        array_pas[key] = array_pas[key]/n_good_det;
        // average of array fwhms in both axes
        double avg_array_fwhm = (std::get<0>(array_fwhms[key]) + std::get<1>(array_fwhms[key]))/2;
        // average array beam area
        array_beam_areas[key] = 2.*pi*pow(avg_array_fwhm*FWHM_TO_STD,2);
    }

    // vector to hold unique fg's in apt
    std::vector<Eigen::Index> fg_temp;
    // init fg
    fg_temp.push_back(apt["fg"](0));

    // loop through detectors
    for (Eigen::Index i=1; i<apt["fg"].size(); i++) {
        // map to Eigen::Vector to use any()
        Eigen::Map<Eigen::VectorXI> x(fg_temp.data(),fg_temp.size());
        // if current fg is not in fg_temp
        if (!(x.array() == apt["fg"](i)).any()) {
            // append to fg_temp
            fg_temp.push_back(apt["fg"](i));
        }
    }
    // allocate fg_temp to fg vector
    fg = Eigen::Map<Eigen::VectorXI>(fg_temp.data(),fg_temp.size());
}

} // namespace engine
