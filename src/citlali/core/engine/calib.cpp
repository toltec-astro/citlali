#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/utils.h>
#include <citlali/core/engine/calib.h>
#include <citlali/core/error/error.h>
#include <citlali/core/utils/toltec_io.h>

#include <cmath>
#include <stdexcept>

namespace {

std::string join_column_names(const std::vector<std::string> &names) {
    std::string result;
    for (const auto &name : names) {
        if (!result.empty()) {
            result += ", ";
        }
        result += name;
    }
    return result;
}

}  // namespace

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

    // reject tables with missing required columns
    if (!missing_header_keys.empty()) {
        throw citlali::error::io(
            "APT table is missing required columns: [" +
            join_column_names(missing_header_keys) + "]");
    }

    // reject tables with empty required columns
    if (!empty_header_keys.empty()) {
        throw citlali::error::io(
            "APT table columns are empty: [" +
            join_column_names(empty_header_keys) + "]");
    }

    // pointing calculations require an altaz APT reference frame
    if (map_with_strs["Radesys"]!="altaz") {
        throw citlali::error::io(
            "APT table reference frame must be altaz");
    }

    // set apt table
    apt = apt_temp;

    // A matched APT can retain fully flagged placeholder rows for a network
    // that was absent from this observation. Restrict the table to the raw
    // interfaces before setup validates its network and array groups.

    // vectors to hold roach indices and missing roaches
    std::vector<Eigen::Index> roach_indices, missing;
    Eigen::Index n_dets_temp = 0;

    // get roach indices from raw data files
    for (Eigen::Index i=0; i<raw_filenames.size(); ++i) {
        netCDF::NcFile fo(raw_filenames[i], netCDF::NcFile::read);
        // get roach index
        int roach_index;
        fo.getVar("Header.Toltec.RoachIndex").getVar(&roach_index);
        roach_indices.push_back(roach_index);
        fo.close();
    }

    // vector to hold interface number
    Eigen::VectorXi interfaces_vec(interfaces.size());

    // get network interfaces
    for (Eigen::Index i=0; i<interfaces.size(); ++i) {
        interfaces_vec(i) = std::stoi(interfaces[i].substr(6));
    }

    // count up number of detectors
    for (Eigen::Index i=0; i<interfaces.size(); ++i) {
        n_dets_temp = n_dets_temp + (apt["nw"].array() == interfaces_vec(i)).count();
    }

    // clear apt
    apt_temp.clear();
    // populate apt temp
    for (auto const& value: apt_header_keys) {
        apt_temp[value].setZero(n_dets_temp);
        Eigen::Index i = 0;
        for (Eigen::Index j=0; j<apt["nw"].size(); ++j) {
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
        fo.getVar(hwpr_install_v).getVar(&run_hwpr);

        // if not enabled or running
        if (run_hwpr) {
            // get hwpr signal
            Eigen::Index n_pts = fo.getVar("Data.Hwp.").getDim(0).getSize();
            hwpr_angle.resize(n_pts);
            // hwpr signal
            fo.getVar("Data.Hwp.").getVar(hwpr_angle.data());

            // if real data
            if (!sim_obs) {
                // get hwpr time for interpolation
                hwpr_ts.resize(n_pts,6);

                // timing for hwpr (temporary)
                fo.getVar("Data.Hwp.Ts").getVar(hwpr_ts.data());
                hwpr_ts.transposeInPlace();

                // UT time for hwpr
                Eigen::Index recvt_n_pts = fo.getVar("Data.Hwp.Uts").getDim(0).getSize();
                hwpr_recvt.resize(recvt_n_pts);
                fo.getVar("Data.Hwp.Uts").getVar(hwpr_recvt.data());

                // fpga frequency
                fo.getVar("Header.Toltec.FpgaFreq").getVar(&hwpr_fpga_freq);
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
    flux_conversion_factor.resize(n_dets);
    mean_flux_conversion_factor.clear();

    // default is mJy/beam (apt should always be in mJy/beam)
    if (units == "mJy/beam") {
        flux_conversion_factor.setOnes();
    }

    // convert to MJy/sr
    else if (units == "MJy/sr") {
        for (Eigen::Index i=0; i<n_dets; ++i) {
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

    // convert to Rayleigh-Jeans uK brightness temperature.
    // mJy/beam is first converted to Jy/sr using the Gaussian beam solid angle.
    else if (units == "uK") {
        engine_utils::toltecIO toltec_io;
        for (Eigen::Index i=0; i<n_dets; ++i) {
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
        for (Eigen::Index i=0; i<n_dets; ++i) {
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
    for (Eigen::Index i=0; i<n_arrays; ++i) {
        auto array = arrays[i];
        // number of good detectors
        Eigen::Index n_good_dets = 0;
        // name of array
        std::string name = array_name_map[array];
        // loop through detectors in current array
        for (const auto &j: array_detector_indices[array]) {
            // if good
            if (apt["flag"](j)==0) {
                mean_flux_conversion_factor[name] += flux_conversion_factor(j);
                n_good_dets++;
            }
        }
        if (n_good_dets <= 0) {
            throw std::runtime_error(
                "cannot calculate mean flux conversion factor: array has no unflagged detectors");
        }
        // calculate mean flux conversion factor
        mean_flux_conversion_factor[name] = mean_flux_conversion_factor[name]/n_good_dets;
    }
}

void Calib::setup() {
    // get number of detectors
    n_dets = apt["uid"].size();

    if (n_dets <= 0) {
        throw std::runtime_error("APT table has no detectors");
    }

    const std::vector<std::string> setup_keys = {
        "uid", "nw", "array", "fg", "flag", "a_fwhm", "b_fwhm", "angle"
    };
    for (const auto &key: setup_keys) {
        auto it = apt.find(key);
        if (it == apt.end()) {
            throw std::runtime_error("APT table is missing required setup column " + key);
        }
        if (it->second.size() != n_dets) {
            throw std::runtime_error("APT column " + key + " length does not match uid length");
        }
    }

    auto read_index = [&](const std::string &key, Eigen::Index i) {
        const double value = apt[key](i);
        const double rounded = std::round(value);
        if (!std::isfinite(value) || std::abs(value - rounded) > 1e-6) {
            throw std::runtime_error("APT column " + key + " contains a non-integer group id");
        }
        return static_cast<Eigen::Index>(rounded);
    };

    auto validate_contiguous = [&](const auto &groups, const std::string &key) {
        for (const auto &[group_id, indices]: groups) {
            if (indices.empty()) {
                throw std::runtime_error("APT " + key + " group has no detector indices");
            }
            for (std::size_t i=1; i<indices.size(); ++i) {
                if (indices[i] != indices[i - 1] + 1) {
                    throw std::runtime_error(
                        "APT rows for " + key + " group are not contiguous; sort the APT before reduction");
                }
            }
        }
    };

    nw_detector_indices.clear();
    array_detector_indices.clear();
    for (Eigen::Index i=0; i<n_dets; ++i) {
        nw_detector_indices[read_index("nw", i)].push_back(i);
        array_detector_indices[read_index("array", i)].push_back(i);
    }
    validate_contiguous(nw_detector_indices, "nw");
    validate_contiguous(array_detector_indices, "array");

    // get number of networks and arrays
    n_nws = nw_detector_indices.size();
    n_arrays = array_detector_indices.size();

    // stores nw number
    nws.setZero(n_nws);
    // stores array number
    arrays.setZero(n_arrays);

    // set up network values
    nw_limits.clear();
    nw_fwhms.clear();
    nw_pas.clear();
    nw_beam_areas.clear();

    for (const auto &[key, indices]: nw_detector_indices) {
        nw_limits[key] = std::tuple<Eigen::Index, Eigen::Index>{indices.front(), indices.back() + 1};
    }

    // get average fwhms for networks
    Eigen::Index j = 0;
    for (auto const& [key, val] : nw_limits) {
        nws(j) = key;
        j++;
        nw_fwhms[key] = std::tuple<double,double>{0, 0};

        // number of good detectors
        Eigen::Index n_good_det = 0;

        // remove flagged dets
        for (const auto &idx: nw_detector_indices[key]) {
            if (apt["flag"](idx)==0) {
                std::get<0>(nw_fwhms[key]) = std::get<0>(nw_fwhms[key]) + apt["a_fwhm"](idx);
                std::get<1>(nw_fwhms[key]) = std::get<1>(nw_fwhms[key]) + apt["b_fwhm"](idx);
                n_good_det++;
            }
        }

        if (n_good_det <= 0) {
            throw std::runtime_error("APT nw group has no unflagged detectors");
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

    for (const auto &[key, indices]: array_detector_indices) {
        array_limits[key] = std::tuple<Eigen::Index, Eigen::Index>{indices.front(), indices.back() + 1};
    }

    // get average fwhms for arrays
    j = 0;
    // loop through arrays
    for (auto const& [key, val] : array_limits) {
        arrays(j) = key;
        j++;
        array_fwhms[key] = std::tuple<double,double>{0, 0};
        array_pas[key] = 0;
        // number of good detectors
        Eigen::Index n_good_det = 0;

        // remove flagged dets
        for (const auto &idx: array_detector_indices[key]) {
            if (apt["flag"](idx)==0) {
                std::get<0>(array_fwhms[key]) = std::get<0>(array_fwhms[key]) + apt["a_fwhm"](idx);
                std::get<1>(array_fwhms[key]) = std::get<1>(array_fwhms[key]) + apt["b_fwhm"](idx);
                array_pas[key] = array_pas[key] + apt["angle"](idx);
                n_good_det++;
            }
        }

        if (n_good_det <= 0) {
            throw std::runtime_error("APT array group has no unflagged detectors");
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
    for (Eigen::Index i=1; i<apt["fg"].size(); ++i) {
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
