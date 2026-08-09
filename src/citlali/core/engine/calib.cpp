#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/utils.h>
#include <citlali/core/engine/calib.h>
#include <citlali/core/error/error.h>
#include <citlali/core/utils/toltec_io.h>
#include <citlali/core/utils/sha256.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

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

bool same_frequency_identity(double left, double right) {
    if (!std::isfinite(left) || !std::isfinite(right)) {
        return false;
    }
    const double scale = std::max({1.0, std::abs(left), std::abs(right)});
    return std::abs(left - right) <=
           64.0 * std::numeric_limits<double>::epsilon() * scale;
}

int interface_network_id(const std::string &interface_name) {
    constexpr std::string_view prefix{"toltec"};
    if (interface_name.rfind(prefix, 0) != 0 ||
        interface_name.size() == prefix.size()) {
        throw citlali::error::io(
            "invalid TolTEC interface name for APT acquisition binding: " +
            interface_name);
    }
    const auto suffix = std::string_view{interface_name}.substr(prefix.size());
    int network = -1;
    const auto parsed =
        std::from_chars(suffix.data(), suffix.data() + suffix.size(), network);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != suffix.data() + suffix.size() || network < 0) {
        throw citlali::error::io(
            "invalid TolTEC interface name for APT acquisition binding: " +
            interface_name);
    }
    return network;
}

}  // namespace

namespace engine {

void Calib::get_apt(const std::string &filepath, std::vector<std::string> &raw_filenames, std::vector<std::string> &interfaces) {
    apt_acquisition_binding = {};
    // store apt filepath
    apt_filepath = filepath;
    apt_acquisition_binding.artifact_sha256 =
        citlali::utils::sha256_file(filepath);
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

    if (raw_filenames.empty() || raw_filenames.size() != interfaces.size()) {
        throw citlali::error::io(
            "APT acquisition binding requires one raw file for every interface");
    }

    const Eigen::Index apt_row_count = apt_temp["nw"].size();
    for (const auto &key : apt_header_keys) {
        if (apt_temp[key].size() != apt_row_count) {
            throw citlali::error::io(
                "APT required-column cardinality mismatch for " + key);
        }
    }

    struct RawNetworkIdentity {
        int network = -1;
        Eigen::VectorXd tone_frequency_hz;
    };
    std::vector<RawNetworkIdentity> raw_networks;
    raw_networks.reserve(raw_filenames.size());
    std::set<int> seen_networks;
    Eigen::Index n_dets_temp = 0;

    for (std::size_t index = 0; index < raw_filenames.size(); ++index) {
        netCDF::NcFile fo(raw_filenames[index], netCDF::NcFile::read);
        const int interface_network = interface_network_id(interfaces[index]);
        int roach_index = -1;
        fo.getVar("Header.Toltec.RoachIndex").getVar(&roach_index);
        if (roach_index != interface_network) {
            throw citlali::error::io(
                "raw-file RoachIndex disagrees with interface identity for " +
                interfaces[index]);
        }
        if (!seen_networks.insert(roach_index).second) {
            throw citlali::error::io(
                "duplicate TolTEC network in APT acquisition binding: " +
                std::to_string(roach_index));
        }

        const auto signal = fo.getVar("Data.Toltec.Is");
        const auto tone_var = fo.getVar("Header.Toltec.ToneFreq");
        const auto lo_var = fo.getVar("Header.Toltec.LoCenterFreq");
        if (signal.isNull() || signal.getDimCount() < 2 || tone_var.isNull() ||
            tone_var.getDimCount() < 2 || lo_var.isNull()) {
            throw citlali::error::io(
                "raw acquisition identity is unavailable for network " +
                std::to_string(roach_index));
        }
        const Eigen::Index detector_count = static_cast<Eigen::Index>(
            signal.getDim(1).getSize());
        const Eigen::Index sweep_count = static_cast<Eigen::Index>(
            tone_var.getDim(0).getSize());
        const Eigen::Index tone_count = static_cast<Eigen::Index>(
            tone_var.getDim(1).getSize());
        if (detector_count <= 0 || sweep_count <= 0 ||
            tone_count != detector_count) {
            throw citlali::error::io(
                "raw detector/tone cardinality mismatch for network " +
                std::to_string(roach_index));
        }
        Eigen::MatrixXd tone_frequency(tone_count, sweep_count);
        tone_var.getVar(tone_frequency.data());
        double lo_frequency = std::numeric_limits<double>::quiet_NaN();
        lo_var.getVar(&lo_frequency);
        if (!std::isfinite(lo_frequency)) {
            throw citlali::error::io(
                "non-finite raw LO frequency for network " +
                std::to_string(roach_index));
        }
        RawNetworkIdentity identity;
        identity.network = roach_index;
        identity.tone_frequency_hz =
            tone_frequency.col(0).array() + lo_frequency;
        if (!identity.tone_frequency_hz.array().isFinite().all()) {
            throw citlali::error::io(
                "non-finite raw tone frequency for network " +
                std::to_string(roach_index));
        }
        for (Eigen::Index left = 0; left < detector_count; ++left) {
            for (Eigen::Index right = left + 1; right < detector_count; ++right) {
                if (same_frequency_identity(identity.tone_frequency_hz(left),
                                            identity.tone_frequency_hz(right))) {
                    throw citlali::error::io(
                        "duplicate raw acquisition tone key for network " +
                        std::to_string(roach_index));
                }
            }
        }
        n_dets_temp += detector_count;
        raw_networks.push_back(std::move(identity));
        fo.close();
    }

    std::map<std::string, Eigen::VectorXd> ordered_apt;
    for (const auto &key : apt_header_keys) {
        ordered_apt[key].setZero(n_dets_temp);
    }
    ordered_apt["kids_tone"].setZero(n_dets_temp);

    Eigen::Index output_row = 0;
    std::vector<bool> used(static_cast<std::size_t>(apt_temp["nw"].size()), false);
    for (const auto &raw : raw_networks) {
        const Eigen::Index raw_count = raw.tone_frequency_hz.size();
        const Eigen::Index apt_count =
            (apt_temp["nw"].array() == raw.network).count();
        if (apt_count != raw_count) {
            throw citlali::error::io(
                "APT/raw tone cardinality mismatch for network " +
                std::to_string(raw.network));
        }
        for (Eigen::Index local_tone = 0; local_tone < raw_count;
             ++local_tone) {
            std::vector<Eigen::Index> matches;
            for (Eigen::Index apt_row = 0; apt_row < apt_temp["nw"].size();
                 ++apt_row) {
                if (!used[static_cast<std::size_t>(apt_row)] &&
                    apt_temp["nw"](apt_row) == raw.network &&
                    same_frequency_identity(
                        apt_temp["tone_freq"](apt_row),
                        raw.tone_frequency_hz(local_tone))) {
                    matches.push_back(apt_row);
                }
            }
            if (matches.size() != 1) {
                throw citlali::error::io(
                    "APT acquisition key is missing or duplicated for network " +
                    std::to_string(raw.network) + " local tone " +
                    std::to_string(local_tone));
            }
            const Eigen::Index apt_row = matches.front();
            used[static_cast<std::size_t>(apt_row)] = true;
            for (const auto &key : apt_header_keys) {
                ordered_apt[key](output_row) = apt_temp[key](apt_row);
            }
            ordered_apt["kids_tone"](output_row) =
                static_cast<double>(local_tone);
            ++output_row;
        }
    }
    if (output_row != apt_row_count ||
        std::find(used.begin(), used.end(), false) != used.end()) {
        throw citlali::error::io(
            "APT contains acquisition keys not present in the raw observation");
    }

    apt = std::move(ordered_apt);
    apt_acquisition_binding.available = true;
    apt_acquisition_binding.valid = true;
    apt_acquisition_binding.mode =
        "explicit_network_local_tone_frequency_join_v1";
    apt_acquisition_binding.key_schema =
        "raw_observation_artifact+network+network_local_tone_frequency";
    apt_acquisition_binding.detail =
        "unique complete raw/APT acquisition-key join; APT row order is not authoritative";
    apt_acquisition_binding.detector_count = n_dets_temp;
    apt_acquisition_binding.network_count =
        static_cast<Eigen::Index>(raw_networks.size());
    std::ostringstream raw_identity;
    raw_identity << std::setprecision(std::numeric_limits<double>::max_digits10)
                 << "raw-observation-acquisition-identity-v1";
    for (std::size_t index = 0; index < raw_networks.size(); ++index) {
        raw_identity << "|file=" << raw_filenames[index]
                     << ",interface=" << interfaces[index]
                     << ",network=" << raw_networks[index].network
                     << ",tones=";
        for (Eigen::Index tone = 0;
             tone < raw_networks[index].tone_frequency_hz.size(); ++tone) {
            if (tone != 0) {
                raw_identity << ',';
            }
            raw_identity << raw_networks[index].tone_frequency_hz(tone);
        }
    }
    apt_acquisition_binding.raw_observation_identity = raw_identity.str();
    std::ostringstream binding_identity;
    binding_identity
        << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "apt-acquisition-binding-v1|apt_sha256="
        << apt_acquisition_binding.artifact_sha256
        << "|raw_identity=" << raw_identity.str()
        << "|ordered_join=";
    for (Eigen::Index row = 0; row < n_dets_temp; ++row) {
        if (row != 0) {
            binding_identity << ';';
        }
        binding_identity
            << "network=" << apt["nw"](row)
            << ",local_tone=" << apt["kids_tone"](row)
            << ",tone_frequency_hz=" << apt["tone_freq"](row)
            << ",uid=" << apt["uid"](row);
    }
    apt_acquisition_binding.binding_sha256 =
        citlali::utils::sha256(binding_identity.str());

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
    (void)pixel_size_rad;
    if (units != "mJy/beam") {
        throw citlali::error::invalid_config(
            "SCI-CAL-001 supports only top-of-atmosphere point-source-peak mJy/beam; unsupported unit " +
            units);
    }
    // flux conversion is per detector
    flux_conversion_factor.resize(n_dets);
    mean_flux_conversion_factor.clear();
    flux_conversion_factor.setOnes();

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
