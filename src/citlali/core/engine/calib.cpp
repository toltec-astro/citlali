#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/utils.h>
#include <citlali/core/engine/calib.h>
#include <citlali/core/error/error.h>
#include <citlali/core/pipeline/apt_detector_relation.h>
#include <citlali/core/utils/toltec_io.h>

#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>

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

std::string read_exact_bytes(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw citlali::error::io(
            "unable to open canonical APT input " + path.string());
    }
    std::string bytes((std::istreambuf_iterator<char>(stream)),
                      std::istreambuf_iterator<char>());
    if (stream.bad()) {
        throw citlali::error::io(
            "unable to read complete canonical APT input " + path.string());
    }
    return bytes;
}

std::filesystem::path receipt_path_for(
    const std::filesystem::path &artifact_path) {
    return std::filesystem::path(artifact_path.string() + ".sha256");
}

std::int64_t parse_toltec_interface(std::string_view interface_name) {
    constexpr std::string_view prefix = "toltec";
    if (!interface_name.starts_with(prefix) ||
        interface_name.size() == prefix.size()) {
        throw citlali::error::io(
            "raw interface is not an exact canonical toltecN identifier");
    }
    std::int64_t network = -1;
    const auto digits = interface_name.substr(prefix.size());
    const auto [end, error] = std::from_chars(
        digits.data(), digits.data() + digits.size(), network);
    if (error != std::errc{} || end != digits.data() + digits.size() ||
        network < 0 ||
        interface_name != std::string(prefix) + std::to_string(network)) {
        throw citlali::error::io(
            "raw interface is not an exact canonical toltecN identifier");
    }
    return network;
}

std::vector<citlali::pipeline::AptDetectorColumnAddress>
make_raw_detector_layout(
    const std::vector<std::string> &raw_filenames,
    const std::vector<std::string> &interfaces) {
    if (raw_filenames.empty() || raw_filenames.size() != interfaces.size()) {
        throw citlali::error::io(
            "canonical APT admission requires equal nonempty raw-file and interface lists");
    }
    std::set<std::int64_t> networks;
    std::set<std::string> interface_names;
    std::vector<citlali::pipeline::AptDetectorColumnAddress> layout;
    for (std::size_t input_index = 0; input_index < raw_filenames.size();
         ++input_index) {
        const auto declared_network =
            parse_toltec_interface(interfaces[input_index]);
        if (!networks.insert(declared_network).second ||
            !interface_names.insert(interfaces[input_index]).second) {
            throw citlali::error::io(
                "canonical APT admission received a duplicate raw network/interface");
        }
        netCDF::NcFile file(raw_filenames[input_index],
                            netCDF::NcFile::read);
        const auto roach = file.getVar("Header.Toltec.RoachIndex");
        if (roach.isNull() || !roach.getDims().empty() ||
            roach.getType().getTypeClass() != netCDF::NcType::nc_INT) {
            throw citlali::error::io(
                "raw file requires scalar int Header.Toltec.RoachIndex");
        }
        int roach_index = -1;
        roach.getVar(&roach_index);
        if (roach_index != declared_network) {
            throw citlali::error::io(
                "raw file RoachIndex disagrees with its paired interface");
        }
        const auto detector_data = file.getVar("Data.Toltec.Is");
        if (detector_data.isNull()) {
            throw citlali::error::io(
                "raw file requires two-dimensional nonempty Data.Toltec.Is");
        }
        const auto dimensions = detector_data.getDims();
        if (dimensions.size() != 2 ||
            dimensions[1].getSize() == 0 ||
            dimensions[1].getSize() >
                static_cast<std::size_t>(
                    citlali::pipeline::canonical_apt_v1::uid_v1_max + 1)) {
            throw citlali::error::io(
                "raw file requires two-dimensional nonempty Data.Toltec.Is");
        }
        for (std::size_t channel = 0; channel < dimensions[1].getSize();
             ++channel) {
            if (layout.size() == std::numeric_limits<std::size_t>::max()) {
                throw citlali::error::io(
                    "raw detector-column count is not representable");
            }
            layout.push_back({layout.size(), declared_network,
                              static_cast<std::int64_t>(channel)});
        }
    }
    return layout;
}

void require_matched_raw_sources(
    const citlali::pipeline::canonical_apt_observation_v1::TargetManifest
        &target,
    const std::vector<std::string> &raw_filenames,
    const std::vector<std::string> &interfaces) {
    if (raw_filenames.size() != interfaces.size() ||
        target.inputs.size() != raw_filenames.size()) {
        throw citlali::error::io(
            "matched APT target and supplied raw-input cardinalities differ");
    }
    std::map<std::int64_t,
             const citlali::pipeline::canonical_apt_observation_v1::
                 TargetInput *>
        inputs;
    for (const auto &input : target.inputs) {
        if (!inputs.emplace(input.network, &input).second) {
            throw citlali::error::io(
                "matched APT target repeats a raw network");
        }
    }
    for (std::size_t index = 0; index < raw_filenames.size(); ++index) {
        const auto network = parse_toltec_interface(interfaces[index]);
        const auto input = inputs.find(network);
        if (input == inputs.end() ||
            input->second->interface_name != interfaces[index]) {
            throw citlali::error::io(
                "matched APT target does not bind a supplied raw network/interface");
        }
        std::error_code error;
        const auto byte_count = std::filesystem::file_size(
            raw_filenames[index], error);
        if (error || byte_count != input->second->raw_source.byte_count ||
            "sha256:" + citlali::utils::sha256_file(raw_filenames[index]) !=
                input->second->raw_source.content_sha256) {
            throw citlali::error::io(
                "matched APT target raw-source byte identity does not match the supplied raw file");
        }
    }
}

double exact_numeric_compatibility_value(
    const citlali::pipeline::canonical_apt_v1::Value &value,
    std::string_view field_name) {
    namespace apt = citlali::pipeline::canonical_apt_v1;
    if (std::holds_alternative<apt::NullValue>(value)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (const auto *typed = std::get_if<std::int64_t>(&value)) {
        const double converted = static_cast<double>(*typed);
        const double int64_limit = std::ldexp(1.0, 63);
        if (converted >= int64_limit || converted < -int64_limit ||
            static_cast<std::int64_t>(converted) != *typed) {
            throw citlali::error::io(
                "canonical APT integral field is not exactly representable in the legacy numeric view: " +
                std::string(field_name));
        }
        return converted;
    }
    if (const auto *typed = std::get_if<double>(&value)) {
        return *typed;
    }
    if (const auto *typed = std::get_if<bool>(&value)) {
        return *typed ? 1.0 : 0.0;
    }
    throw citlali::error::io(
        "canonical APT string field cannot enter the legacy numeric view: " +
        std::string(field_name));
}

Eigen::Index canonical_array_for_network(std::int64_t network) {
    if (network >= 0 && network <= 6) {
        return 0;
    }
    if (network >= 7 && network <= 10) {
        return 1;
    }
    if (network >= 11 && network <= 12) {
        return 2;
    }
    throw citlali::error::io(
        "typed APT network is outside the TolTEC enum {0..12}");
}

template <typename Row>
std::map<std::string, Eigen::VectorXd> make_canonical_numeric_view(
    const std::vector<Row> &rows,
    const citlali::pipeline::AptDetectorRelation &relation,
    const std::vector<std::string> &legacy_keys) {
    using NetworkChannel = std::pair<std::int64_t, std::int64_t>;
    if (relation.bindings().size() >
        static_cast<std::size_t>(
            std::numeric_limits<Eigen::Index>::max())) {
        throw citlali::error::io(
            "typed APT relation is too large for the reduction matrix index");
    }
    std::map<NetworkChannel, const Row *> row_by_relation;
    for (const auto &row : rows) {
        if (!row_by_relation.emplace(NetworkChannel{row.network, row.channel},
                                     &row).second) {
            throw citlali::error::io(
                "canonical APT repeats a typed network/channel relation");
        }
    }
    std::set<std::string> keys(legacy_keys.begin(), legacy_keys.end());
    keys.insert("kids_tone");
    std::map<std::string, Eigen::VectorXd> view;
    for (const auto &key : keys) {
        view[key].resize(static_cast<Eigen::Index>(relation.bindings().size()));
    }
    for (const auto &binding : relation.bindings()) {
        if (binding.detector_column >
            static_cast<std::size_t>(
                std::numeric_limits<Eigen::Index>::max())) {
            throw citlali::error::io(
                "typed APT detector column is not representable by the reduction matrix index");
        }
        const auto row = row_by_relation.find(
            NetworkChannel{binding.network, binding.kids_tone});
        if (row == row_by_relation.end() || row->second->uid != binding.uid) {
            throw citlali::error::io(
                "typed detector binding no longer matches its verified canonical row");
        }
        const auto column = static_cast<Eigen::Index>(
            binding.detector_column);
        view.at("uid")(column) = static_cast<double>(binding.uid);
        view.at("tone_freq")(column) = row->second->tone_frequency_hz;
        view.at("array")(column) = static_cast<double>(row->second->array);
        view.at("nw")(column) = static_cast<double>(binding.network);
        view.at("kids_tone")(column) =
            static_cast<double>(binding.kids_tone);
        for (const auto &key : legacy_keys) {
            if (key == "uid" || key == "tone_freq" || key == "array" ||
                key == "nw") {
                continue;
            }
            const auto field = row->second->fields.find(key);
            if (field == row->second->fields.end()) {
                throw citlali::error::io(
                    "canonical APT omits required legacy compatibility field " +
                    key);
            }
            view.at(key)(column) =
                exact_numeric_compatibility_value(field->second, key);
        }
    }
    return view;
}

}  // namespace

namespace engine {

void Calib::get_apt(const std::string &filepath,
                    std::vector<std::string> &raw_filenames,
                    std::vector<std::string> &interfaces) {
    Calib candidate = *this;
    candidate.apt_detector_relation_.reset();
    candidate.load_legacy_apt_in_place(
        filepath, raw_filenames, interfaces);
    commit_apt_state(std::move(candidate));
}

void Calib::load_legacy_apt_in_place(
    const std::string &filepath, std::vector<std::string> &raw_filenames,
    std::vector<std::string> &interfaces) {
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

void Calib::get_canonical_baseline_apt(
    const std::string &filepath,
    const std::vector<std::string> &raw_filenames,
    const std::vector<std::string> &interfaces) {
    namespace relation = citlali::pipeline;
    namespace observation =
        citlali::pipeline::canonical_apt_observation_v1;
    const auto artifact_bytes = read_exact_bytes(filepath);
    const auto receipt_bytes =
        read_exact_bytes(receipt_path_for(filepath));
    const auto layout = make_raw_detector_layout(raw_filenames, interfaces);
    auto typed_relation = std::make_shared<const relation::AptDetectorRelation>(
        relation::admit_published_baseline_apt_relation(
            artifact_bytes, receipt_bytes, layout));
    const auto descriptor = observation::verify_baseline_descriptor(
        artifact_bytes, receipt_bytes);

    Calib candidate = *this;
    candidate.apt_filepath = filepath;
    candidate.apt = make_canonical_numeric_view(
        descriptor.document().rows, *typed_relation,
        candidate.apt_header_keys);
    candidate.apt_detector_relation_ = std::move(typed_relation);
    candidate.flux_conversion_factor.resize(0);
    candidate.mean_flux_conversion_factor.clear();
    candidate.setup();
    commit_apt_state(std::move(candidate));
}

void Calib::get_canonical_observation_apt(
    const std::string &filepath,
    const std::vector<std::string> &raw_filenames,
    const std::vector<std::string> &interfaces) {
    namespace relation = citlali::pipeline;
    namespace observation =
        citlali::pipeline::canonical_apt_observation_v1;
    const auto artifact_bytes = read_exact_bytes(filepath);
    const auto receipt_bytes =
        read_exact_bytes(receipt_path_for(filepath));
    const auto parsed =
        observation::parse_issued_matched_observation_ecsv_with_receipt(
            artifact_bytes, receipt_bytes);
    require_matched_raw_sources(parsed.target, raw_filenames, interfaces);
    const auto layout = make_raw_detector_layout(raw_filenames, interfaces);
    auto typed_relation = std::make_shared<const relation::AptDetectorRelation>(
        relation::admit_published_observation_apt_relation(
            artifact_bytes, receipt_bytes, layout));
    const auto &scope = typed_relation->published_scope();
    if (parsed.parent_content_revalidated ||
        scope.kind != relation::PublishedAptKind::matched_observation ||
        scope.parent_content_revalidated) {
        throw citlali::error::io(
            "runtime matched APT admission has an invalid consumer assurance scope");
    }

    Calib candidate = *this;
    candidate.apt_filepath = filepath;
    candidate.apt = make_canonical_numeric_view(
        parsed.output.rows, *typed_relation, candidate.apt_header_keys);
    candidate.apt_detector_relation_ = std::move(typed_relation);
    candidate.flux_conversion_factor.resize(0);
    candidate.mean_flux_conversion_factor.clear();
    candidate.setup();
    commit_apt_state(std::move(candidate));
}

void Calib::get_canonical_observation_apt(
    const std::string &filepath, const std::string &baseline_filepath,
    const std::vector<std::string> &raw_filenames,
    const std::vector<std::string> &interfaces) {
    namespace relation = citlali::pipeline;
    namespace observation =
        citlali::pipeline::canonical_apt_observation_v1;
    const auto baseline_bytes = read_exact_bytes(baseline_filepath);
    const auto baseline_receipt_bytes =
        read_exact_bytes(receipt_path_for(baseline_filepath));
    const auto verified_baseline = observation::verify_baseline_descriptor(
        baseline_bytes, baseline_receipt_bytes);
    const auto artifact_bytes = read_exact_bytes(filepath);
    const auto receipt_bytes =
        read_exact_bytes(receipt_path_for(filepath));
    const auto layout = make_raw_detector_layout(raw_filenames, interfaces);
    auto typed_relation = std::make_shared<const relation::AptDetectorRelation>(
        relation::admit_published_observation_apt_relation(
            artifact_bytes, receipt_bytes, verified_baseline, layout));
    const auto parsed =
        observation::parse_matched_observation_ecsv_with_receipt(
            artifact_bytes, receipt_bytes, verified_baseline);
    require_matched_raw_sources(parsed.target, raw_filenames, interfaces);

    Calib candidate = *this;
    candidate.apt_filepath = filepath;
    candidate.apt = make_canonical_numeric_view(
        parsed.output.rows, *typed_relation, candidate.apt_header_keys);
    candidate.apt_detector_relation_ = std::move(typed_relation);
    candidate.flux_conversion_factor.resize(0);
    candidate.mean_flux_conversion_factor.clear();
    candidate.setup();
    commit_apt_state(std::move(candidate));
}

bool Calib::has_apt_detector_relation() const noexcept {
    return apt_detector_relation_ != nullptr;
}

std::shared_ptr<const citlali::pipeline::AptDetectorRelation>
Calib::apt_detector_relation_handle() const noexcept {
    return apt_detector_relation_;
}

const citlali::pipeline::AptDetectorRelation &
Calib::require_apt_detector_relation() const {
    if (apt_detector_relation_ == nullptr) {
        throw std::logic_error(
            "Calib has no admitted typed artifact-scoped APT detector relation");
    }
    return *apt_detector_relation_;
}

void Calib::commit_apt_state(Calib &&candidate) noexcept {
    apt_filepath.swap(candidate.apt_filepath);
    apt.swap(candidate.apt);
    apt_detector_relation_.swap(candidate.apt_detector_relation_);
    fg.swap(candidate.fg);
    nws.swap(candidate.nws);
    arrays.swap(candidate.arrays);
    std::swap(n_dets, candidate.n_dets);
    std::swap(n_nws, candidate.n_nws);
    std::swap(n_arrays, candidate.n_arrays);
    nw_limits.swap(candidate.nw_limits);
    array_limits.swap(candidate.array_limits);
    nw_detector_indices.swap(candidate.nw_detector_indices);
    array_detector_indices.swap(candidate.array_detector_indices);
    nw_fwhms.swap(candidate.nw_fwhms);
    array_fwhms.swap(candidate.array_fwhms);
    nw_pas.swap(candidate.nw_pas);
    array_pas.swap(candidate.array_pas);
    nw_beam_areas.swap(candidate.nw_beam_areas);
    array_beam_areas.swap(candidate.array_beam_areas);
    flux_conversion_factor.swap(candidate.flux_conversion_factor);
    mean_flux_conversion_factor.swap(
        candidate.mean_flux_conversion_factor);
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
    if (apt_detector_relation_ != nullptr) {
        if (apt_detector_relation_->bindings().size() >
            static_cast<std::size_t>(
                std::numeric_limits<Eigen::Index>::max())) {
            throw std::runtime_error(
                "typed APT detector relation is too large for the reduction matrix index");
        }
        n_dets = static_cast<Eigen::Index>(
            apt_detector_relation_->bindings().size());
    }
    else {
        n_dets = apt["uid"].size();
    }

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
        const double tolerance =
            apt_detector_relation_ == nullptr ? 1e-6 : 0.0;
        if (!std::isfinite(value) ||
            std::abs(value - rounded) > tolerance) {
            throw std::runtime_error("APT column " + key + " contains a non-integer group id");
        }
        return static_cast<Eigen::Index>(rounded);
    };

    if (apt_detector_relation_ != nullptr) {
        const auto kids_tone = apt.find("kids_tone");
        if (kids_tone == apt.end() || kids_tone->second.size() != n_dets) {
            throw std::runtime_error(
                "typed APT compatibility view is missing kids_tone");
        }
        for (Eigen::Index i = 0; i < n_dets; ++i) {
            const auto &binding = apt_detector_relation_->binding_for_column(
                static_cast<std::size_t>(i));
            if (apt["uid"](i) != static_cast<double>(binding.uid) ||
                apt["nw"](i) != static_cast<double>(binding.network) ||
                kids_tone->second(i) !=
                    static_cast<double>(binding.kids_tone)) {
                throw std::runtime_error(
                    "legacy numeric APT view disagrees with its immutable typed detector relation");
            }
            const auto flag = read_index("flag", i);
            (void)read_index("fg", i);
            if (flag != 0 && flag != 1) {
                throw std::runtime_error(
                    "typed Calib detector flag is outside the canonical closed set {0,1}");
            }
            if (binding.flag.has_value() && *binding.flag == 1 &&
                flag == 0) {
                throw std::runtime_error(
                    "typed Calib cannot clear an artifact-declared detector flag");
            }
            if (flag == 0 &&
                (!std::isfinite(apt["a_fwhm"](i)) ||
                 !std::isfinite(apt["b_fwhm"](i)) ||
                 !std::isfinite(apt["angle"](i)))) {
                throw std::runtime_error(
                    "typed Calib cannot derive beam groups from missing or nonfinite unflagged detector values");
            }
        }
    }

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
        const auto typed = apt_detector_relation_ != nullptr;
        const auto network = typed
            ? static_cast<Eigen::Index>(
                  apt_detector_relation_->binding_for_column(
                      static_cast<std::size_t>(i)).network)
            : read_index("nw", i);
        const auto array = typed
            ? canonical_array_for_network(network)
            : read_index("array", i);
        if (typed && read_index("array", i) != array) {
            throw std::runtime_error(
                "legacy numeric APT array disagrees with the typed network relation");
        }
        nw_detector_indices[network].push_back(i);
        array_detector_indices[array].push_back(i);
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
        if (apt_detector_relation_ != nullptr &&
            (!std::isfinite(std::get<0>(nw_fwhms[key])) ||
             !std::isfinite(std::get<1>(nw_fwhms[key])) ||
             !std::isfinite(nw_beam_areas[key]))) {
            throw std::runtime_error(
                "typed Calib network beam summary is nonfinite");
        }
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
        if (apt_detector_relation_ != nullptr &&
            (!std::isfinite(std::get<0>(array_fwhms[key])) ||
             !std::isfinite(std::get<1>(array_fwhms[key])) ||
             !std::isfinite(array_pas[key]) ||
             !std::isfinite(array_beam_areas[key]))) {
            throw std::runtime_error(
                "typed Calib array beam summary is nonfinite");
        }
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
