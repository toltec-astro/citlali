#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/utils.h>
#include <citlali/core/engine/calib.h>
#include <citlali/core/error/error.h>
#include <citlali/core/pipeline/canonical_apt_bundle_v2.h>
#include <citlali/core/utils/sha256.h>
#include <citlali/core/utils/toltec_io.h>

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <limits>
#include <set>
#include <stdexcept>

namespace engine {

void Calib::get_apt(
    const std::string &filepath, std::vector<std::string> &raw_filenames,
    std::vector<std::string> &interfaces,
    citlali::pipeline::AptDetectorRelationRetention retention) {
    Calib candidate = *this;
    candidate.apt_meta = YAML::Clone(apt_meta);
    candidate.apt_detector_relation_v2_.reset();
    candidate.load_apt_in_place(
        filepath, raw_filenames, interfaces, retention);
    commit_apt_state(std::move(candidate));
}

void Calib::load_apt_in_place(
    const std::string &filepath, std::vector<std::string> &raw_filenames,
    std::vector<std::string> &interfaces,
    citlali::pipeline::AptDetectorRelationRetention retention) {
    namespace apt2 = citlali::pipeline::canonical_apt_v2;
    try {
        if (raw_filenames.size() != interfaces.size() ||
            raw_filenames.empty()) {
            throw apt2::ContractError(
                "canonical APT v2 admission requires one raw file per interface");
        }
        const auto manifest_path = std::filesystem::absolute(filepath);
        auto verified = apt2::verify_bundle_filesystem(manifest_path, true);
        if (verified.manifest.kind != apt2::BundleKind::matched) {
            throw apt2::ContractError(
                "ordinary APT consumer admission requires a fresh matched v2 bundle");
        }

        std::map<std::int64_t, const apt2::SourceRecord *> raw_sources;
        for (const auto &source : verified.sources) {
            if (source.role == apt2::SourceRole::raw) {
                raw_sources.emplace(source.network, &source);
            }
        }
        std::set<std::int64_t> admitted_networks;
        for (std::size_t index = 0; index < raw_filenames.size(); ++index) {
            const auto &interface_name = interfaces[index];
            if (!interface_name.starts_with("toltec") ||
                interface_name.size() <= 6) {
                throw apt2::ContractError(
                    "raw interface is not exact toltecN");
            }
            std::int64_t network = -1;
            const std::string_view digits{interface_name.data() + 6,
                                          interface_name.size() - 6};
            const auto [end, parse_error] = std::from_chars(
                digits.data(), digits.data() + digits.size(), network);
            if (parse_error != std::errc{} ||
                end != digits.data() + digits.size() || network < 0 ||
                network > 12 ||
                interface_name != "toltec" + std::to_string(network)) {
                throw apt2::ContractError(
                    "raw interface is not exact toltec0..toltec12");
            }
            const auto source = raw_sources.find(network);
            std::error_code size_error;
            const auto byte_count = std::filesystem::file_size(
                raw_filenames[index], size_error);
            if (source == raw_sources.end() ||
                source->second->interface_name != interface_name ||
                size_error || byte_count != source->second->byte_count ||
                "sha256:" + citlali::utils::sha256_file(raw_filenames[index]) !=
                    source->second->content_sha256 ||
                !admitted_networks.insert(network).second) {
                throw apt2::ContractError(
                    "raw observation bytes/interface do not match the verified target manifest");
            }
        }
        if (admitted_networks.size() != raw_sources.size()) {
            throw apt2::ContractError(
                "raw observation does not exactly cover the verified target networks");
        }

        std::shared_ptr<
            const citlali::pipeline::CanonicalAptDetectorRelationV2>
            typed_relation;
        if (retention ==
            citlali::pipeline::AptDetectorRelationRetention::retain) {
            typed_relation = std::make_shared<const
                citlali::pipeline::CanonicalAptDetectorRelationV2>(
                citlali::pipeline::
                    admit_canonical_apt_detector_relation_v2(verified));
        }

        std::vector<const apt2::AptRow *> rows;
        rows.reserve(verified.apt.rows.size());
        for (const auto &row : verified.apt.rows) rows.push_back(&row);
        std::sort(rows.begin(), rows.end(), [](const auto *lhs,
                                              const auto *rhs) {
            return lhs->presentation_rank < rhs->presentation_rank;
        });

        auto rules = verified.fields;
        std::sort(rules.begin(), rules.end(), [](const auto &lhs,
                                                const auto &rhs) {
            return lhs.field_uid < rhs.field_uid;
        });
        apt.clear();
        apt_header_keys.clear();
        apt_header_units.clear();
        apt_header_description.clear();
        for (const auto &rule : rules) {
            apt_header_keys.push_back(rule.name);
            apt_header_units[rule.name] = rule.unit;
            apt_header_description[rule.name] = rule.description;
            apt[rule.name].resize(static_cast<Eigen::Index>(rows.size()));
        }
        const auto exact_double = [](std::int64_t value,
                                     std::string_view field) {
            constexpr std::int64_t exact_integer_limit =
                INT64_C(9007199254740992);
            if (value < -exact_integer_limit ||
                value > exact_integer_limit) {
                throw apt2::ContractError(
                    "canonical int64 is not exactly representable by the legacy Calib adapter: " +
                    std::string(field));
            }
            const auto result = static_cast<double>(value);
            return result;
        };
        const auto stored_value = [&](const apt2::Value &value,
                                      const apt2::FieldRule &rule) {
            if (std::holds_alternative<apt2::NullValue>(value)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            if (const auto integer = std::get_if<std::int64_t>(&value)) {
                return exact_double(*integer, rule.name);
            }
            if (const auto number = std::get_if<double>(&value)) {
                return *number;
            }
            throw apt2::ContractError(
                "canonical APT field cannot enter the numeric Calib adapter: " +
                rule.name);
        };
        for (Eigen::Index index = 0;
             index < static_cast<Eigen::Index>(rows.size()); ++index) {
            const auto &row = *rows[static_cast<std::size_t>(index)];
            if (!admitted_networks.contains(row.network)) {
                throw apt2::ContractError(
                    "verified APT row is outside the admitted raw networks");
            }
            apt["uid"](index) = exact_double(row.uid, "uid");
            apt["tone_freq"](index) = row.tone_frequency_hz;
            apt["array"](index) = exact_double(row.array, "array");
            apt["nw"](index) = exact_double(row.network, "nw");
            apt["kids_tone"](index) = exact_double(row.channel, "kids_tone");
            for (const auto &rule : rules) {
                if (rule.name == "uid" || rule.name == "tone_freq" ||
                    rule.name == "array" || rule.name == "nw" ||
                    rule.name == "kids_tone") {
                    continue;
                }
                apt[rule.name](index) =
                    stored_value(row.fields.at(rule.name), rule);
            }
        }
        apt_filepath = manifest_path.string();
        apt_meta["Radesys"] = "altaz";
        apt_meta["canonical_apt_schema"] = verified.identity.schema;
        apt_meta["canonical_apt_occurrence"] = verified.identity.occurrence;
        apt_meta["canonical_apt_semantic_sha256"] =
            verified.identity.semantic_sha256;
        apt_meta["canonical_apt_envelope_sha256"] =
            verified.identity.envelope_sha256;
        apt_detector_relation_v2_ = std::move(typed_relation);
        flux_conversion_factor.resize(0);
        mean_flux_conversion_factor.clear();
        setup();
    } catch (const apt2::ContractError &error) {
        throw citlali::error::io(
            "canonical APT v2 admission rejected: " +
            std::string(error.what()));
    }
}

bool Calib::has_apt_detector_relation_v2() const noexcept {
    return apt_detector_relation_v2_ != nullptr;
}

std::shared_ptr<
    const citlali::pipeline::CanonicalAptDetectorRelationV2>
Calib::apt_detector_relation_v2_handle() const noexcept {
    return apt_detector_relation_v2_;
}

const citlali::pipeline::CanonicalAptDetectorRelationV2 &
Calib::require_apt_detector_relation_v2() const {
    if (apt_detector_relation_v2_ == nullptr) {
        throw std::logic_error(
            "Calib has no admitted compact-v2 detector relation");
    }
    return *apt_detector_relation_v2_;
}

void Calib::commit_apt_state(Calib &&candidate) noexcept {
    apt_filepath.swap(candidate.apt_filepath);
    apt.swap(candidate.apt);
    apt_header_keys.swap(candidate.apt_header_keys);
    apt_header_units.swap(candidate.apt_header_units);
    apt_header_description.swap(candidate.apt_header_description);
    std::swap(apt_meta, candidate.apt_meta);
    apt_detector_relation_v2_.swap(candidate.apt_detector_relation_v2_);
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
