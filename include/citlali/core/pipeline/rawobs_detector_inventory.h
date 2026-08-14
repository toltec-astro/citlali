#pragma once

#include <citlali/core/pipeline/canonical_apt_v1.h>
#include <citlali/core/utils/netcdf_io.h>

#include <fmt/core.h>
#include <netcdf>
#include <tula/eigen.h>

#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace citlali::pipeline {

namespace canonical_apt = canonical_apt_v1;

struct RawObsInputInventory {
    canonical_apt::RawInput manifest;
    std::int64_t array = 0;
    std::vector<double> tone_frequencies_hz;
};

struct RawObsDetectorInventory {
    canonical_apt::ObservationIdentity observation;
    std::vector<RawObsInputInventory> inputs;

    // Retained for the existing Calib setup adapter. They are required to be
    // exactly redundant with `inputs`, never a second authority.
    Eigen::Index n_dets = 0;
    std::vector<Eigen::Index> dets;
    std::vector<Eigen::Index> nws;
    std::vector<Eigen::Index> arrays;
};

struct RawObsKidsIdentity {
    std::int64_t network = 0;
    std::int64_t observation = 0;
};

inline std::int64_t rawobs_array_id(std::int64_t network) {
    if (network >= 0 && network <= 6) {
        return 0;
    }
    if (network >= 7 && network <= 10) {
        return 1;
    }
    if (network >= 11 && network <= 12) {
        return 2;
    }
    throw std::runtime_error(
        "canonical Beammap raw network is outside TolTEC range 0..12");
}

inline Eigen::Index rawobs_interface_id(std::string_view interface_name) {
    constexpr std::string_view prefix{"toltec"};
    if (!interface_name.starts_with(prefix) ||
        interface_name.size() == prefix.size()) {
        throw std::runtime_error(
            "raw KIDs interface must be exact canonical toltecN");
    }
    std::int64_t value = -1;
    const auto digits = interface_name.substr(prefix.size());
    const auto [end, error] = std::from_chars(
        digits.data(), digits.data() + digits.size(), value);
    if (error != std::errc{} || end != digits.data() + digits.size() ||
        value < 0 || value > 12 ||
        interface_name != prefix.data() + std::to_string(value)) {
        throw std::runtime_error(
            "raw KIDs interface must be exact canonical toltec0..toltec12");
    }
    return static_cast<Eigen::Index>(value);
}

inline void validate_retained_raw_manifest(
    const canonical_apt::RawManifest &raw_manifest) {
    if (raw_manifest.observation.observation < 0 ||
        raw_manifest.observation.subobservation < 0 ||
        raw_manifest.observation.scan < 0 || raw_manifest.inputs.empty()) {
        throw std::runtime_error(
            "retained raw manifest has an invalid observation tuple or no inputs");
    }
    std::set<std::int64_t> networks;
    std::set<std::string> interfaces;
    for (const auto &input : raw_manifest.inputs) {
        if (input.channel_count <= 0 || input.network < 0 ||
            input.network > 12 ||
            rawobs_interface_id(input.interface_name) != input.network ||
            !networks.insert(input.network).second ||
            !interfaces.insert(input.interface_name).second) {
            throw std::runtime_error(
                "retained raw manifest input identity/count is invalid or duplicate");
        }
    }
}

inline void validate_rawobs_detector_inventory(
    const RawObsDetectorInventory &inventory) {
    const auto &observation = inventory.observation;
    if (observation.observation < 0 || observation.subobservation < 0 ||
        observation.scan < 0 || inventory.inputs.empty()) {
        throw std::runtime_error(
            "raw KIDs inventory requires a nonnegative observation tuple and at least one input");
    }
    if (inventory.dets.size() != inventory.inputs.size() ||
        inventory.nws.size() != inventory.inputs.size() ||
        inventory.arrays.size() != inventory.inputs.size()) {
        throw std::runtime_error(
            "raw KIDs legacy inventory vectors disagree with authoritative inputs");
    }

    std::set<std::int64_t> networks;
    std::set<std::string> interfaces;
    std::uint64_t detector_count = 0;
    for (std::size_t index = 0; index < inventory.inputs.size(); ++index) {
        const auto &input = inventory.inputs[index];
        const auto parsed_network = static_cast<std::int64_t>(
            rawobs_interface_id(input.manifest.interface_name));
        if (input.manifest.network != parsed_network ||
            input.manifest.channel_count <= 0 ||
            input.array != rawobs_array_id(input.manifest.network) ||
            !networks.insert(input.manifest.network).second ||
            !interfaces.insert(input.manifest.interface_name).second) {
            throw std::runtime_error(
                "raw KIDs inventory requires one canonical input per unique network/interface");
        }
        if (static_cast<std::uint64_t>(input.manifest.channel_count) !=
                input.tone_frequencies_hz.size() ||
            inventory.dets[index] != input.manifest.channel_count ||
            inventory.nws[index] != input.manifest.network ||
            inventory.arrays[index] != input.array) {
            throw std::runtime_error(
                "raw KIDs detector/ToneFreq/manifest cardinality mismatch");
        }
        for (const auto tone : input.tone_frequencies_hz) {
            if (!std::isfinite(tone)) {
                throw std::runtime_error(
                    "raw KIDs first-sweep ToneFreq values must be finite");
            }
        }
        const auto count =
            static_cast<std::uint64_t>(input.manifest.channel_count);
        const auto local_capacity = static_cast<std::uint64_t>(
            std::numeric_limits<Eigen::Index>::max());
        const auto canonical_capacity =
            static_cast<std::uint64_t>(canonical_apt::uid_v1_max) + 1U;
        if (count > local_capacity ||
            detector_count > local_capacity - count ||
            count > canonical_capacity ||
            detector_count > canonical_capacity - count) {
            throw std::runtime_error(
                "raw KIDs detector inventory exceeds local or canonical v1 UID capacity");
        }
        detector_count += count;
    }
    if (inventory.n_dets < 0 ||
        detector_count != static_cast<std::uint64_t>(inventory.n_dets)) {
        throw std::runtime_error(
            "raw KIDs total detector count disagrees with input manifest");
    }
}

inline canonical_apt::ObservationIdentity validate_rawobs_observation_identity(
    const canonical_apt::RawManifest &raw_manifest,
    const std::vector<RawObsKidsIdentity> &kids_identities,
    const canonical_apt::ObservationIdentity &telescope_observation,
    std::int64_t output_observation) {
    validate_retained_raw_manifest(raw_manifest);
    if (kids_identities.size() != raw_manifest.inputs.size()) {
        throw std::runtime_error(
            "KIDs metadata count disagrees with raw manifest input count");
    }
    for (std::size_t index = 0; index < raw_manifest.inputs.size(); ++index) {
        if (kids_identities[index].network !=
                raw_manifest.inputs[index].network ||
            kids_identities[index].observation !=
                raw_manifest.observation.observation) {
            throw std::runtime_error(
                "KIDs metadata network or obsid disagrees with raw inventory");
        }
    }
    if (!(telescope_observation == raw_manifest.observation) ||
        output_observation != raw_manifest.observation.observation) {
        throw std::runtime_error(
            "KIDs, telescope, and output observation identities disagree");
    }
    return raw_manifest.observation;
}

inline canonical_apt::ObservationIdentity validate_rawobs_observation_identity(
    const canonical_apt::RawManifest &raw_manifest,
    const canonical_apt::ObservationIdentity &telescope_observation,
    std::int64_t output_observation) {
    validate_retained_raw_manifest(raw_manifest);
    if (!(telescope_observation == raw_manifest.observation) ||
        output_observation != raw_manifest.observation.observation) {
        throw std::runtime_error(
            "raw, telescope, and output observation identities disagree");
    }
    return raw_manifest.observation;
}

inline canonical_apt::ObservationIdentity validate_rawobs_observation_identity(
    const RawObsDetectorInventory &inventory,
    const std::vector<RawObsKidsIdentity> &kids_identities,
    const canonical_apt::ObservationIdentity &telescope_observation,
    std::int64_t output_observation) {
    validate_rawobs_detector_inventory(inventory);
    return validate_rawobs_observation_identity(
        canonical_apt::RawManifest{inventory.observation, [&] {
            std::vector<canonical_apt::RawInput> inputs;
            inputs.reserve(inventory.inputs.size());
            for (const auto &input : inventory.inputs) {
                inputs.push_back(input.manifest);
            }
            return inputs;
        }()},
        kids_identities, telescope_observation, output_observation);
}

inline Eigen::Index detector_count_from_rawobs_file(netCDF::NcFile &file) {
    const auto variable = file.getVar("Data.Toltec.Is");
    if (variable.isNull() || variable.getDimCount() != 2) {
        throw std::runtime_error(
            "raw KIDs Data.Toltec.Is must be an exact two-dimensional time/detector array");
    }
    const auto count = variable.getDim(1).getSize();
    if (count == 0 ||
        count > static_cast<std::size_t>(
                    std::numeric_limits<Eigen::Index>::max())) {
        throw std::runtime_error(
            "raw KIDs detector dimension must be positive and representable");
    }
    return static_cast<Eigen::Index>(count);
}

inline std::int64_t rawobs_exact_integer(netCDF::NcFile &file,
                                         const std::string &name) {
    const auto variable = file.getVar(name);
    if (variable.isNull() || variable.getDimCount() != 0) {
        throw std::runtime_error(
            "raw KIDs integer header must be an exact scalar: " + name);
    }
    std::int64_t value = -1;
    const auto type = variable.getType().getTypeClass();
    switch (type) {
    case netCDF::NcType::nc_BYTE: {
        signed char source = -1;
        variable.getVar(&source);
        value = source;
        break;
    }
    case netCDF::NcType::nc_SHORT: {
        short source = -1;
        variable.getVar(&source);
        value = source;
        break;
    }
    case netCDF::NcType::nc_INT: {
        int source = -1;
        variable.getVar(&source);
        value = source;
        break;
    }
    case netCDF::NcType::nc_UBYTE: {
        unsigned char source = 0;
        variable.getVar(&source);
        value = source;
        break;
    }
    case netCDF::NcType::nc_USHORT: {
        unsigned short source = 0;
        variable.getVar(&source);
        value = source;
        break;
    }
    case netCDF::NcType::nc_UINT: {
        unsigned int source = 0;
        variable.getVar(&source);
        value = source;
        break;
    }
    case netCDF::NcType::nc_INT64: {
        long long source = -1;
        variable.getVar(&source);
        if (source < 0 ||
            static_cast<unsigned long long>(source) >
                static_cast<unsigned long long>(
                    std::numeric_limits<std::int64_t>::max())) {
            throw std::runtime_error(
                "raw KIDs integer header is outside canonical int64 range: " +
                name);
        }
        value = static_cast<std::int64_t>(source);
        break;
    }
    case netCDF::NcType::nc_UINT64: {
        unsigned long long source = 0;
        variable.getVar(&source);
        if (source > static_cast<unsigned long long>(
                         std::numeric_limits<std::int64_t>::max())) {
            throw std::runtime_error(
                "raw KIDs integer header is outside canonical int64 range: " +
                name);
        }
        value = static_cast<std::int64_t>(source);
        break;
    }
    default:
        throw std::runtime_error(
            "raw KIDs integer header has a non-integral storage type: " +
            name);
    }
    if (value < 0) {
        throw std::runtime_error("raw KIDs integer header is negative: " +
                                 name);
    }
    return value;
}

inline double rawobs_finite_scalar(netCDF::NcFile &file,
                                   const std::string &name) {
    const auto variable = file.getVar(name);
    if (variable.isNull() || variable.getDimCount() != 0) {
        throw std::runtime_error(
            "raw KIDs floating header must be an exact scalar: " + name);
    }
    const auto type = variable.getType().getTypeClass();
    if (type != netCDF::NcType::nc_FLOAT &&
        type != netCDF::NcType::nc_DOUBLE) {
        throw std::runtime_error(
            "raw KIDs floating header has an unsupported storage type: " +
            name);
    }
    double value = 0.0;
    variable.getVar(&value);
    if (!std::isfinite(value)) {
        throw std::runtime_error("raw KIDs floating header must be finite: " +
                                 name);
    }
    return value;
}

inline canonical_apt::ObservationIdentity rawobs_observation_identity(
    netCDF::NcFile &file) {
    return {
        rawobs_exact_integer(file, "Header.Toltec.ObsNum"),
        rawobs_exact_integer(file, "Header.Toltec.SubObsNum"),
        rawobs_exact_integer(file, "Header.Toltec.ScanNum"),
    };
}

inline std::vector<double> rawobs_first_sweep_tone_frequencies(
    netCDF::NcFile &file, Eigen::Index expected_tones) {
    const auto variable = file.getVar("Header.Toltec.ToneFreq");
    if (variable.isNull() || variable.getDimCount() != 2) {
        throw std::runtime_error(
            "raw KIDs Header.Toltec.ToneFreq must be a two-dimensional sweep/tone array");
    }
    const auto n_sweeps = variable.getDim(0).getSize();
    const auto n_tones = variable.getDim(1).getSize();
    const auto tone_type = variable.getType().getTypeClass();
    if (tone_type != netCDF::NcType::nc_FLOAT &&
        tone_type != netCDF::NcType::nc_DOUBLE) {
        throw std::runtime_error(
            "raw KIDs ToneFreq must have float or double storage");
    }
    if (n_sweeps == 0 ||
        n_tones != static_cast<std::size_t>(expected_tones)) {
        throw std::runtime_error(
            "raw KIDs ToneFreq sweep/tone cardinality disagrees with detectors");
    }
    const auto lo_frequency_hz = rawobs_finite_scalar(
        file, "Header.Toltec.LoCenterFreq");

    std::vector<double> result(static_cast<std::size_t>(expected_tones));
    const std::vector<std::size_t> start{0, 0};
    const std::vector<std::size_t> count{
        1, static_cast<std::size_t>(expected_tones)};
    variable.getVar(start, count, result.data());
    for (Eigen::Index channel = 0; channel < expected_tones; ++channel) {
        const double tone =
            result[static_cast<std::size_t>(channel)] + lo_frequency_hz;
        if (!std::isfinite(tone)) {
            throw std::runtime_error(
                "raw KIDs first-sweep absolute ToneFreq must be finite");
        }
        result[static_cast<std::size_t>(channel)] = tone;
    }
    return result;
}

template <class RawObs, class Logger>
Eigen::Index read_rawobs_detector_count(const RawObs &rawobs,
                                        const Logger &logger) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    Eigen::Index n_dets = 0;
    for (const typename RawObs::DataItem &data_item : rawobs.kidsdata()) {
        try {
            NcFile file(data_item.filepath(), NcFile::read);
            const auto count = detector_count_from_rawobs_file(file);
            if (n_dets > std::numeric_limits<Eigen::Index>::max() - count) {
                throw std::runtime_error(
                    "raw KIDs detector count exceeds local index range");
            }
            n_dets += count;
            file.close();
        }
        catch (NcException &error) {
            logger->error("{}", error.what());
            throw ::DataIOError{fmt::format(
                "failed to load data from netCDF file {}",
                data_item.filepath())};
        }
    }
    return n_dets;
}

template <class RawObs, class NetworkToArrayMap, class Logger>
RawObsDetectorInventory read_rawobs_detector_inventory(
    const RawObs &rawobs, NetworkToArrayMap &nw_to_array_map,
    const Logger &logger) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    RawObsDetectorInventory inventory;
    bool have_observation = false;
    std::set<std::int64_t> networks;
    std::set<std::string> interfaces;
    for (const typename RawObs::DataItem &data_item : rawobs.kidsdata()) {
        const auto interface_id =
            static_cast<std::int64_t>(rawobs_interface_id(
                data_item.interface()));
        if (!networks.insert(interface_id).second ||
            !interfaces.insert(data_item.interface()).second) {
            throw std::runtime_error(
                "duplicate/split raw KIDs network input is unsupported by canonical APT v1");
        }
        try {
            NcFile file(data_item.filepath(), NcFile::read);
            const auto file_network = rawobs_exact_integer(
                file, "Header.Toltec.RoachIndex");
            if (file_network != interface_id) {
                throw std::runtime_error(
                    "raw KIDs configured interface disagrees with Header.Toltec.RoachIndex");
            }
            const auto observation = rawobs_observation_identity(file);
            if (!have_observation) {
                inventory.observation = observation;
                have_observation = true;
            } else if (!(inventory.observation == observation)) {
                throw std::runtime_error(
                    "raw KIDs inputs disagree on observation/subobservation/scan");
            }
            const auto n_file_dets = detector_count_from_rawobs_file(file);
            const auto map_entry = nw_to_array_map.find(
                static_cast<Eigen::Index>(interface_id));
            const auto expected_array = rawobs_array_id(interface_id);
            if (map_entry == nw_to_array_map.end() ||
                static_cast<std::int64_t>(map_entry->second) !=
                    expected_array) {
                throw std::runtime_error(
                    "raw KIDs network is absent from or disagrees with the canonical array map");
            }
            auto tones = rawobs_first_sweep_tone_frequencies(
                file, n_file_dets);
            if (inventory.n_dets >
                std::numeric_limits<Eigen::Index>::max() - n_file_dets) {
                throw std::runtime_error(
                    "raw KIDs detector count exceeds local index range");
            }
            inventory.n_dets += n_file_dets;
            inventory.dets.push_back(n_file_dets);
            inventory.nws.push_back(static_cast<Eigen::Index>(interface_id));
            inventory.arrays.push_back(
                static_cast<Eigen::Index>(expected_array));
            inventory.inputs.push_back({
                {interface_id, data_item.interface(),
                 static_cast<std::int64_t>(n_file_dets)},
                expected_array,
                std::move(tones),
            });
            file.close();
        }
        catch (NcException &error) {
            logger->error("{}", error.what());
            throw ::DataIOError{fmt::format(
                "failed to load data from netCDF file {}",
                data_item.filepath())};
        }
    }
    validate_rawobs_detector_inventory(inventory);
    return inventory;
}

template <class Calib>
void populate_internal_apt_from_detector_inventory(
    Calib &calib, const RawObsDetectorInventory &inventory) {
    validate_rawobs_detector_inventory(inventory);

    // Calib is reused across observations.  Beammap appends its per-observation
    // columns after this population step, so retain exactly the legacy
    // baseline input surface here and let the next Beammap setup rebuild its
    // derived columns once.  This prevents a prior observation's KIDs or
    // Beammap extension names from becoming authority for the next artifact.
    std::set<std::string> baseline_names{"uid", "tone_freq", "array", "nw"};
    for (const auto &field :
         canonical_apt_v1::canonical_field_registry_v1()
             .required_baseline_fields) {
        baseline_names.insert(field.name);
    }
    std::vector<std::string> baseline_headers;
    baseline_headers.reserve(baseline_names.size());
    std::set<std::string> seen_baseline;
    for (const auto &key : calib.apt_header_keys) {
        if (baseline_names.contains(key)) {
            if (!seen_baseline.insert(key).second) {
                throw std::runtime_error(
                    "Beammap baseline APT header contains a duplicate field");
            }
            baseline_headers.push_back(key);
        }
    }
    if (seen_baseline != baseline_names) {
        throw std::runtime_error(
            "Beammap baseline APT header is missing a canonical v1 field");
    }
    calib.apt_header_keys = std::move(baseline_headers);
    calib.apt.clear();

    for (const auto &key : calib.apt_header_keys) {
        if (key == "x_t" || key == "y_t") {
            calib.apt[key].setZero(inventory.n_dets);
        } else {
            calib.apt[key].setOnes(inventory.n_dets);
        }
    }
    calib.apt["flag"].setZero(inventory.n_dets);

    Eigen::Index row_offset = 0;
    for (const auto &input : inventory.inputs) {
        const auto count =
            static_cast<Eigen::Index>(input.manifest.channel_count);
        calib.apt["nw"].segment(row_offset, count)
            .setConstant(input.manifest.network);
        calib.apt["array"].segment(row_offset, count)
            .setConstant(input.array);
        for (Eigen::Index channel = 0; channel < count; ++channel) {
            calib.apt["tone_freq"](row_offset + channel) =
                input.tone_frequencies_hz[static_cast<std::size_t>(channel)];
        }
        row_offset += count;
    }

    calib.apt["uid"] = Eigen::VectorXd::LinSpaced(
        inventory.n_dets, 0, inventory.n_dets - 1);
    calib.setup();
    calib.apt_filepath = "internally generated for beammap";

    auto producer = calib.canonical_apt_producer;
    producer.raw_inventory_ready = false;
    producer.raw_manifest.observation = inventory.observation;
    producer.raw_manifest.inputs.clear();
    producer.rows.clear();
    producer.rows.reserve(static_cast<std::size_t>(inventory.n_dets));
    Eigen::Index uid = 0;
    for (const auto &input : inventory.inputs) {
        producer.raw_manifest.inputs.push_back(input.manifest);
        for (std::int64_t channel = 0;
             channel < input.manifest.channel_count; ++channel, ++uid) {
            producer.rows.push_back({
                static_cast<std::int64_t>(uid),
                input.manifest.network,
                channel,
                input.array,
                input.tone_frequencies_hz[static_cast<std::size_t>(channel)],
            });
        }
    }
    producer.raw_inventory_ready = true;
    calib.canonical_apt_producer = std::move(producer);
}

}  // namespace citlali::pipeline
