#include "successor_input_bindings.h"

#include <citlali/core/utils/sha256.h>
#include <kidscpp_config/gitversion.h>
#include <tula/datatable.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace citlali::cli::input_bindings {
[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

namespace {
std::int64_t parse_integer(const std::string &value,
                           const std::string &option) {
    std::size_t consumed = 0;
    std::int64_t result = 0;
    try {
        result = std::stoll(value, &consumed);
    } catch (const std::exception &) {
        fail(option + " requires an integer");
    }
    require(consumed == value.size(), option + " requires an integer");
    return result;
}

std::string read_netcdf_text(netCDF::NcFile &file,
                             const std::string &name) {
    const auto variable = file.getVar(name);
    require(!variable.isNull() && variable.getDimCount() == 1,
            "missing one-dimensional NetCDF text " + name);
    std::vector<char> buffer(variable.getDim(0).getSize());
    variable.getVar(buffer.data());
    std::string result(buffer.begin(), buffer.end());
    const auto nul = result.find('\0');
    if (nul != std::string::npos) result.erase(nul);
    while (!result.empty() &&
           (result.back() == ' ' || result.back() == '\t' ||
            result.back() == '\r' || result.back() == '\n')) {
        result.pop_back();
    }
    return result;
}

std::string read_units(const netCDF::NcVar &variable,
                       const std::string &name) {
    const auto attribute = variable.getAtt("units");
    require(!attribute.isNull(), name + " lacks a units attribute");
    std::string result;
    attribute.getValues(result);
    return result;
}

Eigen::VectorXd read_vector(netCDF::NcFile &file,
                            const std::string &name,
                            std::size_t expected_count,
                            std::string_view expected_units) {
    const auto variable = file.getVar(name);
    require(!variable.isNull() && variable.getDimCount() == 1 &&
                variable.getDim(0).getSize() == expected_count,
            name + " has an unexpected shape");
    require(read_units(variable, name) == expected_units,
            name + " has unexpected units");
    Eigen::VectorXd result(static_cast<Eigen::Index>(expected_count));
    variable.getVar(result.data());
    return result;
}

YAML::Node metadata_value(const YAML::Node &metadata,
                          const std::string &key) {
    if (metadata.IsMap()) return metadata[key];
    if (metadata.IsSequence()) {
        for (const auto &entry : metadata) {
            if (entry.IsMap() && entry.size() == 1 &&
                entry.begin()->first.as<std::string>() == key) {
                return entry.begin()->second;
            }
        }
    }
    return {};
}

} // namespace

TelescopeInput load_telescope(
    const fs::path &path,
    const pipeline::NativeObservationScope &scope,
    const TelescopeBindingProfile &profile) {
    const auto &[bounded_repeat_census, bounded_point_map,
                 configured_digest, bounded_oof] = profile;
    const bool configured_point = !configured_digest.empty();
    require(!configured_point || (bounded_point_map && !bounded_repeat_census),"configured telescope requires POINT route");
    // One explicitly selected repeat for the offline common-mode census.
    // Existing acceptance callers retain every original artifact guard.
    require(!bounded_repeat_census ||
                (scope.observation == 152392 && scope.subobservation == 0 && scope.scan == 2),
            "repeat telescope binding is outside the fixed census scope");
    require(!bounded_point_map || (!bounded_repeat_census && (configured_point || scope.observation == 152389 || scope.observation == 152430) &&
        scope.subobservation == 0 && (bounded_oof ? (configured_point && scope.observation>=152385 && scope.observation<=152387 && scope.scan==1) : scope.scan==2)), "configured telescope binding outside selected scope");
    const bool independent_point = bounded_point_map && scope.observation == 152430;
    const std::string_view expected_filename = independent_point
        ? "tel_toltec_2026-02-19_152430_00_0002.nc" : bounded_point_map
        ? "tel_toltec_2026-02-19_152389_00_0002.nc" : bounded_repeat_census
        ? "tel_toltec_2026-02-19_152392_00_0002.nc" : telescope_filename;
    const std::string_view expected_digest = independent_point
        ? "f6ac27248692af558a16af94f267a84b344e93f30d2248516d4388b9b25e5f19" : bounded_point_map
        ? "2aec5f1f2be8c365127fcf275a02eb0c780eeda9494cd225e50e50c4386a9392" : bounded_repeat_census
        ? "cf991624411d7f7582977351a5ab4cc228977bc8680578cb74e0ec1ae71d8e47" : telescope_sha256;
    const auto expected_bytes = independent_point ? 1320968u : bounded_point_map ? 1301180u : bounded_repeat_census ? 24247112u : telescope_byte_count;
    auto expected_records = independent_point ? 3251u : bounded_point_map ? 3200u : bounded_repeat_census ? 62339u : telescope_record_count;
    require(configured_point || path.filename() == expected_filename,
            "telescope filename is not the approved artifact");
    std::error_code error;
    const auto byte_count = fs::file_size(path, error);
    require(!error && (configured_point || byte_count == expected_bytes),
            "telescope byte count is not approved");
    const auto digest = citlali::utils::sha256_file(path);
    require(digest == (configured_point ? configured_digest : expected_digest),
            "telescope SHA-256 is not approved");

    netCDF::NcFile file(path.string(), netCDF::NcFile::read);
    require(read_netcdf_scalar<int>(file, "Header.Dcs.ObsNum") ==
                    scope.observation &&
                read_netcdf_scalar<int>(file, "Header.Dcs.SubObsNum") ==
                    scope.subobservation &&
                read_netcdf_scalar<int>(file, "Header.Dcs.ScanNum") ==
                    scope.scan,
            "telescope producer scope is not the admitted observation");

    pipeline::AstScanMotionSourceMetadata metadata;
    metadata.producer_kind =
        pipeline::AstScanMotionProducerKind::real_toltec;
    metadata.dcs_observation_goal =
        read_netcdf_text(file, "Header.Dcs.ObsGoal");
    metadata.dcs_observation_program =
        read_netcdf_text(file, "Header.Dcs.ObsPgm");
    metadata.scan_file_valid =
        read_netcdf_scalar<int>(file, "Header.ScanFile.Valid");
    metadata.source_epoch =
        read_netcdf_scalar<double>(file, "Header.Source.Epoch");
    metadata.source_coordinate_system =
        read_netcdf_scalar<int>(file, "Header.Source.CoordSys");
    metadata.nominal_producer_cadence_hz = 50.0;
    metadata.field_registry = pipeline::AstScanMotionFieldRegistry::
        source_ra_act_source_dec_act_j2000_radians;
    metadata.source_artifact_identity = "sha256:" + digest;

    const auto time_variable =
        file.getVar(std::string{pipeline::ast_scan_motion_time_field});
    if(configured_point) {
        require(bounded_oof ? metadata.dcs_observation_goal=="Oof" && metadata.dcs_observation_program=="Lissajous" : metadata.dcs_observation_goal=="Pointing" || metadata.dcs_observation_program=="Pointing", "configured reference acquisition mode differs");
        require(!time_variable.isNull() && time_variable.getDimCount()==1 && time_variable.getDim(0).getSize()>1,"configured telescope time cardinality");
        expected_records=time_variable.getDim(0).getSize();
    }
    require(!time_variable.isNull() &&
                time_variable.getDimCount() == 1 &&
                time_variable.getDim(0).getSize() ==
                    expected_records,
            "telescope time cardinality is not approved");
    auto times = read_vector(
        file, std::string{pipeline::ast_scan_motion_time_field},
        expected_records, "sec");
    auto ra = read_vector(
        file, std::string{pipeline::ast_scan_motion_ra_field},
        expected_records, "rad");
    auto dec = read_vector(
        file, std::string{pipeline::ast_scan_motion_dec_field},
        expected_records, "rad");
    auto source = pipeline::AstScanMotionSource::admit(
        scope, scope, 0, std::move(metadata), std::move(times),
        std::move(ra), std::move(dec));
    return {std::move(source), digest, byte_count};
}

RuntimeConfig load_runtime_config(const fs::path &path) {
    require(fs::is_regular_file(path), "config is not a regular file");
    const auto root = YAML::LoadFile(path.string());
    RuntimeConfig result;
    const auto offsets = root["interface_sync_offset"];
    require(offsets && offsets.IsSequence(),
            "config lacks interface_sync_offset sequence");
    std::set<std::string> seen;
    for (const auto &entry : offsets) {
        require(entry.IsMap() && entry.size() == 1,
                "interface_sync_offset entry is malformed");
        const auto key = entry.begin()->first.as<std::string>();
        if (!key.starts_with("toltec")) continue;
        const auto suffix = key.substr(6);
        require(!suffix.empty() &&
                    std::all_of(suffix.begin(), suffix.end(),
                                [](unsigned char ch) {
                                    return ch >= '0' && ch <= '9';
                                }),
                "interface sync key is not exact toltecN");
        const auto network = parse_integer(suffix, "interface sync key");
        require(network >= 0 && network < 13 && seen.insert(key).second,
                "interface sync key is duplicate or out of range");
        const auto value = entry.begin()->second.as<double>();
        require(std::isfinite(value),
                "interface sync offset must be finite");
        result.interface_offsets_sec[static_cast<std::size_t>(network)] = value;
        result.interface_offset_present[static_cast<std::size_t>(network)] =
            true;
    }
    const auto model = root["kids"]["fitter"]["modelspec"];
    require(model && model.IsScalar(), "config lacks kids fitter modelspec");
    result.kids_model = model.as<std::string>();
    require(result.kids_model == "gainlintrend",
            "acceptance runner requires the configured gainlintrend model");
    return result;
}

void verify_raw_file(
    const fs::path &path,
    const pipeline::CanonicalAptRawSourceBindingV2 &source) {
    std::error_code error;
    const auto size = fs::file_size(path, error);
    require(!error && size == source.byte_count,
            "raw byte count disagrees with APT source record: " +
                path.string());
    const auto digest = "sha256:" + citlali::utils::sha256_file(path);
    require(digest == source.content_sha256,
            "raw SHA-256 disagrees with APT source record: " +
                path.string());
}

TuneFacts read_tune_facts(const fs::path &path,
                          std::int64_t channel_count) {
    std::vector<std::string> header;
    YAML::Node metadata;
    const auto table = datatable::read<double, datatable::Format::ecsv>(
        path.string(), &header, &metadata);
    const auto flag = std::find(header.begin(), header.end(), "flag");
    require(flag != header.end(), "Tune fit report lacks flag column");
    require(table.rows() == channel_count &&
                table.cols() == static_cast<Eigen::Index>(header.size()),
            "Tune fit report shape disagrees with raw channel inventory");
    const auto column = static_cast<Eigen::Index>(flag - header.begin());
    TuneFacts result;
    result.valid.resize(static_cast<std::size_t>(channel_count));
    for (Eigen::Index row = 0; row < table.rows(); ++row) {
        const double value = table(row, column);
        require(std::isfinite(value) && std::floor(value) == value,
                "Tune validity flag is not a finite integer");
        result.valid[static_cast<std::size_t>(row)] = value == 0.0;
    }
    const auto required_integer = [&](const std::string &key) {
        const auto value = metadata_value(metadata, key);
        require(value && value.IsScalar(),
                "Tune fit report lacks exact metadata " + key);
        return value.as<std::int64_t>();
    };
    result.accumulation_length =
        required_integer("Header.Toltec.AccumLen");
    result.observation = required_integer("Header.Toltec.ObsNum");
    result.subobservation = required_integer("Header.Toltec.SubObsNum");
    result.scan = required_integer("Header.Toltec.ScanNum");
    result.network = required_integer("Header.Toltec.RoachIndex");
    return result;
}

Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
read_timestamp_slice(const fs::path &path, std::int64_t first_row,
                     std::int64_t row_count, double &fpga_frequency_hz,
                     double &sample_frequency_hz,
                     std::int64_t &accumulation_length) {
    netCDF::NcFile file(path.string(), netCDF::NcFile::read);
    fpga_frequency_hz =
        read_netcdf_scalar<double>(file, "Header.Toltec.FpgaFreq");
    sample_frequency_hz =
        read_netcdf_scalar<double>(file, "Header.Toltec.SampleFreq");
    accumulation_length =
        read_netcdf_scalar<int>(file, "Header.Toltec.AccumLen");
    const auto variable = file.getVar("Data.Toltec.Ts");
    require(!variable.isNull() && variable.getDimCount() == 2 &&
                variable.getDim(1).getSize() == 6,
            "Data.Toltec.Ts must have shape (time, 6)");
    require(first_row + row_count <=
                static_cast<std::int64_t>(variable.getDim(0).getSize()),
            "requested native slice exceeds raw timing support");
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        timestamps(row_count, 6);
    variable.getVar(
        {static_cast<std::size_t>(first_row), 0},
        {static_cast<std::size_t>(row_count), 6}, timestamps.data());
    return timestamps;
}

std::vector<pipeline::NativeReadoutDetectorBinding> detector_axis(
    const pipeline::CanonicalAptDetectorRelationV2 &relation,
    const NetworkInput &input) {
    std::vector<std::optional<pipeline::NativeReadoutDetectorBinding>> by_channel(
        static_cast<std::size_t>(input.source.channel_count));
    for (const auto &binding : relation.bindings()) {
        if (binding.network != input.source.network) continue;
        require(binding.raw_source_uid == input.source.source_uid &&
                    binding.channel >= 0 &&
                    binding.channel < input.source.channel_count,
                "APT detector binding disagrees with raw source");
        auto &destination = by_channel.at(
            static_cast<std::size_t>(binding.channel));
        require(!destination.has_value(),
                "APT detector binding repeats a raw channel");
        const auto &relation_id = relation.relation_identity();
        const auto &target = binding.target;
        destination = pipeline::NativeReadoutDetectorBinding{
            static_cast<pipeline::TimestreamNetworkId>(binding.network),
            static_cast<Eigen::Index>(binding.channel),
            target.artifact.schema + ":" + target.artifact.occurrence +
                ":semantic=" + target.artifact.semantic_sha256 +
                ":envelope=" + target.artifact.envelope_sha256 +
                ":row=" + std::to_string(target.local_uid),
            relation_id.schema + ":" + relation_id.occurrence +
                ":semantic=" + relation_id.semantic_sha256 +
                ":envelope=" + relation_id.envelope_sha256 +
                ":relation=" + std::to_string(binding.relation_uid),
            "sha256:" + input.source.content_sha256 + ":network=" +
                std::to_string(binding.network) + ":channel=" +
                std::to_string(binding.channel)};
    }
    std::vector<pipeline::NativeReadoutDetectorBinding> result;
    result.reserve(by_channel.size());
    for (std::size_t channel = 0; channel < by_channel.size(); ++channel) {
        require(by_channel[channel].has_value(),
                "APT detector binding does not cover every raw channel");
        require(by_channel[channel]->storage_column ==
                    static_cast<Eigen::Index>(channel),
                "APT detector channel order is not native solver order");
        result.push_back(*by_channel[channel]);
    }
    return result;
}

std::shared_ptr<const pipeline::NativeReadoutMappingAuthority> mapping_identity(
    const NetworkInput &input, const RuntimeConfig &config) {
    const auto raw_identity =
        input.source.content_sha256 + ":network=" +
        std::to_string(input.source.network);
    const auto tune_identity = "sha256:" + input.tune_sha256;
    const auto transform = std::string{"kidscpp:"} + KIDSCPP_GIT_VERSION +
        ":S21WithGainLinTrend:tune-meta-normalized-v1:raw-accumlen=" +
        std::to_string(input.accumulation_length) + ":tune-accumlen=" +
        std::to_string(input.tune_accumulation_length);
    const auto revision = transform + ":tune=" + tune_identity;
    auto result =
        std::make_shared<pipeline::NativeReadoutMappingAuthority>();
    result->network_id = input.source.network;
    result->producer_id = "kidscpp:timestream-solver:" + config.kids_model;
    result->producer_instance_id = raw_identity + ":tune=" + tune_identity;
    result->producer_interface_id = std::string{producer_interface};
    result->mapping_record_id = result->producer_instance_id;
    result->mapping_revision_id = revision;
    result->tune_id = tune_identity;
    result->readout_interface_id = input.source.interface_name;
    result->input_coordinate_record_id = raw_identity + ":native-iq";
    result->transform_id = transform;
    result->transform_representation_id =
        "kidscpp:S21WithGainLinTrend:complex-jacobian";
    result->applicability_domain_id =
        std::string{IdentityRouteExperimentProfile::mapping_domain_prefix} +
        std::to_string(input.source.network);
    result->event_time_epoch_meaning_id =
        "Header.Kids.tel_time+Header.Kids.pps_time+Header.Kids.clock_time"
        "+interface-sync-offset:unix-sec";
    result->native_time_unit_id = "second";
    result->native_cadence_record_id =
        "Header.Toltec.AccumLen=" +
        std::to_string(input.accumulation_length) +
        ":Header.Toltec.FpgaFreq=" +
        std::to_string(input.fpga_frequency_hz);
    result->native_time_validity_state_id =
        "finite-strictly-increasing-native-network-time";
    result->timing_uncertainty_state_id =
        "not-quantified:producer-native-timing";
    result->parent_readout_record_id = raw_identity + ":packet-counter";
    result->paired_xr_record_id = result->producer_instance_id + ":paired-xr";
    result->runtime_binding_rule_id =
        "verified-apt-raw-source+exact-network+exact-native-channel";
    result->compatibility_rule_id =
        "canonical-paired-d1:exact-x-r-shape-and-authority";
    result->failure_semantics_id =
        "fail-closed:missing-ambiguous-duplicate-or-mismatched-pair";
    auto coordinate = [&](std::string name) {
        return pipeline::NativeReadoutCoordinateAuthority{
            "kidscpp:" + config.kids_model + ":" + name + ":meaning",
            "dimensionless",
            "producer-defined:" + std::string{producer_interface},
            "producer-native-iq-reference",
            "kidscpp:tune-meta-normalized-v1",
            "producer-native-" + name,
            "producer-valid+finite+in-acquisition-support",
            "not-quantified:producer-native-" + name};
    };
    result->x = coordinate("x");
    result->r = coordinate("r");
    return result;
}

std::shared_ptr<const pipeline::NativePairedReadoutOccurrenceAxis> occurrence_axis(
    const NetworkInput &input, std::int64_t first_native_row,
    std::int64_t native_row_count, NativeEventTimeRole event_time_role) {
    const double duration =
        static_cast<double>(input.accumulation_length) /
        input.fpga_frequency_hz;
    std::vector<pipeline::NativePairedReadoutOccurrenceBinding> occurrences;
    occurrences.reserve(static_cast<std::size_t>(native_row_count));
    for (std::int64_t offset = 0; offset < native_row_count; ++offset) {
        const auto row = first_native_row + offset;
        const double event = input.native_timing->identity(row)
                                 .reconstructed_time_unix_sec();
        pipeline::NativeReadoutIntegrationSupport support;
        switch (event_time_role) {
            case NativeEventTimeRole::integration_start:
                support = {event, event + duration};
                break;
            case NativeEventTimeRole::integration_center:
                support = {event - duration / 2.0,
                           event + duration / 2.0};
                break;
            case NativeEventTimeRole::integration_end:
                support = {event - duration, event};
                break;
        }
        const auto occurrence_key =
            input.native_timing->packet_counter(row);
        occurrences.push_back(
            {occurrence_key, occurrence_key, support});
    }
    return std::make_shared<const pipeline::NativePairedReadoutOccurrenceAxis>(
        input.native_timing, first_native_row, std::move(occurrences));
}

std::vector<pipeline::NativeReadoutCoordinateState> member_states(
    const pipeline::NativePairedReadoutMatrix &values,
    const std::vector<bool> &tune_valid) {
    require(values.cols() == static_cast<Eigen::Index>(tune_valid.size()),
            "Tune validity does not match solver columns");
    std::vector<pipeline::NativeReadoutCoordinateState> result;
    result.reserve(static_cast<std::size_t>(values.size()));
    for (Eigen::Index row = 0; row < values.rows(); ++row) {
        for (Eigen::Index column = 0; column < values.cols(); ++column) {
            const bool finite = std::isfinite(values(row, column));
            result.push_back(pipeline::NativeReadoutCoordinateState::measured(
                true, tune_valid[static_cast<std::size_t>(column)] && finite,
                true, finite));
        }
    }
    return result;
}

TemporaryDirectory::TemporaryDirectory() {
        std::random_device random;
        for (int attempt = 0; attempt < 100; ++attempt) {
            std::ostringstream name;
            name << "citlali-timestream-successor-tune-" << std::hex
                 << (static_cast<std::uint64_t>(random()) << 32U |
                     static_cast<std::uint64_t>(random()));
            auto candidate = fs::temp_directory_path() / name.str();
            std::error_code error;
            if (fs::create_directory(candidate, error)) {
                path_ = std::move(candidate);
                return;
            }
            require(!error || error == std::errc::file_exists,
                    "unable to create temporary normalized-Tune directory");
        }
        fail("unable to choose a unique normalized-Tune directory");
    }

TemporaryDirectory::~TemporaryDirectory() {
        std::error_code error;
        fs::remove_all(path_, error);
    }

fs::path normalized_tune_report(const NetworkInput &input,
                                const fs::path &directory) {
    const auto output = directory /
        ("toltec" + std::to_string(input.source.network) + ".ecsv");
    std::ifstream source(input.tune_path);
    require(static_cast<bool>(source),
            "unable to read exact Tune fit report for normalization");
    std::ofstream destination(output, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(destination),
            "unable to create temporary normalized Tune fit report");
    std::size_t replacement_count = 0;
    std::string line;
    while (std::getline(source, line)) {
        constexpr std::string_view prefix =
            "#   - Header.Toltec.AccumLen:";
        if (line.starts_with(prefix)) {
            destination << "#   - accumlen: "
                        << input.tune_accumulation_length << '\n';
            ++replacement_count;
        } else {
            destination << line << '\n';
        }
    }
    require(source.eof() && static_cast<bool>(destination) &&
                replacement_count == 1,
            "Tune fit report normalization did not bind exactly one AccumLen");
    return output;
}

} // namespace citlali::cli::input_bindings
