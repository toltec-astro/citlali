#include <citlali/core/pipeline/canonical_apt_bundle_v2.h>
#include <citlali/core/pipeline/canonical_apt_detector_relation_v2.h>
#include <citlali/core/pipeline/observation_setup_validation.h>
#include <citlali/core/pipeline/paired_readout_kids_adapter.h>
#include <citlali/core/pipeline/rtc_only_route.h>
#include <citlali/core/utils/sha256.h>

#include <citlali_config/gitversion.h>
#include <citlali_wp7/acceptance_build_identity.h>
#include <kidscpp_config/gitversion.h>
#include <tula_config/gitversion.h>

#include <kids/core/kidsdata.h>
#include <kids/timestream/solver.h>
#include <kids/toltec/toltec.h>

#include <tula/container.h>
#include <tula/datatable.h>

#include <netcdf>
#include <spdlog/logger.h>
#include <spdlog/sinks/callback_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <tuple>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace pipeline = citlali::pipeline;
namespace apt = citlali::pipeline::canonical_apt_v2;

constexpr std::string_view acceptance_schema =
    "citlali-wp7-identity-rtc-acceptance-v6";
constexpr std::string_view occurrence_support_assignment_schema =
    "citlali-native-occurrence-support-assignment-v1";
constexpr std::string_view occurrence_support_duration_relation =
    "Header.Toltec.AccumLen / Header.Toltec.FpgaFreq";
constexpr std::string_view occurrence_support_assignment_status =
    "provisional_calibration_pending";
constexpr std::string_view occurrence_support_assignment_id =
    "wp7-provisional-integration-center-152390-v1";
constexpr std::string_view occurrence_support_assignment_sha256 =
    "6fc4e9009b98190c42cc3f6e7e030fa317e8ae5f9e707cd968110a696fac2b6c";
constexpr std::string_view occurrence_support_calibration_disposition =
    "replace_with_calibrated_producer_relation_when_available";
constexpr std::string_view producer_interface =
    "TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE v0.1/r0.1";
constexpr std::string_view producer_interface_sha256 =
    "f9659b34a49a07d4287c4a70db798cdd2ec30049531da603fcca1e9d1fdd5969";
constexpr std::string_view design_commit = "46824f7de";
constexpr std::string_view align_repair_commit = "d55deefb3";
constexpr std::string_view kidscpp_revision =
    "04088da182622c3e879f04314974a7c0d60ee2d6";
constexpr std::string_view kidscpp_patch_sha256 =
    CITLALI_WP7_KIDSCPP_PATCH_SHA256;
constexpr std::string_view tula_revision =
    "f30f81d97c44bd79618273bb842302ef839c6ab1";
constexpr std::string_view tula_patch_sha256 =
    CITLALI_WP7_TULA_PATCH_SHA256;

struct Arguments {
    fs::path data_directory;
    fs::path apt_manifest;
    fs::path config;
    fs::path producer_interface_artifact;
    fs::path occurrence_support_assignment_artifact;
    fs::path kidscpp_build_patch;
    fs::path tula_build_patch;
    fs::path output;
    fs::path executable;
    std::string executable_sha256;
    std::string source_revision;
    std::string dataset_id = "SCI_ALIGN_STAGE7_NGC4449_152390";
    std::int64_t first_native_row = 20000;
    std::int64_t native_row_count = 2048;
    bool owner_run = false;
    bool design_is_ancestor = false;
    bool align_repair_is_ancestor = false;
};

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

std::string usage() {
    return
        "Usage: citlali_wp7_identity_rtc_acceptance\n"
        "  --data-dir PATH --apt-manifest PATH --config PATH\n"
        "  --producer-interface-artifact PATH\n"
        "  --occurrence-support-assignment PATH\n"
        "  --kidscpp-build-patch PATH --tula-build-patch PATH\n"
        "  --output PATH\n"
        "  --source-revision FULL_SHA --owner-run\n"
        "  --design-is-ancestor --align-repair-is-ancestor\n"
        "  [--dataset-id ID] [--first-native-row N] [--native-row-count N]\n";
}

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

Arguments parse_arguments(int argc, char **argv) {
    Arguments result;
    result.executable = fs::absolute(argv[0]);
    auto next = [&](int &index, const std::string &option) {
        require(index + 1 < argc, option + " requires a value");
        return std::string{argv[++index]};
    };
    for (int index = 1; index < argc; ++index) {
        const std::string option{argv[index]};
        if (option == "--help" || option == "-h") {
            std::cout << usage();
            std::exit(0);
        } else if (option == "--data-dir") {
            result.data_directory = next(index, option);
        } else if (option == "--apt-manifest") {
            result.apt_manifest = next(index, option);
        } else if (option == "--config") {
            result.config = next(index, option);
        } else if (option == "--producer-interface-artifact") {
            result.producer_interface_artifact = next(index, option);
        } else if (option == "--occurrence-support-assignment") {
            result.occurrence_support_assignment_artifact =
                next(index, option);
        } else if (option == "--kidscpp-build-patch") {
            result.kidscpp_build_patch = next(index, option);
        } else if (option == "--tula-build-patch") {
            result.tula_build_patch = next(index, option);
        } else if (option == "--output") {
            result.output = next(index, option);
        } else if (option == "--source-revision") {
            result.source_revision = next(index, option);
        } else if (option == "--dataset-id") {
            result.dataset_id = next(index, option);
        } else if (option == "--first-native-row") {
            result.first_native_row = parse_integer(next(index, option), option);
        } else if (option == "--native-row-count") {
            result.native_row_count = parse_integer(next(index, option), option);
        } else if (option == "--owner-run") {
            result.owner_run = true;
        } else if (option == "--design-is-ancestor") {
            result.design_is_ancestor = true;
        } else if (option == "--align-repair-is-ancestor") {
            result.align_repair_is_ancestor = true;
        } else {
            fail("unknown option: " + option);
        }
    }
    require(!result.data_directory.empty(), "--data-dir is required");
    require(!result.apt_manifest.empty(), "--apt-manifest is required");
    require(!result.config.empty(), "--config is required");
    require(!result.producer_interface_artifact.empty(),
            "--producer-interface-artifact is required");
    require(!result.occurrence_support_assignment_artifact.empty(),
            "--occurrence-support-assignment is required");
    require(!result.kidscpp_build_patch.empty(),
            "--kidscpp-build-patch is required");
    require(!result.tula_build_patch.empty(),
            "--tula-build-patch is required");
    require(!result.output.empty(), "--output is required");
    require(!result.source_revision.empty(),
            "--source-revision is required");
    require(result.first_native_row >= 0,
            "--first-native-row must be nonnegative");
    require(result.native_row_count >= 4,
            "--native-row-count must be at least four");
    return result;
}

bool full_lowercase_git_sha(std::string_view value) {
    return value.size() == 40 &&
           std::all_of(value.begin(), value.end(), [](unsigned char ch) {
               return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
           });
}

enum class NativeEventTimeRole : std::uint8_t {
    integration_start,
    integration_center,
    integration_end,
};

constexpr const char *native_event_time_role_name(
    NativeEventTimeRole role) noexcept {
    switch (role) {
        case NativeEventTimeRole::integration_start:
            return "integration_start";
        case NativeEventTimeRole::integration_center:
            return "integration_center";
        case NativeEventTimeRole::integration_end:
            return "integration_end";
    }
    return "unknown";
}

struct NativeOccurrenceSupportAssignment {
    std::string assignment_id;
    std::string assigned_by;
    std::string assigned_at_utc;
    std::int64_t scope_observation = 0;
    NativeEventTimeRole event_time_role =
        NativeEventTimeRole::integration_center;
    std::string artifact_sha256;
};

NativeOccurrenceSupportAssignment load_occurrence_support_assignment(
    const fs::path &path) {
    require(fs::is_regular_file(path),
            "occurrence-support assignment is not a regular file");
    const auto root = YAML::LoadFile(path.string());
    require(root.IsMap(),
            "occurrence-support assignment must be a YAML map");
    const std::set<std::string> required{
        "schema", "assignment_id", "assignment_status", "assigned_by",
        "assigned_at_utc", "scope_observation", "producer_interface_id",
        "producer_interface_sha256", "event_time_role",
        "duration_relation", "calibration_disposition"};
    require(root.size() == required.size(),
            "occurrence-support assignment has an open or incomplete schema");
    for (const auto &entry : root) {
        require(entry.first.IsScalar() &&
                    required.contains(entry.first.as<std::string>()),
                "occurrence-support assignment has an unknown field");
    }
    const auto scalar = [&](const char *name) {
        const auto value = root[name];
        require(value && value.IsScalar(),
                std::string{"occurrence-support assignment lacks scalar "} +
                    name);
        return value.as<std::string>();
    };
    require(scalar("schema") == occurrence_support_assignment_schema,
            "occurrence-support assignment schema is not supported");
    NativeOccurrenceSupportAssignment result;
    result.assignment_id = scalar("assignment_id");
    require(!result.assignment_id.empty(),
            "occurrence-support assignment id is empty");
    require(scalar("assignment_status") ==
                occurrence_support_assignment_status,
            "occurrence-support assignment must remain calibration pending");
    result.assigned_by = scalar("assigned_by");
    result.assigned_at_utc = scalar("assigned_at_utc");
    require(!result.assigned_by.empty() &&
                std::regex_match(
                    result.assigned_at_utc,
                    std::regex{
                        R"(^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$)"}),
            "occurrence-support assignment identity or UTC is invalid");
    result.scope_observation = root["scope_observation"].as<std::int64_t>();
    require(result.scope_observation == 152390,
            "occurrence-support assignment is outside observation 152390");
    require(scalar("producer_interface_id") == producer_interface &&
                scalar("producer_interface_sha256") ==
                    producer_interface_sha256,
            "occurrence-support assignment names a different producer interface");
    require(result.assignment_id == occurrence_support_assignment_id,
            "occurrence-support assignment id is not the approved assignment");
    require(scalar("event_time_role") == "integration_center",
            "occurrence-support assignment must use integration center");
    result.event_time_role = NativeEventTimeRole::integration_center;
    require(scalar("duration_relation") ==
                occurrence_support_duration_relation,
            "occurrence-support assignment duration relation is unsupported");
    require(scalar("calibration_disposition") ==
                occurrence_support_calibration_disposition,
            "occurrence-support assignment calibration disposition is unsupported");
    result.artifact_sha256 = citlali::utils::sha256_file(path);
    require(result.artifact_sha256 == occurrence_support_assignment_sha256,
            "occurrence-support assignment artifact SHA-256 is not approved");
    return result;
}

std::string json_escape(std::string_view value) {
    std::ostringstream stream;
    for (const unsigned char ch : value) {
        switch (ch) {
        case '\"': stream << "\\\""; break;
        case '\\': stream << "\\\\"; break;
        case '\b': stream << "\\b"; break;
        case '\f': stream << "\\f"; break;
        case '\n': stream << "\\n"; break;
        case '\r': stream << "\\r"; break;
        case '\t': stream << "\\t"; break;
        default:
            if (ch < 0x20) {
                stream << "\\u" << std::hex << std::setw(4)
                       << std::setfill('0') << static_cast<int>(ch)
                       << std::dec;
            } else {
                stream << static_cast<char>(ch);
            }
        }
    }
    return stream.str();
}

struct LogCounts {
    std::size_t errors = 0;
    std::size_t criticals = 0;
};

std::pair<std::shared_ptr<spdlog::logger>, std::shared_ptr<LogCounts>>
configure_logging() {
    auto counts = std::make_shared<LogCounts>();
    auto console = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
    auto counter = std::make_shared<spdlog::sinks::callback_sink_mt>(
        [counts](const spdlog::details::log_msg &message) {
            if (message.level == spdlog::level::critical) {
                ++counts->criticals;
            } else if (message.level == spdlog::level::err) {
                ++counts->errors;
            }
        });
    std::vector<spdlog::sink_ptr> sinks{console, counter};
    auto logger = std::make_shared<spdlog::logger>(
        "citlali_logger", sinks.begin(), sinks.end());
    logger->set_level(spdlog::level::warn);
    spdlog::register_logger(logger);
    spdlog::set_default_logger(logger);
    return {std::move(logger), std::move(counts)};
}

template <class T>
T read_netcdf_scalar(netCDF::NcFile &file, const std::string &name) {
    const auto variable = file.getVar(name);
    require(!variable.isNull(), "missing NetCDF scalar " + name);
    T result{};
    variable.getVar(&result);
    return result;
}

struct RuntimeConfig {
    std::array<double, 13> interface_offsets_sec{};
    std::array<bool, 13> interface_offset_present{};
    std::string kids_model;
};

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

fs::path find_raw_file(const fs::path &directory,
                       const pipeline::CanonicalAptRawSourceBindingV2 &source) {
    const auto &observation = source.header_observation;
    std::ostringstream prefix;
    prefix << source.interface_name << '_' << observation.observation << '_'
           << std::setw(3) << std::setfill('0')
           << observation.subobservation << '_' << std::setw(4)
           << observation.scan << '_';
    std::vector<fs::path> candidates;
    for (const auto &entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;
        const auto name = entry.path().filename().string();
        if (name.starts_with(prefix.str()) && entry.path().extension() == ".nc") {
            candidates.push_back(entry.path());
        }
    }
    require(candidates.size() == 1,
            "expected exactly one raw file for " + source.interface_name);
    return fs::absolute(candidates.front());
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

struct NetworkInput {
    pipeline::CanonicalAptRawSourceBindingV2 source;
    fs::path raw_path;
    fs::path tune_path;
    std::string tune_sha256;
    double fpga_frequency_hz = 0.0;
    double sample_frequency_hz = 0.0;
    std::int64_t accumulation_length = 0;
    std::int64_t tune_accumulation_length = 0;
    std::vector<bool> tune_valid;
    std::shared_ptr<const pipeline::NativeNetworkAlignment> native_timing;
};

fs::path find_tune_file(const fs::path &directory,
                        const fs::path &raw_path) {
    const auto [kind, meta] = kids::toltec::get_meta<>(raw_path.string());
    require((kind & kids::KidsDataKind::RawTimeStream) ==
                kids::KidsDataKind::RawTimeStream,
            "raw source is not a KIDs raw timestream");
    require(meta.has("cal_file"),
            "raw timestream lacks its Tune fit-report relation");
    const std::regex pattern{meta.get_str("cal_file")};
    std::vector<fs::path> candidates;
    for (const auto &entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;
        if (std::regex_match(entry.path().filename().string(), pattern)) {
            candidates.push_back(entry.path());
        }
    }
    require(candidates.size() == 1,
            "expected exactly one Tune fit report for " + raw_path.string());
    return fs::absolute(candidates.front());
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

struct TuneFacts {
    std::vector<bool> valid;
    std::int64_t accumulation_length = 0;
    std::int64_t observation = 0;
    std::int64_t subobservation = 0;
    std::int64_t scan = 0;
    std::int64_t network = -1;
};

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

std::vector<NetworkInput> resolve_network_inputs(
    const Arguments &arguments,
    const pipeline::CanonicalAptDetectorRelationV2 &relation,
    const RuntimeConfig &config) {
    std::vector<NetworkInput> result;
    result.reserve(relation.raw_sources().size());
    for (const auto &source : relation.raw_sources()) {
        require(source.network >= 0 && source.network < 13,
                "APT source network is outside toltec0..toltec12");
        require(config.interface_offset_present[
                    static_cast<std::size_t>(source.network)],
                "config lacks the exact participant interface offset");
        auto raw_path = find_raw_file(arguments.data_directory, source);
        verify_raw_file(raw_path, source);
        auto tune_path = find_tune_file(arguments.data_directory, raw_path);
        const auto tune_sha256 = citlali::utils::sha256_file(tune_path);
        auto tune = read_tune_facts(tune_path, source.channel_count);
        double fpga_frequency_hz = 0.0;
        double sample_frequency_hz = 0.0;
        std::int64_t accumulation_length = 0;
        auto timestamps = read_timestamp_slice(
            raw_path, arguments.first_native_row,
            arguments.native_row_count, fpga_frequency_hz,
            sample_frequency_hz, accumulation_length);
        require(std::isfinite(fpga_frequency_hz) &&
                    fpga_frequency_hz > 0.0 &&
                    std::isfinite(sample_frequency_hz) &&
                    sample_frequency_hz > 0.0 && accumulation_length > 0,
                "raw cadence metadata are invalid");
        require(tune.accumulation_length > 0 &&
                    tune.observation == relation.observation().observation &&
                    tune.subobservation == 0 && tune.scan == 1 &&
                    tune.network == source.network,
                "Tune metadata disagrees with its exact raw producer binding");
        const double primitive_duration =
            static_cast<double>(accumulation_length) / fpga_frequency_hz;
        require(std::abs(primitive_duration - 1.0 / sample_frequency_hz) <=
                    8.0 * std::numeric_limits<double>::epsilon() *
                        primitive_duration,
                "AccumLen/FpgaFreq disagrees with SampleFreq");
        auto timing = std::make_shared<const pipeline::NativeNetworkAlignment>(
            pipeline::make_native_network_alignment(
                source.network, arguments.first_native_row, timestamps,
                fpga_frequency_hz,
                config.interface_offsets_sec[
                    static_cast<std::size_t>(source.network)]));
        result.push_back(NetworkInput{
            source, std::move(raw_path), std::move(tune_path),
            tune_sha256, fpga_frequency_hz,
            sample_frequency_hz, accumulation_length,
            tune.accumulation_length,
            std::move(tune.valid),
            std::move(timing)});
    }
    require(!result.empty(), "APT relation has no raw network inputs");
    return result;
}

std::vector<pipeline::PairedReadoutDetectorIdentity> detector_axis(
    const pipeline::CanonicalAptDetectorRelationV2 &relation,
    const NetworkInput &input) {
    std::vector<std::optional<pipeline::PairedReadoutDetectorIdentity>> by_channel(
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
        destination = pipeline::PairedReadoutDetectorIdentity{
            binding.output_uid, binding.array,
            static_cast<pipeline::TimestreamNetworkId>(binding.network),
            binding.raw_source_uid,
            static_cast<Eigen::Index>(binding.channel)};
    }
    std::vector<pipeline::PairedReadoutDetectorIdentity> result;
    result.reserve(by_channel.size());
    for (std::size_t channel = 0; channel < by_channel.size(); ++channel) {
        require(by_channel[channel].has_value(),
                "APT detector binding does not cover every raw channel");
        require(by_channel[channel]->raw_channel ==
                    static_cast<Eigen::Index>(channel),
                "APT detector channel order is not native solver order");
        result.push_back(*by_channel[channel]);
    }
    return result;
}

std::shared_ptr<const pipeline::NativeReadoutMappingIdentity> mapping_identity(
    const NetworkInput &input, const RuntimeConfig &config) {
    const auto raw_identity =
        input.source.content_sha256 + ":network=" +
        std::to_string(input.source.network);
    const auto tune_identity = "sha256:" + input.tune_sha256;
    const auto transform = std::string{"kidscpp:"} + KIDSCPP_GIT_VERSION +
        ":S21WithGainLinTrend:tune-meta-normalized-v1:raw-accumlen=" +
        std::to_string(input.accumulation_length) + ":tune-accumlen=" +
        std::to_string(input.tune_accumulation_length) +
        ":kidscpp-patch=" + std::string{kidscpp_patch_sha256} +
        ":tula-patch=" + std::string{tula_patch_sha256};
    const auto revision = transform + ":tune=" + tune_identity;
    return std::make_shared<const pipeline::NativeReadoutMappingIdentity>(
        pipeline::NativeReadoutMappingIdentity{
            std::string{producer_interface}, raw_identity, tune_identity,
            revision, transform,
            "kidscpp:" + config.kids_model + ":xs:dimensionless",
            "kidscpp:" + config.kids_model + ":rs:dimensionless"});
}

std::shared_ptr<const pipeline::PairedReadoutOccurrenceAxis> occurrence_axis(
    const NetworkInput &input, std::int64_t first_native_row,
    std::int64_t native_row_count, NativeEventTimeRole event_time_role) {
    const double duration =
        static_cast<double>(input.accumulation_length) /
        input.fpga_frequency_hz;
    std::vector<pipeline::NativeOccurrenceInterval> intervals;
    intervals.reserve(static_cast<std::size_t>(native_row_count));
    for (std::int64_t offset = 0; offset < native_row_count; ++offset) {
        const auto row = first_native_row + offset;
        const double event = input.native_timing->identity(row)
                                 .reconstructed_time_unix_sec();
        switch (event_time_role) {
            case NativeEventTimeRole::integration_start:
                intervals.push_back({event, event + duration});
                break;
            case NativeEventTimeRole::integration_center:
                intervals.push_back({event - duration / 2.0,
                                     event + duration / 2.0});
                break;
            case NativeEventTimeRole::integration_end:
                intervals.push_back({event - duration, event});
                break;
        }
    }
    return std::make_shared<const pipeline::PairedReadoutOccurrenceAxis>(
        input.native_timing, first_native_row, std::move(intervals));
}

struct ComparisonMetrics {
    std::size_t paired_ingress_value_comparison_count = 0;
    std::size_t paired_ingress_identity_comparison_count = 0;
    std::size_t paired_ingress_member_state_comparison_count = 0;
    std::size_t rtc_product_value_comparison_count = 0;
    std::size_t identity_comparison_count = 0;
    std::size_t support_comparison_count = 0;
    std::size_t native_time_comparison_count = 0;
    std::size_t representative_native_comparison_count = 0;
    std::size_t pair_decision_comparison_count = 0;
    std::size_t pair_causal_evidence_comparison_count = 0;
    std::size_t assigned_support_binding_count = 0;
    std::size_t x_bitwise_mismatch_count = 0;
    std::size_t r_bitwise_mismatch_count = 0;
    std::size_t paired_ingress_identity_mismatch_count = 0;
    std::size_t paired_ingress_member_state_mismatch_count = 0;
    std::size_t identity_mismatch_count = 0;
    std::size_t support_mismatch_count = 0;
    std::size_t pair_decision_mismatch_count = 0;
    std::size_t pair_causal_evidence_mismatch_count = 0;
    std::size_t member_cause_mismatch_count = 0;
    std::size_t selected_time_mismatch_count = 0;
    std::size_t representative_native_mismatch_count = 0;
    std::size_t chunk_realized_operator_comparison_count = 0;
    std::size_t chunk_realized_operator_mismatch_count = 0;
    std::size_t chunk_scientific_comparison_count = 0;
    std::size_t chunk_scientific_mismatch_count = 0;
    std::size_t assigned_support_binding_mismatch_count = 0;
};

pipeline::NativeOccurrenceInterval assigned_occurrence_interval(
    double event, double duration, NativeEventTimeRole event_time_role) {
    switch (event_time_role) {
        case NativeEventTimeRole::integration_start:
            return {event, event + duration};
        case NativeEventTimeRole::integration_center:
            return {event - duration / 2.0, event + duration / 2.0};
        case NativeEventTimeRole::integration_end:
            return {event - duration, event};
    }
    fail("unsupported native event-time role");
}

void compare_occurrence_axis_to_assignment(
    const pipeline::PairedReadoutOccurrenceAxis &axis,
    const NetworkInput &input, NativeEventTimeRole event_time_role,
    ComparisonMetrics &metrics) {
    const double duration =
        static_cast<double>(input.accumulation_length) /
        input.fpga_frequency_hz;
    for (auto row = axis.first_native_row();
         row < axis.past_last_native_row(); ++row) {
        ++metrics.assigned_support_binding_count;
        const auto event = input.native_timing->identity(row)
                               .reconstructed_time_unix_sec();
        const auto expected = assigned_occurrence_interval(
            event, duration, event_time_role);
        if (!(axis.interval(row) == expected)) {
            ++metrics.assigned_support_binding_mismatch_count;
        }
    }
}

struct NativeValueOracle {
    std::vector<pipeline::PairedReadoutDetectorIdentity> detectors;
    std::vector<std::tuple<pipeline::TimestreamNativeRow, Eigen::Index,
                           std::uint64_t, std::uint64_t>> cells;
};

pipeline::PairedReadoutDetectorIdentity producer_detector_identity(
    const pipeline::CanonicalAptDetectorRelationV2 &relation,
    const NetworkInput &input, Eigen::Index detector) {
    require(detector >= 0 && detector < input.source.channel_count,
            "producer detector oracle index is outside the raw channel axis");
    const auto found = std::find_if(
        relation.bindings().begin(), relation.bindings().end(),
        [&](const auto &binding) {
            return binding.network == input.source.network &&
                   binding.raw_source_uid == input.source.source_uid &&
                   binding.channel == detector;
        });
    require(found != relation.bindings().end(),
            "producer detector oracle lacks an APT binding");
    require(std::find_if(std::next(found), relation.bindings().end(),
                [&](const auto &binding) {
                    return binding.network == input.source.network &&
                           binding.raw_source_uid == input.source.source_uid &&
                           binding.channel == detector;
                }) == relation.bindings().end(),
            "producer detector oracle has an ambiguous APT binding");
    return {found->output_uid, found->array,
            static_cast<pipeline::TimestreamNetworkId>(found->network),
            found->raw_source_uid,
            static_cast<Eigen::Index>(found->channel)};
}

pipeline::ReadoutMemberCause producer_member_causes(
    bool tune_valid, double value) {
    const bool finite = std::isfinite(value);
    auto causes = pipeline::ReadoutMemberCause::none;
    if (!(tune_valid && finite)) {
        causes = causes | pipeline::ReadoutMemberCause::producer_invalid;
    }
    if (!finite) {
        causes = causes | pipeline::ReadoutMemberCause::nonfinite_payload;
    }
    return causes;
}

bool producer_member_valid(bool tune_valid, double value) {
    return tune_valid && std::isfinite(value);
}

pipeline::PairedReadoutCause producer_pair_causes(
    pipeline::ReadoutMemberCause x_causes,
    pipeline::ReadoutMemberCause r_causes) {
    auto causes = pipeline::PairedReadoutCause::none;
    if (pipeline::has_cause(
            x_causes, pipeline::ReadoutMemberCause::producer_invalid)) {
        causes = causes | pipeline::PairedReadoutCause::x_original_invalid;
    }
    if (pipeline::has_cause(
            r_causes, pipeline::ReadoutMemberCause::producer_invalid)) {
        causes = causes | pipeline::PairedReadoutCause::r_original_invalid;
    }
    if (pipeline::has_cause(
            x_causes, pipeline::ReadoutMemberCause::nonfinite_payload)) {
        causes = causes | pipeline::PairedReadoutCause::x_nonfinite;
    }
    if (pipeline::has_cause(
            r_causes, pipeline::ReadoutMemberCause::nonfinite_payload)) {
        causes = causes | pipeline::PairedReadoutCause::r_nonfinite;
    }
    return causes;
}

NativeValueOracle make_native_value_oracle(
    const kids::TimeStreamSolverResult &result,
    std::int64_t first_native_row,
    const pipeline::CanonicalAptDetectorRelationV2 &relation,
    const NetworkInput &input) {
    NativeValueOracle oracle;
    const auto &x = result.data_out.xs.data;
    const auto &r = result.data_out.rs.data;
    oracle.detectors.reserve(static_cast<std::size_t>(x.cols()));
    for (Eigen::Index detector = 0; detector < x.cols(); ++detector) {
        oracle.detectors.push_back(
            producer_detector_identity(relation, input, detector));
    }
    for (Eigen::Index local = 0; local < x.rows(); ++local) {
        const auto native_row = first_native_row +
            static_cast<pipeline::TimestreamNativeRow>(local);
        for (Eigen::Index detector = 0; detector < x.cols(); ++detector) {
            oracle.cells.emplace_back(
                native_row, detector,
                std::bit_cast<std::uint64_t>(x(local, detector)),
                std::bit_cast<std::uint64_t>(r(local, detector)));
        }
    }
    return oracle;
}

void compare_moved_native_ingress(
    const NativeValueOracle &oracle,
    const pipeline::PairedReadoutNetwork &network,
    const std::vector<bool> &tune_valid,
    ComparisonMetrics &metrics) {
    require(oracle.detectors.size() ==
                static_cast<std::size_t>(network.detector_count()) &&
            tune_valid.size() == oracle.detectors.size(),
            "producer oracle does not cover the admitted detector axis");
    for (Eigen::Index detector = 0;
         detector < network.detector_count(); ++detector) {
        ++metrics.paired_ingress_identity_comparison_count;
        if (!(network.detector(detector) ==
              oracle.detectors[static_cast<std::size_t>(detector)])) {
            ++metrics.paired_ingress_identity_mismatch_count;
        }
    }
    for (const auto &[row, detector, x_bits, r_bits] : oracle.cells) {
        ++metrics.paired_ingress_value_comparison_count;
        if (std::bit_cast<std::uint64_t>(network.value(
                pipeline::ReadoutMember::x, row, detector)) != x_bits) {
            ++metrics.x_bitwise_mismatch_count;
        }
        ++metrics.paired_ingress_value_comparison_count;
        if (std::bit_cast<std::uint64_t>(network.value(
                pipeline::ReadoutMember::r, row, detector)) != r_bits) {
            ++metrics.r_bitwise_mismatch_count;
        }
        const auto tune_ok = tune_valid.at(
            static_cast<std::size_t>(detector));
        for (const auto [member, value_bits] :
             {std::pair{pipeline::ReadoutMember::x, x_bits},
              std::pair{pipeline::ReadoutMember::r, r_bits}}) {
            ++metrics.paired_ingress_member_state_comparison_count;
            const auto value = std::bit_cast<double>(value_bits);
            const auto &state = network.state(member, row, detector);
            const auto expected_valid = producer_member_valid(tune_ok, value);
            const auto expected_causes = producer_member_causes(tune_ok, value);
            if (!state.available() || !state.in_acquisition_support() ||
                state.origin() != pipeline::ReadoutMemberOrigin::measured ||
                state.original_valid() != expected_valid ||
                state.valid() != expected_valid ||
                state.causes() != expected_causes) {
                ++metrics.paired_ingress_member_state_mismatch_count;
            }
        }
    }
}

std::vector<pipeline::ReadoutMemberState> member_states(
    const pipeline::PairedReadoutMatrix &values,
    const std::vector<bool> &tune_valid) {
    require(values.cols() == static_cast<Eigen::Index>(tune_valid.size()),
            "Tune validity does not match solver columns");
    std::vector<pipeline::ReadoutMemberState> result;
    result.reserve(static_cast<std::size_t>(values.size()));
    for (Eigen::Index row = 0; row < values.rows(); ++row) {
        for (Eigen::Index column = 0; column < values.cols(); ++column) {
            const bool finite = std::isfinite(values(row, column));
            result.push_back(pipeline::ReadoutMemberState::measured(
                true, tune_valid[static_cast<std::size_t>(column)] && finite,
                true, finite));
        }
    }
    return result;
}

struct PairedBuildResult {
    std::shared_ptr<const pipeline::PairedReadout> paired;
    ComparisonMetrics ingress_comparisons;
    std::string mapping_instance_id;
};

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        std::random_device random;
        for (int attempt = 0; attempt < 100; ++attempt) {
            std::ostringstream name;
            name << "citlali-wp7-tune-" << std::hex
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

    TemporaryDirectory(const TemporaryDirectory &) = delete;
    TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;

    ~TemporaryDirectory() {
        std::error_code error;
        fs::remove_all(path_, error);
    }

    const fs::path &path() const noexcept { return path_; }

private:
    fs::path path_;
};

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

PairedBuildResult build_paired_readout(
    const Arguments &arguments,
    const pipeline::NativeObservationScope &scope,
    const pipeline::CanonicalAptDetectorRelationV2 &relation,
    const std::vector<NetworkInput> &inputs,
    const RuntimeConfig &config,
    const NativeOccurrenceSupportAssignment &support_assignment) {
    std::vector<pipeline::PairedReadoutNetwork> networks;
    std::vector<pipeline::TimestreamNetworkId> inventory;
    ComparisonMetrics metrics;
    citlali::utils::Sha256 mapping_digest;
    TemporaryDirectory temporary_tunes;
    networks.reserve(inputs.size());
    inventory.reserve(inputs.size());
    for (const auto &input : inputs) {
        inventory.push_back(input.source.network);
        const auto slice = tula::container_utils::IndexSlice{
            static_cast<Eigen::Index>(arguments.first_native_row),
            static_cast<Eigen::Index>(arguments.first_native_row +
                                      arguments.native_row_count),
            std::nullopt};
        auto raw = kids::toltec::read_data_slice<
            kids::KidsDataKind::RawTimeStream>(input.raw_path.string(), slice);
        const auto normalized_tune = normalized_tune_report(
            input, temporary_tunes.path());
        kids::TimeStreamSolver solver(kids::TimeStreamSolver::Config{
            {"fitreportfile", normalized_tune.string()},
            {"exmode", std::string{"seq"}},
            {"extra_output", false}});
        auto solved = solver(raw);
        require(solved.data_out.xs.data.rows() ==
                    arguments.native_row_count &&
                    solved.data_out.xs.data.cols() ==
                        input.source.channel_count &&
                    solved.data_out.rs.data.rows() ==
                        solved.data_out.xs.data.rows() &&
                    solved.data_out.rs.data.cols() ==
                        solved.data_out.xs.data.cols(),
                "KIDs solver did not produce the exact paired native shape");
        auto oracle = make_native_value_oracle(
            solved, arguments.first_native_row, relation, input);
        auto mapping = mapping_identity(input, config);
        mapping_digest.update(mapping->producer_instance_id);
        mapping_digest.update(mapping->mapping_revision);
        auto axis = occurrence_axis(
            input, arguments.first_native_row,
            arguments.native_row_count, support_assignment.event_time_role);
        compare_occurrence_axis_to_assignment(
            *axis, input, support_assignment.event_time_role, metrics);
        pipeline::PairedReadoutNetworkIngress ingress{
            std::move(axis),
            detector_axis(relation, input), mapping,
            member_states(solved.data_out.xs.data, input.tune_valid),
            member_states(solved.data_out.rs.data, input.tune_valid)};
        auto network = pipeline::take_paired_kids_solver_result(
            std::move(ingress), std::move(solved));
        compare_moved_native_ingress(
            oracle, network, input.tune_valid, metrics);
        networks.push_back(std::move(network));
    }
    return {
        pipeline::PairedReadout::admit(
            scope, std::move(inventory), std::move(networks)),
        metrics, "sha256:" + mapping_digest.finish()};
}

pipeline::RtcOnlyRouteOutcome run_route(
    std::uint64_t run,
    const std::shared_ptr<const pipeline::PairedReadout> &paired,
    std::vector<pipeline::NativeOccurrenceSpan> logical_spans,
    std::vector<std::vector<pipeline::NativeOccurrenceSpan>>
        engineering_partitions,
    pipeline::RtcOnlyProductSlot &publication) {
    return pipeline::run_identity_rtc_only(
        {{run}, paired, std::move(logical_spans),
         std::move(engineering_partitions)},
        publication);
}

const NetworkInput &producer_input(
    const std::vector<NetworkInput> &inputs,
    pipeline::TimestreamNetworkId network_id) {
    const auto found = std::find_if(
        inputs.begin(), inputs.end(), [&](const auto &input) {
            return input.source.network == network_id;
        });
    require(found != inputs.end(),
            "RTC product network lacks immutable producer facts");
    return *found;
}

void compare_full_product_to_producer_facts(
    const pipeline::RtcTimestream &product,
    const pipeline::CanonicalAptDetectorRelationV2 &relation,
    const std::vector<NetworkInput> &inputs,
    ComparisonMetrics &metrics) {
    const auto &view = *product.input_handle();
    const auto &paired = *product.native_parent_handle();
    const auto &evidence = *product.plan_handle()->evidence_handle();
    for (const auto &span : product.network_spans()) {
        const auto &native = paired.network(span.network_id);
        const auto &input = producer_input(inputs, span.network_id);
        const auto duration =
            static_cast<double>(input.accumulation_length) /
            input.fpga_frequency_hz;
        require(native.detector_count() == input.source.channel_count,
                "RTC product detector axis disagrees with producer facts");
        std::vector<pipeline::PairedReadoutDetectorIdentity>
            expected_detectors;
        expected_detectors.reserve(
            static_cast<std::size_t>(native.detector_count()));
        for (Eigen::Index detector = 0;
             detector < native.detector_count(); ++detector) {
            expected_detectors.push_back(
                producer_detector_identity(relation, input, detector));
        }
        for (auto row = span.first_native_row;
             row < span.past_last_native_row; ++row) {
            const auto expected_native = input.native_timing->identity(row);
            const auto expected_interval = assigned_occurrence_interval(
                expected_native.reconstructed_time_unix_sec(), duration,
                NativeEventTimeRole::integration_center);
            ++metrics.native_time_comparison_count;
            if (std::bit_cast<std::uint64_t>(
                    product.output_time_unix_sec(span.network_id, row)) !=
                std::bit_cast<std::uint64_t>(
                    expected_native.reconstructed_time_unix_sec())) {
                ++metrics.selected_time_mismatch_count;
            }
            ++metrics.representative_native_comparison_count;
            if (!(product.representative_native_identity(
                      span.network_id, row) == expected_native)) {
                ++metrics.representative_native_mismatch_count;
            }
            ++metrics.support_comparison_count;
            if (!(product.representative_interval(span.network_id, row) ==
                  expected_interval)) {
                ++metrics.support_mismatch_count;
            }

            for (Eigen::Index detector = 0;
                 detector < native.detector_count(); ++detector) {
                ++metrics.identity_comparison_count;
                const auto &expected_detector = expected_detectors.at(
                    static_cast<std::size_t>(detector));
                const pipeline::RtcNativeCellIdentity expected_identity{
                    span.network_id, row, expected_detector.output_uid};
                if (!(product.identity(span.network_id, row, detector) ==
                      expected_identity)) {
                    ++metrics.identity_mismatch_count;
                }

                for (const auto member :
                     {pipeline::ReadoutMember::x,
                      pipeline::ReadoutMember::r}) {
                    ++metrics.rtc_product_value_comparison_count;
                    const auto actual = product.value(
                        member, span.network_id, row, detector);
                    const auto expected = native.value(
                        member, row, detector);
                    if (std::bit_cast<std::uint64_t>(actual) !=
                        std::bit_cast<std::uint64_t>(expected)) {
                        if (member == pipeline::ReadoutMember::x) {
                            ++metrics.x_bitwise_mismatch_count;
                        } else {
                            ++metrics.r_bitwise_mismatch_count;
                        }
                    }
                    const auto expected_causes = producer_member_causes(
                        input.tune_valid.at(
                            static_cast<std::size_t>(detector)),
                        actual);
                    if (product.member_local_causes(
                            member, span.network_id, row, detector) !=
                        expected_causes) {
                        ++metrics.member_cause_mismatch_count;
                    }
                }

                ++metrics.pair_decision_comparison_count;
                const auto tune_ok = input.tune_valid.at(
                    static_cast<std::size_t>(detector));
                const auto x_value = product.value(
                    pipeline::ReadoutMember::x,
                    span.network_id, row, detector);
                const auto r_value = product.value(
                    pipeline::ReadoutMember::r,
                    span.network_id, row, detector);
                const auto x_valid = producer_member_valid(tune_ok, x_value);
                const auto r_valid = producer_member_valid(tune_ok, r_value);
                const auto expected_decision = x_valid && r_valid
                    ? pipeline::RtcPairDecision::eligible
                    : pipeline::RtcPairDecision::ineligible;
                if (product.pair_decision(
                        span.network_id, row, detector) != expected_decision) {
                    ++metrics.pair_decision_mismatch_count;
                }

                ++metrics.pair_causal_evidence_comparison_count;
                const auto *actual_evidence = product.pair_causal_evidence(
                    span.network_id, row, detector);
                if (expected_decision == pipeline::RtcPairDecision::eligible) {
                    if (actual_evidence != nullptr) {
                        ++metrics.pair_causal_evidence_mismatch_count;
                    }
                    continue;
                }
                const auto expected_pair_causes = producer_pair_causes(
                    producer_member_causes(tune_ok, x_value),
                    producer_member_causes(tune_ok, r_value));
                if (actual_evidence == nullptr ||
                    actual_evidence->direct_x() != !x_valid ||
                    actual_evidence->direct_r() != !r_valid ||
                    evidence.scientific_identity(*actual_evidence) !=
                        expected_identity ||
                    evidence.member_local_causes(*actual_evidence) !=
                        expected_pair_causes) {
                    ++metrics.pair_causal_evidence_mismatch_count;
                }
            }
        }
    }
    require(view.detector_occurrence_count() ==
                metrics.identity_comparison_count,
            "native RTC view comparison cardinality drifted");
}

void compare_partitioned_to_single(const pipeline::RtcTimestream &partitioned,
                                   const pipeline::RtcTimestream &single,
                                   ComparisonMetrics &metrics) {
    ++metrics.chunk_realized_operator_comparison_count;
    if (partitioned.realized_operator() != single.realized_operator()) {
        ++metrics.chunk_realized_operator_mismatch_count;
    }
    const auto &partitioned_evidence =
        *partitioned.plan_handle()->evidence_handle();
    const auto &full_evidence =
        *single.plan_handle()->evidence_handle();
    for (const auto &span : partitioned.network_spans()) {
        const auto &network =
            partitioned.input_handle()->network(span.network_id);
        for (auto row = span.first_native_row;
             row < span.past_last_native_row; ++row) {
            if (std::bit_cast<std::uint64_t>(
                    partitioned.output_time_unix_sec(span.network_id, row)) !=
                    std::bit_cast<std::uint64_t>(
                        single.output_time_unix_sec(span.network_id, row)) ||
                !(partitioned.representative_native_identity(
                      span.network_id, row) ==
                  single.representative_native_identity(
                      span.network_id, row)) ||
                !(partitioned.representative_interval(
                      span.network_id, row) ==
                  single.representative_interval(span.network_id, row))) {
                ++metrics.chunk_scientific_mismatch_count;
            }
            for (Eigen::Index detector = 0;
                 detector < network.detector_count(); ++detector) {
                ++metrics.chunk_scientific_comparison_count;
                if (!(partitioned.identity(
                          span.network_id, row, detector) ==
                      single.identity(span.network_id, row, detector)) ||
                    partitioned.pair_decision(
                        span.network_id, row, detector) !=
                        single.pair_decision(
                            span.network_id, row, detector)) {
                    ++metrics.chunk_scientific_mismatch_count;
                }
                for (const auto member :
                     {pipeline::ReadoutMember::x,
                      pipeline::ReadoutMember::r}) {
                    if (std::bit_cast<std::uint64_t>(partitioned.value(
                            member, span.network_id, row, detector)) !=
                            std::bit_cast<std::uint64_t>(single.value(
                                member, span.network_id, row, detector)) ||
                        partitioned.member_local_causes(
                            member, span.network_id, row, detector) !=
                            single.member_local_causes(
                                member, span.network_id, row, detector)) {
                        ++metrics.chunk_scientific_mismatch_count;
                    }
                }
                const auto *partitioned_cause =
                    partitioned.pair_causal_evidence(
                        span.network_id, row, detector);
                const auto *full_cause = single.pair_causal_evidence(
                    span.network_id, row, detector);
                if ((partitioned_cause == nullptr) !=
                    (full_cause == nullptr)) {
                    ++metrics.chunk_scientific_mismatch_count;
                } else if (partitioned_cause != nullptr &&
                           (partitioned_cause->origin != full_cause->origin ||
                            partitioned_cause->evidence_class !=
                                full_cause->evidence_class ||
                            partitioned_evidence.scientific_identity(
                                *partitioned_cause) !=
                                full_evidence.scientific_identity(*full_cause) ||
                            partitioned_evidence.member_local_causes(
                                *partitioned_cause) !=
                                full_evidence.member_local_causes(*full_cause))) {
                    ++metrics.chunk_scientific_mismatch_count;
                }
            }
        }
    }
}

std::uint64_t peak_rss_bytes() {
    rusage usage{};
    require(getrusage(RUSAGE_SELF, &usage) == 0,
            "getrusage failed while recording peak RSS");
#if defined(__APPLE__)
    return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
    return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024U;
#endif
}

struct AcceptanceRun {
    std::int64_t observation = 0;
    pipeline::RtcOnlyTerminalResult terminal;
    pipeline::PairedReadoutCardinality native_cardinality;
    pipeline::PairedReadoutMemoryEvidence native_memory;
    ComparisonMetrics comparisons;
    std::string mapping_instance_id;
    double wall_time_sec = 0.0;
    double cpu_time_sec = 0.0;
    std::uint64_t peak_rss = 0;
    std::size_t chunk_partition_count = 0;
    bool product_inspected_in_memory = false;
    bool publication_complete = false;
};

AcceptanceRun execute_acceptance(
    const Arguments &arguments,
    const pipeline::CanonicalAptDetectorRelationV2 &relation,
    const std::vector<NetworkInput> &inputs,
    const RuntimeConfig &config,
    const NativeOccurrenceSupportAssignment &support_assignment) {
    const auto &observation = relation.observation();
    const pipeline::NativeObservationScope scope{
        observation.observation, observation.subobservation,
        observation.scan};
    const auto wall_begin = std::chrono::steady_clock::now();
    const auto cpu_begin = std::clock();

    auto paired_build = build_paired_readout(
        arguments, scope, relation, inputs, config, support_assignment);
    const auto native_cardinality = paired_build.paired->cardinality();
    const auto native_memory = paired_build.paired->memory_evidence();

    pipeline::RtcOnlyProductSlot full_publication;
    const auto full_spans =
        pipeline::full_native_occurrence_spans(*paired_build.paired);
    const auto full = run_route(
        1, paired_build.paired, full_spans, {full_spans},
        full_publication);
    const auto primary_cpu_end = std::clock();
    const auto primary_wall_end = std::chrono::steady_clock::now();
    require(full.complete() &&
                full_publication.snapshot() == full.published_product,
            "full identity RTC route did not publish exactly one completion");
    require(full.terminal.failure_cause ==
                    pipeline::RtcOnlyFailureCause::none &&
                full.terminal.failure_detail.empty(),
            "successful identity RTC route retained a failure cause");
    require(full.published_product->timestream_handle()
                    ->memory_evidence()
                    .owned_numeric_bytes == 0,
            "identity RTC product unexpectedly owns a numerical plane");
    const auto &op = full.published_product->timestream_handle()
                         ->realized_operator();
    require(op.sampling_factor == 1 && op.sampling_phase == 0 &&
                op.x_from_x == 1.0 && op.x_from_r == 0.0 &&
                op.r_from_x == 0.0 && op.r_from_r == 1.0,
            "realized RTC operator is not exact paired identity");

    auto comparisons = paired_build.ingress_comparisons;
    compare_full_product_to_producer_facts(
        *full.published_product->timestream_handle(), relation, inputs,
        comparisons);

    auto first_spans = full_spans;
    auto second_spans = full_spans;
    for (std::size_t index = 0; index < full_spans.size(); ++index) {
        const auto &span = full_spans[index];
        const auto midpoint = span.first_native_row +
            static_cast<pipeline::TimestreamNativeRow>(
                span.occurrence_count() / 2);
        require(midpoint > span.first_native_row &&
                    midpoint < span.past_last_native_row,
                "native occurrence span cannot be divided into two chunks");
        first_spans[index].past_last_native_row = midpoint;
        second_spans[index].first_native_row = midpoint;
    }
    pipeline::RtcOnlyProductSlot partitioned_publication;
    const auto partitioned = run_route(
        2, paired_build.paired, full_spans,
        {std::move(first_spans), std::move(second_spans)},
        partitioned_publication);
    const auto &partitioned_diagnostics = partitioned.terminal.diagnostics;
    require(partitioned.complete() &&
                partitioned_publication.snapshot() ==
                    partitioned.published_product &&
                partitioned_diagnostics.engineering_partition_count == 2 &&
                partitioned_diagnostics.native_admission_entry_count == 1 &&
                partitioned_diagnostics.learn_entry_count == 1 &&
                partitioned_diagnostics.consider_entry_count == 1 &&
                partitioned_diagnostics.apply_entry_count == 1 &&
                partitioned_diagnostics.publication_entry_count == 1,
            "two-chunk identity RTC route did not finalize one publication");
    compare_partitioned_to_single(
        *partitioned.published_product->timestream_handle(),
        *full.published_product->timestream_handle(), comparisons);

    auto incomplete_partition = full_spans;
    for (auto &span : incomplete_partition) {
        span.past_last_native_row = span.first_native_row +
            static_cast<pipeline::TimestreamNativeRow>(
                span.occurrence_count() / 2);
    }
    pipeline::RtcOnlyProductSlot incomplete_publication;
    const auto incomplete = run_route(
        3, paired_build.paired, full_spans, {incomplete_partition},
        incomplete_publication);
    require(!incomplete.complete() &&
                incomplete.terminal.failure_cause ==
                    pipeline::RtcOnlyFailureCause::incomplete_logical_support &&
                !incomplete_publication.snapshot(),
            "missing engineering chunk published a false completion");

    const auto second_publish = run_route(
        4, paired_build.paired, full_spans, {full_spans},
        full_publication);
    require(!second_publish.complete() &&
                second_publish.terminal.state ==
                    pipeline::RtcOnlyTerminalState::publication_failed &&
                second_publish.terminal.failure_cause ==
                    pipeline::RtcOnlyFailureCause::publication_slot_occupied &&
                full_publication.snapshot() == full.published_product,
            "second publication did not preserve the committed product");
    pipeline::RtcOnlyProductSlot failed_publication;
    auto invalid_spans = full_spans;
    ++invalid_spans.front().past_last_native_row;
    const auto failed = run_route(
        5, paired_build.paired, invalid_spans, {invalid_spans},
        failed_publication);
    require(!failed.complete() &&
                failed.terminal.failure_cause ==
                    pipeline::RtcOnlyFailureCause::input_contract_rejected &&
                !failed_publication.snapshot(),
            "failed route published a false completion");

    const auto wall_time =
        std::chrono::duration<double>(primary_wall_end - wall_begin).count();
    const auto cpu_time = static_cast<double>(primary_cpu_end - cpu_begin) /
                          static_cast<double>(CLOCKS_PER_SEC);
    require(wall_time > 0.0 && cpu_time > 0.0,
            "acceptance timing measurements are not positive");
    return {
        observation.observation, full.terminal, native_cardinality,
        native_memory, comparisons,
        std::move(paired_build.mapping_instance_id), wall_time, cpu_time,
        peak_rss_bytes(), 2, true, true};
}

void write_acceptance_record(const Arguments &arguments,
                             const NativeOccurrenceSupportAssignment
                                 &support_assignment,
                             const AcceptanceRun &run,
                             const LogCounts &logs) {
    require(run.comparisons.paired_ingress_value_comparison_count ==
                2 * run.native_cardinality.detector_occurrence_count,
            "paired ingress comparison count is incomplete");
    require(run.comparisons.paired_ingress_identity_comparison_count ==
                run.native_cardinality.detector_count,
            "paired ingress detector-identity comparison count is incomplete");
    require(run.comparisons.paired_ingress_member_state_comparison_count ==
                2 * run.native_cardinality.detector_occurrence_count,
            "paired ingress member-state comparison count is incomplete");
    require(run.comparisons.rtc_product_value_comparison_count ==
                2 * run.native_cardinality.detector_occurrence_count,
            "RTC product comparison count is incomplete");
    require(run.comparisons.identity_comparison_count ==
                run.native_cardinality.detector_occurrence_count,
            "identity comparison count is incomplete");
    require(run.comparisons.support_comparison_count ==
                run.native_cardinality.native_occurrence_count,
            "support comparison count is incomplete");
    require(run.comparisons.native_time_comparison_count ==
                run.native_cardinality.native_occurrence_count,
            "native-time comparison count is incomplete");
    require(run.comparisons.representative_native_comparison_count ==
                run.native_cardinality.native_occurrence_count,
            "representative-native comparison count is incomplete");
    require(run.comparisons.assigned_support_binding_count ==
                run.native_cardinality.native_occurrence_count,
            "assigned-support binding count is incomplete");
    require(run.comparisons.pair_decision_comparison_count ==
                run.native_cardinality.detector_occurrence_count,
            "pair-decision comparison count is incomplete");
    require(run.comparisons.pair_causal_evidence_comparison_count ==
                run.native_cardinality.detector_occurrence_count,
            "pair causal-evidence comparison count is incomplete");
    require(run.chunk_partition_count == 2 &&
                run.comparisons.chunk_realized_operator_comparison_count ==
                    1 &&
                run.comparisons.chunk_scientific_comparison_count ==
                    run.native_cardinality.detector_occurrence_count,
            "partitioned route did not compare the complete logical domain");
    require(run.comparisons.x_bitwise_mismatch_count == 0 &&
                run.comparisons.r_bitwise_mismatch_count == 0 &&
                run.comparisons.paired_ingress_identity_mismatch_count == 0 &&
                run.comparisons.paired_ingress_member_state_mismatch_count == 0 &&
                run.comparisons.identity_mismatch_count == 0 &&
                run.comparisons.support_mismatch_count == 0 &&
                run.comparisons.pair_decision_mismatch_count == 0 &&
                run.comparisons.pair_causal_evidence_mismatch_count == 0 &&
                run.comparisons.member_cause_mismatch_count == 0 &&
                run.comparisons.selected_time_mismatch_count == 0 &&
                run.comparisons.representative_native_mismatch_count == 0 &&
                run.comparisons.chunk_realized_operator_mismatch_count ==
                    0 &&
                run.comparisons.chunk_scientific_mismatch_count == 0 &&
                run.comparisons.assigned_support_binding_mismatch_count == 0,
            "acceptance comparisons contain a scientific mismatch");
    const auto &diagnostics = run.terminal.diagnostics;
    require(diagnostics.native_admission_entry_count == 1 &&
                diagnostics.learn_entry_count == 1 &&
                diagnostics.consider_entry_count == 1 &&
                diagnostics.apply_entry_count == 1 &&
                diagnostics.publication_entry_count == 1,
            "successful RTC-only route did not enter its exact allowed stages once");
    std::ofstream output(arguments.output, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output),
            "unable to create acceptance JSON: " + arguments.output.string());
    const auto q = [](std::string_view value) {
        return "\"" + json_escape(value) + "\"";
    };
    const auto b = [](bool value) { return value ? "true" : "false"; };
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": " << q(acceptance_schema) << ",\n"
           << "  \"source_revision\": " << q(arguments.source_revision)
           << ",\n"
           << "  \"executable_revision\": "
           << q(CITLALI_WP7_SOURCE_REVISION)
           << ",\n"
           << "  \"executable_version\": "
           << q(std::string{CITLALI_GIT_VERSION} + " kids=" +
                KIDSCPP_GIT_VERSION + " tula=" + TULA_GIT_VERSION)
           << ",\n"
           << "  \"citlali_source_clean\": true,\n"
           << "  \"citlali_ignored_source_state_verified\": true,\n"
           << "  \"executable_sha256\": "
           << q(arguments.executable_sha256) << ",\n"
           << "  \"dependency_state_verified\": true,\n"
           << "  \"kidscpp_revision\": "
           << q(CITLALI_WP7_KIDSCPP_REVISION)
           << ",\n"
           << "  \"kidscpp_build_patch_sha256\": "
           << q(kidscpp_patch_sha256) << ",\n"
           << "  \"kidscpp_tree\": "
           << q(CITLALI_WP7_KIDSCPP_TREE) << ",\n"
           << "  \"tula_revision\": "
           << q(CITLALI_WP7_TULA_REVISION) << ",\n"
           << "  \"tula_build_patch_sha256\": "
           << q(tula_patch_sha256) << ",\n"
           << "  \"tula_tree\": "
           << q(CITLALI_WP7_TULA_TREE) << ",\n"
           << "  \"design_commit\": " << q(design_commit) << ",\n"
           << "  \"align_repair_commit\": " << q(align_repair_commit)
           << ",\n"
           << "  \"design_is_ancestor\": "
           << b(arguments.design_is_ancestor) << ",\n"
           << "  \"align_repair_is_ancestor\": "
           << b(arguments.align_repair_is_ancestor) << ",\n"
           << "  \"owner_run\": " << b(arguments.owner_run) << ",\n"
           << "  \"real_paired_data\": true,\n"
           << "  \"apt_bundle_verified\": true,\n"
           << "  \"raw_sources_verified\": true,\n"
           << "  \"tune_bindings_verified\": true,\n"
           << "  \"tune_accumulation_explicit\": true,\n"
           << "  \"product_inspected_in_memory\": "
           << b(run.product_inspected_in_memory) << ",\n"
           << "  \"publication_complete\": "
           << b(run.publication_complete) << ",\n"
           << "  \"representative_dataset_id\": "
           << q(arguments.dataset_id) << ",\n"
           << "  \"observation\": " << run.observation << ",\n"
           << "  \"first_native_row\": "
           << arguments.first_native_row << ",\n"
           << "  \"native_row_count\": "
           << arguments.native_row_count << ",\n"
           << "  \"mapping_instance_id\": "
           << q(run.mapping_instance_id) << ",\n"
           << "  \"producer_interface_id\": " << q(producer_interface)
           << ",\n"
           << "  \"producer_interface_sha256\": "
           << q(producer_interface_sha256) << ",\n"
           << "  \"occurrence_support_assignment_schema\": "
           << q(occurrence_support_assignment_schema) << ",\n"
           << "  \"occurrence_support_assignment_id\": "
           << q(support_assignment.assignment_id) << ",\n"
           << "  \"occurrence_support_assignment_sha256\": "
           << q(support_assignment.artifact_sha256) << ",\n"
           << "  \"occurrence_support_assignment_status\": "
           << q(occurrence_support_assignment_status) << ",\n"
           << "  \"occurrence_support_assigned_by\": "
           << q(support_assignment.assigned_by) << ",\n"
           << "  \"occurrence_support_assigned_at_utc\": "
           << q(support_assignment.assigned_at_utc) << ",\n"
           << "  \"occurrence_support_calibration_pending\": true,\n"
           << "  \"occurrence_support_calibration_disposition\": "
           << q(occurrence_support_calibration_disposition) << ",\n"
           << "  \"occurrence_support_event_time_role\": "
           << q(native_event_time_role_name(
                    support_assignment.event_time_role)) << ",\n"
           << "  \"occurrence_support_duration_relation\": "
           << q(occurrence_support_duration_relation) << ",\n"
           << "  \"terminal_state\": "
           << q(pipeline::rtc_only_terminal_state_name(run.terminal.state))
           << ",\n"
           << "  \"terminal_failure_cause\": "
           << q(pipeline::rtc_only_failure_cause_name(
                    run.terminal.failure_cause)) << ",\n"
           << "  \"terminal_failure_detail\": "
           << q(run.terminal.failure_detail) << ",\n"
           << "  \"metrics\": {\n"
           << "    \"network_count\": "
           << run.terminal.diagnostics.network_count << ",\n"
           << "    \"detector_count\": "
           << run.terminal.diagnostics.detector_count << ",\n"
           << "    \"native_occurrence_count\": "
           << run.native_cardinality.native_occurrence_count << ",\n"
           << "    \"native_detector_occurrence_count\": "
           << run.native_cardinality.detector_occurrence_count << ",\n"
           << "    \"paired_numeric_payload_bytes\": "
           << run.native_memory.numeric_payload_bytes << ",\n"
           << "    \"paired_member_state_bytes\": "
           << run.native_memory.member_state_bytes << ",\n"
           << "    \"paired_occurrence_interval_bytes\": "
           << run.native_memory.occurrence_interval_bytes << ",\n"
           << "    \"paired_detector_axis_bytes\": "
           << run.native_memory.detector_axis_bytes << ",\n"
           << "    \"paired_identity_text_bytes\": "
           << run.native_memory.identity_text_bytes << ",\n"
           << "    \"paired_logical_owned_bytes\": "
           << run.native_memory.logical_owned_bytes() << ",\n"
           << "    \"referenced_native_axis_count\": "
           << run.native_memory.referenced_native_axis_count << ",\n"
           << "    \"rtc_native_occurrence_count\": "
           << run.terminal.diagnostics.native_occurrence_count << ",\n"
           << "    \"rtc_detector_occurrence_count\": "
           << run.terminal.diagnostics.detector_occurrence_count << ",\n"
           << "    \"evidence_event_count\": "
           << run.terminal.diagnostics.evidence_event_count << ",\n"
           << "    \"direct_x_event_count\": "
           << run.terminal.diagnostics.direct_x_event_count << ",\n"
           << "    \"direct_r_event_count\": "
           << run.terminal.diagnostics.direct_r_event_count << ",\n"
           << "    \"x_and_r_event_count\": "
           << run.terminal.diagnostics.x_and_r_event_count << ",\n"
           << "    \"pair_ineligible_cell_count\": "
           << run.terminal.diagnostics.pair_ineligible_cell_count << ",\n"
           << "    \"x_numerically_valid_cell_count\": "
           << run.terminal.diagnostics.x_numerically_valid_cell_count
           << ",\n"
           << "    \"r_numerically_valid_cell_count\": "
           << run.terminal.diagnostics.r_numerically_valid_cell_count
           << ",\n"
           << "    \"derived_evidence_bytes\": "
           << run.terminal.diagnostics.derived_evidence_bytes << ",\n"
           << "    \"derived_plan_bytes\": "
           << run.terminal.diagnostics.derived_plan_bytes << ",\n"
           << "    \"paired_ingress_value_comparison_count\": "
           << run.comparisons.paired_ingress_value_comparison_count << ",\n"
           << "    \"paired_ingress_identity_comparison_count\": "
           << run.comparisons.paired_ingress_identity_comparison_count
           << ",\n"
           << "    \"paired_ingress_member_state_comparison_count\": "
           << run.comparisons.paired_ingress_member_state_comparison_count
           << ",\n"
           << "    \"rtc_product_value_comparison_count\": "
           << run.comparisons.rtc_product_value_comparison_count << ",\n"
           << "    \"identity_comparison_count\": "
           << run.comparisons.identity_comparison_count << ",\n"
           << "    \"support_comparison_count\": "
           << run.comparisons.support_comparison_count << ",\n"
           << "    \"native_time_comparison_count\": "
           << run.comparisons.native_time_comparison_count << ",\n"
           << "    \"representative_native_comparison_count\": "
           << run.comparisons.representative_native_comparison_count << ",\n"
           << "    \"assigned_support_binding_count\": "
           << run.comparisons.assigned_support_binding_count << ",\n"
           << "    \"pair_decision_comparison_count\": "
           << run.comparisons.pair_decision_comparison_count << ",\n"
           << "    \"pair_causal_evidence_comparison_count\": "
           << run.comparisons.pair_causal_evidence_comparison_count << ",\n"
           << "    \"chunk_partition_count\": "
           << run.chunk_partition_count << ",\n"
           << "    \"chunk_realized_operator_comparison_count\": "
           << run.comparisons.chunk_realized_operator_comparison_count
           << ",\n"
           << "    \"chunk_scientific_comparison_count\": "
           << run.comparisons.chunk_scientific_comparison_count << ",\n"
           << "    \"wall_time_sec\": " << run.wall_time_sec << ",\n"
           << "    \"cpu_time_sec\": " << run.cpu_time_sec << ",\n"
           << "    \"process_peak_rss_bytes\": " << run.peak_rss << ",\n"
           << "    \"rtc_owned_numeric_bytes\": "
           << run.terminal.diagnostics.rtc_owned_numeric_bytes << ",\n"
           << "    \"x_bitwise_mismatch_count\": "
           << run.comparisons.x_bitwise_mismatch_count << ",\n"
           << "    \"r_bitwise_mismatch_count\": "
           << run.comparisons.r_bitwise_mismatch_count << ",\n"
           << "    \"paired_ingress_identity_mismatch_count\": "
           << run.comparisons.paired_ingress_identity_mismatch_count
           << ",\n"
           << "    \"paired_ingress_member_state_mismatch_count\": "
           << run.comparisons.paired_ingress_member_state_mismatch_count
           << ",\n"
           << "    \"identity_mismatch_count\": "
           << run.comparisons.identity_mismatch_count << ",\n"
           << "    \"support_mismatch_count\": "
           << run.comparisons.support_mismatch_count << ",\n"
           << "    \"assigned_support_binding_mismatch_count\": "
           << run.comparisons.assigned_support_binding_mismatch_count
           << ",\n"
           << "    \"pair_decision_mismatch_count\": "
           << run.comparisons.pair_decision_mismatch_count << ",\n"
           << "    \"pair_causal_evidence_mismatch_count\": "
           << run.comparisons.pair_causal_evidence_mismatch_count << ",\n"
           << "    \"member_cause_mismatch_count\": "
           << run.comparisons.member_cause_mismatch_count << ",\n"
           << "    \"chunk_realized_operator_mismatch_count\": "
           << run.comparisons.chunk_realized_operator_mismatch_count
           << ",\n"
           << "    \"chunk_scientific_mismatch_count\": "
           << run.comparisons.chunk_scientific_mismatch_count << ",\n"
           << "    \"selected_time_mismatch_count\": "
           << run.comparisons.selected_time_mismatch_count << ",\n"
           << "    \"representative_native_mismatch_count\": "
           << run.comparisons.representative_native_mismatch_count << ",\n"
           << "    \"native_admission_entry_count\": "
           << run.terminal.diagnostics.native_admission_entry_count << ",\n"
           << "    \"learn_entry_count\": "
           << run.terminal.diagnostics.learn_entry_count << ",\n"
           << "    \"consider_entry_count\": "
           << run.terminal.diagnostics.consider_entry_count << ",\n"
           << "    \"apply_entry_count\": "
           << run.terminal.diagnostics.apply_entry_count << ",\n"
           << "    \"publication_entry_count\": "
           << run.terminal.diagnostics.publication_entry_count << ",\n"
           << "    \"unexpected_error_count\": " << logs.errors << ",\n"
           << "    \"unexpected_critical_count\": " << logs.criticals
           << "\n"
           << "  }\n"
           << "}\n";
    require(static_cast<bool>(output), "failed to write acceptance JSON");
}

}  // namespace

int main(int argc, char **argv) {
    try {
        auto arguments = parse_arguments(argc, argv);
        require(full_lowercase_git_sha(arguments.source_revision),
                "source revision must be one full lowercase Git SHA");
        require(CITLALI_WP7_SOURCE_STATE_VERIFIED == 1 &&
                    CITLALI_WP7_IGNORED_SOURCE_STATE_VERIFIED == 1 &&
                    arguments.source_revision == CITLALI_WP7_SOURCE_REVISION,
                "source revision does not match the compiled Citlali revision");
        require(arguments.owner_run, "owner-run authorization is required");
        require(arguments.design_is_ancestor &&
                    arguments.align_repair_is_ancestor,
                "accepted design and ALIGN repair ancestry are required");
        require(fs::is_directory(arguments.data_directory),
                "data directory does not exist");
        require(fs::absolute(arguments.apt_manifest) ==
                    arguments.apt_manifest,
                "APT manifest path must be absolute");
        require(fs::is_regular_file(arguments.producer_interface_artifact),
                "producer interface artifact is not a regular file");
        require(citlali::utils::sha256_file(
                    arguments.producer_interface_artifact) ==
                    producer_interface_sha256,
                "producer interface artifact SHA-256 is not approved");
        const auto support_assignment =
            load_occurrence_support_assignment(
                arguments.occurrence_support_assignment_artifact);
        require(CITLALI_WP7_DEPENDENCY_STATE_VERIFIED == 1 &&
                    std::string_view{CITLALI_WP7_KIDSCPP_REVISION} ==
                        kidscpp_revision &&
                    std::string_view{CITLALI_WP7_TULA_REVISION} ==
                        tula_revision,
                "compiled numerical dependency state is not approved");
        require(fs::is_regular_file(arguments.kidscpp_build_patch) &&
                    citlali::utils::sha256_file(
                        arguments.kidscpp_build_patch) ==
                        kidscpp_patch_sha256,
                "Kidscpp local build patch SHA-256 is not approved");
        require(fs::is_regular_file(arguments.tula_build_patch) &&
                    citlali::utils::sha256_file(
                        arguments.tula_build_patch) == tula_patch_sha256,
                "Tula local build patch SHA-256 is not approved");
        require(fs::is_regular_file(arguments.executable),
                "acceptance executable path is not a regular file");
        arguments.executable_sha256 =
            citlali::utils::sha256_file(arguments.executable);

        auto [logger, log_counts] = configure_logging();
        const auto verified = apt::verify_bundle_filesystem(
            arguments.apt_manifest, true);
        const auto relation =
            pipeline::admit_canonical_apt_detector_relation_v2(verified);
        require(relation.observation().observation == 152390,
                "this bounded acceptance invocation requires observation 152390");
        const auto config = load_runtime_config(arguments.config);
        const auto inputs = resolve_network_inputs(
            arguments, relation, config);
        (void)logger;
        const auto run = execute_acceptance(
            arguments, relation, inputs, config, support_assignment);
        require(log_counts->errors == 0 && log_counts->criticals == 0,
                "acceptance route emitted unexpected error-level records");
        write_acceptance_record(
            arguments, support_assignment, run, *log_counts);
        std::cout << "WP-7 identity RTC acceptance record: "
                  << arguments.output << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "WP-7 identity RTC acceptance runner: FAIL: "
                  << error.what() << '\n';
        return 2;
    }
}
