#include <citlali/core/pipeline/canonical_apt_bundle_v2.h>
#include <citlali/core/pipeline/canonical_apt_detector_relation_v2.h>
#include <citlali/core/pipeline/observation_setup_validation.h>
#include <citlali/core/pipeline/timestream_native_paired_readout_kids_adapter.h>
#include <citlali/core/pipeline/timestream_identity_route_context.h>
#include <citlali/core/utils/sha256.h>

#include <citlali_config/gitversion.h>
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
#include <numbers>
#include <numeric>
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
    "citlali-timestream-successor-identity-route-acceptance-v2";
constexpr std::string_view subject_candidate_revision =
    "b57d9f606549d524ab6bb61faf0cd3d52ac27db6";
constexpr std::string_view subject_candidate_tree =
    "32de9791255c6c52032c0f05d64054b83ff44de5";
constexpr std::string_view representative_dataset =
    "SCI_ALIGN_STAGE7_NGC4449_152390";
constexpr std::string_view telescope_filename =
    "tel_toltec_2026-02-19_152390_00_0002.nc";
constexpr std::string_view telescope_sha256 =
    "2845455a620635955c00a4731e0d9720cfa456fece79d1729cf755a366a1ad6b";
constexpr std::uintmax_t telescope_byte_count = 24157872;
constexpr std::size_t telescope_record_count = 62109;
constexpr pipeline::AstScanMotionIdentityBinding ast_identity_binding{
    1523900001, 1523900002, 1523900003, 1523900004};
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
struct Arguments {
    fs::path data_directory;
    fs::path telescope;
    fs::path apt_manifest;
    fs::path config;
    fs::path producer_interface_artifact;
    fs::path occurrence_support_assignment_artifact;
    fs::path spack_environment;
    fs::path spack_lock;
    fs::path output;
    fs::path executable;
    std::string executable_sha256;
    std::string source_revision;
    std::string build_environment;
    std::string build_profile;
    std::string spack_root_dag;
    std::string dataset_id = std::string{representative_dataset};
    std::int64_t first_native_row = 20000;
    std::int64_t native_row_count = 2048;
    bool owner_run = false;
};

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

std::string usage() {
    return
        "Usage: citlali_timestream_successor_identity_acceptance\n"
        "  --data-dir PATH --telescope PATH --apt-manifest PATH --config PATH\n"
        "  --producer-interface-artifact PATH\n"
        "  --occurrence-support-assignment PATH\n"
        "  --output PATH\n"
        "  --source-revision FULL_SHA --owner-run\n"
        "  --build-environment spack --build-profile unity-gcc13\n"
        "  --spack-environment PATH --spack-lock PATH --spack-root-dag HASH\n"
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
        } else if (option == "--telescope") {
            result.telescope = next(index, option);
        } else if (option == "--apt-manifest") {
            result.apt_manifest = next(index, option);
        } else if (option == "--config") {
            result.config = next(index, option);
        } else if (option == "--producer-interface-artifact") {
            result.producer_interface_artifact = next(index, option);
        } else if (option == "--occurrence-support-assignment") {
            result.occurrence_support_assignment_artifact =
                next(index, option);
        } else if (option == "--build-environment") {
            result.build_environment = next(index, option);
        } else if (option == "--build-profile") {
            result.build_profile = next(index, option);
        } else if (option == "--spack-environment") {
            result.spack_environment = next(index, option);
        } else if (option == "--spack-lock") {
            result.spack_lock = next(index, option);
        } else if (option == "--spack-root-dag") {
            result.spack_root_dag = next(index, option);
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
        } else {
            fail("unknown option: " + option);
        }
    }
    require(!result.data_directory.empty(), "--data-dir is required");
    require(!result.telescope.empty(), "--telescope is required");
    require(!result.apt_manifest.empty(), "--apt-manifest is required");
    require(!result.config.empty(), "--config is required");
    require(!result.producer_interface_artifact.empty(),
            "--producer-interface-artifact is required");
    require(!result.occurrence_support_assignment_artifact.empty(),
            "--occurrence-support-assignment is required");
    require(!result.output.empty(), "--output is required");
    require(!result.source_revision.empty(),
            "--source-revision is required");
    require(result.build_environment == "spack",
            "authoritative acceptance requires --build-environment spack");
    require(result.build_profile == "unity-gcc13",
            "authoritative acceptance requires --build-profile unity-gcc13");
    require(!result.spack_environment.empty(),
            "--spack-environment is required");
    require(!result.spack_lock.empty(), "--spack-lock is required");
    require(!result.spack_root_dag.empty(),
            "--spack-root-dag is required");
    require(result.first_native_row >= 0,
            "--first-native-row must be nonnegative");
    require(result.dataset_id == representative_dataset &&
                result.first_native_row == 20000 &&
                result.native_row_count == 2048,
            "bounded acceptance requires the approved dataset and exact 2048-row slice at row 20000");
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

struct TelescopeInput {
    std::shared_ptr<const pipeline::AstScanMotionSource> source;
    std::string sha256;
    std::uintmax_t byte_count = 0;
};

TelescopeInput load_telescope(
    const fs::path &path,
    const pipeline::NativeObservationScope &scope) {
    require(path.filename() == telescope_filename,
            "telescope filename is not the approved artifact");
    std::error_code error;
    const auto byte_count = fs::file_size(path, error);
    require(!error && byte_count == telescope_byte_count,
            "telescope byte count is not approved");
    const auto digest = citlali::utils::sha256_file(path);
    require(digest == telescope_sha256,
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
    require(!time_variable.isNull() &&
                time_variable.getDimCount() == 1 &&
                time_variable.getDim(0).getSize() ==
                    telescope_record_count,
            "telescope time cardinality is not approved");
    auto times = read_vector(
        file, std::string{pipeline::ast_scan_motion_time_field},
        telescope_record_count, "sec");
    auto ra = read_vector(
        file, std::string{pipeline::ast_scan_motion_ra_field},
        telescope_record_count, "rad");
    auto dec = read_vector(
        file, std::string{pipeline::ast_scan_motion_dec_field},
        telescope_record_count, "rad");
    auto source = pipeline::AstScanMotionSource::admit(
        scope, scope, 0, std::move(metadata), std::move(times),
        std::move(ra), std::move(dec));
    return {std::move(source), digest, byte_count};
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
        "observation=152390:network=" +
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
    std::size_t route_occurrence_binding_count = 0;
    std::size_t route_occurrence_binding_mismatch_count = 0;
    std::size_t ast_mapped_occurrence_count = 0;
    std::size_t ast_available_occurrence_count = 0;
    std::size_t ast_unavailable_occurrence_count = 0;
    std::size_t ast_support_count = 0;
    std::size_t ast_identity_mismatch_count = 0;
    std::size_t ast_support_mismatch_count = 0;
    std::size_t val_binding_comparison_count = 0;
    std::size_t val_binding_mismatch_count = 0;
};

pipeline::NativeReadoutIntegrationSupport assigned_occurrence_interval(
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
    const pipeline::NativePairedReadoutOccurrenceAxis &axis,
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
        if (!(axis.occurrence(row).integration_support == expected)) {
            ++metrics.assigned_support_binding_mismatch_count;
        }
    }
}

struct NativeValueOracle {
    std::vector<pipeline::NativeReadoutDetectorBinding> detectors;
    std::vector<std::tuple<pipeline::TimestreamNativeRow, Eigen::Index,
                           std::uint64_t, std::uint64_t>> cells;
};

pipeline::NativeReadoutDetectorBinding producer_detector_identity(
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
    const auto &relation_id = relation.relation_identity();
    const auto &target = found->target;
    return {
        static_cast<pipeline::TimestreamNetworkId>(found->network),
        static_cast<Eigen::Index>(found->channel),
        target.artifact.schema + ":" + target.artifact.occurrence +
            ":semantic=" + target.artifact.semantic_sha256 +
            ":envelope=" + target.artifact.envelope_sha256 +
            ":row=" + std::to_string(target.local_uid),
        relation_id.schema + ":" + relation_id.occurrence +
            ":semantic=" + relation_id.semantic_sha256 +
            ":envelope=" + relation_id.envelope_sha256 +
            ":relation=" + std::to_string(found->relation_uid),
        "sha256:" + input.source.content_sha256 + ":network=" +
            std::to_string(found->network) + ":channel=" +
            std::to_string(found->channel)};
}

bool same_detector_binding(
    const pipeline::NativeReadoutDetectorBinding &lhs,
    const pipeline::NativeReadoutDetectorBinding &rhs) {
    return lhs.network_id == rhs.network_id &&
           lhs.storage_column == rhs.storage_column &&
           lhs.detector_occurrence_id == rhs.detector_occurrence_id &&
           lhs.detector_association_record_id ==
               rhs.detector_association_record_id &&
           lhs.tone_or_channel_id == rhs.tone_or_channel_id;
}

pipeline::NativeReadoutCoordinateCause producer_member_causes(
    bool tune_valid, double value) {
    const bool finite = std::isfinite(value);
    auto causes = pipeline::NativeReadoutCoordinateCause::none;
    if (!(tune_valid && finite)) {
        causes = causes | pipeline::NativeReadoutCoordinateCause::producer_invalid;
    }
    if (!finite) {
        causes = causes | pipeline::NativeReadoutCoordinateCause::nonfinite_payload;
    }
    return causes;
}

bool producer_member_valid(bool tune_valid, double value) {
    return tune_valid && std::isfinite(value);
}

pipeline::NativePairedReadoutCause producer_pair_causes(
    pipeline::NativeReadoutCoordinateCause x_causes,
    pipeline::NativeReadoutCoordinateCause r_causes) {
    auto causes = pipeline::NativePairedReadoutCause::none;
    if (pipeline::has_cause(
            x_causes, pipeline::NativeReadoutCoordinateCause::producer_invalid)) {
        causes = causes |
            pipeline::NativePairedReadoutCause::x_producer_invalid;
    }
    if (pipeline::has_cause(
            r_causes, pipeline::NativeReadoutCoordinateCause::producer_invalid)) {
        causes = causes |
            pipeline::NativePairedReadoutCause::r_producer_invalid;
    }
    if (pipeline::has_cause(
            x_causes, pipeline::NativeReadoutCoordinateCause::nonfinite_payload)) {
        causes = causes | pipeline::NativePairedReadoutCause::x_nonfinite;
    }
    if (pipeline::has_cause(
            r_causes, pipeline::NativeReadoutCoordinateCause::nonfinite_payload)) {
        causes = causes | pipeline::NativePairedReadoutCause::r_nonfinite;
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
    const pipeline::NativePairedReadoutNetwork &network,
    const std::vector<bool> &tune_valid,
    ComparisonMetrics &metrics) {
    require(oracle.detectors.size() ==
                static_cast<std::size_t>(network.detector_count()) &&
            tune_valid.size() == oracle.detectors.size(),
            "producer oracle does not cover the admitted detector axis");
    for (Eigen::Index detector = 0;
         detector < network.detector_count(); ++detector) {
        ++metrics.paired_ingress_identity_comparison_count;
        if (!same_detector_binding(
                network.detector(detector),
                oracle.detectors[static_cast<std::size_t>(detector)])) {
            ++metrics.paired_ingress_identity_mismatch_count;
        }
    }
    for (const auto &[row, detector, x_bits, r_bits] : oracle.cells) {
        ++metrics.paired_ingress_value_comparison_count;
        if (std::bit_cast<std::uint64_t>(network.value(
                pipeline::NativeReadoutCoordinate::x, row, detector)) != x_bits) {
            ++metrics.x_bitwise_mismatch_count;
        }
        ++metrics.paired_ingress_value_comparison_count;
        if (std::bit_cast<std::uint64_t>(network.value(
                pipeline::NativeReadoutCoordinate::r, row, detector)) != r_bits) {
            ++metrics.r_bitwise_mismatch_count;
        }
        const auto tune_ok = tune_valid.at(
            static_cast<std::size_t>(detector));
        for (const auto [member, value_bits] :
             {std::pair{pipeline::NativeReadoutCoordinate::x, x_bits},
              std::pair{pipeline::NativeReadoutCoordinate::r, r_bits}}) {
            ++metrics.paired_ingress_member_state_comparison_count;
            const auto value = std::bit_cast<double>(value_bits);
            const auto &state = network.state(member, row, detector);
            const auto expected_valid = producer_member_valid(tune_ok, value);
            const auto expected_causes = producer_member_causes(tune_ok, value);
            if (!state.payload_available() ||
                !state.in_acquisition_support() ||
                state.origin() != pipeline::NativeReadoutOrigin::measured ||
                state.producer_valid() != expected_valid ||
                state.valid() != expected_valid ||
                state.causes() != expected_causes) {
                ++metrics.paired_ingress_member_state_mismatch_count;
            }
        }
    }
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

struct PairedBuildResult {
    std::shared_ptr<const pipeline::NativePairedReadoutObservation> paired;
    ComparisonMetrics ingress_comparisons;
    std::string mapping_instance_id;
};

class TemporaryDirectory {
public:
    TemporaryDirectory() {
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
    std::vector<pipeline::NativePairedReadoutNetwork> networks;
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
        mapping_digest.update(mapping->mapping_revision_id);
        auto axis = occurrence_axis(
            input, arguments.first_native_row,
            arguments.native_row_count, support_assignment.event_time_role);
        compare_occurrence_axis_to_assignment(
            *axis, input, support_assignment.event_time_role, metrics);
        pipeline::NativePairedReadoutNetworkIngress ingress{
            std::move(axis),
            detector_axis(relation, input), mapping,
            member_states(solved.data_out.xs.data, input.tune_valid),
            member_states(solved.data_out.rs.data, input.tune_valid)};
        auto network = pipeline::take_native_paired_kids_solver_result(
            std::move(ingress), std::move(solved));
        compare_moved_native_ingress(
            oracle, network, input.tune_valid, metrics);
        networks.push_back(std::move(network));
    }
    auto admitted = pipeline::NativePairedReadoutObservation::admit(
        scope, std::move(inventory), std::move(networks));
    return {
        std::make_shared<const pipeline::NativePairedReadoutObservation>(
            std::move(admitted)),
        metrics, "sha256:" + mapping_digest.finish()};
}

pipeline::IdentityRouteContextOutcome run_route(
    std::uint64_t run,
    const std::shared_ptr<const pipeline::IdentityRouteAlignContext>
        &align_context,
    std::vector<pipeline::NativeOccurrenceSpan> logical_spans,
    std::vector<std::vector<pipeline::NativeOccurrenceSpan>>
        engineering_partitions,
    pipeline::RtcOnlyProductSlot &publication) {
    const auto &paired = align_context->paired_handle();
    std::size_t completed_occurrences = 0;
    std::size_t completed_cells = 0;
    for (const auto &span : logical_spans) {
        const auto occurrences = span.occurrence_count();
        const auto detectors = static_cast<std::size_t>(
            paired->network(span.network_id).detector_count());
        completed_occurrences += occurrences;
        completed_cells += occurrences * detectors;
    }
    pipeline::RtcOnlyLogicalFinalization finalization{
        run, {run}, paired, completed_occurrences, completed_cells, true};
    pipeline::RtcOnlyRouteRequest rtc{
        {run}, paired, align_context->val_snapshot_handle(),
        std::move(logical_spans), std::move(engineering_partitions),
        std::move(finalization)};
    return pipeline::run_identity_route_context(
        {align_context, std::move(rtc)}, publication);
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
        std::vector<pipeline::NativeReadoutDetectorBinding>
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
            if (!(product.integration_support(span.network_id, row) ==
                  expected_interval)) {
                ++metrics.support_mismatch_count;
            }

            for (Eigen::Index detector = 0;
                 detector < native.detector_count(); ++detector) {
                ++metrics.identity_comparison_count;
                const auto &expected_detector = expected_detectors.at(
                    static_cast<std::size_t>(detector));
                const auto &occurrence =
                    native.occurrence_axis().occurrence(row);
                const auto &mapping = native.mapping_authority();
                const pipeline::RtcNativeCellIdentity expected_identity{
                    span.network_id,
                    row,
                    expected_detector.storage_column,
                    occurrence.parent_readout_occurrence_key,
                    occurrence.paired_xr_occurrence_key,
                    expected_detector.detector_occurrence_id,
                    expected_detector.detector_association_record_id,
                    expected_detector.tone_or_channel_id,
                    mapping.mapping_record_id,
                    mapping.mapping_revision_id};
                if (!(product.identity(span.network_id, row, detector) ==
                      expected_identity)) {
                    ++metrics.identity_mismatch_count;
                }

                for (const auto member :
                     {pipeline::NativeReadoutCoordinate::x,
                      pipeline::NativeReadoutCoordinate::r}) {
                    ++metrics.rtc_product_value_comparison_count;
                    const auto actual = product.value(
                        member, span.network_id, row, detector);
                    const auto expected = native.value(
                        member, row, detector);
                    if (std::bit_cast<std::uint64_t>(actual) !=
                        std::bit_cast<std::uint64_t>(expected)) {
                        if (member == pipeline::NativeReadoutCoordinate::x) {
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
                    pipeline::NativeReadoutCoordinate::x,
                    span.network_id, row, detector);
                const auto r_value = product.value(
                    pipeline::NativeReadoutCoordinate::r,
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
                    evidence.pair_local_causes(*actual_evidence) !=
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
                !(partitioned.integration_support(
                      span.network_id, row) ==
                  single.integration_support(span.network_id, row))) {
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
                     {pipeline::NativeReadoutCoordinate::x,
                      pipeline::NativeReadoutCoordinate::r}) {
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
                            partitioned_evidence.pair_local_causes(
                                *partitioned_cause) !=
                                full_evidence.pair_local_causes(*full_cause))) {
                    ++metrics.chunk_scientific_mismatch_count;
                }
            }
        }
    }
}

struct TypedRouteEvidence {
    std::size_t ast_raw_record_count = 0;
    std::size_t ast_raw_owned_bytes = 0;
    std::size_t ast_mapped_owned_bytes = 0;
    std::size_t align_owned_bytes = 0;
    std::size_t rtc_input_owned_bytes = 0;
    std::size_t rtc_output_owned_bytes = 0;
    std::size_t val_owned_bytes = 0;
    std::uint64_t val_generation = 0;
    std::size_t val_finding_count = 0;
    bool ast_present = false;
    bool ast_dependency_not_applicable = false;
    bool val_exact_snapshot_bound = false;
    bool calibration_unavailable = false;
    bool calibration_val_evaluation_unavailable = false;
    bool ptc_unavailable = false;
    bool ptc_val_evaluation_unavailable = false;
    bool map_admission_unavailable = false;
    bool map_action_performed = false;
};

TypedRouteEvidence inspect_typed_route(
    const pipeline::IdentityRouteContextOutcome &outcome,
    const std::shared_ptr<const pipeline::IdentityRouteAlignContext>
        &align_context,
    ComparisonMetrics &metrics) {
    require(outcome.map_facing_context_complete() &&
                outcome.failure_cause ==
                    pipeline::IdentityRouteContextFailureCause::none &&
                outcome.failure_detail.empty(),
            "typed identity route did not reach its MAP-facing context");
    const auto &bundle = outcome.map_facing_bundle;
    const auto &output = bundle->rtc_context_handle();
    const auto &input = output->input_context_handle();
    const auto &terminal = output->rtc_terminal_handle();
    const auto &product = terminal->timestream_handle();
    const auto &plan = terminal->plan_handle();
    const auto &evidence = terminal->evidence_handle();
    const auto &val = align_context->val_snapshot_handle();
    const auto &paired = align_context->paired_handle();
    const auto &ast = align_context->ast_views_handle();

    TypedRouteEvidence result;
    result.ast_present = ast && input->ast_views_handle().get() == ast.get() &&
        output->ast_views_handle().get() == ast.get();
    result.ast_dependency_not_applicable =
        input->ast_dependency() ==
        pipeline::IdentityRtcAstDependency::not_applicable;

    const std::array<const pipeline::ValSnapshot *, 8> val_bindings{
        align_context->val_snapshot_handle().get(),
        input->val_snapshot_handle().get(),
        output->val_snapshot_handle().get(),
        bundle->val_snapshot_handle().get(),
        terminal->val_snapshot_handle().get(),
        product->val_snapshot_handle().get(),
        plan->val_snapshot_handle().get(),
        evidence->val_snapshot_handle().get()};
    for (const auto *binding : val_bindings) {
        ++metrics.val_binding_comparison_count;
        metrics.val_binding_mismatch_count += binding != val.get();
    }
    result.val_generation = val->generation().value;
    result.val_finding_count = val->committed_delta_findings().size();
    result.val_owned_bytes = val->memory_evidence().logical_owned_bytes();
    result.val_exact_snapshot_bound =
        val->paired_handle().get() == paired.get() &&
        val->parent_snapshot_handle() == nullptr &&
        result.val_generation == 0 && result.val_finding_count == 0 &&
        outcome.rtc_terminal.diagnostics.val_generation ==
            val->generation() &&
        metrics.val_binding_mismatch_count == 0;

    const auto &cal = bundle->calibration_state();
    result.calibration_unavailable =
        cal.rtc_context_handle().get() == output.get() &&
        cal.product_state() == pipeline::IdentityCalibrationProductState::
            unavailable_component_not_admitted &&
        cal.unit_state() == pipeline::IdentityCalibrationUnitState::
            unavailable_no_calibration_product &&
        cal.response_state() == pipeline::IdentityCalibrationResponseState::
            unavailable_no_calibration_product &&
        cal.uncertainty_state() ==
            pipeline::IdentityCalibrationUncertaintyState::
                unavailable_no_calibration_product;
    result.calibration_val_evaluation_unavailable =
        bundle->calibration_for_ptc_val_evaluation()
                .rtc_context_handle()
                .get() == output.get() &&
        bundle->calibration_for_ptc_val_evaluation().state() ==
        pipeline::IdentityCalibrationForPtcValEvaluationState::
            unavailable_calibration_product_absent;
    const auto &ptc = bundle->ptc_state();
    result.ptc_unavailable =
        ptc.rtc_context_handle().get() == output.get() &&
        ptc.product_state() == pipeline::IdentityPtcProductState::
            unavailable_component_not_admitted &&
        ptc.conditioning_state() == pipeline::IdentityPtcConditioningState::
            unavailable_no_ptc_product &&
        ptc.response_state() == pipeline::IdentityPtcResponseState::
            unavailable_no_ptc_product &&
        ptc.uncertainty_state() == pipeline::IdentityPtcUncertaintyState::
            unavailable_no_ptc_product;
    result.ptc_val_evaluation_unavailable =
        bundle->ptc_for_map_val_evaluation()
                .rtc_context_handle()
                .get() == output.get() &&
        bundle->ptc_for_map_val_evaluation().state() ==
        pipeline::IdentityPtcForMapValEvaluationState::
            unavailable_ptc_product_absent;
    result.map_admission_unavailable =
        bundle->map_admission_state() ==
        pipeline::IdentityMapAdmissionState::
            unavailable_calibration_and_ptc_products;
    result.map_action_performed = bundle->map_action_performed();

    for (const auto &span : product->network_spans()) {
        const auto &network = paired->network(span.network_id);
        const auto &axis = network.occurrence_axis();
        const auto &mapped = ast->network(span.network_id);
        result.ast_mapped_owned_bytes +=
            mapped.memory_evidence().logical_owned_bytes();
        for (auto row = span.first_native_row;
             row < span.past_last_native_row; ++row) {
            const auto expected_identity = axis.native_identity(row);
            const auto &expected_occurrence = axis.occurrence(row);
            const auto assignment = output->occurrence_assignment(
                span.network_id, row);
            ++metrics.route_occurrence_binding_count;
            if (!(assignment.network_occurrence == expected_identity) ||
                assignment.parent_readout_occurrence_key !=
                    expected_occurrence.parent_readout_occurrence_key ||
                assignment.paired_xr_occurrence_key !=
                    expected_occurrence.paired_xr_occurrence_key ||
                !(assignment.integration_support ==
                  expected_occurrence.integration_support) ||
                std::bit_cast<std::uint64_t>(assignment.assigned_time_unix_sec) !=
                    std::bit_cast<std::uint64_t>(
                        expected_identity.reconstructed_time_unix_sec()) ||
                assignment.assigned_time_unix_sec !=
                    std::midpoint(
                        assignment.integration_support.begin_unix_sec,
                        assignment.integration_support.end_unix_sec)) {
                ++metrics.route_occurrence_binding_mismatch_count;
            }

            ++metrics.ast_mapped_occurrence_count;
            if (!(mapped.identity(row) == expected_identity)) {
                ++metrics.ast_identity_mismatch_count;
            }
            const auto &record = output->ast_motion_record(
                span.network_id, row);
            const auto support = output->ast_motion_support(
                span.network_id, row);
            if (record.available()) {
                ++metrics.ast_available_occurrence_count;
                if (!support) {
                    ++metrics.ast_support_mismatch_count;
                    continue;
                }
                ++metrics.ast_support_count;
                const bool support_valid =
                    support->network_occurrence == expected_identity &&
                    support->lower_source_record.scope == paired->scope() &&
                    support->upper_source_record.scope == paired->scope() &&
                    support->upper_source_record.record ==
                        support->lower_source_record.record + 1 &&
                    support->lower_source_time_unix_sec <=
                        assignment.assigned_time_unix_sec &&
                    assignment.assigned_time_unix_sec <=
                        support->upper_source_time_unix_sec &&
                    support->lower_weight >= 0.0 &&
                    support->upper_weight >= 0.0 &&
                    support->lower_weight <= 1.0 &&
                    support->upper_weight <= 1.0 &&
                    std::abs((support->lower_weight +
                              support->upper_weight) - 1.0) <=
                        4.0 * std::numeric_limits<double>::epsilon();
                metrics.ast_support_mismatch_count += !support_valid;
            } else {
                ++metrics.ast_unavailable_occurrence_count;
                if (support || record.causes() ==
                                   pipeline::AstScanMotionCause::none) {
                    ++metrics.ast_support_mismatch_count;
                }
            }
        }
    }

    result.ast_raw_record_count = ast->raw_product_handle()->record_count();
    result.ast_raw_owned_bytes =
        ast->raw_product_handle()->memory_evidence().logical_owned_bytes();
    result.align_owned_bytes =
        align_context->memory_evidence().logical_owned_bytes();
    result.rtc_input_owned_bytes =
        input->memory_evidence().logical_owned_bytes();
    result.rtc_output_owned_bytes =
        output->memory_evidence().logical_owned_bytes();
    return result;
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
    std::int64_t subobservation = 0;
    std::int64_t scan = 0;
    pipeline::RtcOnlyTerminalResult terminal;
    pipeline::NativePairedReadoutCardinality native_cardinality;
    pipeline::NativePairedReadoutMemoryEvidence native_memory;
    ComparisonMetrics comparisons;
    TypedRouteEvidence typed_route;
    std::string mapping_instance_id;
    std::string telescope_sha256;
    std::uintmax_t telescope_byte_count = 0;
    std::string apt_manifest_sha256;
    std::string apt_bundle_semantic_sha256;
    std::string apt_bundle_envelope_sha256;
    std::string config_sha256;
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

    // VAL V0 is established immediately after Paired-D1 and remains the exact
    // immutable state threaded through ALIGN, RTC, and the MAP-facing bundle.
    auto val = pipeline::ValSnapshot::initial(paired_build.paired);
    const auto telescope = load_telescope(arguments.telescope, scope);
    auto ast_product = pipeline::build_ast_scan_motion_product(
        telescope.source, ast_identity_binding);
    require(ast_product->route_profile() ==
                    pipeline::AstScanMotionRouteProfile::science_lissajous &&
                ast_product->identity_binding().complete() &&
                ast_product->source_time_axis_mapping_eligible(),
            "approved telescope input did not produce the accepted AST route profile");
    std::vector<std::shared_ptr<const pipeline::NativeNetworkAlignment>>
        native_timings;
    native_timings.reserve(inputs.size());
    for (const auto &input : inputs) {
        native_timings.push_back(input.native_timing);
    }
    auto ast_views = pipeline::AstScanMotionNetworkViews::admit(
        scope, std::move(ast_product), std::move(native_timings));
    auto align_context = pipeline::IdentityRouteAlignContext::admit(
        paired_build.paired, std::move(ast_views), val);

    pipeline::RtcOnlyProductSlot full_publication;
    const auto full_spans =
        pipeline::full_native_occurrence_spans(*paired_build.paired);
    const auto full = run_route(
        1, align_context, full_spans, {full_spans},
        full_publication);
    const auto primary_cpu_end = std::clock();
    const auto primary_wall_end = std::chrono::steady_clock::now();
    require(full.map_facing_context_complete(),
            "full typed identity route did not reach the MAP-facing boundary");
    const auto &full_terminal =
        full.map_facing_bundle->rtc_context_handle()->rtc_terminal_handle();
    require(full_publication.snapshot() == full_terminal,
            "full typed identity route did not publish exactly one completion");
    require(full.rtc_terminal.failure_cause ==
                    pipeline::RtcOnlyFailureCause::none &&
                full.rtc_terminal.failure_detail.empty(),
            "successful identity RTC route retained a failure cause");
    require(full_terminal->timestream_handle()
                    ->memory_evidence()
                    .owned_numeric_bytes == 0,
            "identity RTC product unexpectedly owns a numerical plane");
    const auto &op = full_terminal->timestream_handle()
                         ->realized_operator();
    require(op.sampling_factor == 1 && op.sampling_phase == 0 &&
                op.x_from_x == 1.0 && op.x_from_r == 0.0 &&
                op.r_from_x == 0.0 && op.r_from_r == 1.0,
            "realized RTC operator is not exact paired identity");

    auto comparisons = paired_build.ingress_comparisons;
    compare_full_product_to_producer_facts(
        *full_terminal->timestream_handle(), relation, inputs,
        comparisons);
    const auto typed_route = inspect_typed_route(
        full, align_context, comparisons);
    require(typed_route.ast_present &&
                typed_route.ast_dependency_not_applicable &&
                typed_route.val_exact_snapshot_bound &&
                typed_route.calibration_unavailable &&
                typed_route.calibration_val_evaluation_unavailable &&
                typed_route.ptc_unavailable &&
                typed_route.ptc_val_evaluation_unavailable &&
                typed_route.map_admission_unavailable &&
                !typed_route.map_action_performed,
            "typed route endpoint states are incomplete or untruthful");

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
        2, align_context, full_spans,
        {std::move(first_spans), std::move(second_spans)},
        partitioned_publication);
    const auto &partitioned_diagnostics =
        partitioned.rtc_terminal.diagnostics;
    require(partitioned.map_facing_context_complete() &&
                partitioned_publication.snapshot() ==
                    partitioned.map_facing_bundle->rtc_context_handle()
                        ->rtc_terminal_handle() &&
                partitioned_diagnostics.engineering_partition_count == 2 &&
                partitioned_diagnostics.native_admission_entry_count == 1 &&
                partitioned_diagnostics.learn_entry_count == 1 &&
                partitioned_diagnostics.consider_entry_count == 1 &&
                partitioned_diagnostics.apply_entry_count == 1 &&
                partitioned_diagnostics.finalization_entry_count == 1 &&
                partitioned_diagnostics.publication_entry_count == 1,
            "two-chunk identity RTC route did not finalize one publication");
    compare_partitioned_to_single(
        *partitioned.map_facing_bundle->rtc_context_handle()
             ->signal_handle(),
        *full_terminal->timestream_handle(), comparisons);

    auto incomplete_partition = full_spans;
    for (auto &span : incomplete_partition) {
        span.past_last_native_row = span.first_native_row +
            static_cast<pipeline::TimestreamNativeRow>(
                span.occurrence_count() / 2);
    }
    pipeline::RtcOnlyProductSlot incomplete_publication;
    const auto incomplete = run_route(
        3, align_context, full_spans, {incomplete_partition},
        incomplete_publication);
    require(!incomplete.map_facing_context_complete() &&
                incomplete.state ==
                    pipeline::IdentityRouteContextState::rtc_failed &&
                incomplete.rtc_terminal.failure_cause ==
                    pipeline::RtcOnlyFailureCause::incomplete_logical_support &&
                !incomplete_publication.snapshot(),
            "missing engineering chunk published a false completion");

    const auto second_publish = run_route(
        4, align_context, full_spans, {full_spans},
        full_publication);
    require(!second_publish.map_facing_context_complete() &&
                second_publish.state ==
                    pipeline::IdentityRouteContextState::rtc_failed &&
                second_publish.rtc_terminal.state ==
                    pipeline::RtcOnlyTerminalState::publication_failed &&
                second_publish.rtc_terminal.failure_cause ==
                    pipeline::RtcOnlyFailureCause::publication_slot_occupied &&
                full_publication.snapshot() == full_terminal,
            "second publication did not preserve the committed product");
    pipeline::RtcOnlyProductSlot failed_publication;
    auto invalid_spans = full_spans;
    ++invalid_spans.front().past_last_native_row;
    const auto failed = run_route(
        5, align_context, invalid_spans, {invalid_spans},
        failed_publication);
    require(!failed.map_facing_context_complete() &&
                failed.state ==
                    pipeline::IdentityRouteContextState::input_context_failed &&
                failed.failure_cause ==
                    pipeline::IdentityRouteContextFailureCause::
                        input_context_rejected &&
                !failed_publication.snapshot(),
            "failed route published a false completion");

    const auto wall_time =
        std::chrono::duration<double>(primary_wall_end - wall_begin).count();
    const auto cpu_time = static_cast<double>(primary_cpu_end - cpu_begin) /
                          static_cast<double>(CLOCKS_PER_SEC);
    require(wall_time > 0.0 && cpu_time > 0.0,
            "acceptance timing measurements are not positive");
    return {
        observation.observation, observation.subobservation,
        observation.scan, full.rtc_terminal, native_cardinality,
        native_memory, comparisons, typed_route,
        std::move(paired_build.mapping_instance_id), telescope.sha256,
        telescope.byte_count,
        citlali::utils::sha256_file(arguments.apt_manifest),
        relation.bundle_identity().semantic_sha256,
        relation.bundle_identity().envelope_sha256,
        citlali::utils::sha256_file(arguments.config), wall_time, cpu_time,
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
    require(run.comparisons.route_occurrence_binding_count ==
                    run.native_cardinality.native_occurrence_count &&
                run.comparisons.ast_mapped_occurrence_count ==
                    run.native_cardinality.native_occurrence_count &&
                run.comparisons.ast_available_occurrence_count +
                        run.comparisons.ast_unavailable_occurrence_count ==
                    run.native_cardinality.native_occurrence_count &&
                run.comparisons.ast_available_occurrence_count > 0 &&
                run.comparisons.ast_support_count ==
                    run.comparisons.ast_available_occurrence_count &&
                run.comparisons.val_binding_comparison_count == 8,
            "typed route comparison coverage is incomplete");
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
                run.comparisons.assigned_support_binding_mismatch_count == 0 &&
                run.comparisons.route_occurrence_binding_mismatch_count == 0 &&
                run.comparisons.ast_identity_mismatch_count == 0 &&
                run.comparisons.ast_support_mismatch_count == 0 &&
                run.comparisons.val_binding_mismatch_count == 0,
            "acceptance comparisons contain a scientific mismatch");
    require(run.typed_route.ast_present &&
                run.typed_route.ast_dependency_not_applicable &&
                run.typed_route.val_exact_snapshot_bound &&
                run.typed_route.val_generation == 0 &&
                run.typed_route.val_finding_count == 0 &&
                run.typed_route.calibration_unavailable &&
                run.typed_route.calibration_val_evaluation_unavailable &&
                run.typed_route.ptc_unavailable &&
                run.typed_route.ptc_val_evaluation_unavailable &&
                run.typed_route.map_admission_unavailable &&
                !run.typed_route.map_action_performed,
            "typed route evidence does not preserve the approved endpoint states");
    const auto &diagnostics = run.terminal.diagnostics;
    require(diagnostics.native_admission_entry_count == 1 &&
                diagnostics.learn_entry_count == 1 &&
                diagnostics.consider_entry_count == 1 &&
                diagnostics.apply_entry_count == 1 &&
                diagnostics.finalization_entry_count == 1 &&
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
           << "  \"subject_candidate_revision\": "
           << q(subject_candidate_revision) << ",\n"
           << "  \"subject_candidate_tree\": "
           << q(subject_candidate_tree) << ",\n"
           << "  \"tooling_revision\": "
           << q(arguments.source_revision) << ",\n"
           << "  \"source_revision\": " << q(arguments.source_revision)
           << ",\n"
           << "  \"executable_revision\": "
           << q(CITLALI_GIT_REVISION)
           << ",\n"
           << "  \"executable_version\": "
           << q(std::string{CITLALI_GIT_VERSION} + " kids=" +
                KIDSCPP_GIT_VERSION + " tula=" + TULA_GIT_VERSION)
           << ",\n"
           << "  \"citlali_source_clean\": true,\n"
           << "  \"executable_sha256\": "
           << q(arguments.executable_sha256) << ",\n"
           << "  \"build_environment\": "
           << q(arguments.build_environment) << ",\n"
           << "  \"build_profile\": "
           << q(arguments.build_profile) << ",\n"
           << "  \"spack_environment_sha256\": "
           << q(citlali::utils::sha256_file(arguments.spack_environment))
           << ",\n"
           << "  \"spack_environment_byte_count\": "
           << fs::file_size(arguments.spack_environment) << ",\n"
           << "  \"spack_environment_retained\": true,\n"
           << "  \"spack_lock_sha256\": "
           << q(citlali::utils::sha256_file(arguments.spack_lock))
           << ",\n"
           << "  \"spack_lock_byte_count\": "
           << fs::file_size(arguments.spack_lock) << ",\n"
           << "  \"spack_lock_retained\": true,\n"
           << "  \"spack_root_dag\": "
           << q(arguments.spack_root_dag) << ",\n"
           << "  \"dependency_state_verified\": true,\n"
           << "  \"kidscpp_version\": " << q(KIDSCPP_GIT_VERSION)
           << ",\n"
           << "  \"tula_version\": " << q(TULA_GIT_VERSION)
           << ",\n"
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
           << "  \"route_context_state\": \"map_facing_context_complete\",\n"
           << "  \"route_activated\": false,\n"
           << "  \"ordinary_route_changed\": false,\n"
           << "  \"canonical_integration_performed\": false,\n"
           << "  \"representative_science_claim\": false,\n"
           << "  \"representative_dataset_id\": "
           << q(arguments.dataset_id) << ",\n"
           << "  \"observation\": " << run.observation << ",\n"
           << "  \"subobservation\": " << run.subobservation << ",\n"
           << "  \"scan\": " << run.scan << ",\n"
           << "  \"first_native_row\": "
           << arguments.first_native_row << ",\n"
           << "  \"native_row_count\": "
           << arguments.native_row_count << ",\n"
           << "  \"mapping_instance_id\": "
           << q(run.mapping_instance_id) << ",\n"
           << "  \"telescope_filename\": " << q(telescope_filename)
           << ",\n"
           << "  \"telescope_sha256\": " << q(run.telescope_sha256)
           << ",\n"
           << "  \"telescope_byte_count\": "
           << run.telescope_byte_count << ",\n"
           << "  \"telescope_record_count\": "
           << run.typed_route.ast_raw_record_count << ",\n"
           << "  \"apt_manifest_sha256\": "
           << q(run.apt_manifest_sha256) << ",\n"
           << "  \"apt_bundle_semantic_sha256\": "
           << q(run.apt_bundle_semantic_sha256) << ",\n"
           << "  \"apt_bundle_envelope_sha256\": "
           << q(run.apt_bundle_envelope_sha256) << ",\n"
           << "  \"config_sha256\": " << q(run.config_sha256) << ",\n"
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
           << "  \"ast_present_in_rtc_input_context\": "
           << b(run.typed_route.ast_present) << ",\n"
           << "  \"identity_rtc_ast_dependency\": \"not_applicable\",\n"
           << "  \"val_initial_generation\": "
           << run.typed_route.val_generation << ",\n"
           << "  \"val_committed_finding_count\": "
           << run.typed_route.val_finding_count << ",\n"
           << "  \"val_exact_snapshot_bound\": "
           << b(run.typed_route.val_exact_snapshot_bound) << ",\n"
           << "  \"calibration_product_state\": "
              "\"unavailable_component_not_admitted\",\n"
           << "  \"calibration_for_ptc_val_evaluation_state\": "
              "\"unavailable_calibration_product_absent\",\n"
           << "  \"ptc_product_state\": "
              "\"unavailable_component_not_admitted\",\n"
           << "  \"ptc_for_map_val_evaluation_state\": "
              "\"unavailable_ptc_product_absent\",\n"
           << "  \"map_admission_state\": "
              "\"unavailable_calibration_and_ptc_products\",\n"
           << "  \"map_action_performed\": "
           << b(run.typed_route.map_action_performed) << ",\n"
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
           << "    \"paired_coordinate_state_bytes\": "
           << run.native_memory.coordinate_state_bytes << ",\n"
           << "    \"paired_occurrence_axis_bytes\": "
           << run.native_memory.occurrence_axis_bytes << ",\n"
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
           << "    \"x_payload_available_cell_count\": "
           << run.terminal.diagnostics.x_payload_available_cell_count
           << ",\n"
           << "    \"r_payload_available_cell_count\": "
           << run.terminal.diagnostics.r_payload_available_cell_count
           << ",\n"
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
           << "    \"route_occurrence_binding_count\": "
           << run.comparisons.route_occurrence_binding_count << ",\n"
           << "    \"ast_mapped_occurrence_count\": "
           << run.comparisons.ast_mapped_occurrence_count << ",\n"
           << "    \"ast_available_occurrence_count\": "
           << run.comparisons.ast_available_occurrence_count << ",\n"
           << "    \"ast_unavailable_occurrence_count\": "
           << run.comparisons.ast_unavailable_occurrence_count << ",\n"
           << "    \"ast_support_count\": "
           << run.comparisons.ast_support_count << ",\n"
           << "    \"val_binding_comparison_count\": "
           << run.comparisons.val_binding_comparison_count << ",\n"
           << "    \"ast_raw_owned_bytes\": "
           << run.typed_route.ast_raw_owned_bytes << ",\n"
           << "    \"ast_mapped_owned_bytes\": "
           << run.typed_route.ast_mapped_owned_bytes << ",\n"
           << "    \"align_owned_bytes\": "
           << run.typed_route.align_owned_bytes << ",\n"
           << "    \"rtc_input_owned_bytes\": "
           << run.typed_route.rtc_input_owned_bytes << ",\n"
           << "    \"rtc_output_owned_bytes\": "
           << run.typed_route.rtc_output_owned_bytes << ",\n"
           << "    \"val_owned_bytes\": "
           << run.typed_route.val_owned_bytes << ",\n"
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
           << "    \"route_occurrence_binding_mismatch_count\": "
           << run.comparisons.route_occurrence_binding_mismatch_count
           << ",\n"
           << "    \"ast_identity_mismatch_count\": "
           << run.comparisons.ast_identity_mismatch_count << ",\n"
           << "    \"ast_support_mismatch_count\": "
           << run.comparisons.ast_support_mismatch_count << ",\n"
           << "    \"val_binding_mismatch_count\": "
           << run.comparisons.val_binding_mismatch_count << ",\n"
           << "    \"native_admission_entry_count\": "
           << run.terminal.diagnostics.native_admission_entry_count << ",\n"
           << "    \"learn_entry_count\": "
           << run.terminal.diagnostics.learn_entry_count << ",\n"
           << "    \"consider_entry_count\": "
           << run.terminal.diagnostics.consider_entry_count << ",\n"
           << "    \"apply_entry_count\": "
           << run.terminal.diagnostics.apply_entry_count << ",\n"
           << "    \"finalization_entry_count\": "
           << run.terminal.diagnostics.finalization_entry_count << ",\n"
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
        require(arguments.source_revision.starts_with(CITLALI_GIT_REVISION) &&
                    std::string_view{CITLALI_GIT_VERSION}.find("dirty") ==
                        std::string_view::npos,
                "source revision does not match the compiled Citlali revision");
        require(arguments.owner_run, "owner-run authorization is required");
        require(fs::is_directory(arguments.data_directory),
                "data directory does not exist");
        require(fs::absolute(arguments.telescope) == arguments.telescope &&
                    fs::is_regular_file(arguments.telescope),
                "telescope path must be an absolute regular file");
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
        require(fs::is_regular_file(arguments.spack_lock),
                "Spack lock is not a regular file");
        require(fs::is_regular_file(arguments.spack_environment),
                "Spack environment manifest is not a regular file");
        require(fs::is_regular_file(arguments.executable),
                "acceptance executable path is not a regular file");
        arguments.executable_sha256 =
            citlali::utils::sha256_file(arguments.executable);

        auto [logger, log_counts] = configure_logging();
        const auto verified = apt::verify_bundle_filesystem(
            arguments.apt_manifest, true);
        const auto relation =
            pipeline::admit_canonical_apt_detector_relation_v2(verified);
        require(relation.observation().observation == 152390 &&
                    relation.observation().subobservation == 0 &&
                    relation.observation().scan == 2,
                "this bounded acceptance invocation requires observation (152390, 0, 2)");
        const auto config = load_runtime_config(arguments.config);
        const auto inputs = resolve_network_inputs(
            arguments, relation, config);
        require(inputs.size() == 11,
                "representative APT bundle must contain 11 networks");
        (void)logger;
        const auto run = execute_acceptance(
            arguments, relation, inputs, config, support_assignment);
        require(log_counts->errors == 0 && log_counts->criticals == 0,
                "acceptance route emitted unexpected error-level records");
        write_acceptance_record(
            arguments, support_assignment, run, *log_counts);
        std::cout << "Timestream Successor identity-route acceptance record: "
                  << arguments.output << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Timestream Successor identity-route acceptance runner: FAIL: "
                  << error.what() << '\n';
        return 2;
    }
}
