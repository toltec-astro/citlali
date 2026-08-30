#include <citlali/core/pipeline/ast_scan_motion.h>
#include <citlali/core/pipeline/ast_scan_motion_alignment.h>
#include <citlali/core/pipeline/canonical_apt_bundle_v2.h>
#include <citlali/core/pipeline/canonical_apt_detector_relation_v2.h>
#include <citlali/core/utils/sha256.h>

#include <citlali_config/gitversion.h>
#include <citlali_wp7/acceptance_build_identity.h>

#include <netcdf>
#include <yaml-cpp/yaml.h>

#include <Eigen/Core>

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
#include <map>
#include <memory>
#include <numbers>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace pipeline = citlali::pipeline;
namespace apt = citlali::pipeline::canonical_apt_v2;

constexpr std::string_view acceptance_schema =
    "citlali-wp7-ast-scan-motion-acceptance-v1";
constexpr std::string_view representative_dataset_id =
    "SCI_ALIGN_STAGE7_NGC4449_152390";
constexpr std::string_view telescope_filename =
    "tel_toltec_2026-02-19_152390_00_0002.nc";
constexpr std::string_view telescope_sha256 =
    "2845455a620635955c00a4731e0d9720cfa456fece79d1729cf755a366a1ad6b";
constexpr std::uintmax_t telescope_byte_count = 24157872;
constexpr std::size_t telescope_record_count = 62109;
constexpr std::string_view design_commit = "46824f7de";
constexpr std::string_view align_repair_commit = "d55deefb3";
constexpr pipeline::AstScanMotionIdentityBinding identity_binding{
    1523900001, 1523900002, 1523900003, 1523900004};
constexpr double radians_to_arcsec =
    180.0 * 3600.0 / std::numbers::pi_v<double>;

struct Arguments {
    fs::path data_directory;
    fs::path telescope;
    fs::path apt_manifest;
    fs::path config;
    fs::path output;
    fs::path executable;
    std::string executable_sha256;
    std::string source_revision;
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
        "Usage: citlali_wp7_ast_scan_motion_acceptance\n"
        "  --data-dir PATH --telescope PATH --apt-manifest PATH\n"
        "  --config PATH --output PATH --source-revision FULL_SHA\n"
        "  --owner-run --design-is-ancestor --align-repair-is-ancestor\n";
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
        }
        if (option == "--data-dir") {
            result.data_directory = next(index, option);
        }
        else if (option == "--telescope") {
            result.telescope = next(index, option);
        }
        else if (option == "--apt-manifest") {
            result.apt_manifest = next(index, option);
        }
        else if (option == "--config") {
            result.config = next(index, option);
        }
        else if (option == "--output") {
            result.output = next(index, option);
        }
        else if (option == "--source-revision") {
            result.source_revision = next(index, option);
        }
        else if (option == "--owner-run") {
            result.owner_run = true;
        }
        else if (option == "--design-is-ancestor") {
            result.design_is_ancestor = true;
        }
        else if (option == "--align-repair-is-ancestor") {
            result.align_repair_is_ancestor = true;
        }
        else {
            fail("unknown option: " + option);
        }
    }
    require(fs::is_directory(result.data_directory),
            "--data-dir must name a directory");
    require(fs::is_regular_file(result.telescope),
            "--telescope must name a regular file");
    require(fs::is_regular_file(result.apt_manifest),
            "--apt-manifest must name a regular file");
    require(fs::is_regular_file(result.config),
            "--config must name a regular file");
    require(!result.output.empty(), "--output is required");
    require(!result.source_revision.empty(),
            "--source-revision is required");
    return result;
}

bool full_lowercase_git_sha(std::string_view value) {
    return value.size() == 40 &&
        std::all_of(value.begin(), value.end(), [](unsigned char ch) {
            return (ch >= '0' && ch <= '9') ||
                (ch >= 'a' && ch <= 'f');
        });
}

std::string json_escape(std::string_view value) {
    std::ostringstream stream;
    for (const unsigned char ch : value) {
        switch (ch) {
        case '"': stream << "\\\""; break;
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
            }
            else {
                stream << static_cast<char>(ch);
            }
        }
    }
    return stream.str();
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

struct RuntimeConfig {
    std::array<double, 13> interface_offsets_sec{};
    std::array<bool, 13> interface_offset_present{};
};

RuntimeConfig load_runtime_config(const fs::path &path) {
    const auto root = YAML::LoadFile(path.string());
    const auto offsets = root["interface_sync_offset"];
    require(offsets && offsets.IsSequence(),
            "config lacks interface_sync_offset sequence");
    RuntimeConfig result;
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
        const auto network = std::stoll(suffix);
        require(network >= 0 && network < 13 && seen.insert(key).second,
                "interface sync key is duplicate or out of range");
        const auto value = entry.begin()->second.as<double>();
        require(std::isfinite(value),
                "interface sync offset must be finite");
        result.interface_offsets_sec[static_cast<std::size_t>(network)] =
            value;
        result.interface_offset_present[static_cast<std::size_t>(network)] =
            true;
    }
    return result;
}

fs::path find_raw_file(
    const fs::path &directory,
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
        if (name.starts_with(prefix.str()) &&
            entry.path().extension() == ".nc") {
            candidates.push_back(entry.path());
        }
    }
    require(candidates.size() == 1,
            "expected exactly one raw file for " + source.interface_name);
    return fs::absolute(candidates.front());
}

struct NetworkInput {
    std::int64_t network = -1;
    std::string filename;
    std::string sha256;
    std::uintmax_t byte_count = 0;
    std::shared_ptr<const pipeline::NativeNetworkAlignment> timing;
};

NetworkInput load_network_input(
    const fs::path &data_directory,
    const pipeline::CanonicalAptRawSourceBindingV2 &source,
    const RuntimeConfig &config) {
    require(source.network >= 0 && source.network < 13 &&
                config.interface_offset_present[
                    static_cast<std::size_t>(source.network)],
            "network source lacks a valid interface offset");
    const auto path = find_raw_file(data_directory, source);
    std::error_code error;
    const auto byte_count = fs::file_size(path, error);
    require(!error && byte_count == source.byte_count,
            "raw byte count disagrees with APT source record");
    const auto digest = citlali::utils::sha256_file(path);
    require("sha256:" + digest == source.content_sha256,
            "raw SHA-256 disagrees with APT source record");

    netCDF::NcFile file(path.string(), netCDF::NcFile::read);
    const double fpga_frequency =
        read_netcdf_scalar<double>(file, "Header.Toltec.FpgaFreq");
    const auto variable = file.getVar("Data.Toltec.Ts");
    require(std::isfinite(fpga_frequency) && fpga_frequency > 0.0 &&
                !variable.isNull() && variable.getDimCount() == 2 &&
                variable.getDim(1).getSize() == 6,
            "raw network timing facts are invalid");
    const auto row_count = variable.getDim(0).getSize();
    require(row_count > 0 &&
                row_count <= static_cast<std::size_t>(
                    std::numeric_limits<Eigen::Index>::max()),
            "raw network timing cardinality is invalid");
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        timestamps(static_cast<Eigen::Index>(row_count), 6);
    variable.getVar(timestamps.data());
    auto timing =
        std::make_shared<const pipeline::NativeNetworkAlignment>(
            pipeline::make_native_network_alignment(
                source.network, 0, timestamps, fpga_frequency,
                config.interface_offsets_sec[
                    static_cast<std::size_t>(source.network)]));
    return {source.network, path.filename().string(), digest,
            byte_count, std::move(timing)};
}

struct TelescopeLoad {
    std::shared_ptr<const pipeline::AstScanMotionSource> source;
    double minimum_interval_sec = 0.0;
    double maximum_interval_sec = 0.0;
    double direct_adjacent_maximum_arcsec_per_sec = 0.0;
    pipeline::AstTelescopeRecord direct_adjacent_maximizing_record = -1;
};

std::array<double, 3> direction(double ra, double dec) {
    const double cos_dec = std::cos(dec);
    return {cos_dec * std::cos(ra), cos_dec * std::sin(ra),
            std::sin(dec)};
}

double great_circle_angle(const std::array<double, 3> &left,
                          const std::array<double, 3> &right) {
    const std::array<double, 3> cross{
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0]};
    const double cross_norm =
        std::hypot(std::hypot(cross[0], cross[1]), cross[2]);
    const double dot = std::clamp(
        left[0] * right[0] + left[1] * right[1] +
            left[2] * right[2],
        -1.0, 1.0);
    return std::atan2(cross_norm, dot);
}

TelescopeLoad load_telescope(const fs::path &path,
                             const pipeline::NativeObservationScope &scope) {
    require(path.filename() == telescope_filename,
            "telescope filename is not the approved artifact");
    std::error_code error;
    require(fs::file_size(path, error) == telescope_byte_count && !error,
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
    require(!time_variable.isNull() && time_variable.getDimCount() == 1 &&
                time_variable.getDim(0).getSize() == telescope_record_count,
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

    TelescopeLoad result;
    result.minimum_interval_sec = std::numeric_limits<double>::infinity();
    for (Eigen::Index index = 1; index < times.size(); ++index) {
        const double dt = times(index) - times(index - 1);
        require(std::isfinite(dt) && dt > 0.0,
                "telescope time is not finite and strictly increasing");
        result.minimum_interval_sec =
            std::min(result.minimum_interval_sec, dt);
        result.maximum_interval_sec =
            std::max(result.maximum_interval_sec, dt);
        const double speed =
            great_circle_angle(
                direction(ra(index - 1), dec(index - 1)),
                direction(ra(index), dec(index))) /
            dt * radians_to_arcsec;
        require(std::isfinite(speed),
                "direct adjacent diagnostic speed is nonfinite");
        if (speed > result.direct_adjacent_maximum_arcsec_per_sec) {
            result.direct_adjacent_maximum_arcsec_per_sec = speed;
            result.direct_adjacent_maximizing_record =
                static_cast<pipeline::AstTelescopeRecord>(index);
        }
    }
    result.source = pipeline::AstScanMotionSource::admit(
        scope, scope, 0, std::move(metadata), std::move(times),
        std::move(ra), std::move(dec));
    return result;
}

std::uint64_t bit_pattern(double value) {
    return std::bit_cast<std::uint64_t>(value);
}

bool same_record(const pipeline::AstScanMotionDerivedRecord &left,
                 const pipeline::AstScanMotionDerivedRecord &right) {
    return left.causes() == right.causes() &&
        left.continuity_run() == right.continuity_run() &&
        left.raw_direction_structurally_valid() ==
            right.raw_direction_structurally_valid() &&
        left.telemetry_quality_classified() ==
            right.telemetry_quality_classified() &&
        left.telemetry_defect() == right.telemetry_defect() &&
        left.realized_direction_valid() ==
            right.realized_direction_valid() &&
        left.derivative_valid() == right.derivative_valid() &&
        bit_pattern(left.telemetry_residual_arcsec()) ==
            bit_pattern(right.telemetry_residual_arcsec()) &&
        bit_pattern(left.east_velocity_arcsec_per_sec()) ==
            bit_pattern(right.east_velocity_arcsec_per_sec()) &&
        bit_pattern(left.north_velocity_arcsec_per_sec()) ==
            bit_pattern(right.north_velocity_arcsec_per_sec()) &&
        bit_pattern(left.scalar_speed_arcsec_per_sec()) ==
            bit_pattern(right.scalar_speed_arcsec_per_sec());
}

bool same_summary(const pipeline::AstScanMotionScanSummary &left,
                  const pipeline::AstScanMotionScanSummary &right) {
    return left.maximum_available == right.maximum_available &&
        left.causes == right.causes &&
        left.maximizing_record == right.maximizing_record &&
        bit_pattern(left.maximum_speed_arcsec_per_sec) ==
            bit_pattern(right.maximum_speed_arcsec_per_sec) &&
        left.record_count == right.record_count &&
        left.continuity_run_count == right.continuity_run_count &&
        left.derivative_valid_record_count ==
            right.derivative_valid_record_count &&
        left.admitted_candidate_count ==
            right.admitted_candidate_count &&
        left.telemetry_defect_count == right.telemetry_defect_count;
}

std::uint64_t peak_rss_bytes() {
    rusage usage{};
    require(getrusage(RUSAGE_SELF, &usage) == 0,
            "unable to read process resource usage");
#if defined(__APPLE__)
    return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
    return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024U;
#endif
}

struct ProductEvidence {
    std::shared_ptr<const pipeline::AstScanMotionProduct> product;
    std::vector<pipeline::AstTelescopeRecord> defect_records;
    std::size_t raw_direction_valid_count = 0;
    std::size_t quality_classified_count = 0;
    std::size_t realized_direction_valid_count = 0;
    std::size_t derivative_valid_count = 0;
    std::size_t record_mismatch_count = 0;
    std::size_t telemetry_support_mismatch_count = 0;
    std::size_t derivative_support_mismatch_count = 0;
    std::size_t summary_mismatch_count = 0;
    double wall_time_sec = 0.0;
    double cpu_time_sec = 0.0;
};

ProductEvidence build_product_evidence(
    const std::shared_ptr<const pipeline::AstScanMotionSource> &source) {
    const auto wall_begin = std::chrono::steady_clock::now();
    const auto cpu_begin = std::clock();
    auto product = pipeline::build_ast_scan_motion_product(
        source, identity_binding);
    const auto cpu_end = std::clock();
    const auto wall_end = std::chrono::steady_clock::now();

    ProductEvidence evidence;
    evidence.product = product;
    evidence.wall_time_sec =
        std::chrono::duration<double>(wall_end - wall_begin).count();
    evidence.cpu_time_sec =
        static_cast<double>(cpu_end - cpu_begin) / CLOCKS_PER_SEC;
    require(product->source_handle() == source &&
                product->identity_binding() == identity_binding &&
                product->source_time_axis_mapping_eligible(),
            "AST product lost its exact source or lifecycle binding");

    for (std::size_t index = 0; index < product->record_count(); ++index) {
        const auto &record = product->record_at_local(index);
        evidence.raw_direction_valid_count +=
            record.raw_direction_structurally_valid();
        evidence.quality_classified_count +=
            record.telemetry_quality_classified();
        evidence.realized_direction_valid_count +=
            record.realized_direction_valid();
        evidence.derivative_valid_count += record.derivative_valid();
        if (record.telemetry_defect()) {
            evidence.defect_records.push_back(product->record_identity(index));
        }
    }

    const auto one_third = static_cast<pipeline::AstTelescopeRecord>(
        source->record_count() / 3);
    const auto two_thirds = static_cast<pipeline::AstTelescopeRecord>(
        2 * source->record_count() / 3);
    const std::vector<pipeline::AstScanMotionProcessingSpan> schedule{
        {two_thirds, source->past_last_record()},
        {source->first_record(), one_third},
        {one_third, two_thirds}};
    const auto partitioned = pipeline::build_ast_scan_motion_product(
        source, identity_binding, schedule);
    for (std::size_t index = 0; index < product->record_count(); ++index) {
        evidence.record_mismatch_count +=
            !same_record(product->record_at_local(index),
                         partitioned->record_at_local(index));
        const auto identity = product->record_identity(index);
        evidence.telemetry_support_mismatch_count +=
            product->telemetry_support(identity) !=
            partitioned->telemetry_support(identity);
        evidence.derivative_support_mismatch_count +=
            product->derivative_support(identity) !=
            partitioned->derivative_support(identity);
    }
    evidence.summary_mismatch_count =
        !same_summary(product->scan_summary(),
                      partitioned->scan_summary());
    return evidence;
}

struct NetworkEvidence {
    std::int64_t network = -1;
    std::string filename;
    std::string sha256;
    std::uintmax_t byte_count = 0;
    std::size_t occurrence_count = 0;
    std::size_t packet_discontinuity_count = 0;
    std::size_t available_count = 0;
    std::size_t unavailable_count = 0;
    std::size_t telemetry_defect_cause_count = 0;
    double first_time_unix_sec = 0.0;
    double last_time_unix_sec = 0.0;
};

struct MappingEvidence {
    std::vector<NetworkEvidence> networks;
    std::size_t total_occurrence_count = 0;
    std::size_t available_count = 0;
    std::size_t unavailable_count = 0;
    std::size_t support_count = 0;
    std::size_t identity_mismatch_count = 0;
    std::size_t support_mismatch_count = 0;
    std::size_t value_mismatch_count = 0;
    std::size_t missing_unavailable_cause_count = 0;
    std::size_t mapped_owned_bytes = 0;
    double nw0_first_time_unix_sec = 0.0;
    double nw7_first_time_unix_sec = 0.0;
    bool nw0_nw7_distinct = false;
    double wall_time_sec = 0.0;
    double cpu_time_sec = 0.0;
};

MappingEvidence build_mapping_evidence(
    const pipeline::NativeObservationScope &scope,
    const std::shared_ptr<const pipeline::AstScanMotionProduct> &product,
    const std::vector<NetworkInput> &inputs) {
    std::vector<std::shared_ptr<const pipeline::NativeNetworkAlignment>>
        timings;
    timings.reserve(inputs.size());
    for (const auto &input : inputs) timings.push_back(input.timing);
    const auto wall_begin = std::chrono::steady_clock::now();
    const auto cpu_begin = std::clock();
    const auto views = pipeline::AstScanMotionNetworkViews::admit(
        scope, product, std::move(timings));
    MappingEvidence evidence;

    for (const auto &input : inputs) {
        const auto &view = views->network(input.network);
        NetworkEvidence network;
        network.network = input.network;
        network.filename = input.filename;
        network.sha256 = input.sha256;
        network.byte_count = input.byte_count;
        network.occurrence_count = view.occurrence_count();
        network.first_time_unix_sec =
            input.timing->identity(input.timing->first_native_row())
                .reconstructed_time_unix_sec();
        network.last_time_unix_sec =
            input.timing->identity(input.timing->past_last_native_row() - 1)
                .reconstructed_time_unix_sec();
        const auto &counters = input.timing->packet_counters();
        for (std::size_t index = 1; index < counters.size(); ++index) {
            network.packet_discontinuity_count +=
                !pipeline::packet_counters_are_contiguous(
                    counters[index - 1], counters[index]);
        }
        for (pipeline::TimestreamNativeRow row = view.first_native_row();
             row < view.past_last_native_row(); ++row) {
            ++evidence.total_occurrence_count;
            const auto expected_identity = input.timing->identity(row);
            evidence.identity_mismatch_count +=
                !(view.identity(row) == expected_identity);
            const auto &record = view.record(row);
            if (!record.available()) {
                ++network.unavailable_count;
                ++evidence.unavailable_count;
                network.telemetry_defect_cause_count += pipeline::has_cause(
                    record.causes(),
                    pipeline::AstScanMotionCause::telemetry_defect);
                evidence.missing_unavailable_cause_count +=
                    !pipeline::has_cause(
                        record.causes(), pipeline::AstScanMotionCause::
                                             network_mapping_support_unavailable);
                continue;
            }
            ++network.available_count;
            ++evidence.available_count;
            const auto support = view.support(row);
            const auto speed = view.scalar_speed_arcsec_per_sec(row);
            if (!support || !speed) {
                ++evidence.support_mismatch_count;
                continue;
            }
            ++evidence.support_count;
            const auto lower = product->source_handle()->local_index(
                support->lower_source_record.record);
            const auto upper = product->source_handle()->local_index(
                support->upper_source_record.record);
            const double expected_speed =
                support->lower_weight *
                    product->record_at_local(lower)
                        .scalar_speed_arcsec_per_sec() +
                support->upper_weight *
                    product->record_at_local(upper)
                        .scalar_speed_arcsec_per_sec();
            const double target_time =
                expected_identity.reconstructed_time_unix_sec();
            const bool support_valid =
                support->network_occurrence == expected_identity &&
                support->lower_source_record.scope == scope &&
                support->upper_source_record.scope == scope &&
                support->upper_source_record.record ==
                    support->lower_source_record.record + 1 &&
                support->lower_source_time_unix_sec <= target_time &&
                target_time <= support->upper_source_time_unix_sec &&
                support->lower_weight >= 0.0 &&
                support->upper_weight >= 0.0 &&
                support->lower_weight <= 1.0 &&
                support->upper_weight <= 1.0 &&
                std::abs((support->lower_weight +
                          support->upper_weight) - 1.0) <=
                    4.0 * std::numeric_limits<double>::epsilon();
            evidence.support_mismatch_count += !support_valid;
            evidence.value_mismatch_count +=
                bit_pattern(*speed) != bit_pattern(expected_speed);
        }
        evidence.mapped_owned_bytes +=
            view.memory_evidence().logical_owned_bytes();
        evidence.networks.push_back(std::move(network));
    }
    const auto cpu_end = std::clock();
    const auto wall_end = std::chrono::steady_clock::now();
    evidence.wall_time_sec =
        std::chrono::duration<double>(wall_end - wall_begin).count();
    evidence.cpu_time_sec =
        static_cast<double>(cpu_end - cpu_begin) / CLOCKS_PER_SEC;
    const auto nw0 = std::find_if(
        evidence.networks.begin(), evidence.networks.end(),
        [](const auto &network) { return network.network == 0; });
    const auto nw7 = std::find_if(
        evidence.networks.begin(), evidence.networks.end(),
        [](const auto &network) { return network.network == 7; });
    if (nw0 != evidence.networks.end() && nw7 != evidence.networks.end()) {
        evidence.nw0_first_time_unix_sec = nw0->first_time_unix_sec;
        evidence.nw7_first_time_unix_sec = nw7->first_time_unix_sec;
        evidence.nw0_nw7_distinct =
            bit_pattern(evidence.nw0_first_time_unix_sec) !=
            bit_pattern(evidence.nw7_first_time_unix_sec);
    }
    return evidence;
}

void write_record(const Arguments &arguments,
                  const pipeline::CanonicalAptDetectorRelationV2 &relation,
                  const TelescopeLoad &telescope,
                  const ProductEvidence &product,
                  const MappingEvidence &mapping,
                  std::uint64_t peak_rss) {
    std::ofstream output(arguments.output);
    require(static_cast<bool>(output),
            "unable to open AST acceptance output");
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"" << acceptance_schema << "\",\n"
           << "  \"source_revision\": \""
           << json_escape(arguments.source_revision) << "\",\n"
           << "  \"executable_revision\": \""
           << CITLALI_WP7_SOURCE_REVISION << "\",\n"
           << "  \"executable_version\": \""
           << json_escape(CITLALI_GIT_VERSION) << "\",\n"
           << "  \"executable_sha256\": \""
           << arguments.executable_sha256 << "\",\n"
           << "  \"citlali_source_clean\": true,\n"
           << "  \"citlali_ignored_source_state_verified\": true,\n"
           << "  \"dependency_state_verified\": true,\n"
           << "  \"kidscpp_revision\": \""
           << CITLALI_WP7_KIDSCPP_REVISION << "\",\n"
           << "  \"kidscpp_build_patch_sha256\": \""
           << CITLALI_WP7_KIDSCPP_PATCH_SHA256 << "\",\n"
           << "  \"kidscpp_tree\": \""
           << CITLALI_WP7_KIDSCPP_TREE << "\",\n"
           << "  \"tula_revision\": \""
           << CITLALI_WP7_TULA_REVISION << "\",\n"
           << "  \"tula_build_patch_sha256\": \""
           << CITLALI_WP7_TULA_PATCH_SHA256 << "\",\n"
           << "  \"tula_tree\": \""
           << CITLALI_WP7_TULA_TREE << "\",\n"
           << "  \"design_commit\": \"" << design_commit << "\",\n"
           << "  \"align_repair_commit\": \"" << align_repair_commit
           << "\",\n"
           << "  \"design_is_ancestor\": true,\n"
           << "  \"align_repair_is_ancestor\": true,\n"
           << "  \"owner_run\": true,\n"
           << "  \"representative_data\": true,\n"
           << "  \"authority_policy_id\": \""
           << pipeline::ast_scan_motion_policy_id << "\",\n"
           << "  \"product_role\": \""
           << pipeline::ast_scan_motion_product_role << "\",\n"
           << "  \"representative_dataset_id\": \""
           << representative_dataset_id << "\",\n"
           << "  \"observation\": 152390,\n"
           << "  \"subobservation\": 0,\n"
           << "  \"scan\": 2,\n"
           << "  \"common_analysis_grid_requested\": false,\n"
           << "  \"persistent_ast_product_published\": false,\n"
           << "  \"product_inspected_in_memory\": true,\n"
           << "  \"telescope\": {\n"
           << "    \"filename\": \"" << telescope_filename << "\",\n"
           << "    \"sha256\": \"" << telescope_sha256 << "\",\n"
           << "    \"byte_count\": " << telescope_byte_count << ",\n"
           << "    \"record_count\": " << telescope.source->record_count()
           << ",\n"
           << "    \"time_field\": \""
           << pipeline::ast_scan_motion_time_field << "\",\n"
           << "    \"ra_field\": \""
           << pipeline::ast_scan_motion_ra_field << "\",\n"
           << "    \"dec_field\": \""
           << pipeline::ast_scan_motion_dec_field << "\",\n"
           << "    \"observation_goal\": \"Science\",\n"
           << "    \"observation_program\": \"Lissajous\",\n"
           << "    \"scan_file_valid\": 1,\n"
           << "    \"source_epoch\": 2000.0,\n"
           << "    \"source_coordinate_system\": 0,\n"
           << "    \"nominal_cadence_hz\": 50.0,\n"
           << "    \"minimum_interval_sec\": "
           << telescope.minimum_interval_sec << ",\n"
           << "    \"maximum_interval_sec\": "
           << telescope.maximum_interval_sec << ",\n"
           << "    \"direct_adjacent_maximum_arcsec_per_sec\": "
           << telescope.direct_adjacent_maximum_arcsec_per_sec << ",\n"
           << "    \"direct_adjacent_maximizing_record\": "
           << telescope.direct_adjacent_maximizing_record << "\n"
           << "  },\n"
           << "  \"apt_bundle\": {\n"
           << "    \"manifest_sha256\": \""
           << citlali::utils::sha256_file(arguments.apt_manifest)
           << "\",\n"
           << "    \"semantic_sha256\": \""
           << json_escape(relation.bundle_identity().semantic_sha256)
           << "\",\n"
           << "    \"envelope_sha256\": \""
           << json_escape(relation.bundle_identity().envelope_sha256)
           << "\",\n"
           << "    \"participant_network_count\": "
           << mapping.networks.size() << "\n"
           << "  },\n"
           << "  \"identity_binding\": {\n"
           << "    \"requested\": " << identity_binding.requested << ",\n"
           << "    \"effective\": " << identity_binding.effective << ",\n"
           << "    \"observation_resolved\": "
           << identity_binding.observation_resolved << ",\n"
           << "    \"realized\": " << identity_binding.realized << "\n"
           << "  },\n"
           << "  \"raw_product\": {\n"
           << "    \"raw_direction_valid_count\": "
           << product.raw_direction_valid_count << ",\n"
           << "    \"quality_classified_count\": "
           << product.quality_classified_count << ",\n"
           << "    \"telemetry_defect_count\": "
           << product.defect_records.size() << ",\n"
           << "    \"telemetry_defect_records\": [";
    for (std::size_t index = 0; index < product.defect_records.size();
         ++index) {
        if (index != 0) output << ", ";
        output << product.defect_records[index];
    }
    const auto &summary = product.product->scan_summary();
    const auto memory = product.product->memory_evidence();
    output << "],\n"
           << "    \"realized_direction_valid_count\": "
           << product.realized_direction_valid_count << ",\n"
           << "    \"derivative_valid_count\": "
           << product.derivative_valid_count << ",\n"
           << "    \"maximum_available\": "
           << (summary.maximum_available ? "true" : "false") << ",\n"
           << "    \"maximum_causes\": "
           << static_cast<std::uint32_t>(summary.causes) << ",\n"
           << "    \"maximum_speed_arcsec_per_sec\": "
           << summary.maximum_speed_arcsec_per_sec << ",\n"
           << "    \"maximizing_record\": "
           << summary.maximizing_record << ",\n"
           << "    \"continuity_run_count\": "
           << summary.continuity_run_count << ",\n"
           << "    \"admitted_candidate_count\": "
           << summary.admitted_candidate_count << ",\n"
           << "    \"derived_record_bytes\": "
           << memory.derived_record_bytes << ",\n"
           << "    \"referenced_source_axis_count\": "
           << memory.referenced_source_axis_count << ",\n"
           << "    \"referenced_source_direction_plane_count\": "
           << memory.referenced_source_direction_plane_count << "\n"
           << "  },\n"
           << "  \"chunk_invariance\": {\n"
           << "    \"partition_count\": 3,\n"
           << "    \"record_mismatch_count\": "
           << product.record_mismatch_count << ",\n"
           << "    \"telemetry_support_mismatch_count\": "
           << product.telemetry_support_mismatch_count << ",\n"
           << "    \"derivative_support_mismatch_count\": "
           << product.derivative_support_mismatch_count << ",\n"
           << "    \"summary_mismatch_count\": "
           << product.summary_mismatch_count << "\n"
           << "  },\n"
           << "  \"network_mapping\": {\n"
           << "    \"timing_scope\": \"network-specific\",\n"
           << "    \"total_occurrence_count\": "
           << mapping.total_occurrence_count << ",\n"
           << "    \"available_count\": " << mapping.available_count
           << ",\n"
           << "    \"unavailable_count\": " << mapping.unavailable_count
           << ",\n"
           << "    \"support_count\": " << mapping.support_count << ",\n"
           << "    \"identity_mismatch_count\": "
           << mapping.identity_mismatch_count << ",\n"
           << "    \"support_mismatch_count\": "
           << mapping.support_mismatch_count << ",\n"
           << "    \"value_mismatch_count\": "
           << mapping.value_mismatch_count << ",\n"
           << "    \"missing_unavailable_cause_count\": "
           << mapping.missing_unavailable_cause_count << ",\n"
           << "    \"mapped_owned_bytes\": "
           << mapping.mapped_owned_bytes << ",\n"
           << "    \"nw0_first_time_unix_sec\": "
           << mapping.nw0_first_time_unix_sec << ",\n"
           << "    \"nw7_first_time_unix_sec\": "
           << mapping.nw7_first_time_unix_sec << ",\n"
           << "    \"nw0_nw7_times_distinct\": "
           << (mapping.nw0_nw7_distinct ? "true" : "false") << ",\n"
           << "    \"participants\": [\n";
    for (std::size_t index = 0; index < mapping.networks.size(); ++index) {
        const auto &network = mapping.networks[index];
        output << "      {\"network\": " << network.network
               << ", \"filename\": \"" << json_escape(network.filename)
               << "\", \"sha256\": \"" << network.sha256
               << "\", \"byte_count\": " << network.byte_count
               << ", \"occurrence_count\": " << network.occurrence_count
               << ", \"packet_discontinuity_count\": "
               << network.packet_discontinuity_count
               << ", \"available_count\": " << network.available_count
               << ", \"unavailable_count\": " << network.unavailable_count
               << ", \"telemetry_defect_cause_count\": "
               << network.telemetry_defect_cause_count
               << ", \"first_time_unix_sec\": "
               << network.first_time_unix_sec
               << ", \"last_time_unix_sec\": "
               << network.last_time_unix_sec << "}";
        if (index + 1 != mapping.networks.size()) output << ',';
        output << '\n';
    }
    output << "    ]\n"
           << "  },\n"
           << "  \"performance\": {\n"
           << "    \"raw_product_wall_time_sec\": "
           << product.wall_time_sec << ",\n"
           << "    \"raw_product_cpu_time_sec\": "
           << product.cpu_time_sec << ",\n"
           << "    \"network_mapping_wall_time_sec\": "
           << mapping.wall_time_sec << ",\n"
           << "    \"network_mapping_cpu_time_sec\": "
           << mapping.cpu_time_sec << ",\n"
           << "    \"process_peak_rss_bytes\": " << peak_rss << "\n"
           << "  },\n"
           << "  \"unexpected_error_count\": 0\n"
           << "}\n";
    require(static_cast<bool>(output),
            "failed to write AST acceptance record");
}

}  // namespace

int main(int argc, char **argv) {
    try {
        auto arguments = parse_arguments(argc, argv);
        require(full_lowercase_git_sha(arguments.source_revision),
                "source revision must be one full lowercase Git SHA");
        require(CITLALI_WP7_SOURCE_STATE_VERIFIED == 1 &&
                    CITLALI_WP7_IGNORED_SOURCE_STATE_VERIFIED == 1 &&
                    arguments.source_revision ==
                        CITLALI_WP7_SOURCE_REVISION,
                "source revision does not match the compiled Citlali revision");
        require(CITLALI_WP7_DEPENDENCY_STATE_VERIFIED == 1,
                "compiled dependency state is not verified");
        require(arguments.owner_run,
                "owner-run authorization is required");
        require(arguments.design_is_ancestor &&
                    arguments.align_repair_is_ancestor,
                "accepted design and ALIGN repair ancestry are required");
        arguments.executable_sha256 =
            citlali::utils::sha256_file(arguments.executable);

        const auto verified = apt::verify_bundle_filesystem(
            arguments.apt_manifest, true);
        const auto relation =
            pipeline::admit_canonical_apt_detector_relation_v2(verified);
        const auto &observation = relation.observation();
        require(observation.observation == 152390 &&
                    observation.subobservation == 0 &&
                    observation.scan == 2,
                "APT bundle is outside the approved observation scope");
        const pipeline::NativeObservationScope scope{
            observation.observation, observation.subobservation,
            observation.scan};
        const auto config = load_runtime_config(arguments.config);
        std::vector<NetworkInput> inputs;
        inputs.reserve(relation.raw_sources().size());
        for (const auto &source : relation.raw_sources()) {
            inputs.push_back(load_network_input(
                arguments.data_directory, source, config));
        }
        std::sort(inputs.begin(), inputs.end(), [](const auto &left,
                                                   const auto &right) {
            return left.network < right.network;
        });
        require(inputs.size() == 11,
                "representative APT bundle must contain 11 networks");

        const auto telescope = load_telescope(arguments.telescope, scope);
        const auto product = build_product_evidence(telescope.source);
        require(product.defect_records ==
                    std::vector<pipeline::AstTelescopeRecord>{2504, 12971},
                "real telemetry-defect identities disagree with owner evidence");
        require(product.product->scan_summary().maximum_available &&
                    product.product->scan_summary()
                            .maximum_speed_arcsec_per_sec > 200.0 &&
                    product.product->scan_summary()
                            .maximum_speed_arcsec_per_sec < 230.0,
                "real AST maximum is outside the approved evidence envelope");
        require(product.record_mismatch_count == 0 &&
                    product.telemetry_support_mismatch_count == 0 &&
                    product.derivative_support_mismatch_count == 0 &&
                    product.summary_mismatch_count == 0,
                "real AST product is not chunk-partition invariant");
        const auto mapping = build_mapping_evidence(
            scope, product.product, inputs);
        require(mapping.identity_mismatch_count == 0 &&
                    mapping.support_mismatch_count == 0 &&
                    mapping.value_mismatch_count == 0 &&
                    mapping.missing_unavailable_cause_count == 0 &&
                    mapping.nw0_nw7_distinct,
                "real network-specific AST mapping is nonconforming");
        write_record(arguments, relation, telescope, product, mapping,
                     peak_rss_bytes());
        std::cout << "WP-7 AST scan-motion acceptance record: "
                  << arguments.output << '\n';
        return 0;
    }
    catch (const std::exception &error) {
        std::cerr << "WP-7 AST scan-motion acceptance runner: FAIL: "
                  << error.what() << '\n';
        return 2;
    }
}
