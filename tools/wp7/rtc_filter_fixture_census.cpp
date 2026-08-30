#include "rtc_filter_fixture_census_model.h"

#include <citlali/core/pipeline/ast_scan_motion.h>
#include <citlali/core/pipeline/ast_scan_motion_alignment.h>
#include <citlali/core/pipeline/canonical_apt_bundle_v2.h>
#include <citlali/core/pipeline/canonical_apt_detector_relation_v2.h>
#include <citlali/core/utils/sha256.h>

#include <citlali_config/gitversion.h>

#include <netcdf.h>
#include <netcdf>
#include <yaml-cpp/yaml.h>

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace pipeline = citlali::pipeline;
namespace apt = citlali::pipeline::canonical_apt_v2;
namespace fixture = citlali::wp7::rtc_filter_fixture;

constexpr std::string_view schema =
    "citlali-wp7-rtc-filter-fixture-census-v2";
constexpr double maximum_cadence_fractional_uncertainty = 1.0e-4;

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

struct Arguments {
    std::string dataset_id;
    fs::path data_directory;
    fs::path telescope;
    fs::path apt_manifest;
    fs::path config;
    fs::path output;
    fs::path executable;
    std::string source_revision;
    bool source_clean = false;
    std::vector<std::pair<std::string, fs::path>> auxiliary_inputs;
};

std::string usage() {
    return
        "Usage: citlali_wp7_rtc_filter_fixture_census\n"
        "  --dataset-id ID --data-dir PATH --telescope PATH\n"
        "  --apt-manifest PATH --config PATH --output PATH\n"
        "  --source-revision FULL_SHA [--source-clean]\n"
        "  [--auxiliary-input ROLE PATH ...]\n";
}

Arguments parse_arguments(int argc, char **argv) {
    Arguments result;
    result.executable = fs::absolute(argv[0]);
    auto next = [&](int &index, std::string_view option) {
        require(index + 1 < argc,
                std::string(option) + " requires a value");
        return std::string{argv[++index]};
    };
    for (int index = 1; index < argc; ++index) {
        const std::string option{argv[index]};
        if (option == "--help" || option == "-h") {
            std::cout << usage();
            std::exit(0);
        } else if (option == "--dataset-id") {
            result.dataset_id = next(index, option);
        } else if (option == "--data-dir") {
            result.data_directory = next(index, option);
        } else if (option == "--telescope") {
            result.telescope = next(index, option);
        } else if (option == "--apt-manifest") {
            result.apt_manifest = next(index, option);
        } else if (option == "--config") {
            result.config = next(index, option);
        } else if (option == "--output") {
            result.output = next(index, option);
        } else if (option == "--source-revision") {
            result.source_revision = next(index, option);
        } else if (option == "--source-clean") {
            result.source_clean = true;
        } else if (option == "--auxiliary-input") {
            const auto role = next(index, option);
            result.auxiliary_inputs.emplace_back(
                role, next(index, option));
        } else {
            fail("unknown option: " + option);
        }
    }
    require(!result.dataset_id.empty(), "--dataset-id is required");
    require(fs::is_directory(result.data_directory),
            "--data-dir must name a directory");
    require(fs::is_regular_file(result.telescope),
            "--telescope must name a regular file");
    require(fs::is_regular_file(result.apt_manifest),
            "--apt-manifest must name a regular file");
    require(fs::is_regular_file(result.config),
            "--config must name a regular file");
    require(!result.output.empty(), "--output is required");
    require(result.source_revision.size() == 40 &&
                std::all_of(result.source_revision.begin(),
                            result.source_revision.end(),
                            [](unsigned char ch) {
                                return (ch >= '0' && ch <= '9') ||
                                    (ch >= 'a' && ch <= 'f');
                            }),
            "--source-revision must be one full lowercase Git SHA");
    for (const auto &[role, path] : result.auxiliary_inputs) {
        require(!role.empty(), "auxiliary input role is empty");
        require(fs::is_regular_file(path),
                "auxiliary input is not a regular file: " + path.string());
    }
    return result;
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
            } else {
                stream << static_cast<char>(ch);
            }
        }
    }
    return stream.str();
}

struct FileEvidence {
    std::string role;
    fs::path path;
    std::string sha256;
    std::uintmax_t byte_count = 0;
};

FileEvidence file_evidence(std::string role, const fs::path &path) {
    std::error_code error;
    const auto bytes = fs::file_size(path, error);
    require(!error, "unable to read input size: " + path.string());
    return {std::move(role), fs::absolute(path),
            citlali::utils::sha256_file(path), bytes};
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

struct SourceBinding {
    std::int64_t network = -1;
    std::string interface_name;
    std::int64_t channel_count = 0;
    std::string content_sha256;
    std::uint64_t byte_count = 0;
    apt::ObservationIdentity header_observation;
};

fs::path find_raw_file(const fs::path &directory,
                       const SourceBinding &source) {
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

struct CadenceEvidence {
    std::size_t interval_count = 0;
    double minimum_interval_sec = 0.0;
    double median_interval_sec = 0.0;
    double maximum_interval_sec = 0.0;
    double maximum_fractional_deviation = 0.0;
    bool uncertainty_within_margin = false;
};

struct NetworkInput {
    std::int64_t network = -1;
    fixture::Array array = fixture::Array::a1100;
    std::size_t detector_count = 0;
    FileEvidence file;
    double fpga_frequency_hz = 0.0;
    double sample_frequency_hz = 0.0;
    std::int64_t accumulation_length = 0;
    std::shared_ptr<const pipeline::NativeNetworkAlignment> timing;
    std::vector<pipeline::NativeContiguousRun> runs;
    CadenceEvidence cadence;
};

CadenceEvidence measure_cadence(
    const pipeline::NativeNetworkAlignment &timing,
    double sample_frequency_hz) {
    std::vector<double> intervals;
    const auto &times = timing.reconstructed_times_unix_sec();
    const auto &counters = timing.packet_counters();
    intervals.reserve(counters.size() - 1);
    for (std::size_t index = 1; index < counters.size(); ++index) {
        if (!pipeline::packet_counters_are_contiguous(
                counters[index - 1], counters[index])) {
            continue;
        }
        const double dt = times(static_cast<Eigen::Index>(index)) -
            times(static_cast<Eigen::Index>(index - 1));
        require(std::isfinite(dt) && dt > 0.0,
                "native contiguous interval is invalid");
        intervals.push_back(dt);
    }
    require(!intervals.empty(),
            "native timing has no measurable contiguous interval");
    CadenceEvidence result;
    result.interval_count = intervals.size();
    const auto [minimum, maximum] =
        std::minmax_element(intervals.begin(), intervals.end());
    result.minimum_interval_sec = *minimum;
    result.maximum_interval_sec = *maximum;
    auto ordered = intervals;
    std::sort(ordered.begin(), ordered.end());
    const auto middle = ordered.size() / 2;
    result.median_interval_sec = ordered[middle];
    if (ordered.size() % 2 == 0) {
        result.median_interval_sec =
            (ordered[middle - 1] + ordered[middle]) / 2.0;
    }
    const double nominal_interval = 1.0 / sample_frequency_hz;
    for (const double interval : intervals) {
        result.maximum_fractional_deviation = std::max(
            result.maximum_fractional_deviation,
            std::abs(interval / nominal_interval - 1.0));
    }
    result.uncertainty_within_margin =
        result.maximum_fractional_deviation <=
        maximum_cadence_fractional_uncertainty;
    return result;
}

NetworkInput load_network_input(
    const fs::path &data_directory,
    const SourceBinding &source,
    fixture::Array array, std::size_t detector_count,
    const RuntimeConfig &config) {
    require(source.network >= 0 && source.network < 13 &&
                config.interface_offset_present[
                    static_cast<std::size_t>(source.network)],
            "network source lacks a valid interface offset");
    const auto path = find_raw_file(data_directory, source);
    auto input_file = file_evidence("detector_timestream", path);
    require(input_file.byte_count == source.byte_count &&
                "sha256:" + input_file.sha256 == source.content_sha256,
            "raw detector file disagrees with APT source binding");

    netCDF::NcFile file(path.string(), netCDF::NcFile::read);
    const double fpga_frequency =
        read_netcdf_scalar<double>(file, "Header.Toltec.FpgaFreq");
    const double sample_frequency =
        read_netcdf_scalar<double>(file, "Header.Toltec.SampleFreq");
    const auto accumulation_length =
        read_netcdf_scalar<std::int64_t>(file, "Header.Toltec.AccumLen");
    require(std::isfinite(fpga_frequency) && fpga_frequency > 0.0 &&
                std::isfinite(sample_frequency) && sample_frequency > 0.0 &&
                accumulation_length > 0,
            "raw cadence header facts are invalid");
    require(std::abs(fpga_frequency / accumulation_length /
                         sample_frequency -
                     1.0) <= 16.0 * std::numeric_limits<double>::epsilon(),
            "SampleFreq disagrees with FpgaFreq/AccumLen");
    const auto variable = file.getVar("Data.Toltec.Ts");
    require(!variable.isNull() && variable.getDimCount() == 2 &&
                variable.getDim(1).getSize() == 6,
            "raw network timing plane is invalid");
    const auto row_count = variable.getDim(0).getSize();
    require(row_count > 1 &&
                row_count <= static_cast<std::size_t>(
                    std::numeric_limits<Eigen::Index>::max()),
            "raw network timing cardinality is invalid");
    Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic,
                  Eigen::RowMajor>
        timestamps(static_cast<Eigen::Index>(row_count), 6);
    variable.getVar(timestamps.data());
    auto timing =
        std::make_shared<const pipeline::NativeNetworkAlignment>(
            pipeline::make_native_network_alignment(
                source.network, 0, timestamps, fpga_frequency,
                config.interface_offsets_sec[
                    static_cast<std::size_t>(source.network)]));
    auto runs = pipeline::partition_native_contiguous_runs(
        *timing, timing->first_native_row(), timing->past_last_native_row());
    auto cadence = measure_cadence(*timing, sample_frequency);
    return {source.network, array, detector_count, std::move(input_file),
            fpga_frequency, sample_frequency, accumulation_length,
            std::move(timing), std::move(runs), cadence};
}

struct TelescopeEvidence {
    FileEvidence file;
    std::shared_ptr<const pipeline::AstScanMotionSource> source;
    double minimum_interval_sec = 0.0;
    double maximum_interval_sec = 0.0;
};

TelescopeEvidence load_telescope(
    const fs::path &path,
    const pipeline::NativeObservationScope &scope) {
    auto input_file = file_evidence("telescope", path);
    netCDF::NcFile file(path.string(), netCDF::NcFile::read);
    require(read_netcdf_scalar<std::int64_t>(file, "Header.Dcs.ObsNum") ==
                    scope.observation &&
                read_netcdf_scalar<std::int64_t>(
                    file, "Header.Dcs.SubObsNum") == scope.subobservation &&
                read_netcdf_scalar<std::int64_t>(
                    file, "Header.Dcs.ScanNum") == scope.scan,
            "telescope producer scope disagrees with APT observation");
    pipeline::AstScanMotionSourceMetadata metadata;
    metadata.producer_kind =
        pipeline::AstScanMotionProducerKind::real_toltec;
    metadata.dcs_observation_goal =
        read_netcdf_text(file, "Header.Dcs.ObsGoal");
    metadata.dcs_observation_program =
        read_netcdf_text(file, "Header.Dcs.ObsPgm");
    metadata.scan_file_valid =
        read_netcdf_scalar<std::int64_t>(file, "Header.ScanFile.Valid");
    metadata.source_epoch =
        read_netcdf_scalar<double>(file, "Header.Source.Epoch");
    metadata.source_coordinate_system =
        read_netcdf_scalar<std::int64_t>(file, "Header.Source.CoordSys");
    metadata.nominal_producer_cadence_hz = 50.0;
    metadata.field_registry = pipeline::AstScanMotionFieldRegistry::
        source_ra_act_source_dec_act_j2000_radians;
    metadata.source_artifact_identity = "sha256:" + input_file.sha256;

    const auto time_variable =
        file.getVar(std::string{pipeline::ast_scan_motion_time_field});
    require(!time_variable.isNull() && time_variable.getDimCount() == 1,
            "telescope time axis is missing or malformed");
    const auto count = time_variable.getDim(0).getSize();
    require(count > 1 &&
                count <= static_cast<std::size_t>(
                    std::numeric_limits<Eigen::Index>::max()),
            "telescope time cardinality is invalid");
    auto times = read_vector(
        file, std::string{pipeline::ast_scan_motion_time_field}, count,
        "sec");
    auto ra = read_vector(
        file, std::string{pipeline::ast_scan_motion_ra_field}, count,
        "rad");
    auto dec = read_vector(
        file, std::string{pipeline::ast_scan_motion_dec_field}, count,
        "rad");
    TelescopeEvidence result;
    result.file = std::move(input_file);
    result.minimum_interval_sec = std::numeric_limits<double>::infinity();
    for (Eigen::Index index = 1; index < times.size(); ++index) {
        const double dt = times(index) - times(index - 1);
        require(std::isfinite(dt) && dt > 0.0,
                "telescope time axis is not strictly increasing");
        result.minimum_interval_sec =
            std::min(result.minimum_interval_sec, dt);
        result.maximum_interval_sec =
            std::max(result.maximum_interval_sec, dt);
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
        left.admitted_candidate_count == right.admitted_candidate_count &&
        left.telemetry_defect_count == right.telemetry_defect_count;
}

struct ProductEvidence {
    std::shared_ptr<const pipeline::AstScanMotionProduct> product;
    std::size_t chunk_record_mismatch_count = 0;
    bool chunk_summary_matches = false;
};

ProductEvidence build_product_evidence(
    const std::shared_ptr<const pipeline::AstScanMotionSource> &source,
    std::int64_t observation) {
    const auto base = static_cast<std::uint64_t>(observation) * 10000U;
    const pipeline::AstScanMotionIdentityBinding binding{
        base + 1U, base + 2U, base + 3U, base + 4U};
    auto product = pipeline::build_ast_scan_motion_product(source, binding);
    const auto one_third = static_cast<pipeline::AstTelescopeRecord>(
        source->record_count() / 3);
    const auto two_thirds = static_cast<pipeline::AstTelescopeRecord>(
        2 * source->record_count() / 3);
    const std::vector<pipeline::AstScanMotionProcessingSpan> schedule{
        {two_thirds, source->past_last_record()},
        {source->first_record(), one_third},
        {one_third, two_thirds}};
    const auto partitioned = pipeline::build_ast_scan_motion_product(
        source, binding, schedule);
    ProductEvidence result;
    result.product = std::move(product);
    for (std::size_t index = 0; index < result.product->record_count();
         ++index) {
        result.chunk_record_mismatch_count += !same_record(
            result.product->record_at_local(index),
            partitioned->record_at_local(index));
    }
    result.chunk_summary_matches = same_summary(
        result.product->scan_summary(), partitioned->scan_summary());
    return result;
}

struct NetworkMotionEvidence {
    std::vector<double> mapped_speeds_arcsec_per_sec;
    std::vector<std::uint8_t> continues_previous;
};

struct MappingEvidence {
    std::map<std::int64_t, std::size_t> available_by_network;
    std::map<std::int64_t, std::size_t> unavailable_by_network;
    std::map<std::int64_t, std::map<std::uint32_t, std::size_t>>
        unavailable_causes_by_network;
    std::map<std::int64_t, NetworkMotionEvidence> motion_by_network;
    std::size_t identity_mismatch_count = 0;
    std::size_t missing_support_count = 0;
};

MappingEvidence map_ast_to_networks(
    const pipeline::NativeObservationScope &scope,
    const std::shared_ptr<const pipeline::AstScanMotionProduct> &product,
    const std::vector<NetworkInput> &inputs) {
    std::vector<std::shared_ptr<const pipeline::NativeNetworkAlignment>>
        timings;
    for (const auto &input : inputs) timings.push_back(input.timing);
    const auto views = pipeline::AstScanMotionNetworkViews::admit(
        scope, product, std::move(timings));
    MappingEvidence result;
    for (const auto &input : inputs) {
        const auto &view = views->network(input.network);
        result.available_by_network.try_emplace(input.network, 0);
        result.unavailable_by_network.try_emplace(input.network, 0);
        result.unavailable_causes_by_network.try_emplace(input.network);
        auto [motion_position, inserted] =
            result.motion_by_network.try_emplace(input.network);
        require(inserted, "AST mapping repeats a participant network");
        auto &motion = motion_position->second;
        motion.mapped_speeds_arcsec_per_sec.assign(
            view.occurrence_count(),
            std::numeric_limits<double>::quiet_NaN());
        motion.continues_previous.assign(view.occurrence_count(), 0U);
        for (pipeline::TimestreamNativeRow row = view.first_native_row();
             row < view.past_last_native_row(); ++row) {
            const auto local = static_cast<std::size_t>(
                row - view.first_native_row());
            result.identity_mismatch_count +=
                !(view.identity(row) == input.timing->identity(row));
            if (local != 0U) {
                const auto &counters = input.timing->packet_counters();
                motion.continues_previous[local] =
                    pipeline::packet_counters_are_contiguous(
                        counters[local - 1U], counters[local])
                    ? 1U
                    : 0U;
            }
            const auto &record = view.record(row);
            if (record.available()) {
                ++result.available_by_network[input.network];
                result.missing_support_count += !view.support(row).has_value();
                motion.mapped_speeds_arcsec_per_sec[local] =
                    record.scalar_speed_arcsec_per_sec();
            } else {
                ++result.unavailable_by_network[input.network];
                ++result.unavailable_causes_by_network[input.network]
                      [static_cast<std::uint32_t>(record.causes())];
            }
        }
    }
    return result;
}

struct CadenceGroup {
    fixture::Array array = fixture::Array::a1100;
    double sample_frequency_hz = 0.0;
    std::vector<std::int64_t> networks;
    bool cadence_uncertainty_within_margin = true;
};

std::vector<CadenceGroup> cadence_groups(
    const std::vector<NetworkInput> &inputs) {
    using Key = std::pair<std::int64_t, std::uint64_t>;
    std::map<Key, CadenceGroup> grouped;
    for (const auto &input : inputs) {
        const Key key{static_cast<std::int64_t>(input.array),
                      bit_pattern(input.sample_frequency_hz)};
        auto [position, inserted] = grouped.try_emplace(key);
        if (inserted) {
            position->second.array = input.array;
            position->second.sample_frequency_hz =
                input.sample_frequency_hz;
        }
        position->second.networks.push_back(input.network);
        position->second.cadence_uncertainty_within_margin =
            position->second.cadence_uncertainty_within_margin &&
            input.cadence.uncertainty_within_margin;
    }
    std::vector<CadenceGroup> result;
    for (auto &[key, group] : grouped) {
        std::sort(group.networks.begin(), group.networks.end());
        result.push_back(std::move(group));
    }
    return result;
}

const NetworkInput &network_input(
    const std::vector<NetworkInput> &inputs, std::int64_t network) {
    const auto position = std::find_if(
        inputs.begin(), inputs.end(), [network](const auto &input) {
            return input.network == network;
        });
    require(position != inputs.end(),
            "cadence group names an absent participant network");
    return *position;
}

void write_occurrence_admission(
    std::ostream &output, const NetworkInput &input,
    const NetworkMotionEvidence &motion,
    double upper_speed_ceiling_arcsec_per_sec) {
    const auto admission = fixture::summarize_occurrence_admission(
        motion.mapped_speeds_arcsec_per_sec,
        motion.continues_previous,
        upper_speed_ceiling_arcsec_per_sec);
    const double primitive_occurrence_duration_sec =
        static_cast<double>(input.accumulation_length) /
        input.fpga_frequency_hz;
    const auto duration = [&](std::size_t count) {
        return static_cast<double>(count) *
            primitive_occurrence_duration_sec;
    };
    output << "{\"network\": " << input.network
           << ", \"occurrence_count\": " << admission.occurrence_count
           << ", \"ast_unavailable_count\": "
           << admission.ast_unavailable_count
           << ", \"ast_unavailable_duration_sec\": "
           << duration(admission.ast_unavailable_count)
           << ", \"below_minimum_science_scan_speed_count\": "
           << admission.below_minimum_science_speed_count
           << ", \"below_minimum_science_scan_speed_duration_sec\": "
           << duration(admission.below_minimum_science_speed_count)
           << ", \"base_admitted_count\": "
           << admission.base_admitted_count
           << ", \"base_admitted_duration_sec\": "
           << duration(admission.base_admitted_count)
           << ", \"upper_speed_admitted_count\": "
           << admission.upper_speed_admitted_count
           << ", \"upper_speed_admitted_duration_sec\": "
           << duration(admission.upper_speed_admitted_count)
           << ", \"scan_speed_above_mode_support_count\": "
           << admission.scan_speed_above_mode_support_count
           << ", \"scan_speed_above_mode_support_duration_sec\": "
           << duration(admission.scan_speed_above_mode_support_count)
           << ", \"retained_run_count\": "
           << admission.retained_run_count
           << ", \"longest_retained_run_occurrences\": "
           << admission.longest_retained_run_occurrences
           << ", \"longest_retained_run_duration_sec\": "
           << duration(admission.longest_retained_run_occurrences) << '}';
}

void write_file(std::ostream &output, const FileEvidence &file) {
    output << "{\"role\": \"" << json_escape(file.role)
           << "\", \"path\": \"" << json_escape(file.path.string())
           << "\", \"sha256\": \"" << file.sha256
           << "\", \"byte_count\": " << file.byte_count << '}';
}

void write_record(
    const Arguments &arguments, const apt::VerifiedBundle &bundle,
    bool matched_relation_available,
    const FileEvidence &config_file,
    const TelescopeEvidence &telescope,
    const std::vector<FileEvidence> &auxiliary_inputs,
    const std::vector<NetworkInput> &inputs,
    const ProductEvidence &product,
    const MappingEvidence &mapping,
    const std::vector<CadenceGroup> &groups) {
    std::ofstream output(arguments.output);
    require(static_cast<bool>(output),
            "unable to open fixture census output");
    const auto executable = file_evidence("executable", arguments.executable);
    const auto apt_manifest =
        file_evidence("apt_manifest", arguments.apt_manifest);
    const auto &scope = bundle.apt.observation;
    const auto &summary = product.product->scan_summary();
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"" << schema << "\",\n"
           << "  \"dataset_id\": \""
           << json_escape(arguments.dataset_id) << "\",\n"
           << "  \"source_revision\": \""
           << arguments.source_revision << "\",\n"
           << "  \"source_clean_asserted\": "
           << (arguments.source_clean ? "true" : "false") << ",\n"
           << "  \"citlali_version\": \""
           << json_escape(CITLALI_GIT_VERSION) << "\",\n"
           << "  \"compiler\": \"" << json_escape(__VERSION__)
           << "\",\n"
           << "  \"cplusplus\": " << __cplusplus << ",\n"
           << "  \"build_identity\": {\n"
           << "    \"executable_sha256\": \""
           << executable.sha256 << "\",\n"
           << "    \"netcdf_library\": \""
           << json_escape(nc_inq_libvers()) << "\",\n"
           << "    \"eigen_version\": \"" << EIGEN_WORLD_VERSION
           << '.' << EIGEN_MAJOR_VERSION << '.' << EIGEN_MINOR_VERSION
           << "\",\n"
           << "    \"execution_thread_count\": 1,\n"
           << "    \"ast_engineering_partition_count\": 3,\n"
           << "    \"rtc_arithmetic_controls\": \"not-applicable-no-rtc-operator\"\n"
           << "  },\n"
           << "  \"numerical_policy_id\": \""
           << fixture::numerical_policy_id << "\",\n"
           << "  \"speed_admission_policy_id\": \""
           << fixture::speed_admission_policy_id << "\",\n"
           << "  \"observation\": " << scope.observation << ",\n"
           << "  \"subobservation\": " << scope.subobservation << ",\n"
           << "  \"scan\": " << scope.scan << ",\n"
           << "  \"common_analysis_grid_requested\": false,\n"
           << "  \"rtc_route_activated\": false,\n"
           << "  \"filter_coefficients_present\": false,\n"
           << "  \"persistent_scientific_product_published\": false,\n"
           << "  \"inputs\": [\n    ";
    write_file(output, executable);
    output << ",\n    ";
    write_file(output, config_file);
    output << ",\n    ";
    write_file(output, apt_manifest);
    output << ",\n    ";
    write_file(output, telescope.file);
    for (const auto &input : inputs) {
        output << ",\n    ";
        write_file(output, input.file);
    }
    for (const auto &input : auxiliary_inputs) {
        output << ",\n    ";
        write_file(output, input);
    }
    output << "\n  ],\n"
           << "  \"apt_bundle\": {\n"
           << "    \"bundle_kind\": \""
           << apt::bundle_kind_token(bundle.manifest.kind) << "\",\n"
           << "    \"canonical_bundle_verified\": true,\n"
           << "    \"detector_raw_inventory_complete\": true,\n"
           << "    \"semantic_sha256\": \""
           << json_escape(bundle.identity.semantic_sha256)
           << "\",\n"
           << "    \"envelope_sha256\": \""
           << json_escape(bundle.identity.envelope_sha256)
           << "\",\n"
           << "    \"matched_detector_relation_available\": "
           << (matched_relation_available ? "true" : "false") << ",\n"
           << "    \"verified_total_byte_count\": "
           << bundle.total_byte_count << ",\n"
           << "    \"participant_network_count\": "
           << inputs.size() << ",\n"
           << "    \"detector_count\": " << bundle.apt.rows.size()
           << ",\n"
           << "    \"root_receipt_sha256\": \""
           << citlali::utils::sha256(bundle.payload.root_receipt_bytes)
           << "\",\n"
           << "    \"verified_components\": [\n";
    std::size_t component_index = 0;
    for (const auto &[role, bytes] :
         bundle.payload.component_bytes_by_role) {
        output << "      {\"role\": \"" << json_escape(role)
               << "\", \"sha256\": \""
               << citlali::utils::sha256(bytes)
               << "\", \"byte_count\": " << bytes.size() << '}';
        if (++component_index !=
            bundle.payload.component_bytes_by_role.size()) {
            output << ',';
        }
        output << '\n';
    }
    output << "    ]\n  },\n"
           << "  \"telescope_ast\": {\n"
           << "    \"policy_id\": \""
           << pipeline::ast_scan_motion_policy_id << "\",\n"
           << "    \"observation_goal\": \""
           << json_escape(telescope.source->metadata().dcs_observation_goal)
           << "\",\n"
           << "    \"observation_program\": \""
           << json_escape(
                  telescope.source->metadata().dcs_observation_program)
           << "\",\n"
           << "    \"record_count\": "
           << telescope.source->record_count() << ",\n"
           << "    \"minimum_interval_sec\": "
           << telescope.minimum_interval_sec << ",\n"
           << "    \"maximum_interval_sec\": "
           << telescope.maximum_interval_sec << ",\n"
           << "    \"maximum_available\": "
           << (summary.maximum_available ? "true" : "false") << ",\n"
           << "    \"maximum_causes\": "
           << static_cast<std::uint32_t>(summary.causes) << ",\n"
           << "    \"maximum_speed_arcsec_per_sec\": ";
    if (summary.maximum_available) {
        output << summary.maximum_speed_arcsec_per_sec;
    } else {
        output << "null";
    }
    output << ",\n"
           << "    \"maximizing_record\": "
           << summary.maximizing_record << ",\n"
           << "    \"telemetry_defect_count\": "
           << summary.telemetry_defect_count << ",\n"
           << "    \"chunk_record_mismatch_count\": "
           << product.chunk_record_mismatch_count << ",\n"
           << "    \"chunk_summary_matches\": "
           << (product.chunk_summary_matches ? "true" : "false")
           << "\n  },\n"
           << "  \"network_native_census\": [\n";
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        const auto &input = inputs[index];
        std::size_t shortest_run = std::numeric_limits<std::size_t>::max();
        std::size_t longest_run = 0;
        for (const auto &run : input.runs) {
            const auto length = static_cast<std::size_t>(run.row_count());
            shortest_run = std::min(shortest_run, length);
            longest_run = std::max(longest_run, length);
        }
        output << "    {\"network\": " << input.network
               << ", \"array\": \"" << fixture::array_name(input.array)
               << "\", \"detector_count\": " << input.detector_count
               << ", \"occurrence_count\": " << input.timing->row_count()
               << ", \"first_time_unix_sec\": "
               << input.timing->reconstructed_times_unix_sec()(0)
               << ", \"last_time_unix_sec\": "
               << input.timing->reconstructed_times_unix_sec()(
                      input.timing->row_count() - 1)
               << ", \"sample_frequency_hz\": "
               << input.sample_frequency_hz
               << ", \"fpga_frequency_hz\": "
               << input.fpga_frequency_hz
               << ", \"accumulation_length\": "
               << input.accumulation_length
               << ", \"primitive_occurrence_duration_sec\": "
               << static_cast<double>(input.accumulation_length) /
                      input.fpga_frequency_hz
               << ", \"contiguous_run_count\": " << input.runs.size()
               << ", \"shortest_run_occurrences\": " << shortest_run
               << ", \"longest_run_occurrences\": " << longest_run
               << ", \"interval_count\": "
               << input.cadence.interval_count
               << ", \"minimum_interval_sec\": "
               << input.cadence.minimum_interval_sec
               << ", \"median_interval_sec\": "
               << input.cadence.median_interval_sec
               << ", \"maximum_interval_sec\": "
               << input.cadence.maximum_interval_sec
               << ", \"maximum_fractional_deviation\": "
               << input.cadence.maximum_fractional_deviation
               << ", \"cadence_uncertainty_within_100ppm\": "
               << (input.cadence.uncertainty_within_margin
                       ? "true" : "false")
               << ", \"mapped_ast_available_count\": "
               << mapping.available_by_network.at(input.network)
               << ", \"mapped_ast_unavailable_count\": "
               << mapping.unavailable_by_network.at(input.network)
               << ", \"mapped_ast_unavailable_causes\": {";
        const auto &cause_counts =
            mapping.unavailable_causes_by_network.at(input.network);
        std::size_t cause_index = 0;
        for (const auto &[cause, count] : cause_counts) {
            if (cause_index++ != 0U) output << ", ";
            output << '\"' << cause << "\": " << count;
        }
        output << "}}";
        if (index + 1 != inputs.size()) output << ',';
        output << '\n';
    }
    output << "  ],\n"
           << "  \"mapping_checks\": {\n"
           << "    \"timing_scope\": \"network-specific\",\n"
           << "    \"identity_mismatch_count\": "
           << mapping.identity_mismatch_count << ",\n"
           << "    \"missing_support_count\": "
           << mapping.missing_support_count << "\n  },\n"
           << "  \"candidate_mode_domains\": [\n";
    for (std::size_t group_index = 0; group_index < groups.size();
         ++group_index) {
        const auto &group = groups[group_index];
        output << "    {\"array\": \""
               << fixture::array_name(group.array)
               << "\", \"sample_frequency_hz\": "
               << group.sample_frequency_hz << ", \"networks\": [";
        for (std::size_t index = 0; index < group.networks.size(); ++index) {
            if (index != 0) output << ", ";
            output << group.networks[index];
        }
        output << "], \"cadence_uncertainty_within_100ppm\": "
               << (group.cadence_uncertainty_within_margin
                       ? "true" : "false")
               << ", \"duration_basis\": "
                  "\"occurrence-count-times-accumulation-length-divided-by-fpga-frequency\""
               << ", \"automatic_factor_selection_authorized\": false"
               << ", \"factor_candidates\": [\n";
        for (int factor = fixture::minimum_factor;
             factor <= fixture::maximum_factor; ++factor) {
            const auto evidence = fixture::evaluate_structural_mode(
                group.array, group.sample_frequency_hz, factor);
            std::string_view status =
                evidence.has_science_speed_domain()
                ? "structural-upper-bound-available"
                : "structural-domain-below-minimum-science-speed";
            if (!group.cadence_uncertainty_within_margin) {
                status = "cadence-uncertainty-exceeds-margin";
            }
            output << "      {\"factor\": " << factor
                   << ", \"status\": \"" << status
                   << "\", \"ceiling_status\": "
                      "\"structural-upper-bound-pending-filter-certification\""
                   << ", \"output_sample_rate_hz\": "
                   << evidence.output_sample_rate_hz
                   << ", \"safe_output_sample_rate_hz\": "
                   << evidence.safe_output_sample_rate_hz
                   << ", \"safe_output_nyquist_hz\": "
                   << evidence.safe_output_nyquist_hz
                   << ", \"wavelength_m\": " << evidence.wavelength_m
                   << ", \"airy_fwhm_arcsec\": "
                   << evidence.airy_fwhm_arcsec
                   << ", \"science_band_ceiling_arcsec_per_sec\": "
                   << evidence.science_band_ceiling_arcsec_per_sec
                   << ", \"beam_sampling_ceiling_arcsec_per_sec\": "
                   << evidence.beam_sampling_ceiling_arcsec_per_sec
                   << ", \"upper_speed_ceiling_arcsec_per_sec\": "
                   << evidence.upper_speed_ceiling_arcsec_per_sec
                   << ", \"upper_boundary_inclusive\": true"
                   << ", \"governing_constraint\": \""
                   << fixture::constraint_name(
                          evidence.governing_constraint)
                   << "\", \"minimum_science_speed_arcsec_per_sec\": "
                   << fixture::minimum_science_speed_arcsec_per_sec
                   << ", \"upper_speed_typed_cause\": "
                      "\"scan_speed_above_mode_support\""
                   << ", \"occurrence_admission_by_network\": [\n";
            for (std::size_t network_index = 0;
                 network_index < group.networks.size(); ++network_index) {
                const auto network = group.networks[network_index];
                output << "        ";
                write_occurrence_admission(
                    output, network_input(inputs, network),
                    mapping.motion_by_network.at(network),
                    evidence.upper_speed_ceiling_arcsec_per_sec);
                if (network_index + 1U != group.networks.size()) {
                    output << ',';
                }
                output << '\n';
            }
            output << "      ], \"support_erosion\": {";
            if (factor == 1) {
                output << "\"status\": "
                          "\"exact-occurrence-local-m1-no-filter\", "
                          "\"support_eroded_output_count\": 0, "
                          "\"support_eroded_output_duration_sec\": 0";
            } else {
                output << "\"status\": "
                          "\"pending-exact-filter-coefficients-and-half-support\", "
                          "\"support_eroded_output_count\": null, "
                          "\"support_eroded_output_duration_sec\": null";
            }
            output << "}}";
            if (factor != fixture::maximum_factor) output << ',';
            output << '\n';
        }
        output << "    ]}";
        if (group_index + 1 != groups.size()) output << ',';
        output << '\n';
    }
    output << "  ],\n"
           << "  \"d0_fixture_identity_ready\": "
           << (arguments.source_clean ? "true" : "false") << ",\n"
           << "  \"structural_screen_only\": true,\n"
           << "  \"automatic_factor_selection_authorized\": false,\n"
           << "  \"filter_bank_certified\": false,\n"
           << "  \"unexpected_error_count\": 0\n"
           << "}\n";
    require(static_cast<bool>(output),
            "failed to write fixture census record");
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const auto arguments = parse_arguments(argc, argv);
        const auto bundle = apt::verify_bundle_filesystem(
            arguments.apt_manifest, true);
        const bool matched_relation_available = bundle.relation.has_value();
        std::optional<pipeline::CanonicalAptDetectorRelationV2> relation;
        if (matched_relation_available) {
            relation.emplace(
                pipeline::admit_canonical_apt_detector_relation_v2(bundle));
        }
        const auto &observation = bundle.apt.observation;
        const pipeline::NativeObservationScope scope{
            observation.observation, observation.subobservation,
            observation.scan};
        const auto config = load_runtime_config(arguments.config);

        std::map<std::int64_t, std::set<std::int64_t>> arrays_by_network;
        std::map<std::int64_t, std::size_t> detectors_by_network;
        for (const auto &row : bundle.apt.rows) {
            arrays_by_network[row.network].insert(row.array);
            ++detectors_by_network[row.network];
        }
        std::vector<SourceBinding> sources;
        sources.reserve(bundle.sources.size());
        for (const auto &source : bundle.sources) {
            if (source.role != apt::SourceRole::raw) continue;
            sources.push_back({
                source.network, source.interface_name, source.channel_count,
                source.content_sha256, source.byte_count,
                source.header_observation});
        }
        require(!sources.empty(),
                "canonical APT bundle has no raw participant inventory");
        if (relation) {
            require(sources.size() == relation->raw_sources().size() &&
                        bundle.apt.rows.size() == relation->bindings().size(),
                    "matched relation disagrees with bundle inventory");
        }
        std::vector<NetworkInput> inputs;
        for (const auto &source : sources) {
            const auto &arrays = arrays_by_network.at(source.network);
            require(arrays.size() == 1,
                    "one raw network spans multiple array identities");
            inputs.push_back(load_network_input(
                arguments.data_directory, source,
                fixture::array_from_index(*arrays.begin()),
                detectors_by_network.at(source.network), config));
        }
        std::sort(inputs.begin(), inputs.end(),
                  [](const auto &left, const auto &right) {
                      return left.network < right.network;
                  });

        const auto telescope = load_telescope(arguments.telescope, scope);
        const auto product = build_product_evidence(
            telescope.source, observation.observation);
        require(product.chunk_record_mismatch_count == 0 &&
                    product.chunk_summary_matches,
                "AST product is not chunk-partition invariant");
        const auto mapping = map_ast_to_networks(
            scope, product.product, inputs);
        require(mapping.identity_mismatch_count == 0 &&
                    mapping.missing_support_count == 0,
                "network AST mapping lost identity or support");

        std::vector<FileEvidence> auxiliary;
        auxiliary.reserve(arguments.auxiliary_inputs.size());
        for (const auto &[role, path] : arguments.auxiliary_inputs) {
            auxiliary.push_back(file_evidence(role, path));
        }
        write_record(arguments, bundle, matched_relation_available,
                     file_evidence("effective_config", arguments.config),
                     telescope, auxiliary, inputs, product, mapping,
                     cadence_groups(inputs));
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "WP-7 RTC filter fixture census failed: "
                  << error.what() << '\n';
        return 1;
    }
}
