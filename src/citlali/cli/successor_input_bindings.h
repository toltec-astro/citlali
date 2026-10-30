#pragma once

// Internal application-boundary loaders shared by native execution and
// acceptance tooling. YAML/NetCDF/Tune files end at these concrete products;
// scientific admission remains with the existing typed pipeline owners.
#include <citlali/core/pipeline/ast_scan_motion.h>
#include <citlali/core/pipeline/canonical_apt_detector_relation_v2.h>
#include <citlali/core/pipeline/timestream_native_paired_readout.h>
#include <netcdf>
#include "successor_experiment_profiles.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace citlali::cli::input_bindings {
namespace fs = std::filesystem;
namespace pipeline = citlali::pipeline;

[[noreturn]] void fail(const std::string &message);
void require(bool condition, const std::string &message);

// Frozen identity-route experiment defaults. In particular the mapping domain
// is intentionally still observation 152390; execution callers that already
// replace it retain that responsibility. Do not infer scope from a filename.
using experiment_profiles::IdentityRouteExperimentProfile;

inline constexpr auto telescope_filename = IdentityRouteExperimentProfile::telescope_filename;
inline constexpr auto telescope_sha256 = IdentityRouteExperimentProfile::telescope_sha256;
inline constexpr auto telescope_byte_count = IdentityRouteExperimentProfile::telescope_byte_count;
inline constexpr auto telescope_record_count = IdentityRouteExperimentProfile::telescope_record_count;
inline constexpr auto ast_identity_binding = IdentityRouteExperimentProfile::ast_identity_binding;

inline constexpr std::string_view producer_interface =
    "TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE v0.1/r0.1";
inline constexpr std::string_view producer_interface_sha256 =
    "f9659b34a49a07d4287c4a70db798cdd2ec30049531da603fcca1e9d1fdd5969";

enum class NativeEventTimeRole : std::uint8_t {
    integration_start,
    integration_center,
    integration_end,
};

template <class T>
T read_netcdf_scalar(netCDF::NcFile &file, const std::string &name) {
    const auto variable = file.getVar(name);
    require(!variable.isNull(), "missing NetCDF scalar " + name);
    T result{};
    variable.getVar(&result);
    return result;
}

struct TelescopeInput {
    std::shared_ptr<const pipeline::AstScanMotionSource> source;
    std::string sha256;
    std::uintmax_t byte_count = 0;
};

// A named request for the established telescope-binding profiles. Flags and
// guards preserve the existing identity, repeat-census, POINT, and bounded OOF
// behavior; a configured digest is an exact external input association.
struct TelescopeBindingProfile {
    bool bounded_repeat_census = false;
    bool bounded_point_map = false;
    std::string configured_digest;
    bool bounded_oof = false;
};

TelescopeInput load_telescope(
    const fs::path &path, const pipeline::NativeObservationScope &scope,
    const TelescopeBindingProfile &profile = {});

struct RuntimeConfig {
    std::array<double, 13> interface_offsets_sec{};
    std::array<bool, 13> interface_offset_present{};
    std::string kids_model;
};

RuntimeConfig load_runtime_config(const fs::path &path);

void verify_raw_file(const fs::path &path,
                     const pipeline::CanonicalAptRawSourceBindingV2 &source);

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

struct TuneFacts {
    std::vector<bool> valid;
    std::int64_t accumulation_length = 0;
    std::int64_t observation = 0;
    std::int64_t subobservation = 0;
    std::int64_t scan = 0;
    std::int64_t network = -1;
};

TuneFacts read_tune_facts(const fs::path &path, std::int64_t channel_count);

// Ts rows retain their native occurrence order, with six integer producer
// fields. Cadence outputs are Hz and accumulation length is the exact count.
Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
read_timestamp_slice(const fs::path &path, std::int64_t first_row,
                     std::int64_t row_count, double &fpga_frequency_hz,
                     double &sample_frequency_hz,
                     std::int64_t &accumulation_length);

std::vector<pipeline::NativeReadoutDetectorBinding> detector_axis(
    const pipeline::CanonicalAptDetectorRelationV2 &relation,
    const NetworkInput &input);
std::shared_ptr<const pipeline::NativeReadoutMappingAuthority> mapping_identity(
    const NetworkInput &input, const RuntimeConfig &config);
std::shared_ptr<const pipeline::NativePairedReadoutOccurrenceAxis> occurrence_axis(
    const NetworkInput &input, std::int64_t first_native_row,
    std::int64_t native_row_count, NativeEventTimeRole event_time_role);
std::vector<pipeline::NativeReadoutCoordinateState> member_states(
    const pipeline::NativePairedReadoutMatrix &values,
    const std::vector<bool> &tune_valid);

// Owns only scratch normalized Tune copies; the supplied source stays read-only.
class TemporaryDirectory {
public:
    TemporaryDirectory();
    TemporaryDirectory(const TemporaryDirectory &) = delete;
    TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;
    ~TemporaryDirectory();
    const fs::path &path() const noexcept { return path_; }
private:
    fs::path path_;
};
fs::path normalized_tune_report(const NetworkInput &input,
                                const fs::path &directory);

} // namespace citlali::cli::input_bindings
