#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <numeric>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <netcdf>

#include <citlali/core/config/runtime_config.h>
#include <citlali/core/pipeline/reduction_config_netcdf.h>
#include <citlali/core/pipeline/rtc_learned_sampling_metrics.h>
#include <citlali/core/timestream/rtc/filter.h>
#include <citlali/core/utils/netcdf_io.h>
#include <citlali/core/utils/sha256.h>


namespace citlali::pipeline {

#include <citlali/core/pipeline/rtcdiag_layout_config.h>
#include <citlali/core/pipeline/rtcdiag_scan_summary.h>
#include <citlali/core/pipeline/rtcdiag_detector_outputs.h>
#include <citlali/core/pipeline/rtcdiag_network_outputs.h>
#include <citlali/core/pipeline/rtcdiag_impulsive_capture.h>
#include <citlali/core/pipeline/rtcdiag_tod_stream.h>

inline constexpr const char *rtcdiag_successor_contract_epoch =
    "sci-rtc-001-stage-a-successor-2026-08-11";
inline constexpr const char *rtcdiag_raw_input_manifest_schema =
    "citlali-rtcdiag-raw-input-manifest-v1";
inline constexpr const char *rtcdiag_raw_input_manifest_reference =
    "embedded:citlali-rtcdiag-raw-input-manifest-v1";

struct RtcdiagSuccessorProductIdentity {
    std::string mode;
    std::string contract_id;
    std::string validation_profile_id;
};

inline RtcdiagSuccessorProductIdentity rtcdiag_successor_identity_for_mode(
    std::string_view mode) {
    if (mode == "point") {
        return {"point", "sci-rtc-001-stage-a-successor-point-products-v1",
                "sci-rtc-001-stage-a-successor-v1"};
    }
    if (mode == "oof") {
        return {"oof", "sci-rtc-001-stage-a-successor-oof-products-v1",
                "sci-rtc-001-stage-a-successor-oof-v1"};
    }
    if (mode == "beammap") {
        return {std::string{mode},
                "sci-rtc-001-stage-a-successor-beammap-products-v1",
                "sci-rtc-001-stage-a-successor-beammap-v1"};
    }
    if (mode == "science") {
        return {"science",
                "sci-rtc-001-stage-a-successor-science-products-v1",
                "sci-rtc-001-stage-a-successor-science-v1"};
    }
    throw DataIOError("unsupported rtcdiag successor product mode: " +
                      std::string{mode});
}

inline std::string rtcdiag_successor_mode(
    citlali::config::ReductionType reduction_type,
    std::string observation_goal) {
    if (citlali::config::is_beammap_reduction_type(reduction_type)) {
        return "beammap";
    }
    if (citlali::config::is_science_reduction_type(reduction_type)) {
        return "science";
    }
    std::transform(observation_goal.begin(), observation_goal.end(),
                   observation_goal.begin(), [](unsigned char value) {
                       return static_cast<char>(std::tolower(value));
                   });
    return observation_goal.find("oof") != std::string::npos ? "oof"
                                                               : "point";
}

struct RtcdiagRawInputManifest {
    std::string canonical_bytes;
    std::string sha256;
};

inline bool validate_rtcdiag_raw_input_manifest_bytes(
    const std::string &bytes) {
    std::istringstream stream(bytes);
    std::string line;
    if (!std::getline(stream, line) || line != rtcdiag_raw_input_manifest_schema) {
        return false;
    }
    const auto read_field = [&](std::string_view name,
                                std::string &value) {
        if (!std::getline(stream, line)) {
            return false;
        }
        const std::string prefix = std::string{name} + "=";
        if (line.rfind(prefix, 0) != 0) {
            return false;
        }
        const auto colon = line.find(':', prefix.size());
        if (colon == std::string::npos) {
            return false;
        }
        std::size_t length = 0;
        try {
            length = static_cast<std::size_t>(std::stoull(
                line.substr(prefix.size(), colon - prefix.size())));
        }
        catch (...) {
            return false;
        }
        value = line.substr(colon + 1);
        return value.size() == length;
    };
    std::string value;
    if (!read_field("observation", value) || value.empty() ||
        !std::getline(stream, line) || line.rfind("member_count=", 0) != 0) {
        return false;
    }
    std::size_t member_count = 0;
    try {
        member_count = static_cast<std::size_t>(
            std::stoull(line.substr(std::string{"member_count="}.size())));
    }
    catch (...) {
        return false;
    }
    if (member_count == 0) {
        return false;
    }
    for (std::size_t member = 0; member < member_count; ++member) {
        if (!std::getline(stream, line) ||
            line != "member=" + std::to_string(member) ||
            !read_field("role", value) || value != "raw_data_item" ||
            !read_field("interface", value) || value.empty() ||
            !read_field("path", value) || value.empty() ||
            !read_field("sha256", value) || value.size() != 64 ||
            !std::all_of(value.begin(), value.end(), [](unsigned char c) {
                return std::isxdigit(c) && !(c >= 'A' && c <= 'F');
            })) {
            return false;
        }
    }
    return !std::getline(stream, line);
}

inline void append_rtcdiag_manifest_field(std::ostringstream &stream,
                                          std::string_view name,
                                          std::string_view value) {
    stream << name << "=" << value.size() << ":" << value << "\n";
}

template <class RawObs>
RtcdiagRawInputManifest make_rtcdiag_raw_input_manifest(
    const RawObs &rawobs) {
    struct Member {
        std::string interface;
        std::string path;
        std::string sha256;
    };
    std::vector<Member> members;
    members.reserve(rawobs.data_items().size());
    for (const auto &item : rawobs.data_items()) {
        Member member{item.interface(), item.filepath(), {}};
        member.sha256 = citlali::utils::sha256_file(member.path);
        members.push_back(std::move(member));
    }
    std::sort(members.begin(), members.end(),
              [](const Member &left, const Member &right) {
                  return std::tie(left.interface, left.path) <
                         std::tie(right.interface, right.path);
              });
    if (members.empty()) {
        throw DataIOError(
            "rtcdiag canonical raw-input manifest has no members");
    }
    std::ostringstream stream;
    stream << rtcdiag_raw_input_manifest_schema << "\n";
    append_rtcdiag_manifest_field(stream, "observation", rawobs.name());
    stream << "member_count=" << members.size() << "\n";
    for (std::size_t index = 0; index < members.size(); ++index) {
        stream << "member=" << index << "\n";
        append_rtcdiag_manifest_field(stream, "role", "raw_data_item");
        append_rtcdiag_manifest_field(stream, "interface",
                                      members[index].interface);
        append_rtcdiag_manifest_field(stream, "path", members[index].path);
        append_rtcdiag_manifest_field(stream, "sha256",
                                      members[index].sha256);
    }
    RtcdiagRawInputManifest manifest;
    manifest.canonical_bytes = stream.str();
    if (manifest.canonical_bytes.size() >
        rtc_sampling_max_estimated_rtcdiag_bytes) {
        throw DataIOError(
            "rtcdiag canonical raw-input manifest exceeds the storage guard");
    }
    manifest.sha256 = citlali::utils::sha256(manifest.canonical_bytes);
    return manifest;
}
inline constexpr std::array<const char *, 91>
    rtcdiag_successor_candidate_variables{{
        "rtc_sampling_alias_amplitude_max_lower",
        "rtc_sampling_alias_amplitude_max_upper",
        "rtc_sampling_alias_error_enclosure",
        "rtc_sampling_alias_evaluations",
        "rtc_sampling_alias_lipschitz_bound",
        "rtc_sampling_alias_reason",
        "rtc_sampling_alias_status",
        "rtc_sampling_amplitude_reason",
        "rtc_sampling_amplitude_status",
        "rtc_sampling_candidate_factor",
        "rtc_sampling_candidate_output_count",
        "rtc_sampling_candidate_phase",
        "rtc_sampling_candidate_reason",
        "rtc_sampling_candidate_status",
        "rtc_sampling_detector_output_boundary_context_count",
        "rtc_sampling_detector_output_boundary_context_fraction",
        "rtc_sampling_detector_output_cell_count",
        "rtc_sampling_detector_output_fully_supported_count",
        "rtc_sampling_detector_output_fully_supported_fraction",
        "rtc_sampling_detector_output_internal_gap_count",
        "rtc_sampling_detector_output_internal_gap_fraction",
        "rtc_sampling_detector_output_invalid_or_overlimit_motion_count",
        "rtc_sampling_detector_output_invalid_or_overlimit_motion_fraction",
        "rtc_sampling_detector_output_low_velocity_motion_count",
        "rtc_sampling_detector_output_low_velocity_motion_fraction",
        "rtc_sampling_detector_output_nonfinite_input_count",
        "rtc_sampling_detector_output_nonfinite_input_fraction",
        "rtc_sampling_detector_output_per_detector_invalid_count",
        "rtc_sampling_detector_output_per_detector_invalid_fraction",
        "rtc_sampling_detector_output_realized_filter_guard_count",
        "rtc_sampling_detector_output_realized_filter_guard_fraction",
        "rtc_sampling_detector_output_science_flag_count",
        "rtc_sampling_detector_output_science_flag_fraction",
        "rtc_sampling_detector_output_unclassified_count",
        "rtc_sampling_detector_output_unclassified_fraction",
        "rtc_sampling_distortion_reason",
        "rtc_sampling_distortion_status",
        "rtc_sampling_eligible_input_support",
        "rtc_sampling_fir_tap_count",
        "rtc_sampling_full_duration_s",
        "rtc_sampling_full_fraction",
        "rtc_sampling_full_output_count",
        "rtc_sampling_incomplete_boundary_count",
        "rtc_sampling_incomplete_gap_count",
        "rtc_sampling_incomplete_other_count",
        "rtc_sampling_left_context_samples",
        "rtc_sampling_longest_full_run",
        "rtc_sampling_numerical_evaluations",
        "rtc_sampling_output_nyquist_hz",
        "rtc_sampling_output_sample_rate_hz",
        "rtc_sampling_phase_reason",
        "rtc_sampling_phase_status",
        "rtc_sampling_plan_transfer_reason",
        "rtc_sampling_plan_transfer_status",
        "rtc_sampling_power_reason",
        "rtc_sampling_power_status",
        "rtc_sampling_relative_amplitude_at_dc",
        "rtc_sampling_relative_amplitude_error_enclosure",
        "rtc_sampling_relative_amplitude_evaluations",
        "rtc_sampling_relative_amplitude_lipschitz_bound",
        "rtc_sampling_relative_amplitude_max_lower",
        "rtc_sampling_relative_amplitude_max_upper",
        "rtc_sampling_relative_distortion_at_dc",
        "rtc_sampling_relative_distortion_error_enclosure",
        "rtc_sampling_relative_distortion_evaluations",
        "rtc_sampling_relative_distortion_lipschitz_bound",
        "rtc_sampling_relative_distortion_max_lower",
        "rtc_sampling_relative_distortion_max_upper",
        "rtc_sampling_relative_phase_abs_max_lower_rad",
        "rtc_sampling_relative_phase_abs_max_upper_rad",
        "rtc_sampling_relative_phase_at_dc_rad",
        "rtc_sampling_relative_phase_error_enclosure_rad",
        "rtc_sampling_relative_phase_evaluations",
        "rtc_sampling_relative_phase_lipschitz_bound",
        "rtc_sampling_relative_power_at_dc",
        "rtc_sampling_relative_power_error_enclosure",
        "rtc_sampling_relative_power_evaluations",
        "rtc_sampling_relative_power_lipschitz_bound",
        "rtc_sampling_relative_power_max_lower",
        "rtc_sampling_relative_power_max_upper",
        "rtc_sampling_right_context_samples",
        "rtc_sampling_samples_per_fwhm",
        "rtc_sampling_stopband_amplitude_max_lower",
        "rtc_sampling_stopband_amplitude_max_upper",
        "rtc_sampling_stopband_error_enclosure",
        "rtc_sampling_stopband_evaluations",
        "rtc_sampling_stopband_lipschitz_bound",
        "rtc_sampling_stopband_reason",
        "rtc_sampling_stopband_rejection_db_lower",
        "rtc_sampling_stopband_rejection_db_upper",
        "rtc_sampling_stopband_status"}};

enum class RtcdiagFinalizeFailureStage {
    none,
    manifest,
    provenance,
    validation,
    sync,
    close,
    reopen,
    publish,
};

inline void inject_rtcdiag_finalize_failure(
    RtcdiagFinalizeFailureStage configured,
    RtcdiagFinalizeFailureStage current) {
    if (configured == current) {
        throw DataIOError("injected rtcdiag finalization failure");
    }
}

inline std::string read_required_rtcdiag_manifest(
    const std::filesystem::path &path) {
    std::error_code size_error;
    const auto manifest_size = std::filesystem::file_size(path, size_error);
    if (size_error) {
        throw DataIOError("required rtcdiag raw-input manifest is unavailable: " +
                          path.string());
    }
    if (manifest_size > rtc_sampling_max_estimated_rtcdiag_bytes) {
        throw DataIOError(
            "required rtcdiag raw-input manifest exceeds the storage guard");
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw DataIOError("required rtcdiag raw-input manifest is unavailable: " +
                          path.string());
    }
    return std::string{std::istreambuf_iterator<char>(stream),
                       std::istreambuf_iterator<char>()};
}

inline std::string read_netcdf_string_scalar(netCDF::NcFile &file,
                                             const char *name) {
    const auto variable = file.getVar(name);
    if (variable.isNull()) {
        throw DataIOError(std::string{"required rtcdiag variable is absent: "} +
                          name);
    }
    char *raw = nullptr;
    variable.getVar(&raw);
    const std::string value = raw == nullptr ? "" : raw;
    if (raw != nullptr) {
        nc_free_string(1, &raw);
    }
    return value;
}

inline bool rtcdiag_dimensions_equal(
    const netCDF::NcVar &variable,
    const std::vector<std::string> &expected) {
    const auto dimensions = variable.getDims();
    if (dimensions.size() != expected.size()) {
        return false;
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (dimensions[index].getName() != expected[index]) {
            return false;
        }
    }
    return true;
}

inline void validate_rtcdiag_successor_staging(netCDF::NcFile &file) {
    if (read_netcdf_string_scalar(file, "RTC_DIAG_SCHEMA_VERSION") !=
            rtc_sampling_schema_version ||
        read_netcdf_string_scalar(file,
            "RTC_SAMPLING_ALGORITHM_VERSION") !=
            rtc_sampling_algorithm_version ||
        read_netcdf_string_scalar(
            file, "RTC_SAMPLING_STATUS_REASON_VOCABULARY") !=
            rtc_sampling_status_reason_vocabulary()) {
        throw DataIOError("rtcdiag successor schema/algorithm identity mismatch");
    }
    const auto identity = rtcdiag_successor_identity_for_mode(
        read_netcdf_string_scalar(file, "RTC_SAMPLING_PRODUCT_MODE"));
    if (read_netcdf_string_scalar(
            file, "RTC_SAMPLING_PRODUCT_CONTRACT_ID") !=
            identity.contract_id ||
        read_netcdf_string_scalar(
            file, "RTC_SAMPLING_VALIDATION_PROFILE_ID") !=
            identity.validation_profile_id ||
        read_netcdf_string_scalar(file, "RTC_SAMPLING_CONTRACT_EPOCH") !=
            rtcdiag_successor_contract_epoch) {
        throw DataIOError(
            "rtcdiag successor mode/contract/profile identity mismatch");
    }
    const auto commit = read_netcdf_string_scalar(
        file, "RTC_SAMPLING_CITLALI_COMMIT");
    if (commit.size() != 40 || !std::all_of(
            commit.begin(), commit.end(), [](unsigned char value) {
                return std::isxdigit(value) &&
                       !(value >= 'A' && value <= 'F');
            })) {
        throw DataIOError("rtcdiag successor Citlali commit is not 40-hex");
    }
    const auto manifest_reference = read_netcdf_string_scalar(
        file, "RTC_SAMPLING_RAW_MANIFEST_REFERENCE");
    const auto manifest_digest = read_netcdf_string_scalar(
        file, "RTC_SAMPLING_RAW_MANIFEST_SHA256");
    const auto manifest_bytes = read_netcdf_string_scalar(
        file, "RTC_SAMPLING_RAW_MANIFEST_CANONICAL_BYTES");
    if (manifest_reference != rtcdiag_raw_input_manifest_reference ||
        !validate_rtcdiag_raw_input_manifest_bytes(manifest_bytes) ||
        manifest_digest != citlali::utils::sha256(manifest_bytes)) {
        throw DataIOError(
            "rtcdiag canonical raw-input manifest identity/digest mismatch");
    }
    const auto fir = file.getVar("rtc_sampling_realized_fir_coefficients");
    const auto fir_dim = file.getDim("n_rtc_sampling_fir_coefficients");
    if (fir.isNull() || fir_dim.isNull() || fir_dim.getSize() == 0 ||
        !rtcdiag_dimensions_equal(
            fir, {"n_rtc_sampling_fir_coefficients"})) {
        throw DataIOError("rtcdiag realized FIR shape is invalid");
    }
    std::vector<double> fir_coefficients(fir_dim.getSize());
    fir.getVar(fir_coefficients.data());
    if (read_netcdf_string_scalar(file, "RTC_SAMPLING_FIR_DIGEST") !=
        rtc_sampling_fir_digest(fir_coefficients)) {
        throw DataIOError("rtcdiag realized FIR digest mismatch");
    }
    for (const char *name : {
             "rtc_sampling_prerequisite_status",
             "rtc_sampling_prerequisite_reason",
             "rtc_sampling_candidate_mmax",
             "rtc_sampling_candidate_range_status",
             "rtc_sampling_candidate_range_reason",
             "rtc_sampling_applied_scan_status",
             "rtc_sampling_applied_scan_reason",
             "rtc_sampling_beam_fwhm_arcsec",
             "rtc_sampling_temporal_sigma_s",
             "rtc_sampling_input_total_detector_cells",
             "rtc_sampling_input_fully_supported_count",
             "rtc_sampling_input_boundary_context_count",
             "rtc_sampling_input_internal_gap_count",
             "rtc_sampling_input_low_velocity_motion_count",
             "rtc_sampling_input_invalid_or_overlimit_motion_count",
             "rtc_sampling_input_per_detector_invalid_count",
             "rtc_sampling_input_science_flag_count",
             "rtc_sampling_input_nonfinite_input_count",
             "rtc_sampling_input_realized_filter_guard_count",
             "rtc_sampling_input_unclassified_count"}) {
        const auto variable = file.getVar(name);
        if (variable.isNull() ||
            !rtcdiag_dimensions_equal(variable, {"n_scans", "n_arrays"})) {
            throw DataIOError(std::string{
                "rtcdiag scan-array compact-state shape mismatch for "} +
                name);
        }
    }
    const auto scan_dim = file.getDim("n_scans");
    const auto array_dim = file.getDim("n_arrays");
    if (scan_dim.isNull() || array_dim.isNull() || scan_dim.getSize() == 0 ||
        array_dim.getSize() == 0 ||
        scan_dim.getSize() > std::numeric_limits<std::size_t>::max() /
                                 array_dim.getSize()) {
        throw DataIOError("rtcdiag scan-array compact-state shape is invalid");
    }
    const std::size_t scan_array_cells =
        scan_dim.getSize() * array_dim.getSize();
    std::vector<int> input_total(scan_array_cells);
    file.getVar("rtc_sampling_input_total_detector_cells")
        .getVar(input_total.data());
    std::vector<long long> input_sum(scan_array_cells, 0);
    std::vector<double> input_fraction_sum(scan_array_cells, 0.0);
    const std::array<const char *, 10> input_count_names{{
        "rtc_sampling_input_fully_supported_count",
        "rtc_sampling_input_boundary_context_count",
        "rtc_sampling_input_internal_gap_count",
        "rtc_sampling_input_low_velocity_motion_count",
        "rtc_sampling_input_invalid_or_overlimit_motion_count",
        "rtc_sampling_input_per_detector_invalid_count",
        "rtc_sampling_input_science_flag_count",
        "rtc_sampling_input_nonfinite_input_count",
        "rtc_sampling_input_realized_filter_guard_count",
        "rtc_sampling_input_unclassified_count"}};
    const std::array<const char *, 10> input_fraction_names{{
        "rtc_sampling_input_fully_supported_fraction",
        "rtc_sampling_input_boundary_context_fraction",
        "rtc_sampling_input_internal_gap_fraction",
        "rtc_sampling_input_low_velocity_motion_fraction",
        "rtc_sampling_input_invalid_or_overlimit_motion_fraction",
        "rtc_sampling_input_per_detector_invalid_fraction",
        "rtc_sampling_input_science_flag_fraction",
        "rtc_sampling_input_nonfinite_input_fraction",
        "rtc_sampling_input_realized_filter_guard_fraction",
        "rtc_sampling_input_unclassified_fraction"}};
    for (std::size_t category = 0; category < input_count_names.size();
         ++category) {
        std::vector<int> counts(scan_array_cells);
        std::vector<double> fractions(scan_array_cells);
        file.getVar(input_count_names[category]).getVar(counts.data());
        const auto fraction_variable =
            file.getVar(input_fraction_names[category]);
        if (fraction_variable.isNull() ||
            !rtcdiag_dimensions_equal(
                fraction_variable, {"n_scans", "n_arrays"})) {
            throw DataIOError(
                "rtcdiag input context fraction shape mismatch");
        }
        fraction_variable.getVar(fractions.data());
        for (std::size_t cell = 0; cell < scan_array_cells; ++cell) {
            const double expected = input_total[cell] == 0
                ? 0.0
                : static_cast<double>(counts[cell]) /
                      static_cast<double>(input_total[cell]);
            if (counts[cell] < 0 || !std::isfinite(fractions[cell]) ||
                std::abs(fractions[cell] - expected) > 1e-12) {
                throw DataIOError(
                    "rtcdiag input context count/fraction mismatch");
            }
            input_sum[cell] += counts[cell];
            input_fraction_sum[cell] += fractions[cell];
        }
    }
    for (std::size_t cell = 0; cell < scan_array_cells; ++cell) {
        if (input_total[cell] < 0 || input_sum[cell] != input_total[cell] ||
            (input_total[cell] == 0
                 ? input_fraction_sum[cell] != 0.0
                 : std::abs(input_fraction_sum[cell] - 1.0) > 1e-12)) {
            throw DataIOError(
                "rtcdiag input context categories do not sum to total");
        }
    }
    int available = 0;
    long long declared_count = -1;
    file.getVar("RTC_SAMPLING_CANDIDATE_TABLE_AVAILABLE").getVar(&available);
    file.getVar("RTC_SAMPLING_CANDIDATE_COUNT").getVar(&declared_count);
    const auto candidate_dim = file.getDim("n_rtc_sampling_candidates");
    if (available != 0 && available != 1) {
        throw DataIOError(
            "rtcdiag candidate-table availability is not normalized to 0/1");
    }
    if (available == 1) {
        if (candidate_dim.isNull() || candidate_dim.getSize() == 0 ||
            declared_count != static_cast<long long>(candidate_dim.getSize())) {
            throw DataIOError(
                "available rtcdiag candidate count/dimension mismatch");
        }
        for (const char *name : rtcdiag_successor_candidate_variables) {
            const auto variable = file.getVar(name);
            if (variable.isNull()) {
                throw DataIOError(std::string{
                    "required rtcdiag candidate variable is absent: "} + name);
            }
            const bool axis_only = std::string_view{name} ==
                    "rtc_sampling_candidate_factor" ||
                std::string_view{name} == "rtc_sampling_candidate_phase";
            const std::vector<std::string> expected = axis_only
                ? std::vector<std::string>{"n_rtc_sampling_candidates"}
                : std::vector<std::string>{"n_scans", "n_arrays",
                                           "n_rtc_sampling_candidates"};
            if (!rtcdiag_dimensions_equal(variable, expected)) {
                throw DataIOError(std::string{
                    "rtcdiag candidate full-shape mismatch for "} + name);
            }
        }
        if (scan_dim.isNull() || array_dim.isNull() ||
            scan_dim.getSize() == 0 || array_dim.getSize() == 0 ||
            candidate_dim.getSize() >
                std::numeric_limits<std::size_t>::max() /
                    scan_dim.getSize() ||
            scan_dim.getSize() * candidate_dim.getSize() >
                std::numeric_limits<std::size_t>::max() /
                    array_dim.getSize()) {
            throw DataIOError("rtcdiag candidate context shape is invalid");
        }
        const std::size_t cells = scan_dim.getSize() * array_dim.getSize() *
                                  candidate_dim.getSize();
        std::vector<int> total(cells);
        file.getVar("rtc_sampling_detector_output_cell_count").getVar(
            total.data());
        std::vector<long long> sum(cells, 0);
        std::vector<double> fraction_sum(cells, 0.0);
        const std::array<const char *, 10> count_names{{
            "rtc_sampling_detector_output_fully_supported_count",
            "rtc_sampling_detector_output_boundary_context_count",
            "rtc_sampling_detector_output_internal_gap_count",
            "rtc_sampling_detector_output_low_velocity_motion_count",
            "rtc_sampling_detector_output_invalid_or_overlimit_motion_count",
            "rtc_sampling_detector_output_per_detector_invalid_count",
            "rtc_sampling_detector_output_science_flag_count",
            "rtc_sampling_detector_output_nonfinite_input_count",
            "rtc_sampling_detector_output_realized_filter_guard_count",
            "rtc_sampling_detector_output_unclassified_count"}};
        const std::array<const char *, 10> fraction_names{{
            "rtc_sampling_detector_output_fully_supported_fraction",
            "rtc_sampling_detector_output_boundary_context_fraction",
            "rtc_sampling_detector_output_internal_gap_fraction",
            "rtc_sampling_detector_output_low_velocity_motion_fraction",
            "rtc_sampling_detector_output_invalid_or_overlimit_motion_fraction",
            "rtc_sampling_detector_output_per_detector_invalid_fraction",
            "rtc_sampling_detector_output_science_flag_fraction",
            "rtc_sampling_detector_output_nonfinite_input_fraction",
            "rtc_sampling_detector_output_realized_filter_guard_fraction",
            "rtc_sampling_detector_output_unclassified_fraction"}};
        for (std::size_t category = 0; category < count_names.size();
             ++category) {
            const char *name = count_names[category];
            std::vector<int> values(cells);
            const auto variable = file.getVar(name);
            if (variable.isNull()) {
                throw DataIOError(std::string{
                    "rtcdiag candidate context category is absent: "} + name);
            }
            variable.getVar(values.data());
            for (std::size_t i = 0; i < cells; ++i) {
                if (values[i] < 0) {
                    throw DataIOError(
                        "rtcdiag candidate context category is negative");
                }
                sum[i] += values[i];
                if (std::string_view{name}.find("unclassified") !=
                        std::string_view::npos &&
                    values[i] != 0) {
                    throw DataIOError(
                        "rtcdiag candidate context contains unclassified cells");
                }
            }
            std::vector<double> fractions(cells);
            const auto fraction_variable = file.getVar(fraction_names[category]);
            if (fraction_variable.isNull()) {
                throw DataIOError(
                    "rtcdiag candidate context fraction is absent");
            }
            fraction_variable.getVar(fractions.data());
            for (std::size_t i = 0; i < cells; ++i) {
                const double expected = total[i] == 0
                    ? 0.0
                    : static_cast<double>(values[i]) /
                          static_cast<double>(total[i]);
                if (!std::isfinite(fractions[i]) ||
                    std::abs(fractions[i] - expected) > 1e-12) {
                    throw DataIOError(
                        "rtcdiag candidate context count/fraction mismatch");
                }
                fraction_sum[i] += fractions[i];
            }
        }
        for (std::size_t i = 0; i < cells; ++i) {
            if (total[i] < 0 || sum[i] != total[i] ||
                (total[i] == 0 ? fraction_sum[i] != 0.0
                               : std::abs(fraction_sum[i] - 1.0) > 1e-12)) {
                throw DataIOError(
                    "rtcdiag candidate context categories do not sum to total");
            }
        }
    }
    else {
        if (declared_count != 0 || !candidate_dim.isNull()) {
            throw DataIOError(
                "unavailable rtcdiag candidate table has forbidden rows/dimension");
        }
        for (const char *name : rtcdiag_successor_candidate_variables) {
            if (!file.getVar(name).isNull()) {
                throw DataIOError(std::string{
                    "unavailable rtcdiag candidate table has forbidden variable: "} +
                    name);
            }
        }
    }
}

inline std::string finalize_rtcdiag_successor_staging(
    const std::string &staging_path,
    const RtcdiagRawInputManifest &raw_manifest,
    const RtcdiagSuccessorProductIdentity &identity,
    RtcdiagFinalizeFailureStage failure =
        RtcdiagFinalizeFailureStage::none) {
    if (!is_netcdf_atomic_staging_path(staging_path)) {
        throw DataIOError(
            "rtcdiag finalization refuses an already-published path: " +
            staging_path);
    }
    if (raw_manifest.canonical_bytes.empty() ||
        raw_manifest.sha256 !=
            citlali::utils::sha256(raw_manifest.canonical_bytes)) {
        throw DataIOError(
            "required rtcdiag canonical raw-input manifest is invalid");
    }
    try {
        inject_rtcdiag_finalize_failure(
            failure, RtcdiagFinalizeFailureStage::manifest);
        std::error_code staging_size_error;
        const auto staging_size =
            std::filesystem::file_size(staging_path, staging_size_error);
        const auto manifest_size = raw_manifest.canonical_bytes.size();
        std::uintmax_t bounded_size = 0;
        if (staging_size_error ||
            manifest_size > rtc_sampling_max_estimated_rtcdiag_bytes ||
            staging_size > rtc_sampling_max_estimated_rtcdiag_bytes ||
            manifest_size >
                std::numeric_limits<std::uintmax_t>::max() - staging_size) {
            throw DataIOError(
                "rtcdiag staging plus raw-input manifest cannot satisfy the storage guard");
        }
        bounded_size = staging_size + manifest_size;
        if (bounded_size > rtc_sampling_max_estimated_rtcdiag_bytes) {
            throw DataIOError(
                "rtcdiag staging plus raw-input manifest exceeds the storage guard");
        }
        netCDF::NcFile file(staging_path, netCDF::NcFile::write);
        inject_rtcdiag_finalize_failure(
            failure, RtcdiagFinalizeFailureStage::provenance);
        add_netcdf_var(file, "RTC_SAMPLING_PRODUCT_CONTRACT_ID",
                       identity.contract_id);
        add_netcdf_var(file, "RTC_SAMPLING_VALIDATION_PROFILE_ID",
                       identity.validation_profile_id);
        add_netcdf_var(file, "RTC_SAMPLING_PRODUCT_MODE", identity.mode);
        add_netcdf_var(file, "RTC_SAMPLING_CONTRACT_EPOCH",
                       std::string{rtcdiag_successor_contract_epoch});
        add_netcdf_var(file, "RTC_SAMPLING_RAW_MANIFEST_REFERENCE",
                       std::string{rtcdiag_raw_input_manifest_reference});
        add_netcdf_var(file, "RTC_SAMPLING_RAW_MANIFEST_SHA256",
                       raw_manifest.sha256);
        add_netcdf_var(file, "RTC_SAMPLING_RAW_MANIFEST_CANONICAL_BYTES",
                       raw_manifest.canonical_bytes);
        inject_rtcdiag_finalize_failure(
            failure, RtcdiagFinalizeFailureStage::validation);
        validate_rtcdiag_successor_staging(file);
        inject_rtcdiag_finalize_failure(
            failure, RtcdiagFinalizeFailureStage::sync);
        file.sync();
        inject_rtcdiag_finalize_failure(
            failure, RtcdiagFinalizeFailureStage::close);
        file.close();
        inject_rtcdiag_finalize_failure(
            failure, RtcdiagFinalizeFailureStage::reopen);
        {
            netCDF::NcFile reopened(staging_path, netCDF::NcFile::read);
            validate_rtcdiag_successor_staging(reopened);
            reopened.close();
        }
        std::error_code ec;
        const auto size = std::filesystem::file_size(staging_path, ec);
        if (ec || size > rtc_sampling_max_estimated_rtcdiag_bytes) {
            throw DataIOError(
                "completed rtcdiag staging artifact exceeds the storage guard");
        }
        inject_rtcdiag_finalize_failure(
            failure, RtcdiagFinalizeFailureStage::publish);
        return publish_netcdf_atomic_staging(staging_path);
    }
    catch (...) {
        cleanup_netcdf_atomic_staging(staging_path);
        throw;
    }
}

}  // namespace citlali::pipeline
