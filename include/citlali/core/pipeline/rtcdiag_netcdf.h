#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cmath>
#include <limits>
#include <numeric>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <netcdf>

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

inline constexpr const char *rtcdiag_successor_product_contract_id =
    "sci-rtc-001-stage-a-successor-products-v1";
inline constexpr const char *rtcdiag_successor_validation_profile_id =
    "sci-rtc-001-stage-a-successor-v1";
inline constexpr const char *rtcdiag_successor_contract_epoch =
    "sci-rtc-001-stage-a-successor-2026-08-11";
inline constexpr std::array<const char *, 81>
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
        "rtc_sampling_detector_output_cell_count",
        "rtc_sampling_detector_output_fully_supported_count",
        "rtc_sampling_detector_output_internal_gap_count",
        "rtc_sampling_detector_output_invalid_or_overlimit_motion_count",
        "rtc_sampling_detector_output_low_velocity_motion_count",
        "rtc_sampling_detector_output_nonfinite_input_count",
        "rtc_sampling_detector_output_per_detector_invalid_count",
        "rtc_sampling_detector_output_realized_filter_guard_count",
        "rtc_sampling_detector_output_science_flag_count",
        "rtc_sampling_detector_output_unclassified_count",
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

inline void validate_rtcdiag_successor_staging(netCDF::NcFile &file) {
    if (read_netcdf_string_scalar(file, "RTC_DIAG_SCHEMA_VERSION") !=
            rtc_sampling_schema_version ||
        read_netcdf_string_scalar(file,
            "RTC_SAMPLING_ALGORITHM_VERSION") !=
            rtc_sampling_algorithm_version) {
        throw DataIOError("rtcdiag successor schema/algorithm identity mismatch");
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
            const auto dimensions = variable.getDims();
            if (dimensions.empty() ||
                dimensions.back().getName() != candidate_dim.getName() ||
                dimensions.back().getSize() != candidate_dim.getSize()) {
                throw DataIOError(std::string{
                    "rtcdiag candidate cardinality mismatch for "} + name);
            }
        }
        const auto scan_dim = file.getDim("n_scans");
        const auto array_dim = file.getDim("n_arrays");
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
        for (const char *name : {
                 "rtc_sampling_detector_output_fully_supported_count",
                 "rtc_sampling_detector_output_boundary_context_count",
                 "rtc_sampling_detector_output_internal_gap_count",
                 "rtc_sampling_detector_output_low_velocity_motion_count",
                 "rtc_sampling_detector_output_invalid_or_overlimit_motion_count",
                 "rtc_sampling_detector_output_per_detector_invalid_count",
                 "rtc_sampling_detector_output_science_flag_count",
                 "rtc_sampling_detector_output_nonfinite_input_count",
                 "rtc_sampling_detector_output_realized_filter_guard_count",
                 "rtc_sampling_detector_output_unclassified_count"}) {
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
        }
        for (std::size_t i = 0; i < cells; ++i) {
            if (total[i] < 0 || sum[i] != total[i]) {
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
    const std::optional<std::filesystem::path> &raw_manifest_path,
    RtcdiagFinalizeFailureStage failure =
        RtcdiagFinalizeFailureStage::none) {
    if (!is_netcdf_atomic_staging_path(staging_path)) {
        throw DataIOError(
            "rtcdiag finalization refuses an already-published path: " +
            staging_path);
    }
    if (!raw_manifest_path) {
        throw DataIOError("required rtcdiag raw-input manifest was not published");
    }
    try {
        inject_rtcdiag_finalize_failure(
            failure, RtcdiagFinalizeFailureStage::manifest);
        const std::string manifest =
            read_required_rtcdiag_manifest(*raw_manifest_path);
        if (manifest.empty()) {
            throw DataIOError(
                "required rtcdiag raw-input manifest is empty");
        }
        const std::string digest = citlali::utils::sha256(manifest);
        netCDF::NcFile file(staging_path, netCDF::NcFile::write);
        inject_rtcdiag_finalize_failure(
            failure, RtcdiagFinalizeFailureStage::provenance);
        add_netcdf_var(file, "RTC_SAMPLING_PRODUCT_CONTRACT_ID",
                       std::string{rtcdiag_successor_product_contract_id});
        add_netcdf_var(file, "RTC_SAMPLING_VALIDATION_PROFILE_ID",
                       std::string{rtcdiag_successor_validation_profile_id});
        add_netcdf_var(file, "RTC_SAMPLING_CONTRACT_EPOCH",
                       std::string{rtcdiag_successor_contract_epoch});
        add_netcdf_var(file, "RTC_SAMPLING_RAW_MANIFEST_REFERENCE",
                       raw_manifest_path->filename().string());
        add_netcdf_var(file, "RTC_SAMPLING_RAW_MANIFEST_SHA256", digest);
        add_netcdf_var(file, "RTC_SAMPLING_RAW_MANIFEST_CANONICAL_BYTES",
                       manifest);
        inject_rtcdiag_finalize_failure(
            failure, RtcdiagFinalizeFailureStage::validation);
        validate_rtcdiag_successor_staging(file);
        inject_rtcdiag_finalize_failure(
            failure, RtcdiagFinalizeFailureStage::sync);
        file.sync();
        inject_rtcdiag_finalize_failure(
            failure, RtcdiagFinalizeFailureStage::close);
        file.close();
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
