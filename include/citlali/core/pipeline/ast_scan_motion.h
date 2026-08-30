#pragma once

#include <citlali/core/pipeline/timestream_native_timing.h>

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace citlali::pipeline {

inline constexpr std::string_view ast_scan_motion_policy_id =
    "wp7-ast-scan-motion-v1";
inline constexpr std::string_view ast_scan_motion_product_role =
    "SCI-AST:scan_motion_planning@1";
inline constexpr std::string_view ast_scan_motion_time_field =
    "Data.TelescopeBackend.TelTime";
inline constexpr std::string_view ast_scan_motion_ra_field =
    "Data.TelescopeBackend.SourceRaAct";
inline constexpr std::string_view ast_scan_motion_dec_field =
    "Data.TelescopeBackend.SourceDecAct";

using AstTelescopeRecord = std::int64_t;

enum class AstScanMotionProducerKind : std::uint8_t {
    real_toltec,
    simulation,
    unsupported,
};

enum class AstScanMotionFieldRegistry : std::uint8_t {
    source_ra_act_source_dec_act_j2000_radians,
    unsupported,
};

struct AstScanMotionSourceMetadata {
    AstScanMotionProducerKind producer_kind =
        AstScanMotionProducerKind::unsupported;
    std::string dcs_observation_goal;
    std::string dcs_observation_program;
    std::int64_t scan_file_valid = 0;
    double source_epoch = 0.0;
    std::int64_t source_coordinate_system = -1;
    double nominal_producer_cadence_hz = 0.0;
    AstScanMotionFieldRegistry field_registry =
        AstScanMotionFieldRegistry::unsupported;
    std::string source_artifact_identity;
};

struct AstTelescopeRecordIdentity {
    NativeObservationScope scope;
    AstTelescopeRecord record = -1;

    friend bool operator==(const AstTelescopeRecordIdentity &,
                           const AstTelescopeRecordIdentity &) = default;
};

// Immutable producer facts. This is the single owner of the raw telescope
// Unix-second TelTime axis and J2000-radian SourceRaAct/SourceDecAct planes
// used by the bounded AST role. The derived product below retains this handle
// and does not copy those planes.
class AstScanMotionSource {
public:
    static std::shared_ptr<const AstScanMotionSource> admit(
        NativeObservationScope telescope_scope,
        NativeObservationScope admitted_detector_scope,
        AstTelescopeRecord first_record,
        AstScanMotionSourceMetadata metadata,
        Eigen::VectorXd producer_times_unix_sec,
        Eigen::VectorXd source_ra_act_rad,
        Eigen::VectorXd source_dec_act_rad);

    const NativeObservationScope &telescope_scope() const noexcept;
    const NativeObservationScope &admitted_detector_scope() const noexcept;
    AstTelescopeRecord first_record() const noexcept;
    AstTelescopeRecord past_last_record() const noexcept;
    std::size_t record_count() const noexcept;
    const AstScanMotionSourceMetadata &metadata() const noexcept;
    const Eigen::VectorXd &producer_times_unix_sec() const noexcept;
    const Eigen::VectorXd &source_ra_act_rad() const noexcept;
    const Eigen::VectorXd &source_dec_act_rad() const noexcept;
    AstTelescopeRecordIdentity identity(AstTelescopeRecord record) const;
    std::size_t local_index(AstTelescopeRecord record) const;

private:
    AstScanMotionSource(
        NativeObservationScope telescope_scope,
        NativeObservationScope admitted_detector_scope,
        AstTelescopeRecord first_record,
        AstScanMotionSourceMetadata metadata,
        Eigen::VectorXd producer_times_unix_sec,
        Eigen::VectorXd source_ra_act_rad,
        Eigen::VectorXd source_dec_act_rad);

    NativeObservationScope telescope_scope_;
    NativeObservationScope admitted_detector_scope_;
    AstTelescopeRecord first_record_ = -1;
    AstScanMotionSourceMetadata metadata_;
    Eigen::VectorXd producer_times_unix_sec_;
    Eigen::VectorXd source_ra_act_rad_;
    Eigen::VectorXd source_dec_act_rad_;
};

enum class AstScanMotionCause : std::uint32_t {
    none = 0,
    not_science_observation = 1U << 0,
    invalid_scan_file = 1U << 1,
    unsupported_scan_program = 1U << 2,
    unsupported_source_frame = 1U << 3,
    observation_scope_mismatch = 1U << 4,
    unsupported_producer_kind = 1U << 5,
    unsupported_producer_cadence = 1U << 6,
    unsupported_field_registry = 1U << 7,
    nonfinite_telescope_time = 1U << 8,
    nonmonotonic_telescope_time = 1U << 9,
    telescope_gap = 1U << 10,
    nonfinite_or_unnormalizable_direction = 1U << 11,
    spherical_topology_unavailable = 1U << 12,
    telemetry_defect = 1U << 13,
    telemetry_quality_support_unavailable = 1U << 14,
    derivative_support_intersects_invalidity = 1U << 15,
    rank_deficient_derivative_fit = 1U << 16,
    nonfinite_derivative = 1U << 17,
    network_mapping_support_unavailable = 1U << 18,
    scan_maximum_incomplete = 1U << 19,
    no_admitted_scan_motion = 1U << 20,
};

constexpr AstScanMotionCause operator|(AstScanMotionCause lhs,
                                       AstScanMotionCause rhs) noexcept {
    return static_cast<AstScanMotionCause>(
        static_cast<std::uint32_t>(lhs) |
        static_cast<std::uint32_t>(rhs));
}

constexpr AstScanMotionCause &operator|=(AstScanMotionCause &lhs,
                                         AstScanMotionCause rhs) noexcept {
    lhs = lhs | rhs;
    return lhs;
}

constexpr bool has_cause(AstScanMotionCause value,
                         AstScanMotionCause cause) noexcept {
    return (static_cast<std::uint32_t>(value) &
            static_cast<std::uint32_t>(cause)) != 0;
}

constexpr bool ast_scan_motion_continuous_interval(double dt_sec) noexcept {
    return dt_sec > 0.0 && dt_sec <= 0.030;
}

constexpr bool ast_scan_motion_telemetry_defect(
    double radial_residual_arcsec) noexcept {
    return radial_residual_arcsec > 2.0;
}

constexpr bool ast_scan_motion_speed_admitted(
    double scalar_speed_arcsec_per_sec) noexcept {
    return scalar_speed_arcsec_per_sec >= 1.0;
}

struct AstScanMotionIdentityBinding {
    std::uint64_t requested = 0;
    std::uint64_t effective = 0;
    std::uint64_t observation_resolved = 0;
    std::uint64_t realized = 0;

    bool complete() const noexcept;

    friend bool operator==(const AstScanMotionIdentityBinding &,
                           const AstScanMotionIdentityBinding &) = default;
};

struct AstScanMotionProcessingSpan {
    AstTelescopeRecord first_record = -1;
    AstTelescopeRecord past_last_record = -1;

    friend bool operator==(const AstScanMotionProcessingSpan &,
                           const AstScanMotionProcessingSpan &) = default;
};

struct AstScanMotionSupport {
    AstTelescopeRecord first_record = -1;
    AstTelescopeRecord past_last_record = -1;
    // Inclusive endpoint times for the half-open record interval above.
    double first_time_unix_sec = 0.0;
    double last_time_unix_sec = 0.0;

    friend bool operator==(const AstScanMotionSupport &,
                           const AstScanMotionSupport &) = default;
};

class AstScanMotionProduct;

class AstScanMotionDerivedRecord {
public:
    AstScanMotionCause causes() const noexcept;
    std::int32_t continuity_run() const noexcept;
    bool raw_direction_structurally_valid() const noexcept;
    bool telemetry_quality_classified() const noexcept;
    bool telemetry_defect() const noexcept;
    bool realized_direction_valid() const noexcept;
    bool derivative_valid() const noexcept;
    double telemetry_residual_arcsec() const noexcept;
    double east_velocity_arcsec_per_sec() const noexcept;
    double north_velocity_arcsec_per_sec() const noexcept;
    double scalar_speed_arcsec_per_sec() const noexcept;

private:
    friend class AstScanMotionProduct;
    friend std::shared_ptr<const AstScanMotionProduct>
    build_ast_scan_motion_product(
        std::shared_ptr<const AstScanMotionSource>,
        AstScanMotionIdentityBinding,
        std::span<const AstScanMotionProcessingSpan>);

    static constexpr std::uint8_t raw_direction_valid_bit = 1U << 0;
    static constexpr std::uint8_t quality_classified_bit = 1U << 1;
    static constexpr std::uint8_t telemetry_defect_bit = 1U << 2;
    static constexpr std::uint8_t derivative_valid_bit = 1U << 3;

    AstScanMotionCause causes_ = AstScanMotionCause::none;
    std::int32_t continuity_run_ = -1;
    std::uint8_t flags_ = 0;
    double telemetry_residual_arcsec_ = 0.0;
    double east_velocity_arcsec_per_sec_ = 0.0;
    double north_velocity_arcsec_per_sec_ = 0.0;
    double scalar_speed_arcsec_per_sec_ = 0.0;
};

struct AstScanMotionScanSummary {
    bool maximum_available = false;
    AstScanMotionCause causes = AstScanMotionCause::none;
    AstTelescopeRecord maximizing_record = -1;
    double maximum_speed_arcsec_per_sec = 0.0;
    std::size_t record_count = 0;
    std::size_t continuity_run_count = 0;
    std::size_t derivative_valid_record_count = 0;
    std::size_t admitted_candidate_count = 0;
    std::size_t telemetry_defect_count = 0;
};

struct AstScanMotionMemoryEvidence {
    std::size_t derived_record_bytes = 0;
    std::size_t referenced_source_axis_count = 0;
    std::size_t referenced_source_direction_plane_count = 0;

    std::size_t logical_owned_bytes() const noexcept {
        return derived_record_bytes;
    }
};

class AstScanMotionProduct {
public:
    const std::shared_ptr<const AstScanMotionSource> &source_handle()
        const noexcept;
    const NativeObservationScope &scope() const noexcept;
    const AstScanMotionIdentityBinding &identity_binding() const noexcept;
    std::size_t record_count() const noexcept;
    const AstScanMotionDerivedRecord &record(
        AstTelescopeRecord record) const;
    const AstScanMotionDerivedRecord &record_at_local(
        std::size_t local_index) const;
    AstTelescopeRecord record_identity(std::size_t local_index) const;
    std::optional<AstScanMotionSupport> telemetry_support(
        AstTelescopeRecord record) const;
    std::optional<AstScanMotionSupport> derivative_support(
        AstTelescopeRecord record) const;
    const AstScanMotionScanSummary &scan_summary() const noexcept;
    AstScanMotionMemoryEvidence memory_evidence() const noexcept;
    bool source_time_axis_mapping_eligible() const noexcept;

private:
    friend std::shared_ptr<const AstScanMotionProduct>
    build_ast_scan_motion_product(
        std::shared_ptr<const AstScanMotionSource>,
        AstScanMotionIdentityBinding,
        std::span<const AstScanMotionProcessingSpan>);

    AstScanMotionProduct(
        std::shared_ptr<const AstScanMotionSource> source,
        AstScanMotionIdentityBinding identity_binding,
        std::vector<AstScanMotionDerivedRecord> records,
        AstScanMotionScanSummary scan_summary,
        bool source_time_axis_mapping_eligible);

    std::optional<AstScanMotionSupport> support(
        AstTelescopeRecord record, bool require_derivative) const;

    std::shared_ptr<const AstScanMotionSource> source_;
    AstScanMotionIdentityBinding identity_binding_;
    std::vector<AstScanMotionDerivedRecord> records_;
    AstScanMotionScanSummary scan_summary_;
    bool source_time_axis_mapping_eligible_ = false;
};

std::shared_ptr<const AstScanMotionProduct> build_ast_scan_motion_product(
    std::shared_ptr<const AstScanMotionSource> source,
    AstScanMotionIdentityBinding identity_binding,
    std::span<const AstScanMotionProcessingSpan> engineering_schedule = {});

}  // namespace citlali::pipeline
