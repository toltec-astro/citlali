#pragma once

#include <citlali/core/pipeline/ast_scan_motion_alignment.h>
#include <citlali/core/pipeline/timestream_identity_rtc_only_route.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace citlali::pipeline {

enum class IdentityRouteOccurrenceTimePolicy : std::uint8_t {
    integration_interval_midpoint,
};

// ALIGN assigns one Paired-D1 occurrence to the owner-selected midpoint of
// its integration interval. The native identity remains authoritative only
// when its exact time is that midpoint; no common grid is introduced.
struct IdentityRouteOccurrenceAssignment {
    NativeSampleIdentity network_occurrence;
    std::int64_t parent_readout_occurrence_key = -1;
    std::int64_t paired_xr_occurrence_key = -1;
    NativeReadoutIntegrationSupport integration_support;
    double assigned_time_unix_sec = 0.0;

    friend bool operator==(const IdentityRouteOccurrenceAssignment &,
                           const IdentityRouteOccurrenceAssignment &) =
        default;
};

struct IdentityRouteAlignMemoryEvidence {
    std::size_t owned_numeric_bytes = 0;
    std::size_t owned_occurrence_axis_bytes = 0;
    std::size_t referenced_paired_product_count = 0;
    std::size_t referenced_ast_view_set_count = 0;

    std::size_t logical_owned_bytes() const noexcept {
        return owned_numeric_bytes + owned_occurrence_axis_bytes;
    }
};

// Concrete ALIGN-owned binding around canonical Paired-D1 and the accepted
// AST per-network views. It stores only exact immutable handles and derives
// occurrence assignments on demand.
class IdentityRouteAlignContext {
public:
    static std::shared_ptr<const IdentityRouteAlignContext> admit(
        std::shared_ptr<const NativePairedReadoutObservation> paired,
        std::shared_ptr<const AstScanMotionNetworkViews> ast_views);

    const NativeObservationScope &scope() const noexcept;
    const std::shared_ptr<const NativePairedReadoutObservation> &
    paired_handle() const noexcept;
    const std::shared_ptr<const AstScanMotionNetworkViews> &
    ast_views_handle() const noexcept;
    std::span<const TimestreamNetworkId> participant_network_ids()
        const noexcept;
    IdentityRouteOccurrenceTimePolicy occurrence_time_policy()
        const noexcept;
    IdentityRouteOccurrenceAssignment occurrence_assignment(
        TimestreamNetworkId network_id,
        TimestreamNativeRow native_row) const;
    const AstScanMotionMappedRecord &ast_motion_record(
        TimestreamNetworkId network_id,
        TimestreamNativeRow native_row) const;
    std::optional<AstScanMotionMappedSupport> ast_motion_support(
        TimestreamNetworkId network_id,
        TimestreamNativeRow native_row) const;
    IdentityRouteAlignMemoryEvidence memory_evidence() const noexcept;

private:
    IdentityRouteAlignContext(
        std::shared_ptr<const NativePairedReadoutObservation> paired,
        std::shared_ptr<const AstScanMotionNetworkViews> ast_views);

    std::shared_ptr<const NativePairedReadoutObservation> paired_;
    std::shared_ptr<const AstScanMotionNetworkViews> ast_views_;
};

enum class IdentityRtcAstDependency : std::uint8_t {
    not_applicable,
};

struct IdentityRtcInputContextMemoryEvidence {
    std::size_t owned_numeric_bytes = 0;
    std::size_t owned_span_bytes = 0;
    std::size_t referenced_align_context_count = 0;
    std::size_t referenced_paired_view_count = 0;

    std::size_t logical_owned_bytes() const noexcept {
        return owned_numeric_bytes + owned_span_bytes;
    }
};

// AST is structurally present in every RTC input context. The identity
// operator alone declares that consuming AST motion is not applicable.
class IdentityRtcInputContext {
public:
    static std::shared_ptr<const IdentityRtcInputContext> admit(
        std::shared_ptr<const IdentityRouteAlignContext> align_context,
        std::vector<NativeOccurrenceSpan> logical_spans);

    const std::shared_ptr<const IdentityRouteAlignContext> &
    align_context_handle() const noexcept;
    const std::shared_ptr<const NativePairedReadoutView> &
    signal_handle() const noexcept;
    const std::shared_ptr<const AstScanMotionNetworkViews> &
    ast_views_handle() const noexcept;
    IdentityRtcAstDependency ast_dependency() const noexcept;
    IdentityRtcInputContextMemoryEvidence memory_evidence() const noexcept;

private:
    IdentityRtcInputContext(
        std::shared_ptr<const IdentityRouteAlignContext> align_context,
        std::shared_ptr<const NativePairedReadoutView> signal);

    std::shared_ptr<const IdentityRouteAlignContext> align_context_;
    std::shared_ptr<const NativePairedReadoutView> signal_;
};

enum class IdentityRtcSamplingRelation : std::uint8_t {
    native_factor_one_phase_zero,
};

enum class IdentityRouteSignalUnitState : std::uint8_t {
    raw_paired_xr,
};

struct IdentityRtcOutputContextMemoryEvidence {
    std::size_t owned_numeric_bytes = 0;
    std::size_t owned_coordinate_plane_bytes = 0;
    std::size_t referenced_input_context_count = 0;
    std::size_t referenced_rtc_terminal_count = 0;

    std::size_t logical_owned_bytes() const noexcept {
        return owned_numeric_bytes + owned_coordinate_plane_bytes;
    }
};

// AST owns the output-domain coordinate/motion association through the exact
// factor-one native occurrence relation. The RTC result remains raw x/r and
// owns no coordinate or signal plane copy.
class IdentityRtcOutputContext {
public:
    static std::shared_ptr<const IdentityRtcOutputContext> admit(
        std::shared_ptr<const IdentityRtcInputContext> input_context,
        std::shared_ptr<const RtcOnlyTerminalProduct> rtc_terminal);

    const std::shared_ptr<const IdentityRtcInputContext> &
    input_context_handle() const noexcept;
    const std::shared_ptr<const RtcOnlyTerminalProduct> &
    rtc_terminal_handle() const noexcept;
    const std::shared_ptr<const RtcTimestream> &signal_handle()
        const noexcept;
    const std::shared_ptr<const AstScanMotionNetworkViews> &
    ast_views_handle() const noexcept;
    IdentityRtcSamplingRelation sampling_relation() const noexcept;
    IdentityRouteSignalUnitState signal_unit_state() const noexcept;
    IdentityRouteOccurrenceAssignment occurrence_assignment(
        TimestreamNetworkId network_id,
        TimestreamNativeRow native_row) const;
    const AstScanMotionMappedRecord &ast_motion_record(
        TimestreamNetworkId network_id,
        TimestreamNativeRow native_row) const;
    std::optional<AstScanMotionMappedSupport> ast_motion_support(
        TimestreamNetworkId network_id,
        TimestreamNativeRow native_row) const;
    IdentityRtcOutputContextMemoryEvidence memory_evidence() const noexcept;

private:
    IdentityRtcOutputContext(
        std::shared_ptr<const IdentityRtcInputContext> input_context,
        std::shared_ptr<const RtcOnlyTerminalProduct> rtc_terminal);

    std::shared_ptr<const IdentityRtcInputContext> input_context_;
    std::shared_ptr<const RtcOnlyTerminalProduct> rtc_terminal_;
};

enum class IdentityCalibrationProductState : std::uint8_t {
    unavailable_component_not_admitted,
};

enum class IdentityCalibrationUnitState : std::uint8_t {
    unavailable_no_calibration_product,
};

enum class IdentityCalibrationResponseState : std::uint8_t {
    unavailable_no_calibration_product,
};

enum class IdentityCalibrationUncertaintyState : std::uint8_t {
    unavailable_no_calibration_product,
};

enum class IdentityCalibrationForPtcAdmissionState : std::uint8_t {
    not_evaluated_product_unavailable,
};

class IdentityRouteCalibrationState {
public:
    const std::shared_ptr<const IdentityRtcOutputContext> &
    rtc_context_handle() const noexcept;
    IdentityCalibrationProductState product_state() const noexcept;
    IdentityCalibrationUnitState unit_state() const noexcept;
    IdentityCalibrationResponseState response_state() const noexcept;
    IdentityCalibrationUncertaintyState uncertainty_state() const noexcept;

private:
    friend class IdentityRouteMapFacingBundle;
    explicit IdentityRouteCalibrationState(
        std::shared_ptr<const IdentityRtcOutputContext> rtc_context);

    std::shared_ptr<const IdentityRtcOutputContext> rtc_context_;
};

enum class IdentityPtcProductState : std::uint8_t {
    unavailable_component_not_admitted,
};

enum class IdentityPtcConditioningState : std::uint8_t {
    unavailable_no_ptc_product,
};

enum class IdentityPtcResponseState : std::uint8_t {
    unavailable_no_ptc_product,
};

enum class IdentityPtcUncertaintyState : std::uint8_t {
    unavailable_no_ptc_product,
};

enum class IdentityPtcForMapAdmissionState : std::uint8_t {
    not_evaluated_product_unavailable,
};

class IdentityRoutePtcState {
public:
    const std::shared_ptr<const IdentityRtcOutputContext> &
    rtc_context_handle() const noexcept;
    IdentityPtcProductState product_state() const noexcept;
    IdentityPtcConditioningState conditioning_state() const noexcept;
    IdentityPtcResponseState response_state() const noexcept;
    IdentityPtcUncertaintyState uncertainty_state() const noexcept;

private:
    friend class IdentityRouteMapFacingBundle;
    explicit IdentityRoutePtcState(
        std::shared_ptr<const IdentityRtcOutputContext> rtc_context);

    std::shared_ptr<const IdentityRtcOutputContext> rtc_context_;
};

// VAL owns consumer-specific use dispositions separately from both producer
// validity and the unavailable CAL/PTC product states.
class IdentityRouteCalibrationForPtcValDisposition {
public:
    const std::shared_ptr<const IdentityRtcOutputContext> &
    rtc_context_handle() const noexcept;
    IdentityCalibrationForPtcAdmissionState state() const noexcept;

private:
    friend class IdentityRouteMapFacingBundle;
    explicit IdentityRouteCalibrationForPtcValDisposition(
        std::shared_ptr<const IdentityRtcOutputContext> rtc_context);

    std::shared_ptr<const IdentityRtcOutputContext> rtc_context_;
};

class IdentityRoutePtcForMapValDisposition {
public:
    const std::shared_ptr<const IdentityRtcOutputContext> &
    rtc_context_handle() const noexcept;
    IdentityPtcForMapAdmissionState state() const noexcept;

private:
    friend class IdentityRouteMapFacingBundle;
    explicit IdentityRoutePtcForMapValDisposition(
        std::shared_ptr<const IdentityRtcOutputContext> rtc_context);

    std::shared_ptr<const IdentityRtcOutputContext> rtc_context_;
};

enum class IdentityMapAdmissionState : std::uint8_t {
    unavailable_calibration_and_ptc_products,
};

// Complete here means only that the truthful typed context reaches the MAP
// boundary. MAP is not admitted and no MAP action or product exists.
class IdentityRouteMapFacingBundle {
public:
    static std::shared_ptr<const IdentityRouteMapFacingBundle> assemble(
        std::shared_ptr<const IdentityRtcOutputContext> rtc_context);

    const std::shared_ptr<const IdentityRtcOutputContext> &
    rtc_context_handle() const noexcept;
    const IdentityRouteCalibrationState &calibration_state() const noexcept;
    const IdentityRouteCalibrationForPtcValDisposition &
    calibration_for_ptc_val_disposition() const noexcept;
    const IdentityRoutePtcState &ptc_state() const noexcept;
    const IdentityRoutePtcForMapValDisposition &
    ptc_for_map_val_disposition() const noexcept;
    IdentityMapAdmissionState map_admission_state() const noexcept;
    bool map_action_performed() const noexcept;

private:
    explicit IdentityRouteMapFacingBundle(
        std::shared_ptr<const IdentityRtcOutputContext> rtc_context);

    std::shared_ptr<const IdentityRtcOutputContext> rtc_context_;
    IdentityRouteCalibrationState calibration_;
    IdentityRouteCalibrationForPtcValDisposition calibration_val_;
    IdentityRoutePtcState ptc_;
    IdentityRoutePtcForMapValDisposition ptc_val_;
};

enum class IdentityRouteContextState : std::uint8_t {
    map_facing_context_complete,
    input_context_failed,
    rtc_failed,
    output_context_failed,
};

enum class IdentityRouteContextFailureCause : std::uint8_t {
    none,
    missing_align_context,
    align_input_binding_mismatch,
    input_context_rejected,
    rtc_route_rejected,
    output_context_rejected,
};

struct IdentityRouteContextRequest {
    std::shared_ptr<const IdentityRouteAlignContext> align_context;
    RtcOnlyRouteRequest rtc;
};

struct IdentityRouteContextOutcome {
    IdentityRouteContextState state =
        IdentityRouteContextState::input_context_failed;
    IdentityRouteContextFailureCause failure_cause =
        IdentityRouteContextFailureCause::none;
    std::string failure_detail;
    RtcOnlyTerminalResult rtc_terminal;
    std::shared_ptr<const IdentityRouteMapFacingBundle> map_facing_bundle;

    bool map_facing_context_complete() const noexcept {
        return state ==
                   IdentityRouteContextState::map_facing_context_complete &&
            map_facing_bundle != nullptr;
    }
};

IdentityRouteContextOutcome run_identity_route_context(
    const IdentityRouteContextRequest &request,
    RtcOnlyProductSlot &rtc_publication);

}  // namespace citlali::pipeline
