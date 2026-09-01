#include <citlali/core/pipeline/timestream_identity_route_context.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace citlali::pipeline {
namespace {

double occurrence_midpoint(
    const NativeReadoutIntegrationSupport &support) {
    const double midpoint =
        std::midpoint(support.begin_unix_sec, support.end_unix_sec);
    if (!std::isfinite(midpoint)) {
        throw std::invalid_argument(
            "identity route occurrence midpoint is not finite");
    }
    return midpoint;
}

bool exact_spans(std::span<const NativeOccurrenceSpan> lhs,
                 std::span<const NativeOccurrenceSpan> rhs) noexcept {
    if (lhs.size() != rhs.size()) return false;
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        if (lhs[index].network_id != rhs[index].network_id ||
            lhs[index].first_native_row != rhs[index].first_native_row ||
            lhs[index].past_last_native_row !=
                rhs[index].past_last_native_row) {
            return false;
        }
    }
    return true;
}

}  // namespace

std::shared_ptr<const IdentityRouteAlignContext>
IdentityRouteAlignContext::admit(
    std::shared_ptr<const NativePairedReadoutObservation> paired,
    std::shared_ptr<const AstScanMotionNetworkViews> ast_views) {
    if (!paired || !ast_views || !(paired->scope() == ast_views->scope())) {
        throw std::invalid_argument(
            "identity route ALIGN context requires matching Paired-D1 and AST scopes");
    }

    const auto &paired_ids = paired->participant_network_ids();
    const auto ast_ids = ast_views->participant_network_ids();
    if (paired_ids.size() != ast_ids.size() ||
        !std::equal(paired_ids.begin(), paired_ids.end(),
                    ast_ids.begin())) {
        throw std::invalid_argument(
            "identity route ALIGN context requires exact network participants");
    }

    for (const auto network_id : paired_ids) {
        const auto &paired_network = paired->network(network_id);
        const auto &axis = paired_network.occurrence_axis();
        const auto &ast_network = ast_views->network(network_id);
        if (ast_network.network_timing_handle().get() !=
                axis.native_timing_handle().get() ||
            axis.first_native_row() < ast_network.first_native_row() ||
            axis.past_last_native_row() >
                ast_network.past_last_native_row()) {
            throw std::invalid_argument(
                "identity route ALIGN context requires the exact Paired-D1 native timing handle and support");
        }
        for (auto row = axis.first_native_row();
             row < axis.past_last_native_row(); ++row) {
            const auto paired_identity = axis.native_identity(row);
            if (!(paired_identity == ast_network.identity(row))) {
                throw std::invalid_argument(
                    "identity route ALIGN occurrence identity differs from AST");
            }
            const double midpoint = occurrence_midpoint(
                axis.occurrence(row).integration_support);
            if (midpoint !=
                paired_identity.reconstructed_time_unix_sec()) {
                throw std::invalid_argument(
                    "identity route native event time is not the approved occurrence midpoint");
            }
        }
    }

    return std::shared_ptr<const IdentityRouteAlignContext>(
        new IdentityRouteAlignContext{std::move(paired),
                                      std::move(ast_views)});
}

IdentityRouteAlignContext::IdentityRouteAlignContext(
    std::shared_ptr<const NativePairedReadoutObservation> paired,
    std::shared_ptr<const AstScanMotionNetworkViews> ast_views)
    : paired_{std::move(paired)}, ast_views_{std::move(ast_views)} {}

const NativeObservationScope &IdentityRouteAlignContext::scope()
    const noexcept {
    return paired_->scope();
}

const std::shared_ptr<const NativePairedReadoutObservation> &
IdentityRouteAlignContext::paired_handle() const noexcept {
    return paired_;
}

const std::shared_ptr<const AstScanMotionNetworkViews> &
IdentityRouteAlignContext::ast_views_handle() const noexcept {
    return ast_views_;
}

std::span<const TimestreamNetworkId>
IdentityRouteAlignContext::participant_network_ids() const noexcept {
    return paired_->participant_network_ids();
}

IdentityRouteOccurrenceTimePolicy
IdentityRouteAlignContext::occurrence_time_policy() const noexcept {
    return IdentityRouteOccurrenceTimePolicy::
        integration_interval_midpoint;
}

IdentityRouteOccurrenceAssignment
IdentityRouteAlignContext::occurrence_assignment(
    TimestreamNetworkId network_id,
    TimestreamNativeRow native_row) const {
    const auto &axis = paired_->network(network_id).occurrence_axis();
    const auto &binding = axis.occurrence(native_row);
    const auto identity = axis.native_identity(native_row);
    const double midpoint =
        occurrence_midpoint(binding.integration_support);
    if (midpoint != identity.reconstructed_time_unix_sec() ||
        !(identity ==
          ast_views_->network(network_id).identity(native_row))) {
        throw std::logic_error(
            "identity route occurrence no longer satisfies exact ALIGN binding");
    }
    return {identity,
            binding.parent_readout_occurrence_key,
            binding.paired_xr_occurrence_key,
            binding.integration_support,
            midpoint};
}

const AstScanMotionMappedRecord &
IdentityRouteAlignContext::ast_motion_record(
    TimestreamNetworkId network_id,
    TimestreamNativeRow native_row) const {
    (void)occurrence_assignment(network_id, native_row);
    return ast_views_->network(network_id).record(native_row);
}

std::optional<AstScanMotionMappedSupport>
IdentityRouteAlignContext::ast_motion_support(
    TimestreamNetworkId network_id,
    TimestreamNativeRow native_row) const {
    (void)occurrence_assignment(network_id, native_row);
    return ast_views_->network(network_id).support(native_row);
}

IdentityRouteAlignMemoryEvidence
IdentityRouteAlignContext::memory_evidence() const noexcept {
    return {0, 0, 1, 1};
}

std::shared_ptr<const IdentityRtcInputContext>
IdentityRtcInputContext::admit(
    std::shared_ptr<const IdentityRouteAlignContext> align_context,
    std::vector<NativeOccurrenceSpan> logical_spans) {
    if (!align_context) {
        throw std::invalid_argument(
            "identity RTC input requires an ALIGN context");
    }
    auto signal = NativePairedReadoutView::admit(
        align_context->paired_handle(), std::move(logical_spans));
    for (const auto &span : signal->spans()) {
        for (auto row = span.first_native_row;
             row < span.past_last_native_row; ++row) {
            (void)align_context->occurrence_assignment(
                span.network_id, row);
        }
    }
    return std::shared_ptr<const IdentityRtcInputContext>(
        new IdentityRtcInputContext{std::move(align_context),
                                    std::move(signal)});
}

IdentityRtcInputContext::IdentityRtcInputContext(
    std::shared_ptr<const IdentityRouteAlignContext> align_context,
    std::shared_ptr<const NativePairedReadoutView> signal)
    : align_context_{std::move(align_context)},
      signal_{std::move(signal)} {}

const std::shared_ptr<const IdentityRouteAlignContext> &
IdentityRtcInputContext::align_context_handle() const noexcept {
    return align_context_;
}

const std::shared_ptr<const NativePairedReadoutView> &
IdentityRtcInputContext::signal_handle() const noexcept {
    return signal_;
}

const std::shared_ptr<const AstScanMotionNetworkViews> &
IdentityRtcInputContext::ast_views_handle() const noexcept {
    return align_context_->ast_views_handle();
}

IdentityRtcAstDependency IdentityRtcInputContext::ast_dependency()
    const noexcept {
    return IdentityRtcAstDependency::not_applicable;
}

IdentityRtcInputContextMemoryEvidence
IdentityRtcInputContext::memory_evidence() const noexcept {
    return {0, signal_->spans().size() * sizeof(NativeOccurrenceSpan), 1,
            1};
}

std::shared_ptr<const IdentityRtcOutputContext>
IdentityRtcOutputContext::admit(
    std::shared_ptr<const IdentityRtcInputContext> input_context,
    std::shared_ptr<const RtcOnlyTerminalProduct> rtc_terminal) {
    if (!input_context || !rtc_terminal ||
        !rtc_terminal->terminal_result().complete() ||
        !rtc_terminal->timestream_handle()) {
        throw std::invalid_argument(
            "identity RTC output context requires a complete terminal");
    }
    const auto &product = *rtc_terminal->timestream_handle();
    if (product.input_handle().get() !=
            input_context->signal_handle().get() ||
        product.native_parent_handle().get() !=
            input_context->align_context_handle()
                ->paired_handle()
                .get() ||
        !exact_spans(product.network_spans(),
                     input_context->signal_handle()->spans())) {
        throw std::invalid_argument(
            "identity RTC output does not bind the exact input context");
    }
    const RtcIdentityOperator expected_operator;
    if (!(product.realized_operator() == expected_operator) ||
        rtc_terminal->realization().realized_sampling_factor != 1 ||
        product.memory_evidence().owned_numeric_bytes != 0) {
        throw std::invalid_argument(
            "identity RTC output is not the exact zero-copy factor-one realization");
    }
    return std::shared_ptr<const IdentityRtcOutputContext>(
        new IdentityRtcOutputContext{std::move(input_context),
                                     std::move(rtc_terminal)});
}

IdentityRtcOutputContext::IdentityRtcOutputContext(
    std::shared_ptr<const IdentityRtcInputContext> input_context,
    std::shared_ptr<const RtcOnlyTerminalProduct> rtc_terminal)
    : input_context_{std::move(input_context)},
      rtc_terminal_{std::move(rtc_terminal)} {}

const std::shared_ptr<const IdentityRtcInputContext> &
IdentityRtcOutputContext::input_context_handle() const noexcept {
    return input_context_;
}

const std::shared_ptr<const RtcOnlyTerminalProduct> &
IdentityRtcOutputContext::rtc_terminal_handle() const noexcept {
    return rtc_terminal_;
}

const std::shared_ptr<const RtcTimestream> &
IdentityRtcOutputContext::signal_handle() const noexcept {
    return rtc_terminal_->timestream_handle();
}

const std::shared_ptr<const AstScanMotionNetworkViews> &
IdentityRtcOutputContext::ast_views_handle() const noexcept {
    return input_context_->ast_views_handle();
}

IdentityRtcSamplingRelation
IdentityRtcOutputContext::sampling_relation() const noexcept {
    return IdentityRtcSamplingRelation::native_factor_one_phase_zero;
}

IdentityRouteSignalUnitState
IdentityRtcOutputContext::signal_unit_state() const noexcept {
    return IdentityRouteSignalUnitState::raw_paired_xr;
}

IdentityRouteOccurrenceAssignment
IdentityRtcOutputContext::occurrence_assignment(
    TimestreamNetworkId network_id,
    TimestreamNativeRow native_row) const {
    (void)signal_handle()->representative_native_identity(
        network_id, native_row);
    return input_context_->align_context_handle()->occurrence_assignment(
        network_id, native_row);
}

const AstScanMotionMappedRecord &
IdentityRtcOutputContext::ast_motion_record(
    TimestreamNetworkId network_id,
    TimestreamNativeRow native_row) const {
    (void)signal_handle()->representative_native_identity(
        network_id, native_row);
    return input_context_->align_context_handle()->ast_motion_record(
        network_id, native_row);
}

std::optional<AstScanMotionMappedSupport>
IdentityRtcOutputContext::ast_motion_support(
    TimestreamNetworkId network_id,
    TimestreamNativeRow native_row) const {
    (void)signal_handle()->representative_native_identity(
        network_id, native_row);
    return input_context_->align_context_handle()->ast_motion_support(
        network_id, native_row);
}

IdentityRtcOutputContextMemoryEvidence
IdentityRtcOutputContext::memory_evidence() const noexcept {
    return {0, 0, 1, 1};
}

IdentityRouteCalibrationState::IdentityRouteCalibrationState(
    std::shared_ptr<const IdentityRtcOutputContext> rtc_context)
    : rtc_context_{std::move(rtc_context)} {}

const std::shared_ptr<const IdentityRtcOutputContext> &
IdentityRouteCalibrationState::rtc_context_handle() const noexcept {
    return rtc_context_;
}

IdentityCalibrationProductState
IdentityRouteCalibrationState::product_state() const noexcept {
    return IdentityCalibrationProductState::
        unavailable_component_not_admitted;
}

IdentityCalibrationUnitState
IdentityRouteCalibrationState::unit_state() const noexcept {
    return IdentityCalibrationUnitState::
        unavailable_no_calibration_product;
}

IdentityCalibrationResponseState
IdentityRouteCalibrationState::response_state() const noexcept {
    return IdentityCalibrationResponseState::
        unavailable_no_calibration_product;
}

IdentityCalibrationUncertaintyState
IdentityRouteCalibrationState::uncertainty_state() const noexcept {
    return IdentityCalibrationUncertaintyState::
        unavailable_no_calibration_product;
}

IdentityRoutePtcState::IdentityRoutePtcState(
    std::shared_ptr<const IdentityRtcOutputContext> rtc_context)
    : rtc_context_{std::move(rtc_context)} {}

const std::shared_ptr<const IdentityRtcOutputContext> &
IdentityRoutePtcState::rtc_context_handle() const noexcept {
    return rtc_context_;
}

IdentityPtcProductState IdentityRoutePtcState::product_state()
    const noexcept {
    return IdentityPtcProductState::unavailable_component_not_admitted;
}

IdentityPtcConditioningState
IdentityRoutePtcState::conditioning_state() const noexcept {
    return IdentityPtcConditioningState::unavailable_no_ptc_product;
}

IdentityPtcResponseState IdentityRoutePtcState::response_state()
    const noexcept {
    return IdentityPtcResponseState::unavailable_no_ptc_product;
}

IdentityPtcUncertaintyState
IdentityRoutePtcState::uncertainty_state() const noexcept {
    return IdentityPtcUncertaintyState::unavailable_no_ptc_product;
}

IdentityRouteCalibrationForPtcValDisposition::
    IdentityRouteCalibrationForPtcValDisposition(
        std::shared_ptr<const IdentityRtcOutputContext> rtc_context)
    : rtc_context_{std::move(rtc_context)} {}

const std::shared_ptr<const IdentityRtcOutputContext> &
IdentityRouteCalibrationForPtcValDisposition::rtc_context_handle()
    const noexcept {
    return rtc_context_;
}

IdentityCalibrationForPtcAdmissionState
IdentityRouteCalibrationForPtcValDisposition::state() const noexcept {
    return IdentityCalibrationForPtcAdmissionState::
        not_evaluated_product_unavailable;
}

IdentityRoutePtcForMapValDisposition::
    IdentityRoutePtcForMapValDisposition(
        std::shared_ptr<const IdentityRtcOutputContext> rtc_context)
    : rtc_context_{std::move(rtc_context)} {}

const std::shared_ptr<const IdentityRtcOutputContext> &
IdentityRoutePtcForMapValDisposition::rtc_context_handle()
    const noexcept {
    return rtc_context_;
}

IdentityPtcForMapAdmissionState
IdentityRoutePtcForMapValDisposition::state() const noexcept {
    return IdentityPtcForMapAdmissionState::
        not_evaluated_product_unavailable;
}

std::shared_ptr<const IdentityRouteMapFacingBundle>
IdentityRouteMapFacingBundle::assemble(
    std::shared_ptr<const IdentityRtcOutputContext> rtc_context) {
    if (!rtc_context) {
        throw std::invalid_argument(
            "MAP-facing identity bundle requires an RTC context");
    }
    return std::shared_ptr<const IdentityRouteMapFacingBundle>(
        new IdentityRouteMapFacingBundle{std::move(rtc_context)});
}

IdentityRouteMapFacingBundle::IdentityRouteMapFacingBundle(
    std::shared_ptr<const IdentityRtcOutputContext> rtc_context)
    : rtc_context_{std::move(rtc_context)},
      calibration_{rtc_context_},
      calibration_val_{rtc_context_},
      ptc_{rtc_context_},
      ptc_val_{rtc_context_} {}

const std::shared_ptr<const IdentityRtcOutputContext> &
IdentityRouteMapFacingBundle::rtc_context_handle() const noexcept {
    return rtc_context_;
}

const IdentityRouteCalibrationState &
IdentityRouteMapFacingBundle::calibration_state() const noexcept {
    return calibration_;
}

const IdentityRouteCalibrationForPtcValDisposition &
IdentityRouteMapFacingBundle::calibration_for_ptc_val_disposition()
    const noexcept {
    return calibration_val_;
}

const IdentityRoutePtcState &
IdentityRouteMapFacingBundle::ptc_state() const noexcept {
    return ptc_;
}

const IdentityRoutePtcForMapValDisposition &
IdentityRouteMapFacingBundle::ptc_for_map_val_disposition()
    const noexcept {
    return ptc_val_;
}

IdentityMapAdmissionState
IdentityRouteMapFacingBundle::map_admission_state() const noexcept {
    return IdentityMapAdmissionState::
        unavailable_calibration_and_ptc_products;
}

bool IdentityRouteMapFacingBundle::map_action_performed() const noexcept {
    return false;
}

IdentityRouteContextOutcome run_identity_route_context(
    const IdentityRouteContextRequest &request,
    RtcOnlyProductSlot &rtc_publication) {
    IdentityRouteContextOutcome result;
    result.rtc_terminal.identity = request.rtc.identity;
    if (!request.align_context) {
        result.failure_cause =
            IdentityRouteContextFailureCause::missing_align_context;
        result.failure_detail =
            "identity route requires an ALIGN context";
        return result;
    }
    if (request.rtc.native_input.get() !=
        request.align_context->paired_handle().get()) {
        result.failure_cause = IdentityRouteContextFailureCause::
            align_input_binding_mismatch;
        result.failure_detail =
            "identity route RTC request differs from the ALIGN-bound Paired-D1 input";
        return result;
    }

    std::shared_ptr<const IdentityRtcInputContext> input_context;
    try {
        input_context = IdentityRtcInputContext::admit(
            request.align_context, request.rtc.logical_spans);
    } catch (const std::exception &error) {
        result.failure_cause =
            IdentityRouteContextFailureCause::input_context_rejected;
        result.failure_detail = error.what();
        return result;
    }

    auto rtc_request = request.rtc;
    rtc_request.admitted_logical_input = input_context->signal_handle();
    auto rtc_outcome =
        run_identity_rtc_only(rtc_request, rtc_publication);
    result.rtc_terminal = rtc_outcome.terminal;
    if (!rtc_outcome.complete()) {
        result.state = IdentityRouteContextState::rtc_failed;
        result.failure_cause =
            IdentityRouteContextFailureCause::rtc_route_rejected;
        result.failure_detail = rtc_outcome.terminal.failure_detail;
        return result;
    }

    try {
        auto output_context = IdentityRtcOutputContext::admit(
            std::move(input_context), rtc_outcome.published_product);
        result.map_facing_bundle =
            IdentityRouteMapFacingBundle::assemble(
                std::move(output_context));
    } catch (const std::exception &error) {
        result.state = IdentityRouteContextState::output_context_failed;
        result.failure_cause =
            IdentityRouteContextFailureCause::output_context_rejected;
        result.failure_detail = error.what();
        return result;
    }

    result.state =
        IdentityRouteContextState::map_facing_context_complete;
    result.failure_cause = IdentityRouteContextFailureCause::none;
    return result;
}

}  // namespace citlali::pipeline
