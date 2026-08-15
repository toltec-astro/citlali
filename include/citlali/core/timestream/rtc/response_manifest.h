#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace timestream {

enum class RTCResponseState {
    complete,
    unavailable,
};

enum class RTCManifestObjectState {
    complete,
    failed,
};

enum class RTCResponseRole {
    signal_changing,
    support_or_eligibility,
    identity_boundary,
};

enum class RTCResponseStageState {
    complete,
    partial,
    unavailable,
    not_applicable,
};

enum class RTCUnavailableReasonCode {
    missing_or_mismatched_artifact_identity,
    incompatible_grid_wcs_or_units,
    invalid_detector_selection_or_weight,
    invalid_gaussian_parameter,
    invalid_support_denominator,
    incomplete_rectangular_support,
    missing_stage_identity,
    omitted_enabled_response_stage,
    incomplete_response_provenance,
};

enum class RTCResponseStageSlot {
    enabled_pre_calibration_response_changes,
    calibration_and_extinction,
    reference_kernel_realization,
    donor_replacement,
    pre_filter_response_changes,
    primary_fir_and_configured_notch,
    highpass_and_detector_local_filtering,
    edge_support_eligibility,
    inner_crop,
    phase_zero_downsampling,
    post_downsampling_response_changes,
    post_downsampling_masks_and_eligibility,
    altaz_projection_when_enabled,
    rtc_conditioned_inner_terminal,
    ptc_consumer_boundary_link,
};

inline constexpr std::array<RTCResponseStageSlot, 15>
    rtc_response_stage_slots{
        RTCResponseStageSlot::enabled_pre_calibration_response_changes,
        RTCResponseStageSlot::calibration_and_extinction,
        RTCResponseStageSlot::reference_kernel_realization,
        RTCResponseStageSlot::donor_replacement,
        RTCResponseStageSlot::pre_filter_response_changes,
        RTCResponseStageSlot::primary_fir_and_configured_notch,
        RTCResponseStageSlot::highpass_and_detector_local_filtering,
        RTCResponseStageSlot::edge_support_eligibility,
        RTCResponseStageSlot::inner_crop,
        RTCResponseStageSlot::phase_zero_downsampling,
        RTCResponseStageSlot::post_downsampling_response_changes,
        RTCResponseStageSlot::post_downsampling_masks_and_eligibility,
        RTCResponseStageSlot::altaz_projection_when_enabled,
        RTCResponseStageSlot::rtc_conditioned_inner_terminal,
        RTCResponseStageSlot::ptc_consumer_boundary_link,
    };

inline constexpr std::string_view rtc_response_stage_slot_name(
    RTCResponseStageSlot slot) {
    switch (slot) {
        case RTCResponseStageSlot::enabled_pre_calibration_response_changes:
            return "enabled_pre_calibration_response_changes";
        case RTCResponseStageSlot::calibration_and_extinction:
            return "calibration_and_extinction";
        case RTCResponseStageSlot::reference_kernel_realization:
            return "reference_kernel_realization";
        case RTCResponseStageSlot::donor_replacement:
            return "donor_replacement";
        case RTCResponseStageSlot::pre_filter_response_changes:
            return "pre_filter_response_changes";
        case RTCResponseStageSlot::primary_fir_and_configured_notch:
            return "primary_FIR_and_configured_notch";
        case RTCResponseStageSlot::highpass_and_detector_local_filtering:
            return "highpass_and_detector_local_filtering";
        case RTCResponseStageSlot::edge_support_eligibility:
            return "edge_support_eligibility";
        case RTCResponseStageSlot::inner_crop:
            return "inner_crop";
        case RTCResponseStageSlot::phase_zero_downsampling:
            return "phase_zero_downsampling";
        case RTCResponseStageSlot::post_downsampling_response_changes:
            return "post_downsampling_response_changes";
        case RTCResponseStageSlot::post_downsampling_masks_and_eligibility:
            return "post_downsampling_masks_and_eligibility";
        case RTCResponseStageSlot::altaz_projection_when_enabled:
            return "altaz_projection_when_enabled";
        case RTCResponseStageSlot::rtc_conditioned_inner_terminal:
            return "rtc_conditioned_inner_terminal";
        case RTCResponseStageSlot::ptc_consumer_boundary_link:
            return "PTC_consumer_boundary_link";
    }
    return "";
}

inline bool rtc_exact_sha256_digest(std::string_view value) {
    constexpr std::string_view prefix = "sha256:";
    if (value.size() != prefix.size() + 64 ||
        value.substr(0, prefix.size()) != prefix) {
        return false;
    }
    return std::all_of(
        value.begin() + static_cast<std::ptrdiff_t>(prefix.size()),
        value.end(), [](char value_char) {
            const unsigned char character =
                static_cast<unsigned char>(value_char);
            return std::isdigit(character) != 0 ||
                   (value_char >= 'a' && value_char <= 'f');
        });
}

struct RTCArtifactIdentity {
    std::string id;
    std::string digest;
    std::optional<std::uint64_t> byte_count;
};

struct RTCUnavailableReason {
    RTCUnavailableReasonCode code =
        RTCUnavailableReasonCode::missing_or_mismatched_artifact_identity;
    std::string affected_object;
    std::string evidence;
};

struct RTCResponseStage {
    RTCResponseStageSlot slot =
        RTCResponseStageSlot::enabled_pre_calibration_response_changes;
    std::string stage_id;
    std::string stage_version;
    bool enabled = false;
    bool applicable = false;
    std::size_t order = 0;
    RTCResponseRole response_role = RTCResponseRole::identity_boundary;
    std::vector<RTCArtifactIdentity> input_parents;
    RTCArtifactIdentity output_parent;
    std::string operator_identity;
    std::string normalization_state;
    RTCArtifactIdentity grid_identity;
    RTCArtifactIdentity wcs_identity;
    RTCArtifactIdentity support_identity;
    RTCArtifactIdentity validity_identity;
    RTCResponseStageState response_state = RTCResponseStageState::unavailable;
    RTCArtifactIdentity artifact;
};

struct RTCResponseManifest {
    RTCArtifactIdentity input_observation;
    RTCArtifactIdentity input_reduction;
    RTCArtifactIdentity detector;
    RTCArtifactIdentity cohort;
    RTCArtifactIdentity scan;
    RTCArtifactIdentity sample_origin;
    RTCArtifactIdentity native_time_origin;
    RTCArtifactIdentity grid;
    RTCArtifactIdentity wcs;
    RTCArtifactIdentity support;
    RTCArtifactIdentity validity;
    std::vector<RTCArtifactIdentity> input_parents;
    std::string requested_state;
    std::string effective_state;
    std::string observation_resolved_state;
    std::string realized_state;
    std::string output_stage_identity;
    RTCResponseState response_state = RTCResponseState::unavailable;
    std::vector<RTCUnavailableReason> unavailable_reasons;
    std::string normalization_at_rtc_terminal_ptc_boundary;
    std::string normalization_before_jinc_consumption;
    std::string final_manifest_digest;
    std::optional<std::uint64_t> final_manifest_byte_count;
    std::vector<std::string> final_manifest_bundle_identities;
    std::vector<std::string> final_manifest_member_identities;
    RTCManifestObjectState object_state = RTCManifestObjectState::failed;
    std::vector<RTCResponseStage> stages;
};

struct RTCResponseManifestValidation {
    bool valid = false;
    std::vector<std::string> errors;

    bool contains(std::string_view text) const {
        return std::any_of(errors.begin(), errors.end(), [&](const auto &error) {
            return error.find(text) != std::string::npos;
        });
    }

    std::string summary() const {
        std::ostringstream stream;
        for (std::size_t index = 0; index < errors.size(); ++index) {
            if (index != 0) {
                stream << "; ";
            }
            stream << errors[index];
        }
        return stream.str();
    }
};

inline RTCResponseManifestValidation validate_rtc_response_manifest(
    const RTCResponseManifest &manifest) {
    RTCResponseManifestValidation validation;
    auto fail = [&](std::string message) {
        validation.errors.push_back(std::move(message));
    };
    auto require_artifact = [&](const RTCArtifactIdentity &identity,
                                std::string_view field) {
        if (identity.id.empty()) {
            fail(std::string(field) + " identity is missing");
        }
        if (!rtc_exact_sha256_digest(identity.digest)) {
            fail(std::string(field) + " digest is not exact SHA-256");
        }
        if (!identity.byte_count.has_value()) {
            fail(std::string(field) + " byte count is missing");
        }
    };
    auto require_text = [&](const std::string &value, std::string_view field) {
        if (value.empty()) {
            fail(std::string(field) + " is missing");
        }
    };

    require_artifact(manifest.input_observation, "input observation");
    require_artifact(manifest.input_reduction, "input reduction");
    require_artifact(manifest.detector, "detector");
    require_artifact(manifest.cohort, "cohort");
    require_artifact(manifest.scan, "scan");
    require_artifact(manifest.sample_origin, "sample origin");
    require_artifact(manifest.native_time_origin, "native-time origin");
    require_artifact(manifest.grid, "grid");
    require_artifact(manifest.wcs, "WCS");
    require_artifact(manifest.support, "support");
    require_artifact(manifest.validity, "validity");
    if (manifest.input_parents.empty()) {
        fail("input parents are missing");
    }
    for (const auto &parent : manifest.input_parents) {
        require_artifact(parent, "input parent");
    }
    require_text(manifest.requested_state, "requested state");
    require_text(manifest.effective_state, "effective state");
    require_text(manifest.observation_resolved_state,
                 "observation-resolved state");
    require_text(manifest.realized_state, "realized state");
    if (manifest.output_stage_identity != "rtc_conditioned_inner_terminal") {
        fail("output stage is not rtc_conditioned_inner_terminal");
    }
    require_text(manifest.normalization_at_rtc_terminal_ptc_boundary,
                 "RTC-terminal/PTC normalization state");
    require_text(manifest.normalization_before_jinc_consumption,
                 "pre-JINC normalization state");
    if (!rtc_exact_sha256_digest(manifest.final_manifest_digest)) {
        fail("final manifest digest is not exact SHA-256");
    }
    if (!manifest.final_manifest_byte_count.has_value()) {
        fail("final manifest byte count is missing");
    }
    if (manifest.final_manifest_bundle_identities.empty()) {
        fail("final manifest bundle identities are missing");
    }
    if (manifest.final_manifest_member_identities.empty()) {
        fail("final manifest member identities are missing");
    }
    if (manifest.object_state == RTCManifestObjectState::failed) {
        fail("failed manifest object is not an admissible response parent");
    }

    if (manifest.response_state == RTCResponseState::unavailable) {
        if (manifest.unavailable_reasons.empty()) {
            fail("unavailable response has no controlled reason");
        }
        for (const auto &reason : manifest.unavailable_reasons) {
            require_text(reason.affected_object,
                         "unavailable reason affected object");
            require_text(reason.evidence, "unavailable reason evidence");
        }
    }
    else if (!manifest.unavailable_reasons.empty()) {
        fail("complete response carries unavailable reasons");
    }

    if (manifest.stages.size() != rtc_response_stage_slots.size()) {
        fail("RTC response stage count is not exactly 15");
    }
    const std::size_t stage_count =
        std::min(manifest.stages.size(), rtc_response_stage_slots.size());
    for (std::size_t index = 0; index < stage_count; ++index) {
        const auto &stage = manifest.stages[index];
        if (stage.slot != rtc_response_stage_slots[index]) {
            fail("RTC response stage order does not match frozen coverage");
        }
        if (stage.order != index + 1) {
            fail("RTC response stage ordinal does not match frozen coverage");
        }
        require_text(stage.stage_id, "stage id");
        require_text(stage.stage_version, "stage version");
        if (stage.input_parents.empty()) {
            fail("stage input parents are missing");
        }
        for (const auto &parent : stage.input_parents) {
            require_artifact(parent, "stage input parent");
        }
        require_artifact(stage.output_parent, "stage output parent");
        require_text(stage.operator_identity, "stage operator identity");
        require_text(stage.normalization_state,
                     "stage normalization state");
        require_artifact(stage.grid_identity, "stage grid");
        require_artifact(stage.wcs_identity, "stage WCS");
        require_artifact(stage.support_identity, "stage support");
        require_artifact(stage.validity_identity, "stage validity");
        require_artifact(stage.artifact, "stage artifact");
        if (manifest.response_state == RTCResponseState::complete &&
            stage.enabled && stage.applicable &&
            stage.response_state != RTCResponseStageState::complete) {
            fail("complete response contains a non-complete stage");
        }
    }

    validation.valid = validation.errors.empty();
    return validation;
}

inline constexpr std::string_view
    sci_map_003_governing_application_commit =
        "46ad23888a40f5102cdfd50c06e49a549bdf8a20";

inline RTCResponseManifestValidation
validate_rtc_response_manifest_at_sci_map_003_governing_base(
    const RTCResponseManifest &manifest) {
    auto validation = validate_rtc_response_manifest(manifest);
    auto fail = [&](std::string message) {
        validation.errors.push_back(std::move(message));
    };
    if (manifest.response_state != RTCResponseState::unavailable) {
        fail("governing application RTC response must remain unavailable");
    }
    constexpr std::array<RTCUnavailableReasonCode, 3> required_codes{
        RTCUnavailableReasonCode::omitted_enabled_response_stage,
        RTCUnavailableReasonCode::omitted_enabled_response_stage,
        RTCUnavailableReasonCode::incomplete_response_provenance,
    };
    constexpr std::array<std::string_view, 3> required_objects{
        "donor_replacement",
        "altaz_projection",
        "support_eligibility_causes_and_realized_response_identity",
    };
    if (manifest.unavailable_reasons.size() != required_codes.size()) {
        fail("governing application requires exactly three unavailable reasons");
    }
    const std::size_t reason_count =
        std::min(manifest.unavailable_reasons.size(), required_codes.size());
    for (std::size_t index = 0; index < reason_count; ++index) {
        if (manifest.unavailable_reasons[index].code != required_codes[index] ||
            manifest.unavailable_reasons[index].affected_object !=
                required_objects[index]) {
            fail("governing application unavailable reasons do not match the frozen disposition");
        }
    }
    validation.valid = validation.errors.empty();
    return validation;
}

}  // namespace timestream
