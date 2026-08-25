#pragma once

// Bounded canonical SCI-ALIGN provenance. Runtime sample masks and revision
// ledgers remain authoritative in memory; this product records authorities,
// natural-scope causes, reconciled populations, and deterministic identities
// without serializing detector-by-sample execution history. See ADR 0013.

#include <citlali/core/pipeline/native_cohort_product_provenance_v2.h>
#include <citlali/core/pipeline/native_fruit_loop_feedback.h>
#include <citlali/core/pipeline/native_noise_assignment.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace citlali::pipeline {

inline constexpr std::string_view native_cohort_product_provenance_v3_schema =
    "citlali-native-cohort-product-provenance-v3";
inline constexpr std::string_view native_cohort_policy_schema_v3 =
    "citlali-native-cohort-bounded-provenance-policy-v1";
inline constexpr NativeDetectorFlagBits
    native_cohort_duplicate_tone_exclusion_bit_v3 =
        NativeDetectorFlagBits{1} << 62U;
inline constexpr NativeDetectorFlagBits
    native_cohort_learned_rtc_exclusion_bit_v3 =
        NativeDetectorFlagBits{1} << 61U;
inline constexpr NativeDetectorFlagBits
    native_cohort_rtc_processing_flag_bit_v3 =
        NativeDetectorFlagBits{1} << 63U;

inline std::size_t native_cohort_checked_product_v3(
    std::size_t lhs, std::size_t rhs, const char *field) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error(std::string{field} + " count overflow");
    }
    return lhs * rhs;
}

inline void native_cohort_checked_add_v3(
    std::size_t &target, std::size_t value, const char *field) {
    if (value > std::numeric_limits<std::size_t>::max() - target) {
        throw std::overflow_error(std::string{field} + " count overflow");
    }
    target += value;
}

struct NativeCohortDetectorExclusionV3 {
    TimestreamDetectorColumn detector_column = -1;
    std::int64_t output_uid = -1;
    std::int64_t network = -1;
    std::int64_t channel = -1;
    std::optional<std::int64_t> apt_flag;
    std::string authority;
    std::string cause;
};

struct NativeCohortDetectorPopulationV3 {
    std::size_t detector_count = 0;
    std::size_t apt_eligible_detector_count = 0;
    std::size_t apt_excluded_detector_count = 0;
};

inline std::pair<NativeCohortDetectorPopulationV3,
                 std::vector<NativeCohortDetectorExclusionV3>>
native_cohort_detector_population_v3(
    const CanonicalAptDetectorRelationV2 &relation) {
    NativeCohortDetectorPopulationV3 population;
    population.detector_count = relation.bindings().size();
    std::vector<NativeCohortDetectorExclusionV3> exclusions;
    for (const auto &binding : relation.bindings()) {
        if (binding.flag.has_value() && *binding.flag == 0) {
            ++population.apt_eligible_detector_count;
            continue;
        }
        ++population.apt_excluded_detector_count;
        exclusions.push_back({
            static_cast<TimestreamDetectorColumn>(binding.detector_column),
            binding.output_uid, binding.network,
            binding.channel, binding.flag,
            "canonical_apt_v2.matched_detector_relation",
            binding.flag.has_value() ? "apt_flag_nonzero"
                                     : "apt_flag_typed_null"});
    }
    if (population.detector_count == 0 ||
        population.apt_eligible_detector_count +
                population.apt_excluded_detector_count !=
            population.detector_count ||
        exclusions.size() != population.apt_excluded_detector_count) {
        throw std::logic_error(
            "native cohort detector population does not reconcile");
    }
    return {population, exclusions};
}

struct NativeCohortRtcSummaryV3 {
    std::size_t run_count = 0;
    std::size_t loaded_input_row_count = 0;
    std::size_t selected_input_row_count = 0;
    std::size_t output_row_count = 0;
    std::size_t exact_support_identity_count = 0;
    std::size_t detector_support_count = 0;
    std::size_t flagged_detector_support_count = 0;
    std::size_t final_short_support_count = 0;
    std::string interval_authority;
};

struct NativeCohortPtcSummaryV3 {
    std::string requested_grouping;
    std::string effective_grouping;
    std::size_t segment_count = 0;
    std::size_t group_count = 0;
    std::size_t pca_clean_group_count = 0;
    std::size_t pass_through_group_count = 0;
    std::size_t detector_membership_count = 0;
};

struct NativeCohortScopedCauseV3 {
    std::string scope;
    std::string authority;
    std::string cause;
    std::string count_unit;
    std::optional<NativeDetectorFlagBits> flag_bits;
    std::optional<std::size_t> start_row;
    std::optional<std::size_t> end_row;
    std::size_t affected_count = 0;
    std::vector<TimestreamDetectorColumn> detector_columns;
};

struct NativeCohortPopulationV3 {
    std::size_t row_count = 0;
    std::size_t detector_count = 0;
    std::size_t detector_sample_count = 0;
    std::size_t mapped_valid_sample_count = 0;
    std::size_t mapped_invalid_sample_count = 0;
    std::size_t delivered_flagged_sample_count = 0;
    std::size_t raw_input_flagged_sample_count = 0;
    std::size_t rtc_processing_flagged_sample_count = 0;
    std::size_t learned_rtc_excluded_sample_count = 0;
    std::size_t operation_excluded_sample_count = 0;
    std::size_t apt_excluded_sample_count = 0;
    std::size_t ptc_second_pass_excluded_sample_count = 0;
    std::size_t learned_ptc_excluded_sample_count = 0;
    std::size_t postclean_outlier_excluded_sample_count = 0;
    std::size_t final_excluded_sample_count = 0;
    std::size_t replaced_by_pca_sample_count = 0;
    std::size_t preserved_pca_invalid_sample_count = 0;
    std::size_t preserved_pass_through_sample_count = 0;
    std::size_t positive_weight_detector_count = 0;
    std::size_t zero_weight_detector_count = 0;
    std::size_t eligible_map_input_sample_count = 0;
};

struct NativeCohortMapOccurrenceV3 {
    bool mapmaking_enabled = false;
    std::string method;
    std::string eligible_weight_digest;
    std::string map_index_digest;
    std::size_t map_index_count = 0;
    std::vector<TimestreamDetectorColumn> zero_weight_detector_columns;
    std::string product_occurrence;
    std::string product_identity_digest;
    std::optional<std::string> jinc_processing_configuration_digest;
    NativeNoiseAssignmentSummaryV3 noise_assignment;
    NativeFruitLoopFeedbackSummaryV3 fruit_loop_feedback;
};

struct NativeCohortScanProvenanceV3 {
    std::string observation_binding_digest;
    NativeScanChunkScope scope{NativeObservationScope{1, 0, 0}, 0, 0};
    NativeOperationIdentity operation{0, 0};
    NativeCohortRtcSummaryV3 rtc;
    NativeCohortPtcSummaryV3 ptc;
    NativeCohortPopulationV3 population;
    std::vector<NativeCohortScopedCauseV3> scoped_causes;
    NativeCohortMapOccurrenceV3 map_occurrence;

    void validate(const NativeCohortObservationBindingV2 &binding,
                  const NativeCohortDetectorPopulationV3 &detectors,
                  std::size_t scan_count) const;
};

struct NativeCohortProductProvenanceV3 {
    NativeCohortObservationBindingV2 binding;
    NativeCohortDetectorPopulationV3 detector_population;
    std::vector<NativeCohortDetectorExclusionV3> detector_exclusions;
    std::vector<NativeCohortScanProvenanceV3> scans;

    void validate_complete(std::size_t expected_scan_count) const;
};

inline void NativeCohortScanProvenanceV3::validate(
    const NativeCohortObservationBindingV2 &binding,
    const NativeCohortDetectorPopulationV3 &detectors,
    std::size_t scan_count) const {
    binding.validate();
    const auto cells = native_cohort_checked_product_v3(
        population.row_count, population.detector_count,
        "native detector sample");
    if (observation_binding_digest != binding.digest() ||
        scope.observation_scope.observation != binding.observation.observation ||
        scope.observation_scope.subobservation !=
            binding.observation.subobservation ||
        scope.observation_scope.scan != binding.observation.scan ||
        scope.scan_index != operation.scan_index || operation.scan_index < 0 ||
        static_cast<std::size_t>(operation.scan_index) >= scan_count ||
        rtc.run_count == 0 || rtc.loaded_input_row_count == 0 ||
        rtc.selected_input_row_count == 0 || rtc.output_row_count == 0 ||
        rtc.loaded_input_row_count < rtc.selected_input_row_count ||
        rtc.interval_authority !=
            "telescope.scan_indices.inner_and_outer_intervals" ||
        ptc.segment_count == 0 ||
        ptc.group_count == 0 || ptc.requested_grouping.empty() ||
        ptc.effective_grouping.empty() ||
        population.detector_count != detectors.detector_count ||
        population.detector_sample_count != cells ||
        population.mapped_valid_sample_count +
                population.mapped_invalid_sample_count != cells ||
        population.replaced_by_pca_sample_count +
                population.preserved_pca_invalid_sample_count +
                population.preserved_pass_through_sample_count != cells ||
        population.preserved_pca_invalid_sample_count !=
            population.mapped_invalid_sample_count ||
        ptc.pca_clean_group_count + ptc.pass_through_group_count !=
            ptc.group_count ||
        (!map_occurrence.mapmaking_enabled &&
         (map_occurrence.noise_assignment.enabled ||
          map_occurrence.fruit_loop_feedback.enabled))) {
        throw std::logic_error(
            "bounded native cohort scan provenance does not reconcile");
    }
    map_occurrence.noise_assignment.validate(population.detector_count);
    map_occurrence.fruit_loop_feedback.validate();
    std::size_t raw_input_cause_samples = 0;
    std::size_t rtc_processing_cause_samples = 0;
    std::size_t learned_rtc_cause_samples = 0;
    std::size_t operation_cause_samples = 0;
    std::size_t second_pass_cause_samples = 0;
    std::size_t learned_ptc_cause_samples = 0;
    std::size_t postclean_outlier_cause_samples = 0;
    for (const auto &cause : scoped_causes) {
        if (cause.scope.empty() || cause.authority.empty() ||
            cause.cause.empty() || cause.count_unit.empty() ||
            cause.affected_count == 0 ||
            cause.detector_columns.empty()) {
            throw std::logic_error(
                "bounded native cohort cause is incomplete");
        }
        if (cause.authority == "raw_kids_input.sample_flags") {
            if (cause.scope != "scan_summary" ||
                cause.cause != "raw_input_flag_bits" ||
                cause.count_unit != "detector_samples" ||
                !cause.flag_bits || *cause.flag_bits == 0 ||
                (*cause.flag_bits &
                 (native_cohort_rtc_processing_flag_bit_v3 |
                  native_cohort_learned_rtc_exclusion_bit_v3 |
                  native_cohort_duplicate_tone_exclusion_bit_v3)) != 0) {
                throw std::logic_error(
                    "bounded raw-input cause has invalid authority fields");
            }
            native_cohort_checked_add_v3(
                raw_input_cause_samples, cause.affected_count,
                "raw input flag cause");
        }
        else if (cause.authority ==
                 "citlali.native_rtc_processing_policy_v2") {
            if (cause.scope != "scan_summary" ||
                cause.cause != "rtc_processing_generated_flag" ||
                cause.count_unit != "detector_samples" ||
                cause.flag_bits !=
                    native_cohort_rtc_processing_flag_bit_v3) {
                throw std::logic_error(
                    "bounded RTC cause has invalid authority fields");
            }
            native_cohort_checked_add_v3(
                rtc_processing_cause_samples,
                cause.affected_count,
                "RTC processing flag cause");
        }
        else if (cause.authority ==
                 "citlali.learning.native_rtc_application_v1") {
            if (cause.scope != "scan_summary" ||
                cause.cause != "learned_rtc_sample_mask" ||
                cause.count_unit != "detector_samples" ||
                cause.flag_bits !=
                    native_cohort_learned_rtc_exclusion_bit_v3 ||
                cause.start_row || cause.end_row) {
                throw std::logic_error(
                    "bounded learned RTC cause has invalid authority fields");
            }
            native_cohort_checked_add_v3(
                learned_rtc_cause_samples, cause.affected_count,
                "learned RTC flag cause");
        }
        else if (cause.authority ==
                 "citlali.native_duplicate_tone_policy_v2") {
            if (cause.scope != "scan_summary" ||
                cause.cause != "duplicate_tone" ||
                cause.count_unit != "detector_samples" ||
                cause.flag_bits !=
                    native_cohort_duplicate_tone_exclusion_bit_v3) {
                throw std::logic_error(
                    "bounded duplicate-tone cause has invalid authority fields");
            }
            native_cohort_checked_add_v3(
                operation_cause_samples, cause.affected_count,
                "operation exclusion cause");
        }
        else if (cause.authority ==
                 "citlali.weighting.detector_weight_contract_v1") {
            if (cause.scope != "scan_detector" ||
                cause.cause != "nonpositive_final_detector_weight" ||
                cause.count_unit != "detectors" || cause.flag_bits ||
                cause.affected_count != cause.detector_columns.size()) {
                throw std::logic_error(
                    "bounded weight cause has invalid authority fields");
            }
        }
        else if (cause.authority ==
                 "citlali.learning.native_map_application_v1") {
            if (cause.scope != "scan_detector" ||
                cause.cause != "learned_pre_map_detector_exclusion" ||
                cause.count_unit != "detectors" || cause.flag_bits ||
                cause.start_row || cause.end_row ||
                cause.affected_count != cause.detector_columns.size()) {
                throw std::logic_error(
                    "bounded learned map cause has invalid authority fields");
            }
        }
        else if (cause.authority ==
                 "citlali.ptc.second_pass_local_v1") {
            if (cause.scope != "scan_detector_interval" ||
                cause.cause != "accepted_second_pass_event" ||
                cause.count_unit != "detector_samples" ||
                cause.flag_bits || !cause.start_row || !cause.end_row ||
                cause.detector_columns.size() != 1 ||
                *cause.start_row > *cause.end_row ||
                *cause.end_row >= population.row_count ||
                cause.affected_count !=
                    *cause.end_row - *cause.start_row + 1) {
                throw std::logic_error(
                    "bounded PTC second-pass cause has invalid interval fields");
            }
            native_cohort_checked_add_v3(
                second_pass_cause_samples, cause.affected_count,
                "PTC second-pass interval cause");
        }
        else if (cause.authority ==
                 "citlali.learning.native_ptc_application_v1") {
            if (cause.scope != "scan_detector_interval" ||
                cause.cause != "learned_ptc_sample_or_detector_mask" ||
                cause.count_unit != "detector_samples" ||
                cause.flag_bits || !cause.start_row || !cause.end_row ||
                cause.detector_columns.size() != 1 ||
                *cause.start_row > *cause.end_row ||
                *cause.end_row >= population.row_count ||
                cause.affected_count !=
                    *cause.end_row - *cause.start_row + 1) {
                throw std::logic_error(
                    "bounded learned PTC cause has invalid interval fields");
            }
            native_cohort_checked_add_v3(
                learned_ptc_cause_samples, cause.affected_count,
                "learned PTC interval cause");
        }
        else if (cause.authority ==
                 "citlali.ptc.postclean_outlier_policy_v1") {
            if (cause.scope != "scan_detector_interval" ||
                cause.cause != "postclean_detector_outlier" ||
                cause.count_unit != "detector_samples" ||
                cause.flag_bits || !cause.start_row || !cause.end_row ||
                cause.detector_columns.size() != 1 ||
                *cause.start_row > *cause.end_row ||
                *cause.end_row >= population.row_count ||
                cause.affected_count !=
                    *cause.end_row - *cause.start_row + 1) {
                throw std::logic_error(
                    "bounded post-clean outlier cause has invalid interval fields");
            }
            native_cohort_checked_add_v3(
                postclean_outlier_cause_samples, cause.affected_count,
                "post-clean outlier interval cause");
        }
        else {
            throw std::logic_error(
                "bounded native cohort cause has no admitted authority");
        }
        if (!std::is_sorted(cause.detector_columns.begin(),
                            cause.detector_columns.end()) ||
            std::adjacent_find(cause.detector_columns.begin(),
                               cause.detector_columns.end()) !=
                cause.detector_columns.end()) {
            throw std::logic_error(
                "bounded native cohort cause detector scope is not canonical");
        }
        for (const auto detector : cause.detector_columns) {
            if (detector < 0 ||
                static_cast<std::size_t>(detector) >=
                    population.detector_count) {
                throw std::logic_error(
                    "bounded native cause detector is outside support");
            }
        }
    }
    if (raw_input_cause_samples !=
            population.raw_input_flagged_sample_count ||
        rtc_processing_cause_samples !=
            population.rtc_processing_flagged_sample_count ||
        learned_rtc_cause_samples !=
            population.learned_rtc_excluded_sample_count ||
        population.delivered_flagged_sample_count <
            population.raw_input_flagged_sample_count ||
        population.delivered_flagged_sample_count <
            population.rtc_processing_flagged_sample_count ||
        population.delivered_flagged_sample_count <
            population.learned_rtc_excluded_sample_count ||
        population.delivered_flagged_sample_count >
            population.raw_input_flagged_sample_count +
                population.rtc_processing_flagged_sample_count +
                population.learned_rtc_excluded_sample_count ||
        operation_cause_samples !=
            population.operation_excluded_sample_count ||
        second_pass_cause_samples !=
            population.ptc_second_pass_excluded_sample_count ||
        learned_ptc_cause_samples !=
            population.learned_ptc_excluded_sample_count ||
        postclean_outlier_cause_samples !=
            population.postclean_outlier_excluded_sample_count ||
        population.final_excluded_sample_count !=
            population.mapped_invalid_sample_count +
                population.learned_ptc_excluded_sample_count +
                population.ptc_second_pass_excluded_sample_count +
                population.postclean_outlier_excluded_sample_count) {
        throw std::logic_error(
            "bounded native cohort named causes do not reconcile");
    }
    if (!map_occurrence.mapmaking_enabled) {
        if (!map_occurrence.method.empty() ||
            !map_occurrence.eligible_weight_digest.empty() ||
            !map_occurrence.map_index_digest.empty() ||
            map_occurrence.map_index_count != 0 ||
            !map_occurrence.zero_weight_detector_columns.empty() ||
            !map_occurrence.product_occurrence.empty() ||
            !map_occurrence.product_identity_digest.empty() ||
            map_occurrence.jinc_processing_configuration_digest ||
            population.positive_weight_detector_count != 0 ||
            population.zero_weight_detector_count != 0 ||
            population.eligible_map_input_sample_count != 0) {
            throw std::logic_error(
                "disabled native map occurrence carries provenance");
        }
        return;
    }
    if ((map_occurrence.method != "naive" &&
         map_occurrence.method != "jinc") ||
        map_occurrence.eligible_weight_digest.empty() ||
        map_occurrence.map_index_digest.empty() ||
        map_occurrence.map_index_count != population.detector_count ||
        map_occurrence.product_occurrence.empty() ||
        map_occurrence.product_identity_digest.empty() ||
        population.positive_weight_detector_count +
                population.zero_weight_detector_count !=
            population.detector_count ||
        population.eligible_map_input_sample_count >
            population.mapped_valid_sample_count ||
        map_occurrence.zero_weight_detector_columns.size() !=
            population.zero_weight_detector_count) {
        throw std::logic_error(
            "bounded native map occurrence is incomplete");
    }
    const bool jinc = map_occurrence.method == "jinc";
    if (jinc !=
            map_occurrence.jinc_processing_configuration_digest.has_value()) {
        throw std::logic_error(
            "bounded native JINC occurrence is incomplete");
    }
    if (!std::is_sorted(
            map_occurrence.zero_weight_detector_columns.begin(),
            map_occurrence.zero_weight_detector_columns.end()) ||
        std::adjacent_find(
            map_occurrence.zero_weight_detector_columns.begin(),
            map_occurrence.zero_weight_detector_columns.end()) !=
            map_occurrence.zero_weight_detector_columns.end()) {
        throw std::logic_error(
            "bounded native zero-weight detector scope is not canonical");
    }
    const std::set<TimestreamDetectorColumn> zero_weights{
        map_occurrence.zero_weight_detector_columns.begin(),
        map_occurrence.zero_weight_detector_columns.end()};
    for (const auto detector : zero_weights) {
        if (detector < 0 ||
            static_cast<std::size_t>(detector) >=
                population.detector_count) {
            throw std::logic_error(
                "bounded native zero-weight detector is outside support");
        }
    }
    for (const auto &cause : scoped_causes) {
        if (cause.authority ==
                "citlali.weighting.detector_weight_contract_v1" ||
            cause.authority ==
                "citlali.learning.native_map_application_v1") {
            for (const auto detector : cause.detector_columns) {
                if (!zero_weights.contains(detector)) {
                    throw std::logic_error(
                        "bounded native weight cause retains a positive-weight detector");
                }
            }
        }
    }
}

inline void NativeCohortProductProvenanceV3::validate_complete(
    std::size_t expected_scan_count) const {
    binding.validate();
    if (expected_scan_count == 0 || scans.size() != expected_scan_count ||
        detector_exclusions.size() !=
            detector_population.apt_excluded_detector_count ||
        detector_population.apt_eligible_detector_count +
                detector_population.apt_excluded_detector_count !=
            detector_population.detector_count) {
        throw std::logic_error(
            "bounded native cohort provenance is incomplete");
    }
    std::set<TimestreamDetectorColumn> apt_excluded;
    TimestreamDetectorColumn previous_detector = -1;
    for (const auto &exclusion : detector_exclusions) {
        const bool valid_flag_cause = exclusion.apt_flag.has_value()
            ? exclusion.cause == "apt_flag_nonzero" &&
                *exclusion.apt_flag != 0
            : exclusion.cause == "apt_flag_typed_null";
        if (exclusion.detector_column < 0 ||
            static_cast<std::size_t>(exclusion.detector_column) >=
                detector_population.detector_count ||
            exclusion.detector_column <= previous_detector ||
            exclusion.output_uid < 0 || exclusion.network < 0 ||
            exclusion.channel < 0 ||
            exclusion.authority !=
                "canonical_apt_v2.matched_detector_relation" ||
            !valid_flag_cause) {
            throw std::logic_error(
                "bounded native detector exclusion is incomplete");
        }
        previous_detector = exclusion.detector_column;
        apt_excluded.insert(exclusion.detector_column);
    }
    for (std::size_t scan = 0; scan < scans.size(); ++scan) {
        scans[scan].validate(binding, detector_population,
                             expected_scan_count);
        if (scans[scan].operation.scan_index !=
            static_cast<std::int64_t>(scan)) {
            throw std::logic_error(
                "native cohort scan publication order is nondeterministic");
        }
        if (scans[scan].map_occurrence.mapmaking_enabled) {
            std::set<TimestreamDetectorColumn> named_zero_weights{
                apt_excluded};
            for (const auto &cause : scans[scan].scoped_causes) {
                if (cause.authority ==
                        "citlali.native_duplicate_tone_policy_v2" ||
                    cause.authority ==
                        "citlali.weighting.detector_weight_contract_v1" ||
                    cause.authority ==
                        "citlali.learning.native_map_application_v1") {
                    named_zero_weights.insert(
                        cause.detector_columns.begin(),
                        cause.detector_columns.end());
                }
            }
            const std::set<TimestreamDetectorColumn> realized_zero_weights{
                scans[scan].map_occurrence
                    .zero_weight_detector_columns.begin(),
                scans[scan].map_occurrence
                    .zero_weight_detector_columns.end()};
            if (named_zero_weights != realized_zero_weights) {
                throw std::logic_error(
                    "bounded native zero-weight authorities do not reconcile");
            }
        }
    }
}

struct NativeCohortMapPublicationRequestV3 {
    bool mapmaking_enabled = false;
    std::string method;
    std::string product_occurrence;
    std::string product_identity_digest;
    std::string eligible_weight_digest;
    std::size_t positive_weight_detector_count = 0;
    std::size_t zero_weight_detector_count = 0;
    std::vector<TimestreamDetectorColumn> zero_weight_detector_columns;
    std::optional<std::string> jinc_processing_configuration_digest;
    std::vector<TimestreamDetectorColumn>
        learned_map_zero_weight_detector_columns;
    NativeNoiseAssignmentSummaryV3 noise_assignment;
    NativeFruitLoopFeedbackSummaryV3 fruit_loop_feedback;
};

NativeCohortScanProvenanceV3 make_native_cohort_scan_provenance_v3(
    const NativeCohortObservationBindingV2 &binding,
    const NativeMeasuredDetectorLedger &ledger,
    const NativeRtcDispatchResult &rtc,
    const NativePtcPreparedOperation &prepared,
    const NativeScienceProjection &projection,
    const NativePtcExclusionMatrix &ptc_preclean_flags,
    const NativePtcExclusionMatrix &ptc_runtime_flags,
    const NativePtcExclusionMatrix &final_flags,
    NativeCohortMapPublicationRequestV3 map_request);

inline std::string native_cohort_map_index_digest_v3(
    const NativeScienceProjection &projection) {
    NativeCohortDigestBuilderV2 digest;
    digest.add("schema", "citlali-native-map-index-v3");
    for (const auto &detector : projection.detectors()) {
        digest.add_integer("detector", detector.detector_column);
        digest.add_integer("map-index", detector.map_index);
    }
    return digest.finish();
}

inline NativeCohortScanProvenanceV3
make_native_cohort_scan_provenance_v3(
    const NativeCohortObservationBindingV2 &binding,
    const NativeMeasuredDetectorLedger &ledger,
    const NativeRtcDispatchResult &rtc,
    const NativePtcPreparedOperation &prepared,
    const NativeScienceProjection &projection,
    const NativePtcExclusionMatrix &ptc_preclean_flags,
    const NativePtcExclusionMatrix &ptc_runtime_flags,
    const NativePtcExclusionMatrix &final_flags,
    NativeCohortMapPublicationRequestV3 map_request) {
    const auto mapping = ledger.mapping_handle();
    if (!mapping || prepared.mapping_handle().get() != mapping.get() ||
        !ledger.last_operation() || !ledger.last_committed_operation() ||
        !(*ledger.last_operation() == prepared.operation()) ||
        !(*ledger.last_committed_operation() == prepared.operation()) ||
        !(projection.operation() == prepared.operation()) ||
        !(projection.scope() == mapping->scope()) ||
        ptc_preclean_flags.rows() != projection.row_count() ||
        ptc_preclean_flags.cols() != projection.detector_count() ||
        ptc_runtime_flags.rows() != projection.row_count() ||
        ptc_runtime_flags.cols() != projection.detector_count() ||
        final_flags.rows() != projection.row_count() ||
        final_flags.cols() != projection.detector_count() ||
        (projection.flags().array() &&
         !ptc_preclean_flags.array()).any() ||
        (ptc_preclean_flags.array() &&
         !ptc_runtime_flags.array()).any() ||
        (ptc_runtime_flags.array() && !final_flags.array()).any() ||
        rtc.runs.empty() ||
        prepared.groups().empty() ||
        binding.detector_relation_digest !=
            native_cohort_detector_relation_digest_v2(
                *mapping->relation_handle()) ||
        binding.raw_manifest_digest !=
            native_cohort_raw_manifest_digest_v2(
                *mapping->relation_handle()) ||
        binding.alignment_plan_digest !=
            native_cohort_alignment_plan_digest_v2(
                *mapping->carriers_handle()->alignment_handle()) ||
        binding.pointing_plan_digest !=
            native_cohort_pointing_plan_digest_v2(
                *mapping->carriers_handle()->pointing_handle())) {
        throw std::logic_error(
            "bounded native cohort provenance requires one exact committed operation");
    }

    NativeCohortScanProvenanceV3 result;
    result.observation_binding_digest = binding.digest();
    result.scope = mapping->scope();
    result.operation = prepared.operation();
    result.rtc.run_count = rtc.runs.size();
    result.rtc.output_row_count = rtc.output_row_count();
    result.rtc.interval_authority =
        "telescope.scan_indices.inner_and_outer_intervals";
    std::set<std::size_t> summarized_rtc_segments;
    for (const auto &run : rtc.runs) {
        if (summarized_rtc_segments.insert(
                run.input.segment_ordinal).second) {
            native_cohort_checked_add_v3(
                result.rtc.loaded_input_row_count,
                run.input.common_slots.size(),
                "RTC loaded input row");
            native_cohort_checked_add_v3(
                result.rtc.selected_input_row_count,
                run.input.selected_row_count(),
                "RTC selected input row");
        }
        for (const auto &support : run.support) {
            if (support.final_short_support) {
                ++result.rtc.final_short_support_count;
            }
            native_cohort_checked_add_v3(
                result.rtc.exact_support_identity_count,
                support.exact_native_support.size(),
                "RTC support identity");
            native_cohort_checked_add_v3(
                result.rtc.detector_support_count,
                support.detector_columns.size(),
                "RTC detector support");
            if (support.detector_columns.size() !=
                support.ored_flag_support.size()) {
                throw std::logic_error(
                    "bounded native RTC support has unequal detector flags");
            }
            for (std::size_t detector = 0;
                 detector < support.detector_columns.size(); ++detector) {
                if (support.ored_flag_support[detector] != 0) {
                    ++result.rtc.flagged_detector_support_count;
                }
            }
        }
    }
    result.ptc.requested_grouping = prepared.requested_grouping();
    result.ptc.effective_grouping = prepared.effective_grouping();
    result.ptc.segment_count = prepared.segment_count();
    result.ptc.group_count = prepared.groups().size();
    std::map<std::pair<std::string, NativeDetectorFlagBits>,
             NativeCohortScopedCauseV3> causes;
    for (const auto &group : prepared.groups()) {
        const bool clean = group.role() == NativePtcGroupRole::pca_clean;
        if (clean) {
            ++result.ptc.pca_clean_group_count;
        }
        else {
            ++result.ptc.pass_through_group_count;
        }
        native_cohort_checked_add_v3(
            result.ptc.detector_membership_count,
            group.detector_columns().size(),
            "PTC detector membership");
        for (Eigen::Index row = 0; row < group.slot_count(); ++row) {
            for (Eigen::Index local = 0;
                 local < group.detector_count(); ++local) {
                const auto &cell = group.cell(row, local);
                if (!cell.identity) {
                    throw std::logic_error(
                        "bounded native PTC summary lost measured identity");
                }
                const auto detector = group.detector_columns().at(
                    static_cast<std::size_t>(local));
                const auto current = ledger.record(
                    {cell.identity->key(), detector});
                if (!(current.identity == *cell.identity) ||
                    current.revision != cell.expected_revision + 1) {
                    throw std::logic_error(
                        "bounded native revision summary is stale");
                }
                const bool invalid = cell.state ==
                    CoincidenceCellState::mapped_invalid;
                if (invalid) {
                    ++result.population.mapped_invalid_sample_count;
                    ++result.population.preserved_pca_invalid_sample_count;
                }
                else {
                    ++result.population.mapped_valid_sample_count;
                    if (group.role() == NativePtcGroupRole::pass_through) {
                        ++result.population
                              .preserved_pass_through_sample_count;
                    }
                    else {
                        ++result.population.replaced_by_pca_sample_count;
                    }
                }
                if (cell.delivered_flag_bits != 0) {
                    ++result.population.delivered_flagged_sample_count;
                    const auto raw_bits = cell.delivered_flag_bits &
                        ~(native_cohort_rtc_processing_flag_bit_v3 |
                          native_cohort_learned_rtc_exclusion_bit_v3);
                    if (raw_bits != 0) {
                        ++result.population.raw_input_flagged_sample_count;
                        auto &cause = causes[{
                            "raw_input_flag_bits", raw_bits}];
                        cause.scope = "scan_summary";
                        cause.authority =
                            "raw_kids_input.sample_flags";
                        cause.cause = "raw_input_flag_bits";
                        cause.count_unit = "detector_samples";
                        cause.flag_bits = raw_bits;
                        ++cause.affected_count;
                        cause.detector_columns.push_back(detector);
                    }
                    if ((cell.delivered_flag_bits &
                         native_cohort_rtc_processing_flag_bit_v3) != 0) {
                        ++result.population
                              .rtc_processing_flagged_sample_count;
                        auto &cause = causes[{
                            "rtc_processing_generated_flag",
                            native_cohort_rtc_processing_flag_bit_v3}];
                        cause.scope = "scan_summary";
                        cause.authority =
                            "citlali.native_rtc_processing_policy_v2";
                        cause.cause = "rtc_processing_generated_flag";
                        cause.count_unit = "detector_samples";
                        cause.flag_bits =
                            native_cohort_rtc_processing_flag_bit_v3;
                        ++cause.affected_count;
                        cause.detector_columns.push_back(detector);
                    }
                    if ((cell.delivered_flag_bits &
                         native_cohort_learned_rtc_exclusion_bit_v3) != 0) {
                        ++result.population
                              .learned_rtc_excluded_sample_count;
                        auto &cause = causes[{
                            "learned_rtc_sample_mask",
                            native_cohort_learned_rtc_exclusion_bit_v3}];
                        cause.scope = "scan_summary";
                        cause.authority =
                            "citlali.learning.native_rtc_application_v1";
                        cause.cause = "learned_rtc_sample_mask";
                        cause.count_unit = "detector_samples";
                        cause.flag_bits =
                            native_cohort_learned_rtc_exclusion_bit_v3;
                        ++cause.affected_count;
                        cause.detector_columns.push_back(detector);
                    }
                }
                if (cell.operation_exclusion_bits != 0) {
                    ++result.population.operation_excluded_sample_count;
                    if (cell.operation_exclusion_bits !=
                        native_cohort_duplicate_tone_exclusion_bit_v3) {
                        throw std::logic_error(
                            "native operation exclusion lacks a named bounded cause");
                    }
                    auto &cause = causes[{
                        "duplicate_tone",
                        cell.operation_exclusion_bits}];
                    cause.scope = "scan_summary";
                    cause.authority =
                        "citlali.native_duplicate_tone_policy_v2";
                    cause.cause = "duplicate_tone";
                    cause.count_unit = "detector_samples";
                    cause.flag_bits = cell.operation_exclusion_bits;
                    ++cause.affected_count;
                    cause.detector_columns.push_back(detector);
                }
                if (!cell.apt_flag.has_value() || *cell.apt_flag != 0) {
                    ++result.population.apt_excluded_sample_count;
                }
            }
        }
    }
    for (auto &[identity, cause] : causes) {
        (void)identity;
        std::sort(cause.detector_columns.begin(),
                  cause.detector_columns.end());
        cause.detector_columns.erase(
            std::unique(cause.detector_columns.begin(),
                        cause.detector_columns.end()),
            cause.detector_columns.end());
        result.scoped_causes.push_back(std::move(cause));
    }

    result.population.row_count =
        static_cast<std::size_t>(projection.row_count());
    result.population.detector_count =
        static_cast<std::size_t>(projection.detector_count());
    result.population.detector_sample_count =
        native_cohort_checked_product_v3(
            result.population.row_count,
            result.population.detector_count,
            "native detector sample");
    const NativePtcExclusionMatrix learned_ptc_added =
        (ptc_preclean_flags.array() && !projection.flags().array()).matrix();
    const NativePtcExclusionMatrix second_pass_added =
        (ptc_runtime_flags.array() && !ptc_preclean_flags.array()).matrix();
    const NativePtcExclusionMatrix postclean_outlier_added =
        (final_flags.array() && !ptc_runtime_flags.array()).matrix();
    result.population.learned_ptc_excluded_sample_count =
        static_cast<std::size_t>(learned_ptc_added.array().count());
    result.population.ptc_second_pass_excluded_sample_count =
        static_cast<std::size_t>(second_pass_added.array().count());
    result.population.postclean_outlier_excluded_sample_count =
        static_cast<std::size_t>(postclean_outlier_added.array().count());
    result.population.final_excluded_sample_count =
        static_cast<std::size_t>(final_flags.array().count());

    const auto append_interval_causes =
        [&](const NativePtcExclusionMatrix &added,
            std::string_view authority, std::string_view cause) {
            for (Eigen::Index detector = 0;
                 detector < added.cols(); ++detector) {
                Eigen::Index row = 0;
                while (row < added.rows()) {
                    if (!added(row, detector)) {
                        ++row;
                        continue;
                    }
                    const auto start = row;
                    while (row + 1 < added.rows() &&
                           added(row + 1, detector)) {
                        ++row;
                    }
                    const auto end = row;
                    NativeCohortScopedCauseV3 interval;
                    interval.scope = "scan_detector_interval";
                    interval.authority = std::string{authority};
                    interval.cause = std::string{cause};
                    interval.count_unit = "detector_samples";
                    interval.start_row =
                        static_cast<std::size_t>(start);
                    interval.end_row = static_cast<std::size_t>(end);
                    interval.affected_count =
                        static_cast<std::size_t>(end - start + 1);
                    interval.detector_columns.push_back(detector);
                    result.scoped_causes.push_back(std::move(interval));
                    ++row;
                }
            }
        };
    append_interval_causes(
        learned_ptc_added,
        "citlali.learning.native_ptc_application_v1",
        "learned_ptc_sample_or_detector_mask");
    append_interval_causes(
        second_pass_added, "citlali.ptc.second_pass_local_v1",
        "accepted_second_pass_event");
    append_interval_causes(
        postclean_outlier_added,
        "citlali.ptc.postclean_outlier_policy_v1",
        "postclean_detector_outlier");

    map_request.noise_assignment.validate(mapping->detector_count());
    map_request.fruit_loop_feedback.validate();
    if (map_request.mapmaking_enabled) {
        if ((map_request.method != "naive" &&
             map_request.method != "jinc") ||
            map_request.product_occurrence.empty() ||
            map_request.product_identity_digest.empty() ||
            map_request.eligible_weight_digest.empty() ||
            map_request.positive_weight_detector_count +
                    map_request.zero_weight_detector_count !=
                result.population.detector_count ||
            map_request.zero_weight_detector_columns.size() !=
                map_request.zero_weight_detector_count ||
            !std::is_sorted(
                map_request.zero_weight_detector_columns.begin(),
                map_request.zero_weight_detector_columns.end()) ||
            std::adjacent_find(
                map_request.zero_weight_detector_columns.begin(),
                map_request.zero_weight_detector_columns.end()) !=
                map_request.zero_weight_detector_columns.end() ||
            !std::is_sorted(
                map_request.learned_map_zero_weight_detector_columns.begin(),
                map_request.learned_map_zero_weight_detector_columns.end()) ||
            std::adjacent_find(
                map_request.learned_map_zero_weight_detector_columns.begin(),
                map_request.learned_map_zero_weight_detector_columns.end()) !=
                map_request.learned_map_zero_weight_detector_columns.end()) {
            throw std::invalid_argument(
                "bounded native map publication request is incomplete");
        }
        result.map_occurrence.mapmaking_enabled = true;
        result.map_occurrence.method = std::move(map_request.method);
        result.map_occurrence.eligible_weight_digest =
            std::move(map_request.eligible_weight_digest);
        result.map_occurrence.product_occurrence =
            std::move(map_request.product_occurrence);
        result.map_occurrence.product_identity_digest =
            std::move(map_request.product_identity_digest);
        result.map_occurrence.jinc_processing_configuration_digest =
            std::move(
                map_request.jinc_processing_configuration_digest);
        result.map_occurrence.map_index_count =
            projection.detectors().size();
        result.map_occurrence.map_index_digest =
            native_cohort_map_index_digest_v3(projection);
        result.population.positive_weight_detector_count =
            map_request.positive_weight_detector_count;
        result.population.zero_weight_detector_count =
            map_request.zero_weight_detector_count;
        result.map_occurrence.zero_weight_detector_columns =
            std::move(map_request.zero_weight_detector_columns);
        result.map_occurrence.noise_assignment =
            std::move(map_request.noise_assignment);
        result.map_occurrence.fruit_loop_feedback =
            std::move(map_request.fruit_loop_feedback);
        const std::set<TimestreamDetectorColumn> zero_weight_detectors{
            result.map_occurrence.zero_weight_detector_columns.begin(),
            result.map_occurrence.zero_weight_detector_columns.end()};
        NativeCohortScopedCauseV3 learned_map_cause;
        learned_map_cause.scope = "scan_detector";
        learned_map_cause.authority =
            "citlali.learning.native_map_application_v1";
        learned_map_cause.cause =
            "learned_pre_map_detector_exclusion";
        learned_map_cause.count_unit = "detectors";
        learned_map_cause.detector_columns =
            std::move(
                map_request.learned_map_zero_weight_detector_columns);
        learned_map_cause.affected_count =
            learned_map_cause.detector_columns.size();
        for (const auto detector :
             learned_map_cause.detector_columns) {
            if (!zero_weight_detectors.contains(detector)) {
                throw std::logic_error(
                    "learned map exclusion retains a positive detector weight");
            }
        }
        if (learned_map_cause.affected_count != 0) {
            result.scoped_causes.push_back(
                std::move(learned_map_cause));
        }
        std::set<TimestreamDetectorColumn> explained_zero_weights;
        for (const auto &detector : mapping->relation_handle()->bindings()) {
            if (!detector.flag.has_value() || *detector.flag != 0) {
                explained_zero_weights.insert(
                    static_cast<TimestreamDetectorColumn>(
                        detector.detector_column));
            }
        }
        for (const auto &cause : result.scoped_causes) {
            if (cause.authority ==
                    "citlali.native_duplicate_tone_policy_v2" ||
                cause.authority ==
                    "citlali.learning.native_map_application_v1") {
                explained_zero_weights.insert(
                    cause.detector_columns.begin(),
                    cause.detector_columns.end());
            }
        }
        NativeCohortScopedCauseV3 weight_cause;
        weight_cause.scope = "scan_detector";
        weight_cause.authority =
            "citlali.weighting.detector_weight_contract_v1";
        weight_cause.cause = "nonpositive_final_detector_weight";
        weight_cause.count_unit = "detectors";
        for (const auto detector : zero_weight_detectors) {
            if (!explained_zero_weights.contains(detector)) {
                weight_cause.detector_columns.push_back(detector);
            }
        }
        weight_cause.affected_count =
            weight_cause.detector_columns.size();
        if (weight_cause.affected_count != 0) {
            result.scoped_causes.push_back(std::move(weight_cause));
        }
        for (const auto &detector : projection.detectors()) {
            if (detector.detector_column < 0 || detector.map_index < 0) {
                throw std::logic_error(
                    "bounded native map occurrence has invalid detector identity");
            }
        }
        for (Eigen::Index row = 0; row < projection.row_count(); ++row) {
            for (Eigen::Index detector = 0;
                 detector < projection.detector_count(); ++detector) {
                if (!final_flags(row, detector) &&
                    !zero_weight_detectors.contains(detector)) {
                    ++result.population.eligible_map_input_sample_count;
                }
            }
        }
    }
    else {
        if (!map_request.method.empty() ||
            !map_request.product_occurrence.empty() ||
            !map_request.product_identity_digest.empty() ||
            !map_request.eligible_weight_digest.empty() ||
            map_request.positive_weight_detector_count != 0 ||
            map_request.zero_weight_detector_count != 0 ||
            !map_request.zero_weight_detector_columns.empty() ||
            map_request.jinc_processing_configuration_digest ||
            !map_request.learned_map_zero_weight_detector_columns.empty() ||
            map_request.noise_assignment.enabled ||
            map_request.fruit_loop_feedback.enabled) {
            throw std::invalid_argument(
                "disabled native map publication request carries identity");
        }
    }

    const auto [detector_population, detector_exclusions] =
        native_cohort_detector_population_v3(
            *mapping->relation_handle());
    (void)detector_exclusions;
    std::sort(
        result.scoped_causes.begin(), result.scoped_causes.end(),
        [](const auto &lhs, const auto &rhs) {
            return std::tie(lhs.authority, lhs.cause, lhs.count_unit,
                            lhs.flag_bits, lhs.start_row, lhs.end_row,
                            lhs.detector_columns) <
                std::tie(rhs.authority, rhs.cause, rhs.count_unit,
                         rhs.flag_bits, rhs.start_row, rhs.end_row,
                         rhs.detector_columns);
        });
    result.validate(
        binding, detector_population,
        static_cast<std::size_t>(result.operation.scan_index + 1));
    return result;
}

class NativeCohortObservationLineageV3
    : public std::enable_shared_from_this<
          NativeCohortObservationLineageV3> {
public:
    enum class SlotPhase : std::uint8_t { empty, pending, committed };

    class Reservation {
    public:
        Reservation() = default;
        Reservation(const Reservation &) = delete;
        Reservation &operator=(const Reservation &) = delete;
        Reservation(Reservation &&other) noexcept
            : owner_{std::move(other.owner_)}, scan_{other.scan_},
              active_{other.active_} {
            other.active_ = false;
        }
        Reservation &operator=(Reservation &&other) noexcept {
            if (this != &other) {
                rollback();
                owner_ = std::move(other.owner_);
                scan_ = other.scan_;
                active_ = other.active_;
                other.active_ = false;
            }
            return *this;
        }
        ~Reservation() { rollback(); }

        void commit() noexcept {
            if (!active_ || !owner_) return;
            owner_->slots_[scan_]->phase.store(
                SlotPhase::committed, std::memory_order_release);
            active_ = false;
        }
        void rollback() noexcept {
            if (!active_ || !owner_) return;
            auto &slot = *owner_->slots_[scan_];
            slot.record.reset();
            slot.phase.store(
                SlotPhase::empty, std::memory_order_release);
            active_ = false;
        }

    private:
        friend class NativeCohortObservationLineageV3;
        Reservation(
            std::shared_ptr<NativeCohortObservationLineageV3> owner,
            std::size_t scan) noexcept
            : owner_{std::move(owner)}, scan_{scan}, active_{true} {}
        std::shared_ptr<NativeCohortObservationLineageV3> owner_;
        std::size_t scan_ = 0;
        bool active_ = false;
    };

    static std::shared_ptr<NativeCohortObservationLineageV3> create(
        NativeCohortObservationBindingV2 binding,
        const CanonicalAptDetectorRelationV2 &relation,
        std::size_t scan_count) {
        binding.validate();
        if (scan_count == 0 ||
            binding.detector_relation_digest !=
                native_cohort_detector_relation_digest_v2(relation)) {
            throw std::invalid_argument(
                "bounded native cohort lineage has invalid observation scope");
        }
        auto [population, exclusions] =
            native_cohort_detector_population_v3(relation);
        return std::shared_ptr<NativeCohortObservationLineageV3>(
            new NativeCohortObservationLineageV3{
                std::move(binding), std::move(population),
                std::move(exclusions), scan_count});
    }

    const NativeCohortObservationBindingV2 &binding() const noexcept {
        return binding_;
    }
    std::size_t scan_count() const noexcept { return slots_.size(); }

    Reservation reserve(NativeCohortScanProvenanceV3 record) {
        record.validate(binding_, detector_population_, scan_count());
        const auto scan = static_cast<std::size_t>(
            record.operation.scan_index);
        auto &slot = *slots_.at(scan);
        SlotPhase expected = SlotPhase::empty;
        if (!slot.phase.compare_exchange_strong(
                expected, SlotPhase::pending,
                std::memory_order_acq_rel)) {
            throw std::logic_error(
                "bounded native cohort scan is stale, duplicate, or pending");
        }
        try {
            slot.record.emplace(std::move(record));
        }
        catch (...) {
            slot.record.reset();
            slot.phase.store(
                SlotPhase::empty, std::memory_order_release);
            throw;
        }
        return Reservation{shared_from_this(), scan};
    }

    NativeCohortProductProvenanceV3 snapshot_complete() const {
        NativeCohortProductProvenanceV3 result{
            binding_, detector_population_, detector_exclusions_, {}};
        result.scans.reserve(slots_.size());
        for (const auto &slot_ptr : slots_) {
            const auto &slot = *slot_ptr;
            if (slot.phase.load(std::memory_order_acquire) !=
                    SlotPhase::committed ||
                !slot.record) {
                throw std::logic_error(
                    "bounded native cohort observation is incomplete");
            }
            result.scans.push_back(*slot.record);
        }
        result.validate_complete(scan_count());
        return result;
    }

private:
    struct Slot {
        std::atomic<SlotPhase> phase{SlotPhase::empty};
        std::optional<NativeCohortScanProvenanceV3> record;
    };

    NativeCohortObservationLineageV3(
        NativeCohortObservationBindingV2 binding,
        NativeCohortDetectorPopulationV3 detector_population,
        std::vector<NativeCohortDetectorExclusionV3> detector_exclusions,
        std::size_t scan_count)
        : binding_{std::move(binding)},
          detector_population_{std::move(detector_population)},
          detector_exclusions_{std::move(detector_exclusions)} {
        slots_.reserve(scan_count);
        for (std::size_t scan = 0; scan < scan_count; ++scan) {
            slots_.push_back(std::make_unique<Slot>());
        }
    }

    NativeCohortObservationBindingV2 binding_;
    NativeCohortDetectorPopulationV3 detector_population_;
    std::vector<NativeCohortDetectorExclusionV3> detector_exclusions_;
    std::vector<std::unique_ptr<Slot>> slots_;
};

}  // namespace citlali::pipeline
