#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <citlali/core/config/runtime_config.h>
#include <citlali/core/config/timestream_enums.h>

struct ReductionLearningState {
    enum class IterationPhase {
        Inactive,
        Learn,
        LearnWithModel,
        Apply
    };

    struct Options {
        bool enabled = false;
        bool diagnostics_enabled = true;
        int learn_iters = 2;
        int apply_start_iter = 2;
        int max_records_per_type = 200000;
        bool apply_sample_masks_enabled = true;
        double apply_max_new_flagged_fraction = 0.02;
        bool map_pixel_outlier_diagnostics_enabled = true;
        bool map_pixel_outlier_contributor_diagnostics_enabled = false;
        bool map_pixel_outlier_targeted_contributor_diagnostics_enabled = false;
        bool map_pixel_outlier_detector_exclusion_enabled = false;
        bool map_pixel_outlier_detector_exclusion_feedback_bypass_enabled = false;
        citlali::config::MapPixelOutlierDetectorExclusionApplication
            map_pixel_outlier_detector_exclusion_application =
                citlali::config::MapPixelOutlierDetectorExclusionApplication::
                    pre_cleaning;
        int map_pixel_outlier_top_n = 8;
        int map_pixel_outlier_targeted_contributor_max_pixels = 32;
        int map_pixel_outlier_detector_exclusion_min_pixels = 4;
        double map_pixel_outlier_min_abs_z = 8.0;
        double map_pixel_outlier_min_n_eff = 4.0;
        double map_pixel_outlier_source_radius_arcsec = 30.0;
        bool busy_detector_exclusion_enabled = true;
        bool scan_network_pathology_enabled = true;
        bool scan_network_pathology_apply_pre_rtc = false;
        bool scan_network_pathology_apply_pre_ptc = false;
        bool scan_network_pathology_apply_pre_mapmaking = true;
        int scan_network_pathology_min_candidate_clusters = 4;
        int scan_network_pathology_min_candidate_events = 100;
        double scan_network_pathology_min_max_residual_z = 25.0;
        int scan_network_pathology_severe_candidate_events = 250;
        double scan_network_pathology_severe_max_residual_z = 50.0;
        double scan_network_pathology_max_new_flagged_fraction = 0.35;
    };

    struct LearnedSampleMask {
        std::string obsnum;
        std::string producer;
        std::string reason;
        int iter = -1;
        int scan = -1;
        int uid = -1;
        int nw = -1;
        int array = -1;
        long long raw_start = -1;
        long long raw_stop = -1;
        long long ptc_start = -1;
        long long ptc_stop = -1;
        double score = nan_value();
        double z = nan_value();
        double value = nan_value();
        double confidence = nan_value();
        double source_distance_arcsec = nan_value();
        bool source_protected = false;
        bool apply_pre_rtc = false;
    };

    struct EffectiveSampleMask {
        int iter = -1;
        int uid = -1;
        long long start = -1;
        long long stop = -1;
    };

    struct DetectorPenalty {
        std::string obsnum;
        std::string producer;
        std::string reason;
        int iter = -1;
        int scan = -1;
        int uid = -1;
        int nw = -1;
        int array = -1;
        double factor = 1.0;
        double score = nan_value();
        // Midpoint of the processed time chunk in Unix seconds.  This is
        // diagnostic provenance only; it is not part of the learned-state
        // identity or application policy.
        double event_time_unix_sec = nan_value();
        bool scan_local = false;
    };

    struct HighWeightDetector {
        std::string obsnum;
        std::string grouping;
        std::string reason;
        int iter = -1;
        int scan = -1;
        int uid = -1;
        int nw = -1;
        int array = -1;
        double weight = nan_value();
        double final_weight = nan_value();
        double group_median = nan_value();
        double robust_z = nan_value();
        double cap = nan_value();
        double validation_factor = nan_value();
        bool cap_recommended = false;
        bool cap_applied = false;
        bool validated = false;
    };

    struct MapPixelOutlier {
        std::string obsnum;
        std::string producer;
        std::string reason;
        int iter = -1;
        int map_index = -1;
        int scan = -1;
        int uid = -1;
        int row = -1;
        int col = -1;
        long long sample = -1;
        double value = nan_value();
        double weight = nan_value();
        double n_eff = nan_value();
        double leave_one_out_z = nan_value();
        double source_distance_arcsec = nan_value();
        bool source_protected = false;
    };

    struct MapPixelTarget {
        int map_index = -1;
        int row = -1;
        int col = -1;
    };

    struct ResolvedMapPixelTargetSet {
        std::string obsnum;
        std::string producer;
        int source_iter = -1;
        int apply_iter = -1;
        int map_count = -1;
        int n_rows = -1;
        int n_cols = -1;
        std::vector<MapPixelTarget> targets;
    };

    struct SourceProtectionSummary {
        std::string obsnum;
        std::string producer;
        std::string mode;
        int iter = -1;
        int scan = -1;
        int map_index = -1;
        int protected_samples = 0;
        int total_samples = 0;
        double radius_arcsec = nan_value();
        double support_npix = nan_value();
    };

    struct BusyNetworkSummary {
        std::string obsnum;
        std::string producer;
        std::string reason;
        int iter = -1;
        int scan = -1;
        int nw = -1;
        int n_candidate_clusters = 0;
        int n_candidate_events = 0;
        int n_accepted_clusters = 0;
        int n_accepted_events = 0;
        int n_rejected_clusters = 0;
        int n_rejected_events = 0;
        int n_source_protected_clusters = 0;
        int n_source_protected_events = 0;
        int max_unflagged_residual_uid = -1;
        int top_candidate_sample = -1;
        double top_candidate_score = nan_value();
        double max_unflagged_residual_z = nan_value();
        bool busy_vetoed = false;
        bool selective_acceptance_recommended = false;
    };

    struct LearnedMaskApplicationSummary {
        std::string obsnum;
        std::string producer;
        std::string stage;
        int iter = -1;
        int scan = -1;
        int candidate_records = 0;
        int matched_records = 0;
        int invalid_records = 0;
        int proposed_samples = 0;
        int newly_flagged_samples = 0;
        int already_flagged_samples = 0;
        int source_protected_samples = 0;
        double newly_flagged_fraction = nan_value();
        double max_new_flagged_fraction = nan_value();
        bool applied = false;
    };

    Options options;
    int current_iter = -1;
    IterationPhase current_phase = IterationPhase::Inactive;
    std::string current_reduction_type = "unknown";
    bool current_source_model_available = false;
    int begin_count = 0;
    int finalize_count = 0;

    // Operational learned state is intentionally separate from the bounded
    // diagnostic event history below.  The effective sample-mask state is an
    // interval union keyed by observation, scan, application stage, and UID;
    // detector penalties are reduced to one effective record per scientific
    // identity.  Neither collection is subject to the diagnostic record cap.
    using EffectiveSampleMaskKey =
        std::tuple<std::string, int, bool, int>;
    using EffectiveDetectorPenaltyKey =
        std::tuple<std::string, std::string, std::string, int, int, int, int,
                   bool>;
    using ResolvedMapPixelTargetKey =
        std::tuple<std::string, std::string>;
    std::map<EffectiveSampleMaskKey, std::vector<EffectiveSampleMask>>
        effective_sample_masks;
    std::map<EffectiveDetectorPenaltyKey, DetectorPenalty>
        effective_detector_penalties;
    // This is bounded operational apply state for one next iteration, not
    // diagnostic event history.  One record is kept per observation and
    // producer/stage, and each target list is capped by the effective policy.
    std::map<ResolvedMapPixelTargetKey, ResolvedMapPixelTargetSet>
        resolved_map_pixel_target_sets;
    // Transient evidence for resolving the next boundary state. This is
    // cleared at every iteration start and never checkpointed. Keeping it
    // separate makes diagnostic-history retention and its record cap
    // non-causal without changing the target ranking rule.
    std::map<ResolvedMapPixelTargetKey, std::vector<MapPixelOutlier>>
        current_map_pixel_target_candidates;

    // These vectors are diagnostic event history only.  max_records_per_type
    // bounds their output/memory cost but never truncates effective state.
    std::vector<LearnedSampleMask> learned_sample_mask_events;
    std::vector<DetectorPenalty> detector_penalty_events;
    std::vector<HighWeightDetector> high_weight_detectors;
    std::vector<MapPixelOutlier> map_pixel_outliers;
    std::vector<SourceProtectionSummary> source_protection_summaries;
    std::vector<BusyNetworkSummary> busy_network_summaries;
    std::vector<LearnedMaskApplicationSummary> learned_mask_applications;

    std::size_t dropped_learned_sample_masks = 0;
    std::size_t dropped_detector_penalties = 0;
    std::size_t dropped_high_weight_detectors = 0;
    std::size_t dropped_map_pixel_outliers = 0;
    std::size_t dropped_source_protection_summaries = 0;
    std::size_t dropped_busy_network_summaries = 0;
    std::size_t dropped_learned_mask_applications = 0;

    std::shared_ptr<std::mutex> mutex = std::make_shared<std::mutex>();

    static double nan_value() {
        return std::numeric_limits<double>::quiet_NaN();
    }

    static std::string phase_name(IterationPhase phase) {
        switch (phase) {
            case IterationPhase::Learn:
                return "learn";
            case IterationPhase::LearnWithModel:
                return "learn_with_model";
            case IterationPhase::Apply:
                return "apply";
            case IterationPhase::Inactive:
            default:
                return "inactive";
        }
    }

    void configure(Options new_options) {
        std::lock_guard<std::mutex> lock(*mutex);
        new_options.learn_iters = std::max(0, new_options.learn_iters);
        new_options.apply_start_iter = std::max(0, new_options.apply_start_iter);
        new_options.max_records_per_type = std::max(0, new_options.max_records_per_type);
        new_options.apply_max_new_flagged_fraction =
            std::max(0.0, new_options.apply_max_new_flagged_fraction);
        new_options.map_pixel_outlier_top_n =
            std::max(0, new_options.map_pixel_outlier_top_n);
        new_options.map_pixel_outlier_targeted_contributor_max_pixels =
            std::max(0, new_options.map_pixel_outlier_targeted_contributor_max_pixels);
        new_options.map_pixel_outlier_detector_exclusion_min_pixels =
            std::max(1, new_options.map_pixel_outlier_detector_exclusion_min_pixels);
        new_options.map_pixel_outlier_min_abs_z =
            std::max(0.0, new_options.map_pixel_outlier_min_abs_z);
        new_options.map_pixel_outlier_min_n_eff =
            std::max(0.0, new_options.map_pixel_outlier_min_n_eff);
        new_options.map_pixel_outlier_source_radius_arcsec =
            std::max(0.0, new_options.map_pixel_outlier_source_radius_arcsec);
        new_options.scan_network_pathology_min_candidate_clusters =
            std::max(0, new_options.scan_network_pathology_min_candidate_clusters);
        new_options.scan_network_pathology_min_candidate_events =
            std::max(0, new_options.scan_network_pathology_min_candidate_events);
        new_options.scan_network_pathology_min_max_residual_z =
            std::max(0.0, new_options.scan_network_pathology_min_max_residual_z);
        new_options.scan_network_pathology_severe_candidate_events =
            std::max(0, new_options.scan_network_pathology_severe_candidate_events);
        new_options.scan_network_pathology_severe_max_residual_z =
            std::max(0.0, new_options.scan_network_pathology_severe_max_residual_z);
        new_options.scan_network_pathology_max_new_flagged_fraction =
            std::max(0.0, new_options.scan_network_pathology_max_new_flagged_fraction);
        options = new_options;
        if (!map_pixel_target_state_required_unlocked()) {
            resolved_map_pixel_target_sets.clear();
            current_map_pixel_target_candidates.clear();
        }
        if (!options.enabled) {
            current_phase = IterationPhase::Inactive;
        }
    }

    IterationPhase phase_for_iteration(int iter, bool source_model_available) const {
        if (!options.enabled) {
            return IterationPhase::Inactive;
        }
        if (iter >= options.apply_start_iter) {
            return IterationPhase::Apply;
        }
        if (source_model_available && iter > 0) {
            return IterationPhase::LearnWithModel;
        }
        if (iter < options.learn_iters) {
            return IterationPhase::Learn;
        }
        return IterationPhase::Apply;
    }

    void begin_iteration(int iter, bool source_model_available,
                         const std::string &reduction_type) {
        std::lock_guard<std::mutex> lock(*mutex);
        current_iter = iter;
        current_source_model_available = source_model_available;
        current_reduction_type = reduction_type;
        current_phase = phase_for_iteration(iter, source_model_available);
        current_map_pixel_target_candidates.clear();
        begin_count++;
    }

    void begin_iteration(int iter, bool source_model_available,
                         citlali::config::ReductionType reduction_type) {
        begin_iteration(
            iter, source_model_available,
            std::string{citlali::config::to_string(reduction_type)});
    }

    void finalize_iteration(int iter) {
        std::lock_guard<std::mutex> lock(*mutex);
        current_iter = iter;
        finalize_count++;
    }

    bool is_enabled() const {
        return options.enabled;
    }

    bool diagnostics_enabled() const {
        return options.diagnostics_enabled;
    }

    bool learning_active() const {
        return current_phase == IterationPhase::Learn ||
               current_phase == IterationPhase::LearnWithModel;
    }

    bool apply_active() const {
        return current_phase == IterationPhase::Apply;
    }

    std::string current_phase_name() const {
        return phase_name(current_phase);
    }

    void clear_records() {
        std::lock_guard<std::mutex> lock(*mutex);
        effective_sample_masks.clear();
        effective_detector_penalties.clear();
        resolved_map_pixel_target_sets.clear();
        current_map_pixel_target_candidates.clear();
        learned_sample_mask_events.clear();
        detector_penalty_events.clear();
        high_weight_detectors.clear();
        map_pixel_outliers.clear();
        source_protection_summaries.clear();
        busy_network_summaries.clear();
        learned_mask_applications.clear();
        dropped_learned_sample_masks = 0;
        dropped_detector_penalties = 0;
        dropped_high_weight_detectors = 0;
        dropped_map_pixel_outliers = 0;
        dropped_source_protection_summaries = 0;
        dropped_busy_network_summaries = 0;
        dropped_learned_mask_applications = 0;
    }

    void record_learned_sample_mask(LearnedSampleMask record) {
        std::lock_guard<std::mutex> lock(*mutex);
        if (!options.enabled || !learning_active()) {
            return;
        }
        merge_effective_sample_mask(record);
        if (options.diagnostics_enabled) {
            push_with_cap(learned_sample_mask_events, std::move(record),
                          dropped_learned_sample_masks);
        }
    }

    void record_detector_penalty(DetectorPenalty record,
                                 bool allow_apply_phase = false) {
        std::lock_guard<std::mutex> lock(*mutex);
        if (!options.enabled ||
            (!learning_active() && !(allow_apply_phase && apply_active()))) {
            return;
        }
        merge_effective_detector_penalty(record);
        if (options.diagnostics_enabled) {
            push_with_cap(detector_penalty_events, std::move(record),
                          dropped_detector_penalties);
        }
    }

    void record_high_weight_detector(HighWeightDetector record) {
        std::lock_guard<std::mutex> lock(*mutex);
        if (!options.enabled || !options.diagnostics_enabled) {
            return;
        }
        push_with_cap(high_weight_detectors, std::move(record),
                      dropped_high_weight_detectors);
    }

    void record_map_pixel_outlier(MapPixelOutlier record) {
        std::lock_guard<std::mutex> lock(*mutex);
        if (!options.enabled || !options.diagnostics_enabled) {
            return;
        }
        if (map_pixel_target_state_required_unlocked() &&
            record.iter == current_iter) {
            current_map_pixel_target_candidates[
                ResolvedMapPixelTargetKey{record.obsnum, record.producer}]
                .push_back(record);
        }
        push_with_cap(map_pixel_outliers, std::move(record),
                      dropped_map_pixel_outliers);
    }

    void record_source_protection_summary(SourceProtectionSummary record) {
        std::lock_guard<std::mutex> lock(*mutex);
        if (!options.enabled || !options.diagnostics_enabled) {
            return;
        }
        push_with_cap(source_protection_summaries, std::move(record),
                      dropped_source_protection_summaries);
    }

    void record_busy_network_summary(BusyNetworkSummary record) {
        std::lock_guard<std::mutex> lock(*mutex);
        if (!options.enabled || !options.diagnostics_enabled) {
            return;
        }
        push_with_cap(busy_network_summaries, std::move(record),
                      dropped_busy_network_summaries);
    }

    void record_learned_mask_application(LearnedMaskApplicationSummary record) {
        std::lock_guard<std::mutex> lock(*mutex);
        if (!options.enabled || !options.diagnostics_enabled) {
            return;
        }
        push_with_cap(learned_mask_applications, std::move(record),
                      dropped_learned_mask_applications);
    }

    std::string summary_string() const {
        std::lock_guard<std::mutex> lock(*mutex);
        std::ostringstream os;
        std::size_t effective_sample_mask_interval_count = 0;
        for (const auto &[key, intervals] : effective_sample_masks) {
            (void) key;
            effective_sample_mask_interval_count += intervals.size();
        }
        os << "enabled=" << options.enabled
           << " diagnostics=" << options.diagnostics_enabled
           << " iter=" << current_iter
           << " phase=" << phase_name(current_phase)
           << " reduction_type=" << current_reduction_type
           << " source_model_available=" << current_source_model_available
           << " effective_sample_mask_intervals="
           << effective_sample_mask_interval_count
           << " effective_detector_penalties="
           << effective_detector_penalties.size()
           << " resolved_map_pixel_target_scopes="
           << resolved_map_pixel_target_sets.size()
           << " resolved_map_pixel_targets="
           << resolved_map_pixel_target_count_unlocked()
           << " sample_mask_events=" << learned_sample_mask_events.size()
           << " detector_penalty_events=" << detector_penalty_events.size()
           << " high_weight_detectors=" << high_weight_detectors.size()
           << " map_pixel_outliers=" << map_pixel_outliers.size()
           << " source_protection_summaries=" << source_protection_summaries.size()
           << " busy_network_summaries=" << busy_network_summaries.size()
           << " learned_mask_applications=" << learned_mask_applications.size()
           << " dropped_diagnostic_records="
           << (dropped_learned_sample_masks + dropped_detector_penalties +
               dropped_high_weight_detectors + dropped_map_pixel_outliers +
               dropped_source_protection_summaries +
               dropped_busy_network_summaries +
               dropped_learned_mask_applications);
        return os.str();
    }

    std::vector<EffectiveSampleMask> effective_sample_masks_for(
        const std::string &obsnum, int scan, bool apply_pre_rtc,
        int before_iter) const {
        std::lock_guard<std::mutex> lock(*mutex);
        std::vector<EffectiveSampleMask> result;
        const EffectiveSampleMaskKey first{
            obsnum, scan, apply_pre_rtc, std::numeric_limits<int>::min()};
        auto it = effective_sample_masks.lower_bound(first);
        for (; it != effective_sample_masks.end(); ++it) {
            const auto &[record_obsnum, record_scan, record_pre_rtc,
                         record_uid] = it->first;
            if (record_obsnum != obsnum || record_scan != scan ||
                record_pre_rtc != apply_pre_rtc) {
                break;
            }
            for (const auto &interval : it->second) {
                if (interval.iter >= 0 && interval.iter < before_iter) {
                    auto record = interval;
                    record.uid = record_uid;
                    result.push_back(record);
                }
            }
        }
        return result;
    }

    std::vector<DetectorPenalty> effective_detector_penalty_records() const {
        std::lock_guard<std::mutex> lock(*mutex);
        std::vector<DetectorPenalty> result;
        result.reserve(effective_detector_penalties.size());
        for (const auto &[key, record] : effective_detector_penalties) {
            (void) key;
            result.push_back(record);
        }
        return result;
    }

    std::size_t effective_sample_mask_interval_count() const {
        std::lock_guard<std::mutex> lock(*mutex);
        std::size_t count = 0;
        for (const auto &[key, intervals] : effective_sample_masks) {
            (void) key;
            count += intervals.size();
        }
        return count;
    }

    bool map_pixel_target_state_required() const {
        std::lock_guard<std::mutex> lock(*mutex);
        return map_pixel_target_state_required_unlocked();
    }

    void resolve_map_pixel_targets_for_next_iteration(
        const std::string &obsnum, const std::string &producer,
        int completed_iter, int map_count, int n_rows, int n_cols) {
        std::lock_guard<std::mutex> lock(*mutex);
        const ResolvedMapPixelTargetKey key{obsnum, producer};
        if (!map_pixel_target_state_required_unlocked()) {
            resolved_map_pixel_target_sets.erase(key);
            return;
        }
        if (obsnum.empty() || producer.empty() || completed_iter < 0 ||
            map_count < 0 || n_rows < 0 || n_cols < 0) {
            throw std::invalid_argument(
                "invalid map-pixel target resolution boundary");
        }

        const auto evidence = current_map_pixel_target_candidates.find(key);
        int source_iter = -1;
        if (evidence != current_map_pixel_target_candidates.end()) {
            for (const auto &record : evidence->second) {
                if (record.iter == completed_iter &&
                    record.map_index >= 0 && record.map_index < map_count &&
                    record.row >= 0 && record.row < n_rows && record.col >= 0 &&
                    record.col < n_cols) {
                    source_iter = completed_iter;
                    break;
                }
            }
        }

        if (source_iter < 0) {
            const auto previous = resolved_map_pixel_target_sets.find(key);
            if (previous != resolved_map_pixel_target_sets.end()) {
                auto &resolved = previous->second;
                const bool grid_matches =
                    resolved.targets.empty() ||
                    (resolved.map_count == map_count &&
                     resolved.n_rows == n_rows && resolved.n_cols == n_cols);
                if (!grid_matches) {
                    throw std::logic_error(
                        "map grid changed while carrying resolved map-pixel targets");
                }
                resolved.apply_iter = completed_iter + 1;
                if (resolved.targets.empty()) {
                    resolved.map_count = map_count;
                    resolved.n_rows = n_rows;
                    resolved.n_cols = n_cols;
                }
                return;
            }
            ResolvedMapPixelTargetSet empty;
            empty.obsnum = obsnum;
            empty.producer = producer;
            empty.apply_iter = completed_iter + 1;
            empty.map_count = map_count;
            empty.n_rows = n_rows;
            empty.n_cols = n_cols;
            resolved_map_pixel_target_sets.emplace(key, std::move(empty));
            return;
        }

        struct Candidate {
            MapPixelTarget target;
            double score = 0.0;
        };
        std::vector<Candidate> candidates;
        for (const auto &record : evidence->second) {
            if (record.iter == source_iter &&
                record.map_index >= 0 && record.map_index < map_count &&
                record.row >= 0 && record.row < n_rows && record.col >= 0 &&
                record.col < n_cols) {
                const double raw_score =
                    std::isfinite(record.leave_one_out_z)
                        ? std::abs(record.leave_one_out_z)
                        : std::abs(record.value);
                candidates.push_back(
                    {{record.map_index, record.row, record.col},
                     std::isfinite(raw_score) ? raw_score : 0.0});
            }
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto &left, const auto &right) {
                      return left.score > right.score;
                  });

        ResolvedMapPixelTargetSet resolved;
        resolved.obsnum = obsnum;
        resolved.producer = producer;
        resolved.source_iter = source_iter;
        resolved.apply_iter = completed_iter + 1;
        resolved.map_count = map_count;
        resolved.n_rows = n_rows;
        resolved.n_cols = n_cols;
        const auto max_targets = static_cast<std::size_t>(
            options.map_pixel_outlier_targeted_contributor_max_pixels);
        resolved.targets.reserve(
            std::min(candidates.size(), max_targets));
        for (const auto &candidate : candidates) {
            if (resolved.targets.size() >= max_targets) {
                break;
            }
            const bool duplicate = std::any_of(
                resolved.targets.begin(), resolved.targets.end(),
                [&](const auto &target) {
                    return target.map_index == candidate.target.map_index &&
                           target.row == candidate.target.row &&
                           target.col == candidate.target.col;
                });
            if (!duplicate) {
                resolved.targets.push_back(candidate.target);
            }
        }
        resolved_map_pixel_target_sets[key] = std::move(resolved);
    }

    void finalize_map_pixel_target_state(
        const std::vector<std::string> &observation_ids,
        const std::string &producer, int completed_iter) {
        std::lock_guard<std::mutex> lock(*mutex);
        if (!map_pixel_target_state_required_unlocked()) {
            resolved_map_pixel_target_sets.clear();
            return;
        }
        if (producer.empty() || completed_iter < 0 || observation_ids.empty()) {
            throw std::invalid_argument(
                "invalid map-pixel target finalization boundary");
        }

        for (auto it = resolved_map_pixel_target_sets.begin();
             it != resolved_map_pixel_target_sets.end();) {
            const auto &[record_obsnum, record_producer] = it->first;
            if (record_producer == producer &&
                std::find(observation_ids.begin(), observation_ids.end(),
                          record_obsnum) == observation_ids.end()) {
                it = resolved_map_pixel_target_sets.erase(it);
            }
            else {
                ++it;
            }
        }

        const int apply_iter = completed_iter + 1;
        for (const auto &obsnum : observation_ids) {
            const ResolvedMapPixelTargetKey key{obsnum, producer};
            const auto found = resolved_map_pixel_target_sets.find(key);
            if (found == resolved_map_pixel_target_sets.end()) {
                ResolvedMapPixelTargetSet empty;
                empty.obsnum = obsnum;
                empty.producer = producer;
                empty.apply_iter = apply_iter;
                resolved_map_pixel_target_sets.emplace(key, std::move(empty));
                continue;
            }
            auto &resolved = found->second;
            if (resolved.source_iter > completed_iter ||
                resolved.apply_iter > apply_iter) {
                throw std::logic_error(
                    "map-pixel target state is ahead of the completed iteration");
            }
            if (resolved.apply_iter != apply_iter) {
                // No new usable outlier evidence was produced for this scope.
                // Preserve the same resolved target membership, as the
                // historical in-process resolver preserved the latest usable
                // source iteration.
                resolved.apply_iter = apply_iter;
            }
        }
    }

    std::optional<ResolvedMapPixelTargetSet>
    resolved_map_pixel_targets_for(const std::string &obsnum,
                                   const std::string &producer,
                                   int apply_iter) const {
        std::lock_guard<std::mutex> lock(*mutex);
        const auto found = resolved_map_pixel_target_sets.find(
            ResolvedMapPixelTargetKey{obsnum, producer});
        if (found == resolved_map_pixel_target_sets.end() ||
            found->second.apply_iter != apply_iter) {
            return std::nullopt;
        }
        return found->second;
    }

    std::vector<ResolvedMapPixelTargetSet>
    resolved_map_pixel_target_records() const {
        std::lock_guard<std::mutex> lock(*mutex);
        std::vector<ResolvedMapPixelTargetSet> result;
        result.reserve(resolved_map_pixel_target_sets.size());
        for (const auto &[key, record] : resolved_map_pixel_target_sets) {
            (void) key;
            result.push_back(record);
        }
        return result;
    }

    std::size_t resolved_map_pixel_target_count() const {
        std::lock_guard<std::mutex> lock(*mutex);
        return resolved_map_pixel_target_count_unlocked();
    }

private:
    bool map_pixel_target_state_required_unlocked() const {
        const bool full_contribution_diagnostics =
            options.enabled && options.diagnostics_enabled &&
            options.map_pixel_outlier_diagnostics_enabled &&
            options.map_pixel_outlier_contributor_diagnostics_enabled;
        return options.enabled && options.diagnostics_enabled &&
               options.map_pixel_outlier_diagnostics_enabled &&
               options.map_pixel_outlier_targeted_contributor_diagnostics_enabled &&
               options.map_pixel_outlier_targeted_contributor_max_pixels > 0 &&
               !full_contribution_diagnostics;
    }

    std::size_t resolved_map_pixel_target_count_unlocked() const {
        std::size_t count = 0;
        for (const auto &[key, record] : resolved_map_pixel_target_sets) {
            (void) key;
            count += record.targets.size();
        }
        return count;
    }

    void merge_effective_sample_mask(const LearnedSampleMask &record) {
        const long long start =
            record.apply_pre_rtc ? record.raw_start : record.ptc_start;
        const long long stop =
            record.apply_pre_rtc ? record.raw_stop : record.ptc_stop;
        if (record.iter < 0 || record.uid < 0 || start < 0 || stop < start ||
            record.source_protected) {
            return;
        }

        const EffectiveSampleMaskKey key{
            record.obsnum, record.scan, record.apply_pre_rtc, record.uid};
        auto &intervals = effective_sample_masks[key];
        EffectiveSampleMask merged{record.iter, record.uid, start, stop};
        auto it = std::lower_bound(
            intervals.begin(), intervals.end(), merged.start,
            [](const EffectiveSampleMask &interval, long long value) {
                return interval.start < value;
            });
        if (it != intervals.begin()) {
            auto previous = std::prev(it);
            if (previous->stop >= merged.start - 1) {
                it = previous;
            }
        }
        while (it != intervals.end() && it->start <= merged.stop + 1) {
            if (it->stop < merged.start - 1) {
                ++it;
                continue;
            }
            merged.start = std::min(merged.start, it->start);
            merged.stop = std::max(merged.stop, it->stop);
            merged.iter = std::min(merged.iter, it->iter);
            it = intervals.erase(it);
        }
        intervals.insert(it, merged);
    }

    void merge_effective_detector_penalty(const DetectorPenalty &record) {
        if (record.iter < 0) {
            return;
        }
        const EffectiveDetectorPenaltyKey key{
            record.obsnum, record.producer, record.reason, record.scan,
            record.uid, record.nw, record.array, record.scan_local};
        const auto [it, inserted] =
            effective_detector_penalties.emplace(key, record);
        if (inserted) {
            return;
        }
        auto &effective = it->second;
        effective.iter = std::min(effective.iter, record.iter);
        if (std::isfinite(record.factor)) {
            if (!std::isfinite(effective.factor)) {
                effective.factor = record.factor;
            }
            else {
                effective.factor = std::min(effective.factor, record.factor);
            }
        }
        if (std::isfinite(record.score) &&
            (!std::isfinite(effective.score) || record.score > effective.score)) {
            effective.score = record.score;
        }
    }

    template <typename Record>
    void push_with_cap(std::vector<Record> &records, Record &&record,
                       std::size_t &dropped_count) {
        const auto cap = static_cast<std::size_t>(options.max_records_per_type);
        if (cap > 0 && records.size() >= cap) {
            dropped_count++;
            return;
        }
        records.emplace_back(std::forward<Record>(record));
    }
};
