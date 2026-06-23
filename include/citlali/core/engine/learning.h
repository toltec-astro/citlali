#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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
        double group_median = nan_value();
        double robust_z = nan_value();
        double cap = nan_value();
        bool cap_recommended = false;
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
        int max_unflagged_residual_uid = -1;
        int top_candidate_sample = -1;
        double top_candidate_score = nan_value();
        double max_unflagged_residual_z = nan_value();
        bool busy_vetoed = false;
        bool selective_acceptance_recommended = false;
    };

    Options options;
    int current_iter = -1;
    IterationPhase current_phase = IterationPhase::Inactive;
    std::string current_reduction_type = "unknown";
    bool current_source_model_available = false;
    int begin_count = 0;
    int finalize_count = 0;

    std::vector<LearnedSampleMask> learned_sample_masks;
    std::vector<DetectorPenalty> detector_penalties;
    std::vector<HighWeightDetector> high_weight_detectors;
    std::vector<MapPixelOutlier> map_pixel_outliers;
    std::vector<SourceProtectionSummary> source_protection_summaries;
    std::vector<BusyNetworkSummary> busy_network_summaries;

    std::size_t dropped_learned_sample_masks = 0;
    std::size_t dropped_detector_penalties = 0;
    std::size_t dropped_high_weight_detectors = 0;
    std::size_t dropped_map_pixel_outliers = 0;
    std::size_t dropped_source_protection_summaries = 0;
    std::size_t dropped_busy_network_summaries = 0;

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
        options = new_options;
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
        begin_count++;
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
        learned_sample_masks.clear();
        detector_penalties.clear();
        high_weight_detectors.clear();
        map_pixel_outliers.clear();
        source_protection_summaries.clear();
        busy_network_summaries.clear();
        dropped_learned_sample_masks = 0;
        dropped_detector_penalties = 0;
        dropped_high_weight_detectors = 0;
        dropped_map_pixel_outliers = 0;
        dropped_source_protection_summaries = 0;
        dropped_busy_network_summaries = 0;
    }

    void record_learned_sample_mask(LearnedSampleMask record) {
        std::lock_guard<std::mutex> lock(*mutex);
        if (!options.enabled) {
            return;
        }
        push_with_cap(learned_sample_masks, std::move(record),
                      dropped_learned_sample_masks);
    }

    void record_detector_penalty(DetectorPenalty record) {
        std::lock_guard<std::mutex> lock(*mutex);
        if (!options.enabled) {
            return;
        }
        push_with_cap(detector_penalties, std::move(record),
                      dropped_detector_penalties);
    }

    void record_high_weight_detector(HighWeightDetector record) {
        std::lock_guard<std::mutex> lock(*mutex);
        if (!options.enabled) {
            return;
        }
        push_with_cap(high_weight_detectors, std::move(record),
                      dropped_high_weight_detectors);
    }

    void record_map_pixel_outlier(MapPixelOutlier record) {
        std::lock_guard<std::mutex> lock(*mutex);
        if (!options.enabled) {
            return;
        }
        push_with_cap(map_pixel_outliers, std::move(record),
                      dropped_map_pixel_outliers);
    }

    void record_source_protection_summary(SourceProtectionSummary record) {
        std::lock_guard<std::mutex> lock(*mutex);
        if (!options.enabled) {
            return;
        }
        push_with_cap(source_protection_summaries, std::move(record),
                      dropped_source_protection_summaries);
    }

    void record_busy_network_summary(BusyNetworkSummary record) {
        std::lock_guard<std::mutex> lock(*mutex);
        if (!options.enabled) {
            return;
        }
        push_with_cap(busy_network_summaries, std::move(record),
                      dropped_busy_network_summaries);
    }

    std::string summary_string() const {
        std::lock_guard<std::mutex> lock(*mutex);
        std::ostringstream os;
        os << "enabled=" << options.enabled
           << " diagnostics=" << options.diagnostics_enabled
           << " iter=" << current_iter
           << " phase=" << phase_name(current_phase)
           << " reduction_type=" << current_reduction_type
           << " source_model_available=" << current_source_model_available
           << " sample_masks=" << learned_sample_masks.size()
           << " detector_penalties=" << detector_penalties.size()
           << " high_weight_detectors=" << high_weight_detectors.size()
           << " map_pixel_outliers=" << map_pixel_outliers.size()
           << " source_protection_summaries=" << source_protection_summaries.size()
           << " busy_network_summaries=" << busy_network_summaries.size()
           << " dropped_records="
           << (dropped_learned_sample_masks + dropped_detector_penalties +
               dropped_high_weight_detectors + dropped_map_pixel_outliers +
               dropped_source_protection_summaries +
               dropped_busy_network_summaries);
        return os.str();
    }

private:
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
