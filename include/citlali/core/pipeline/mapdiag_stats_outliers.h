#pragma once

// Included by mapdiag_stats.h inside namespace citlali::pipeline.

inline bool mapdiag_has_valid_contributor(int uid, int fill_int,
                                          double contribution_signal) {
    return uid != fill_int && std::isfinite(contribution_signal);
}

inline bool mapdiag_has_full_leave_one_out_inputs(
    double total_signal, double total_weight, double contribution_weight,
    double contribution_variance_weight, double total_variance_weight,
    double remaining_weight) {
    return std::isfinite(total_signal) && std::isfinite(total_weight) &&
           std::isfinite(contribution_weight) &&
           std::isfinite(contribution_variance_weight) &&
           std::isfinite(total_variance_weight) &&
           contribution_weight >= 0.0 && contribution_variance_weight >= 0.0 &&
           remaining_weight > std::numeric_limits<double>::epsilon() &&
           total_variance_weight > contribution_variance_weight;
}

inline double mapdiag_remaining_contribution_weight(double total_weight,
                                                    double contribution_weight) {
    return total_weight - contribution_weight;
}

inline double mapdiag_full_leave_one_out_value(double total_signal,
                                               double contribution_signal,
                                               double remaining_weight) {
    return (total_signal - contribution_signal) / remaining_weight;
}

inline void mapdiag_assign_leave_one_out_z(double value, double weight,
                                           double leave_one_out_value,
                                           double &leave_one_out_z) {
    const double residual = value - leave_one_out_value;
    if (std::isfinite(residual) && std::isfinite(weight) && weight > 0.0) {
        leave_one_out_z = residual * std::sqrt(weight);
    }
}

inline bool mapdiag_has_fallback_leave_one_out_inputs(
    double weight, double contribution_weight) {
    return std::isfinite(contribution_weight) && contribution_weight >= 0.0 &&
           weight > contribution_weight &&
           (weight - contribution_weight) >
               std::numeric_limits<double>::epsilon();
}

inline double mapdiag_raw_weighted_signal(double value, double weight) {
    return value * weight;
}

inline double mapdiag_fallback_leave_one_out_value(
    double raw_weighted_signal, double contribution_signal, double weight,
    double contribution_weight) {
    return (raw_weighted_signal - contribution_signal) /
           (weight - contribution_weight);
}

template <class NoiseList>
bool mapdiag_has_noise_realizations(
    const NoiseList &noise, Eigen::Index i, Eigen::Index n_noise) {
    return !noise.empty() && i >= 0 &&
           i < static_cast<Eigen::Index>(noise.size()) && n_noise > 0;
}

inline Eigen::Index mapdiag_noise_realization_size(Eigen::Index n_rows,
                                                   Eigen::Index n_cols) {
    return n_rows * n_cols;
}

inline Eigen::Index mapdiag_noise_realization_offset(
    Eigen::Index realization_index, Eigen::Index n_rows, Eigen::Index n_cols) {
    return realization_index * mapdiag_noise_realization_size(n_rows, n_cols);
}

inline MapdiagNoiseMatrix mapdiag_noise_matrix(
    double *noise_data, Eigen::Index realization_index, Eigen::Index n_rows,
    Eigen::Index n_cols) {
    return MapdiagNoiseMatrix(
        noise_data + mapdiag_noise_realization_offset(
                         realization_index, n_rows, n_cols),
        n_rows, n_cols);
}

template <class MapBuffer>
MapdiagNoiseMatrix mapdiag_noise_matrix(
    const MapBuffer &mb, Eigen::Index map_index,
    Eigen::Index realization_index) {
    return mapdiag_noise_matrix(
        mb->noise[map_index].data(), realization_index,
        mapdiag_n_rows(mb), mapdiag_n_cols(mb));
}

inline double mapdiag_center_pixel_coordinate(Eigen::Index n_pixels) {
    return (static_cast<double>(n_pixels) - 1.0) / 2.0;
}

inline MapdiagSourceDistanceContext mapdiag_source_distance_context(
    Eigen::Index n_rows, Eigen::Index n_cols, double pixel_size_arcsec,
    double fill_value) {
    return {mapdiag_center_pixel_coordinate(n_rows),
            mapdiag_center_pixel_coordinate(n_cols),
            pixel_size_arcsec,
            fill_value};
}

template <class MapBuffer>
MapdiagSourceDistanceContext mapdiag_source_distance_context(
    const MapBuffer &mb, double rad_to_arcsec, double fill_value) {
    return mapdiag_source_distance_context(
        mapdiag_n_rows(mb), mapdiag_n_cols(mb),
        mb->pixel_size_rad * rad_to_arcsec, fill_value);
}

inline double mapdiag_source_distance_arcsec(
    Eigen::Index row, Eigen::Index col, double center_row,
    double center_col, double pixel_size_arcsec, double fill_value) {
    if (!std::isfinite(pixel_size_arcsec) || pixel_size_arcsec <= 0.0) {
        return fill_value;
    }
    const double drow =
        (static_cast<double>(row) - center_row) * pixel_size_arcsec;
    const double dcol =
        (static_cast<double>(col) - center_col) * pixel_size_arcsec;
    return std::hypot(drow, dcol);
}

inline double mapdiag_source_distance_arcsec(
    Eigen::Index row, Eigen::Index col,
    const MapdiagSourceDistanceContext &context) {
    return mapdiag_source_distance_arcsec(
        row, col, context.center_row, context.center_col,
        context.pixel_size_arcsec, context.fill_value);
}

inline bool mapdiag_is_source_protected(double distance_arcsec,
                                        double protect_radius_arcsec) {
    return protect_radius_arcsec > 0.0 && std::isfinite(distance_arcsec) &&
           distance_arcsec <= protect_radius_arcsec;
}

inline void apply_mapdiag_source_protection_mask(
    Eigen::ArrayXXd &mask, const MapdiagSourceDistanceContext &context,
    double protect_radius_arcsec) {
    for (Eigen::Index r = 0; r < mask.rows(); ++r) {
        for (Eigen::Index c = 0; c < mask.cols(); ++c) {
            const double dist_arcsec =
                mapdiag_source_distance_arcsec(r, c, context);
            if (mapdiag_is_source_protected(
                    dist_arcsec, protect_radius_arcsec)) {
                mask(r, c) = 0.0;
            }
        }
    }
}

inline Eigen::ArrayXXd mapdiag_off_source_core_mask(
    const Eigen::ArrayXXd &core_mask,
    const MapdiagSourceDistanceContext &context,
    double protect_radius_arcsec) {
    Eigen::ArrayXXd off_source_mask = core_mask;
    apply_mapdiag_source_protection_mask(
        off_source_mask, context, protect_radius_arcsec);
    return off_source_mask;
}

inline bool mapdiag_has_minimum_samples(std::size_t n_values,
                                        std::size_t min_values) {
    return n_values >= min_values;
}

inline bool mapdiag_has_enough_off_source_values(
    const std::vector<double> &values) {
    return mapdiag_has_minimum_samples(values.size(), 8);
}

inline std::vector<double> mapdiag_absolute_deviations(
    const std::vector<double> &values, double center) {
    std::vector<double> abs_dev;
    abs_dev.reserve(values.size());
    for (const auto &value : values) {
        abs_dev.push_back(std::abs(value - center));
    }
    return abs_dev;
}

inline double mapdiag_robust_sigma(
    const MapdiagStatsContext &stats, const std::vector<double> &values,
    double center) {
    return 1.4826 * stats.median(mapdiag_absolute_deviations(values, center));
}

inline MapdiagRobustCenterStats mapdiag_robust_center_stats(
    const MapdiagStatsContext &stats, const std::vector<double> &values) {
    const double center = stats.median(values);
    return {center, mapdiag_robust_sigma(stats, values, center)};
}

inline bool mapdiag_has_valid_robust_center_stats(
    const MapdiagRobustCenterStats &stats) {
    return std::isfinite(stats.center) && std::isfinite(stats.robust_sigma) &&
           stats.robust_sigma > std::numeric_limits<double>::epsilon();
}

inline bool mapdiag_is_valid_outlier_pixel_value(double value, double weight,
                                                 double sig2noise) {
    return std::isfinite(value) && std::isfinite(weight) && weight > 0.0 &&
           std::isfinite(sig2noise);
}

inline bool mapdiag_passes_min_effective_samples(double n_eff,
                                                 double min_n_eff) {
    return !std::isfinite(n_eff) || n_eff >= min_n_eff;
}

inline double mapdiag_robust_z(double sig2noise,
                               const MapdiagRobustCenterStats &stats) {
    return (sig2noise - stats.center) / stats.robust_sigma;
}

inline bool mapdiag_passes_min_abs_z(double z, double min_abs_z) {
    return std::isfinite(z) && std::abs(z) >= min_abs_z;
}

inline MapdiagMapPixelCandidate make_mapdiag_map_pixel_candidate(
    Eigen::Index row, Eigen::Index col, double value, double weight,
    double n_eff, double robust_z, double source_distance_arcsec,
    int fill_int, double fill_double) {
    return {static_cast<int>(row),
            static_cast<int>(col),
            fill_int,
            fill_int,
            fill_int,
            value,
            weight,
            n_eff,
            robust_z,
            robust_z,
            source_distance_arcsec,
            false,
            false};
}

inline std::vector<MapdiagMapPixelCandidate> make_mapdiag_pixel_candidates() {
    return {};
}

inline void append_mapdiag_pixel_candidate(
    std::vector<MapdiagMapPixelCandidate> &candidates,
    const MapdiagMapPixelCandidate &candidate) {
    candidates.push_back(candidate);
}

inline const MapdiagMapPixelCandidate &mapdiag_emitted_candidate(
    const std::vector<MapdiagMapPixelCandidate> &candidates,
    std::size_t index) {
    return candidates[index];
}

inline void assign_mapdiag_candidate_contributor(
    MapdiagMapPixelCandidate &candidate, int uid, int scan,
    long long sample) {
    candidate.has_contributor = true;
    candidate.uid = uid;
    candidate.scan = scan;
    candidate.sample = sample;
}

template <class ScanMatrix, class SampleMatrix>
void assign_mapdiag_candidate_contributor_from_products(
    MapdiagMapPixelCandidate &candidate, int uid,
    const ScanMatrix &scan_matrix, const SampleMatrix &sample_matrix,
    Eigen::Index row, Eigen::Index col) {
    assign_mapdiag_candidate_contributor(
        candidate, uid, mapdiag_matrix_value(scan_matrix, row, col),
        mapdiag_matrix_value(sample_matrix, row, col));
}

inline bool mapdiag_candidate_abs_z_greater(
    const MapdiagMapPixelCandidate &a,
    const MapdiagMapPixelCandidate &b) {
    return std::abs(a.robust_z) > std::abs(b.robust_z);
}

inline void sort_mapdiag_pixel_candidates(
    std::vector<MapdiagMapPixelCandidate> &candidates) {
    std::sort(
        candidates.begin(), candidates.end(),
        mapdiag_candidate_abs_z_greater);
}

inline std::size_t mapdiag_candidate_emit_count(std::size_t n_candidates,
                                                int top_n) {
    return std::min<std::size_t>(
        n_candidates, static_cast<std::size_t>(top_n));
}

inline bool mapdiag_candidate_has_dominance_key(
    const MapdiagMapPixelCandidate &candidate, int fill_int) {
    return candidate.has_contributor && !candidate.source_protected &&
           candidate.uid != fill_int && candidate.scan != fill_int &&
           candidate.uid >= 0 && candidate.scan >= 0;
}

inline std::vector<MapdiagDetectorDominance>
make_mapdiag_detector_dominance_list() {
    return {};
}

inline bool mapdiag_dominance_matches_candidate(
    const MapdiagDetectorDominance &entry,
    const MapdiagMapPixelCandidate &candidate) {
    return entry.uid == candidate.uid && entry.scan == candidate.scan;
}

inline MapdiagDetectorDominance make_mapdiag_detector_dominance_entry(
    const MapdiagMapPixelCandidate &candidate) {
    return {candidate.uid, candidate.scan, 0, 0.0, 0.0};
}

inline void update_mapdiag_detector_dominance_stats(
    MapdiagDetectorDominance &entry,
    const MapdiagMapPixelCandidate &candidate) {
    ++entry.count;
    if (std::isfinite(candidate.value)) {
        entry.max_abs_value =
            std::max(entry.max_abs_value, std::abs(candidate.value));
    }
    if (std::isfinite(candidate.leave_one_out_z)) {
        entry.max_abs_leave_one_out_z =
            std::max(entry.max_abs_leave_one_out_z,
                     std::abs(candidate.leave_one_out_z));
    }
}

inline void update_mapdiag_detector_dominance(
    std::vector<MapdiagDetectorDominance> &dominance,
    const MapdiagMapPixelCandidate &candidate, int fill_int) {
    if (!mapdiag_candidate_has_dominance_key(candidate, fill_int)) {
        return;
    }
    auto it = std::find_if(
        dominance.begin(), dominance.end(),
        [&](const auto &entry) {
            return mapdiag_dominance_matches_candidate(entry, candidate);
        });
    if (it == dominance.end()) {
        dominance.push_back(make_mapdiag_detector_dominance_entry(candidate));
        it = dominance.end() - 1;
    }
    update_mapdiag_detector_dominance_stats(*it, candidate);
}

inline const char *mapdiag_map_pixel_outlier_reason(bool has_contributor,
                                                    bool targeted) {
    if (!has_contributor) {
        return "extreme_pixel_no_contributor";
    }
    return targeted ? "extreme_pixel_targeted_contributor"
                    : "extreme_pixel_contributor";
}

template <class MapBuffer>
const char *mapdiag_map_pixel_outlier_reason(
    const MapdiagMapPixelCandidate &candidate, const MapBuffer &mb) {
    return mapdiag_map_pixel_outlier_reason(
        candidate.has_contributor, mb->contribution_diag_targeted);
}

inline const char *mapdiag_detector_dominance_penalty_reason() {
    return "map_pixel_outlier_detector_dominance";
}

template <class Record, class Obsnum, class Producer, class Reason>
void assign_mapdiag_outlier_record_context(
    Record &record, const Obsnum &obsnum, const Producer &producer,
    const Reason &reason, int iter, int map_index) {
    record.obsnum = obsnum;
    record.producer = producer;
    record.reason = reason;
    record.iter = iter;
    record.map_index = map_index;
}

template <class Record>
void assign_mapdiag_outlier_record_candidate(
    Record &record, const MapdiagMapPixelCandidate &candidate) {
    record.scan = candidate.scan;
    record.uid = candidate.uid;
    record.row = candidate.row;
    record.col = candidate.col;
    record.sample = candidate.sample;
    record.value = candidate.value;
    record.weight = candidate.weight;
    record.n_eff = candidate.n_eff;
    record.leave_one_out_z = candidate.leave_one_out_z;
    record.source_distance_arcsec = candidate.source_distance_arcsec;
    record.source_protected = candidate.source_protected;
}

template <class Record, class Obsnum, class Producer, class Reason>
Record make_mapdiag_outlier_record(
    const Obsnum &obsnum, const Producer &producer, const Reason &reason,
    int iter, int map_index, const MapdiagMapPixelCandidate &candidate) {
    Record record;
    assign_mapdiag_outlier_record_context(
        record, obsnum, producer, reason, iter, map_index);
    assign_mapdiag_outlier_record_candidate(record, candidate);
    return record;
}

template <class Penalty, class Obsnum, class Producer, class Reason>
void assign_mapdiag_detector_penalty_context(
    Penalty &penalty, const Obsnum &obsnum, const Producer &producer,
    const Reason &reason, int iter, const MapdiagDetectorDominance &entry,
    int array_id) {
    penalty.obsnum = obsnum;
    penalty.producer = producer;
    penalty.reason = reason;
    penalty.iter = iter;
    penalty.scan = entry.scan;
    penalty.uid = entry.uid;
    penalty.nw = -1;
    penalty.array = array_id;
}

template <class Penalty>
void assign_mapdiag_detector_penalty_dominance(
    Penalty &penalty, const MapdiagDetectorDominance &entry) {
    penalty.factor = 0.0;
    penalty.score = static_cast<double>(entry.count);
    penalty.scan_local = true;
}

template <class Penalty, class Obsnum, class Producer, class Reason>
Penalty make_mapdiag_detector_penalty(
    const Obsnum &obsnum, const Producer &producer, const Reason &reason,
    int iter, const MapdiagDetectorDominance &entry, int array_id) {
    Penalty penalty;
    assign_mapdiag_detector_penalty_context(
        penalty, obsnum, producer, reason, iter, entry, array_id);
    assign_mapdiag_detector_penalty_dominance(penalty, entry);
    return penalty;
}

inline bool mapdiag_dominance_meets_min_pixels(
    const MapdiagDetectorDominance &entry, int min_pixels) {
    return entry.count >= min_pixels;
}

template <class ArrayIds>
int mapdiag_array_id_or_default(Eigen::Index map_index,
                                const ArrayIds &array_ids,
                                int default_array_id) {
    if (map_index >= 0 &&
        map_index < static_cast<Eigen::Index>(array_ids.size())) {
        return array_ids[map_index];
    }
    return default_array_id;
}

inline int mapdiag_display_scan_index(int scan) {
    return scan + 1;
}

inline bool mapdiag_mask_pixel_is_selected(const Eigen::ArrayXXd &mask,
                                           Eigen::Index row,
                                           Eigen::Index col) {
    return mask(row, col) > 0.0;
}

template <class ReductionLearning>
bool mapdiag_outlier_diagnostics_enabled(
    const ReductionLearning &reduction_learning) {
    return reduction_learning.is_enabled() &&
           reduction_learning.diagnostics_enabled() &&
           reduction_learning.options.map_pixel_outlier_diagnostics_enabled &&
           reduction_learning.options.map_pixel_outlier_top_n > 0;
}

template <class ReductionLearning>
double mapdiag_source_protect_radius_arcsec(
    const ReductionLearning &reduction_learning) {
    return reduction_learning.options.map_pixel_outlier_source_radius_arcsec;
}

template <class ReductionLearning>
double mapdiag_min_effective_samples(
    const ReductionLearning &reduction_learning) {
    return reduction_learning.options.map_pixel_outlier_min_n_eff;
}

template <class ReductionLearning>
double mapdiag_min_abs_z(const ReductionLearning &reduction_learning) {
    return reduction_learning.options.map_pixel_outlier_min_abs_z;
}

template <class ReductionLearning>
int mapdiag_candidate_top_n(const ReductionLearning &reduction_learning) {
    return reduction_learning.options.map_pixel_outlier_top_n;
}

template <class ReductionLearning>
bool mapdiag_detector_exclusion_enabled(
    const ReductionLearning &reduction_learning) {
    return reduction_learning.options
        .map_pixel_outlier_detector_exclusion_enabled;
}

template <class ReductionLearning>
int mapdiag_detector_exclusion_min_pixels(
    const ReductionLearning &reduction_learning) {
    return reduction_learning.options
        .map_pixel_outlier_detector_exclusion_min_pixels;
}

