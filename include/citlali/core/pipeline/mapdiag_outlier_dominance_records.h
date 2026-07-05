#pragma once

// Included by mapdiag_stats_outliers.h inside namespace citlali::pipeline.

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

