#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_prior_qc_stats.h>

Beammap::BeammapPriorAlignmentOverlapBox Beammap::beammap_prior_alignment_overlap_box(
    const Beammap::BeammapPriorAlignmentSamples &alignment_samples,
    const citlali::config::BeammapPriorsConfig &priors_config) const {
    const double q_low =
        priors_config.alignment_common_support_quantile;
    const double q_high =
        1.0 - priors_config.alignment_common_support_quantile;
    BeammapPriorAlignmentOverlapBox overlap_box;
    bool overlap_valid = true;

    for (const auto &[array, pairs] : alignment_samples.pairs_by_array) {
        static_cast<void>(array);
        std::vector<double> xs;
        std::vector<double> ys;
        xs.reserve(pairs.size());
        ys.reserve(pairs.size());
        for (const auto &pair : pairs) {
            if (std::isfinite(pair.slot_x) && std::isfinite(pair.slot_y)) {
                xs.push_back(pair.slot_x);
                ys.push_back(pair.slot_y);
            }
        }
        const double x_low = beammap_prior_qc_stats::quantile(xs, q_low);
        const double x_high = beammap_prior_qc_stats::quantile(xs, q_high);
        const double y_low = beammap_prior_qc_stats::quantile(ys, q_low);
        const double y_high = beammap_prior_qc_stats::quantile(ys, q_high);
        if (!(std::isfinite(x_low) && std::isfinite(x_high) &&
              std::isfinite(y_low) && std::isfinite(y_high))) {
            overlap_valid = false;
            break;
        }
        overlap_box.x_low = std::max(overlap_box.x_low, x_low);
        overlap_box.x_high = std::min(overlap_box.x_high, x_high);
        overlap_box.y_low = std::max(overlap_box.y_low, y_low);
        overlap_box.y_high = std::min(overlap_box.y_high, y_high);
    }

    overlap_box.valid = overlap_valid &&
                        overlap_box.x_low < overlap_box.x_high &&
                        overlap_box.y_low < overlap_box.y_high;
    return overlap_box;
}

std::vector<Beammap::BeammapPriorAlignmentPair>
Beammap::filter_beammap_prior_alignment_pairs_to_overlap_box(
    const Beammap::BeammapPriorAlignmentSamples &alignment_samples,
    const Beammap::BeammapPriorAlignmentOverlapBox &overlap_box) const {
    std::vector<BeammapPriorAlignmentPair> filtered_pairs;
    filtered_pairs.reserve(alignment_samples.all_pairs.size());
    for (const auto &pair : alignment_samples.all_pairs) {
        if (pair.slot_x >= overlap_box.x_low && pair.slot_x <= overlap_box.x_high &&
            pair.slot_y >= overlap_box.y_low && pair.slot_y <= overlap_box.y_high) {
            filtered_pairs.push_back(pair);
        }
    }
    return filtered_pairs;
}

std::vector<Beammap::BeammapPriorAlignmentPair>
Beammap::select_common_beammap_prior_alignment_pairs(
    const Beammap::BeammapPriorAlignmentSamples &alignment_samples,
    const citlali::config::BeammapPriorsConfig &priors_config) {
    auto common_pairs = alignment_samples.all_pairs;
    if (citlali::config::uses_overlap_box_prior_alignment_support(
            priors_config) &&
        alignment_samples.pairs_by_array.size() >= 2) {
        const auto overlap_box =
            beammap_prior_alignment_overlap_box(alignment_samples, priors_config);
        if (overlap_box.valid) {
            auto filtered_pairs =
                filter_beammap_prior_alignment_pairs_to_overlap_box(
                    alignment_samples, overlap_box);
            if (filtered_pairs.size() >=
                static_cast<std::size_t>(
                    priors_config.alignment_min_matches)) {
                common_pairs.swap(filtered_pairs);
            }
            logger->info(
                "beammap prior common alignment overlap_box (iter {}): q={} x=[{}, {}] y=[{}, {}] kept={}/{}",
                current_iter,
                priors_config.alignment_common_support_quantile,
                overlap_box.x_low, overlap_box.x_high, overlap_box.y_low, overlap_box.y_high,
                common_pairs.size(), alignment_samples.all_pairs.size());
        }
        else {
            logger->debug(
                "beammap prior common alignment overlap_box skipped: invalid overlap x=[{}, {}] y=[{}, {}]",
                overlap_box.x_low, overlap_box.x_high, overlap_box.y_low, overlap_box.y_high);
        }
    }

    return common_pairs;
}
