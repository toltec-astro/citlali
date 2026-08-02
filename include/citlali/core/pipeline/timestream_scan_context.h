#pragma once

#include <citlali/core/config/runtime_config.h>
#include <citlali/core/config/timestream_enums.h>
#include <citlali/core/error/error.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/sci_align_scan_contract.h>
#include <citlali/core/pipeline/timestream_alignment_state.h>

#include <Eigen/Core>
#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <tuple>
#include <vector>

namespace citlali::pipeline {

struct RtcScanSampleWindow {
    Eigen::Index start = 0;
    Eigen::Index length = 0;
};

template <class RtcData, class Telescope, class PointingOffsets>
RtcScanSampleWindow copy_rtc_scan_context(
    RtcData &rtcdata, const Telescope &telescope,
    const PointingOffsets &pointing_offsets_arcsec) {
    if (rtcdata.scan_indices.data.size() < 4) {
        throw citlali::error::runtime(
            "RTC scan context requires four inclusive scan indices");
    }
    const Eigen::Index start = rtcdata.scan_indices.data(2);
    const Eigen::Index inclusive_stop = rtcdata.scan_indices.data(3);
    if (start < 0 || inclusive_stop < start ||
        inclusive_stop - start ==
            std::numeric_limits<Eigen::Index>::max()) {
        throw citlali::error::runtime(
            "RTC scan context has invalid inclusive sample bounds");
    }
    const Eigen::Index length = inclusive_stop - start + 1;

    for (const auto &[key, values] : telescope.tel_data) {
        if (start > values.size() || length > values.size() - start) {
            throw citlali::error::runtime(
                "telescope field does not cover the RTC scan context");
        }
        rtcdata.tel_data.data[key] = values.segment(start, length);
    }

    for (const auto &[axis, offset] : pointing_offsets_arcsec) {
        if (start > offset.size() || length > offset.size() - start) {
            throw citlali::error::runtime(
                "pointing offset does not cover the RTC scan context");
        }
        rtcdata.pointing_offsets_arcsec.data[axis] =
            offset.segment(start, length);
    }

    return {start, length};
}

template <class RtcData, class Calib>
void copy_hwpr_angle_if_enabled(
    RtcData &rtcdata, const Calib &calib, bool run_polarization,
    bool run_hwpr, Eigen::Index hwpr_start_index, Eigen::Index scan_start,
    Eigen::Index scan_length) {
    if (run_polarization && run_hwpr) {
        if (scan_start < 0 || hwpr_start_index < 0 || scan_length < 0 ||
            hwpr_start_index >
                std::numeric_limits<Eigen::Index>::max() - scan_start) {
            throw citlali::error::runtime(
                "HWPR scan context has invalid sample bounds");
        }
        const Eigen::Index start = scan_start + hwpr_start_index;
        if (start > calib.hwpr_angle.size() ||
            scan_length > calib.hwpr_angle.size() - start) {
            throw citlali::error::runtime(
                "HWPR angle does not cover the RTC scan context");
        }
        rtcdata.hwpr_angle.data =
            calib.hwpr_angle.segment(start, scan_length);
    }
}

template <class RtcData>
void initialize_rtc_flags(RtcData &rtcdata) {
    rtcdata.flags.data.resize(rtcdata.scans.data.rows(),
                              rtcdata.scans.data.cols());
    rtcdata.flags.data.setConstant(0);
}

struct RtcGapChunkClassification {
    Eigen::VectorXi exact_missing;
    Eigen::VectorXi processing_guard;
    Eigen::Index cumulative_missing = 0;
    Eigen::Index longest_missing_run = 0;
    bool network_chunk_unusable = false;
};

template <class BinaryVector>
std::vector<AlignmentIndexRun> compact_binary_runs(
    const BinaryVector &values, Eigen::Index global_start) {
    if (global_start < 0 ||
        values.size() >
            std::numeric_limits<Eigen::Index>::max() - global_start) {
        throw citlali::error::runtime(
            "binary run support exceeds Eigen index range");
    }
    std::vector<AlignmentIndexRun> result;
    Eigen::Index local = 0;
    while (local < values.size()) {
        while (local < values.size() && values(local) == 0) {
            ++local;
        }
        if (local == values.size()) {
            break;
        }
        const Eigen::Index begin = local;
        while (local < values.size() && values(local) != 0) {
            ++local;
        }
        result.push_back(
            {global_start + begin, global_start + local});
    }
    return result;
}

template <class Mask, class ContextSamples>
RtcGapChunkClassification classify_gap_mask_chunk(
    const Mask &mask, Eigen::Index chunk_start, Eigen::Index chunk_size,
    ContextSamples context_samples,
    bool apply_local_unusable_short_circuit = true) {
    if (chunk_start < 0 || chunk_size < 0 || chunk_start > mask.size() ||
        chunk_size > mask.size() - chunk_start) {
        throw citlali::error::runtime(
            "gap mask does not cover the realized processing chunk");
    }
    if constexpr (std::is_signed_v<ContextSamples>) {
        if (context_samples < 0) {
            throw citlali::error::runtime(
                "gap processing guard must be nonnegative");
        }
    }
    const auto context_value = static_cast<long double>(context_samples);
    if (!std::isfinite(context_value) ||
        context_value >
            static_cast<long double>(
                std::numeric_limits<Eigen::Index>::max())) {
        throw citlali::error::runtime(
            "gap processing guard is outside the supported sample range");
    }
    const Eigen::Index context =
        static_cast<Eigen::Index>(context_samples);

    RtcGapChunkClassification result;
    result.exact_missing = Eigen::VectorXi::Zero(chunk_size);
    result.processing_guard = Eigen::VectorXi::Zero(chunk_size);
    if (chunk_size == 0) {
        return result;
    }

    const Eigen::Index chunk_end = chunk_start + chunk_size;
    const Eigen::Index extended_start =
        context > chunk_start ? 0 : chunk_start - context;
    const Eigen::Index extended_end =
        context > mask.size() - chunk_end
            ? mask.size()
            : chunk_end + context;
    for (Eigen::Index global = extended_start; global < extended_end;
         ++global) {
        if (mask(global) != 0 && mask(global) != 1) {
            throw citlali::error::runtime(
                "gap mask must contain only zero or one");
        }
    }

    Eigen::Index run = 0;
    for (Eigen::Index local = 0; local < chunk_size; ++local) {
        if (mask(chunk_start + local) == 0) {
            result.exact_missing(local) = 1;
            ++result.cumulative_missing;
            ++run;
            result.longest_missing_run =
                std::max(result.longest_missing_run, run);
        } else {
            run = 0;
        }
    }

    // Fixed-cadence common-grid cells make the corresponding elapsed-duration
    // ratios identical. Keep both approved run and cumulative tests explicit;
    // exactly one quarter is allowed and only a strict excess is unusable.
    const auto exceeds_quarter = [chunk_size](Eigen::Index missing) {
        return static_cast<long double>(missing) * 4.0L >
               static_cast<long double>(chunk_size);
    };
    result.network_chunk_unusable =
        exceeds_quarter(result.longest_missing_run) ||
        exceeds_quarter(result.cumulative_missing);
    if ((apply_local_unusable_short_circuit &&
         result.network_chunk_unusable) ||
        context == 0) {
        return result;
    }

    // A gap just outside the chunk can still contribute a processing guard
    // inside it. The exact-missing identity remains separate from guard-only
    // samples even though the existing RTC output is a binary union.
    for (Eigen::Index gap = extended_start; gap < extended_end; ++gap) {
        if (mask(gap) != 0) {
            continue;
        }
        const Eigen::Index guarded_global_begin =
            context > gap ? 0 : gap - context;
        const Eigen::Index guarded_global_end =
            context >= mask.size() - gap
                ? mask.size()
                : gap + context + 1;
        const Eigen::Index guarded_begin =
            std::max(chunk_start, guarded_global_begin);
        const Eigen::Index guarded_end =
            std::min(chunk_end, guarded_global_end);
        for (Eigen::Index global = guarded_begin; global < guarded_end;
             ++global) {
            const Eigen::Index local = global - chunk_start;
            if (result.exact_missing(local) == 0) {
                result.processing_guard(local) = 1;
            }
        }
    }
    return result;
}

inline std::uint64_t run_cardinality(
    const std::vector<AlignmentIndexRun> &runs) {
    std::uint64_t result = 0;
    for (const auto &run : runs) {
        if (run.start < 0 || run.stop < run.start) {
            throw citlali::error::runtime(
                "ALIGN run has invalid half-open support");
        }
        result = checked_alignment_count_add(
            result, static_cast<std::uint64_t>(run.stop - run.start),
            "ALIGN run cardinality");
    }
    return result;
}

inline void add_to_compact_alignment_run_union(
    std::vector<AlignmentIndexRun> &runs, AlignmentIndexRun incoming) {
    if (incoming.start < 0 || incoming.stop <= incoming.start) {
        throw citlali::error::runtime(
            "ALIGN run union received invalid half-open support");
    }
    auto first = std::lower_bound(
        runs.begin(), runs.end(), incoming.start,
        [](const AlignmentIndexRun &run, Eigen::Index start) {
            return run.stop < start;
        });
    auto after = first;
    while (after != runs.end() && after->start <= incoming.stop) {
        incoming.start = std::min(incoming.start, after->start);
        incoming.stop = std::max(incoming.stop, after->stop);
        ++after;
    }
    first = runs.erase(first, after);
    runs.insert(first, incoming);
}

inline std::uint64_t compact_alignment_run_union_cardinality(
    const std::vector<AlignmentIndexRun> &runs) {
    std::uint64_t result = 0;
    Eigen::Index previous_stop = 0;
    bool first = true;
    for (const auto &run : runs) {
        if (run.start < 0 || run.stop <= run.start ||
            (!first && run.start <= previous_stop)) {
            throw citlali::error::runtime(
                "ALIGN run union is not compact and deterministically ordered");
        }
        result = checked_alignment_count_add(
            result, static_cast<std::uint64_t>(run.stop - run.start),
            "ALIGN compact run-union cardinality");
        previous_stop = run.stop;
        first = false;
    }
    return result;
}

inline bool alignment_chunk_disposition_key_less(
    const AlignmentChunkDisposition &left,
    const AlignmentChunkDisposition &right) {
    if (left.compatibility_ordinal != right.compatibility_ordinal) {
        return left.compatibility_ordinal < right.compatibility_ordinal;
    }
    if (left.roach_index != right.roach_index) {
        return left.roach_index < right.roach_index;
    }
    return left.interface_id < right.interface_id;
}

inline bool alignment_chunk_disposition_is_exceptional(
    const AlignmentChunkDisposition &disposition) {
    return disposition.cumulative_missing_count != 0 ||
           disposition.longest_missing_run_count != 0 ||
           disposition.full_network_unusable ||
           !disposition.synthesized_missing_runs.empty() ||
           !disposition.unavailable_missing_runs.empty() ||
           !disposition.processing_guard_runs.empty();
}

inline const AlignmentChunkDisposition *find_alignment_chunk_disposition(
    const TimestreamAlignmentState &alignment,
    Eigen::Index compatibility_ordinal, Eigen::Index roach_index) {
    if (!alignment.processing_support.observation_resolved) {
        throw citlali::error::runtime(
            "ALIGN gap processing plan is not observation-resolved");
    }
    if (compatibility_ordinal < 0) {
        throw citlali::error::runtime(
            "ALIGN gap disposition has an invalid scan ordinal");
    }
    const auto interface_it = std::find_if(
        alignment.interfaces.begin(), alignment.interfaces.end(),
        [roach_index](const AlignmentInterfaceSummary &interface) {
            return interface.roach_index == roach_index;
        });
    if (interface_it == alignment.interfaces.end()) {
        throw citlali::error::runtime(fmt::format(
            "missing ALIGN detector interface for network {}", roach_index));
    }
    const auto disposition_it = std::lower_bound(
        alignment.chunk_dispositions.begin(),
        alignment.chunk_dispositions.end(),
        std::make_pair(compatibility_ordinal, roach_index),
        [](const AlignmentChunkDisposition &disposition,
           const std::pair<Eigen::Index, Eigen::Index> &key) {
            return disposition.compatibility_ordinal < key.first ||
                   (disposition.compatibility_ordinal == key.first &&
                    disposition.roach_index < key.second);
        });
    if (disposition_it == alignment.chunk_dispositions.end() ||
        disposition_it->compatibility_ordinal != compatibility_ordinal ||
        disposition_it->roach_index != roach_index) {
        // Ordinary all-original, zero-gap scan/interface state is the
        // generative default and intentionally has no stored disposition.
        return nullptr;
    }
    if (disposition_it->interface_id != interface_it->interface_id) {
        throw citlali::error::runtime(
            "ALIGN sparse gap disposition conflicts with interface identity");
    }
    return &*disposition_it;
}

class AlignmentGapSynthesisPermissionView {
public:
    AlignmentGapSynthesisPermissionView(
        const TimestreamAlignmentState &alignment,
        Eigen::Index compatibility_ordinal)
        : alignment_(&alignment),
          compatibility_ordinal_(compatibility_ordinal) {}

    std::size_t size() const noexcept {
        return alignment_->interfaces.size();
    }

    unsigned char operator[](std::size_t interface_index) const {
        const auto &interface = alignment_->interfaces.at(interface_index);
        if (compatibility_ordinal_ < 0) {
            throw citlali::error::runtime(
                "ALIGN gap-permission view has an invalid scan ordinal");
        }
        const auto *disposition = find_alignment_chunk_disposition(
            *alignment_, compatibility_ordinal_, interface.roach_index);
        if (disposition == nullptr) {
            return static_cast<unsigned char>(citlali::config::is_xs_tod_type(
                alignment_->processing_support.signal_domain));
        }
        return static_cast<unsigned char>(
            disposition->continuity_surrogate_permitted &&
            !disposition->full_network_unusable);
    }

    unsigned char front() const {
        if (size() == 0) {
            throw citlali::error::runtime(
                "ALIGN gap-permission view has no detector interfaces");
        }
        return (*this)[0];
    }

private:
    const TimestreamAlignmentState *alignment_ = nullptr;
    Eigen::Index compatibility_ordinal_ = -1;
};

inline AlignmentGapSynthesisPermissionView
alignment_gap_synthesis_permissions(
    const TimestreamAlignmentState &alignment,
    Eigen::Index compatibility_ordinal) {
    if (!alignment.processing_support.observation_resolved) {
        throw citlali::error::runtime(
            "ALIGN gap processing plan is not observation-resolved");
    }
    return {alignment, compatibility_ordinal};
}

inline void finalize_alignment_gap_processing_plan(
    TimestreamAlignmentState &alignment,
    const sci_align::ScanWindowPlan &scan_plan,
    Eigen::Index guard_context_samples,
    citlali::config::TodType signal_domain) {
    if (!alignment.grid.initialized ||
        alignment.masks.size() != alignment.interfaces.size()) {
        throw citlali::error::runtime(
            "cannot resolve gap processing without complete alignment state");
    }
    sci_align::validate_scan_window_plan(scan_plan);
    alignment.chunk_dispositions.clear();
    alignment.processing_support = AlignmentProcessingSupportSummary{};
    alignment.processing_support.signal_domain =
        std::string{citlali::config::to_string(signal_domain)};
    alignment.support.guarded_original_count = 0;
    alignment.support.gap_policy_eligible_original_count = 0;
    const bool continuity_permitted =
        citlali::config::is_xs_tod_type(signal_domain);
    std::vector<std::vector<AlignmentIndexRun>> synthesized_unique_support(
        alignment.masks.size());

    alignment.chunk_dispositions.reserve(alignment.exceptions.size());
    for (const auto stable_id : scan_plan.compatibility_to_stable_id) {
        const auto &record = scan_plan.records.at(
            static_cast<std::size_t>(stable_id));
        const auto &compatibility_context =
            sci_align::compatibility_context_window(record);
        const auto &compatibility_science =
            sci_align::compatibility_science_window(record);
        const Eigen::Index chunk_start = compatibility_context.start;
        const Eigen::Index chunk_size = compatibility_context.size();
        const Eigen::Index science_start = compatibility_science.start;
        const Eigen::Index science_size = compatibility_science.size();
        for (std::size_t interface_index = 0;
             interface_index < alignment.interfaces.size();
             ++interface_index) {
            const auto &interface = alignment.interfaces[interface_index];
            const auto &mask = alignment.masks[interface_index];
            // Admission is a property of the stable record's science window.
            // Expanded context may load endpoints and define guard/action
            // support, but must never dilute or inflate the >25% decision.
            const auto science_classification = classify_gap_mask_chunk(
                mask, science_start, science_size, 0);
            const auto context_classification = classify_gap_mask_chunk(
                mask, chunk_start, chunk_size, guard_context_samples,
                false);

            AlignmentChunkDisposition disposition;
            disposition.stable_scan_id = stable_id;
            disposition.compatibility_ordinal =
                record.compatibility_ordinal;
            disposition.interface_id = interface.interface_id;
            disposition.roach_index = interface.roach_index;
            disposition.context_start = compatibility_context.start;
            disposition.context_stop = compatibility_context.stop;
            disposition.cumulative_missing_count =
                science_classification.cumulative_missing;
            disposition.longest_missing_run_count =
                science_classification.longest_missing_run;
            disposition.full_network_unusable =
                science_classification.network_chunk_unusable;
            disposition.continuity_surrogate_permitted =
                continuity_permitted;

            const auto local_missing_runs = compact_binary_runs(
                context_classification.exact_missing, chunk_start);
            for (const auto &local_run : local_missing_runs) {
                Eigen::Index observation_run_start = local_run.start;
                while (observation_run_start > 0 &&
                       mask(observation_run_start - 1) == 0) {
                    --observation_run_start;
                }
                Eigen::Index observation_run_stop = local_run.stop;
                while (observation_run_stop < mask.size() &&
                       mask(observation_run_stop) == 0) {
                    ++observation_run_stop;
                }
                const bool bounded_internal =
                    observation_run_start > 0 &&
                    observation_run_stop < mask.size();
                if (!science_classification.network_chunk_unusable &&
                    continuity_permitted && bounded_internal) {
                    disposition.synthesized_missing_runs.push_back(
                        local_run);
                }
                else {
                    disposition.unavailable_missing_runs.push_back(
                        local_run);
                }
            }
            for (const auto &run :
                 disposition.synthesized_missing_runs) {
                add_to_compact_alignment_run_union(
                    synthesized_unique_support[interface_index], run);
            }
            std::uint64_t science_guarded_original_count = 0;
            std::uint64_t gap_policy_eligible_original_count = 0;
            if (!science_classification.network_chunk_unusable) {
                disposition.processing_guard_runs = compact_binary_runs(
                    context_classification.processing_guard, chunk_start);
                for (Eigen::Index sample = compatibility_science.start;
                     sample < compatibility_science.stop; ++sample) {
                    if (mask(sample) == 0) {
                        // A continuity surrogate is a processing convenience,
                        // never an acquired or gap-policy-eligible original.
                        continue;
                    }
                    const Eigen::Index context_local =
                        sample - chunk_start;
                    if (context_classification.processing_guard(
                            context_local) != 0) {
                        ++science_guarded_original_count;
                    }
                    else {
                        ++gap_policy_eligible_original_count;
                    }
                }
            }

            alignment.support.guarded_original_count =
                checked_alignment_count_add(
                    alignment.support.guarded_original_count,
                    science_guarded_original_count,
                    "ALIGN guarded-original count");
            alignment.support.gap_policy_eligible_original_count =
                checked_alignment_count_add(
                    alignment.support.gap_policy_eligible_original_count,
                    gap_policy_eligible_original_count,
                    "ALIGN science-eligible count");

            alignment.processing_support
                .synthesized_processing_occurrence_count =
                checked_alignment_count_add(
                    alignment.processing_support
                        .synthesized_processing_occurrence_count,
                    run_cardinality(disposition.synthesized_missing_runs),
                    "ALIGN synthesized processing occurrence count");
            alignment.processing_support
                .unavailable_processing_occurrence_count =
                checked_alignment_count_add(
                    alignment.processing_support
                        .unavailable_processing_occurrence_count,
                    run_cardinality(disposition.unavailable_missing_runs),
                    "ALIGN unavailable processing occurrence count");
            alignment.processing_support
                .guarded_original_processing_occurrence_count =
                checked_alignment_count_add(
                    alignment.processing_support
                        .guarded_original_processing_occurrence_count,
                    run_cardinality(disposition.processing_guard_runs),
                    "ALIGN guarded processing occurrence count");
            if (science_classification.network_chunk_unusable) {
                alignment.processing_support
                    .full_network_unusable_original_occurrence_count =
                    checked_alignment_count_add(
                        alignment.processing_support
                            .full_network_unusable_original_occurrence_count,
                        static_cast<std::uint64_t>(
                            chunk_size -
                            context_classification.cumulative_missing),
                        "ALIGN unusable original occurrence count");
            }
            if (alignment_chunk_disposition_is_exceptional(disposition)) {
                alignment.chunk_dispositions.push_back(
                    std::move(disposition));
            }
        }
    }
    std::sort(alignment.chunk_dispositions.begin(),
              alignment.chunk_dispositions.end(),
              alignment_chunk_disposition_key_less);
    alignment.support.synthesized_count = 0;
    for (const auto &support : synthesized_unique_support) {
        alignment.support.synthesized_count =
            checked_alignment_count_add(
                alignment.support.synthesized_count,
                compact_alignment_run_union_cardinality(support),
                "ALIGN unique synthesized count");
    }
    const auto interface_slot_capacity =
        checked_alignment_interface_slot_capacity(
            alignment.support.nominal_slot_count,
            alignment.interfaces.size());
    if (alignment.support.acquired_original_count >
            interface_slot_capacity ||
        alignment.support.synthesized_count >
            interface_slot_capacity -
                alignment.support.acquired_original_count) {
        throw citlali::error::runtime(
            "ALIGN support-origin counts exceed common-grid capacity");
    }
    alignment.support.unavailable_count =
        interface_slot_capacity -
        alignment.support.acquired_original_count -
        alignment.support.synthesized_count;
    alignment.processing_support.observation_resolved = true;
}

template <class RtcData, class Calib, class NetworkMasks, class ContextSamples,
          class Logger>
void apply_gap_masks_to_rtc_flags(
    RtcData &rtcdata, const Calib &calib, const NetworkMasks &nw_masks,
    Eigen::Index scan_start, ContextSamples context_samples,
    const Logger &logger) {
    for (const auto &[network_id, limits] : calib.nw_limits) {
        auto mask_it = nw_masks.find(network_id);
        if (mask_it == nw_masks.end()) {
            logger->error(
                "missing gap mask for nw {}; cannot apply gap flagging",
                network_id);
            throw citlali::error::runtime(fmt::format(
                "missing gap mask for network {}; cannot apply gap flagging",
                network_id));
        }
        const auto &mask = mask_it->second;

        const Eigen::Index start = std::get<0>(limits);
        const Eigen::Index end = std::get<1>(limits) - 1;
        if (start < 0 || end < start ||
            end >= rtcdata.flags.data.cols()) {
            throw citlali::error::runtime(fmt::format(
                "invalid detector limits for gap-mask network {}",
                network_id));
        }

        const auto classification = classify_gap_mask_chunk(
            mask, scan_start, rtcdata.flags.data.rows(), context_samples);
        if (classification.network_chunk_unusable) {
            rtcdata.flags.data
                .block(0, start, rtcdata.flags.data.rows(), end - start + 1)
                .setOnes();
        } else {
            for (Eigen::Index row = 0;
                 row < rtcdata.flags.data.rows(); ++row) {
                if (classification.exact_missing(row) == 0 &&
                    classification.processing_guard(row) == 0) {
                    continue;
                }
                rtcdata.flags.data
                    .block(row, start, 1, end - start + 1)
                    .setOnes();
            }
        }
        logger->debug("{}/{} gaps flagged",
                      rtcdata.flags.data.col(start)
                          .template cast<std::int64_t>()
                          .sum(),
                      rtcdata.flags.data.rows());
    }
}

template <class RtcData, class Calib, class Logger>
void apply_planned_gap_dispositions_to_rtc_flags(
    RtcData &rtcdata, const Calib &calib,
    const TimestreamAlignmentState &alignment,
    Eigen::Index compatibility_ordinal, Eigen::Index scan_start,
    const Logger &logger) {
    const Eigen::Index chunk_size = rtcdata.flags.data.rows();
    if (scan_start < 0 || chunk_size < 0 ||
        chunk_size >
            std::numeric_limits<Eigen::Index>::max() - scan_start) {
        throw citlali::error::runtime(
            "planned ALIGN gap context exceeds Eigen index range");
    }
    const Eigen::Index chunk_stop = scan_start + chunk_size;
    auto flag_run = [&](const AlignmentIndexRun &run,
                        Eigen::Index detector_start,
                        Eigen::Index detector_count) {
        if (run.start < scan_start || run.stop > chunk_stop ||
            run.stop <= run.start) {
            throw citlali::error::runtime(
                "planned ALIGN gap run is outside its processing context");
        }
        rtcdata.flags.data
            .block(run.start - scan_start, detector_start,
                   run.stop - run.start, detector_count)
            .setOnes();
    };

    for (const auto &[network_id, limits] : calib.nw_limits) {
        const Eigen::Index detector_start = std::get<0>(limits);
        const Eigen::Index detector_stop = std::get<1>(limits);
        if (detector_start < 0 || detector_stop <= detector_start ||
            detector_stop > rtcdata.flags.data.cols()) {
            throw citlali::error::runtime(fmt::format(
                "invalid detector limits for gap-mask network {}",
                network_id));
        }
        const auto *disposition = find_alignment_chunk_disposition(
            alignment, compatibility_ordinal, network_id);
        if (disposition == nullptr) {
            logger->debug(
                "0/{} planned ALIGN gap samples flagged for nw {}",
                chunk_size, network_id);
            continue;
        }
        if (disposition->context_start != scan_start ||
            disposition->context_stop != chunk_stop) {
            throw citlali::error::runtime(fmt::format(
                "ALIGN gap disposition/context mismatch for scan {} network {}",
                compatibility_ordinal, network_id));
        }
        const Eigen::Index detector_count =
            detector_stop - detector_start;
        if (disposition->full_network_unusable) {
            rtcdata.flags.data
                .block(0, detector_start, chunk_size, detector_count)
                .setOnes();
        }
        else {
            for (const auto &run :
                 disposition->synthesized_missing_runs) {
                flag_run(run, detector_start, detector_count);
            }
            for (const auto &run :
                 disposition->unavailable_missing_runs) {
                flag_run(run, detector_start, detector_count);
            }
            for (const auto &run : disposition->processing_guard_runs) {
                flag_run(run, detector_start, detector_count);
            }
        }
        logger->debug(
            "{}/{} planned ALIGN gap samples flagged for nw {}",
            rtcdata.flags.data.col(detector_start)
                .template cast<std::int64_t>()
                .sum(),
            chunk_size, network_id);
    }
}

template <class Engine, class RtcData>
RtcScanSampleWindow prepare_standard_rtc_scan_context(
    Engine &engine, RtcData &rtcdata) {
    const auto scan_window = copy_rtc_scan_context(
        rtcdata, engine.telescope, engine.pointing_offsets.arcsec);
    copy_hwpr_angle_if_enabled(
        rtcdata, engine.calib, engine.rtcproc.run_polarization,
        engine.calib.run_hwpr, engine.alignment.hwpr_start_index,
        scan_window.start, scan_window.length);
    initialize_rtc_flags(rtcdata);
    if (citlali::config::timing_gap_interpolation_active(
            effective_runtime_values(engine))) {
        apply_planned_gap_dispositions_to_rtc_flags(
            rtcdata, engine.calib, engine.alignment,
            rtcdata.index.data, scan_window.start, engine.logger);
    }
    return scan_window;
}

}  // namespace citlali::pipeline
