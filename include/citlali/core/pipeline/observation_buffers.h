#pragma once

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/mapmaking/science_map_contract.h>
#include <citlali/core/pipeline/observation_buffer_allocation.h>
#include <citlali/core/pipeline/observation_map_access.h>
#include <citlali/core/pipeline/mapmaking_provenance_lifecycle.h>
#include <citlali/core/pipeline/native_cohort_product_provenance.h>
#include <citlali/core/pipeline/pointing_provenance_lifecycle.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace citlali::pipeline {

template <class Engine>
inline constexpr bool has_native_cohort_observation_sources_v =
    has_raw_timestream_plan_v<Engine> && requires(Engine &engine) {
        engine.calib.apt_detector_relation_handle();
        engine.alignment.native_consumer_plan;
        engine.alignment.native_pointing_plan;
        engine.telescope.scan_indices.cols();
    };

inline bool native_cohort_observation_bindings_equal(
    const NativeCohortObservationBinding &lhs,
    const NativeCohortObservationBinding &rhs) {
    return lhs.observation_index == rhs.observation_index &&
           lhs.raw_observation == rhs.raw_observation &&
           lhs.artifact_scope == rhs.artifact_scope &&
           lhs.detector_relation_digest == rhs.detector_relation_digest &&
           lhs.raw_manifest_digest == rhs.raw_manifest_digest &&
           lhs.alignment_plan_digest == rhs.alignment_plan_digest &&
           lhs.pointing_plan_digest == rhs.pointing_plan_digest;
}

template <class Engine>
bool native_cohort_observation_sources_active(const Engine &engine) {
    if constexpr (has_native_cohort_observation_sources_v<Engine>) {
        return static_cast<bool>(
                   engine.calib.apt_detector_relation_handle()) ||
               static_cast<bool>(engine.alignment.native_consumer_plan) ||
               static_cast<bool>(engine.alignment.native_pointing_plan);
    }
    return false;
}

template <class Engine>
NativeCohortObservationBinding native_cohort_observation_binding_from_engine(
    const Engine &engine, std::size_t observation_index) {
    static_assert(has_native_cohort_observation_sources_v<Engine>);
    const auto relation = engine.calib.apt_detector_relation_handle();
    const auto &alignment = engine.alignment.native_consumer_plan;
    const auto &pointing = engine.alignment.native_pointing_plan;
    if (!relation || !alignment || !pointing) {
        throw std::logic_error(
            "native cohort observation activation requires typed APT, alignment, and pointing authorities");
    }
    return make_native_cohort_observation_binding(
        observation_index, *relation, alignment, pointing);
}

template <bool EnableNativeCohortLineage = true, class Engine>
void begin_native_cohort_observation_if_available(
    Engine &engine, std::size_t observation_index) {
    if constexpr (!EnableNativeCohortLineage) {
        // Beammap is an APT producer, not an observation-matched APT
        // consumer.  Until its producer-specific lineage is introduced in
        // the BEAM development cycle, native alignment/pointing may be used
        // without activating the consumer lineage that requires an input
        // typed-APT authority.
        return;
    }
    if constexpr (has_native_cohort_observation_sources_v<Engine>) {
        if (!native_cohort_observation_sources_active(engine)) {
            return;
        }

        auto &plan = raw_timestream_plan(engine);
        if (!plan.initialized || !plan.observation) {
            throw std::logic_error(
                "native cohort observation activation requires an active raw timestream observation");
        }
        if (plan.observation->native_cohort_lineage ||
            plan.realized.native_cohort_provenance) {
            throw std::logic_error(
                "native cohort observation lineage is already active or realized");
        }
        const auto scan_count = engine.telescope.scan_indices.cols();
        if (scan_count <= 0) {
            throw std::logic_error(
                "native cohort observation requires positive telescope scan cardinality");
        }

        auto candidate = NativeCohortObservationLineage::create(
            native_cohort_observation_binding_from_engine(
                engine, observation_index),
            static_cast<std::size_t>(scan_count));
        plan.observation->native_cohort_lineage = std::move(candidate);
    }
}

template <class Engine>
std::optional<NativeCohortObservationLineage::Reservation>
prepare_native_cohort_scan_provenance_if_available(
    Engine &engine, std::size_t observation_index,
    NativeCohortScanProvenance record) {
    if constexpr (has_native_cohort_observation_sources_v<Engine>) {
        auto &plan = raw_timestream_plan(engine);
        const bool sources_active =
            native_cohort_observation_sources_active(engine);
        if (!sources_active) {
            if (plan.observation &&
                plan.observation->native_cohort_lineage) {
                throw std::logic_error(
                    "native cohort observation authorities disappeared after activation");
            }
            return std::nullopt;
        }
        if (!plan.initialized || !plan.observation ||
            !plan.observation->native_cohort_lineage) {
            throw std::logic_error(
                "native cohort scan cannot run before observation lineage activation");
        }
        if (plan.realized.execution_completed ||
            plan.realized.native_cohort_provenance) {
            throw std::logic_error(
                "native cohort scan cannot mutate completed observation lineage");
        }

        const auto &lineage = plan.observation->native_cohort_lineage;
        const auto current_binding =
            native_cohort_observation_binding_from_engine(
                engine, observation_index);
        if (!native_cohort_observation_bindings_equal(
                lineage->binding(), current_binding) ||
            lineage->scan_count() != static_cast<std::size_t>(
                                         engine.telescope.scan_indices.cols())) {
            throw std::logic_error(
                "native cohort scan authorities or telescope scan cardinality changed after activation");
        }
        return lineage->reserve(std::move(record));
    }
    return std::nullopt;
}

template <class Engine>
std::optional<NativeCohortObservationLineage::Reservation>
prepare_native_cohort_scan_provenance_if_available(
    Engine &engine, NativeCohortScanProvenance record) {
    std::size_t observation_index = 0;
    if constexpr (has_native_cohort_observation_sources_v<Engine>) {
        const auto &plan = raw_timestream_plan(engine);
        if (plan.observation && plan.observation->native_cohort_lineage) {
            observation_index = plan.observation->native_cohort_lineage
                                    ->binding()
                                    .observation_index;
        }
    }
    return prepare_native_cohort_scan_provenance_if_available(
        engine, observation_index, std::move(record));
}

template <class NativeScan>
TimestreamNativeRevision native_cohort_scan_uniform_revision(
    const NativeScan &scan) {
    if (!scan.is_processed_projection() || scan.row_count() <= 0 ||
        scan.detector_count() <= 0) {
        throw std::logic_error(
            "native cohort provenance requires a processed measured scan projection");
    }
    const auto revision = scan.require_cell(0, 0).expected_revision;
    for (Eigen::Index row = 0; row < scan.row_count(); ++row) {
        for (Eigen::Index column = 0; column < scan.detector_count(); ++column) {
            if (scan.require_cell(row, column).expected_revision != revision) {
                throw std::logic_error(
                    "native cohort scan contains mixed transactional revisions");
            }
        }
    }
    return revision;
}

template <class NativeScan>
NativeCohortScanProvenance make_native_cohort_scan_provenance(
    const NativeCohortObservationBinding &binding, const NativeScan &scan,
    TimestreamNativeRevision input_revision) {
    const auto output_revision = native_cohort_scan_uniform_revision(scan);
    if (output_revision < input_revision) {
        throw std::logic_error(
            "native cohort output revision precedes its PTC input revision");
    }
    const auto &rtc_rows = scan.rtc_output_rows();
    if (rtc_rows.size() != static_cast<std::size_t>(scan.row_count())) {
        throw std::logic_error(
            "native cohort final scan lacks exact RTC row support");
    }

    NativeCohortScanProvenance record;
    record.observation_binding_digest = binding.digest();
    record.operation = scan.operation();
    record.input_revision = input_revision;
    record.output_revision = output_revision;
    record.rows.reserve(rtc_rows.size());
    for (const auto &rtc_row : rtc_rows) {
        if (rtc_row.output_row !=
            static_cast<Eigen::Index>(record.rows.size())) {
            throw std::logic_error(
                "native cohort final output rows are not deterministic");
        }
        NativeCohortOutputRow row;
        row.output_row = rtc_row.output_row;
        row.relational_common_slot = rtc_row.relational_common_slot;
        row.participant_support = rtc_row.participant_support;
        row.participants.reserve(row.participant_support.size());
        for (const auto &support : row.participant_support) {
            for (const auto detector_column : support.detector_columns) {
                const auto cell =
                    scan.require_cell(row.output_row, detector_column);
                if (!(cell.identity == support.selected_anchor) ||
                    cell.expected_revision != output_revision) {
                    throw std::logic_error(
                        "native cohort final scan differs from its exact measured anchor or revision");
                }
            }
            row.participants.push_back(NativeCohortParticipantRow{
                support.selected_anchor, input_revision, output_revision,
                CoincidenceCellState::mapped_valid});
        }
        record.rows.push_back(std::move(row));
    }
    return record;
}

template <class MapIndices>
std::vector<Eigen::Index> native_cohort_ordered_map_indices(
    const MapIndices &map_indices, Eigen::Index detector_count,
    Eigen::Index map_count) {
    if (map_indices.size() != detector_count || map_count <= 0) {
        throw std::logic_error(
            "native cohort map join has incompatible detector or map cardinality");
    }
    std::vector<Eigen::Index> result;
    result.reserve(static_cast<std::size_t>(map_indices.size()));
    for (Eigen::Index column = 0; column < map_indices.size(); ++column) {
        const auto map_index = map_indices(column);
        if (map_index < 0 || map_index >= map_count) {
            throw std::logic_error(
                "native cohort map join references an invalid map index");
        }
        result.push_back(map_index);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

template <class MapIndices>
std::string native_cohort_eligible_input_digest(
    const NativeCohortScanProvenance &record,
    const MapIndices &map_indices) {
    NativeCohortDigestBuilder digest;
    digest.add("schema", "citlali-native-cohort-map-eligible-input-v1");
    digest.add("observation-binding", record.observation_binding_digest);
    digest.add_integer("operation.sequence", record.operation.sequence);
    digest.add_integer("operation.scan", record.operation.scan_index);
    digest.add_integer("input.revision", record.input_revision);
    digest.add_integer("output.revision", record.output_revision);
    for (const auto &row : record.rows) {
        digest.add_integer("row.output", row.output_row);
        digest.add_integer("row.common-slot", row.relational_common_slot);
        for (const auto &support : row.participant_support) {
            digest.add_integer("support.run", support.run_ordinal);
            digest.add_integer("support.run-output-row",
                               support.run_output_row);
            digest.add_integer("support.network",
                               support.selected_anchor.network_id());
            digest.add_integer("support.anchor-row",
                               support.selected_anchor.native_row());
            digest.add_double(
                "support.anchor-time",
                support.selected_anchor.reconstructed_time_unix_sec());
            digest.add_integer("support.factor", support.factor);
            digest.add_integer("support.first-row",
                               support.first_support_native_row);
            digest.add_integer("support.past-row",
                               support.past_last_support_native_row);
            digest.add_integer("support.final-short",
                               support.final_short_support);
            for (const auto &identity : support.exact_support_rows) {
                digest.add_integer("support.row.network",
                                   identity.network_id());
                digest.add_integer("support.row.native",
                                   identity.native_row());
                digest.add_double("support.row.time",
                                  identity.reconstructed_time_unix_sec());
            }
            for (std::size_t column = 0;
                 column < support.detector_columns.size(); ++column) {
                digest.add_integer("support.detector",
                                   support.detector_columns[column]);
                digest.add_integer("support.flags",
                                   support.ored_flag_support[column]);
            }
        }
    }
    for (Eigen::Index column = 0; column < map_indices.size(); ++column) {
        digest.add_integer("map.detector-column", column);
        digest.add_integer("map.index", map_indices(column));
    }
    return digest.finish();
}

template <class JincTrace>
std::string native_cohort_jinc_scan_trace_digest(const JincTrace &trace) {
    NativeCohortDigestBuilder digest;
    digest.add("schema", "citlali-native-cohort-jinc-scan-trace-v1");
    digest.add_integer("detectors", trace.detector_count);
    digest.add_integer("detector-samples", trace.detector_sample_count);
    digest.add_integer("rtc-flagged", trace.rtc_flagged_sample_count);
    digest.add_integer("ptc-flagged", trace.ptc_flagged_sample_count);
    digest.add_integer("apt-flagged", trace.apt_flagged_detector_count);
    digest.add_integer("rtc-source-masked",
                       trace.rtc_source_masked_sample_count);
    digest.add_integer("ptc-mean-masked",
                       trace.ptc_mean_masked_sample_count);
    digest.add_integer("pca-solves", trace.pca_solve_count);
    digest.add_integer("configured-notches",
                       trace.configured_notch_applied_count);
    digest.add_integer("fixed-notches", trace.fixed_notch_count);
    digest.add_integer("dynamic-notches", trace.dynamic_notch_count);
    digest.add_integer("detector-notches", trace.detector_notch_count);
    digest.add("rtc-flags", trace.rtc_flags_digest);
    digest.add("ptc-flags", trace.ptc_flags_digest);
    digest.add("apt-flags", trace.apt_flags_digest);
    digest.add("map-indices", trace.map_indices_digest);
    digest.add("ptc-signal", trace.ptc_signal_digest);
    digest.add("ptc-kernel", trace.ptc_kernel_digest);
    digest.add("rtc-source-mask", trace.rtc_source_mask_digest);
    digest.add("rtc-notch-operators", trace.rtc_notch_operators_digest);
    digest.add("ptc-mean-mask", trace.ptc_mean_mask_digest);
    digest.add("pca", trace.pca_realization_digest);
    return digest.finish();
}

template <class Engine, class NativeScan, class MapIndices>
NativeCohortScanProvenance make_native_cohort_scan_provenance_for_map(
    Engine &engine, const NativeScan &scan,
    TimestreamNativeRevision input_revision,
    const MapIndices &map_indices, citlali::config::MapMethod method,
    bool mapmaking_enabled) {
    auto &plan = raw_timestream_plan(engine);
    if (!plan.observation || !plan.observation->native_cohort_lineage) {
        throw std::logic_error(
            "native cohort scan record requires active observation lineage");
    }
    auto record = make_native_cohort_scan_provenance(
        plan.observation->native_cohort_lineage->binding(), scan,
        input_revision);
    if (!mapmaking_enabled) {
        return record;
    }
    if (method == citlali::config::MapMethod::maximum_likelihood) {
        throw std::logic_error(
            "native cohort product lineage does not admit maximum-likelihood mapmaking");
    }

    record.map_join.mapmaking_enabled = true;
    record.map_join.method = std::string{citlali::config::to_string(method)};
    record.map_join.eligible_input_digest =
        native_cohort_eligible_input_digest(record, map_indices);
    record.map_join.ordered_map_indices = native_cohort_ordered_map_indices(
        map_indices, scan.detector_count(), engine.map_indices.n_maps);
    if (method == citlali::config::MapMethod::naive) {
        const auto &identity = engine.omb.science_products.bundle_identity;
        if (!identity) {
            throw std::logic_error(
                "native cohort naive map join lacks admitted science-map identity");
        }
        record.map_join.product_identity_digest =
            mapmaking::science_map_bundle_identity_digest(*identity);
        return record;
    }

    auto &products = engine.omb.jinc_products;
    if (!products.initialized || !products.provenance.available ||
        !products.provenance.processing_configuration_bound ||
        products.provenance.effective_digest.empty() ||
        products.provenance.processing_configuration_identity ==
            "unavailable") {
        throw std::logic_error(
            "native cohort JINC map join lacks an admitted product configuration");
    }
    record.map_join.product_identity_digest =
        products.provenance.effective_digest;
    record.map_join.jinc_processing_configuration_digest =
        products.provenance.processing_configuration_identity;
    {
        std::scoped_lock<std::mutex> lock(*products.processing_trace_mutex);
        const auto trace = products.processing_scan_traces.find(
            scan.operation().scan_index);
        if (trace == products.processing_scan_traces.end()) {
            throw std::logic_error(
                "native cohort JINC map join lacks its completed scan trace");
        }
        record.map_join.jinc_scan_trace_digest =
            native_cohort_jinc_scan_trace_digest(trace->second);
    }
    return record;
}

inline void commit_native_cohort_scan_provenance(
    std::optional<NativeCohortObservationLineage::Reservation> &reservation)
    noexcept {
    if (reservation) {
        reservation->commit();
        reservation.reset();
    }
}

template <bool EnableNativeCohortLineage = true, class TodProc,
          class MapExtents, class MapCoords, class Logger>
void allocate_observation_map_buffers_if_needed(
    TodProc &todproc, MapExtents &map_extents, MapCoords &map_coords,
    std::size_t observation_index, const Logger &logger) {
    auto &engine = todproc.engine();

    if (!should_allocate_observation_map_buffers(engine)) {
        begin_native_cohort_observation_if_available<
            EnableNativeCohortLineage>(engine, observation_index);
        return;
    }

    allocate_observation_map_buffers(
        todproc, observation_map_extent_at(map_extents, observation_index),
        observation_map_coord_at(map_coords, observation_index),
        logger);
    begin_mapmaking_observation_if_available(engine, observation_index);
    begin_pointing_observation_if_available(engine);
    begin_native_cohort_observation_if_available<EnableNativeCohortLineage>(
        engine, observation_index);
}

template <bool EnableNativeCohortLineage = true, class TodProc,
          class MapExtents, class MapCoords, class Logger>
void allocate_reduction_observation_map_buffers_if_needed(
    TodProc &todproc, MapExtents &map_extents, MapCoords &map_coords,
    std::size_t observation_index, const Logger &logger) {
    allocate_observation_map_buffers_if_needed<EnableNativeCohortLineage>(
        todproc, map_extents, map_coords, observation_index, logger);
}

}  // namespace citlali::pipeline
