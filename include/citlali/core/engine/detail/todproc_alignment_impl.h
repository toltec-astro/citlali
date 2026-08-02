#pragma once

// Implementation detail included by todproc.h.

#include <citlali/core/engine/detail/sci_align_netcdf_input_contract.h>
#include <citlali/core/engine/detail/sci_align_packet_slot_contract.h>
#include <citlali/core/pipeline/interface_sync_config_adapter.h>
#include <citlali/core/pipeline/observation_setup_validation.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>
#include <citlali/core/pipeline/sci_align_contract.h>
#include <citlali/core/pipeline/sci_align_field_registry.h>
#include <citlali/core/pipeline/timestream_alignment_helpers.h>
#include <citlali/core/pipeline/timestream_alignment_state.h>

#include <netcdf>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace citlali::engine_detail {

struct PreparedDetectorAlignment {
    citlali::pipeline::sci_align::DetectorLattice lattice;
    double raw_overlap_end_sec = 0.0;
    Eigen::VectorXd common_time;
    std::vector<Eigen::VectorXd> network_times;
    std::vector<Eigen::VectorXi> masks;
    std::vector<Eigen::Index> network_ids;
    std::vector<std::vector<citlali::pipeline::sci_align::PacketGap>>
        packet_gaps;
};

inline Eigen::VectorXd common_time_from_lattice(
    const citlali::pipeline::sci_align::DetectorLattice &lattice) {
    const auto count = lattice.slot_count();
    if (count == 0 || count > static_cast<std::uint64_t>(
                                std::numeric_limits<Eigen::Index>::max())) {
        throw std::runtime_error("detector union lattice has invalid size");
    }
    Eigen::VectorXd result(static_cast<Eigen::Index>(count));
    for (Eigen::Index local = 0; local < result.size(); ++local) {
        result(local) = lattice.time_for_global_slot(
            lattice.first_global_slot + static_cast<std::int64_t>(local));
    }
    citlali::pipeline::sci_align::require_strictly_increasing(
        result, "detector common lattice");
    return result;
}

inline std::vector<Eigen::VectorXi> masks_from_lattice(
    const citlali::pipeline::sci_align::DetectorLattice &lattice,
    const std::vector<
        citlali::pipeline::sci_align::DetectorInterfaceCoordinates>
        &coordinates) {
    const auto count = lattice.slot_count();
    if (coordinates.size() != lattice.interfaces.size()) {
        throw std::logic_error(
            "detector coordinate/mapping cardinality mismatch");
    }
    std::vector<Eigen::VectorXi> result;
    result.reserve(lattice.interfaces.size());
    for (std::size_t interface_index = 0;
         interface_index < lattice.interfaces.size(); ++interface_index) {
        const auto &times =
            coordinates[interface_index].corrected_time.seconds;
        Eigen::VectorXi mask = Eigen::VectorXi::Zero(
            static_cast<Eigen::Index>(count));
        for (Eigen::Index row = 0; row < times.size(); ++row) {
            const auto global_slot =
                citlali::pipeline::sci_align::round_half_up_slot(
                    (times(row) - lattice.phase_seconds) /
                    lattice.cadence_seconds);
            const auto local64 = global_slot -
                                 lattice.first_global_slot;
            if (local64 < 0 ||
                local64 >= static_cast<std::int64_t>(count)) {
                throw std::logic_error(
                    "detector assignment lies outside union lattice");
            }
            const auto local = static_cast<Eigen::Index>(local64);
            if (mask(local) != 0) {
                throw std::logic_error(
                    "detector assignment collision survived lattice validation");
            }
            mask(local) = 1;
        }
        result.push_back(std::move(mask));
    }
    return result;
}

inline void append_missing_runs(
    citlali::pipeline::TimestreamAlignmentState &state,
    const std::string &interface_id, const Eigen::VectorXi &mask) {
    Eigen::Index index = 0;
    while (index < mask.size()) {
        while (index < mask.size() && mask(index) != 0) {
            ++index;
        }
        if (index == mask.size()) {
            break;
        }
        const Eigen::Index start = index;
        while (index < mask.size() && mask(index) == 0) {
            ++index;
        }
        const bool edge = start == 0 || index == mask.size();
        state.exceptions.push_back({
            interface_id,
            "detector_acquisition",
            start,
            index,
            "unavailable",
            "unavailable",
            edge ? "none" : "bounded_continuity_candidate",
            edge ? "union_edge_no_extrapolation" : "packet_gap",
            edge ? -1 : start - 1,
            edge ? -1 : index,
        });
    }
}

inline void populate_compact_alignment_state(
    citlali::pipeline::TimestreamAlignmentState &state,
    const PreparedDetectorAlignment &prepared) {
    const auto &lattice = prepared.lattice;
    state.grid.initialized = true;
    state.grid.phase_sec = lattice.phase_seconds;
    state.grid.cadence_sec = lattice.cadence_seconds;
    state.grid.exclusive_half_cell_sec =
        lattice.exclusive_half_cell_seconds;
    state.grid.first_global_slot = lattice.first_global_slot;
    state.grid.last_global_slot = lattice.last_global_slot;
    state.governing_compatibility_axis =
        citlali::pipeline::make_governing_gap_compatibility_axis(
            state.grid, prepared.raw_overlap_end_sec);
    state.field_registry_version =
        std::string{citlali::pipeline::sci_align::active_field_registry_version};
    state.support.nominal_slot_count = lattice.slot_count();
    state.support.nominal_span_sec =
        static_cast<double>(lattice.slot_count()) * lattice.cadence_seconds;

    for (std::size_t i = 0; i < lattice.interfaces.size(); ++i) {
        const auto &mapping = lattice.interfaces[i];
        citlali::pipeline::AlignmentInterfaceSummary summary;
        summary.interface_id = mapping.interface_id;
        summary.roach_index = prepared.network_ids[i];
        summary.native_row_count = mapping.native_row_count;
        summary.accepted_row_count = summary.native_row_count;
        summary.first_global_slot = mapping.first_global_slot;
        summary.last_global_slot = mapping.last_global_slot;
        summary.minimum_residual_sec = mapping.minimum_residual_seconds;
        summary.maximum_residual_sec = mapping.maximum_residual_seconds;
        summary.maximum_absolute_residual_sec =
            mapping.maximum_absolute_residual_seconds;
        if (mapping.leading_unavailable) {
            summary.leading_unavailable_count = static_cast<std::int64_t>(
                mapping.leading_unavailable->size());
        }
        if (mapping.trailing_unavailable) {
            summary.trailing_unavailable_count = static_cast<std::int64_t>(
                mapping.trailing_unavailable->size());
        }
        state.interfaces.push_back(summary);
        const auto native_row_count =
            static_cast<std::uint64_t>(mapping.native_row_count);
        state.support.acquired_original_count =
            citlali::pipeline::checked_alignment_count_add(
                state.support.acquired_original_count, native_row_count,
                "ALIGN acquired-original count");
        state.support.timing_coordinate_valid_original_count =
            citlali::pipeline::checked_alignment_count_add(
                state.support.timing_coordinate_valid_original_count,
                native_row_count,
                "ALIGN timing-coordinate-valid original count");
        state.support.unavailable_count =
            citlali::pipeline::checked_alignment_count_add(
                state.support.unavailable_count,
                lattice.slot_count() - native_row_count,
                "ALIGN unavailable interface-slot count");
        append_missing_runs(state, mapping.interface_id, prepared.masks[i]);
    }
    state.support.acquired_original_cadence_weighted_support_sec =
        static_cast<double>(state.support.acquired_original_count) *
        lattice.cadence_seconds;
    if (!std::isfinite(
            state.support
                .acquired_original_cadence_weighted_support_sec)) {
        throw std::overflow_error(
            "ALIGN cadence-weighted acquired support overflows");
    }
}

template <class Engine>
void copy_interface_lifecycle_to_raw_plan(Engine &engine) {
    if constexpr (citlali::pipeline::has_raw_timestream_plan_v<Engine>) {
        auto &plan = citlali::pipeline::raw_timestream_plan(engine);
        if (plan.observation.has_value()) {
            plan.observation->interface_offsets =
                engine.interface_sync.lifecycle;
        }
    }
}

template <class Engine, class RawObs>
PreparedDetectorAlignment prepare_detector_alignment(
    Engine &engine, const RawObs &rawobs) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    citlali::pipeline::reset_alignment_observation_state(engine.alignment);
    engine.alignment.hwpr =
        citlali::pipeline::bounded_nonpolarimetric_hwpr_summary(
            citlali::pipeline::observation_hwpr_input_present(rawobs));
    if (engine.interface_sync.lifecycle.empty()) {
        citlali::pipeline::adapt_interface_sync_config_one_way(
            citlali::config::InterfaceSyncOffsetConfig{},
            engine.interface_sync);
    }
    citlali::pipeline::begin_interface_sync_observation(
        engine.interface_sync);

    if (engine.calib.run_hwpr) {
        throw std::runtime_error(
            "enabled HWPR alignment is unavailable in the bounded nonpolarimetric SCI-ALIGN-001 profile");
    }

    const auto kids_data = rawobs.kidsdata();
    if (kids_data.empty()) {
        throw std::runtime_error(
            "no detector interfaces are available for alignment");
    }

    PreparedDetectorAlignment prepared;
    std::vector<citlali::pipeline::sci_align::DetectorInterfaceCoordinates>
        coordinates;
    coordinates.reserve(kids_data.size());
    std::set<std::string> supplied_identities;
    std::set<int> raw_roach_ids;

    for (const auto &data_item_ref : kids_data) {
        const auto &data_item = data_item_ref.get();
        const std::string interface_id = data_item.interface();
        const int supplied_id =
            citlali::pipeline::sci_align::parse_toltec_interface_identity(
                interface_id);
        if (!supplied_identities.insert(interface_id).second) {
            throw std::runtime_error(
                "duplicate selected detector interface " + interface_id);
        }

        try {
            NcFile file(data_item.filepath(), NcFile::read);
            namespace nc_contract =
                citlali::engine_detail::sci_align_netcdf;

            const auto roach_index_var = nc_contract::require_variable(
                file, "Header.Toltec.RoachIndex");
            nc_contract::require_scalar(
                roach_index_var, "Header.Toltec.RoachIndex");
            nc_contract::require_type(
                roach_index_var, "Header.Toltec.RoachIndex",
                {NcType::nc_INT});
            int roach_index = -1;
            roach_index_var.getVar(&roach_index);
            if (roach_index != supplied_id || roach_index < 0 ||
                roach_index >= static_cast<int>(
                    citlali::config::toltec_interface_count)) {
                throw std::runtime_error(fmt::format(
                    "selected interface '{}' conflicts with Header.Toltec.RoachIndex={}",
                    interface_id, roach_index));
            }
            if (!raw_roach_ids.insert(roach_index).second) {
                throw std::runtime_error(fmt::format(
                    "multiple detector inputs claim Header.Toltec.RoachIndex={}",
                    roach_index));
            }

            double fpga_frequency_hz = 0.0;
            double sample_frequency_hz = 0.0;
            int accumulation_length_ticks = 0;
            const auto fpga_frequency_var = nc_contract::require_variable(
                file, "Header.Toltec.FpgaFreq");
            const auto accumulation_length_var =
                nc_contract::require_variable(file,
                                              "Header.Toltec.AccumLen");
            const auto sample_frequency_var = nc_contract::require_variable(
                file, "Header.Toltec.SampleFreq");
            for (const auto *entry : {
                     &fpga_frequency_var, &accumulation_length_var,
                     &sample_frequency_var}) {
                nc_contract::require_scalar(*entry, entry->getName());
            }
            nc_contract::require_type(
                fpga_frequency_var, "Header.Toltec.FpgaFreq",
                {NcType::nc_DOUBLE});
            nc_contract::require_type(
                accumulation_length_var, "Header.Toltec.AccumLen",
                {NcType::nc_INT});
            nc_contract::require_type(
                sample_frequency_var, "Header.Toltec.SampleFreq",
                {NcType::nc_DOUBLE});
            nc_contract::require_units(
                fpga_frequency_var, "Header.Toltec.FpgaFreq", {"Hz"});
            nc_contract::require_units(
                sample_frequency_var, "Header.Toltec.SampleFreq", {"Hz"});
            fpga_frequency_var.getVar(&fpga_frequency_hz);
            accumulation_length_var.getVar(&accumulation_length_ticks);
            sample_frequency_var.getVar(&sample_frequency_hz);
            const citlali::pipeline::sci_align::NativeTimingHeader header{
                fpga_frequency_hz,
                static_cast<double>(accumulation_length_ticks),
                sample_frequency_hz,
            };
            (void)citlali::pipeline::sci_align::
                validate_bounded_production_native_timing_header(header);

            const auto is_var = nc_contract::require_variable(
                file, "Data.Toltec.Is");
            const auto qs_var = nc_contract::require_variable(
                file, "Data.Toltec.Qs");
            const auto ts_var = nc_contract::require_variable(
                file, "Data.Toltec.Ts");
            nc_contract::require_legacy_toltec_timing_schema(
                ts_var, is_var, qs_var);
            const auto ts_dims = ts_var.getDims();
            const auto native_count = nc_contract::require_eigen_matrix_rows(
                ts_dims[0].getSize(), 6, "Data.Toltec.Ts");
            Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor>
                ts(native_count, 6);
            ts_var.getVar(ts.data());
            Eigen::MatrixXd ts_double = ts.cast<double>();
            std::vector<std::int64_t> packet_counts(
                static_cast<std::size_t>(native_count));
            for (Eigen::Index row = 0; row < native_count; ++row) {
                packet_counts[static_cast<std::size_t>(row)] = ts(row, 3);
            }
            auto reconstructed =
                citlali::pipeline::sci_align::
                    reconstruct_legacy_detector_timestamps(
                        ts_double, packet_counts, fpga_frequency_hz);

            const double realized_offset =
                citlali::pipeline::realize_interface_offset(
                    engine.interface_sync, interface_id, false);
            citlali::pipeline::sci_align::ClockCoordinates native{
                std::move(reconstructed.seconds),
                citlali::pipeline::sci_align::ClockCoordinateStage::
                    native_legacy,
            };
            citlali::pipeline::sci_align::InterfaceOffset offset;
            offset.seconds = realized_offset;
            offset.authority_resolved = realized_offset == 0.0;
            offset.source = "bounded_offset_lifecycle";
            offset.reference_interface = "detector_clock";
            auto corrected =
                citlali::pipeline::sci_align::apply_interface_offset_once(
                    std::move(native), offset);

            prepared.network_ids.push_back(roach_index);
            prepared.packet_gaps.push_back(
                std::move(reconstructed.packet_gaps));
            coordinates.push_back({interface_id, header,
                                   std::move(corrected)});
            file.close();
        }
        catch (const DataIOError &error) {
            throw DataIOError{fmt::format(
                "failed detector alignment input contract for '{}': {}",
                data_item.filepath(), error.what())};
        }
        catch (const NcException &error) {
            throw std::runtime_error(fmt::format(
                "failed to read detector alignment input '{}': {}",
                data_item.filepath(), error.what()));
        }
    }

    for (auto &record : engine.interface_sync.lifecycle) {
        if (record.interface_id.rfind("toltec", 0) == 0 &&
            supplied_identities.find(record.interface_id) ==
                supplied_identities.end() &&
            record.effective_sec != 0.0) {
            record.availability =
                citlali::pipeline::OffsetAvailability::
                    unavailable_authority;
            throw std::runtime_error(
                "nonzero offset was requested for absent interface " +
                record.interface_id);
        }
    }
    auto &hwpr = citlali::pipeline::require_interface_offset_record(
        engine.interface_sync, "hwpr");
    if (hwpr.effective_sec != 0.0) {
        hwpr.availability =
            citlali::pipeline::OffsetAvailability::unavailable_authority;
        throw std::runtime_error(
            "nonzero HWPR offset is unavailable when HWPR is inactive");
    }
    (void)citlali::pipeline::realize_interface_offset(
        engine.interface_sync, "lmt", false);

    prepared.raw_overlap_end_sec =
        std::numeric_limits<double>::infinity();
    for (const auto &coordinate : coordinates) {
        const auto &times = coordinate.corrected_time.seconds;
        if (times.size() <= 0 || !std::isfinite(times(times.size() - 1))) {
            throw std::logic_error(
                "detector interface has no finite overlap endpoint");
        }
        prepared.raw_overlap_end_sec = std::min(
            prepared.raw_overlap_end_sec, times(times.size() - 1));
    }
    if (!std::isfinite(prepared.raw_overlap_end_sec)) {
        throw std::logic_error(
            "detector overlap endpoint is unavailable");
    }

    prepared.lattice =
        citlali::pipeline::sci_align::build_detector_union_lattice(
            coordinates, false);
    if (prepared.packet_gaps.size() != coordinates.size()) {
        throw std::logic_error(
            "detector packet/coordinate cardinality mismatch");
    }
    for (std::size_t interface_index = 0;
         interface_index < coordinates.size(); ++interface_index) {
        const auto &coordinate = coordinates[interface_index];
        const auto summary = require_packet_slot_consistency(
            coordinate.interface_id, coordinate.corrected_time.seconds,
            prepared.packet_gaps[interface_index],
            prepared.lattice.phase_seconds,
            prepared.lattice.cadence_seconds);
        if (summary.gap_event_count > 0) {
            if (summary.gap_event_count >
                static_cast<std::uint64_t>(
                    std::numeric_limits<int>::max())) {
                throw std::runtime_error(
                    "detector gap-event count exceeds compact summary range");
            }
            engine.alignment.gaps[coordinate.interface_id] =
                static_cast<int>(summary.gap_event_count);
        }
    }
    prepared.common_time = common_time_from_lattice(prepared.lattice);
    prepared.masks = masks_from_lattice(prepared.lattice, coordinates);
    prepared.network_times.reserve(coordinates.size());
    for (auto &coordinate : coordinates) {
        prepared.network_times.push_back(
            std::move(coordinate.corrected_time.seconds));
    }
    populate_compact_alignment_state(engine.alignment, prepared);
    // Keep deterministic union-only support on the shared slot formula, but
    // restore the exact Eigen endpoint construction used by governing 9aae
    // over global slots [0,n).  Moving through observation state avoids a
    // second dense setup allocation; publish_prepared_alignment installs the
    // completed hybrid axis after telescope interpolation succeeds.
    engine.alignment.common_time = std::move(prepared.common_time);
    citlali::pipeline::install_governing_compatibility_assigned_times(
        engine.alignment);
    prepared.common_time = std::move(engine.alignment.common_time);
    copy_interface_lifecycle_to_raw_plan(engine);
    return prepared;
}

template <class Engine>
void publish_prepared_alignment(Engine &engine,
                                PreparedDetectorAlignment prepared) {
    engine.alignment.common_time = std::move(prepared.common_time);
    engine.alignment.network_times = std::move(prepared.network_times);
    engine.alignment.masks = std::move(prepared.masks);
    for (std::size_t i = 0; i < prepared.network_ids.size(); ++i) {
        engine.alignment.network_masks[prepared.network_ids[i]] =
            engine.alignment.masks[i];
    }
}

}  // namespace citlali::engine_detail

template <class EngineType>
void TimeOrderedDataProc<EngineType>::align_timestreams(
    const RawObs &rawobs) {
    (void)rawobs;
    citlali::pipeline::reset_alignment_observation_state(
        engine().alignment);
    // The configuration boundary already rejects interp_over_gaps=false.
    // Reusing the gap lattice here would falsely claim the governing direct
    // path's native max-start-interface coordinate semantics.  Keep the
    // unsupported path explicit until that separate authority is restored.
    throw std::runtime_error(
        "direct detector/telescope alignment is unavailable; runtime.interp_over_gaps must remain true");
}

template <class EngineType>
void TimeOrderedDataProc<EngineType>::align_timestreams_gaps(
    const RawObs &rawobs) {
    auto prepared = citlali::engine_detail::prepare_detector_alignment(
        engine(), rawobs);
    citlali::pipeline::interpolate_telescope_data_to_common_time(
        engine().telescope.tel_data, prepared.common_time, true,
        &engine().alignment);
    citlali::engine_detail::publish_prepared_alignment(
        engine(), std::move(prepared));
}
