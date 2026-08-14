#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

#include <citlali/core/engine/detail/beammap_apt_table_output_helpers.h>
#include <citlali/core/pipeline/beammap_provenance_lifecycle.h>
#include <citlali/core/pipeline/rawobs_detector_inventory.h>

template <class KidsProc, class RawObs>
void Beammap::pipeline(
    KidsProc &kidsproc, RawObs &rawobs,
    citlali::pipeline::StageProfileCollector &stage_profile) {
    // Simulation changes the telescope header namespace, not the canonical
    // raw/telescope/output tuple requirement. Simulated KIDs metadata does not
    // independently promise the real-data metadata key set, so only the real
    // path adds that fourth authority below.
    const auto telescope_identity =
        beammap_apt_table_output_helpers::telescope_observation_identity(
            telescope.tel_header, telescope.sim_obs);
    const auto output_observation =
        beammap_apt_table_output_helpers::exact_nonnegative_decimal(
            observation_identity.obsnum, "output observation number");
    (void)citlali::pipeline::validate_rawobs_observation_identity(
        calib.canonical_apt_producer.raw_manifest, telescope_identity,
        output_observation);
    beammap_apt_table_output_helpers::validate_current_raw_binding(calib);

    // Only real observations carry the current external KIDs fit report.
    if (!telescope.sim_obs) {
        // Each loader record binds a matrix to the header returned by the same
        // read, plus the raw network/obsid used to select it. There is no
        // final-header-only result and no second schema-reading path.
        auto fit_reports = kidsproc.load_fit_report(rawobs);
        std::vector<citlali::pipeline::RawObsKidsIdentity> kids_identities;
        kids_identities.reserve(fit_reports.size());
        for (const auto &report : fit_reports) {
            kids_identities.push_back({
                report.network,
                report.observation,
            });
        }
        (void)citlali::pipeline::validate_rawobs_observation_identity(
            calib.canonical_apt_producer.raw_manifest, kids_identities,
            telescope_identity, output_observation);
        beammap_apt_table_output_helpers::apply_atomic_kids_fit_report_overlay(
            calib, fit_reports,
            calib.canonical_apt_producer.raw_manifest);
    }

    // run timestream pipeline
    rtcproc.kernel.clear_source_centers();
    timestream_pipeline(kidsproc, rawobs, stage_profile);

    citlali::pipeline::begin_beammap_observation_if_available(
        *this, calib.n_dets, ptcs0.size());

    // placeholder vectors of size nscans for grppi maps
    scan_in_vec.resize(ptcs0.size());
    std::iota(scan_in_vec.begin(), scan_in_vec.end(), 0);
    scan_out_vec.resize(ptcs0.size());

    // placeholder vectors of size ndet for grppi maps
    det_in_vec.resize(map_indices.n_maps);
    std::iota(det_in_vec.begin(), det_in_vec.end(), 0);
    det_out_vec.resize(map_indices.n_maps);

    // run iterative pipeline
    loop_pipeline(kidsproc, rawobs, stage_profile);
}
