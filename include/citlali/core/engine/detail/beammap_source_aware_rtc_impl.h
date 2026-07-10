#pragma once

// Beammap mapmaking stage implementation detail.
// Include this only after Beammap has been declared.

template <class KidsProc, class RawObs>
bool Beammap::maybe_run_beammap_source_aware_rtc(KidsProc &kidsproc,
                                                 RawObs &rawobs,
                                                 bool first_measurement_iter,
                                                 bool detector_grouping) {
    configure_detector_source_centers_from_previous_fit();

    const bool detector_kernel_source_centers_active =
        detector_grouping &&
        rtcproc.run_kernel &&
        rtcproc.kernel.has_source_centers();
    const bool rerun_source_aware_rtc =
        first_measurement_iter && detector_kernel_source_centers_active;
    if (!rerun_source_aware_rtc) {
        return false;
    }

    logger->info(
        "beammap iter {} rerunning RTC with previous-fit detector source centers; regular RTC TOD output disabled for this internal pass",
        current_iter);
    const auto profile_scope =
        citlali::pipeline::profile_stage(
            "beammap.rtc.source_aware_rerun", logger,
            "iter=" + std::to_string(current_iter));
    timestream_pipeline(kidsproc, rawobs, false);
    return true;
}
