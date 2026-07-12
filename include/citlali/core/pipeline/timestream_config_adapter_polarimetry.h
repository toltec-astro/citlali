#pragma once

namespace citlali::pipeline {

template <class SourceRtcProc, class TargetRtcProc>
void adapt_legacy_polarimetry_runtime(
    const SourceRtcProc &source, TargetRtcProc &target) {
    target.run_polarization = source.run_polarization;
    target.polarization.grouping = source.polarization.grouping;
    target.polarization.stokes_params = source.polarization.stokes_params;
}

}  // namespace citlali::pipeline
