#pragma once

// Included by map_source_finding.h inside namespace citlali::pipeline.

struct SourceFitUnitScale {
    double pixel_to_arcsec;
    double source_fwhm_to_arcsec;
};

struct SourceFitUnitConstants {
    double rad_to_arcsec;
    double std_to_fwhm;
    double arcsec_to_rad;
    double rad_to_deg;
    double deg_to_rad;
    double arcsec_to_deg;
};

struct SourceInitialPosition {
    double row;
    double col;
};

inline double source_window_arcsec_to_rad(double source_window_arcsec,
                                          double arcsec_to_rad) {
    return source_window_arcsec * arcsec_to_rad;
}

inline double source_fitting_arcsec_to_pixels(double value_arcsec,
                                              double arcsec_to_rad,
                                              double pixel_size_rad) {
    return arcsec_to_rad * value_arcsec / pixel_size_rad;
}

inline bool source_fitting_config_needed(citlali::config::ReductionType reduction_type,
                                         bool run_map_filter,
                                         bool run_source_finder) {
    return citlali::config::is_pointing_reduction_type(reduction_type) ||
           citlali::config::is_beammap_reduction_type(reduction_type) ||
           run_map_filter || run_source_finder;
}

template <class PostProcessingConfig>
bool source_fitting_config_needed(
    citlali::config::ReductionType reduction_type,
    const PostProcessingConfig &post_processing_config) {
    return source_fitting_config_needed(
        reduction_type,
        citlali::config::map_filtering_active(post_processing_config),
        citlali::config::source_finding_active(post_processing_config));
}

template <class MapFitter>
void apply_positive_source_fit_limits(MapFitter &map_fitter) {
    if (map_fitter.flux_limits(0) > 0) {
        map_fitter.flux_low = map_fitter.flux_limits(0);
    }
    if (map_fitter.flux_limits(1) > 0) {
        map_fitter.flux_high = map_fitter.flux_limits(1);
    }
    if (map_fitter.fwhm_limits(0) > 0) {
        map_fitter.fwhm_low = map_fitter.fwhm_limits(0);
    }
    if (map_fitter.fwhm_limits(1) > 0) {
        map_fitter.fwhm_high = map_fitter.fwhm_limits(1);
    }
}

template <class ObservationMapBuffer, class CoaddMapBuffer>
void mirror_source_finding_config_to_coadd(
    const ObservationMapBuffer &omb, CoaddMapBuffer &cmb,
    bool run_coadd) {
    if (!run_coadd) {
        return;
    }
    cmb.source_sigma = omb.source_sigma;
    cmb.source_window_rad = omb.source_window_rad;
    cmb.source_finder_mode = omb.source_finder_mode;
}

template <class MapsToArrays, class InitFwhmForArray, class FitMapSources>
struct SourceFitCallbacks {
    MapsToArrays maps_to_arrays;
    InitFwhmForArray init_fwhm_for_array;
    FitMapSources fit_map_sources;
};

template <class MapsToArrays, class InitFwhmForArray, class FitMapSources>
SourceFitCallbacks<MapsToArrays, InitFwhmForArray, FitMapSources>
make_source_fit_callbacks(const MapsToArrays &maps_to_arrays,
                          const InitFwhmForArray &init_fwhm_for_array,
                          const FitMapSources &fit_map_sources) {
    return {maps_to_arrays, init_fwhm_for_array, fit_map_sources};
}

template <class MapToArray, class CalcStdDev, class WriteSourceTable>
struct SourceTableCallbacks {
    MapToArray maps_to_arrays;
    CalcStdDev calc_std_dev;
    WriteSourceTable write_source_table;
};

template <class MapToArray, class CalcStdDev, class WriteSourceTable>
SourceTableCallbacks<MapToArray, CalcStdDev, WriteSourceTable>
make_source_table_callbacks(const MapToArray &maps_to_arrays,
                            const CalcStdDev &calc_std_dev,
                            const WriteSourceTable &write_source_table) {
    return {maps_to_arrays, calc_std_dev, write_source_table};
}

constexpr int missing_source_location() {
    return -99;
}
