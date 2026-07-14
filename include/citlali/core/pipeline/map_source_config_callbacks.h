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
