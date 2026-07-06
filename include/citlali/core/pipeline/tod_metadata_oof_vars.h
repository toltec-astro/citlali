#pragma once

// Included by tod_output_reduction_metadata.h inside namespace citlali::pipeline.

inline void add_oof_telescope_vars(netCDF::NcFile &fo, double m2x_microns,
                                   double m2y_microns,
                                   double m2z_microns) {
    add_netcdf_var(fo, "OOF_T", 3.0);
    add_netcdf_var(fo, "OOF_M2X", m2x_microns);
    add_netcdf_var(fo, "OOF_M2Y", m2y_microns);
    add_netcdf_var(fo, "OOF_M2Z", m2z_microns);
    add_netcdf_var(fo, "OOF_RO", 25.);
    add_netcdf_var(fo, "OOF_RI", 1.65);
}

template <class MapBuffer, class Calib, class ArrayNameMap,
          class WavelengthMap>
void add_oof_array_vars(netCDF::NcFile &fo, const MapBuffer &mb,
                        citlali::config::ReductionType reduction_type,
                        bool run_mapmaking, const Calib &calib,
                        ArrayNameMap &array_name_map,
                        WavelengthMap &array_wavelength_map) {
    for (decltype(calib.arrays.size()) i=0; i<calib.arrays.size(); ++i) {
        double rms = 0.0;
        if (reduction_type != citlali::config::ReductionType::beammap &&
            run_mapmaking) {
            rms = std::pow(mb->median_err(i), 0.5);
        }
        const auto array = calib.arrays(i);
        const auto &name = array_name_map[array];
        add_netcdf_var(fo, "OOF_RMS_" + name, rms);
        add_netcdf_var(fo, "OOF_W_" + name,
                       array_wavelength_map[array]/1000.);
        add_netcdf_var(fo, "OOF_ID_" + name,
                       static_cast<int>(array_wavelength_map[array]*1000));
    }
}

template <class TelescopeHeader, class MapBuffer, class Calib,
          class ArrayNameMap, class WavelengthMap>
void add_oof_header_vars_if_observed(
    netCDF::NcFile &fo, bool simulated_observation,
    TelescopeHeader &tel_header, const MapBuffer &mb,
    citlali::config::ReductionType reduction_type, bool run_mapmaking,
    const Calib &calib, ArrayNameMap &array_name_map,
    WavelengthMap &array_wavelength_map) {
    if (simulated_observation) {
        return;
    }

    add_oof_telescope_vars(
        fo, tel_header["Header.M2.XReq"](0) / 1000. * 1e6,
        tel_header["Header.M2.YReq"](0) / 1000. * 1e6,
        tel_header["Header.M2.ZReq"](0) / 1000. * 1e6);
    add_oof_array_vars(
        fo, mb, reduction_type, run_mapmaking, calib, array_name_map,
        array_wavelength_map);
}
