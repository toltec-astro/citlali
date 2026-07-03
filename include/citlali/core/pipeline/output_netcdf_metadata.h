#pragma once

#include <cstddef>
#include <cmath>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <netcdf>

#include <citlali/core/pipeline/phdu_beammap.h>
#include <citlali/core/utils/netcdf_io.h>

namespace citlali::pipeline {

inline double tod_output_fill_double() {
    return std::numeric_limits<double>::quiet_NaN();
}

constexpr int tod_output_fill_int() {
    return -2147483647;
}

struct TodFileDims {
    netCDF::NcDim n_pts;
    netCDF::NcDim n_raw_scan_indices;
    netCDF::NcDim n_scan_indices;
    netCDF::NcDim n_scans;
    netCDF::NcDim n_dets;
    std::vector<netCDF::NcDim> signal;
    std::vector<netCDF::NcDim> raw_scans;
    std::vector<netCDF::NcDim> scans;
};

struct TodStreamLayout {
    Eigen::Index n_output_scans;
    bool mini_output;
    bool outer_output;
};

struct TodFileCounts {
    std::size_t n_output_scans;
    std::size_t n_raw_scan_indices;
    std::size_t n_dets;
};

struct TodChunking {
    netCDF::NcVar::ChunkMode mode;
    std::vector<std::size_t> sizes;
};

inline TodFileCounts tod_file_counts(Eigen::Index n_output_scans,
                                     Eigen::Index n_raw_scan_indices,
                                     Eigen::Index n_dets) {
    return {
        static_cast<std::size_t>(n_output_scans),
        static_cast<std::size_t>(n_raw_scan_indices),
        static_cast<std::size_t>(n_dets),
    };
}

template <class RtcProc, class PtcProc>
TodStreamLayout tod_stream_layout(bool is_rtc_stream,
                                  Eigen::Index n_rtc_output_scans,
                                  Eigen::Index n_ptc_output_scans,
                                  const RtcProc &rtcproc,
                                  const PtcProc &ptcproc) {
    return {
        is_rtc_stream ? n_rtc_output_scans : n_ptc_output_scans,
        is_rtc_stream ? rtcproc.tod_output_mini : ptcproc.tod_output_mini,
        is_rtc_stream ? rtcproc.tod_output_outer : ptcproc.tod_output_outer,
    };
}

inline TodFileDims add_tod_file_dims(netCDF::NcFile &fo,
                                     std::size_t n_output_scans,
                                     std::size_t n_raw_scan_indices,
                                     std::size_t n_dets) {
    TodFileDims dims;
    dims.n_pts = fo.addDim("n_pts");
    dims.n_raw_scan_indices =
        fo.addDim("n_raw_scan_indices", n_raw_scan_indices);
    dims.n_scan_indices = fo.addDim("n_scan_indices", 2);
    dims.n_scans = fo.addDim("n_scans", n_output_scans);
    dims.n_dets = fo.addDim("n_dets", n_dets);
    dims.signal = {dims.n_pts, dims.n_dets};
    dims.raw_scans = {dims.n_scans, dims.n_raw_scan_indices};
    dims.scans = {dims.n_scans, dims.n_scan_indices};
    return dims;
}

template <class ScanIndices>
std::vector<std::size_t> tod_data_chunk_sizes(const ScanIndices &scan_indices,
                                              std::size_t n_dets) {
    const auto mean_scan_size =
        ((scan_indices.row(3) - scan_indices.row(2)).array() + 1).mean();
    return {static_cast<std::size_t>(mean_scan_size), n_dets};
}

template <class ScanIndices>
TodChunking tod_data_chunking(const ScanIndices &scan_indices,
                              std::size_t n_dets) {
    return {
        netCDF::NcVar::nc_CHUNKED,
        tod_data_chunk_sizes(scan_indices, n_dets),
    };
}

inline void add_tod_output_type_label(netCDF::NcFile &fo,
                                      const std::string &label) {
    netCDF::NcDim dim = fo.addDim("n_tod_output_type", 1);
    netCDF::NcVar var = fo.addVar("tod_output_type", netCDF::ncString, dim);
    const std::vector<std::size_t> index = {0};
    std::string value = label;
    var.putVar(index, value);
}

inline void add_obsnum_var(netCDF::NcFile &fo, int obsnum) {
    netCDF::NcVar var = fo.addVar("obsnum", netCDF::ncInt);
    var.putAtt("units", "N/A");
    var.putVar(&obsnum);
}

inline void add_source_radec_vars(netCDF::NcFile &fo, double source_ra,
                                  double source_dec) {
    netCDF::NcVar source_ra_v = fo.addVar("SourceRa", netCDF::ncDouble);
    source_ra_v.putAtt("units", "rad");
    source_ra_v.putVar(&source_ra);

    netCDF::NcVar source_dec_v = fo.addVar("SourceDec", netCDF::ncDouble);
    source_dec_v.putAtt("units", "rad");
    source_dec_v.putVar(&source_dec);
}

inline void add_observation_identity_vars(netCDF::NcFile &fo, int obsnum,
                                          double source_ra,
                                          double source_dec) {
    add_obsnum_var(fo, obsnum);
    add_source_radec_vars(fo, source_ra, source_dec);
}

inline void add_pipeline_identity_vars(
    netCDF::NcFile &fo, const std::string &citlali_version,
    const std::string &kids_version, const std::string &tula_version,
    const std::string &project_id, const std::string &reduction_goal,
    const std::string &obs_goal, const std::string &tod_type) {
    add_netcdf_var<std::string>(fo, "INSTRUME", "TolTEC");
    add_netcdf_var<std::string>(fo, "TELESCOP", "LMT");
    add_netcdf_var<std::string>(fo, "PIPELINE", "CITLALI");
    add_netcdf_var<std::string>(fo, "VERSION", citlali_version);
    add_netcdf_var<std::string>(fo, "KIDS", kids_version);
    add_netcdf_var<std::string>(fo, "TULA", tula_version);
    add_netcdf_var<std::string>(fo, "PROJID", project_id);
    add_netcdf_var<std::string>(fo, "GOAL", reduction_goal);
    add_netcdf_var<std::string>(fo, "OBSGOAL", obs_goal);
    add_netcdf_var<std::string>(fo, "TYPE", tod_type);
}

inline void add_observation_date_source_vars(netCDF::NcFile &fo,
                                             const std::string &date_obs,
                                             const std::string &source_name) {
    add_netcdf_var<std::string>(fo, "DATEOBS0", date_obs);
    add_netcdf_var<std::string>(fo, "SOURCE", source_name);
}

inline void add_tod_map_geometry_vars(
    netCDF::NcFile &fo, const std::string &map_grouping,
    const std::string &map_method, double exposure_time,
    const std::string &radec_system, double tangent_ra, double tangent_dec,
    double mean_el_deg, double mean_az_deg, double mean_pa_deg) {
    add_netcdf_var<std::string>(fo, "GROUPING", map_grouping);
    add_netcdf_var<std::string>(fo, "METHOD", map_method);
    add_netcdf_var(fo, "EXPTIME", exposure_time);
    add_netcdf_var<std::string>(fo, "RADESYS", radec_system);
    add_netcdf_var(fo, "TAN_RA", tangent_ra);
    add_netcdf_var(fo, "TAN_DEC", tangent_dec);
    add_netcdf_var(fo, "MEAN_EL", mean_el_deg);
    add_netcdf_var(fo, "MEAN_AZ", mean_az_deg);
    add_netcdf_var(fo, "MEAN_PA", mean_pa_deg);
}

inline void add_tod_signal_unit_var(netCDF::NcFile &fo,
                                    const std::string &signal_unit) {
    add_netcdf_var(fo, "BUNIT", signal_unit);
}

inline void add_tod_auxiliary_metadata_vars(netCDF::NcFile &fo,
                                            double sample_rate_hz,
                                            const std::string &apt_name,
                                            int fruit_loop_iter) {
    add_netcdf_var(fo, "SAMPRATE", sample_rate_hz);
    add_netcdf_var<std::string>(fo, "APT", apt_name);
    add_netcdf_var(fo, "FRUITLOOPS_ITER", fruit_loop_iter);
}

inline void add_unit_conversion_basis_vars(netCDF::NcFile &fo) {
    add_netcdf_var<std::string>(fo, "UNITCONV.UK_CONVENTION",
                                "Rayleigh-Jeans brightness temperature");
    add_netcdf_var<std::string>(
        fo, "UNITCONV.UK_BASIS",
        "monochromatic array center frequency; mJy/beam uses Gaussian beam solid angle to Jy/sr");
}

inline void add_unit_conversion_array_vars(
    netCDF::NcFile &fo, const std::string &array_name,
    const std::string &signal_unit, double mjy_sr_to_mjy_beam,
    double mjy_beam_to_uk, double mjy_beam_to_jy_pixel) {
    if (signal_unit == "mJy/beam") {
        add_netcdf_var(fo, "to_mJy_beam_" + array_name, 1);
        add_netcdf_var(fo, "to_MJy_sr_" + array_name,
                       1/mjy_sr_to_mjy_beam);
        add_netcdf_var(fo, "to_uK_" + array_name, mjy_beam_to_uk);
        add_netcdf_var(fo, "to_Jy_pixel_" + array_name,
                       mjy_beam_to_jy_pixel);
    }
    else if (signal_unit == "MJy/sr") {
        add_netcdf_var(fo, "to_mJy_beam_" + array_name,
                       mjy_sr_to_mjy_beam);
        add_netcdf_var(fo, "to_MJy_sr_" + array_name, 1);
        add_netcdf_var(fo, "to_uK_" + array_name,
                       mjy_sr_to_mjy_beam*mjy_beam_to_uk);
        add_netcdf_var(fo, "to_Jy_pixel_" + array_name,
                       mjy_sr_to_mjy_beam*mjy_beam_to_jy_pixel);
    }
    else if (signal_unit == "uK") {
        add_netcdf_var(fo, "to_mJy_beam_" + array_name, 1/mjy_beam_to_uk);
        add_netcdf_var(fo, "to_MJy_sr_" + array_name,
                       1/mjy_beam_to_uk/mjy_sr_to_mjy_beam);
        add_netcdf_var(fo, "to_uK_" + array_name, 1);
        add_netcdf_var(fo, "to_Jy_pixel_" + array_name,
                       (1/mjy_beam_to_uk)*mjy_beam_to_jy_pixel);
    }
    else if (signal_unit == "Jy/pixel") {
        add_netcdf_var(fo, "to_mJy_beam_" + array_name,
                       1/mjy_beam_to_jy_pixel);
        add_netcdf_var(fo, "to_MJy_sr_" + array_name,
                       (1/mjy_beam_to_jy_pixel)/mjy_sr_to_mjy_beam);
        add_netcdf_var(fo, "to_uK_" + array_name,
                       mjy_beam_to_uk/mjy_beam_to_jy_pixel);
        add_netcdf_var(fo, "to_Jy_pixel_" + array_name, 1);
    }
}

template <class Arrays, class FwhmMap, class PositionAngleMap,
          class ArrayNameMap>
void add_array_beam_geometry_vars(netCDF::NcFile &fo, const Arrays &arrays,
                                  FwhmMap &array_fwhms,
                                  PositionAngleMap &array_pas,
                                  ArrayNameMap &array_name_map,
                                  double rad_to_deg,
                                  double pa_quadrature_offset_rad) {
    for (const auto &arr: arrays) {
        const auto &fwhm = array_fwhms[arr];
        const auto &name = array_name_map[arr];
        if (std::get<0>(fwhm) >= std::get<1>(fwhm)) {
            add_netcdf_var(fo, "BMAJ_" + name, std::get<0>(fwhm));
            add_netcdf_var(fo, "BMIN_" + name, std::get<1>(fwhm));
            add_netcdf_var(fo, "BPA_" + name, array_pas[arr]*rad_to_deg);
        }
        else {
            add_netcdf_var(fo, "BMAJ_" + name, std::get<1>(fwhm));
            add_netcdf_var(fo, "BMIN_" + name, std::get<0>(fwhm));
            add_netcdf_var(fo, "BPA_" + name,
                           (array_pas[arr] + pa_quadrature_offset_rad)*
                               rad_to_deg);
        }
    }
}

template <class Arrays, class FwhmMap, class PositionAngleMap,
          class ArrayNameMap>
void add_tod_identity_geometry_vars(
    netCDF::NcFile &fo, const std::string &citlali_version,
    const std::string &kids_version, const std::string &tula_version,
    const std::string &project_id, const std::string &reduction_goal,
    const std::string &obs_goal, const std::string &tod_type, bool run_hwpr,
    const std::string &map_grouping, const std::string &map_method,
    double exposure_time, const std::string &pixel_axes, double tangent_ra,
    double tangent_dec, double mean_el_deg, double mean_az_deg,
    double mean_pa_deg, const Arrays &arrays, FwhmMap &array_fwhms,
    PositionAngleMap &array_pas, ArrayNameMap &array_name_map,
    double rad_to_deg, double pa_quadrature_offset_rad,
    const std::string &signal_unit) {
    add_pipeline_identity_vars(
        fo, citlali_version, kids_version, tula_version, project_id,
        reduction_goal, obs_goal, tod_type);
    add_netcdf_var(fo, "HWPR", run_hwpr);
    add_tod_map_geometry_vars(
        fo, map_grouping, map_method, exposure_time, pixel_axes, tangent_ra,
        tangent_dec, mean_el_deg, mean_az_deg, mean_pa_deg);
    add_array_beam_geometry_vars(
        fo, arrays, array_fwhms, array_pas, array_name_map, rad_to_deg,
        pa_quadrature_offset_rad);
    add_tod_signal_unit_var(fo, signal_unit);
}

template <class Arrays, class ShapeParams, class ArrayNameMap>
void add_jinc_shape_config_vars(netCDF::NcFile &fo, const Arrays &arrays,
                                ShapeParams &shape_params,
                                ArrayNameMap &array_name_map,
                                double r_max) {
    add_netcdf_var(fo, "JINC_R", r_max);
    for (const auto &arr: arrays) {
        const auto &name = array_name_map[arr];
        add_netcdf_var(fo, "JINC_A_" + name, shape_params[arr][0]);
        add_netcdf_var(fo, "JINC_B_" + name, shape_params[arr][1]);
        add_netcdf_var(fo, "JINC_C_" + name, shape_params[arr][2]);
    }
}

template <class Arrays, class ShapeParams, class ArrayNameMap>
void add_jinc_shape_config_vars_if_needed(
    netCDF::NcFile &fo, const std::string &map_method, const Arrays &arrays,
    ShapeParams &shape_params, ArrayNameMap &array_name_map, double r_max) {
    if (map_method == "jinc") {
        add_jinc_shape_config_vars(
            fo, arrays, shape_params, array_name_map, r_max);
    }
}

template <class TauByFrequency, class Calib, class ArrayNameMap>
void add_mean_tau_vars(netCDF::NcFile &fo, const TauByFrequency &tau_freq,
                       const Calib &calib, ArrayNameMap &array_name_map) {
    decltype(calib.arrays.size()) i = 0;
    for (auto const& [key, val] : tau_freq) {
        add_netcdf_var(
            fo, "MEAN_TAU_" + array_name_map[calib.arrays(i)], val[0]);
        i++;
    }
}

template <class Calib, class ArrayNameMap>
void add_zero_mean_tau_vars(netCDF::NcFile &fo, const Calib &calib,
                            ArrayNameMap &array_name_map) {
    for (decltype(calib.arrays.size()) i=0; i<calib.arrays.size(); ++i) {
        add_netcdf_var(
            fo, "MEAN_TAU_" + array_name_map[calib.arrays(i)], 0.);
    }
}

template <class Rtcproc, class TelescopeData, class Calib, class ArrayNameMap>
void add_tod_mean_tau_vars(netCDF::NcFile &fo, Rtcproc &rtcproc,
                           TelescopeData &tel_data, double tau_225_ghz,
                           const Calib &calib, ArrayNameMap &array_name_map) {
    if (rtcproc.run_extinction) {
        Eigen::VectorXd tau_el(1);
        tau_el << tel_data["TelElAct"].mean();
        auto tau_freq = rtcproc.calibration.calc_tau(tau_el, tau_225_ghz);
        add_mean_tau_vars(fo, tau_freq, calib, array_name_map);
    }
    else {
        add_zero_mean_tau_vars(fo, calib, array_name_map);
    }
}

template <class Arrays, class ArrayNameMap, class FluxMap>
void add_beammap_source_flux_vars(netCDF::NcFile &fo, const Arrays &arrays,
                                  ArrayNameMap &array_name_map,
                                  FluxMap &flux_mjy_beam,
                                  FluxMap &flux_mjy_sr) {
    for (const auto &arr: arrays) {
        const auto &name = array_name_map[arr];
        add_netcdf_var(fo, "HEADER.SOURCE.FLUX_MJYPERBEAM_" + name,
                       flux_mjy_beam[name]);
        add_netcdf_var(fo, "HEADER.SOURCE.FLUX_MJYPERSR_" + name,
                       flux_mjy_sr[name]);
    }
}

inline void add_beammap_tuning_vars(
    netCDF::NcFile &fo, double iter_tolerance,
    double convergence_radius_arcsec, int iter_max, bool phase_split_enabled,
    int locator_iter, int measurement_start_iter, bool is_derotated) {
    add_netcdf_var(fo, "BEAMMAP.ITER_TOLERANCE", iter_tolerance);
    add_netcdf_var(fo, "BEAMMAP.CONVERGENCE_RADIUS_ARCSEC",
                   convergence_radius_arcsec);
    add_netcdf_var(fo, "BEAMMAP.ITER_MAX", iter_max);
    add_netcdf_var(fo, "BEAMMAP.PHASE_SPLIT_ENABLED",
                   phase_split_enabled);
    add_netcdf_var(fo, "BEAMMAP.LOCATOR_ITER", locator_iter);
    add_netcdf_var(fo, "BEAMMAP.MEASUREMENT_START_ITER",
                   measurement_start_iter);
    add_netcdf_var(fo, "BEAMMAP.IS_DEROTATED", is_derotated);
}

inline void add_beammap_reference_vars(netCDF::NcFile &fo, int det_index,
                                       double ref_x_t, double ref_y_t) {
    add_netcdf_var(fo, "BEAMMAP.REF_DET_INDEX", det_index);
    add_netcdf_var(fo, "BEAMMAP.REF_X_T", ref_x_t);
    add_netcdf_var(fo, "BEAMMAP.REF_Y_T", ref_y_t);
}

template <class Calib, class ArrayNameMap, class FluxMap, class ReferenceDet>
void add_beammap_tod_header_vars(
    netCDF::NcFile &fo, Calib &calib, ArrayNameMap &array_name_map,
    FluxMap &flux_mjy_beam, FluxMap &flux_mjy_sr, double iter_tolerance,
    double convergence_radius_arcsec, int iter_max,
    bool phase_split_enabled, int locator_iter, int measurement_start_iter,
    bool is_derotated, bool subtract_reference,
    const ReferenceDet &reference_det) {
    add_beammap_source_flux_vars(
        fo, calib.arrays, array_name_map, flux_mjy_beam, flux_mjy_sr);
    add_beammap_tuning_vars(
        fo, iter_tolerance, convergence_radius_arcsec, iter_max,
        phase_split_enabled, locator_iter, measurement_start_iter,
        is_derotated);

    int ref_det_index = -99;
    double ref_x_t = -99.0;
    double ref_y_t = -99.0;
    if (subtract_reference) {
        const auto reference_values =
            beammap_reference_header_values(calib, reference_det);
        ref_det_index = reference_values.det_index;
        ref_x_t = reference_values.x_t;
        ref_y_t = reference_values.y_t;
    }
    add_beammap_reference_vars(fo, ref_det_index, ref_x_t, ref_y_t);
}

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
                        const std::string &reduction_type,
                        bool run_mapmaking, const Calib &calib,
                        ArrayNameMap &array_name_map,
                        WavelengthMap &array_wavelength_map) {
    for (decltype(calib.arrays.size()) i=0; i<calib.arrays.size(); ++i) {
        double rms = 0.0;
        if (reduction_type != "beammap" && run_mapmaking) {
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
    const std::string &reduction_type, bool run_mapmaking,
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

inline void add_tod_scan_index_placeholders(
    netCDF::NcFile &fo, const std::vector<netCDF::NcDim> &raw_scans_dims,
    const std::vector<netCDF::NcDim> &scans_dims,
    netCDF::NcDim n_scans_dim, std::size_t n_output_scans,
    std::size_t n_raw_scan_indices, bool tod_output_outer, int fill_value) {
    netCDF::NcVar raw_scan_indices_v =
        fo.addVar("raw_scan_indices", netCDF::ncInt, raw_scans_dims);
    raw_scan_indices_v.putAtt("units", "N/A");
    raw_scan_indices_v.putAtt(
        "comment",
        tod_output_outer
            ? "indices in output timebase: inner_start, inner_end, outer_start, outer_end"
            : "indices in output timebase; outer=inner (output stores inner scans only)");
    std::vector<int> raw_scan_init(n_output_scans * n_raw_scan_indices,
                                   fill_value);
    raw_scan_indices_v.putVar(raw_scan_init.data());

    netCDF::NcVar scan_indices_v =
        fo.addVar("scan_indices", netCDF::ncInt, scans_dims);
    scan_indices_v.putAtt("units", "N/A");
    std::vector<int> scan_init(n_output_scans * 2, fill_value);
    scan_indices_v.putVar(scan_init.data());

    netCDF::NcVar output_scan_index_v =
        fo.addVar("output_scan_index", netCDF::ncInt, n_scans_dim);
    output_scan_index_v.putAtt("units", "N/A");
    output_scan_index_v.putAtt(
        "comment", "1-based original scan index from the full observation");
    std::vector<int> output_scan_init(n_output_scans, fill_value);
    output_scan_index_v.putVar(output_scan_init.data());
}

inline void add_tod_scan_int_placeholder_var(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, netCDF::NcDim n_scans_dim,
    std::size_t n_output_scans, int fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, n_scans_dim);
    v.putAtt("units", "samples");
    v.putAtt("comment", comment);
    std::vector<int> init(n_output_scans, fill_value);
    v.putVar(init.data());
}

inline void add_tod_scan_double_placeholder_var(
    netCDF::NcFile &fo, const std::string &name, const std::string &units,
    const std::string &comment, netCDF::NcDim n_scans_dim,
    std::size_t n_output_scans, double fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, n_scans_dim);
    v.putAtt("units", units);
    v.putAtt("comment", comment);
    std::vector<double> init(n_output_scans, fill_value);
    v.putVar(init.data());
}

inline void set_tod_var_chunking(
    netCDF::NcVar &var, netCDF::NcVar::ChunkMode chunk_mode,
    const std::vector<std::size_t> &chunk_sizes) {
    auto chunks = chunk_sizes;
    var.setChunking(chunk_mode, chunks);
}

inline void add_tod_signal_var(
    netCDF::NcFile &fo, const std::vector<netCDF::NcDim> &dims,
    bool mini_output, const std::string &signal_unit,
    netCDF::NcVar::ChunkMode chunk_mode,
    const std::vector<std::size_t> &chunk_sizes) {
    netCDF::NcVar signal_v;
    if (mini_output) {
        signal_v = fo.addVar("signal", netCDF::ncFloat, dims);
    }
    else {
        signal_v = fo.addVar("signal", netCDF::ncDouble, dims);
    }
    signal_v.putAtt("units", signal_unit);
    set_tod_var_chunking(signal_v, chunk_mode, chunk_sizes);
}

inline void add_tod_flags_var(
    netCDF::NcFile &fo, const std::vector<netCDF::NcDim> &dims,
    bool mini_output, netCDF::NcVar::ChunkMode chunk_mode,
    const std::vector<std::size_t> &chunk_sizes) {
    netCDF::NcVar flags_v;
    if (mini_output) {
        flags_v = fo.addVar("flags", netCDF::ncByte, dims);
    }
    else {
        flags_v = fo.addVar("flags", netCDF::ncDouble, dims);
    }
    flags_v.putAtt("units", "N/A");
    if (mini_output) {
        flags_v.putAtt("comment", "0=good,1=flagged");
    }
    set_tod_var_chunking(flags_v, chunk_mode, chunk_sizes);
}

inline void add_tod_kernel_var_if_requested(
    netCDF::NcFile &fo, const std::vector<netCDF::NcDim> &dims,
    bool run_kernel, bool mini_output,
    netCDF::NcVar::ChunkMode chunk_mode,
    const std::vector<std::size_t> &chunk_sizes) {
    if (!run_kernel || mini_output) {
        return;
    }
    netCDF::NcVar kernel_v = fo.addVar("kernel", netCDF::ncDouble, dims);
    kernel_v.putAtt("units", "N/A");
    set_tod_var_chunking(kernel_v, chunk_mode, chunk_sizes);
}

inline void add_tod_detector_pointing_vars(
    netCDF::NcFile &fo, const std::vector<netCDF::NcDim> &dims,
    bool mini_output, const std::string &pixel_axes,
    netCDF::NcVar::ChunkMode chunk_mode,
    const std::vector<std::size_t> &chunk_sizes) {
    if (mini_output) {
        return;
    }

    netCDF::NcVar det_lat_v = fo.addVar("det_lat", netCDF::ncDouble, dims);
    det_lat_v.putAtt("units", "rad");
    set_tod_var_chunking(det_lat_v, chunk_mode, chunk_sizes);

    netCDF::NcVar det_lon_v = fo.addVar("det_lon", netCDF::ncDouble, dims);
    det_lon_v.putAtt("units", "rad");
    set_tod_var_chunking(det_lon_v, chunk_mode, chunk_sizes);

    if (pixel_axes == "radec") {
        netCDF::NcVar det_ra_v = fo.addVar("det_ra", netCDF::ncDouble, dims);
        det_ra_v.putAtt("units", "rad");
        set_tod_var_chunking(det_ra_v, chunk_mode, chunk_sizes);

        netCDF::NcVar det_dec_v =
            fo.addVar("det_dec", netCDF::ncDouble, dims);
        det_dec_v.putAtt("units", "rad");
        set_tod_var_chunking(det_dec_v, chunk_mode, chunk_sizes);
    }
}

inline void add_tod_core_data_vars(
    netCDF::NcFile &fo, const std::vector<netCDF::NcDim> &dims,
    bool mini_output, const std::string &signal_unit, bool run_kernel,
    const std::string &pixel_axes, netCDF::NcVar::ChunkMode chunk_mode,
    const std::vector<std::size_t> &chunk_sizes) {
    add_tod_signal_var(fo, dims, mini_output, signal_unit, chunk_mode,
                       chunk_sizes);
    add_tod_flags_var(fo, dims, mini_output, chunk_mode, chunk_sizes);
    add_tod_kernel_var_if_requested(
        fo, dims, run_kernel, mini_output, chunk_mode, chunk_sizes);
    add_tod_detector_pointing_vars(
        fo, dims, mini_output, pixel_axes, chunk_mode, chunk_sizes);
}

template <class AptTable, class AptUnits>
void add_tod_apt_table_vars(netCDF::NcFile &fo, const AptTable &apt,
                            const AptUnits &apt_header_units,
                            netCDF::NcDim n_dets_dim) {
    for (const auto &item : apt) {
        netCDF::NcVar apt_v =
            fo.addVar("apt_" + item.first, netCDF::ncDouble, n_dets_dim);
        const auto units_it = apt_header_units.find(item.first);
        const std::string units =
            (units_it == apt_header_units.end()) ? "" : units_it->second;
        apt_v.putAtt("units", units);
    }
}

template <class TelescopeData>
void add_telescope_data_vars(
    netCDF::NcFile &fo, const TelescopeData &tel_data,
    netCDF::NcDim n_pts_dim, netCDF::NcVar::ChunkMode chunk_mode,
    const std::vector<std::size_t> &chunk_sizes) {
    for (const auto &item : tel_data) {
        netCDF::NcVar tel_data_v =
            fo.addVar(item.first, netCDF::ncDouble, n_pts_dim);
        tel_data_v.putAtt("units", "rad");
        set_tod_var_chunking(tel_data_v, chunk_mode, chunk_sizes);
    }
}

template <class PointingOffsets, class Logger>
void add_tod_pointing_offset_vars(
    netCDF::NcFile &fo, const PointingOffsets &pointing_offsets_arcsec,
    const Logger &logger, netCDF::NcDim n_pts_dim,
    netCDF::NcVar::ChunkMode chunk_mode,
    const std::vector<std::size_t> &chunk_sizes) {
    for (const auto &item : pointing_offsets_arcsec) {
        logger->info("pointing_offsets_arcsec.second {} {}", item.first,
                     item.second);
        netCDF::NcVar offsets_v = fo.addVar(
            "pointing_offset_" + item.first, netCDF::ncDouble, n_pts_dim);
        offsets_v.putAtt("units", "arcsec");
        set_tod_var_chunking(offsets_v, chunk_mode, chunk_sizes);
    }
}

template <class AddInt, class AddDouble>
void add_tod_filter_edge_guard_scan_vars(const AddInt &add_int,
                                         const AddDouble &add_double) {
    add_int("tod_filter_edge_guard_pre_samples",
            "samples flagged at the start of this output scan by the TOD filter edge guard");
    add_int("tod_filter_edge_guard_post_samples",
            "samples flagged at the end of this output scan by the TOD filter edge guard");
    add_int("tod_filter_edge_guard_flagged_samples",
            "detector-samples flagged by the TOD filter edge guard");
    add_double("tod_filter_edge_guard_flagged_frac", "N/A",
               "fraction of time samples guarded at this output scan edge");
}

inline void add_tod_filter_edge_guard_scan_placeholders(
    netCDF::NcFile &fo, netCDF::NcDim n_scans_dim,
    std::size_t n_output_scans, int fill_int, double fill_double) {
    auto add_scan_int_var = [&](const std::string &name,
                                const std::string &comment) {
        add_tod_scan_int_placeholder_var(
            fo, name, comment, n_scans_dim, n_output_scans, fill_int);
    };
    auto add_scan_double_var = [&](const std::string &name,
                                   const std::string &units,
                                   const std::string &comment) {
        add_tod_scan_double_placeholder_var(
            fo, name, units, comment, n_scans_dim, n_output_scans,
            fill_double);
    };
    add_tod_filter_edge_guard_scan_vars(add_scan_int_var,
                                        add_scan_double_var);
}

inline void add_tod_hwpr_var(netCDF::NcFile &fo, netCDF::NcDim n_pts_dim) {
    netCDF::NcVar hwpr_v = fo.addVar("hwpr", netCDF::ncDouble, n_pts_dim);
    hwpr_v.putAtt("units", "rad");
}

inline void add_tod_hwpr_var_if_requested(netCDF::NcFile &fo,
                                          bool run_polarization,
                                          bool run_hwpr,
                                          netCDF::NcDim n_pts_dim) {
    if (run_polarization && run_hwpr) {
        add_tod_hwpr_var(fo, n_pts_dim);
    }
}

template <class TelescopeHeader>
void add_telescope_header_vars(netCDF::NcFile &fo,
                               const TelescopeHeader &tel_header) {
    netCDF::NcDim tel_header_dim = fo.addDim("tel_header_n_pts", 1);
    for (const auto &[key, val] : tel_header) {
        netCDF::NcVar tel_header_v =
            fo.addVar(key, netCDF::ncDouble, tel_header_dim);
        tel_header_v.putVar(&val(0));
    }
}

}  // namespace citlali::pipeline
