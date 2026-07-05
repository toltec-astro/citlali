#pragma once

// TOD output NetCDF metadata implementation detail.
// Include this only from output_netcdf_metadata.h inside citlali::pipeline.

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

inline void add_diagnostic_file_identity_vars(netCDF::NcFile &fo,
                                              const std::string &output_type,
                                              int obsnum, double source_ra,
                                              double source_dec) {
    add_tod_output_type_label(fo, output_type);
    add_observation_identity_vars(fo, obsnum, source_ra, source_dec);
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
