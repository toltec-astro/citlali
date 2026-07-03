#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <netcdf>

namespace citlali::pipeline {

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

}  // namespace citlali::pipeline
