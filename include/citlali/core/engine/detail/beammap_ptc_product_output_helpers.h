#pragma once

// Beammap PTC product output helpers.

#include <netcdf>

#include <exception>
#include <map>
#include <stdexcept>
#include <string>

namespace beammap_ptc_product_output_helpers {

template <class Logger>
void update_ptc_tod_fruitloops_iter(
    const std::map<std::string, std::string> &tod_filename,
    int output_iter,
    const Logger &) {
    auto ptc_filename_it = tod_filename.find("ptc");
    if (ptc_filename_it == tod_filename.end() ||
        ptc_filename_it->second.empty()) {
        throw std::runtime_error(
            "processed TOD output is enabled but the PTC TOD filename is unavailable");
    }

    try {
        netCDF::NcFile ptc_tod_file(
            ptc_filename_it->second, netCDF::NcFile::write);
        netCDF::NcVar fruit_iter_var =
            ptc_tod_file.getVar("FRUITLOOPS_ITER");
        if (!fruit_iter_var.isNull()) {
            fruit_iter_var.putVar(&output_iter);
        }
        else {
            throw std::runtime_error(
                "required PTC TOD file has no FRUITLOOPS_ITER variable");
        }
    }
    catch (const std::exception &e) {
        throw std::runtime_error(
            "failed to update required PTC TOD FRUITLOOPS_ITER in " +
            ptc_filename_it->second + ": " + e.what());
    }
}

} // namespace beammap_ptc_product_output_helpers
