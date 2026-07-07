#pragma once

// Beammap PTC product output helpers.

#include <netcdf>

#include <exception>
#include <map>
#include <string>

namespace beammap_ptc_product_output_helpers {

template <class Logger>
void update_ptc_tod_fruitloops_iter(
    const std::map<std::string, std::string> &tod_filename,
    int output_iter,
    const Logger &logger) {
    auto ptc_filename_it = tod_filename.find("ptc");
    if (ptc_filename_it == tod_filename.end() ||
        ptc_filename_it->second.empty()) {
        return;
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
            logger->warn("PTC TOD file {} has no FRUITLOOPS_ITER variable",
                         ptc_filename_it->second);
        }
    }
    catch (const std::exception &e) {
        logger->warn("failed to update PTC TOD FRUITLOOPS_ITER in {}: {}",
                     ptc_filename_it->second, e.what());
    }
}

} // namespace beammap_ptc_product_output_helpers
