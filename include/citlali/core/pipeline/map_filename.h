#pragma once

#include <string>

#include <citlali/core/config/runtime_config.h>
#include <citlali/core/mapmaking/map.h>
#include <citlali/core/utils/toltec_io.h>

namespace citlali::pipeline {

template <mapmaking::MapType map_t, engine_utils::toltecIO::DataType data_t,
          engine_utils::toltecIO::ProdType prod_t, class ToltecIO>
std::string map_output_filename(ToltecIO &toltec_io, const std::string &dir_name,
                                citlali::config::ReductionType reduction_type,
                                const std::string &obsnum, bool sim_obs) {
    if constexpr (map_t == mapmaking::RawObs) {
        const std::string reduction_type_name{
            citlali::config::to_string(reduction_type)};
        return toltec_io.template create_filename<
            data_t, prod_t, engine_utils::toltecIO::raw>(
            dir_name, reduction_type_name, "", obsnum, sim_obs);
    }
    else if constexpr (map_t == mapmaking::FilteredObs) {
        const std::string reduction_type_name{
            citlali::config::to_string(reduction_type)};
        return toltec_io.template create_filename<
            data_t, prod_t, engine_utils::toltecIO::filtered>(
            dir_name, reduction_type_name, "", obsnum, sim_obs);
    }
    else if constexpr (map_t == mapmaking::RawCoadd) {
        return toltec_io.template create_filename<
            data_t, prod_t, engine_utils::toltecIO::raw>(
            dir_name, "", "", "", sim_obs);
    }
    else if constexpr (map_t == mapmaking::FilteredCoadd) {
        return toltec_io.template create_filename<
            data_t, prod_t, engine_utils::toltecIO::filtered>(
            dir_name, "", "", "", sim_obs);
    }
    else {
        return "";
    }
}

}  // namespace citlali::pipeline
