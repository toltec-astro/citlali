#pragma once

#include <string>

#include <citlali/core/mapmaking/map.h>
#include <citlali/core/utils/toltec_io.h>

namespace citlali::pipeline {

template <mapmaking::MapType map_t, engine_utils::toltecIO::DataType data_t,
          engine_utils::toltecIO::ProdType prod_t, class ToltecIO>
std::string map_output_filename(ToltecIO &toltec_io, const std::string &dir_name,
                                const std::string &redu_type,
                                const std::string &obsnum, bool sim_obs) {
    if constexpr (map_t == mapmaking::RawObs) {
        return toltec_io.template create_filename<
            data_t, prod_t, engine_utils::toltecIO::raw>(
            dir_name, redu_type, "", obsnum, sim_obs);
    }
    else if constexpr (map_t == mapmaking::FilteredObs) {
        return toltec_io.template create_filename<
            data_t, prod_t, engine_utils::toltecIO::filtered>(
            dir_name, redu_type, "", obsnum, sim_obs);
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
