#pragma once

#include <netcdf>
#include <tula/eigen.h>

#include <vector>

namespace citlali::pipeline {

inline Eigen::Matrix<short,Eigen::Dynamic, Eigen::Dynamic>
read_adc_snap_matrix(netCDF::NcFile &fo) {
    const Eigen::Index adc_snap_dim = static_cast<Eigen::Index>(
        fo.getVar("Header.Toltec.AdcSnapData").getDim(0).getSize());
    const Eigen::Index adc_snap_data_dim = static_cast<Eigen::Index>(
        fo.getVar("Header.Toltec.AdcSnapData").getDim(1).getSize());

    Eigen::Matrix<short,Eigen::Dynamic, Eigen::Dynamic> adcsnap(
        adc_snap_data_dim, adc_snap_dim);
    fo.getVar("Header.Toltec.AdcSnapData").getVar(adcsnap.data());
    return adcsnap;
}

template <class RawObs, class AdcSnapData, class Logger>
void read_rawobs_adc_snap_data(const RawObs &rawobs,
                               AdcSnapData &adc_snap_data,
                               const Logger &logger) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    adc_snap_data.clear();
    for (const typename RawObs::DataItem &data_item : rawobs.kidsdata()) {
        try {
            NcFile fo(data_item.filepath(), NcFile::read);
            adc_snap_data.push_back(read_adc_snap_matrix(fo));
            fo.close();
        }
        catch (NcException &e) {
            logger->warn("{} adc data not found", data_item.filepath());
        }
    }
}

}  // namespace citlali::pipeline
