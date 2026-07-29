#pragma once

#include <netcdf>
#include <tula/eigen.h>

#include <cstddef>
#include <vector>

namespace citlali::pipeline {

// Raw TolTEC input schema for Header.Toltec.AdcSnapData.
//
// The NetCDF variable is [file_boundary, adc_sample].  The producer-confirmed
// boundary ordering is beginning then ending; it is not an ADC-channel axis.
// Values are signed 12-bit ADC counts stored in an ncShort container.
inline constexpr const char *adc_snap_variable_name =
    "Header.Toltec.AdcSnapData";
inline constexpr std::size_t adc_snap_boundary_count = 2;
inline constexpr std::size_t adc_snap_sample_count = 4096;
inline constexpr short adc_snap_min_count = -2048;
inline constexpr short adc_snap_max_count = 2047;

enum class AdcSnapBoundary : std::size_t {
    beginning = 0,
    ending = 1,
};

inline Eigen::Matrix<short,Eigen::Dynamic, Eigen::Dynamic>
read_adc_snap_matrix(netCDF::NcFile &fo) {
    const auto adc_snap_var = fo.getVar(adc_snap_variable_name);
    const Eigen::Index boundary_count = static_cast<Eigen::Index>(
        adc_snap_var.getDim(0).getSize());
    const Eigen::Index sample_count = static_cast<Eigen::Index>(
        adc_snap_var.getDim(1).getSize());

    Eigen::Matrix<short,Eigen::Dynamic, Eigen::Dynamic> adcsnap(
        sample_count, boundary_count);
    adc_snap_var.getVar(adcsnap.data());
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
