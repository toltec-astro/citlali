#pragma once

// Implementation detail included by todproc.h.

template <class EngineType>
void TimeOrderedDataProc<EngineType>::get_adc_snap_from_files(const RawObs &rawobs) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    // explicitly clear adc vector
    engine().diagnostics.adc_snap_data.clear();

    // loop through input files
    for (const RawObs::DataItem &data_item : rawobs.kidsdata()) {
        try {
            // load data file
            NcFile fo(data_item.filepath(), NcFile::read);

            // dimension 0 of adc data
            Eigen::Index adcSnapDim = fo.getVar("Header.Toltec.AdcSnapData").getDim(0).getSize();
            // dimension 1 of adc data
            Eigen::Index adcSnapDataDim = fo.getVar("Header.Toltec.AdcSnapData").getDim(1).getSize();

            // matrix to hold adc data for current file
            Eigen::Matrix<short,Eigen::Dynamic, Eigen::Dynamic> adcsnap(adcSnapDataDim,adcSnapDim);
            // load adc data
            fo.getVar("Header.Toltec.AdcSnapData").getVar(adcsnap.data());
            // append to vector of adc data
            engine().diagnostics.adc_snap_data.push_back(adcsnap);

            fo.close();

        } catch (NcException &e) {
            logger->warn("{} adc data not found",data_item.filepath());
        }
    }
}
