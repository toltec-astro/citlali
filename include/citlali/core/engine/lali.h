#pragma once

#include <functional>
#include <mutex>
#include <condition_variable>

#include <citlali/core/engine/engine.h>
#include <citlali/core/pipeline/timestream_output_context.h>

using timestream::TCData;
using timestream::RTCProc;
using timestream::PTCProc;

// selects the type of TCData
using timestream::TCDataKind;

class Lali: public Engine {
public:
    using FitsOutput = fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>;
    using input_t = TCData<TCDataKind::RTC, Eigen::MatrixXd>;
    using run_stage_t = decltype(grppi::farm(
        std::declval<int>(), std::declval<std::function<void(input_t &)>>() ));

    // initial setup for each obs
    void setup();

    // main grppi pipeline
    template <class KidsProc, class RawObs>
    void pipeline(KidsProc &, RawObs &);

    // run the reduction for the obs
    auto run() -> run_stage_t;
    template <class CalibScan>
    void write_lali_rtc_outputs(
        input_t &rtcdata,
        TCData<TCDataKind::PTC, Eigen::MatrixXd> &ptcdata,
        input_t &rtc_outer_output,
        CalibScan &calib_scan,
        const citlali::pipeline::TimestreamOutputFlags &output_flags,
        const citlali::pipeline::TimestreamOutputWriters &output_writers,
        Eigen::Index rtc_scan_row,
        bool write_this_rtc,
        const std::string &map_grouping);
    template <class CalibScan>
    void write_lali_ptc_outputs(
        TCData<TCDataKind::PTC, Eigen::MatrixXd> &ptcdata,
        CalibScan &calib_scan,
        const citlali::pipeline::TimestreamOutputFlags &output_flags,
        const citlali::pipeline::TimestreamOutputWriters &output_writers,
        const std::string &map_grouping);
    template <class CalibScan>
    void populate_lali_final_maps(
        TCData<TCDataKind::PTC, Eigen::MatrixXd> &ptcdata,
        CalibScan &calib_scan,
        Eigen::VectorXI &map_indices,
        const std::string &map_grouping,
        citlali::config::MapMethod mapmaking_method,
        bool make_maps,
        bool make_noise_maps);

    // output files
    void finalize_lali_map_fits_outputs(
        std::vector<FitsOutput> &data_outputs,
        std::vector<FitsOutput> &noise_outputs);
    template <typename FitsIoVector>
    void write_lali_map_fits_products(FitsIoVector *f_io,
                                      FitsIoVector *n_io,
                                      mapmaking::MapBuffer *mb);
    template <mapmaking::MapType map_type>
    void output();
};


#include <citlali/core/engine/detail/lali_setup_pipeline_impl.h>
#include <citlali/core/engine/detail/lali_timestream_output_impl.h>
#include <citlali/core/engine/detail/lali_map_population_impl.h>
#include <citlali/core/engine/detail/lali_run_impl.h>
#include <citlali/core/engine/detail/lali_output_impl.h>
