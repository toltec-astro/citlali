#pragma once

#include <mutex>
#include <condition_variable>

#include <citlali/core/engine/engine.h>
#include <citlali/core/pipeline/timestream_output_context.h>

using timestream::TCData;
using timestream::RTCProc;
using timestream::PTCProc;

// selects the type of TCData
using timestream::TCDataKind;

class Pointing: public Engine {
public:
    // fit parameters
    Eigen::MatrixXd params, perrors;
    Eigen::VectorXi fit_valid;

    // meta information for ppt table
    YAML::Node ppt_meta;

    // ppt header information
    std::vector<std::string> ppt_header = {
        "array",
        "amp",
        "amp_err",
        "x_t",
        "x_t_err",
        "y_t",
        "y_t_err",
        "a_fwhm",
        "a_fwhm_err",
        "b_fwhm",
        "b_fwhm_err",
        "angle",
        "angle_err",
        "sig2noise"
    };

    // ppt header units
    std::map<std::string,std::string> ppt_header_units;

    // initial setup for each obs
    void setup();

    // main grppi pipeline
    template <class KidsProc, class RawObs>
    void pipeline(KidsProc &, RawObs &);

    // run the reduction for the obs
    template <class KidsProc>
    auto run(KidsProc &,
             const citlali::pipeline::TimestreamOutputFlags &,
             const citlali::pipeline::TimestreamOutputWriters &);
    template <class CalibScan>
    bool write_pointing_rtc_outputs(
        TCData<TCDataKind::RTC, Eigen::MatrixXd> &rtcdata,
        TCData<TCDataKind::PTC, Eigen::MatrixXd> &ptcdata,
        TCData<TCDataKind::RTC, Eigen::MatrixXd> &rtc_outer_output,
        CalibScan &calib_scan,
        const citlali::pipeline::TimestreamOutputFlags &output_flags,
        const citlali::pipeline::TimestreamOutputWriters &output_writers,
        Eigen::Index rtc_scan_row,
        bool write_this_rtc,
        const std::string &map_grouping);
    template <class CalibScan>
    bool write_pointing_ptc_outputs(
        TCData<TCDataKind::PTC, Eigen::MatrixXd> &ptcdata,
        CalibScan &calib_scan,
        const citlali::pipeline::TimestreamOutputFlags &output_flags,
        const citlali::pipeline::TimestreamOutputWriters &output_writers,
        const std::string &map_grouping);
    template <class CalibScan>
    void populate_pointing_final_maps(
        TCData<TCDataKind::PTC, Eigen::MatrixXd> &ptcdata,
        CalibScan &calib_scan,
        Eigen::VectorXI &map_indices,
        const std::string &map_grouping,
        citlali::config::MapMethod mapmaking_method,
        bool make_maps,
        bool make_noise_maps);
    template <class CalibScan>
    void maybe_subtract_pointing_fruitloop_model(
        TCData<TCDataKind::PTC, Eigen::MatrixXd> &ptcdata,
        CalibScan &calib_scan,
        Eigen::VectorXI &map_indices,
        const std::string &map_grouping,
        const citlali::pipeline::FruitLoopWeightPolicy &fruit_weight_policy);
    template <class CalibScan>
    void run_pointing_fruitloop_noise_pass(
        TCData<TCDataKind::PTC, Eigen::MatrixXd> &ptcdata,
        CalibScan &calib_scan,
        Eigen::VectorXI &map_indices,
        const std::string &map_grouping,
        citlali::config::MapMethod mapmaking_method,
        bool make_maps,
        bool make_noise_maps,
        const citlali::pipeline::FruitLoopWeightPolicy &fruit_weight_policy);

    // fit the maps
    void fit_maps();

    // output files
    Eigen::MatrixXf make_pointing_ppt_table(mapmaking::MapBuffer *mb);
    void add_pointing_fit_header_keys(CCfits::ExtHDU &hdu,
                                      const Eigen::MatrixXf &ppt_table,
                                      Eigen::Index map_row);
    template <typename FitsIoVector>
    void write_pointing_map_fits_products(FitsIoVector *f_io,
                                          FitsIoVector *n_io,
                                          mapmaking::MapBuffer *mb,
                                          const Eigen::MatrixXf &ppt_table);
    template <mapmaking::MapType map_type>
    void output();
};


#include <citlali/core/engine/detail/pointing_setup_impl.h>
#include <citlali/core/engine/detail/pointing_pipeline_impl.h>
#include <citlali/core/engine/detail/pointing_timestream_output_impl.h>
#include <citlali/core/engine/detail/pointing_map_population_impl.h>
#include <citlali/core/engine/detail/pointing_fruitloop_impl.h>
#include <citlali/core/engine/detail/pointing_run_impl.h>
#include <citlali/core/engine/detail/pointing_fit_maps_impl.h>
#include <citlali/core/engine/detail/pointing_output_impl.h>
