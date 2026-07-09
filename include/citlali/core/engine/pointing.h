#pragma once

#include <mutex>
#include <condition_variable>

#include <citlali/core/engine/engine.h>

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
    auto run(KidsProc &);

    // fit the maps
    void fit_maps();

    // output files
    Eigen::MatrixXf make_pointing_ppt_table(mapmaking::MapBuffer *mb);
    void add_pointing_fit_header_keys(CCfits::ExtHDU &hdu,
                                      const Eigen::MatrixXf &ppt_table,
                                      Eigen::Index map_row);
    template <mapmaking::MapType map_type>
    void output();
};


#include <citlali/core/engine/detail/pointing_setup_impl.h>
#include <citlali/core/engine/detail/pointing_pipeline_impl.h>
#include <citlali/core/engine/detail/pointing_run_impl.h>
#include <citlali/core/engine/detail/pointing_fit_maps_impl.h>
#include <citlali/core/engine/detail/pointing_output_impl.h>
