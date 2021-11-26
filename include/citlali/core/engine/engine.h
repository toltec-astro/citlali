#pragma once

#include <memory>
#include <string>
#include <tula/config/yamlconfig.h>
#include <tula/config/flatconfig.h>
#include <omp.h>

#include <citlali/core/engine/telescope.h>
#include <citlali/core/engine/observation.h>
#include <citlali/core/map/map.h>
#include <citlali/core/engine/calib.h>
#include <citlali/core/utils/utils.h>

#include <citlali/core/utils/toltec_io.h>
#include <citlali/core/utils/fits_io.h>

#include <citlali/core/timestream/timestream.h>

#include <citlali/core/timestream/rtc/kernel.h>
#include <citlali/core/timestream/rtc/despike.h>
#include <citlali/core/timestream/rtc/filter.h>
#include <citlali/core/timestream/rtc/downsample.h>
#include <citlali/core/timestream/rtc/calibrate.h>

#include <citlali/core/timestream/ptc/clean.h>

#include <citlali/core/timestream/rtc/rtcproc.h>
#include <citlali/core/timestream/ptc/ptcproc.h>

#include <citlali/core/map/naive_mm.h>

class EngineBase: public Telescope, public Observation, public MapBase, public Calib {
public:

    using key_vec_t = std::vector<std::vector<std::string>>;

    // citlali config file
    tula::config::YamlConfig engine_config;

    std::string filepath;

    // vectors to hold missing/invalid keys
    key_vec_t missing_keys;
    key_vec_t invalid_keys;

    // grppi execution name
    std::string ex_name;

    // reduction type
    std::string reduction_type;

    // requested fitting model type
    std::string fit_model;

    // number of cores to parallelize over
    int nthreads;

    // sample rate
    double fsmp;

    // controls for each stage
    bool run_despike, run_kernel, run_filter,
    run_downsample, run_clean;

    // beammap config options included here
    // for initial validation
    int max_iterations;
    double cutoff;

    // control for fitting
    bool run_fit;

    // size of fit bounding box
    double bounding_box;

    // starting point for fit
    std::string fit_init_guess;

    // control for coadd
    bool run_coadd;

    // control for noise maps
    bool run_noise;

    // control for coadd filter
    bool run_coadd_filter;

    // requested coadd filter type
    std::string coadd_filter_type;

    // kernel class
    timestream::Kernel kernel;

    // despike class
    timestream::Despiker despiker;

    // lowpass and highpass class
    timestream::Filter filter;

    // downsample class
    timestream::Downsampler downsampler;

    // pca clean class
    timestream::Cleaner cleaner;

    //WienerFilter wiener_filter;

    // weight type
    std::string approx_weights;

    // Output files
    std::vector<FitsIO<fileType::write_fits, CCfits::ExtHDU*>> fits_ios;
    std::vector<FitsIO<fileType::write_fits, CCfits::ExtHDU*>> coadd_fits_ios;
    std::vector<FitsIO<fileType::write_fits, CCfits::ExtHDU*>> noise_fits_ios;

    template<typename param_t, typename option_t>
    void check_allowed(param_t param, std::vector<param_t> allowed, option_t option) {
        // loop through allowed values and see if param is contained within it
        if (!std::any_of(allowed.begin(), allowed.end(), [&](const auto i){return i==param;})) {
            // temporary vector to hold current invalid param's keys
            std::vector<std::string> invalid_temp;
            // push back invalid keys into temp vector
            engine_utils::for_each_in_tuple(option, [&](const auto &x) {
                invalid_temp.push_back(x);
            });

            // push temp invalid keys vector into invalid keys vector
            invalid_keys.push_back(invalid_temp);
        }
    }
    template<typename param_t, typename option_t>
    void check_range(param_t param, std::vector<param_t> min_val,  std::vector<param_t> max_val,
                     option_t option) {

        bool invalid = false;

        // check if param is larger than minimum
        if (!min_val.empty()) {
            if (param < min_val.at(0)) {
                    invalid = true;
            }
        }

        // check if param is smaller than maximum
        if (!max_val.empty()) {
            if (param > max_val.at(0)) {
                    invalid = true;
            }
        }

        // if param is invalid
        if (invalid) {
            // temporary vector to hold current invalid param's keys
            std::vector<std::string> invalid_temp;
            // push back invalid keys into temp vector
            engine_utils::for_each_in_tuple(option, [&](const auto &x) {
                invalid_temp.push_back(x);
            });

            // push temp invalid keys vector into invalid keys vector
            invalid_keys.push_back(invalid_temp);
        }
    }

    template <typename param_t, typename option_t>
    void get_config(param_t &param, option_t option, std::vector<param_t> allowed={},
                      std::vector<param_t> min_val={}, std::vector<param_t> max_val={}) {

        // check if config option exists
        if (engine_config.has(option_t(option))) {
            // if it exists, try to get it with intended type
            try {
                // get the parameter from config
                param = engine_config.get_typed<param_t>(option_t(option));

                // if allowed values is specified, check against them
                if (!allowed.empty()) {
                    check_allowed(param, allowed, option);
                }

                // if a range is specified, check against them
                if (!min_val.empty() || !max_val.empty()) {
                    check_range(param, min_val, max_val, option);
                }

               // mark as invalid if get_typed fails
            }  catch (YAML::TypedBadConversion<param_t>) {

                // temporary vector to hold current invalid param's keys
                std::vector<std::string> invalid_temp;
                // push back invalid keys into temp vector
                engine_utils::for_each_in_tuple(option, [&](const auto &x) {
                    invalid_temp.push_back(x);
                });

                // push temp invalid keys vector into invalid keys vector
                invalid_keys.push_back(invalid_temp);
            }
        }
        // else mark as missing
        else {
            // temporary vector to hold current missing param's keys
            std::vector<std::string> missing_temp;
            // push back missing keys into temp vector
            engine_utils::for_each_in_tuple(option, [&](const auto &x) {
                missing_temp.push_back(x);
            });
            // push temp missing keys vector into invalid keys vector
            missing_keys.push_back(missing_temp);
        }
    }

    void from_config(tula::config::YamlConfig _c) {
        engine_config = _c;
        SPDLOG_INFO("getting config options");

        // get runtime config options
        get_config(ex_name,std::tuple{"runtime","parallel_policy"},{"omp","seq","tbb"});
        get_config(nthreads,std::tuple{"runtime","n_threads"});
        get_config(filepath,std::tuple{"runtime","output_dir"});
        get_config(reduction_type,std::tuple{"runtime","reduction_type"},{"science","pointing","beammap"});

        get_config(time_chunk,std::tuple{"timestream","chunking","length_sec"});
        get_config(approx_weights,std::tuple{"timestream","weighting","type"});

        // get despike config options
        get_config(run_despike,std::tuple{"timestream","despike","enabled"});
        if (run_despike) {
            get_config(despiker.sigma,std::tuple{"timestream","despike","min_spike_sigma"});
            get_config(despiker.time_constant,std::tuple{"timestream","despike","time_constant_sec"});
        }

        // get filter config options
        get_config(run_filter,std::tuple{"timestream","filter","enabled"});
        if (run_filter) {
            get_config(filter.a_gibbs,std::tuple{"timestream","filter","a_gibbs"});
            get_config(filter._flow,std::tuple{"timestream","filter","freq_low_Hz"});
            get_config(filter._fhigh,std::tuple{"timestream","filter","freq_high_Hz"});
            get_config(filter.nterms,std::tuple{"timestream","filter","n_terms"});

            filter.fsmp = fsmp;

            // if filter is requested, set the despike window to the filter window
            despiker.run_filter = true;
            despiker.despike_window = filter.nterms;
        }

        else {
            filter.nterms = 0;
        }

        // get downsample config options
        get_config(run_downsample,std::tuple{"timestream","downsample","enabled"});
        if (run_downsample) {
            get_config(downsampler.dsf,std::tuple{"timestream","downsample","factor"});
        }

        // get kernel config options
        get_config(run_kernel,std::tuple{"timestream","kernel","enabled"});
        if (run_kernel) {
            get_config(kernel.filepath,std::tuple{"timestream","kernel","filepath"});
            get_config(kernel.kernel_type,std::tuple{"timestream","kernel","type"});
            kernel.setup();
        }

        // get cleaning config options
        get_config(run_clean,std::tuple{"timestream","clean","enabled"});
        if (run_clean) {
            get_config(cleaner.neig,std::tuple{"timestream","clean","n_eig_to_cut"});
            get_config(cleaner.grouping,std::tuple{"timestream","clean","grouping"});
            get_config(cleaner.cut_std,std::tuple{"timestream","clean","cut_std"});
        }

        // get mapmaking config options
        get_config(map_grouping,std::tuple{"mapmaking","grouping"});
        get_config(mapping_method,std::tuple{"mapmaking","method"});
        get_config(map_type,std::tuple{"mapmaking","pixel_axes"});
        get_config(pixel_size,std::tuple{"mapmaking","pixel_size_arcsec"});

        // convert pixel size to radians at start
        pixel_size *= RAD_ASEC;
        // assign pixel size to map buffer and coadded map buffer for convenience
        mb.pixel_size = pixel_size;
        cmb.pixel_size = pixel_size;

        // override default map parameters
        get_config(crval1_J2000,std::tuple{"mapmaking","crval1_J2000"});
        get_config(crval2_J2000,std::tuple{"mapmaking","crval2_J2000"});
        get_config(x_size_pix,std::tuple{"mapmaking","x_size_pix"});
        get_config(y_size_pix,std::tuple{"mapmaking","y_size_pix"});

        // get beammap config options
        get_config(cutoff,std::tuple{"beammap","iter_tolerance"});
        get_config(max_iterations,std::tuple{"beammap","iter_max"});

        // check if point source fitting is requested
        get_config(run_fit,std::tuple{"source_fitting","enabled"});
         if (run_fit) {
            get_config(fit_model,std::tuple{"source_fitting","model"});
            get_config(bounding_box,std::tuple{"source_fitting","bounding_box_arcsec"});
            get_config(fit_init_guess,std::tuple{"source_fitting","initial_guess"});
        }

        // get coadd config options
        get_config(run_coadd,std::tuple{"coadd","enabled"});
        if (run_coadd) {
            cmb.pixel_size = pixel_size;

            get_config(run_noise,std::tuple{"coadd","noise_maps","enabled"});
            if (run_noise) {
                get_config(cmb.nnoise,std::tuple{"coadd","noise_maps","n_noise_maps"});
            }

            else {
                cmb.nnoise = 0;
            }
            get_config(run_coadd_filter,std::tuple{"coadd","filtering","enabled"});
            if (run_coadd_filter) {
                get_config(coadd_filter_type,std::tuple{"coadd","filtering","type"});

                //get_config(wiener_filter.gauss_template,std::tuple{"wiener_filter","gaussian_template"});
                //get_config(wiener_filter.gaussian_template_fwhm_arcsec,std::tuple{"wiener_filter","gaussian_template_fwhm_arcsec"});
                //get_config(wiener_filter.lowpass_only,std::tuple{"wiener_filter","lowpass_only"});
                //get_config(wiener_filter.highpass_only,std::tuple{"wiener_filter","highpass_only"});
                //get_config(wiener_filter.normalize_error,std::tuple{"wiener_filter","normalize_error"});
            }
        }
    }
};
