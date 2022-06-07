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
#include <citlali/core/utils/diagnostics.h>

#include <citlali/core/utils/toltec_io.h>
#include <citlali/core/utils/fits_io.h>

#include <citlali/core/timestream/timestream.h>

#include <citlali/core/timestream/rtc/polarization.h>
#include <citlali/core/timestream/rtc/kernel.h>
#include <citlali/core/timestream/rtc/despike.h>
#include <citlali/core/timestream/rtc/filter.h>
#include <citlali/core/timestream/rtc/downsample.h>
#include <citlali/core/timestream/rtc/calibrate.h>

#include <citlali/core/timestream/ptc/clean.h>
#include <citlali/core/timestream/ptc/sensitivity.h>

#include <citlali/core/timestream/rtc/rtcproc.h>
#include <citlali/core/timestream/ptc/ptcproc.h>

#include <citlali/core/map/naive_mm.h>
#include <citlali/core/map/jinc_mm.h>
#include <citlali/core/map/wiener_filter.h>

struct reduControls {
    // reduction sub directory
    bool use_subdir;

    // timestream controls
    bool run_timestream;
    bool run_polarization;
    bool run_despike;
    bool run_kernel;
    bool run_filter;
    bool run_downsample;
    bool run_clean;

    // output timestreams?
    bool run_tod_output;

    // create maps?
    bool run_maps;

    // fit maps?
    bool run_fitting;

    // do coaddition?
    bool run_coadd;
    // create noise maps?
    bool run_noise;
    // filter coadded maps?
    bool run_coadd_filter;
};

struct reduClasses {
    // rtc classes
    timestream::Polarization polarization;
    timestream::Kernel kernel;
    timestream::Despiker despiker;
    timestream::Filter filter;
    timestream::Downsampler downsampler;

    // ptc classes
    timestream::Cleaner cleaner;

    // map classes
    mapmaking::WienerFilter wiener_filter;
};

class EngineBase: public reduControls, public reduClasses, public Telescope, public Observation, public mapmaking::MapBase, public Calib {
public:
    using key_vec_t = std::vector<std::vector<std::string>>;

    // class for outputs
    ToltecIO toltec_io;

    // citlali config file
    tula::config::YamlConfig engine_config;

    // path to output files and directories
    std::string filepath;

    // path to timestream outuput file
    std::vector<std::string> ts_filepath;

    std::string beammap_source_name;
    double beammap_ra, beammap_dec;

    std::map<std::string,double> beammap_fluxes, beammap_uncer;

    std::string extinction_model;
    double tau;

    std::map<std::string,double> pointing_offsets;

    // format for output timestream file
    std::string ts_format;

    // output rtc, ptc, or both
    std::string ts_chunk_type;

    // timestream offsets
    std::vector<double> interface_sync_offset;

    // reduction number
    int redu_num;

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

    // current obs number
    int nobs;

    // sample rate
    double fsmp;

    // downsampled sample rate
    double dfsmp;

    // beammap config options included here
    // for initial validation
    int max_iterations;
    double cutoff;

    // jinc map-maker
    Eigen::VectorXd radius, jinc_weights;

    // size of fit bounding box
    double bounding_box_pix;

    // starting point for fit
    std::string fit_init_guess;

    // requested coadd filter type
    std::string coadd_filter_type;

    // gaussian template fwhm in radians
    std::map<std::string, double> gaussian_template_fwhm_rad;

    // weight type
    std::string weighting_type;

    // timestream output files
    std::vector<FitsIO<fileType::write_fits, CCfits::ExtHDU*>> ts_out_ios;
    std::vector<netCDF::NcFile*> ts_out_ncs;
    Eigen::Index ts_rows;

    // map output files
    std::vector<FitsIO<fileType::write_fits, CCfits::ExtHDU*>> fits_ios;
    std::vector<FitsIO<fileType::write_fits, CCfits::ExtHDU*>> coadd_fits_ios;
    std::vector<FitsIO<fileType::write_fits, CCfits::ExtHDU*>> noise_fits_ios;
    std::vector<FitsIO<fileType::write_fits, CCfits::ExtHDU*>> filtered_coadd_fits_ios;
    std::vector<FitsIO<fileType::write_fits, CCfits::ExtHDU*>> filtered_noise_fits_ios;

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

        // make sure param is larger than minimum
        if (!min_val.empty()) {
            if (param < min_val.at(0)) {
                    invalid = true;
            }
        }

        // make sure param is smaller than maximum
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

        get_config(run_tod_output,std::tuple{"timestream","output","enabled"});
        if (run_tod_output) {
            get_config(ts_format,std::tuple{"timestream","output","format"});
            get_config(ts_chunk_type,std::tuple{"timestream","output","chunk_type"});
        }

        // get runtime config options
        get_config(ex_name,std::tuple{"runtime","parallel_policy"});//,{"omp","seq","tbb"});

        if (run_tod_output) {
            if (ex_name != "seq") {
                SPDLOG_INFO("timestream output requires sequential execution policy.  setting policy to seq");
                ex_name = "seq";
            }
        }

        get_config(nthreads,std::tuple{"runtime","n_threads"});
        get_config(filepath,std::tuple{"runtime","output_dir"});
        get_config(reduction_type,std::tuple{"runtime","reduction_type"},{"science","pointing","beammap"});
        get_config(use_subdir,std::tuple{"runtime","use_subdir"});

        get_config(time_chunk,std::tuple{"timestream","chunking","length_sec"});
        get_config(weighting_type,std::tuple{"timestream","weighting","type"});

        // get config
        get_config(run_polarization,std::tuple{"timestream","polarimetry","enabled"});

        if (run_polarization == false) {
            polarization.stokes_params = {{"I",0}};
        }

        else {
            polarization.stokes_params = {{"I",0},
                                          {"Q",1},
                                          {"U",2}};
        }

        // get despike config options
        get_config(run_despike,std::tuple{"timestream","despike","enabled"});
        if (run_despike) {
            get_config(despiker.sigma,std::tuple{"timestream","despike","min_spike_sigma"});
            get_config(despiker.time_constant,std::tuple{"timestream","despike","time_constant_sec"});
	    despiker.fsmp = fsmp;
        }

        // get filter config options
        get_config(run_filter,std::tuple{"timestream","filter","enabled"});
        if (run_filter) {
            get_config(filter.a_gibbs,std::tuple{"timestream","filter","a_gibbs"});
            get_config(filter._flow,std::tuple{"timestream","filter","freq_low_Hz"});
            get_config(filter._fhigh,std::tuple{"timestream","filter","freq_high_Hz"});
            get_config(filter.nterms,std::tuple{"timestream","filter","n_terms"});

            // filter is run before downsampling
            filter.fsmp = fsmp;

            // if filter is requested, set the despike window to the filter window
            despiker.run_filter = true;
            despiker.despike_window = filter.nterms;
        }

        else {
            // controls inner scan start index
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
            get_config(kernel.kernel_type,std::tuple{"timestream","kernel","type"},{"internal_gaussian","internal_airy","image"});

            auto kernel_node = engine_config.get_node(std::tuple{"timestream","kernel","image_ext_name"});
            auto kernel_node_size = kernel_node.size();

            get_config(kernel.hdu_name,std::tuple{"timestream","kernel","image_ext_name"});
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

        if (mapping_method == "jinc") {
            double r_max = 1.5;
            double a = 1.1;
            double b = 4.75;
            double c = 2.;

            radius = Eigen::VectorXd::LinSpaced(1000, 1e-10, r_max);
            jinc_weights.resize(radius.size());

            for (Eigen::Index i=0; i<radius.size(); i++) {
                jinc_weights(i) = jinc_func(radius(i),a,b,c,r_max,1);
            }
        }

        get_config(map_type,std::tuple{"mapmaking","pixel_axes"});
        get_config(pixel_size,std::tuple{"mapmaking","pixel_size_arcsec"});
        get_config(cmb.cov_cut,std::tuple{"coadd","cov_cut"});

        // convert pixel size to radians at start
        pixel_size *= ASEC_TO_RAD;
        // assign pixel size to map buffer and coadded map buffer for convenience
        mb.pixel_size = pixel_size;
        cmb.pixel_size = pixel_size;

        // override default map parameters
        get_config(crval1_J2000,std::tuple{"mapmaking","crval1_J2000"});
        get_config(crval2_J2000,std::tuple{"mapmaking","crval2_J2000"});
        get_config(x_size_pix,std::tuple{"mapmaking","x_size_pix"});
        get_config(y_size_pix,std::tuple{"mapmaking","y_size_pix"});
        get_config(crpix1,std::tuple{"mapmaking","crpix1"});
        get_config(crpix2,std::tuple{"mapmaking","crpix2"});
        get_config(cunit,std::tuple{"mapmaking","cunit"},{"MJy/Sr","mJy/beam", "uK/arcmin^2"});

        // get beammap config options
        get_config(cutoff,std::tuple{"beammap","iter_tolerance"});
        get_config(max_iterations,std::tuple{"beammap","iter_max"});

        // check if point source fitting is requested
        //get_config(run_fitting,std::tuple{"source_fitting","enabled"});
         //if (run_fitting) {
        get_config(fit_model,std::tuple{"source_fitting","model"});
        get_config(bounding_box_pix,std::tuple{"source_fitting","bounding_box_arcsec"});
        bounding_box_pix = std::floor(bounding_box_pix/pixel_size*ASEC_TO_RAD);
        get_config(fit_init_guess,std::tuple{"source_fitting","initial_guess"});
        //}

        // get coadd config options
        get_config(run_coadd,std::tuple{"coadd","enabled"});
        if (run_coadd) {
            get_config(cmb.cov_cut,std::tuple{"coadd","cov_cut"});
            cmb.pixel_size = pixel_size;

            // get noise config options
            get_config(run_noise,std::tuple{"coadd","noise_maps","enabled"});
            if (run_noise) {
                get_config(cmb.nnoise,std::tuple{"coadd","noise_maps","n_noise_maps"});
            }

            else {
                cmb.nnoise = 0;
            }

            // get coadd filter config options
            get_config(run_coadd_filter,std::tuple{"coadd","filtering","enabled"});
            if (run_coadd_filter) {
                get_config(coadd_filter_type,std::tuple{"coadd","filtering","type"});

                if (run_noise == false) {
                    SPDLOG_ERROR("noise maps are needed for map filtering.");
                    std::exit(EXIT_FAILURE);
                }

                get_config(wiener_filter.run_gaussian_template,std::tuple{"wiener_filter","gaussian_template"});
                get_config(wiener_filter.run_lowpass_only,std::tuple{"wiener_filter","lowpass_only"});
                get_config(wiener_filter.run_highpass_only,std::tuple{"wiener_filter","highpass_only"});
                get_config(wiener_filter.normalize_error,std::tuple{"wiener_filter","normalize_error"});

                get_config(gaussian_template_fwhm_rad["a1100"],std::tuple{"wiener_filter","gaussian_template_fwhm_arcsec","a1100"});
                get_config(gaussian_template_fwhm_rad["a1400"],std::tuple{"wiener_filter","gaussian_template_fwhm_arcsec","a1400"});
                get_config(gaussian_template_fwhm_rad["a2000"],std::tuple{"wiener_filter","gaussian_template_fwhm_arcsec","a2000"});

                for (auto const& pair : gaussian_template_fwhm_rad) {
                    gaussian_template_fwhm_rad[pair.first] = gaussian_template_fwhm_rad[pair.first]*ASEC_TO_RAD;
                }

                wiener_filter.run_kernel = run_kernel;
            }
        }
    }
};
