#pragma once

#include <memory>
#include <string>
#include <vector>
#include <omp.h>
#include <Eigen/Core>
#include <fstream>

#include <citlali_config/config.h>
#include <citlali_config/gitversion.h>
#include <citlali_config/default_config.h>
#include <kids/core/kidsdata.h>
#include <kids/sweep/fitter.h>
#include <kids/timestream/solver.h>
#include <kids/toltec/toltec.h>
#include <kidscpp_config/gitversion.h>
#include <tula/cli.h>
#include <tula/config/core.h>
#include <tula/config/flatconfig.h>
#include <tula/config/yamlconfig.h>
#include <tula/enum.h>
#include <tula/filesystem.h>
#include <tula/formatter/container.h>
#include <tula/formatter/enum.h>
#include <tula/grppi.h>
#include <tula/logging.h>
#include <tula/switch_invoke.h>

#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/fits_io.h>
#include <citlali/core/utils/toltec_io.h>
#include <citlali/core/utils/gauss_models.h>
#include <citlali/core/utils/fitting.h>

#include <citlali/core/engine/config.h>

#include <citlali/core/engine/calib.h>
#include <citlali/core/engine/telescope.h>

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

#include <citlali/core/mapmaking/map.h>
#include <citlali/core/mapmaking/naive_mm.h>
#include <citlali/core/mapmaking/jinc_mm.h>
#include <citlali/core/mapmaking/wiener_filter.h>

#include <citlali/core/engine/io.h>
#include <citlali/core/engine/kidsproc.h>
#include <citlali/core/engine/todproc.h>

struct reduControls {
    // create reduction subdirectories
    bool use_subdir;

    // run or skip tod processing
    bool run_tod;

    // output timestreams
    bool run_tod_output;

    // controls for mapmaking
    bool run_mapmaking;
    bool run_coadd;
    bool run_noise;
    bool run_map_filter;

    // run source finding
    bool run_source_finder;
};

struct reduClasses {
    // reduction classes
    engine::Calib calib;
    engine::Telescope telescope;
    engine_utils::toltecIO toltec_io;
    engine_utils::mapFitter map_fitter;

    // rtc proc
    timestream::RTCProc rtcproc;

    // ptc proc
    timestream::PTCProc ptcproc;

    // map classes
    mapmaking::ObsMapBuffer omb;
    mapmaking::ObsMapBuffer cmb;
    mapmaking::WienerFilter wiener_filter;
};

struct beammapControls {
    // source name
    std::string beammap_source_name;

    // beammap source position
    double beammap_ra_rad, beammap_dec_rad;

    // fluxes and errs
    std::map<std::string, double> beammap_fluxes_mJy_beam, beammap_err_mJy_beam;
    std::map<std::string, double> beammap_fluxes_MJy_Sr, beammap_err_MJy_Sr;

    // maximum beammap iterations
    int beammap_iter_max;

    // beammap tolerance
    double beammap_iter_tolerance;

    // subtract reference detector
    bool beammap_subtract_reference;

    // beammap reference detector
    Eigen::Index beammap_reference_det;

    // derotate fitted detectors
    bool beammap_derotate;

    // iteration to write out ptcdata
    int beammap_tod_output_iter = 0;

    // upper and lower limits of psd for sensitivity calc
    Eigen::VectorXd sens_psd_limits;

    // limits on fwhm, sig2noise, and distance from center for flagging
    std::map<std::string, double> lower_fwhm_arcsec, upper_fwhm_arcsec, lower_sig2noise,
        max_dist_arcsec;

    // limits on sensitivity for flagging
    double lower_sens_factor, upper_sens_factor;
};

class Engine: public reduControls, public reduClasses, public beammapControls {
public:
    using key_vec_t = std::vector<std::vector<std::string>>;

    // add extra output for debugging
    bool verbose_mode;

    // time gaps
    std::map<std::string,int> gaps;

    // output directory and optional sub directory name
    std::string output_dir, redu_dir_name;

    // obsnum and coadded directory names
    std::string obsnum_dir_name, coadd_dir_name;

    // tod output file name
    std::map<std::string, std::string> tod_filename;

    // vectors to hold missing/invalid keys
    key_vec_t missing_keys, invalid_keys;

    // number of threads
    int n_threads;

    // parallel execution policy
    std::string parallel_policy;

    // number of scans completed
    Eigen::Index n_scans_done;

    // manual offsets for nws and hwp
    std::map<std::string,double> interface_sync_offset;

    // vectors for tod alignment offsets
    std::vector<Eigen::Index> start_indices, end_indices;

    // xs or rs
    std::string tod_type;

    // reduction type (science, pointing, beammap)
    std::string redu_type;

    // obsnum
    std::string obsnum;

    // weight type (approximate or full)
    std::string weighting_type;

    // rtc or ptc types
    std::string tod_output_type, tod_output_subdir_name;

    // map grouping and algorithm
    std::string map_grouping, map_method;

    // number of maps
    Eigen::Index n_maps;

    // jinc mapmaking params
    double jinc_r_max, jinc_a, jinc_b, jinc_c;

    // mapping from index in map vector to array index
    Eigen::VectorXI maps_to_arrays, arrays_to_maps;

    // mapping from index in map vector to array index
    Eigen::VectorXI maps_to_stokes;

    // manual pointing offsets
    std::map<std::string, double> pointing_offsets_arcsec;

    // map output files
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> fits_io_vec;
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> noise_fits_io_vec;
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> filtered_fits_io_vec;
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> filtered_noise_fits_io_vec;

    // coadded map output files
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> coadd_fits_io_vec;
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> coadd_noise_fits_io_vec;
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> filtered_coadd_fits_io_vec;
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> filtered_coadd_noise_fits_io_vec;


    template<typename CT>
    void get_rtc_config(CT &);

    template<typename CT>
    void get_ptc_config(CT &);

    template<typename CT>
    void get_beammap_config(CT &);

    template<typename CT>
    void get_mapmaking_config(CT &);

    template<typename CT>
    void get_map_filter_config(CT &);

    template<typename CT>
    void get_citlali_config(CT &);

    template<typename CT>
    void get_photometry_config(CT &);

    template<typename CT>
    void get_astrometry_config(CT &);

    template <typename Derived>
    auto calc_map_indices(Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &,
                          Eigen::DenseBase<Derived> &, std::string);

    void create_map_files();
    template <engine_utils::toltecIO::ProdType prod_t>
    void create_tod_files();

    //template <TCDataKind tc_t>
    //void print_summary(TCData<tc_t, Eigen::MatrixXd> &);

    void print_summary();

    template <TCDataKind tc_t>
    void write_chunk_summary(TCData<tc_t, Eigen::MatrixXd> &);

    template <typename map_buffer_t>
    void write_map_summary(map_buffer_t &);

    template <typename fits_io_type, class map_buffer_t>
    void add_phdu(fits_io_type &, map_buffer_t &, Eigen::Index);

    template <typename fits_io_type, class map_buffer_t>
    void write_maps(fits_io_type &, fits_io_type &, map_buffer_t &, Eigen::Index);

    template <mapmaking::MapType map_t, class map_buffer_t>
    void write_psd(map_buffer_t &, std::string);

    template <mapmaking::MapType map_t, class map_buffer_t>
    void write_hist(map_buffer_t &, std::string);

    template <mapmaking::MapType map_t, class map_buffer_t>
    void run_wiener_filter(map_buffer_t &);

    template <mapmaking::MapType map_t, class map_buffer_t>
    void find_sources(map_buffer_t &);

    template <mapmaking::MapType map_t, class map_buffer_t>
    void write_sources(map_buffer_t &, std::string);
};

template<typename CT>
void Engine::get_rtc_config(CT &config) {
    get_config_value(config, rtcproc.lower_std_dev, missing_keys, invalid_keys, std::tuple{"timestream","raw_time_chunk","flagging",
                                                                                           "lower_weight_factor"});
    get_config_value(config, rtcproc.upper_std_dev, missing_keys, invalid_keys, std::tuple{"timestream", "raw_time_chunk","flagging",
                                                                                           "upper_weight_factor"});
    get_config_value(config, rtcproc.delta_f_min_Hz, missing_keys, invalid_keys, std::tuple{"timestream","raw_time_chunk","flagging",
                                                                                            "delta_f_min_Hz"});

    /* polarization */
    get_config_value(config, rtcproc.run_polarization, missing_keys, invalid_keys, std::tuple{"timestream","polarimetry", "enabled"});

    if (!rtcproc.run_polarization) {
        rtcproc.polarization.stokes_params.clear();
        rtcproc.polarization.stokes_params[0] = "I";
    }

    /* kernel */
    get_config_value(config, rtcproc.run_kernel, missing_keys, invalid_keys, std::tuple{"timestream","raw_time_chunk","kernel","enabled"});
    if (rtcproc.run_kernel) {
        get_config_value(config, rtcproc.kernel.filepath, missing_keys, invalid_keys, std::tuple{"timestream","raw_time_chunk","kernel",
                                                                                                 "filepath"});
        get_config_value(config, rtcproc.kernel.type, missing_keys, invalid_keys, std::tuple{"timestream","raw_time_chunk","kernel","type"});
        get_config_value(config, rtcproc.kernel.fwhm_rad, missing_keys, invalid_keys, std::tuple{"timestream","raw_time_chunk","kernel",
                                                                                          "fwhm_arcsec"});

        rtcproc.kernel.fwhm_rad *=ASEC_TO_RAD;
        rtcproc.kernel.sigma_rad = rtcproc.kernel.fwhm_rad*FWHM_TO_STD;

        if (rtcproc.kernel.type == "fits") {
            auto img_ext_name_node = config.get_node(std::tuple{"timestream","raw_time_chunk","kernel", "image_ext_names"});
            for (Eigen::Index i=0; i<img_ext_name_node.size(); i++) {
                std::string img_ext_name = config.template get_str(std::tuple{"timestream","raw_time_chunk","kernel", "image_ext_names",
                                                                              i, std::to_string(i)});
                rtcproc.kernel.img_ext_names.push_back(img_ext_name);
            }
        }
    }

    /* despike */
    get_config_value(config, rtcproc.run_despike, missing_keys, invalid_keys, std::tuple{"timestream","raw_time_chunk","despike","enabled"});
    if (rtcproc.run_despike) {
        get_config_value(config, rtcproc.despiker.min_spike_sigma, missing_keys, invalid_keys, std::tuple{"timestream","raw_time_chunk",
                                                                                                   "despike","min_spike_sigma"});
        get_config_value(config, rtcproc.despiker.time_constant_sec, missing_keys, invalid_keys, std::tuple{"timestream","raw_time_chunk",
                                                                                                     "despike","time_constant_sec"});
        get_config_value(config, rtcproc.despiker.window_size, missing_keys, invalid_keys, std::tuple{"timestream","raw_time_chunk","despike",
                                                                                               "window_size"});

        rtcproc.despiker.grouping = "nw";
    }

    /* filter */
    get_config_value(config, rtcproc.run_tod_filter, missing_keys, invalid_keys, std::tuple{"timestream","raw_time_chunk","filter","enabled"});

    // set scan limits
    if (rtcproc.run_tod_filter) {
        get_config_value(config, rtcproc.filter.a_gibbs, missing_keys, invalid_keys, std::tuple{"timestream","raw_time_chunk","filter","a_gibbs"});
        get_config_value(config, rtcproc.filter.freq_low_Hz, missing_keys, invalid_keys, std::tuple{"timestream","raw_time_chunk","filter",
                                                                                             "freq_low_Hz"});
        get_config_value(config, rtcproc.filter.freq_high_Hz, missing_keys, invalid_keys, std::tuple{"timestream","raw_time_chunk","filter",
                                                                                              "freq_high_Hz"});
        get_config_value(config, rtcproc.filter.n_terms, missing_keys, invalid_keys, std::tuple{"timestream","raw_time_chunk","filter","n_terms"});

        telescope.inner_scans_chunk = rtcproc.filter.n_terms;
        rtcproc.despiker.window_size = rtcproc.filter.n_terms;
    }
    else {
        rtcproc.filter.n_terms = 0;
        telescope.inner_scans_chunk = 0;
    }

    /* downsample */
    get_config_value(config, rtcproc.run_downsample, missing_keys, invalid_keys, std::tuple{"timestream","raw_time_chunk","downsample","enabled"});
    if (rtcproc.run_downsample) {
        get_config_value(config, rtcproc.downsampler.factor, missing_keys, invalid_keys, std::tuple{"timestream","raw_time_chunk","downsample",
                                                                                                    "factor"},{},{0});
    }

    /* calibration */
    get_config_value(config, rtcproc.run_calibrate, missing_keys, invalid_keys, std::tuple{"timestream","raw_time_chunk","calibration",
                                                                                           "enabled"});
    get_config_value(config, rtcproc.calibration.extinction_model, missing_keys, invalid_keys, std::tuple{"timestream","raw_time_chunk",
                                                                                                          "calibration","extinction_model"},
                                                                                                         {"am_q25","am_q50","am_q75"});
    // setup atm model
    rtcproc.calibration.setup();
}

template<typename CT>
void Engine::get_ptc_config(CT &config) {
    get_config_value(config, ptcproc.weighting_type, missing_keys, invalid_keys, std::tuple{"timestream","processed_time_chunk","weighting",
                                                                                            "type"},{"full","approximate"});
    get_config_value(config, ptcproc.lower_std_dev, missing_keys, invalid_keys, std::tuple{"timestream","processed_time_chunk","flagging",
                                                                                           "lower_weight_factor"});
    get_config_value(config, ptcproc.upper_std_dev, missing_keys, invalid_keys, std::tuple{"timestream","processed_time_chunk","flagging",
                                                                                           "upper_weight_factor"});

    // for sensitivity fcf
    ptcproc.run_calibrate = rtcproc.run_calibrate;

    // override calibration if in beammap mode

    /*if (redu_type=="beammap") {
        rtcproc.run_calibrate = false;
        ptcproc.run_calibrate = false;
    }*/

    /* cleaning */
    get_config_value(config, ptcproc.run_clean, missing_keys, invalid_keys, std::tuple{"timestream","processed_time_chunk","clean","enabled"});

    // vector of groupings
    ptcproc.cleaner.grouping = config.template get_typed<std::vector<std::string>>(std::tuple{"timestream","processed_time_chunk","clean","grouping"});
    // vector of eigenvalues to cut
    auto n_eig_to_cut = config.template get_typed<std::vector<Eigen::Index>>(std::tuple{"timestream","processed_time_chunk","clean","n_eig_to_cut"});
    // map to eigen vector
    ptcproc.cleaner.n_eig_to_cut = Eigen::Map<Eigen::VectorXI>(n_eig_to_cut.data(),n_eig_to_cut.size());
    get_config_value(config, ptcproc.cleaner.stddev_limit, missing_keys, invalid_keys, std::tuple{"timestream","processed_time_chunk",
                                                                                             "clean","stddev_limit"});
}

template<typename CT>
void Engine::get_mapmaking_config(CT &config) {
    get_config_value(config, map_grouping, missing_keys, invalid_keys, std::tuple{"mapmaking","grouping"},{"auto","array","nw","detector"});

    // coverage cut
    get_config_value(config, omb.cov_cut, missing_keys, invalid_keys, std::tuple{"mapmaking","coverage_cut"});
    cmb.cov_cut = omb.cov_cut;

    // copy map grouping for simplicity
    omb.map_grouping = map_grouping;
    cmb.map_grouping = omb.map_grouping;

    rtcproc.kernel.map_grouping = omb.map_grouping;

    // map_method
    get_config_value(config, map_method, missing_keys, invalid_keys, std::tuple{"mapmaking","method"},{"naive","jinc"});

    if (map_method=="jinc") {
        get_config_value(config, jinc_r_max, missing_keys, invalid_keys, std::tuple{"mapmaking","jinc_filter","r_max"},{},{0});
        get_config_value(config, jinc_a, missing_keys, invalid_keys, std::tuple{"mapmaking","jinc_filter","a"});
        get_config_value(config, jinc_b, missing_keys, invalid_keys, std::tuple{"mapmaking","jinc_filter","b"});
        get_config_value(config, jinc_c, missing_keys, invalid_keys, std::tuple{"mapmaking","jinc_filter","c"});
    }

    // histogram
    get_config_value(config, omb.hist_n_bins, missing_keys, invalid_keys, std::tuple{"post_processing","map_histogram_n_bins"},{},{0});
    cmb.hist_n_bins = omb.hist_n_bins;

    /* wcs */

    // pixel size
    get_config_value(config, omb.pixel_size_rad, missing_keys, invalid_keys, std::tuple{"mapmaking","pixel_size_arcsec"},{},{0});

    // convert to radians
    omb.pixel_size_rad *= ASEC_TO_RAD;
    cmb.pixel_size_rad = omb.pixel_size_rad;

    omb.wcs.cdelt.push_back(-omb.pixel_size_rad);
    omb.wcs.cdelt.push_back(omb.pixel_size_rad);

    omb.wcs.cdelt.push_back(1);
    omb.wcs.cdelt.push_back(1);

    // icrs or altaz
    get_config_value(config, telescope.pixel_axes, missing_keys, invalid_keys, std::tuple{"mapmaking","pixel_axes"},{"icrs","altaz"});

    double wcs_double;

    // crpix
    get_config_value(config, wcs_double, missing_keys, invalid_keys, std::tuple{"mapmaking","crpix1"});
    omb.wcs.crpix.push_back(wcs_double);

    get_config_value(config, wcs_double, missing_keys, invalid_keys, std::tuple{"mapmaking","crpix2"});
    omb.wcs.crpix.push_back(wcs_double);

    omb.wcs.crpix.push_back(1);
    omb.wcs.crpix.push_back(1);

    // crval
    get_config_value(config, wcs_double, missing_keys, invalid_keys, std::tuple{"mapmaking","crval1_J2000"});
    omb.wcs.crval.push_back(wcs_double);

    get_config_value(config, wcs_double, missing_keys, invalid_keys, std::tuple{"mapmaking","crval2_J2000"});
    omb.wcs.crval.push_back(wcs_double);

    omb.wcs.crval.push_back(1);
    omb.wcs.crval.push_back(1);

    // naxis
    get_config_value(config, wcs_double, missing_keys, invalid_keys, std::tuple{"mapmaking","x_size_pix"});
    omb.wcs.naxis.push_back(wcs_double);

    get_config_value(config, wcs_double, missing_keys, invalid_keys, std::tuple{"mapmaking","y_size_pix"});
    omb.wcs.naxis.push_back(wcs_double);

    omb.wcs.naxis.push_back(1);
    omb.wcs.naxis.push_back(1);

    // map units
    if (rtcproc.run_calibrate) {
        get_config_value(config, omb.sig_unit, missing_keys, invalid_keys, std::tuple{"mapmaking","cunit"},{"mJy/beam","MJy/sr"});
        cmb.sig_unit = omb.sig_unit;
    }

    else {
        omb.sig_unit = tod_type;
        cmb.sig_unit = tod_type;
    }

    // icrs frame
    if (telescope.pixel_axes == "icrs") {
        omb.wcs.ctype.push_back("RA---TAN");
        omb.wcs.ctype.push_back("DEC--TAN");
        omb.wcs.ctype.push_back("FREQ");
        omb.wcs.ctype.push_back("STOKES");

        omb.wcs.cunit.push_back("deg");
        omb.wcs.cunit.push_back("deg");
        omb.wcs.cunit.push_back("Hz");
        omb.wcs.cunit.push_back("");

        omb.wcs.cdelt[0] *= RAD_TO_DEG;
        omb.wcs.cdelt[1] *= RAD_TO_DEG;
    }

    // altaz frame
    else if (telescope.pixel_axes == "altaz") {
        omb.wcs.ctype.push_back("AZOFFSET");
        omb.wcs.ctype.push_back("ELOFFSET");
        omb.wcs.ctype.push_back("FREQ");
        omb.wcs.ctype.push_back("STOKES");

        // arcsec if pointing or beammap
        if (redu_type == "pointing" || redu_type == "beammap") {
            omb.wcs.cunit.push_back("arcsec");
            omb.wcs.cunit.push_back("arcsec");
            omb.wcs.cdelt[0] *= RAD_TO_ASEC;
            omb.wcs.cdelt[1] *= RAD_TO_ASEC;
        }
        // degrees if science
        else {
            omb.wcs.cunit.push_back("deg");
            omb.wcs.cunit.push_back("deg");
            omb.wcs.cdelt[0] *= RAD_TO_DEG;
            omb.wcs.cdelt[1] *= RAD_TO_DEG;
        }
        omb.wcs.cunit.push_back("Hz");
        omb.wcs.cunit.push_back("");
    }

    // copy omb wcs to cmb wcs
    cmb.wcs = omb.wcs;

    // set parallelization for psd filter ffts (maintained with tod output/verbose mode)
    omb.parallel_policy = parallel_policy;
    cmb.parallel_policy = parallel_policy;
}


template<typename CT>
void Engine::get_beammap_config(CT &config) {
    get_config_value(config, beammap_iter_max, missing_keys, invalid_keys, std::tuple{"beammap","iter_max"});
    get_config_value(config, beammap_iter_tolerance, missing_keys, invalid_keys, std::tuple{"beammap","iter_tolerance"});
    get_config_value(config, beammap_reference_det, missing_keys, invalid_keys, std::tuple{"beammap","reference_det"});
    get_config_value(config, beammap_subtract_reference, missing_keys, invalid_keys, std::tuple{"beammap","subtract_reference_det"});
    get_config_value(config, beammap_derotate, missing_keys, invalid_keys, std::tuple{"beammap","derotate"});

    // limits for flagging
    for (auto const& [arr_index, arr_name] : toltec_io.array_name_map) {
        get_config_value(config, lower_fwhm_arcsec[arr_name], missing_keys, invalid_keys, std::tuple{"beammap","flagging","lower_fwhm_arcsec",
                                                                                              arr_name});
        get_config_value(config, upper_fwhm_arcsec[arr_name], missing_keys, invalid_keys, std::tuple{"beammap","flagging","upper_fwhm_arcsec",
                                                                                              arr_name});
        get_config_value(config, lower_sig2noise[arr_name], missing_keys, invalid_keys, std::tuple{"beammap","flagging","lower_sig2noise",
                                                                                            arr_name});
        get_config_value(config, max_dist_arcsec[arr_name], missing_keys, invalid_keys, std::tuple{"beammap","flagging","max_dist_arcsec",
                                                                                            arr_name});
    }

    get_config_value(config, lower_sens_factor, missing_keys, invalid_keys, std::tuple{"beammap","flagging","lower_sens_factor"});
    get_config_value(config, upper_sens_factor, missing_keys, invalid_keys, std::tuple{"beammap","flagging","upper_sens_factor"});

    // sensitiivty
    sens_psd_limits.resize(2);
    double sens;
    get_config_value(config, sens, missing_keys, invalid_keys, std::tuple{"beammap","sens_psd_lower_limit"});
    sens_psd_limits(0) = sens;

    get_config_value(config, sens, missing_keys, invalid_keys, std::tuple{"beammap","sens_psd_upper_limit"});
    sens_psd_limits(1) = sens;

    // if no tolerance is specified, write out max iteration tod
    if (run_tod_output) {
        if (beammap_iter_tolerance <=0) {
            beammap_tod_output_iter = beammap_iter_max;
        }
        else {
            beammap_tod_output_iter = 0;
        }
    }
}

template<typename CT>
void Engine::get_map_filter_config(CT &config) {
    get_config_value(config, wiener_filter.template_type, missing_keys, invalid_keys, std::tuple{"wiener_filter","template_type"},
                     {"kernel","gaussian","airy","highpass"});
    get_config_value(config, wiener_filter.run_lowpass, missing_keys, invalid_keys, std::tuple{"wiener_filter","lowpass_only"});
    get_config_value(config, wiener_filter.normalize_error, missing_keys, invalid_keys, std::tuple{"post_processing","map_filtering",
                                                                                                   "normalize_errors"});
    if (wiener_filter.template_type=="kernel") {
        if (!rtcproc.run_kernel) {
            SPDLOG_ERROR("wiener filter kernel template requires kernel");
            std::exit(EXIT_FAILURE);
        }
        else {
            wiener_filter.map_fitter = map_fitter;
        }
    }

    if (!run_noise) {
        SPDLOG_ERROR("wiener filter requires noise maps");
        std::exit(EXIT_FAILURE);
    }

    // gaussian template fwhms
    if (wiener_filter.template_type=="gaussian" || wiener_filter.template_type=="airy") {
        get_config_value(config, wiener_filter.gaussian_template_fwhm_rad["a1100"], missing_keys, invalid_keys,
                   std::tuple{"wiener_filter","gaussian_template_fwhm_arcsec","a1100"});
        get_config_value(config, wiener_filter.gaussian_template_fwhm_rad["a1400"], missing_keys, invalid_keys,
                   std::tuple{"wiener_filter","gaussian_template_fwhm_arcsec","a1400"});
        get_config_value(config, wiener_filter.gaussian_template_fwhm_rad["a2000"], missing_keys, invalid_keys,
                   std::tuple{"wiener_filter","gaussian_template_fwhm_arcsec","a2000"});

        for (auto const& pair : wiener_filter.gaussian_template_fwhm_rad) {
            wiener_filter.gaussian_template_fwhm_rad[pair.first] = wiener_filter.gaussian_template_fwhm_rad[pair.first]*ASEC_TO_RAD;
        }
    }

    // set parallelization for ffts (maintained with tod output/verbose mode)
    wiener_filter.parallel_policy = parallel_policy;
}

template<typename CT>
void Engine::get_citlali_config(CT &config) {
    /* interface offsets */
    if (config.has(std::tuple{"interface_sync_offset"})) {

        auto interface_node = config.get_node(std::tuple{"interface_sync_offset"});

        std::vector<std::string> interface_keys = {
            "toltec0",
            "toltec1",
            "toltec2",
            "toltec3",
            "toltec4",
            "toltec5",
            "toltec6",
            "toltec7",
            "toltec8",
            "toltec9",
            "toltec10",
            "toltec11",
            "toltec12",
            "hwp"
        };

        for (Eigen::Index i=0; i<interface_node.size(); i++) {
            try {
                auto offset = config.template get_typed<double>(std::tuple{"interface_sync_offset",i, interface_keys[i]});
                interface_sync_offset[interface_keys[i]] = offset;
            } catch (YAML::TypedBadConversion<double>) {
               std::vector<std::string> invalid_temp;
                engine_utils::for_each_in_tuple(
                    std::tuple{"interface_sync_offset", std::to_string(i), interface_keys[i]},
                    [&](const auto &x) { invalid_temp.push_back(x); });
                invalid_keys.push_back(invalid_temp);
            }

            catch (YAML::InvalidNode) {
                std::vector<std::string> invalid_temp;
                engine_utils::for_each_in_tuple(
                    std::tuple{"interface_sync_offset", std::to_string(i), interface_keys[i]},
                    [&](const auto &x) { invalid_temp.push_back(x); });
                invalid_keys.push_back(invalid_temp);
            }
        }
    }

    else {
        std::vector<std::string> missing_temp;
        engine_utils::for_each_in_tuple(
            std::tuple{"interface_sync_offset"},
            [&](const auto &x) { missing_temp.push_back(x); });
        missing_keys.push_back(missing_temp);
    }

    /* runtime */
    get_config_value(config, verbose_mode, missing_keys, invalid_keys, std::tuple{"runtime","verbose"});
    get_config_value(config, output_dir, missing_keys, invalid_keys, std::tuple{"runtime","output_dir"});
    get_config_value(config, n_threads, missing_keys, invalid_keys, std::tuple{"runtime","n_threads"});
    get_config_value(config, parallel_policy, missing_keys, invalid_keys, std::tuple{"runtime","parallel_policy"},{"seq","omp"});

    get_config_value(config, redu_type, missing_keys, invalid_keys, std::tuple{"runtime","reduction_type"},{"science","pointing","beammap"});
    get_config_value(config, use_subdir, missing_keys, invalid_keys, std::tuple{"runtime","use_subdir"});

    /* timestream */
    get_config_value(config, run_tod, missing_keys, invalid_keys, std::tuple{"timestream","enabled"});
    get_config_value(config, tod_type, missing_keys, invalid_keys, std::tuple{"timestream","type"});

    bool run_tod_output_rtc, run_tod_output_ptc;
    get_config_value(config, run_tod_output_rtc, missing_keys, invalid_keys, std::tuple{"timestream","raw_time_chunk","output","enabled"});
    get_config_value(config, run_tod_output_ptc, missing_keys, invalid_keys, std::tuple{"timestream","processed_time_chunk","output",
                                                                                        "enabled"});
    // set tod output to false by default
    run_tod_output = false;

    // check if rtc output is requested
    if (run_tod_output_rtc) {
        run_tod_output = true;
        tod_output_type = "rtc";
    }
    // if ptc output is requested
    if (run_tod_output_ptc) {
        // check if rtc output was requested
        if (run_tod_output == true) {
            tod_output_type = "both";
        }
        // else just output ptc
        else {
            run_tod_output = true;
            tod_output_type = "ptc";
        }
    }

    if (run_tod_output) {
        get_config_value(config, tod_output_subdir_name, missing_keys, invalid_keys, std::tuple{"timestream","output", "subdir_name"});
    }
    get_config_value(config, telescope.time_chunk, missing_keys, invalid_keys, std::tuple{"timestream","chunking", "length_sec"});
    get_config_value(config, telescope.force_chunk, missing_keys, invalid_keys, std::tuple{"timestream","chunking", "force_chunking"});

    /* rtc */
    get_rtc_config(config);

    /* ptc */
    get_ptc_config(config);

    /* mapmaking */
    get_config_value(config, run_mapmaking, missing_keys, invalid_keys, std::tuple{"mapmaking","enabled"});
    get_mapmaking_config(config);

    /* map filtering */
    get_config_value(config, run_map_filter, missing_keys, invalid_keys, std::tuple{"post_processing","map_filtering","enabled"});

    /* fitting */
    if (redu_type=="pointing" || redu_type=="beammap" || run_map_filter) {
        get_config_value(config, map_fitter.bounding_box_pix, missing_keys, invalid_keys, std::tuple{"post_processing","source_fitting",
                                                                                                     "bounding_box_arcsec"},{},{0});
        get_config_value(config, map_fitter.fitting_region_pix, missing_keys, invalid_keys, std::tuple{"post_processing","source_fitting",
                                                                                                       "fitting_region_arcsec"});
        map_fitter.bounding_box_pix = ASEC_TO_RAD*map_fitter.bounding_box_pix/omb.pixel_size_rad;
        map_fitter.fitting_region_pix = ASEC_TO_RAD*map_fitter.fitting_region_pix/omb.pixel_size_rad;

        // temp setup to get limits
        map_fitter.flux_limits.resize(2);
        map_fitter.fwhm_limits.resize(2);

        map_fitter.flux_limits(0) = config.template get_typed<double>(std::tuple{"post_processing","source_fitting","gauss_model",
                                                                                 "amp_limits",0});
        map_fitter.flux_limits(1) = config.template get_typed<double>(std::tuple{"post_processing","source_fitting","gauss_model",
                                                                                 "amp_limits",1});

        map_fitter.fwhm_limits(0) = config.template get_typed<double>(std::tuple{"post_processing","source_fitting","gauss_model",
                                                                                 "fwhm_limits",0});
        map_fitter.fwhm_limits(1) = config.template get_typed<double>(std::tuple{"post_processing","source_fitting","gauss_model",
                                                                                 "fwhm_limits",1});

        // flux lower factor
        if (map_fitter.flux_limits(0) > 0) {
            map_fitter.flux_low = map_fitter.flux_limits(0);
        }

        // flux lower factor
        if (map_fitter.flux_limits(1) > 0) {
            map_fitter.flux_high = map_fitter.flux_limits(1);
        }

        // fwhm lower factor
        if (map_fitter.fwhm_limits(0) > 0) {
            map_fitter.fwhm_low = map_fitter.fwhm_limits(0);
        }

        // fwhm upper factor
        if (map_fitter.fwhm_limits(1) > 0) {
            map_fitter.fwhm_high = map_fitter.fwhm_limits(1);
        }
    }

    /* coaddition */
    get_config_value(config, run_coadd, missing_keys, invalid_keys, std::tuple{"coadd","enabled"});

    /* noise maps */
    get_config_value(config, run_noise, missing_keys, invalid_keys, std::tuple{"noise_maps","enabled"});
    if (run_noise) {
        get_config_value(config, omb.n_noise, missing_keys, invalid_keys, std::tuple{"noise_maps","n_noise_maps"},{},{0},{});
        cmb.n_noise = omb.n_noise;
    }
    else {
        omb.n_noise = 0;
        cmb.n_noise = 0;
    }

    /* wiener filter */
    if (run_map_filter) {
        get_map_filter_config(config);
    }

    /* beammap */
    if (redu_type=="beammap") {
        get_beammap_config(config);
    }

    /* source finding */
    get_config_value(config, run_source_finder, missing_keys, invalid_keys, std::tuple{"post_processing","source_finding","enabled"});

    if (run_source_finder) {
        get_config_value(config, omb.source_sigma, missing_keys, invalid_keys, std::tuple{"post_processing","source_finding","source_sigma"});
        get_config_value(config, omb.source_window_rad, missing_keys, invalid_keys, std::tuple{"post_processing","source_finding","source_window_arcsec"});
        get_config_value(config, omb.source_finder_mode, missing_keys, invalid_keys, std::tuple{"post_processing","source_finding","mode"});

        omb.source_window_rad = omb.source_window_rad*ASEC_TO_RAD;

        cmb.source_sigma = omb.source_sigma;
        cmb.source_window_rad = omb.source_window_rad;
        cmb.source_finder_mode = omb.source_finder_mode;
    }

    // disable map related keys if map-making is disabled
    if (!run_mapmaking) {
        run_coadd = false;
        run_noise = false;
        run_map_filter = false;
        run_source_finder = false;
        // we don't need to do iterations if no maps are made
        beammap_iter_max = 1;
    }
}

template<typename CT>
void Engine::get_photometry_config(CT &config) {
    // beammap name
    get_config_value(config, beammap_source_name, missing_keys, invalid_keys, std::tuple{"beammap_source","name"});

    // beammap source ra
    get_config_value(config, beammap_ra_rad, missing_keys, invalid_keys, std::tuple{"beammap_source","ra_deg"});
    beammap_ra_rad = beammap_ra_rad*DEG_TO_RAD;

    // beammap source dec
    get_config_value(config, beammap_dec_rad, missing_keys, invalid_keys, std::tuple{"beammap_source","dec_deg"});
    beammap_dec_rad = beammap_dec_rad*DEG_TO_RAD;

    Eigen::Index n_fluxes = config.get_node(std::tuple{"beammap_source","fluxes"}).size();

    // get source fluxes
    for (Eigen::Index i=0; i<n_fluxes; i++) {
        auto array = config.get_str(std::tuple{"beammap_source","fluxes",i,"array_name"});

        auto flux = config.template get_typed<double>(std::tuple{"beammap_source","fluxes",i,"value_mJy"});
        auto uncertainty_mJy = config.template get_typed<double>(std::tuple{"beammap_source","fluxes",i,"uncertainty_mJy"});

        beammap_fluxes_mJy_beam[array] = flux;
        beammap_err_mJy_beam[array] = uncertainty_mJy;
    }
}

template<typename CT>
void Engine::get_astrometry_config(CT &config) {
    pointing_offsets_arcsec["az"] = config.template get_typed<double>(std::tuple{"pointing_offsets",0,"value_arcsec"});
    pointing_offsets_arcsec["alt"] = config.template get_typed<double>(std::tuple{"pointing_offsets",1,"value_arcsec"});
}

template <typename Derived>
auto Engine::calc_map_indices(Eigen::DenseBase<Derived> &det_indices, Eigen::DenseBase<Derived> &nw_indices,
                              Eigen::DenseBase<Derived> &array_indices, std::string stokes_param) {
    // indices for maps
    Eigen::VectorXI indices(array_indices.size()), map_indices(array_indices.size());

    // overwrite map indices for networks
    if (map_grouping == "nw") {
        indices = nw_indices;
    }

    // overwrite map indices for arrays
    else if (map_grouping == "array") {
        indices = array_indices;
    }

    // overwrite map indices for detectors
    else if (map_grouping == "detector") {
        indices = det_indices;
    }

    // start at 0
    Eigen::Index map_index = 0;
    map_indices(0) = 0;
    // loop through and populate map indices
    for (Eigen::Index i=0; i<indices.size()-1; i++) {
        // if next index is larger than current index, increment map index
        if (indices(i+1) > indices(i)) {
            map_index++;
        }
        map_indices(i+1) = map_index;
    }

    // loop through and calculate map indices
    /*if (map_grouping != "detector") {
        for (Eigen::Index i=0; i<map_indices.size()-1; i++) {
            if (map_indices(i+1) > (map_indices(i)+1)) {
                auto map_indices_temp = map_indices;
                auto mi_lower = map_indices(i);
                auto mi_upper = map_indices(i+1);
                (map_indices.array() == mi_upper).select(mi_lower+1,map_indices.array());
            }
        }
    }*/

    // make sure first value is zero
    //auto min_index = map_indices.minCoeff();
    //map_indices = map_indices.array() - min_index;

    if (rtcproc.run_polarization) {
        if (stokes_param == "Q") {
            map_indices = map_indices.array() + n_maps/3;
        }
        else if (stokes_param == "U") {
            map_indices = map_indices.array() + 2*n_maps/3;
        }
    }

    return std::move(map_indices);
}

void Engine::create_map_files() {
    // clear for each observation
    fits_io_vec.clear();

    if (!run_coadd) {
        if (run_noise) {
            noise_fits_io_vec.clear();
        }
    }

    for (Eigen::Index i=0; i<calib.n_arrays; i++) {
        auto array = calib.arrays[i];
        std::string array_name = toltec_io.array_name_map[array];
        auto filename = toltec_io.create_filename<engine_utils::toltecIO::toltec,
                                                  engine_utils::toltecIO::map,
                                                  engine_utils::toltecIO::raw>(obsnum_dir_name + "raw/", redu_type, array_name,
                                                                               obsnum, telescope.sim_obs);
        fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*> fits_io(filename);

        fits_io_vec.push_back(std::move(fits_io));

        // if noise maps are requested but coadding is not, populate noise fits vector
        if (!run_coadd) {
            if (run_noise) {
                auto filename = toltec_io.create_filename<engine_utils::toltecIO::toltec,
                                                          engine_utils::toltecIO::noise,
                                                          engine_utils::toltecIO::raw>(obsnum_dir_name + "raw/", redu_type, array_name,
                                                                                         obsnum, telescope.sim_obs);
                fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*> fits_io(filename);

                noise_fits_io_vec.push_back(std::move(fits_io));
            }

            // map filtering
            if (run_map_filter) {
                auto filename = toltec_io.create_filename<engine_utils::toltecIO::toltec,
                                                          engine_utils::toltecIO::map,
                                                          engine_utils::toltecIO::filtered>(obsnum_dir_name + "filtered/",
                                                                                            redu_type, array_name,
                                                                                            obsnum, telescope.sim_obs);
                fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*> fits_io(filename);

                filtered_fits_io_vec.push_back(std::move(fits_io));

                // filtered noise maps
                if (run_noise) {
                    auto filename = toltec_io.create_filename<engine_utils::toltecIO::toltec,
                                                              engine_utils::toltecIO::noise,
                                                              engine_utils::toltecIO::filtered>(obsnum_dir_name + "filtered/", redu_type,
                                                                                             array_name, obsnum, telescope.sim_obs);
                    fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*> fits_io(filename);

                    filtered_noise_fits_io_vec.push_back(std::move(fits_io));
                }
            }
        }
    }
}

template <engine_utils::toltecIO::ProdType prod_t>
void Engine::create_tod_files() {
    // loop through stokes indices
    for (const auto &[stokes_index,stokes_param]: rtcproc.polarization.stokes_params) {
        // name for std map
        std::string name;
        // subdirectory name
        std::string dir_name = obsnum_dir_name + "raw/";

        // if config subdirectory name is specified, add it
        if (tod_output_subdir_name != "null") {
            dir_name = dir_name + tod_output_subdir_name + "/";
        }

        // rtc tod output filename setup
        if constexpr (prod_t == engine_utils::toltecIO::rtc_timestream) {
            auto filename = toltec_io.create_filename<engine_utils::toltecIO::toltec,
                                                      engine_utils::toltecIO::rtc_timestream,
                                                      engine_utils::toltecIO::raw>(dir_name, redu_type, "",
                                                                                   obsnum, telescope.sim_obs);

            tod_filename["rtc_" + stokes_param] = filename + "_" + stokes_param + ".nc";
            name = "rtc_" + stokes_param;
        }

        // ptc tod output filename setup
        else if constexpr (prod_t == engine_utils::toltecIO::ptc_timestream) {
            auto filename = toltec_io.create_filename<engine_utils::toltecIO::toltec,
                                                      engine_utils::toltecIO::ptc_timestream,
                                                      engine_utils::toltecIO::raw>(dir_name, redu_type, "",
                                                                                   obsnum, telescope.sim_obs);

            tod_filename["ptc_" + stokes_param] = filename + "_" + stokes_param + ".nc";
            name = "ptc_" + stokes_param;
        }

        // create netcdf file
        netCDF::NcFile fo(tod_filename[name], netCDF::NcFile::replace);

        // add tod output type to file
        netCDF::NcDim n_tod_output_type_dim = fo.addDim("n_tod_output_type",1);
        netCDF::NcVar tod_output_type_var = fo.addVar("tod_output_type",netCDF::ncString, n_tod_output_type_dim);
        const std::vector< size_t > tod_output_type_index = {0};

        if constexpr (prod_t == engine_utils::toltecIO::rtc_timestream) {
            std::string tod_output_type_name = "rtc";
            tod_output_type_var.putVar(tod_output_type_index,tod_output_type_name);
        }
        else if constexpr (prod_t == engine_utils::toltecIO::ptc_timestream) {
            std::string tod_output_type_name = "ptc";
            tod_output_type_var.putVar(tod_output_type_index,tod_output_type_name);
        }

        // add obsnum
        netCDF::NcVar obsnum_v = fo.addVar("obsnum",netCDF::ncInt);
        int obsnum_int = std::stoi(obsnum);
        obsnum_v.putVar(&obsnum_int);

        // add source ra
        netCDF::NcVar source_ra_v = fo.addVar("SourceRa",netCDF::ncDouble);
        source_ra_v.putVar(&telescope.tel_header["Header.Source.Ra"](0));

        // add source dec
        netCDF::NcVar source_dec_v = fo.addVar("SourceDec",netCDF::ncDouble);
        source_dec_v.putVar(&telescope.tel_header["Header.Source.Dec"](0));

        netCDF::NcDim n_pts_dim = fo.addDim("n_pts");
        netCDF::NcDim n_raw_scan_indices_dim = fo.addDim("n_raw_scan_indices", telescope.scan_indices.rows());
        netCDF::NcDim n_scan_indices_dim = fo.addDim("n_scan_indices", 2);
        netCDF::NcDim n_scans_dim = fo.addDim("n_scans", telescope.scan_indices.cols());

        Eigen::Index n_dets;

        // set number of dets for unpolarized timestreams
        if (stokes_param=="I") {
            n_dets = calib.apt["array"].size();
        }

        // set number of detectors for polarized timestreams
        else if ((stokes_param == "Q") || (stokes_param == "U")) {
            //n_dets = (calib.apt["fg"].array() == 0).count() + (calib.apt["fg"].array() == 1).count();
            n_dets = (calib.apt["loc"].array()!=-1).count()/2;
        }

        netCDF::NcDim n_dets_dim = fo.addDim("n_dets", n_dets);

        std::vector<netCDF::NcDim> dims = {n_pts_dim, n_dets_dim};
        std::vector<netCDF::NcDim> raw_scans_dims = {n_scans_dim, n_raw_scan_indices_dim};
        std::vector<netCDF::NcDim> scans_dims = {n_scans_dim, n_scan_indices_dim};

        // raw file scan indices
        netCDF::NcVar raw_scan_indices_v = fo.addVar("raw_scan_indices",netCDF::ncInt, raw_scans_dims);
        //Eigen::MatrixXI scans_indices_transposed = telescope.scan_indices;
        raw_scan_indices_v.putVar(telescope.scan_indices.data());

        // scan indices for data
        netCDF::NcVar scan_indices_v = fo.addVar("scan_indices",netCDF::ncInt, scans_dims);

        // signal
        netCDF::NcVar signal_v = fo.addVar("signal",netCDF::ncDouble, dims);
        // chunk sizes
        std::vector<std::size_t> chunkSizes;
        // set chunk mode
        netCDF::NcVar::ChunkMode chunkMode = netCDF::NcVar::nc_CHUNKED;

        // set chunking to mean scan size and n_dets
        chunkSizes.push_back(((telescope.scan_indices.row(3) - telescope.scan_indices.row(2)).array() + 1).mean());
        chunkSizes.push_back(n_dets);

        // set signal chunking
        signal_v.setChunking(chunkMode, chunkSizes);

        // flags
        netCDF::NcVar flags_v = fo.addVar("flags",netCDF::ncDouble, dims);
        flags_v.setChunking(chunkMode, chunkSizes);
        // kernel
        if (rtcproc.run_kernel) {
            netCDF::NcVar kernel_v = fo.addVar("kernel",netCDF::ncDouble, dims);
            kernel_v.setChunking(chunkMode, chunkSizes);
        }

        // detector lat
        netCDF::NcVar det_lat_v = fo.addVar("det_lat",netCDF::ncDouble, dims);
        det_lat_v.setChunking(chunkMode, chunkSizes);

        // detector lon
        netCDF::NcVar det_lon_v = fo.addVar("det_lon",netCDF::ncDouble, dims);
        det_lon_v.setChunking(chunkMode, chunkSizes);

        // calc absolute pointing if in icrs frame
        if (telescope.pixel_axes == "icrs") {
            // detector absolute ra
            netCDF::NcVar det_ra_v = fo.addVar("det_ra",netCDF::ncDouble, dims);
            det_ra_v.setChunking(chunkMode, chunkSizes);

            // detector absolute dec
            netCDF::NcVar det_dec_v = fo.addVar("det_dec",netCDF::ncDouble, dims);
            det_dec_v.setChunking(chunkMode, chunkSizes);
        }

        // add apt table
        for (auto const& x: calib.apt) {
            netCDF::NcVar apt_v = fo.addVar("apt_" + x.first,netCDF::ncDouble, n_dets_dim);
        }

        // add telescope parameters
        for (auto const& x: telescope.tel_data) {
            netCDF::NcVar tel_data_v = fo.addVar(x.first,netCDF::ncDouble, n_pts_dim);
        }

        // weights
        if constexpr (prod_t == engine_utils::toltecIO::ptc_timestream) {
            std::vector<netCDF::NcDim> weight_dims = {n_scans_dim, n_dets_dim};
            netCDF::NcVar weights_v = fo.addVar("weights",netCDF::ncDouble, weight_dims);
        }

        fo.close();
    }
}

//template <TCDataKind tc_t>
void Engine::print_summary() {
    SPDLOG_INFO("\n\nreduction info:\n\n");
    SPDLOG_INFO("map buffer rows: {}", omb.n_rows);
    SPDLOG_INFO("map buffer cols: {}", omb.n_cols);
    SPDLOG_INFO("number of maps: {}", omb.signal.size());
    SPDLOG_INFO("map units: {}",omb.sig_unit);

    // total size of all maps
    double mb_size_total = 0;

    // make a rough estimate of memory usage for obs map buffer
    double omb_size = 8*omb.n_rows*omb.n_cols*(omb.signal.size() + omb.weight.size() +
                                               omb.kernel.size() + omb.coverage.size())/1e9;

    SPDLOG_INFO("estimated size of map buffer {} GB", omb_size);

    mb_size_total = mb_size_total + omb_size;

    // print info if coadd is requested
    if (run_coadd) {
        SPDLOG_INFO("coadd map buffer rows: {}", cmb.n_rows);
        SPDLOG_INFO("coadd map buffer cols: {}", cmb.n_cols);

        // make a rough estimate of memory usage for coadd map buffer
        double cmb_size = 8*cmb.n_rows*cmb.n_cols*(cmb.signal.size() + cmb.weight.size() +
                                                   cmb.kernel.size() + cmb.coverage.size())/1e9;

        SPDLOG_INFO("estimated size of coadd buffer {} GB", cmb_size);

        mb_size_total = mb_size_total + cmb_size;

        // output info if coadd noise maps are requested
        if (run_noise) {
            SPDLOG_INFO("coadd map buffer noise maps: {}", cmb.n_noise);
            // make a rough estimate of memory usage for coadd noise maps
            double nmb_size = 8*cmb.n_rows*cmb.n_cols*cmb.noise.size()*cmb.n_noise/1e9;
            SPDLOG_INFO("estimated size of noise buffer {} GB", nmb_size);
            mb_size_total = mb_size_total + nmb_size;
        }
    }
    else {
        // output info if obs noise maps are requested
        if (run_noise) {
            SPDLOG_INFO("observation map buffer noise maps: {}", omb.n_noise);
            // make a rough estimate of memory usage for obs noise maps
            double nmb_size = 8*omb.n_rows*omb.n_cols*omb.noise.size()*omb.n_noise/1e9;
            SPDLOG_INFO("estimated size of noise buffer {} GB", nmb_size);
            mb_size_total = mb_size_total + nmb_size;
        }
    }

    SPDLOG_INFO("estimated size of all maps {} GB", mb_size_total);

    SPDLOG_INFO("number of scans: {}\n\n",telescope.scan_indices.cols());

}

template <TCDataKind tc_t>
void Engine::write_chunk_summary(TCData<tc_t, Eigen::MatrixXd> &in) {

    SPDLOG_DEBUG("writing summary files for chunk {}",in.index.data);

    std::string filename = "chunk_summary_" + std::to_string(in.index.data);

    // write summary log file
    std::ofstream f;
    f.open (obsnum_dir_name+"/logs/" + filename + ".log");

    f << "Summary file for scan " << in.index.data << "\n";
    f << "-Citlali version: " << CITLALI_GIT_VERSION << "\n";
    f << "-Kidscpp version: " << KIDSCPP_GIT_VERSION << "\n";
    f << "-Time of time chunk creation: " + in.creation_time + "\n";
    f << "-Time of file writing: " << engine_utils::current_date_time() << "\n";

    f << "-Reduction type: " << redu_type << "\n";
    f << "-TOD type: " << tod_type << "\n";
    f << "-TOD unit: " << omb.sig_unit << "\n";

    f << "-Demodulated: " << in.demodulated << "\n";
    f << "-Kernel Generated: " << in.kernel_generated << "\n";
    f << "-Despiked: " << in.despiked << "\n";
    f << "-TOD filtered: " << in.tod_filtered << "\n";
    f << "-Downsampled: " << in.downsampled << "\n";
    f << "-Calibrated: " << in.calibrated << "\n";
    f << "-Cleaned: " << in.cleaned << "\n";

    f << "-Scan length: " << in.scans.data.rows() << "\n";

    f << "-Number of detectors: " << in.scans.data.cols() << "\n";
    f << "-Number of detectors flagged in APT table: " << (calib.apt["flag"].array()==0).count() << "\n";
    f << "-Number of detectors flagged below weight limit: " << in.n_low_dets <<"\n";
    f << "-Number of detectors flagged above weight limit: " << in.n_high_dets << "\n";
    Eigen::Index n_flagged = in.n_low_dets + in.n_high_dets + (calib.apt["flag"].array()==0).count();
    f << "-Number of detectors flagged: " << n_flagged << " (" << 100*float(n_flagged)/float(in.scans.data.cols()) << "%)\n";

    f << "-NaNs found: " << in.scans.data.array().isNaN().count() << "\n";
    f << "-Infs found: " << in.scans.data.array().isInf().count() << "\n";
    f << "-Data min: " << in.scans.data.minCoeff() << "\n";
    f << "-Data max: " << in.scans.data.maxCoeff() << "\n";
    f << "-Data mean: " << in.scans.data.mean() << "\n";

    f.close();

    using netCDF::NcDim;
    using netCDF::NcFile;
    using netCDF::NcType;
    using netCDF::NcVar;
    using namespace netCDF::exceptions;

    try {
        netCDF::NcFile fo(obsnum_dir_name+"/logs/" + filename + ".nc", netCDF::NcFile::replace);

        // number of samples
        unsigned long n_pts = in.scans.data.rows();

        // make sure its even
        if (n_pts % 2 == 1) {
            n_pts--;
        }

        // containers for frequency domain
        Eigen::Index n_freqs = n_pts / 2 + 1; // number of one sided freq bins
        double d_freq = telescope.d_fsmp / n_pts;

        netCDF::NcDim n_dets_dim = fo.addDim("n_dets", in.scans.data.cols());
        netCDF::NcDim n_hist_dim = fo.addDim("n_hist_bins", omb.hist_n_bins);

        // get number of bins
        auto n_hist_bins = n_hist_dim.getSize();

        std::vector<netCDF::NcDim> hist_dims = {n_dets_dim, n_hist_dim};

        // histogram variable
        netCDF::NcVar hist_var = fo.addVar("hist",netCDF::ncDouble, hist_dims);
        // histogram bins variable
        NcVar hist_bin_var = fo.addVar("hist_bins",netCDF::ncDouble, hist_dims);

        // add psd variable
        NcDim n_psd_dim = fo.addDim("n_psd", n_freqs);
        std::vector<netCDF::NcDim> psd_dims = {n_dets_dim, n_psd_dim};

        // psd variable
        netCDF::NcVar psd_var = fo.addVar("psd",netCDF::ncDouble, psd_dims);
        psd_var.putAtt("Units","V * s^(1/2)");
        // psd freq variable
        NcVar psd_freq_var = fo.addVar("psd_freq",netCDF::ncDouble, n_psd_dim);
        psd_freq_var.putAtt("Units","Hz");

        Eigen::VectorXd psd_freq = d_freq * Eigen::VectorXd::LinSpaced(n_freqs, 0, n_pts / 2);
        psd_freq_var.putVar(psd_freq.data());

        // start index for hist
        std::vector<std::size_t> start_index_hist = {0,0};
        // size for hist
        std::vector<std::size_t> size_hist = {1,n_hist_bins};

        // size for psd
        std::vector<std::size_t> size_psd = {1,TULA_SIZET(n_freqs)};

        // loop through detectors
        for (Eigen::Index i=0; i<in.scans.data.cols(); i++) {
            // increment starting row
            start_index_hist[0] = i;
            // get data for detector
            Eigen::VectorXd scan = in.scans.data.col(i);
            // calculate histogram
            auto [h, h_bins] = engine_utils::calc_hist(scan, n_hist_bins);
            // add data to histogram variable
            hist_var.putVar(start_index_hist, size_hist, h.data());
            // add data to histogram bins variable
            hist_bin_var.putVar(start_index_hist, size_hist, h_bins.data());

            // apply hanning window
            Eigen::VectorXd hanning = (0.5 - 0.5 * Eigen::ArrayXd::LinSpaced(n_pts, 0, 2.0 * pi / n_pts * (n_pts - 1)).cos());

            scan.noalias() = (scan.array()*hanning.array()).matrix();

            // setup fft
            Eigen::FFT<double> fft;
            fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
            fft.SetFlag(Eigen::FFT<double>::Unscaled);

            // vector to hold fft data
            Eigen::VectorXcd freqdata;

            // do fft
            fft.fwd(freqdata, scan.head(n_pts));
            // calc psd
            Eigen::VectorXd psd = freqdata.cwiseAbs2() / d_freq / n_pts / telescope.d_fsmp;
            // account for negative freqs
            psd.segment(1, n_freqs - 2) *= 2.;

            Eigen::VectorXd smoothed_psd(psd.size());
            engine_utils::smooth_edge_truncate(psd, smoothed_psd, omb.smooth_window);
            psd = std::move(smoothed_psd);

            // put detector's psd into variable
            psd_var.putVar(start_index_hist, size_psd, psd.data());
        }

        fo.close();

    } catch (NcException &e) {
        SPDLOG_ERROR("{}", e.what());
    }
}

template <typename map_buffer_t>
void Engine::write_map_summary(map_buffer_t &mb) {

    SPDLOG_DEBUG("writing map summary files");

    std::string filename = "map_summary";
    std::ofstream f;
    f.open (obsnum_dir_name+"/logs/" + filename + ".log");

    f << "Summary file for maps\n";
    f << "-Citlali version: " << CITLALI_GIT_VERSION << "\n";
    f << "-Kidscpp version: " << KIDSCPP_GIT_VERSION << "\n";
    f << "-Time of file writing: " << engine_utils::current_date_time() << "\n";

    f << "-Reduction type: " << redu_type << "\n";
    f << "-Map type: " << tod_type << "\n";
    f << "-Map grouping: " << map_grouping << "\n";
    f << "-Rows: " << mb.n_rows << "\n";
    f << "-Cols: " << mb.n_cols << "\n";
    f << "-Number of maps: " << n_maps << "\n";
    f << "-Signal map unit: " << mb.sig_unit << "\n";
    f << "-Weight map unit: " << "1/(" + mb.sig_unit + ")^2" << "\n";
    f << "-Kernel maps generated: " << !mb.kernel.empty() << "\n";
    f << "-Coverage maps generated: " << !mb.coverage.empty() << "\n";
    f << "-Noise maps generated: " << !mb.noise.empty() << "\n";
    f << "-Number of noise maps: " << mb.noise.size() << "\n";

    // map to count nans for all maps
    std::map<std::string,int> n_nans;
    n_nans["signal"] = 0;
    n_nans["weight"] = 0;
    n_nans["kernel"] = 0;
    n_nans["coverage"] = 0;
    n_nans["noise"] = 0;

    // maps to hold infs for all maps
    std::map<std::string,int> n_infs;
    n_infs["signal"] = 0;
    n_infs["weight"] = 0;
    n_nans["kernel"] = 0;
    n_infs["coverage"] = 0;
    n_infs["noise"] = 0;

    // loop through maps and count up nans and infs
    for (Eigen::Index i=0; i<mb.signal.size(); i++) {
        n_nans["signal"] = n_nans["signal"] + mb.signal[i].array().isNaN().count();
        n_nans["weight"] = n_nans["weight"] + mb.weight[i].array().isNaN().count();

        // check kernel for nans if requested
        if (!mb.kernel.empty()) {
            n_nans["kernel"] = n_nans["kernel"] + mb.kernel[i].array().isNaN().count();
        }
        // check coverage map for nans if available
        if (!mb.coverage.empty()) {
            n_nans["coverage"] = n_nans["coverage"] + mb.coverage[i].array().isNaN().count();
        }

        n_infs["signal"] = n_infs["signal"] + mb.signal[i].array().isInf().count();
        n_infs["weight"] = n_infs["weight"] + mb.weight[i].array().isInf().count();

        // check kernel for infs if requested
        if (!mb.kernel.empty()) {
            n_infs["kernel"] = n_infs["kernel"] + mb.kernel[i].array().isInf().count();
        }
        // check coverage map for infs if available
        if (!mb.coverage.empty()) {
            n_infs["coverage"] = n_infs["coverage"] + mb.coverage[i].array().isInf().count();
        }

        // loop through noise maps and check for nans and infs
        if (!mb.noise.empty()) {
            for (Eigen::Index j=0; j<mb.noise.size(); j++) {
                Eigen::Tensor<double,2> out = mb.noise[i].chip(j,2);
                auto out_matrix = Eigen::Map<Eigen::MatrixXd>(out.data(), out.dimension(0), out.dimension(1));
                n_nans["noise"] = n_nans["noise"] + out_matrix.array().isNaN().count();
                n_infs["noise"] = n_infs["noise"] + out_matrix.array().isInf().count();
            }
        }
    }

    for (auto const& [key, val] : n_nans) {
         f << "-Number of "+ key + " NaNs: " << val << "\n";
    }

    for (auto const& [key, val] : n_infs) {
        f << "-Number of "+ key + " Infs: " << val << "\n";
    }
}

template <typename fits_io_type, class map_buffer_t>
void Engine::add_phdu(fits_io_type &fits_io, map_buffer_t &mb, Eigen::Index i) {
    // array name
    std::string name = toltec_io.array_name_map[calib.arrays(i)];

    // add unit conversions
    if (rtcproc.run_calibrate) {
        if (mb->sig_unit == "MJy/sr") {
            fits_io->at(i).pfits->pHDU().addKey("to_mJy/beam", calib.array_beam_areas[calib.arrays(i)]*MJY_SR_TO_mJY_ASEC,
                                            "Conversion to mJy/beam");
            fits_io->at(i).pfits->pHDU().addKey("to_MJy/sr", 1, "Conversion to MJy/sr");
        }

        else if (mb->sig_unit == "mJy/beam") {
            fits_io->at(i).pfits->pHDU().addKey("to_mJy/beam", 1, "Conversion to mJy/beam");
            fits_io->at(i).pfits->pHDU().addKey("to_MJy/sr", 1/(calib.array_beam_areas[calib.arrays(i)]*MJY_SR_TO_mJY_ASEC),
                                                "Conversion to MJy/sr");
        }
    }

    else {
        fits_io->at(i).pfits->pHDU().addKey("to_mJy/beam", "N/A", "Conversion to mJy/beam");
        fits_io->at(i).pfits->pHDU().addKey("to_MJy/sr", "N/A", "Conversion to MJy/sr");
    }

    // add obsnums
    for (Eigen::Index j=0; j<mb->obsnums.size(); j++) {
        fits_io->at(i).pfits->pHDU().addKey("OBSNUM"+std::to_string(j), mb->obsnums.at(j), "Observation Number " + std::to_string(j));
    }

    // add source
    fits_io->at(i).pfits->pHDU().addKey("SOURCE", telescope.source_name, "Source name");

    // add source flux for beammaps
    if (redu_type == "beammap") {
        fits_io->at(i).pfits->pHDU().addKey("HEADER.SOURCE.FLUX_MJYPERBEAM", beammap_fluxes_mJy_beam[name], "Source flux (mJy/beam)");
        fits_io->at(i).pfits->pHDU().addKey("HEADER.SOURCE.FLUX_MJYPERSR", beammap_fluxes_MJy_Sr[name], "Sorce flux (MJy/Sr)");

        fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.ITER_TOLERANCE", beammap_iter_tolerance, "Beammap iteration tolerance");
        fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.ITER_MAX", beammap_iter_max, "Beammap max iterations");
        fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.IS_DEROTATED", beammap_derotate, "Beammap derotated");
        // add reference detector information
        if (beammap_subtract_reference) {
            fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.REF_DET_INDEX", beammap_reference_det, "Beammap Reference det (rotation center)");
            fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.REF_X_T", calib.apt["x_t"](beammap_reference_det), "Az rotation center (arcsec)");
            fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.REF_Y_T", calib.apt["y_t"](beammap_reference_det), "Alt rotation center (arcsec)");
        }
        else {
            fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.REF_DET_INDEX", -99, "Beammap Reference det (rotation center)");
            fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.REF_X_T", "N/A", "Az rotation center (arcsec)");
            fits_io->at(i).pfits->pHDU().addKey("BEAMMAP.REF_Y_T", "N/A", "Alt rotation center (arcsec)");
        }
    }

    // add instrument
    fits_io->at(i).pfits->pHDU().addKey("INSTRUME", "TolTEC", "Instrument");
    // add telescope
    fits_io->at(i).pfits->pHDU().addKey("TELESCOP", "LMT", "Telescope");
    // add wavelength
    fits_io->at(i).pfits->pHDU().addKey("WAV", name, "wavelength");
    // add citlali version
    fits_io->at(i).pfits->pHDU().addKey("VERSION", CITLALI_GIT_VERSION, "CITLALI_GIT_VERSION");
    // add kids version
    fits_io->at(i).pfits->pHDU().addKey("KIDS", KIDSCPP_GIT_VERSION, "KIDSCPP_GIT_VERSION");
    // add redu type
    fits_io->at(i).pfits->pHDU().addKey("GOAL", redu_type, "Reduction type");
    // add tod type
    fits_io->at(i).pfits->pHDU().addKey("TYPE", tod_type, "TOD Type");
    // add map grouping
    fits_io->at(i).pfits->pHDU().addKey("GROUPING", map_grouping, "Map grouping");
    // add exposure time
    fits_io->at(i).pfits->pHDU().addKey("EXPTIME", mb->exposure_time, "Exposure time (sec)");
    // add pixel axes
    fits_io->at(i).pfits->pHDU().addKey("RADESYS", telescope.pixel_axes, "Coord Reference Frame");
    // add source ra
    fits_io->at(i).pfits->pHDU().addKey("SRC_RA", telescope.tel_header["Header.Source.Ra"][0], "Source RA (radians)");
    // add source dec
    fits_io->at(i).pfits->pHDU().addKey("SRC_DEC", telescope.tel_header["Header.Source.Dec"][0], "Source Dec (radians)");
    // add map tangent point ra
    fits_io->at(i).pfits->pHDU().addKey("TAN_RA", telescope.tel_header["Header.Source.Ra"][0], "Map Tangent Point RA (radians)");
    // add map tangent point dec
    fits_io->at(i).pfits->pHDU().addKey("TAN_DEC", telescope.tel_header["Header.Source.Dec"][0], "Map Tangent Point Dec (radians)");

    // add mean elev
    fits_io->at(i).pfits->pHDU().addKey("MEAN_EL", RAD_TO_DEG*telescope.tel_data["TelElAct"].mean(), "mean elevation (deg)");

    // add mean tau
    Eigen::VectorXd tau_el(1);
    tau_el << telescope.tel_data["TelElAct"].mean();
    auto tau_freq = rtcproc.calibration.calc_tau(tau_el, telescope.tau_225_GHz);

    fits_io->at(i).pfits->pHDU().addKey("MEAN_TAU", tau_freq[i](0), "mean tau (" + name + ")");

    // add sample rate
    fits_io->at(i).pfits->pHDU().addKey("SAMPRATE", telescope.fsmp, "sample rate (Hz)");

    // add apt table to header
    std::vector<string> result;
    std::stringstream ss(calib.apt_filepath);
    std::string item;
    char delim = '/';

    while (getline (ss, item, delim)) {
        result.push_back(item);
    }
    fits_io->at(i).pfits->pHDU().addKey("APT", result.back(), "APT table used");

    // add control/runtime parameters
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.VERBOSE", verbose_mode, "Reduced in verbose mode");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.POLARIZED", rtcproc.run_polarization, "Polarized Obs");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DESPIKED", rtcproc.run_despike, "Despiked");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.TODFILTERED", rtcproc.run_tod_filter, "TOD Filtered");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.DOWNSAMPLED", rtcproc.run_downsample, "Downsampled");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CALIBRATED", rtcproc.run_calibrate, "Calibrated");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CALIBRATED.EXTMODEL", rtcproc.calibration.extinction_model, "Extinction model");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.TYPE", ptcproc.weighting_type, "Weighting scheme");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.WTLOW", ptcproc.lower_std_dev, "Lower weight cutoff");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.WEIGHT.WTHIGH", ptcproc.upper_std_dev, "Upper weight cutoff");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CLEANED", ptcproc.run_clean, "Cleaned");
    fits_io->at(i).pfits->pHDU().addKey("CONFIG.CLEANED.NEIG", ptcproc.cleaner.n_eig_to_cut.sum(), "Number of eigenvalues removed");

    // add telescope file header information
    for (auto const& [key, val] : telescope.tel_header) {
        fits_io->at(i).pfits->pHDU().addKey(key, val(0), key);
    }
}

template <typename fits_io_type, class map_buffer_t>
void Engine::write_maps(fits_io_type &fits_io, fits_io_type &noise_fits_io, map_buffer_t &mb, Eigen::Index i) {

    // get name for extension layer
    std::string map_name = "";

    if (map_grouping!="array") {
        if (map_grouping=="nw") {
            map_name = map_name + "nw_" + std::to_string(calib.nws(i)) + "_";
        }

        else if (map_grouping=="detector") {
            map_name = map_name + "det_" + std::to_string(i) + "_";
        }
    }

    // get the array for the given map
    Eigen::Index map_index = arrays_to_maps(i);
    // get the stokes parameter for the given map
    Eigen::Index stokes_index = maps_to_stokes(i);

    // array name
    std::string name = toltec_io.array_name_map[map_index];

    // signal map
    fits_io->at(map_index).add_hdu("signal_" + map_name + rtcproc.polarization.stokes_params[stokes_index], mb->signal[i]);
    fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(),mb->wcs);
    fits_io->at(map_index).hdus.back()->addKey("UNIT", mb->sig_unit, "Unit of map");

    // weight map
    fits_io->at(map_index).add_hdu("weight_" + map_name + rtcproc.polarization.stokes_params[stokes_index], mb->weight[i]);
    fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(),mb->wcs);
    fits_io->at(map_index).hdus.back()->addKey("UNIT", "1/("+mb->sig_unit+")^2", "Unit of map");

    // kernel map
    if (rtcproc.run_kernel) {
        fits_io->at(map_index).add_hdu("kernel_" + map_name + rtcproc.polarization.stokes_params[stokes_index], mb->kernel[i]);
        fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(),mb->wcs);
        fits_io->at(map_index).hdus.back()->addKey("UNIT", mb->sig_unit, "Unit of map");
    }

    // coverage map
    if (!mb->coverage.empty()) {
        fits_io->at(map_index).add_hdu("coverage_" + map_name + rtcproc.polarization.stokes_params[stokes_index], mb->coverage[i]);
        fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(),mb->wcs);
        fits_io->at(map_index).hdus.back()->addKey("UNIT", "sec", "Unit of map");
    }

    /* coverage bool and signal-to-noise maps */
    if (!mb->coverage.empty()) {
        Eigen::MatrixXd ones, zeros;
        ones.setOnes(mb->weight[i].rows(), mb->weight[i].cols());
        zeros.setZero(mb->weight[i].rows(), mb->weight[i].cols());

        // get weight threshold for current map
        auto [weight_threshold, cov_ranges, cov_n_rows, cov_n_cols] = mb->calc_cov_region(i);
        // if weight is less than threshold, set to zero, otherwise set to one
        Eigen::MatrixXd coverage_bool = (mb->weight[i].array() < weight_threshold).select(zeros,ones);

        // coverage bool map
        fits_io->at(map_index).add_hdu("coverage_bool_" + map_name + rtcproc.polarization.stokes_params[stokes_index], coverage_bool);
        fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(),mb->wcs);
        fits_io->at(map_index).hdus.back()->addKey("UNIT", "N/A", "Unit of map");

        // signal-to-noise map
        Eigen::MatrixXd sig2noise = mb->signal[i].array()*sqrt(mb->weight[i].array());
        fits_io->at(map_index).add_hdu("sig2noise_" + map_name + rtcproc.polarization.stokes_params[stokes_index], sig2noise);
        fits_io->at(map_index).add_wcs(fits_io->at(map_index).hdus.back(),mb->wcs);
        fits_io->at(map_index).hdus.back()->addKey("UNIT", "N/A", "Unit of map");
    }

    // add primary hdu
    //add_phdu(fits_io, mb, map_index);

    // write noise maps
    if (!mb->noise.empty()) {
        for (Eigen::Index n=0; n<mb->n_noise; n++) {
            Eigen::Tensor<double,2> out = mb->noise[i].chip(n,2);
            auto out_matrix = Eigen::Map<Eigen::MatrixXd>(out.data(), out.dimension(0), out.dimension(1));

            noise_fits_io->at(map_index).add_hdu("signal_" + map_name + std::to_string(n) + "_" +
                                                 rtcproc.polarization.stokes_params[stokes_index],
                                                 out_matrix);
            noise_fits_io->at(map_index).add_wcs(noise_fits_io->at(map_index).hdus.back(),mb->wcs);
            noise_fits_io->at(map_index).hdus.back()->addKey("UNIT", mb->sig_unit, "Unit of map");
        }
        // add primary hdu to noise files
        //add_phdu(noise_fits_io, mb, map_index);
    }
}

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::write_psd(map_buffer_t &mb, std::string dir_name) {
    std::string filename, noise_filename;

    // raw obs maps
    if constexpr (map_t == mapmaking::RawObs) {
        filename = toltec_io.create_filename<engine_utils::toltecIO::toltec, engine_utils::toltecIO::psd,
                                             engine_utils::toltecIO::raw>
                   (dir_name, redu_type, "", obsnum, telescope.sim_obs);
    }
    // filtered obs maps
    else if constexpr (map_t == mapmaking::FilteredObs) {
        filename = toltec_io.create_filename<engine_utils::toltecIO::toltec, engine_utils::toltecIO::psd,
                                             engine_utils::toltecIO::filtered>
                   (dir_name, redu_type, "", obsnum, telescope.sim_obs);
    }
    // raw coadded maps
    else if constexpr (map_t == mapmaking::RawCoadd) {
        filename = toltec_io.create_filename<engine_utils::toltecIO::toltec, engine_utils::toltecIO::psd,
                                             engine_utils::toltecIO::raw>
                   (dir_name, "", "", "", telescope.sim_obs);
    }
    // filtered coadded maps
    else if constexpr (map_t == mapmaking::FilteredCoadd) {
        filename = toltec_io.create_filename<engine_utils::toltecIO::toltec, engine_utils::toltecIO::psd,
                                             engine_utils::toltecIO::filtered>
                   (dir_name, "", "", "", telescope.sim_obs);
    }

    // create file
    netCDF::NcFile fo(filename + ".nc", netCDF::NcFile::replace);

    // loop through psd vector
    for (Eigen::Index i=0; i<mb->psds.size(); i++) {
        std::string map_name = "";

        if (map_grouping!="array") {
            if (map_grouping=="nw") {
                map_name = map_name + "nw_" + std::to_string(calib.nws(i)) + "_";
            }

            else if (map_grouping=="detector") {
                map_name = map_name + "det_" + std::to_string(i) + "_";
            }
        }

        // get the array for the given map
        Eigen::Index map_index = arrays_to_maps(i);
        // get the stokes parameter for the given map
        Eigen::Index stokes_index = maps_to_stokes(i);

        auto array = calib.arrays[map_index];
        std::string name = toltec_io.array_name_map[array] + "_" + map_name + rtcproc.polarization.stokes_params[stokes_index];

        // add dimensions
        netCDF::NcDim psd_dim = fo.addDim(name + "_nfreq",mb->psds[i].size());
        netCDF::NcDim pds_2d_row_dim = fo.addDim(name + "_rows",mb->psd_2ds[i].rows());
        netCDF::NcDim pds_2d_col_dim = fo.addDim(name + "_cols",mb->psd_2ds[i].cols());

        std::vector<netCDF::NcDim> dims;
        dims.push_back(pds_2d_row_dim);
        dims.push_back(pds_2d_col_dim);

        // psd
        netCDF::NcVar psd_v = fo.addVar(name + "_psd",netCDF::ncDouble, psd_dim);
        psd_v.putVar(mb->psds[i].data());

        // psd freq
        netCDF::NcVar psd_freq_v = fo.addVar(name + "_psd_freq",netCDF::ncDouble, psd_dim);
        psd_freq_v.putVar(mb->psd_freqs[i].data());

        // transpose 2d psd and freq
        Eigen::MatrixXd psd_2d_transposed = mb->psd_2ds[i].transpose();
        Eigen::MatrixXd psd_2d_freq_transposed = mb->psd_2d_freqs[i].transpose();

        // 2d psd
        netCDF::NcVar psd_2d_v = fo.addVar(name + "_psd_2d",netCDF::ncDouble, dims);
        psd_2d_v.putVar(psd_2d_transposed.data());

        // 2d psd freq
        netCDF::NcVar psd_2d_freq_v = fo.addVar(name + "_psd_2d_freq",netCDF::ncDouble, dims);
        psd_2d_freq_v.putVar(psd_2d_freq_transposed.data());
    }
    // close file
    fo.close();
}

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::write_hist(map_buffer_t &mb, std::string dir_name) {
    std::string filename;

    // raw obs maps
    if constexpr (map_t == mapmaking::RawObs) {
        filename = toltec_io.create_filename<engine_utils::toltecIO::toltec, engine_utils::toltecIO::hist,
                                             engine_utils::toltecIO::raw>
                   (dir_name, redu_type, "", obsnum, telescope.sim_obs);
    }

    // filtered obs maps
    else if constexpr (map_t == mapmaking::FilteredObs) {
        filename = toltec_io.create_filename<engine_utils::toltecIO::toltec, engine_utils::toltecIO::hist,
                                             engine_utils::toltecIO::filtered>
                    (dir_name, redu_type, "", obsnum, telescope.sim_obs);
    }

    // raw coadded maps
    else if constexpr (map_t == mapmaking::RawCoadd) {
        filename = toltec_io.create_filename<engine_utils::toltecIO::toltec, engine_utils::toltecIO::hist,
                                             engine_utils::toltecIO::raw>
                   (dir_name, "", "", "", telescope.sim_obs);
    }

    // filtered coadded maps
    else if constexpr (map_t == mapmaking::FilteredCoadd) {
        filename = toltec_io.create_filename<engine_utils::toltecIO::toltec, engine_utils::toltecIO::hist,
                                             engine_utils::toltecIO::filtered>
                   (dir_name, "", "", "", telescope.sim_obs);
    }

    netCDF::NcFile fo(filename + ".nc", netCDF::NcFile::replace);
    netCDF::NcDim hist_bins_dim = fo.addDim("n_bins", mb->hist_n_bins);

    Eigen::Index k = 0;

    for (Eigen::Index i=0; i<mb->hists.size(); i++) {

        std::string map_name = "";

        if (map_grouping!="array") {
            if (map_grouping=="nw") {
                map_name = map_name + "nw_" + std::to_string(calib.nws(i)) + "_";
            }

            else if (map_grouping=="detector") {
                map_name = map_name + "det_" + std::to_string(i) + "_";
            }
        }
        // get the array for the given map
        Eigen::Index map_index = arrays_to_maps(i);
        // get the stokes parameter for the given map
        Eigen::Index stokes_index = maps_to_stokes(i);

        auto array = calib.arrays[map_index];
        std::string name = toltec_io.array_name_map[array] + "_" + map_name + rtcproc.polarization.stokes_params[stokes_index];

        // histogram bins
        netCDF::NcVar hist_bins_v = fo.addVar(name + "_bins",netCDF::ncDouble, hist_bins_dim);
        hist_bins_v.putVar(mb->hist_bins[i].data());

        // histogram
        netCDF::NcVar hist_v = fo.addVar(name + "_hist",netCDF::ncDouble, hist_bins_dim);
        hist_v.putVar(mb->hists[i].data());
    }
    // close file
    fo.close();
}

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::run_wiener_filter(map_buffer_t &mb) {
    // pointer to map buffer
    mapmaking::ObsMapBuffer* pmb = &mb;
    // pointer to data file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* f_io = NULL;
    // pointer to noise file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* n_io = NULL;

    // directory name
    std::string dir_name;

    // filtered obs maps
    if constexpr (map_t == mapmaking::FilteredObs) {
        f_io = &filtered_fits_io_vec;
        n_io = &filtered_noise_fits_io_vec;
        dir_name = obsnum_dir_name + "filtered/";
    }

    // filtered coadded maps
    else if constexpr (map_t == mapmaking::FilteredCoadd) {
        f_io = &filtered_coadd_fits_io_vec;
        n_io = &filtered_coadd_noise_fits_io_vec;
        dir_name = coadd_dir_name + "filtered/";
    }

    for (Eigen::Index i=0; i<f_io->size(); i++) {
        // get the array for the given map
        //Eigen::Index map_index = arrays_to_maps(i);
        // add primary hdu
        add_phdu(f_io, pmb, i);

        if (!pmb->noise.empty()) {
            add_phdu(n_io, pmb, i);
        }
    }

    for (Eigen::Index i=0; i<n_maps; i++) {
        // current array
        auto array = maps_to_arrays(i);
        // get file index
        auto map_index = arrays_to_maps(i);
        // init fwhm in pixels
        wiener_filter.init_fwhm = toltec_io.array_fwhm_arcsec[array]*ASEC_TO_RAD/omb.pixel_size_rad;
        // make wiener filter template
        wiener_filter.make_template(mb,calib.apt, wiener_filter.gaussian_template_fwhm_rad[toltec_io.array_name_map[i]],i);
        // run the filter for the current map
        wiener_filter.filter_maps(mb,i);

        // filter noise maps
        if (run_noise) {
            tula::logging::progressbar pb(
                [](const auto &msg) { SPDLOG_INFO("{}", msg); }, 100,
                "filtering noise");

            for (Eigen::Index j=0; j<mb.n_noise; j++) {
                wiener_filter.filter_noise(mb, i, j);
                pb.count(mb.n_noise, mb.n_noise / 100);
            }
        }

        if (wiener_filter.normalize_error) {
            SPDLOG_INFO("renormalizing errors");
            // get mean error from weight maps
            mb.calc_mean_err();

            // get mean map rms from noise maps
            mb.calc_mean_rms();

            // get rescaled normalization factor
            auto noise_factor = (1./pow(mb.mean_rms.array(),2.))*mb.mean_err.array();

            // re-normalize weight map
            mb.weight[i].noalias() = mb.weight[i]*noise_factor(i);
        }

        // write maps immediately after filtering due to computation time
        write_maps(f_io,n_io,pmb,i);

        SPDLOG_INFO("file has been written to:");
        SPDLOG_INFO("{}.fits",f_io->at(map_index).filepath);

        // explicitly destroy the fits file after we're done with it
        if (i > 0) {
            // check if we're moving onto a new file
            if (arrays_to_maps(i) > arrays_to_maps(i-1)) {
                f_io->at(map_index).pfits->destroy();
            }
        }
    }

    // clear fits file vectors to ensure its closed.
    f_io->clear();
    n_io->clear();
}

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::find_sources(map_buffer_t &mb) {
    // clear all source vectors
    mb.n_sources.clear();
    mb.row_source_locs.clear();
    mb.col_source_locs.clear();
    for (Eigen::Index i=0; i<n_maps; i++) {
        // update source vectors
        mb.n_sources.push_back(0);
        mb.row_source_locs.push_back(Eigen::VectorXi::Ones(1));
        mb.col_source_locs.push_back(Eigen::VectorXi::Ones(1));

        // default value of -99 to keep size of vectors same as map vector
        mb.row_source_locs.back()*=-99;
        mb.col_source_locs.back()*=-99;

        // run source finder
        auto sources_found = mb.find_sources(i);

        if (sources_found) {
            SPDLOG_INFO("{} source(s) found", mb.n_sources.back());
        }
        else {
            SPDLOG_INFO("no sources found");
        }
    }

    // count up the total number of sources
    Eigen::Index n_sources = 0;
    for (const auto &sources: mb.n_sources) {
        n_sources += sources;
    }

    // matrix to store source parameters
    mb.source_params.setZero(n_sources,6);
    mb.source_perror.setZero(n_sources,6);

    // keep track of row in total source count
    Eigen::Index k = 0;

    // now loop through and fit the sources
    for (Eigen::Index i=0; i<n_maps; i++) {
        // skip map if no sources found
        if (mb.n_sources[i] > 0) {
            // current array
            auto array = maps_to_arrays(i);
            // get file index
            auto map_index = arrays_to_maps(i);
            // init fwhm in pixels
            auto init_fwhm = toltec_io.array_fwhm_arcsec[array]*ASEC_TO_RAD/mb.pixel_size_rad;

            // placeholder vectors for grppi map
            std::vector<int> source_in_vec, source_out_vec;

            source_in_vec.resize(mb.n_sources[i]);
            std::iota(source_in_vec.begin(), source_in_vec.end(), 0);
            source_out_vec.resize(mb.n_sources[i]);

            // loop through sources and fit them
            grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), source_in_vec, source_out_vec, [&](auto j) {
                // update source rows and cols
                double init_row = mb.row_source_locs[i](j);
                double init_col = mb.col_source_locs[i](j);

                // fit source
                auto [params, perror, good_fit] =
                    map_fitter.fit_to_gaussian<engine_utils::mapFitter::pointing>(mb.signal[i], mb.weight[i],
                                                                                  init_fwhm, init_row, init_col);
                if (good_fit) {
                    // rescale fit params from pixel to on-sky units
                    params(1) = RAD_TO_ASEC*mb.pixel_size_rad*(params(1) - (mb.n_cols)/2);
                    params(2) = RAD_TO_ASEC*mb.pixel_size_rad*(params(2) - (mb.n_rows)/2);
                    params(3) = RAD_TO_ASEC*STD_TO_FWHM*mb.pixel_size_rad*(params(3));
                    params(4) = RAD_TO_ASEC*STD_TO_FWHM*mb.pixel_size_rad*(params(4));

                    // rescale fit errors from pixel to on-sky units
                    perror(1) = RAD_TO_ASEC*mb.pixel_size_rad*(perror(1));
                    perror(2) = RAD_TO_ASEC*mb.pixel_size_rad*(perror(2));
                    perror(3) = RAD_TO_ASEC*STD_TO_FWHM*mb.pixel_size_rad*(perror(3));
                    perror(4) = RAD_TO_ASEC*STD_TO_FWHM*mb.pixel_size_rad*(perror(4));

                    // add source params and errors to table
                    mb.source_params.row(k+j) = params;
                    mb.source_perror.row(k+j) = perror;
                }
                return 0;
            });

            // update row
            k = k + mb.n_sources[i];
        }
    }
}

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::write_sources(map_buffer_t &mb, std::string dir_name) {
    std::string source_filename;
    if constexpr (map_t == mapmaking::FilteredObs) {
        source_filename = toltec_io.create_filename<engine_utils::toltecIO::source, engine_utils::toltecIO::map,
                                                    engine_utils::toltecIO::filtered>
                          (dir_name, redu_type, "", obsnum, telescope.sim_obs);
    }
    else if constexpr (map_t == mapmaking::FilteredCoadd) {
        source_filename = toltec_io.create_filename<engine_utils::toltecIO::source, engine_utils::toltecIO::map,
                                                    engine_utils::toltecIO::filtered>
                          (dir_name, "", "", "", telescope.sim_obs);
    }

    std::vector<std::string> source_header = {
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
        "angle_err"
    };

    // meta information for source table
    YAML::Node source_meta;

    // add obsnums
    for (Eigen::Index i=0; i<mb->obsnums.size(); i++) {
        // add obsnum to meta data
        source_meta["obsnum" + std::to_string(i)] = mb->obsnums[i];
    }

    // add source name
    source_meta["Source"] = telescope.source_name;

    // add date
    source_meta["Date"] = engine_utils::current_date_time();

    // array
    source_meta["array"].push_back("units: N/A");
    source_meta["array"].push_back("array");

    // amplitude
    source_meta["amp"].push_back("units: " + mb->sig_unit);
    source_meta["amp"].push_back("fitted amplitude");
    // amplitude error
    source_meta["amp_err"].push_back("units: " + mb->sig_unit);
    source_meta["amp_err"].push_back("fitted amplitude error");
    // x position
    source_meta["x_t"].push_back("units: arcsec");
    source_meta["x_t"].push_back("fitted azimuthal offset");
    // x position error
    source_meta["x_t_err"].push_back("units: arcsec");
    source_meta["x_t_err"].push_back("fitted azimuthal offset error");
    // y position
    source_meta["y_t"].push_back("units: arcsec");
    source_meta["y_t"].push_back("fitted altitude offset");
    // y position error
    source_meta["y_t_err"].push_back("units: arcsec");
    source_meta["y_t_err"].push_back("fitted altitude offset error");
    // azimuthal fwhm
    source_meta["a_fwhm"].push_back("units: arcsec");
    source_meta["a_fwhm"].push_back("fitted azimuthal FWHM");
    // azimuthal fwhm error
    source_meta["a_fwhm_err"].push_back("units: arcsec");
    source_meta["a_fwhm_err"].push_back("fitted azimuthal FWHM error");
    // elevation fwhm
    source_meta["b_fwhm"].push_back("units: arcsec");
    source_meta["b_fwhm"].push_back("fitted altitude FWMH");
    // elevation fwhm error
    source_meta["b_fwhm_err"].push_back("units: arcsec");
    source_meta["b_fwhm_err"].push_back("fitted altitude FWMH error");
    // rotation angle
    source_meta["angle"].push_back("units: radians");
    source_meta["angle"].push_back("fitted rotation angle");
    // rotation angle error
    source_meta["angle_err"].push_back("units: radians");
    source_meta["angle_err"].push_back("fitted rotation angle error");

    // count up the total number of sources
    Eigen::Index n_sources = 0;
    for (const auto &sources: mb->n_sources) {
        n_sources += sources;
    }

    // matrix to hold source information
    Eigen::MatrixXf source_table(n_sources, 13);

    // loop through params and add arrays
    Eigen::Index k=0;
    for (Eigen::Index i=0; i<mb->n_sources.size(); i++) {
        if (mb->n_sources[i]!=0) {
            for (Eigen::Index j=0; j<mb->n_sources[i]; j++) {
                source_table(k,0) = maps_to_arrays(i);
                k++;
            }
        }
    }

    // populate table
    Eigen::Index j = 0;
    for (Eigen::Index i=1; i<12; i=i+2) {
        source_table.col(i) = mb->source_params.col(j).template cast <float> ();
        source_table.col(i+1) = mb->source_perror.col(j).template cast <float> ();
        j++;
    }

    // write table
    to_ecsv_from_matrix(source_filename, source_table, source_header, source_meta);
}
