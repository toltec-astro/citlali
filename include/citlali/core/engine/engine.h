#pragma once

#include <memory>
#include <string>
#include <vector>
#include <omp.h>
#include <Eigen/Core>

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
#include <citlali/core/mapmaking/psd.h>
#include <citlali/core/mapmaking/histogram.h>
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
    bool run_timestream;

    // output timestreams
    bool run_tod_output;

    // controls for mapmaking
    bool run_mapmaking;
    bool run_coadd;
    bool run_noise;
    bool run_map_filter;
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
    mapmaking::PSD psd;
    mapmaking::Histogram hist;
};

struct beammapControls {
    // source name
    std::string beammap_source_name;

    // beammap source position
    double beammap_ra_rad, beammap_dec_rad;

    // fluxes and errs
    std::map<std::string, double> beammap_fluxes, beammap_err;

    // maximum beammap iterations
    int beammap_iter_max;

    // beammap tolerance
    double beammap_iter_tolerance;

    // beammap reference detector
    Eigen::Index beammap_reference_det;

    // upper and lower limits of psd for sensitivity calc
    Eigen::VectorXd sens_psd_limits;

    // limits for fwhm and sig2noise  for flagging
    std::map<std::string, double> lower_fwhm_arcsec, upper_fwhm_arcsec, lower_sig2noise;
};

class Engine: public reduControls, public reduClasses, public beammapControls {
public:
    using key_vec_t = std::vector<std::vector<std::string>>;

    // add extra output for debugging
    bool verbose_mode;

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
    std::string tod_output_type;

    // map grouping and algorithm
    std::string map_grouping, map_method;

    // number of maps
    Eigen::Index n_maps;

    // mapping from index in map vector to array index
    Eigen::VectorXI maps_to_arrays;

    // manual pointing offsets
    std::map<std::string, double> pointing_offsets_arcsec;

    // map output files
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> fits_io_vec;
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> coadd_fits_io_vec;
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> noise_fits_io_vec;
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> filtered_coadd_fits_io_vec;
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> filtered_noise_fits_io_vec;

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
    void create_tod_files();

    template <typename fits_io_type, class map_buffer_t>
    void write_maps(fits_io_type &, map_buffer_t &, Eigen::Index, Eigen::Index, Eigen::Index);
    void write_psd();
    void write_hist();
};

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
    get_value(config, verbose_mode, missing_keys, invalid_keys, std::tuple{"runtime","verbose"});
    get_value(config, output_dir, missing_keys, invalid_keys, std::tuple{"runtime","output_dir"});
    get_value(config, n_threads, missing_keys, invalid_keys, std::tuple{"runtime","n_threads"});
    get_value(config, parallel_policy, missing_keys, invalid_keys, std::tuple{"runtime","parallel_policy"},{"seq","omp"});

    // parallelization for ffts
    omb.parallel_policy = parallel_policy;
    cmb.parallel_policy = parallel_policy;

    get_value(config, redu_type, missing_keys, invalid_keys, std::tuple{"runtime","reduction_type"},{"science","pointing","beammap"});
    get_value(config, use_subdir, missing_keys, invalid_keys, std::tuple{"runtime","use_subdir"});

    /* timestream */
    get_value(config, run_timestream, missing_keys, invalid_keys, std::tuple{"timestream","enabled"});
    get_value(config, tod_type, missing_keys, invalid_keys, std::tuple{"timestream","type"});
    get_value(config, run_tod_output, missing_keys, invalid_keys, std::tuple{"timestream","output","enabled"});

    // tod output requires sequential policy
    if (run_tod_output) {
        SPDLOG_INFO("tod output requires sequential policy");
        parallel_policy = "seq";
        omb.parallel_policy = parallel_policy;
        cmb.parallel_policy = parallel_policy;
    }

    get_value(config, tod_output_type, missing_keys, invalid_keys, std::tuple{"timestream","output", "chunk_type"});
    get_value(config, telescope.time_chunk, missing_keys, invalid_keys, std::tuple{"timestream","chunking", "length_sec"});
    get_value(config, ptcproc.weighting_type, missing_keys, invalid_keys, std::tuple{"timestream","weighting", "type"});
    get_value(config, ptcproc.lower_std_dev, missing_keys, invalid_keys, std::tuple{"timestream","weighting", "lower_std_factor"});
    get_value(config, ptcproc.upper_std_dev, missing_keys, invalid_keys, std::tuple{"timestream","weighting", "upper_std_factor"});

    /* polarization */
    get_value(config, rtcproc.run_polarization, missing_keys, invalid_keys, std::tuple{"timestream","polarimetry", "enabled"});

    if (!rtcproc.run_polarization) {
        rtcproc.polarization.stokes_params = {"I"};
    }

    /* kernel */
    get_value(config, rtcproc.run_kernel, missing_keys, invalid_keys, std::tuple{"timestream","kernel","enabled"});
    get_value(config, rtcproc.kernel.filepath, missing_keys, invalid_keys, std::tuple{"timestream","kernel","filepath"});
    get_value(config, rtcproc.kernel.type, missing_keys, invalid_keys, std::tuple{"timestream","kernel","type"});

    if (rtcproc.kernel.type == "fits") {
        auto img_ext_name_node = config.get_node(std::tuple{"timestream","kernel", "image_ext_names"});
        for (Eigen::Index i=0; i<img_ext_name_node.size(); i++) {
            std::string img_ext_name = config.template get_str(std::tuple{"timestream","kernel", "image_ext_names", i, std::to_string(i)});
            rtcproc.kernel.img_ext_names.push_back(img_ext_name);
        }
    }

    /* despike */
    get_value(config, rtcproc.run_despike, missing_keys, invalid_keys, std::tuple{"timestream","despike","enabled"});
    get_value(config, rtcproc.despiker.min_spike_sigma, missing_keys, invalid_keys, std::tuple{"timestream","despike","min_spike_sigma"});
    get_value(config, rtcproc.despiker.time_constant_sec, missing_keys, invalid_keys, std::tuple{"timestream","despike","time_constant_sec"});
    get_value(config, rtcproc.despiker.window_size, missing_keys, invalid_keys, std::tuple{"timestream","despike","window_size"});

    /* filter */
    get_value(config, rtcproc.run_tod_filter, missing_keys, invalid_keys, std::tuple{"timestream","filter","enabled"});
    get_value(config, rtcproc.filter.a_gibbs, missing_keys, invalid_keys, std::tuple{"timestream","filter","a_gibbs"});
    get_value(config, rtcproc.filter.freq_low_Hz, missing_keys, invalid_keys, std::tuple{"timestream","filter","freq_low_Hz"});
    get_value(config, rtcproc.filter.freq_high_Hz, missing_keys, invalid_keys, std::tuple{"timestream","filter","freq_high_Hz"});
    get_value(config, rtcproc.filter.n_terms, missing_keys, invalid_keys, std::tuple{"timestream","filter","n_terms"});

    // set scan limits
    if (rtcproc.run_tod_filter) {
        telescope.inner_scans_chunk = rtcproc.filter.n_terms;
        rtcproc.despiker.window_size = rtcproc.filter.n_terms;
    }
    else {
        rtcproc.filter.n_terms = 0;
        telescope.inner_scans_chunk = 0;
    }

    /* downsample */
    get_value(config, rtcproc.run_downsample, missing_keys, invalid_keys, std::tuple{"timestream","downsample","enabled"});
    get_value(config, rtcproc.downsampler.factor, missing_keys, invalid_keys, std::tuple{"timestream","downsample","factor"});

    /* calibration */
    get_value(config, rtcproc.run_calibrate, missing_keys, invalid_keys, std::tuple{"timestream","calibration","enabled"});

    // override calibration if in beammap mode
    if (redu_type=="beammap") {
        rtcproc.run_calibrate = false;
    }

    /* cleaning */
    get_value(config, ptcproc.run_clean, missing_keys, invalid_keys, std::tuple{"timestream","clean","enabled"});
    get_value(config, ptcproc.cleaner.n_eig_to_cut, missing_keys, invalid_keys, std::tuple{"timestream","clean","n_eig_to_cut"});
    get_value(config, ptcproc.cleaner.grouping, missing_keys, invalid_keys, std::tuple{"timestream","clean","grouping"});
    get_value(config, ptcproc.cleaner.cut_std, missing_keys, invalid_keys, std::tuple{"timestream","clean","cut_std"});

    /* mapmaking */
    get_value(config, run_mapmaking, missing_keys, invalid_keys, std::tuple{"mapmaking","enabled"});
    get_value(config, map_grouping, missing_keys, invalid_keys, std::tuple{"mapmaking","grouping"});
    get_value(config, map_method, missing_keys, invalid_keys, std::tuple{"mapmaking","method"});
    // histogram
    get_value(config, omb.hist_n_bins, missing_keys, invalid_keys, std::tuple{"mapmaking","histogram","n_bins"});
    cmb.hist_n_bins = omb.hist_n_bins;

    /* wcs */

    // pixel size
    get_value(config, omb.pixel_size_rad, missing_keys, invalid_keys, std::tuple{"mapmaking","pixel_size_arcsec"});

    // convert to radians
    omb.pixel_size_rad *= ASEC_TO_RAD;
    cmb.pixel_size_rad = omb.pixel_size_rad;

    omb.wcs.cdelt.push_back(omb.pixel_size_rad);
    omb.wcs.cdelt.push_back(-omb.pixel_size_rad);

    omb.wcs.cdelt.push_back(1);
    omb.wcs.cdelt.push_back(1);

    // icrs or altaz
    get_value(config, telescope.pixel_axes, missing_keys, invalid_keys, std::tuple{"mapmaking","pixel_axes"});

    double wcs_double;

    // crpix
    get_value(config, wcs_double, missing_keys, invalid_keys, std::tuple{"mapmaking","crpix1"});
    omb.wcs.crpix.push_back(wcs_double);

    get_value(config, wcs_double, missing_keys, invalid_keys, std::tuple{"mapmaking","crpix2"});
    omb.wcs.crpix.push_back(wcs_double);

    omb.wcs.crpix.push_back(1);
    omb.wcs.crpix.push_back(1);

    // crval
    get_value(config, wcs_double, missing_keys, invalid_keys, std::tuple{"mapmaking","crval1_J2000"});
    omb.wcs.crval.push_back(wcs_double);

    get_value(config, wcs_double, missing_keys, invalid_keys, std::tuple{"mapmaking","crval2_J2000"});
    omb.wcs.crval.push_back(wcs_double);

    omb.wcs.crval.push_back(1);
    omb.wcs.crval.push_back(1);

    // naxis
    get_value(config, wcs_double, missing_keys, invalid_keys, std::tuple{"mapmaking","x_size_pix"});
    omb.wcs.naxis.push_back(wcs_double);

    get_value(config, wcs_double, missing_keys, invalid_keys, std::tuple{"mapmaking","y_size_pix"});
    omb.wcs.naxis.push_back(wcs_double);

    omb.wcs.naxis.push_back(1);
    omb.wcs.naxis.push_back(1);

    // map units
    if (rtcproc.run_calibrate) {
        get_value(config, omb.sig_unit, missing_keys, invalid_keys, std::tuple{"mapmaking","cunit"});
    }

    else {
        omb.sig_unit = tod_type;
    }

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

    else if (telescope.pixel_axes == "altaz") {
        omb.wcs.ctype.push_back("AZOFFSET");
        omb.wcs.ctype.push_back("ELOFFSET");
        omb.wcs.ctype.push_back("FREQ");
        omb.wcs.ctype.push_back("STOKES");

        // arcsec if pointing
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

    /* fitting */
    get_value(config, map_fitter.bounding_box_pix, missing_keys, invalid_keys, std::tuple{"source_fitting","bounding_box_arcsec"});
    map_fitter.bounding_box_pix = ASEC_TO_RAD*map_fitter.bounding_box_pix/omb.pixel_size_rad;

    /* coadd */
    get_value(config, run_coadd, missing_keys, invalid_keys, std::tuple{"coadd","enabled"});
    get_value(config, omb.cov_cut, missing_keys, invalid_keys, std::tuple{"coadd","cov_cut"});
    cmb.cov_cut = omb.cov_cut;

    /* noise */
    get_value(config, run_noise, missing_keys, invalid_keys, std::tuple{"coadd","noise_maps","enabled"});
    get_value(config, omb.n_noise, missing_keys, invalid_keys, std::tuple{"coadd","noise_maps","n_noise_maps"});
    cmb.n_noise = omb.n_noise;

    /* filtering */
    get_value(config, run_map_filter, missing_keys, invalid_keys, std::tuple{"coadd","filtering","enabled"});

    /* beammap */
    get_value(config, beammap_iter_max, missing_keys, invalid_keys, std::tuple{"beammap","iter_max"});
    get_value(config, beammap_iter_tolerance, missing_keys, invalid_keys, std::tuple{"beammap","iter_tolerance"});
    get_value(config, beammap_reference_det, missing_keys, invalid_keys, std::tuple{"beammap","reference_det"});

    for (auto const& [arr_index, arr_name] : toltec_io.array_name_map) {
        get_value(config, lower_fwhm_arcsec[arr_name], missing_keys, invalid_keys, std::tuple{"beammap","lower_fwhm_arcsec",arr_name});
        get_value(config, upper_fwhm_arcsec[arr_name], missing_keys, invalid_keys, std::tuple{"beammap","upper_fwhm_arcsec",arr_name});
        get_value(config, lower_sig2noise[arr_name], missing_keys, invalid_keys, std::tuple{"beammap","lower_sig2noise",arr_name});
    }

    // sensitiivty
    sens_psd_limits.resize(2);
    double sens;
    get_value(config, sens, missing_keys, invalid_keys, std::tuple{"beammap","sens_psd_lower_limit"});
    sens_psd_limits(0) = sens;

    get_value(config, sens, missing_keys, invalid_keys, std::tuple{"beammap","sens_psd_upper_limit"});
    sens_psd_limits(1) = sens;
}

template<typename CT>
void Engine::get_photometry_config(CT &config) {
    // beammap name
    get_value(config, beammap_source_name, missing_keys, invalid_keys, std::tuple{"beammap_source","name"});

    // beammap source ra
    get_value(config, beammap_ra_rad, missing_keys, invalid_keys, std::tuple{"beammap_source","ra_deg"});
    beammap_ra_rad = beammap_ra_rad*DEG_TO_RAD;

    // beammap source dec
    get_value(config, beammap_dec_rad, missing_keys, invalid_keys, std::tuple{"beammap_source","dec_deg"});
    beammap_dec_rad = beammap_dec_rad*DEG_TO_RAD;

    Eigen::Index n_fluxes = config.get_node(std::tuple{"beammap_source","fluxes"}).size();

    for (Eigen::Index i=0; i<n_fluxes; i++) {
        auto array = config.get_str(std::tuple{"beammap_source","fluxes",i,"array_name"});

        auto flux = config.template get_typed<double>(std::tuple{"beammap_source","fluxes",i,"value_mJy"});
        auto uncertainty_mJy = config.template get_typed<double>(std::tuple{"beammap_source","fluxes",i,"uncertainty_mJy"});

        beammap_fluxes[array] = flux;
        beammap_err[array] = uncertainty_mJy;
    }
}

template<typename CT>
void Engine::get_astrometry_config(CT &config) {
    // initialize pointing az offset
    pointing_offsets_arcsec["az"] = 0.0;

    // initialize pointing alt offset
    pointing_offsets_arcsec["alt"] = 0.0;
}

template <typename Derived>
auto Engine::calc_map_indices(Eigen::DenseBase<Derived> &det_indices, Eigen::DenseBase<Derived> &nw_indices,
                              Eigen::DenseBase<Derived> &array_indices, std::string stokes_param) {
    // indices for maps
    Eigen::VectorXI map_indices(array_indices.size());

    // set map indices
    if (redu_type == "science") {
        map_indices = array_indices;
    }

    else if (redu_type == "pointing") {
        map_indices = array_indices;
    }

    else if (redu_type == "beammap") {
        map_indices = det_indices;
    }

    // overwrite map indices for networks
    if (map_grouping == "nw") {
        map_indices = nw_indices;
    }

    // overwrite map indices for arrays
    else if (map_grouping == "array") {
        map_indices = array_indices;
    }

    // overwrite map indices for detectors
    else if (map_grouping == "detector") {
        map_indices = det_indices;
    }

    // loop through and calculate map indices
    if (map_grouping != "detector") {
        for (Eigen::Index i=0; i<map_indices.size()-1; i++) {
            if (map_indices(i+1) > (map_indices(i)+1)) {
                auto map_indices_temp = map_indices;
                auto mi_lower = map_indices(i);
                auto mi_upper = map_indices(i+1);
                (map_indices.array() == mi_upper).select(mi_lower+1,map_indices.array());
            }
        }
    }

    // make sure first value is zero
    auto min_index = map_indices.minCoeff();
    map_indices = map_indices.array() - min_index;

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

    for (Eigen::Index i=0; i<calib.n_arrays; i++) {
        auto array = calib.arrays[i];
        std::string array_name = toltec_io.array_name_map[array];
        auto filename = toltec_io.create_filename<engine_utils::toltecIO::toltec,
                                                  engine_utils::toltecIO::map>(obsnum_dir_name, redu_type, array_name,
                                                                               obsnum, telescope.sim_obs);
        fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*> fits_io(filename);

        fits_io_vec.push_back(std::move(fits_io));
    }
}

void Engine::create_tod_files() {
    for (const auto &stokes_param: rtcproc.polarization.stokes_params) {
        auto filename = toltec_io.create_filename<engine_utils::toltecIO::toltec,
                                                  engine_utils::toltecIO::timestream>(obsnum_dir_name, redu_type, "",
                                                                                      obsnum, telescope.sim_obs);

        SPDLOG_INFO("tod_filename {}", filename);
        tod_filename[stokes_param] = filename + "_" + stokes_param + ".nc";
        netCDF::NcFile fo(tod_filename[stokes_param], netCDF::NcFile::replace);

        // add tod output type to file
        netCDF::NcDim n_tod_output_type_dim = fo.addDim("n_tod_output_type",1);
        netCDF::NcVar tod_output_type_var = fo.addVar("tod_output_type",netCDF::ncString, n_tod_output_type_dim);
        const std::vector< size_t > tod_output_type_index = {0};
        tod_output_type_var.putVar(tod_output_type_index, tod_output_type);

        netCDF::NcDim n_pts_dim = fo.addDim("n_pts");
        netCDF::NcDim n_scan_indices_dim = fo.addDim("n_scan_indices", telescope.scan_indices.rows());
        netCDF::NcDim n_scans_dim = fo.addDim("n_scans", telescope.scan_indices.cols());

        Eigen::Index n_dets;

        // set number of dets for unpolarized timestreams
        if (stokes_param=="I") {
            n_dets = calib.apt["array"].size();
        }

        // set number of detectors for polarized timestreams
        else if ((stokes_param == "Q") || (stokes_param == "U")) {
            n_dets = (calib.apt["fg"].array() == 0).count() + (calib.apt["fg"].array() == 1).count();
        }

        netCDF::NcDim n_dets_dim = fo.addDim("n_dets", n_dets);

        std::vector<netCDF::NcDim> dims = {n_pts_dim, n_dets_dim};
        std::vector<netCDF::NcDim> scans_dims = {n_scan_indices_dim, n_scans_dim};

        // scan indices
        netCDF::NcVar scan_indices_v = fo.addVar("scan_indices",netCDF::ncInt, scans_dims);
        Eigen::MatrixXI scans_indices_transposed = telescope.scan_indices.transpose();
        scan_indices_v.putVar(scans_indices_transposed.data());

        // scans
        netCDF::NcVar scans_v = fo.addVar("signal",netCDF::ncDouble, dims);
        // flags
        netCDF::NcVar flags_v = fo.addVar("flags",netCDF::ncDouble, dims);
        // kernel
        if (rtcproc.run_kernel) {
            netCDF::NcVar kernel_v = fo.addVar("kernel",netCDF::ncDouble, dims);
        }

        // detector lat
        netCDF::NcVar det_lat_v = fo.addVar("det_lat",netCDF::ncDouble, dims);
        // detector lon
        netCDF::NcVar det_lon_v = fo.addVar("det_lon",netCDF::ncDouble, dims);

        // add apt table
        for (auto const& x: calib.apt) {
            netCDF::NcVar apt_v = fo.addVar("apt_" + x.first,netCDF::ncDouble, n_dets_dim);
        }

        // add telescope parameters
        for (auto const& x: telescope.tel_data) {
            netCDF::NcVar tel_data_v = fo.addVar(x.first,netCDF::ncDouble, n_pts_dim);
        }

        // weights
        if (tod_output_type == "ptc") {
            std::vector<netCDF::NcDim> weight_dims = {n_scans_dim, n_dets_dim};
            netCDF::NcVar weights_v = fo.addVar("weights",netCDF::ncDouble, weight_dims);
        }

        // add psd and hist variables if in verbose mode
        if (verbose_mode) {
            netCDF::NcDim n_hist_dim = fo.addDim("n_hist_bins", omb.hist_n_bins);
            std::vector<netCDF::NcDim> hist_dims = {n_dets_dim, n_hist_dim};
            netCDF::NcVar hist_v = fo.addVar("hist",netCDF::ncDouble, hist_dims);
            netCDF::NcVar hist_bins_v = fo.addVar("hist_bins",netCDF::ncDouble, hist_dims);
        }

        fo.close();

        SPDLOG_INFO("made timestream files");
    }
}

template <typename fits_io_type, class map_buffer_t>
void Engine::write_maps(fits_io_type &fits_io, map_buffer_t &mb, Eigen::Index i, Eigen::Index j, Eigen::Index k) {
    // array name
    std::string name = toltec_io.array_name_map[j];
    // signal map
    fits_io[j].add_hdu("signal_" + rtcproc.polarization.stokes_params[i], omb.signal[k]);
    fits_io[j].add_wcs(fits_io_vec[j].hdus.back(),omb.wcs);
    fits_io[j].hdus.back()->addKey("UNIT", mb.sig_unit, "Unit of map");

    // weight map
    fits_io[j].add_hdu("weight_" + rtcproc.polarization.stokes_params[i], omb.weight[k]);
    fits_io[j].add_wcs(fits_io_vec[j].hdus.back(),omb.wcs);
    fits_io[j].hdus.back()->addKey("UNIT", "1/("+mb.sig_unit+")", "Unit of map");

    // kernel map
    if (rtcproc.run_kernel) {
        fits_io[j].add_hdu("kernel_" + rtcproc.polarization.stokes_params[i], omb.kernel[k]);
        fits_io[j].add_wcs(fits_io_vec[j].hdus.back(),omb.wcs);
        fits_io[j].hdus.back()->addKey("UNIT", mb.sig_unit, "Unit of map");
    }

    // coverage map
    fits_io[j].add_hdu("coverage_" + rtcproc.polarization.stokes_params[i], omb.coverage[k]);
    fits_io[j].add_wcs(fits_io_vec[j].hdus.back(),omb.wcs);
    fits_io[j].hdus.back()->addKey("UNIT", "sec", "Unit of map");

    // coverage bool map
    Eigen::MatrixXd ones, zeros;
    ones.setOnes(omb.weight[k].rows(), omb.weight[k].cols());
    zeros.setZero(omb.weight[k].rows(), omb.weight[k].cols());

    auto [weight_threshold, cov_ranges, cov_n_rows, cov_n_cols] = omb.calc_cov_region(omb.weight[k]);
    auto coverage_bool = (omb.weight[k].array() < weight_threshold).select(zeros,ones);

    fits_io[j].add_hdu("coverage_bool_" + rtcproc.polarization.stokes_params[i], coverage_bool);
    fits_io[j].add_wcs(fits_io_vec[j].hdus.back(),omb.wcs);
    fits_io[j].hdus.back()->addKey("UNIT", "N/A", "Unit of map");

    // signal-to-noise map
    Eigen::MatrixXd sig2noise = omb.signal[k].array()*sqrt(omb.weight[k].array());
    fits_io[j].add_hdu("sig2noise_" + rtcproc.polarization.stokes_params[i], sig2noise);
    fits_io[j].add_wcs(fits_io_vec[j].hdus.back(),omb.wcs);
    fits_io[j].hdus.back()->addKey("UNIT", "N/A", "Unit of map");

    if (rtcproc.run_calibrate) {
        if (mb.sig_unit == "Mjy/sr") {
            fits_io[j].pfits->pHDU().addKey("to_mJy/beam", calib.array_beam_areas[calib.arrays(j)]*MJY_SR_TO_mJY_ASEC, "Conversion to mJy/beam");
            fits_io[j].pfits->pHDU().addKey("to_Mjy/sr", 1, "Conversion to MJy/sr");
        }

        else if (mb.sig_unit == "mJy/beam") {
            fits_io[j].pfits->pHDU().addKey("to_mJy/beam", 1, "Conversion to mJy/beam");
            fits_io[j].pfits->pHDU().addKey("to_Mjy/sr", 1/calib.mean_flux_conversion_factor[name], "Conversion to MJy/sr");
        }
    }

    else {
        fits_io[j].pfits->pHDU().addKey("to_mJy/beam", "N/A", "Conversion to mJy/beam");
        fits_io[j].pfits->pHDU().addKey("to_Mjy/sr", "N/A", "Conversion to MJy/sr");
    }

    // add obsnum
    fits_io[j].pfits->pHDU().addKey("OBSNUM", obsnum, "Observation Number");
    // add citlali version
    fits_io[j].pfits->pHDU().addKey("VERSION", CITLALI_GIT_VERSION, "CITLALI_GIT_VERSION");
    // add tod type
    fits_io[j].pfits->pHDU().addKey("TYPE", tod_type, "TOD Type");
    // add exposure time
    fits_io[j].pfits->pHDU().addKey("EXPTIME", mb.exposure_time, "Exposure Time");

    // add source ra
    fits_io[j].pfits->pHDU().addKey("SRC_RA", telescope.tel_header["Header.Source.Ra"][0], "Source RA (radians)");
    // add source dec
    fits_io[j].pfits->pHDU().addKey("SRC_DEC", telescope.tel_header["Header.Source.Dec"][0], "Source Dec (radians)");
    // add map tangent point ra
    fits_io[j].pfits->pHDU().addKey("TAN_RA", telescope.tel_header["Header.Source.Ra"][0], "Map Tangent Point RA (radians)");
    // add map tangent point dec
    fits_io[j].pfits->pHDU().addKey("TAN_DEC", telescope.tel_header["Header.Source.Dec"][0], "Map Tangent Point Dec (radians)");

    // add telescope file header information
    for (auto const& [key, val] : telescope.tel_header) {
        fits_io[j].pfits->pHDU().addKey(key, val(0), key);
    }
}

void Engine::write_psd() {
    auto filename = toltec_io.create_filename<engine_utils::toltecIO::toltec,
                                              engine_utils::toltecIO::psd>(obsnum_dir_name, redu_type, "",
                                                                           obsnum, telescope.sim_obs);
    netCDF::NcFile fo(filename + ".nc", netCDF::NcFile::replace);

    Eigen::Index k = 0;

    for (Eigen::Index i=0; i<rtcproc.polarization.stokes_params.size(); i++) {
        for (Eigen::Index j=0; j<n_maps/rtcproc.polarization.stokes_params.size(); j++) {

            auto array = calib.arrays[j];
            std::string name = toltec_io.array_name_map[array] + "_" + rtcproc.polarization.stokes_params[k];

            // add dimensions
            netCDF::NcDim psd_dim = fo.addDim(name + "_nfreq",omb.psds[k].size());
            netCDF::NcDim pds_2d_row_dim = fo.addDim(name + "_rows",omb.psd_2ds[k].rows());
            netCDF::NcDim pds_2d_col_dim = fo.addDim(name + "_cols",omb.psd_2ds[k].cols());

            std::vector<netCDF::NcDim> dims;
            dims.push_back(pds_2d_row_dim);
            dims.push_back(pds_2d_col_dim);

            // psd
            netCDF::NcVar psd_v = fo.addVar(name + "_psd",netCDF::ncDouble, psd_dim);
            psd_v.putVar(omb.psds[k].data());

            // psd freq
            netCDF::NcVar psd_freq_v = fo.addVar(name + "_psd_freq",netCDF::ncDouble, psd_dim);
            psd_freq_v.putVar(omb.psd_freqs[k].data());

            // transpose 2d psd and freq
            Eigen::MatrixXd psd_2d_transposed = omb.psd_2ds[k].transpose();
            Eigen::MatrixXd psd_2d_freq_transposed = omb.psd_2d_freqs[k].transpose();

            // 2d psd
            netCDF::NcVar psd_2d_v = fo.addVar(name + "_psd_2d",netCDF::ncDouble, dims);
            psd_2d_v.putVar(psd_2d_transposed.data());

            // 2d psd freq
            netCDF::NcVar psd_2d_freq_v = fo.addVar(name + "_psd_2d_freq",netCDF::ncDouble, dims);
            psd_2d_freq_v.putVar(psd_2d_freq_transposed.data());

            k++;
        }
    }
    // close file
    fo.close();
}

void Engine::write_hist() {
    auto filename = toltec_io.create_filename<engine_utils::toltecIO::toltec, engine_utils::toltecIO::hist>
                    (obsnum_dir_name, redu_type, "", obsnum, telescope.sim_obs);

    netCDF::NcFile fo(filename + ".nc", netCDF::NcFile::replace);
    netCDF::NcDim hist_bins_dim = fo.addDim("n_bins", omb.hist_n_bins);

    Eigen::Index k = 0;

    for (Eigen::Index i=0; i<rtcproc.polarization.stokes_params.size(); i++) {
        for (Eigen::Index j=0; j<n_maps/rtcproc.polarization.stokes_params.size(); j++) {

            auto array = calib.arrays[j];
            std::string name = toltec_io.array_name_map[array] + "_" + rtcproc.polarization.stokes_params[k];

            // histogram bins
            netCDF::NcVar hist_bins_v = fo.addVar(name + "_bins",netCDF::ncDouble, hist_bins_dim);
            hist_bins_v.putVar(omb.hist_bins[k].data());

            // histogram
            netCDF::NcVar hist_v = fo.addVar(name + "_hist",netCDF::ncDouble, hist_bins_dim);
            hist_v.putVar(omb.hists[k].data());
            k++;
        }
    }
    // close file
    fo.close();
}

