#include <citlali_config/gitversion.h>
#include <citlali/core/engine/engine.h>
#include <citlali/core/utils/utils.h>

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
    get_value(config, output_dir, missing_keys, invalid_keys, std::tuple{"runtime","output_dir"});
    get_value(config, n_threads, missing_keys, invalid_keys, std::tuple{"runtime","n_threads"});
    get_value(config, parallel_policy, missing_keys, invalid_keys, std::tuple{"runtime","parallel_policy"});
    get_value(config, redu_type, missing_keys, invalid_keys, std::tuple{"runtime","reduction_type"},{"science","pointing","beammap"});
    get_value(config, use_subdir, missing_keys, invalid_keys, std::tuple{"runtime","use_subdir"});

    /* timestream */
    get_value(config, run_timestream, missing_keys, invalid_keys, std::tuple{"timestream","enabled"});
    get_value(config, tod_type, missing_keys, invalid_keys, std::tuple{"timestream","type"});
    get_value(config, run_tod_output, missing_keys, invalid_keys, std::tuple{"timestream","output","enabled"});
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

    /* cleaning */
    get_value(config, ptcproc.run_clean, missing_keys, invalid_keys, std::tuple{"timestream","clean","enabled"});
    get_value(config, ptcproc.cleaner.n_eig_to_cut, missing_keys, invalid_keys, std::tuple{"timestream","clean","n_eig_to_cut"});
    get_value(config, ptcproc.cleaner.grouping, missing_keys, invalid_keys, std::tuple{"timestream","clean","grouping"});
    get_value(config, ptcproc.cleaner.cut_std, missing_keys, invalid_keys, std::tuple{"timestream","clean","cut_std"});

    /* mapmaking */
    get_value(config, run_mapmaking, missing_keys, invalid_keys, std::tuple{"mapmaking","enabled"});
    get_value(config, map_grouping, missing_keys, invalid_keys, std::tuple{"mapmaking","grouping"});
    get_value(config, map_method, missing_keys, invalid_keys, std::tuple{"mapmaking","method"});

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
    get_value(config, omb.sig_unit, missing_keys, invalid_keys, std::tuple{"mapmaking","cunit"});

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

        omb.wcs.cunit.push_back("arcsec");
        omb.wcs.cunit.push_back("arcsec");
        omb.wcs.cunit.push_back("Hz");
        omb.wcs.cunit.push_back("");

        omb.wcs.cdelt[0] *= RAD_TO_ASEC;
        omb.wcs.cdelt[1] *= RAD_TO_ASEC;
    }

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

    for (auto const& [arr_index, arr_name] : toltec_io.array_name_map) {
        get_value(config, lower_fwhm_arcsec[arr_name], missing_keys, invalid_keys, std::tuple{"beammap","lower_fwhm_arcsec",arr_name});
        get_value(config, upper_fwhm_arcsec[arr_name], missing_keys, invalid_keys, std::tuple{"beammap","upper_fwhm_arcsec",arr_name});
        get_value(config, lower_sig2noise[arr_name], missing_keys, invalid_keys, std::tuple{"beammap","lower_sig2noise",arr_name});
    }

}

template<typename CT>
void Engine::get_photometry_config(CT &config) {

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

        tod_filename[stokes_param] = filename + "_" + stokes_param + ".nc";
        SPDLOG_INFO("tod_filename {}", tod_filename);
        netCDF::NcFile fo(tod_filename[stokes_param], netCDF::NcFile::replace);
        netCDF::NcDim n_pts_dim = fo.addDim("n_pts");
        netCDF::NcDim n_scan_indices_dim = fo.addDim("n_scan_indices", telescope.scan_indices.rows());
        netCDF::NcDim n_scans_dim = fo.addDim("n_scans", telescope.scan_indices.cols());

        Eigen::Index n_dets;

        if (stokes_param=="I") {
            n_dets = calib.apt["array"].size();
        }

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
        netCDF::NcVar scans_v = fo.addVar("scans",netCDF::ncDouble, dims);
        // flags
        netCDF::NcVar flags_v = fo.addVar("flags",netCDF::ncDouble, dims);
        // kernel
        if (rtcproc.run_kernel) {
            netCDF::NcVar kernel_v = fo.addVar("kernel",netCDF::ncDouble, dims);
        }

        // add apt table
        for (auto const& x: calib.apt) {
            netCDF::NcVar apt_v = fo.addVar(x.first,netCDF::ncDouble, n_dets_dim);
        }

        // add telescope parameters
        for (auto const& x: telescope.tel_data) {
            netCDF::NcVar tel_data_v = fo.addVar(x.first,netCDF::ncDouble, n_pts_dim);
        }

        // weights
        if (tod_output_type == "ptc") {
            netCDF::NcDim n_weights_dim = fo.addDim("n_weights");
            std::vector<netCDF::NcDim> weight_dims = {n_scans_dim, n_weights_dim};
            netCDF::NcVar weights_v = fo.addVar("weights",netCDF::ncDouble, n_weights_dim);
        }

        fo.close();
    }
}

void Engine::write_maps(Eigen::Index i, Eigen::Index j, Eigen::Index k) {
    // signal map
    fits_io_vec[j].add_hdu("signal_" + rtcproc.polarization.stokes_params[i], omb.signal[k]);
    fits_io_vec[j].add_wcs(fits_io_vec[j].hdus.back(),omb.wcs);

    // weight map
    fits_io_vec[j].add_hdu("weight_" + rtcproc.polarization.stokes_params[i], omb.weight[k]);
    fits_io_vec[j].add_wcs(fits_io_vec[j].hdus.back(),omb.wcs);

    // kernel map
    if (rtcproc.run_kernel) {
        fits_io_vec[j].add_hdu("kernel_" + rtcproc.polarization.stokes_params[i], omb.kernel[k]);
        fits_io_vec[j].add_wcs(fits_io_vec[j].hdus.back(),omb.wcs);
    }

    // coverage map
    fits_io_vec[j].add_hdu("coverage_" + rtcproc.polarization.stokes_params[i], omb.coverage[k]);
    fits_io_vec[j].add_wcs(fits_io_vec[j].hdus.back(),omb.wcs);

    // signal-to-noise map
    Eigen::MatrixXd sig2noise = omb.signal[k].array()*sqrt(omb.weight[k].array());
    fits_io_vec[j].add_hdu("sig2noise_" + rtcproc.polarization.stokes_params[i], sig2noise);
    fits_io_vec[j].add_wcs(fits_io_vec[j].hdus.back(),omb.wcs);

    // add telescope file header information
    for (auto const& [key, val] : telescope.tel_header) {
        fits_io_vec[j].pfits->pHDU().addKey(key, val(0), key);
    }

    fits_io_vec[j].pfits->pHDU().addKey("OBSNUM", obsnum, "Observation Number");
    fits_io_vec[j].pfits->pHDU().addKey("CITLALI_GIT_VERSION", CITLALI_GIT_VERSION, "CITLALI_GIT_VERSION");
}
