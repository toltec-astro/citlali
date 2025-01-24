# pragma once

struct ObsContainer {
    Telescope telescope;
    Instrument toltec;
    Hwpr hwpr;
    DataMapsContainer obs_maps, coadd_maps;
    DataMapsContainer noise_maps;

    void write_obs();
};

auto ObsContainer::create_phdu() {
    // create primary header
    FitsHeader phdu_base;
    phdu_base.add_key("OBSNUM", obsnum, "Observation number");
    phdu_base.add_key("SOURCE", telescope.source_name, "Source name");
    phdu_base.add_key("INSTRUME", "TolTEC", "Instrument");
    phdu_base.add_key("TELESCOP", "LMT", "Telescope");
    phdu_base.add_key("HWPR", hwpr.run_hwpr, "HWPR installed");
    phdu_base.add_key("PIPELINE", "CITLALI", "Redu pipeline");
    phdu_base.add_key("VERSION", CITLALI_GIT_VERSION, "CITLALI_GIT_VERSION");
    phdu_base.add_key("KIDS", KIDSCPP_GIT_VERSION, "KIDSCPP_GIT_VERSION");
    phdu_base.add_key("TULA", TULA_GIT_VERSION, "TULA_GIT_VERSION");
    phdu_base.add_key("PROJID", telescope.project_id, "Project ID");
    phdu_base.add_key("GOAL", redu_type, "Reduction type");
    phdu_base.add_key("OBSGOAL", telescope.obs_goal, "Obs goal");
    phdu_base.add_key("TYPE", tod_type, "TOD Type");
    phdu_base.add_key("GROUPING", map_grouping, "Map grouping");
    phdu_base.add_key("METHOD", map_method, "Map method");
    phdu_base.add_key("RADESYS", telescope.pixel_axes, "Coord Reference Frame");
    phdu_base.add_key("SRC_RA", telescope.header.at("Header.Source.Ra")(0), "Source RA (radians)");
    phdu_base.add_key("SRC_DEC", telescope.header.at("Header.Source.Dec")(0), "Source Dec (radians)");
    phdu_base.add_key("MEAN_EL", RAD_TO_DEG*telescope.data.at("TelElAct").mean(), "Mean Elevation (deg)");
    phdu_base.add_key("MEAN_AZ", RAD_TO_DEG*telescope.data.at("TelAzAct").mean(), "Mean Azimuth (deg)");
    phdu_base.add_key("MEAN_PA", RAD_TO_DEG*telescope.data.at("ActParAng").mean(), "Mean Parallactic angle (deg)");

    phdu_base.add_key("CONFIG.VERBOSE", verbose, "Reduced in verbose mode");
    phdu_base.add_key("CONFIG.POLARIZED", run_polarization, "Polarized Obs");
    phdu_base.add_key("CONFIG.DESPIKED", run_despike, "Despiked");
    phdu_base.add_key("CONFIG.TODFILTERED", run_tod_filter, "TOD Filtered");
    phdu_base.add_key("CONFIG.DOWNSAMPLED", run_downsample, "Downsampled");
    phdu_base.add_key("CONFIG.CALIBRATED", run_flux_calib, "Calibrated");
    phdu_base.add_key("CONFIG.EXTINCTION", run_extinction, "Extinction corrected");
    phdu_base.add_key("CONFIG.CLEANED", run_pca_clean, "Cleaned");
    phdu_base.add_key("CONFIG.RTCTODOUT", run_tod_output_rtc, "RTC Output");
    phdu_base.add_key("CONFIG.PTCTODOUT", run_tod_output_ptc, "PTC Output");
    phdu_base.add_key("CONFIG.MAPMAKING", run_mapmaking, "Mapmaking");
    phdu_base.add_key("CONFIG.NOISEMAPS", run_noise_maps, "Noise Maps");
    phdu_base.add_key("CONFIG.COADDED", run_map_coadd, "Coadd");
    phdu_base.add_key("CONFIG.MAPFILTER", run_map_filter, "Map filter");
    phdu_base.add_key("CONFIG.FRUITLOOPED", run_fruit_loops, "Fruit looped");

    for (auto const& [key, val] : telescope.header) {
        phdu_base.add_key("HEADER." + key, val(0), key);
    }

    return phdu_base;
}

void ObsContainer::create_signal_hdu() {
    FitsHeader signal_base;
    phdu_base.add_key(("UNIT", mb->sig_unit, "Unit of map");
}

void ObsContainer::write_obs() {
    WCS wcs = obs_maps.wcs;

    // setup wcs
    wcs.cdelt[0] = obs_maps.pix_size_radians;
    wcs.cdelt[1] = obs_maps.pix_size_radians;
    wcs.crpix[0] = (obs_maps.n_cols - 1) / 2.0;
    wcs.crpix[1] = (obs_maps.n_rows - 1) / 2.0;
    wcs.naxis[0] = obs_maps.n_cols;
    wcs.naxis[1] = obs_maps.n_rows;
    wcs.epoch = telescope.header["Source.Epoch"](0);

    if (telescope.pixel_axes == "radec") {
        wcs.ctype[0] = "RA---TAN";
        wcs.ctype[1] = "DEC--TAN";
        wcs.crval[0] = telescope.header["Source.Ra"](0) * RAD_TO_DEG;
        wcs.crval[1] = telescope.header["Source.Dec"](0) * RAD_TO_DEG;
        wcs.cdelt[0] *= RAD_TO_DEG;
        wcs.cdelt[1] *= RAD_TO_DEG;
        wcs.cunit[0] = "deg";
        wcs.cunit[1] = "deg";
    } else if (telescope.pixel_axes == "altaz") {
        wcs.ctype[0] = "AZOFFSET";
        wcs.ctype[1] = "ELOFFSET";
        wcs.crval[0] = 0.0;
        wcs.crval[1] = 0.0;
        wcs.cdelt[0] *= RAD_TO_ASEC;
        wcs.cdelt[1] *= RAD_TO_ASEC;
        wcs.cunit[0] = "arcsec";
        wcs.cunit[1] = "arcsec";
    } else if (telescope.pixel_axes == "galactic") {
        wcs.ctype[0] = "GLON-TAN";
        wcs.ctype[1] = "GLAT-TAN";
        wcs.crval[0] = telescope.header["Source.L"](0) * RAD_TO_DEG;
        wcs.crval[1] = telescope.header["Source.B"](0) * RAD_TO_DEG;
        wcs.cdelt[0] *= RAD_TO_DEG;
        wcs.cdelt[1] *= RAD_TO_DEG;
        wcs.cunit[0] = "deg";
        wcs.cunit[1] = "deg";
    }

    create_phdu();

    // output obs maps
    for (const auto& array: toltec.apt.arrays) {
        auto filename = toltec.create_filename(reduction_directory + obsnum + "/raw/", "toltec", "", "raw",
                                               redu_type, array_index_to_name[array], obsnum, telescope.sim_obs);
        fitsIO<FitsMode::WriteFits, CCfits::ExtHDU*> fits_io(filename);

        for (const auto& [key, val] : phdu.headers) {
                phdu.add_key_to_header(fits_io.pfits->pHDU(), key, val.first, val.second);
        }

        for (auto& [set_key, map_set] : obs_maps.map_sets) {
            fits_io.add_hdu("signal_I", map_set.signal.i, wcs);
            fits_io.add_hdu("weight_I", map_set.weight.i, wcs);
            fits_io.add_hdu("sig2noise_I", map_set.signal.i.cwiseProduct(map_set.weight.i.cwiseSqrt()), wcs);

            // write kernel map if present
            if (map_set.kernel.has_value()) {
                fits_io.add_hdu("kernel_I", map_set.kernel->i, wcs);
            }

            // write signal, weight, and s/n q map
            if (map_set.signal.q.has_value() && map_set.weight.q.has_value()) {
                fits_io.add_hdu("signal_Q", *map_set.signal.q, wcs);
                fits_io.add_hdu("weight_Q", *map_set.weight.q, wcs);
                fits_io.add_hdu("sig2noise_Q", (*map_set.signal.q).cwiseProduct((*map_set.weight.q).cwiseSqrt()), wcs);
            }
            // write signal, weight, and s/n u map
            if (map_set.signal.u.has_value() && map_set.weight.u.has_value()) {
                fits_io.add_hdu("signal_U", *map_set.signal.u, wcs);
                fits_io.add_hdu("weight_U", *map_set.weight.u, wcs);
                fits_io.add_hdu("sig2noise_U", (*map_set.signal.u).cwiseProduct((*map_set.weight.u).cwiseSqrt()), wcs);
            }

            // write coverage map if present
            if (map_set.coverage.has_value()) {
                fits_io.add_hdu("coverage_I", map_set.coverage->i, wcs);
            }
            // write noise maps if present
            if (map_set.noise.has_value()) {
                for (size_t i = 0; i < map_set.noise->size(); ++i) {
                    fits_io.add_hdu("noise_I_" + std::to_string(i), map_set.noise->at(i).i, wcs);
                }
            }
        }
    }
}
