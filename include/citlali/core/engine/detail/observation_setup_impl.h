#pragma once

// Engine member-function implementations split from engine.h.
// Include this only after Engine has been declared.

#include <citlali/core/config/mapmaking_config.h>
#include <citlali/core/pipeline/output_policy.h>

void Engine::obsnum_setup() {
    if (rtcproc.run_extinction) {
        // get atm model
        rtcproc.calibration.setup(telescope.tau_225_GHz);

        logger->info("using {} model for extinction correction",rtcproc.calibration.extinction_model);

        // check tau (may be unnecessary now)
        if (!telescope.sim_obs) {
            Eigen::VectorXd tau_el(1);
            // get mean elevation
            tau_el << telescope.tel_data["TelElAct"].mean();
            // get tau at mean elevation for each band
            auto tau_freq = rtcproc.calibration.calc_tau(tau_el, telescope.tau_225_GHz);
            // loop through and make sure average tau is not negative (implies wrong model)
            for (auto const& [key, val] : tau_freq) {
                if (val[0] < 0) {
                    logger->error("calculated mean {} tau {} < 0",toltec_io.array_name_map[key], val[0]);
                    std::exit(EXIT_FAILURE);
                }
            }
        }
    }
    else {
        rtcproc.calibration.extinction_model = "N/A";
    }

    // make sure there are matched fg's in apt if reducing in polarized mode
    if (rtcproc.run_polarization) {
        if ((calib.apt["fg"].array()==-1).all()) {
            logger->error("no matched freq groups.  cannot run in polarized mode");
            std::exit(EXIT_FAILURE);
        }
    }

    // setup kernel
    if (rtcproc.run_kernel) {
        rtcproc.kernel.setup(n_maps);
    }

    // set despiker sample rate
    rtcproc.despiker.fsmp = telescope.fsmp;
    // set processed timestream sample rate for optional adaptive PTC mode selection
    ptcproc.cleaner.sample_rate_Hz = telescope.d_fsmp;

    // if filter is requested, make it here
    if (rtcproc.run_tod_filter) {
        rtcproc.filter.make_filter(telescope.fsmp);
        if (rtcproc.run_tod_notch) {
            rtcproc.filter.make_notch_filter(telescope.fsmp);
        }
    }
    if (rtcproc.run_tod_iir_highpass) {
        const double nyquist_Hz = telescope.fsmp / 2.0;
        if (rtcproc.filter.iir_highpass_freq_Hz >= nyquist_Hz) {
            logger->error("timestream.raw_time_chunk.IIR_filter.freq_Hz ({}) must be less than Nyquist ({})",
                          rtcproc.filter.iir_highpass_freq_Hz, nyquist_Hz);
            std::exit(EXIT_FAILURE);
        }
    }

    // set map wcs crvals to source ra/dec
    if (citlali::config::is_radec_map_pixel_axes(telescope.pixel_axes)) {
        omb.wcs.crval[0] = telescope.tel_header["Header.Source.Ra"](0)*RAD_TO_DEG;
        omb.wcs.crval[1] = telescope.tel_header["Header.Source.Dec"](0)*RAD_TO_DEG;

        if (citlali::pipeline::coadd_outputs_enabled(*this)) {
            cmb.wcs.crval[0] = telescope.tel_header["Header.Source.Ra"](0)*RAD_TO_DEG;
            cmb.wcs.crval[1] = telescope.tel_header["Header.Source.Dec"](0)*RAD_TO_DEG;
        }
    }

    // set map wcs crvals to source l/b
    else if (citlali::config::is_galactic_map_pixel_axes(
                 telescope.pixel_axes)) {
        omb.wcs.crval[0] = telescope.tel_header["Header.Source.L"](0)*RAD_TO_DEG;
        omb.wcs.crval[1] = telescope.tel_header["Header.Source.B"](0)*RAD_TO_DEG;

        if (citlali::pipeline::coadd_outputs_enabled(*this)) {
            cmb.wcs.crval[0] = telescope.tel_header["Header.Source.L"](0)*RAD_TO_DEG;
            cmb.wcs.crval[1] = telescope.tel_header["Header.Source.B"](0)*RAD_TO_DEG;
        }
    }

    setup_tod_output_chunk_selection();
    // create output subdirectory if requested
    if (tod_output_subdir_name != "null") {
        fs::create_directories(obsnum_dir_name + "raw/" + tod_output_subdir_name);
    }
    // create timestream files
    if (run_tod_output) {
        // make rtc tod output file
        if (run_tod_output_rtc) {
            create_tod_files<engine_utils::toltecIO::rtc_timestream>();
        }
        // make ptc tod output file
        if (run_tod_output_ptc) {
            create_tod_files<engine_utils::toltecIO::ptc_timestream>();
        }
    }
    // don't calculate any eigenvalues
    else if (!diagnostics.write_evals) {
        ptcproc.cleaner.n_calc = 0;
    }
    create_rtcdiag_file();
    create_ptcdiag_file();

    // output basic info for obs reduction to command line
    cli_summary();

    // set up per-det stats file values
    for (const auto &stat: diagnostics.det_stats_header) {
        diagnostics.stats[stat].setZero(calib.n_dets, telescope.scan_indices.cols());
    }
    // set up per-group stats file values
    for (const auto &stat: diagnostics.grp_stats_header) {
        diagnostics.stats[stat].setZero(calib.n_arrays, telescope.scan_indices.cols());
    }
    // clear stored eigenvalues
    std::map<Eigen::Index, std::vector<std::vector<Eigen::VectorXd>>>().swap(diagnostics.evals);
}
