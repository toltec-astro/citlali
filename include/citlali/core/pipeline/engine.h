#pragma once

#include <vector>
#include <string>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/SpecialFunctions>

#include <gsl/gsl_rng.h>

#include <tula/config/core.h>
#include <tula/config/flatconfig.h>
#include <tula/config/yamlconfig.h>

#include <citlali/core/utils/ecsv.h>

#include <citlali/core/utils/config.h>
#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/utils.h>
#include <citlali/core/utils/time.h>
#include <citlali/core/utils/fits.h>

#include <citlali/core/pipeline/pipeline.h>
#include <citlali/core/pipeline/telescope.h>
//#include <citlali/core/pipeline/obs.h>
#include <citlali/core/pipeline/apt.h>
#include <citlali/core/pipeline/ppt.h>
#include <citlali/core/pipeline/toltec.h>
#include <citlali/core/mapmaking/map.h>

#include <citlali/core/timestream/timestream.h>
#include <citlali/core/timestream/flagging.h>
#include <citlali/core/timestream/rtc/kernel.h>
#include <citlali/core/timestream/rtc/despike.h>
#include <citlali/core/timestream/rtc/filter.h>
#include <citlali/core/timestream/rtc/downsample.h>
#include <citlali/core/timestream/rtc/flux_calib.h>
#include <citlali/core/timestream/rtc/extinction.h>

#include <citlali/core/timestream/ptc/demodulate.h>
#include <citlali/core/timestream/ptc/weights.h>
#include <citlali/core/timestream/ptc/pca_clean.h>
#include <citlali/core/timestream/ptc/source.h>
//#include <citlali/core/timestream/ptc/fruit_loops.h>
//#include <citlali/core/timestream/ptc/sensitivity.h>

//#include <citlali/core/timestream/output.h>

#include <citlali/core/mapmaking/naive_mm.h>
#include <citlali/core/mapmaking/jinc_mm.h>
#include <citlali/core/mapmaking/normalize.h>
#include <citlali/core/mapmaking/fit.h>
/*#include <citlali/core/mapmaking/psd.h>
#include <citlali/core/mapmaking/hist.h>
//#include <citlali/core/mapmaking/filter.h>
#include <citlali/core/mapmaking/coadd.h>
*/

using namespace citlali::config::options;

class Engine {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // current obsnum
    std::string obsnum;

    // classes
    Telescope telescope;
    Instrument toltec;
    DataMapsContainer obs_maps, coadd_maps;
    NoiseMapsContainer noise_maps;
    PointingPropertyTable ppt;

    // main pipelines
    Pipeline<TCData> rtc_pipeline, ptc_pipeline;
    Pipeline<DataMapsContainer> obs_map_pipeline, coadd_map_pipeline, map_process_pipeline;

    // vectors to hold initial and final indices for each network and hwpr (last index if hwpr.run_hwpr)
    Eigen::VectorXI init_indices, final_indices;

    // current directory (output path or output path + reduNN)
    std::string reduction_directory;

    template <typename ConfigType>
    void get_configs(ConfigType &config) {
        // get runtime control config params
        get_runtime_configs(config);

        // get reduction control configs
        get_reduction_configs(config);

        // get telescope configs
        telescope.get_configs(config);

        if (redu_type == "beammap") {
            // get beammap control configs
            get_beammap_configs(config);
        }

        if (run_fruit_loops) {
            // get fruit loops configs
            get_fruit_configs(config);
        }
    }

    template <typename ConfigType>
    void build_pipelines(ConfigType &config) {
        // add Kernel
        if (run_kernel) {
            rtc_pipeline.add_component("Kernel", std::make_shared<Kernel<TCData>>(
                toltec, telescope, config));
        }
        // add Despike
        if (run_despike) {
            rtc_pipeline.add_component("Despike", std::make_shared<Despike<TCData>>(
                toltec, telescope, config));
        }
        // add Filter
        if (run_tod_filter) {
            rtc_pipeline.add_component("Filter", std::make_shared<Filter<TCData>>(
                toltec, telescope, config));
        }
        // add Downsample
        if (run_downsample) {
            rtc_pipeline.add_component("Downsample", std::make_shared<Downsample<TCData>>(
                toltec, telescope, config));
        }
        // add FluxCalib
        if (run_flux_calib) {
            rtc_pipeline.add_component("FluxCalib", std::make_shared<FluxCalib<TCData>>(
                toltec, telescope));
        }
        // add Extinction
        if (run_extinction) {
            rtc_pipeline.add_component("Extinction", std::make_shared<Extinction<TCData>>(
                toltec, telescope));
        }
        // add TodFlagging to rtc pipeline (checks flags itself)
        rtc_pipeline.add_component("TodFlagging", std::make_shared<TodFlagging<TCData>>(
            "raw_time_chunk", toltec, telescope, config));

        // add Source (Beammap)
        if (redu_type == "beammap") {
            ptc_pipeline.add_component("Source", std::make_shared<Source<TCData>>(
                SourceType::Subtract, toltec, telescope, obs_maps, config));
        }
        //else if (run_fruit_loops) {
        //    ptc_pipeline.add_component("FruitLoops", std::make_shared<FruitLoops<TCData>>(
        //        FruitType::Subtract, toltec, telescope, config));
        //}

        // add PCA Clean
        if (run_pca_clean) {
            ptc_pipeline.add_component("PcaClean", std::make_shared<PcaClean<TCData>>(
                toltec, telescope, config));
        }
        // add Source (Beammap)
        if (redu_type == "beammap") {
            ptc_pipeline.add_component("Source", std::make_shared<Source<TCData>>(
                SourceType::Add, toltec, telescope, obs_maps, config));
        }
        /*else if (run_fruit_loops) {
            // Make noise maps after source subtraction and cleaning
            if (run_mapmaking) {
                if (map_method == "naive") {
                    ptc_pipeline.add_component("NaiveMapMaker", std::make_shared<NaiveMapmaker<TCData>>(
                        MapType::Noise, toltec, telescope, obs_maps, coadd_maps, config));
                }
                else if (map_method == "jinc") {
                    ptc_pipeline.add_component("JincMapmaker", std::make_shared<JincMapmaker<TCData>>(
                        MapType::Noise, toltec, telescope, obs_maps, coadd_maps, config));
                }
            }
            ptc_pipeline.add_component("FruitLoops", std::make_shared<FruitLoops<TCData>>(
                FruitType::Add, toltec, telescope, config));
        }*/
        // add TodFlagging to ptc pipeline (checks flags itself)
        ptc_pipeline.add_component("TodFlagging", std::make_shared<TodFlagging<TCData>>(
            "processed_time_chunk", toltec, telescope, config));

        // add Demodulate
        if (run_polarization) {
            ptc_pipeline.add_component("Demodulate", std::make_shared<Demodulate<TCData>>(
                toltec, telescope, config));
        }
        // add Weights
        ptc_pipeline.add_component("Weights", std::make_shared<Weights<TCData>>(
            toltec, telescope, config));

        // add Mapmaking Components
        if (run_mapmaking) {
            MapMode map_type = MapMode::Both;

            if (run_fruit_loops) {
                map_type = MapMode::Obs;
            }
            if (map_method == "naive") {
                ptc_pipeline.add_component("NaiveMapMaker", std::make_shared<NaiveMapmaker<TCData>>(
                    map_type, toltec, telescope, obs_maps, coadd_maps, noise_maps, config));
            }
            else if (map_method == "jinc") {
                ptc_pipeline.add_component("JincMapmaker", std::make_shared<JincMapmaker<TCData>>(
                    map_type, toltec, telescope, obs_maps, coadd_maps, noise_maps, config));
            }
            // add Normalize
            obs_map_pipeline.add_component("Normalize", std::make_shared<Normalize<DataMapsContainer>>(
                    toltec, telescope, config));

            // add Fit (for non-science reduction types)
            if (redu_type != "science") {
                obs_map_pipeline.add_component("Fit", std::make_shared<Fit<DataMapsContainer>>(
                    toltec, telescope, config));
            }
            // add Psd (for non-beammap reduction types)
            /*if (redu_type != "beammap") {
                obs_map_pipeline.add_component("MapPsd", std::make_shared<MapPsd<DataMapsContainer>>(
                    toltec, telescope, config));
            }
            if (redu_type != "beammap") {
                obs_map_pipeline.add_component("MapHist", std::make_shared<MapHist<DataMapsContainer>>(
                    toltec, telescope, config));
            }
            // add Coadd
            if (run_map_coadd) {
                obs_map_pipeline.add_component("Coadd", std::make_shared<Coadd<DataMapsContainer>>(
                    coadd_maps, toltec, telescope, config));
            }*/
        }
    }

    FitsHeader create_phdu() {
        // create primary header
        FitsHeader phdu_base;
        phdu_base.add_key("OBSNUM", obsnum, "Observation number");
        phdu_base.add_key("SOURCE", telescope.source_name, "Source name");
        phdu_base.add_key("INSTRUME", "TolTEC", "Instrument");
        phdu_base.add_key("TELESCOP", "LMT", "Telescope");
        phdu_base.add_key("HWPR", toltec.hwpr.run_hwpr, "HWPR installed");
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
        phdu_base.add_key("SRC_RA", telescope.header.at("Source.Ra")(0), "Source RA (radians)");
        phdu_base.add_key("SRC_DEC", telescope.header.at("Source.Dec")(0), "Source Dec (radians)");
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

    template <typename Derived>
    FitsHeader create_signal_hdu(Eigen::DenseBase<Derived> &params, Eigen::DenseBase<Derived> &errors) {
        FitsHeader signal_hdu;
        signal_hdu.add_key("UNIT", units, "Unit of map");

        for (int i = 0; i < params.size(); ++i) {
            signal_hdu.add_key("p" + std::to_string(i), params(i), "test");
            signal_hdu.add_key("e" + std::to_string(i), errors(i), "test");
        }

        return signal_hdu;
    }

    void write_obs() {
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

        // output obs maps
        for (const auto& array_index: toltec.apt.arrays) {
            auto filename = toltec.create_filename(reduction_directory + obsnum + "/raw/", "toltec", "", "raw",
                                                   redu_type, toltec.array_index_to_name[array_index], obsnum, telescope.sim_obs);
            logger->info("outputting maps to {}", filename + ".fits");
            fitsIO<FitsMode::WriteFits, CCfits::ExtHDU*> fits_io(filename + ".fits");

            auto phdu = create_phdu();
            phdu.write_to_fits(fits_io.pfits->pHDU());

            std::string grp_name = "";

            for (auto& [key, key_map] : obs_maps[array_index]) {

                if (map_grouping != "array") {
                    grp_name = "_" + map_grouping + "_" + std::to_string(key);
                }

                fits_io.add_hdu("signal_I" + grp_name, key_map.signal.i, wcs);
                auto signal_hdu = create_signal_hdu(obs_maps.fits[array_index][key].params.i, obs_maps.fits[array_index][key].errors.i);
                signal_hdu.write_to_fits(*fits_io.hdus.back());

                fits_io.add_hdu("weight_I" + grp_name, key_map.weight.i, wcs);
                fits_io.add_hdu("sig2noise_I" + grp_name, key_map.signal.i.cwiseProduct(key_map.weight.i.cwiseSqrt()), wcs);

                // write kernel map if present
                if (key_map.kernel.i.size() > 0) {
                    fits_io.add_hdu("kernel_I" + grp_name, key_map.kernel.i, wcs);
                }

                // write signal, weight, and s/n q map
                if (key_map.signal.q.size() > 0 && key_map.weight.q.size() > 0) {
                    fits_io.add_hdu("signal_Q" + grp_name, key_map.signal.q, wcs);
                    auto signal_hdu = create_signal_hdu(obs_maps.fits[array_index][key].params.q, obs_maps.fits[array_index][key].errors.q);
                    signal_hdu.write_to_fits(*fits_io.hdus.back());

                    fits_io.add_hdu("weight_Q" + grp_name, key_map.weight.q, wcs);
                    fits_io.add_hdu("sig2noise_Q" + grp_name, (key_map.signal.q).cwiseProduct(key_map.weight.q.cwiseSqrt()), wcs);
                }
                // write signal, weight, and s/n u map
                if (key_map.signal.u.size() > 0 && key_map.weight.u.size() > 0) {
                    auto signal_hdu = create_signal_hdu(obs_maps.fits[array_index][key].params.u, obs_maps.fits[array_index][key].errors.u);
                    signal_hdu.write_to_fits(*fits_io.hdus.back());
                    fits_io.add_hdu("signal_U" + grp_name, key_map.signal.u, wcs);

                    fits_io.add_hdu("weight_U" + grp_name, key_map.weight.u, wcs);
                    fits_io.add_hdu("sig2noise_U" + grp_name, (key_map.signal.u).cwiseProduct(key_map.weight.u.cwiseSqrt()), wcs);
                }

                // write coverage map if present
                if (key_map.coverage.i.size() > 0) {
                    fits_io.add_hdu("coverage_I" + grp_name, key_map.coverage.i, wcs);
                }
            }
        }
    }

    // Process input through the pipeline
    template <class KidsProc, class RawObs>
    void run(KidsProc& kidsproc, RawObs& rawobs) {

        // run initialize on all pipelines
        rtc_pipeline.init();
        ptc_pipeline.init();
        obs_map_pipeline.init();
        coadd_map_pipeline.init();

        // vector to store time chunks for beammap mode
        std::vector<TCData> tc_vector;
        // vectors for grppi maps over tod chunks
        std::vector<int> tc_in, tc_out;
        // vectors for grppi maps over maps
        std::vector<int> map_in, map_out;

        // resize tod, map, and convergence vector if in beammap mode
        if (redu_type == "beammap") {
            tc_vector.resize(telescope.n_chunks);

            // for timestream-based grppi maps
            tc_in.resize(tc_vector.size());
            std::iota(tc_in.begin(), tc_in.end(), 0);
            tc_out.resize(tc_vector.size());

            // for map-based grppi maps
            map_in.resize(obs_maps.signal_map.size());
            std::iota(map_in.begin(), map_in.end(), 0);
            map_out.resize(obs_maps.signal_map.size());

            converged.resize(obs_maps.signal_map.size());
            converged.setConstant(false);
            convergence_iter.setZero(obs_maps.signal_map.size());
        }

        // grppi pipeline that reads files, runs kidsproc, and returns each chunk sequentially
        grppi::pipeline(tula::grppi_utils::dyn_ex(exec_mode), [&]() -> std::optional<TCData> {
                static int current_chunk = 0;
                while (current_chunk < telescope.n_chunks) {
                    logger->info("beginning reduction of chunk {}/{}", current_chunk, telescope.n_chunks);

                    // declare time chunk data class
                    TCData tcdata;
                    tcdata.chunk = current_chunk;
                    tcdata.chunk_indices = telescope.chunk_indices.col(current_chunk);

                    // vector to store kids data
                    std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>> chunk_rawobs;

                    // get kids data
                    chunk_rawobs = kidsproc.load_rawobs(rawobs, current_chunk, telescope.chunk_indices, init_indices, final_indices);

                    // current length of outer chunks
                    Eigen::Index chunk_size = tcdata.chunk_indices(3) - tcdata.chunk_indices(2) + 1;

                    // get raw tod from files after kidsproc
                    tcdata.signal = kidsproc.populate_rtc(chunk_rawobs, chunk_size, toltec.apt.n_dets, tod_type);

                    // clear input vector
                    std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>>().swap(chunk_rawobs);

                    // increment chunk number
                    current_chunk++;
                    return tcdata;
                }
                current_chunk = 0;
                return {};
            },
            grppi::farm(citlali::utils::threads::n_chunk_threads, [&](TCData& tcdata) {
                // set flags
                tcdata.flag.setZero(tcdata.n_pts(), tcdata.n_dets());
                // set chunk apt flags to initial apt flags (cleared at mapmaking if no coverage found)
                tcdata.apt_flag = toltec.apt["flag"].data;
                // set sampling rate (updated after downsampling)
                tcdata.data_fs_hz = toltec.data_fs_hz;
                // hold fcf for extinction correction
                tcdata.fcf.setOnes(tcdata.n_dets());

                // current length of outer chunks
                Eigen::Index chunk_size = tcdata.chunk_indices(3) - tcdata.chunk_indices(2) + 1;

                // copy chunk's telescope data vectors
                for (const auto& [key,value]: telescope.data) {
                    tcdata.tel_data[key] = value.segment(tcdata.chunk_indices(2), chunk_size);
                }

                // copy chunk's hwpr angle
                if (toltec.hwpr.hwpr_theta.size() > 0) {
                    tcdata.hwpr_theta = toltec.hwpr.hwpr_theta.segment(tcdata.chunk_indices(2), chunk_size);
                }

                // run the raw time chunk pipeline
                rtc_pipeline.process(tcdata);
                logger->info("done with rtc reduction of chunk {}", tcdata.chunk);

                // rtc tod output
                if (run_tod_output_rtc) {
                    logger->info("writing rtc chunk {}", tcdata.chunk);
                    //rtc_output.process(tcdata);
                }

                // accumulate time chunks for beammap mode
                if (redu_type == "beammap") {
                    tc_vector[tcdata.chunk] = std::move(tcdata);
                } else {
                    // run the processed time chunk pipeline
                    ptc_pipeline.process(tcdata);

                    // ptc tod output
                    if (run_tod_output_ptc) {
                        logger->info("writing ptc chunk {}", tcdata.chunk);
                        //ptc_output.process(tcdata);
                    }

                    logger->info("done with ptc reduction of chunk {}", tcdata.chunk);

                    // doesn't like being in the destructor
                    tcdata.gsl_free();
                }
            }));

        // normalize maps if not in beammap mode and fit maps if in pointing mode
        if (redu_type != "beammap") {
            logger->info("running obs map pipeline");
            obs_map_pipeline.process(obs_maps);

            if (redu_type == "pointing") {

                for (const auto& column : ppt.column_order) {
                    ppt[column].data.resize(obs_maps.params_map.size());
                }

                // populate apt with fits and errors
                for (int i = 0; i < obs_maps.params_map.size(); ++i) {
                    ppt["array"].data(i) = obs_maps.arrays[i];
                    ppt["amp"].data(i) = obs_maps.params_map[i](0);
                    ppt["x_t"].data(i) = obs_maps.params_map[i](1);
                    ppt["y_t"].data(i) = obs_maps.params_map[i](2);
                    ppt["a_fwhm"].data(i) = obs_maps.params_map[i](3);
                    ppt["b_fwhm"].data(i) = obs_maps.params_map[i](4);
                    ppt["angle"].data(i) = obs_maps.params_map[i](5);

                    ppt["amp_err"].data(i) = obs_maps.errors_map[i](0);
                    ppt["x_t_err"].data(i) = obs_maps.errors_map[i](1);
                    ppt["y_t_err"].data(i) = obs_maps.errors_map[i](2);
                    ppt["a_fwhm_err"].data(i) = obs_maps.errors_map[i](3);
                    ppt["b_fwhm_err"].data(i) = obs_maps.errors_map[i](4);
                    ppt["angle_err"].data(i) = obs_maps.errors_map[i](5);
                }

                ppt["sig2noise"].data = ppt["amp"].data.array() / ppt["amp_err"].data.array();

                // write apt
                auto filename = toltec.create_filename(reduction_directory + obsnum + "/", "ppt", "", "raw",
                                                       redu_type, "", obsnum, telescope.sim_obs);

                logger->info("outputting ppt to {}", filename + ".ecsv");
                ppt.write(filename);
            }

        } else {
            // current beammap iteration
            int bmp_iter = 0;

            // boolean control for beammap loop
            bool keep_going = true;

            // don't need a copy of uncertainties
            std::vector<Eigen::VectorXd> params_copy;

            // start the iterative loop
            while (keep_going) {
                // copy time chunk vector so we don't overwrite the original
                std::vector<TCData> tc_vector_copy = tc_vector;

                // copy params
                params_copy.clear();
                for (const auto& map : obs_maps.params_map) {
                    params_copy.emplace_back(map);
                }

                // re-initialize maps to zero
                for (const auto& [key, obs_map] : obs_maps.maps) {
                    Eigen::VectorXi lower_keys(obs_map.size());
                    int i = 0;
                    for (const auto& [lower_key, obs_key_map] : obs_map) {
                        lower_keys(i) = lower_key;
                        i++;
                    }
                    obs_maps.init_array(key, lower_keys);
                }

                obs_maps.build_vectors();

                // run the processed time chunk pipeline
                grppi::map(tula::grppi_utils::dyn_ex(exec_mode), tc_in, tc_out, [&](int i) {
                    ptc_pipeline.process(tc_vector_copy[i]);
                    return 0;
                });

                // ptc tod output
                if (run_tod_output_ptc) {
                    for (auto &chunk: tc_vector_copy) {
                        //ptc_output.process(chunk);
                        chunk.gsl_free();
                    }
                }

                logger->info("running obs map pipeline");
                obs_map_pipeline.process(obs_maps);

                // check convergence
                if (bmp_iter < bmp_iter_max - 1) {
                    if (converged.array().all()) {

                        logger->info("all maps have converged");
                        keep_going = false;

                    } else if (bmp_iter > 0 && bmp_iter_tolerance > 0) {

                        logger->info("checking convergence");
                         grppi::map(tula::grppi_utils::dyn_ex(exec_mode), map_in, map_out, [&](int i) {
                            if (!converged(i)) {

                                Eigen::VectorXd abs_diff = (params_copy[i].array() -
                                                         obs_maps.params_map[i].array()).abs() / params_copy[i].array().abs();

                                // if a variable is constant, make sure no nans are present
                                (abs_diff.array()).isNaN().select(0, abs_diff.array());

                                if ((abs_diff.array() <= bmp_iter_tolerance).all()) {
                                    converged(i) = true;
                                    convergence_iter(i) = bmp_iter;
                                }
                            }
                            return 0;
                        });

                        logger->info("iter {}: {} maps converged", bmp_iter, (converged.array() == true).count());

                         // check again to see if all maps converged
                         if ((converged.array() == true).all()) {
                             logger->info("all maps converged");
                             keep_going = false;
                         }
                    } else if (bmp_iter_tolerance > 0) {
                        logger->info("done with iteration {}", bmp_iter);
                        bmp_iter++;
                    } else {
                        logger->info("bypassing convergence check");
                    }
                } else {
                    logger->info("max iteration reached");
                    keep_going = false;
                }
            }

            // populate apt with fits and errors
            for (int i = 0; i < obs_maps.params_map.size(); ++i) {
                toltec.apt["amp"].data(i) = obs_maps.params_map[i](0);
                toltec.apt["x_t_raw"].data(i) = obs_maps.params_map[i](1);
                toltec.apt["y_t_raw"].data(i) = obs_maps.params_map[i](2);
                toltec.apt["a_fwhm"].data(i) = obs_maps.params_map[i](3);
                toltec.apt["b_fwhm"].data(i) = obs_maps.params_map[i](4);
                toltec.apt["angle"].data(i) = obs_maps.params_map[i](5);

                toltec.apt["amp_err"].data(i) = obs_maps.errors_map[i](0);
                toltec.apt["x_t_err"].data(i) = obs_maps.errors_map[i](1);
                toltec.apt["y_t_err"].data(i) = obs_maps.errors_map[i](2);
                toltec.apt["a_fwhm_err"].data(i) = obs_maps.errors_map[i](3);
                toltec.apt["b_fwhm_err"].data(i) = obs_maps.errors_map[i](4);
                toltec.apt["angle_err"].data(i) = obs_maps.errors_map[i](5);
            }

            toltec.apt["sig2noise"].data = toltec.apt["amp"].data.array() / toltec.apt["amp_err"].data.array();
            toltec.apt["converge_iter"].data = convergence_iter.cast<double>();

            // coordinates of reference detector
            std::pair<double, double> reference_coord;
            // find reference detector coordinates
            if (bmp_subtract_reference_det) {
                reference_coord = toltec.apt.find_reference(bmp_reference_det);
            } else {
                reference_coord = std::make_pair(0.0, 0.0);
            }

            // angle to rotate apt x_t_raw and y_t_raw by
            Eigen::VectorXd theta;
            // use mean telescope boresight elevation
            if (bmp_derotate_apt) {
                theta = Eigen::VectorXd::Constant(telescope.data.at("TelElAct").size(), -telescope.data.at("TelElAct").mean());
            } else {
                theta = Eigen::VectorXd::Zero(telescope.data.at("TelElAct").size());
            }
            // subtract reference det and derotate apt
            toltec.apt.rotate(theta, reference_coord);

            // flag detectors by fwhm and sig2noise
            for (int i = 0; i < toltec.apt.n_dets; ++i) {
                int array_index = toltec.apt["array"].data(i);
                if (toltec.apt["a_fwhm"].data(i) < bmp_fwhm_lower[array_index] || toltec.apt["a_fwhm"].data(i) > bmp_fwhm_upper[array_index]) {
                    toltec.apt["flag"].data(i) = true;
                }
                if (toltec.apt["b_fwhm"].data(i) < bmp_fwhm_lower[array_index] || toltec.apt["b_fwhm"].data(i) > bmp_fwhm_upper[array_index]) {
                    toltec.apt["flag"].data(i) = true;
                }
                if (toltec.apt["amp"].data(i)/toltec.apt["amp_err"].data(i) < bmp_sig2noise_lower[array_index]
                    || toltec.apt["amp"].data(i)/toltec.apt["amp_err"].data(i) > bmp_sig2noise_upper[array_index]) {
                    toltec.apt["flag"].data(i) = true;
                }
            }

            // nw median sens
            auto nw_median_sens = toltec.apt.calc_median("sens", "nw");
            // flag sens after other flagging since it depends on multiple dets
            for (int i = 0; i < toltec.apt.n_dets; ++i) {
                int nw_index = toltec.apt["nw"].data(i);
                if (toltec.apt["sens"].data(i) < bmp_sens_factors[0]*nw_median_sens[nw_index]
                    || toltec.apt["sens"].data(i) > bmp_sens_factors[1]*nw_median_sens[nw_index]) {
                    toltec.apt["flag"].data(i) = true;
                }
            }

            // array median positions
            auto array_median_x_t = toltec.apt.calc_median("x_t", "array");
            auto array_median_y_t = toltec.apt.calc_median("y_t", "array");
            // flag footprint
            for (int i = 0; i < toltec.apt.n_dets; ++i) {
                int array_index = toltec.apt["array"].data(i);
                double dist = sqrt(pow(toltec.apt["x_t"].data(i) - array_median_x_t[array_index],2)
                                   + pow(toltec.apt["y_t"].data(i) - array_median_y_t[array_index],2));
                if (dist > bmp_dist_max_arcsec[array_index]) {
                    toltec.apt["flag"].data(i) = true;
                }
            }

            // calculate flux scale
            int i = 0;
            for (const auto& pair : toltec.apt.array_indices) {
                auto flxscale = toltec.apt["flxscale"].data(Eigen::seq(pair.first, pair.second - 1));
                auto amp = toltec.apt["amp"].data(Eigen::seq(pair.first, pair.second - 1));
                auto sens = toltec.apt["sens"].data(Eigen::seq(pair.first, pair.second - 1));
                // Calculate the flux scale, setting it to 0 if flagged
                flxscale = (toltec.apt["flags"]
                                .data(Eigen::seq(pair.first, pair.second - 1))
                                .array() == false)
                               .select(0.0, amp / bmp_flux_mJy_beam[toltec.array_index_to_name[toltec.apt.arrays(i)]]);
                sens = (toltec.apt["flags"]
                                .data(Eigen::seq(pair.first, pair.second - 1))
                                .array() == false)
                               .select(0.0, sens * flxscale);
                i++;
            }

            // clear vector of time chunks
            std::vector<TCData>().swap(tc_vector);

            // write apt
            auto filename = toltec.create_filename(reduction_directory + obsnum + "/", "apt", "", "raw",
                                                   redu_type, "", obsnum, telescope.sim_obs);

            logger->info("outputting apt to {}", filename + ".ecsv");
            toltec.apt.write(filename);
        }

        // write maps
        write_obs();
    }

    void process_coadd() {
        coadd_map_pipeline.process(coadd_maps);
    }
};
