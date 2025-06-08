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
#include <citlali/core/utils/pointing.h>
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
#include <citlali/core/timestream/ptc/fruit_loops.h>
//#include <citlali/core/timestream/ptc/sensitivity.h>

//#include <citlali/core/timestream/output.h>

#include <citlali/core/mapmaking/naive_mm.h>
#include <citlali/core/mapmaking/jinc_mm.h>
#include <citlali/core/mapmaking/bilinear_mm.h>
#include <citlali/core/mapmaking/normalize.h>
#include <citlali/core/mapmaking/fit.h>
#include <citlali/core/mapmaking/psd.h>
//#include <citlali/core/mapmaking/hist.h>
//#include <citlali/core/mapmaking/filter.h>
#include <citlali/core/mapmaking/coadd.h>

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
    ObsMaps<MapKey> obs_maps, coadd_maps;
    ObsMaps<MapKey, std::vector<ObsMatrix<MapKey>>> noise_maps;

    // pipelines
    Pipeline<TCData> rtc_pipeline, ptc_pipeline;
    Pipeline<ObsMaps<>> obs_map_pipeline, coadd_map_pipeline;
    Pipeline<ObsMaps<MapKey, std::vector<ObsMatrix<MapKey>>>> noise_map_pipeline;

    // vectors to hold initial and final indices for each network and hwpr (last index if hwpr.run_hwpr)
    Eigen::VectorXI init_indices, final_indices;

    // current directory (output path or output path + reduNN)
    std::string reduction_directory;

    template <typename MapType, typename UniqueKeyType>
    void output_maps(MapType& maps, UniqueKeyType& unique_map_keys);

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
                SourceMode::Subtract, toltec, telescope, obs_maps, config));
        }
        else if (run_fruit_loops) {
            ptc_pipeline.add_component("FruitLoops", std::make_shared<FruitLoops<TCData>>(
                FruitMode::Subtract, toltec, telescope, config));
        }

        // add PCA Clean
        if (run_pca_clean) {
            ptc_pipeline.add_component("PcaClean", std::make_shared<PcaClean<TCData>>(
                toltec, telescope, config));
        }
        // add Weights
        ptc_pipeline.add_component("Weights", std::make_shared<Weights<TCData>>(
                toltec, telescope, config));

        // add Source (Beammap)
        if (redu_type == "beammap") {
            ptc_pipeline.add_component("Source", std::make_shared<Source<TCData>>(
                SourceMode::Add, toltec, telescope, obs_maps, config));
        }
        else if (run_fruit_loops) {
            // Make noise maps after source subtraction and cleaning
            if (run_mapmaking) {
                MapMode map_mode = MapMode::Noise;
                if (map_method == "naive") {
                    ptc_pipeline.add_component("NaiveMapMaker", std::make_shared<NaiveMapmaker<TCData>>(
                         toltec, telescope, obs_maps, coadd_maps, noise_maps, map_mode, config));
                }
                else if (map_method == "jinc") {
                    ptc_pipeline.add_component("JincMapmaker", std::make_shared<JincMapmaker<TCData>>(
                        toltec, telescope, obs_maps, coadd_maps, noise_maps, map_mode, config));
                }
                else if (map_method == "bilinear") {
                    ptc_pipeline.add_component("BilinearMapmaker", std::make_shared<BilinearMapmaker<TCData>>(
                        toltec, telescope, obs_maps, coadd_maps, noise_maps, map_mode, config));
                }
            }
            ptc_pipeline.add_component("FruitLoops", std::make_shared<FruitLoops<TCData>>(
                FruitMode::Add, toltec, telescope, config));
        }
        // add TodFlagging to ptc pipeline (checks flags itself)
        ptc_pipeline.add_component("TodFlagging", std::make_shared<TodFlagging<TCData>>(
            "processed_time_chunk", toltec, telescope, config));

        // add Demodulate
        if (run_polarization) {
            ptc_pipeline.add_component("Demodulate", std::make_shared<Demodulate<TCData>>(
                toltec, telescope, config));
        }

        // // add Mapmaking Components
        if (run_mapmaking) {
            MapMode map_mode = MapMode::Both;

            if (run_fruit_loops) {
                map_mode = MapMode::Obs;
            }
            if (map_method == "naive") {
                ptc_pipeline.add_component("NaiveMapMaker", std::make_shared<NaiveMapmaker<TCData>>(
                    toltec, telescope, obs_maps, coadd_maps, noise_maps, map_mode, config));
            }
            else if (map_method == "jinc") {
                ptc_pipeline.add_component("JincMapmaker", std::make_shared<JincMapmaker<TCData>>(
                    toltec, telescope, obs_maps, coadd_maps, noise_maps, map_mode, config));
            }
            else if (map_method == "bilinear") {
                ptc_pipeline.add_component("BilinearMapmaker", std::make_shared<BilinearMapmaker<TCData>>(
                    toltec, telescope, obs_maps, coadd_maps, noise_maps, map_mode, config));
            }
            // add Normalize
            obs_map_pipeline.add_component("Normalize", std::make_shared<Normalize<ObsMaps<>>>(
                    toltec, telescope, config));

            // add Fit (for non-science reduction types)
            if (redu_type != "science") {
                obs_map_pipeline.add_component("Fit", std::make_shared<Fit<ObsMaps<>>>(
                    toltec, telescope, config));
            }
            // // add Psd (for non-beammap reduction types)
            if (redu_type != "beammap") {
                obs_map_pipeline.add_component("MapPsd", std::make_shared<MapPsd<ObsMaps<>>>(
                    toltec, telescope, config));

                // obs_map_pipeline.add_component("MapHist", std::make_shared<MapHist<DataMapsContainer>>(
                //     toltec, telescope, config));
            }
            // add Coadd
            if (run_map_coadd) {
                obs_map_pipeline.add_component("Coadd", std::make_shared<Coadd<ObsMaps<>>>(
                    coadd_maps, toltec, telescope, config));

                coadd_map_pipeline.add_component("Normalize", std::make_shared<Normalize<ObsMaps<>>>(
                    toltec, telescope, config));
            }
        }
    }

    void output_ppt() {
        PointingPropertyTable ppt;

        for (const auto& column : ppt.column_order) {
            ppt[column].data.resize(obs_maps.params.cols());
        }

        // populate apt with fits and errors
        for (int i = 0; i < obs_maps.params.cols(); ++i) {
            ppt["array"].data(i) = obs_maps.signal[i].key.array_index;
            ppt["amp"].data(i) = obs_maps.params(0, i);
            ppt["x_t"].data(i) = obs_maps.params(1, i);
            ppt["y_t"].data(i) = obs_maps.params(2, i);
            ppt["a_fwhm"].data(i) = obs_maps.params(3, i);
            ppt["b_fwhm"].data(i) = obs_maps.params(4, i);
            ppt["angle"].data(i) = obs_maps.params(5, i);

            ppt["amp_err"].data(i) = obs_maps.errors(0, i);
            ppt["x_t_err"].data(i) = obs_maps.errors(1, i);
            ppt["y_t_err"].data(i) = obs_maps.errors(2, i);
            ppt["a_fwhm_err"].data(i) = obs_maps.errors(3, i);
            ppt["b_fwhm_err"].data(i) = obs_maps.errors(4, i);
            ppt["angle_err"].data(i) = obs_maps.errors(5, i);
        }

        ppt["sig2noise"].data = ppt["amp"].data.array() / ppt["amp_err"].data.array();

        // write ppt
        auto filename = toltec.create_filename(reduction_directory + obsnum + "/", "ppt", "", "raw",
                                               redu_type, "", obsnum, telescope.sim_obs);

        logger->info("outputting ppt to {}", filename + ".ecsv");
        ppt.write(filename);
    }

    template <class KidsProc, class RawObs>
    void run_obs(KidsProc& kidsproc, RawObs& rawobs) {
        // run initialize on all pipelines
        rtc_pipeline.init();
        ptc_pipeline.init();
        obs_map_pipeline.init();

        // vector to store time chunks for beammap mode
        std::vector<TCData> tc_vector;

        // resize tod, map, and convergence vector if in beammap mode
        if (redu_type == "beammap") {
            tc_vector.resize(telescope.n_chunks);

            converged.resize(obs_maps.signal.size());
            converged.setConstant(false);
            convergence_iter.setZero(obs_maps.signal.size());
        }

        // for timestream-based grppi maps
        auto [tc_in, tc_out] = citlali::utils::threads::get_grppi_vectors(tc_vector.size());
        // for map-based grppi maps
        auto [map_in, map_out] = citlali::utils::threads::get_grppi_vectors(obs_maps.n_maps);

        // grppi pipeline that reads files, runs kidsproc, and returns each chunk sequentially
        grppi::pipeline(tula::grppi_utils::dyn_ex(exec_mode), [&]() -> std::optional<TCData> {
                static int current_chunk = 0;
                while (current_chunk < telescope.n_chunks) {
                    logger->info("beginning reduction of chunk {}/{}", current_chunk, telescope.n_chunks);
                    // declare time chunk data class
                    TCData tcdata;
                    // set random seed to chunk number
                    tcdata.set_seed(current_chunk);
                    // set chunk
                    tcdata.chunk = current_chunk;
                    // set current chunk inner and outer indices
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
                output_ppt();
            }

        } else {
            // current beammap iteration
            int bmp_iter = 0;

            // boolean control for beammap loop
            bool keep_going = true;

            // don't need a copy of uncertainties
            Eigen::MatrixXd params_copy;

            // start the iterative loop
            while (keep_going) {
                // copy time chunk vector so we don't overwrite the original
                std::vector<TCData> tc_vector_copy = tc_vector;

                // copy params
                params_copy = obs_maps.params;

                // clear maps
                if (bmp_iter > 0) {
                    obs_maps.set_zero();
                    noise_maps.set_zero();
                }

                // run the processed time chunk pipeline
                grppi::map(tula::grppi_utils::dyn_ex(exec_mode), tc_in, tc_out, [&](int i) {
                    ptc_pipeline.process(tc_vector_copy[i]);
                    return 0;
                });

                // ptc tod output
                if (run_tod_output_ptc) {
                    for (auto &chunk: tc_vector_copy) {
                        // ptc_output.process(chunk);
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

                                Eigen::VectorXd abs_diff = (params_copy.col(i).array() -
                                                            obs_maps.params.col(i).array()).abs() / params_copy.col(i).array().abs();

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
                    } else {
                        logger->info("bypassing convergence check");
                    }
                    bmp_iter++;
                } else {
                    logger->info("max iteration reached");
                    keep_going = false;
                }
            }

            // populate apt with fits and errors
            toltec.apt["amp"].data = obs_maps.params.row(0).transpose();
            toltec.apt["x_t_raw"].data = obs_maps.params.row(1).transpose();
            toltec.apt["y_t_raw"].data = obs_maps.params.row(2).transpose();
            toltec.apt["a_fwhm"].data = obs_maps.params.row(3).transpose();
            toltec.apt["b_fwhm"].data = obs_maps.params.row(4).transpose();
            toltec.apt["angle"].data = obs_maps.params.row(5).transpose();

            toltec.apt["amp_err"].data = obs_maps.errors.row(0).transpose();
            toltec.apt["x_t_err"].data = obs_maps.errors.row(1).transpose();
            toltec.apt["y_t_err"].data = obs_maps.errors.row(2).transpose();
            toltec.apt["a_fwhm_err"].data = obs_maps.errors.row(3).transpose();
            toltec.apt["b_fwhm_err"].data = obs_maps.errors.row(4).transpose();
            toltec.apt["angle_err"].data = obs_maps.errors.row(5).transpose();

            toltec.apt["sig2noise"].data = (toltec.apt["amp_err"].data.array() > 0)
                                        .select(toltec.apt["amp"].data.array() / toltec.apt["amp_err"].data.array(), 0);

            toltec.apt["converge_iter"].data = convergence_iter.cast<double>();

            // coordinates of reference detector
            std::pair<double, double> reference_coord;
            // find reference detector coordinates
            if (bmp_subtract_reference_det) {
                reference_coord = toltec.apt.find_reference(bmp_reference_det);
            } else {
                reference_coord = std::make_pair(0.0, 0.0);
            }

            auto [in, out] = citlali::utils::threads::get_grppi_vectors(toltec.apt.n_dets);
            auto exec_mode = citlali::utils::threads::get_map_exec_mode();

            // angle to rotate apt x_t_raw and y_t_raw by
            Eigen::VectorXd theta(toltec.apt.n_dets);
            // use mean telescope boresight elevation
            if (bmp_derotate_apt) {
                logger->info("derotating apt");
                theta.resize(toltec.apt.n_dets);
                grppi::map(exec_mode, in, out, [&](int det) {
                //for (int det = 0; det < toltec.apt.n_dets; ++det) {
                    // get detector altaz
                    auto xy = calc_pointing(toltec.apt["x_t"].data(det), toltec.apt["y_t"].data(det), telescope.data, "altaz");
                    theta(det) = -(xy.second + telescope.data.at("TelElAct")).mean();
                    return 0;
                });
            } else {
                theta.setZero();
            }
            // subtract reference det and derotate apt
            logger->info("derotating apt and subtracting det at ({}, {})", reference_coord.first, reference_coord.second);
            toltec.apt.rotate(theta, reference_coord);

            // flag detectors by fwhm and sig2noise
            logger->info("flagging fwhm and sig2noise");
            for (int i = 0; i < toltec.apt.n_dets; ++i) {
                int array_index = toltec.apt["array"].data(i);
                if (toltec.apt["a_fwhm"].data(i) < bmp_fwhm_lower[array_index] || toltec.apt["a_fwhm"].data(i) > bmp_fwhm_upper[array_index]) {
                    toltec.apt["flag"].data(i) = true;
                }
                if (toltec.apt["b_fwhm"].data(i) < bmp_fwhm_lower[array_index] || toltec.apt["b_fwhm"].data(i) > bmp_fwhm_upper[array_index]) {
                    toltec.apt["flag"].data(i) = true;
                }
                if (toltec.apt["sig2noise"].data(i) < bmp_sig2noise_lower[array_index]) {
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

            // nw median sens
            // auto nw_median_sens = toltec.apt.calc_median("sens", "nw");
            // logger->info("nw_median_sens {}", nw_median_sens);
            // // flag sens after other flagging since it depends on multiple dets
            // for (int i = 0; i < toltec.apt.n_dets; ++i) {
            //     int nw_index = toltec.apt["nw"].data(i);
            //     if (toltec.apt["sens"].data(i) < bmp_sens_factors[0]*nw_median_sens[nw_index]
            //         || toltec.apt["sens"].data(i) > bmp_sens_factors[1]*nw_median_sens[nw_index]) {
            //         toltec.apt["flag"].data(i) = true;
            //     }
            // }


            // calculate flux scale
            int i = 0;
            for (const auto& pair : toltec.apt.array_indices) {
                auto flxscale = toltec.apt["flxscale"].data(Eigen::seq(pair.first, pair.second - 1));
                auto amp = toltec.apt["amp"].data(Eigen::seq(pair.first, pair.second - 1));
                auto sens = toltec.apt["sens"].data(Eigen::seq(pair.first, pair.second - 1));
                // Calculate the flux scale, setting it to 0 if flagged
                flxscale = (toltec.apt["flag"]
                                .data(Eigen::seq(pair.first, pair.second - 1))
                                .array() == true)
                               .select(0.0, amp / bmp_flux_mJy_beam[toltec.array_index_to_name[toltec.apt.arrays(i)]]);
                sens = (toltec.apt["flag"]
                                .data(Eigen::seq(pair.first, pair.second - 1))
                                .array() == true)
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
    }

    void run_coadd() {
        coadd_map_pipeline.init();
        coadd_map_pipeline.process(coadd_maps);
    }
};
