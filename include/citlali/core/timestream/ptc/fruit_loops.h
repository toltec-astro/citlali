# pragma once

#include <citlali/core/utils/beam.h>

// add or subtract gaussian source
enum FruitType {
    Add = 1,
    Subtract = -1,
};

// FruitLoops
template <typename TCDataType>
class FruitLoops : public PipelineComponent<TCDataType> {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    ObsMaps<> fruit_maps;

    bool save_all_fruit_iters;
    std::string fruit_path;
    std::string fruit_type;
    double fruit_sig2noise;
    std::vector<double> fruit_flux;
    int fruit_iters;

    bool run_noise;

    double add_subtract_factor;
    Instrument& toltec;
    Telescope& telescope;
    std::map<int, std::map<int, double>> median_rms_I, median_rms_Q, median_rms_U;

    template <typename ConfigType>
    FruitLoops(FruitType fruit_type, Instrument& toltec_ref, Telescope& telescope_ref, const ConfigType& config)
        : toltec(toltec_ref), telescope(telescope_ref) {

        add_subtract_factor = static_cast<double>(fruit_type);

        config.get(fruit_path, std::tuple{"timestream","fruit_loops","path"});
        config.get(fruit_sig2noise, std::tuple{"timestream","fruit_loops","lower_sig2noise_limit"});
        config.get(fruit_flux, std::tuple{"timestream","fruit_loops","fruit_flux"});
        config.get(fruit_iters, std::tuple{"timestream","fruit_loops","max_iters"});
    }

    void init() {
        std::vector<std::string> files, noise_files;

        for (const auto& array : toltec.arrays) {
            for (const auto& entry : fs::directory_iterator(fruit_path)) {
                if (entry.is_regular_file()) {
                    const std::string filename = entry.path().filename().string();

                    if (entry.path().extension() == ".fits") {
                        if (filename.find(array) != std::string::npos) {
                            if (filename.find("noise") == std::string::npos) {
                                files.push_back(entry.path().string());
                            }
                            else {
                                noise_files.push_back(entry.path().string());
                            }
                        }
                    }
                }
            }
        }

        if (files.size() == 0) {
            throw std::runtime_error("no map files found");
        }

        run_noise = !noise_files.empty();

        // get signal and weight
        for (int i = 0; i < files.size(); ++i) {
            int array = toltec.apt.arrays[i];

            fitsIO<FitsMode::WriteFits, CCfits::ExtHDU*> fits_io(files[i]);
            int n_extensions = fits_io.get_n_extensions();

            if (n_extensions == 0) {
                throw std::runtime_error(fmt::format("{} is empty", f));
            }

            for (int j = 1; j < n_extensions; ++j) {
                CCfits::ExtHDU& ext = fits_io.pfits->extension(j);
                std::string hdu_name;
                ext.readKey("EXTNAME", hdu_name);

                int group;
                ext.readKey("GROUP", group);

                if (hdu_name.find("signal") != std::string::npos) {
                    if (hdu_name.find("I") != std::string::npos) {
                        fruit_maps[array][group].signal.i = fits_io.get_hdu(hdu_name);

                        if (i == 0 && j == 0) {
                            fruit_maps.wcs = fits_io.get_wcs(hdu_name);
                            fruit_maps.n_cols = wcs.naxis[0];
                            fruit_maps.n_rows = wcs.naxis[1];
                        }
                    }
                    else if (hdu_name.find("Q") != std::string::npos) {
                        fruit_maps[array][group].signal.q = (fits_io.get_hdu(hdu_name));
                    }
                    else if (hdu_name.find("U") != std::string::npos) {
                        fruit_maps[array][group].signal.u = (fits_io.get_hdu(hdu_name));
                    }
                }
                if (hdu_name.find("weight") != std::string::npos) {
                    if (hdu_name.find("I") != std::string::npos) {
                        fruit_maps[array][group].weight.i = fits_io.get_hdu(hdu_name);
                    }
                    else if (hdu_name.find("Q") != std::string::npos) {
                        fruit_maps[array][group].weight.q = (fits_io.get_hdu(hdu_name));
                    }
                    else if (hdu_name.find("U") != std::string::npos) {
                        fruit_maps[array][group].weight.u = (fits_io.get_hdu(hdu_name));
                    }
                }
            }
        }

        // get median rms
        for (int i = 0; i < noise_files.size(); ++i) {
            int array = toltec.apt.arrays[i];

            fitsIO<FitsMode::WriteFits, CCfits::ExtHDU*> fits_io(files[i]);
            int n_extensions = fits_io.get_n_extensions();

            if (n_extensions == 0) {
                throw std::runtime_error(fmt::format("{} is empty", f));
            }

            for (int j = 1; j < n_extensions; ++j) {
                CCfits::ExtHDU& ext = fits_io.pfits->extension(j);
                std::string hdu_name;
                ext.readKey("EXTNAME", hdu_name);

                int group;
                ext.readKey("GROUP", group);

                if (hdu_name.find("noise") != std::string::npos) {
                    if (hdu_name.find("I") != std::string::npos) {
                        ext.readKey("MEDRMS", median_rms_I[array][group]);
                    }
                    if (hdu_name.find("Q") != std::string::npos) {
                        ext.readKey("MEDRMS", median_rms_Q[array][group]);
                    }
                    if (hdu_name.find("U") != std::string::npos) {
                        ext.readKey("MEDRMS", median_rms_U[array][group]);
                    }
                }
            }
        }

        fruit_maps.build_vectors();
    }

    void process(TCDataType& tcdata) override {
        logger->info("fruit loop processing");

        int n_dets = tcdata.n_dets();
        int n_pts = tcdata.n_pts();

        // loop through dets
        for (int det = 0; det < n_dets; ++det) {
            if (toltec.apt["flag"].data(det)) continue;

            // keys of current detector
            int array = toltec.apt["array"].data(det);
            int group = toltec.apt[fruit_maps.map_grouping].data(det);

            // get indices of maps
            int sig_i_index = fruit_maps.signal_lookup.at(MapKey(toltec.apt["array"].data(det), toltec.apt[map_grouping].data(det), "I"));
            int sig_q_index, sig_u_index;

            if (run_polarization) {
                sig_q_index = fruit_maps.signal_lookup.at(MapKey(toltec.apt["array"].data(det), toltec.apt[map_grouping].data(det), "Q"));
                sig_u_index = fruit_maps.signal_lookup.at(MapKey(toltec.apt["array"].data(det), toltec.apt[map_grouping].data(det), "U"));
            }

            auto xy = calc_pointing(toltec.apt["x_t"].data(det), toltec.apt["y_t"].data(det), tcdata.tel_data,
                                    telescope.pixel_axes);
            // pixels for samples
            Eigen::VectorXI pix_x = (xy.first.array() / fruit_maps.pix_size_radians + fruit_maps.n_cols / 2.0).template cast<Eigen::Index>();
            Eigen::VectorXI pix_y = (xy.second.array() / fruit_maps.pix_size_radians + fruit_maps.n_rows / 2.0).template cast<Eigen::Index>();

            // loop through data points
            for (int i = 0; i < n_pts; ++i) {
                // don't run flagged detectors
                if (tcdata.flag(i, det)) continue;

                // if pixel in map
                if (pix_x(i) >= 0 && pix_x(i) < fruit_maps.n_cols && pix_y(i) >= 0 && pix_y(i) < fruit_maps.n_rows) {
                    // check whether we should include pixel
                    bool run_pix_s2n_I = run_noise && (obs_map.signal[sig_i_index](pix_y(i), pix_x(i)) / median_rms_I[array][group] >= fruit_sig2noise);
                    bool run_pix_flux_I = obs_map.signal.i(pix_y(i), pix_x(i)) >= fruit_loops_flux(array);

                    if (run_pix_s2n_I || run_pix_flux_I) {
                        tcdata.signal(j, i) += add_subtract_factor * obs_map.signal[sig_i_index](pix_y(i), pix_x(i));
                    }

                    if (tcdata.signal_q) {
                        // check whether we should include pixel
                        bool run_pix_s2n_Q = run_noise && (obs_map.signal[sig_q_index](pix_y(i), pix_x(i)) / median_rms_Q[array][group] >= fruit_sig2noise);
                        bool run_pix_flux_Q = obs_map.signal.q(pix_y(i), pix_x(i)) >= fruit_loops_flux(array);
                        if (run_pix_s2n_Q || run_pix_flux_Q) {
                            tcdata.signal_q(j, i) += add_subtract_factor * obs_map.signal[sig_q_index](pix_y(i), pix_x(i));
                        }
                    }

                    if (tcdata.signal_u) {
                        // check whether we should include pixel
                        bool run_pix_s2n_U = run_noise && (obs_map.signal[sig_u_index](pix_y(i), pix_x(i)) / median_rms_U[array][group] >= fruit_sig2noise);
                        bool run_pix_flux_U = obs_map.signal.u(pix_y(i), pix_x(i)) >= fruit_loops_flux(array);
                        if (run_pix_s2n_U || run_pix_flux_U) {
                            tcdata.signal_u(j, i) += add_subtract_factor * obs_map.signal[sig_u_index](pix_y(i), pix_x(i));
                        }
                    }
                }
            }
        }
    }
};
