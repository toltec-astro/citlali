# pragma once

#include <citlali/core/utils/beam.h>

using namespace citlali::config::options;

// add or subtract gaussian source
enum class FruitMode {
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
    std::string fruit_path, fruit_source, curr_fruit_dir;
    double fruit_sig2noise;
    std::vector<double> fruit_flux;
    int fruit_iter, fruit_iters;

    double add_subtract_factor;
    Instrument& toltec;
    Telescope& telescope;
    std::map<int, std::map<int, double>> median_rms_I, median_rms_Q, median_rms_U;

    template <typename ConfigType>
    FruitLoops(FruitMode fruit_mode, Instrument& toltec_, Telescope& telescope_, ConfigType& config)
        : toltec(toltec_), telescope(telescope_) {

        add_subtract_factor = static_cast<double>(fruit_mode);

        config.get(fruit_path, std::tuple{"timestream","fruit_loops","path"});
        config.get(fruit_source, std::tuple{"timestream","fruit_loops","source"});
        config.get(fruit_sig2noise, std::tuple{"timestream","fruit_loops","lower_sig2noise_limit"});
        config.get(fruit_flux, std::tuple{"timestream","fruit_loops","fruit_flux"});
        config.get(fruit_iters, std::tuple{"timestream","fruit_loops","max_iters"});
    }

    void init() {
        namespace fs = std::filesystem;

        if (fruit_iter > 0 || fruit_path != "null") {
            std::vector<std::string> files, noise_files;

            for (const auto& array : toltec.apt.arrays) {
                for (const auto& entry : fs::directory_iterator(curr_fruit_dir)) {
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

            // get signal and weight
            for (int i = 0; i < files.size(); ++i) {
                int array = toltec.apt.arrays[i];

                fitsIO<FitsMode::ReadFits, CCfits::ExtHDU*> fits_io(files[i]);
                int n_extensions = fits_io.get_n_extensions();

                if (n_extensions == 0) {
                    throw std::runtime_error(fmt::format("{} is empty", files[i]));
                }

                WCS wcs;

                CCfits::ExtHDU& phdu = fits_io.pfits->extension(0);

                bool has_kernel;
                phdu.readKey("CONFIG.KERNEL", has_kernel);

                for (int j = 1; j < n_extensions; ++j) {
                    CCfits::ExtHDU& ext = fits_io.pfits->extension(j);
                    std::string hdu_name;
                    ext.readKey("EXTNAME", hdu_name);

                    int group;
                    ext.readKey("GROUP", group);

                    if (hdu_name.find("signal") != std::string::npos) {
                        if (hdu_name.find("I") != std::string::npos) {
                            if (i == 0 && j == 0) {
                                wcs = fits_io.get_wcs(hdu_name);
                            }

                            MapKey i_key(array, group, "I");
                            fruit_maps.add(i_key, {wcs.naxis[1], wcs.naxis[0]}, true, has_kernel, false);
                            fruit_maps.signal[fruit_maps.signal_lookup[i_key]].data = fits_io.get_hdu(hdu_name);
                        }
                        else if (hdu_name.find("Q") != std::string::npos) {
                            MapKey q_key(array, group, "Q");
                            fruit_maps.add(q_key, {wcs.naxis[1], wcs.naxis[0]}, true, false, false);
                            fruit_maps.signal[fruit_maps.signal_lookup[q_key]].data = fits_io.get_hdu(hdu_name);
                        }
                        else if (hdu_name.find("U") != std::string::npos) {
                            MapKey u_key(array, group, "U");
                            fruit_maps.add(u_key, {wcs.naxis[1], wcs.naxis[0]}, true, false, false);
                            fruit_maps.signal[fruit_maps.signal_lookup[u_key]].data = fits_io.get_hdu(hdu_name);
                        }
                    }
                    else if (hdu_name.find("weight") != std::string::npos) {
                        if (hdu_name.find("I") != std::string::npos) {
                            MapKey i_key(array, group, "I");
                            fruit_maps.weight[fruit_maps.weight_lookup[i_key]].data = fits_io.get_hdu(hdu_name);
                        }
                        else if (hdu_name.find("Q") != std::string::npos) {
                            MapKey q_key(array, group, "Q");
                            fruit_maps.weight[fruit_maps.weight_lookup[q_key]].data = fits_io.get_hdu(hdu_name);
                        }
                        else if (hdu_name.find("U") != std::string::npos) {
                            MapKey u_key(array, group, "U");
                            fruit_maps.weight[fruit_maps.weight_lookup[u_key]].data = fits_io.get_hdu(hdu_name);
                        }
                    }
                }
            }

            // get median rms
            for (int i = 0; i < noise_files.size(); ++i) {
                int array = toltec.apt.arrays[i];

                fitsIO<FitsMode::ReadFits, CCfits::ExtHDU*> fits_io(noise_files[i]);
                int n_extensions = fits_io.get_n_extensions();

                if (n_extensions == 0) {
                    throw std::runtime_error(fmt::format("{} is empty", noise_files[i]));
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
        }
    }

    void process(TCDataType& tcdata) override {
        logger->info("fruit loop processing");

        if (fruit_maps.signal.size() > 0) {

            int n_dets = tcdata.n_dets();
            int n_pts = tcdata.n_pts();

            // loop through dets
            for (int det = 0; det < n_dets; ++det) {
                if (toltec.apt["flag"].data(det)) continue;

                // keys of current detector
                int array = toltec.apt["array"].data(det);
                int group = toltec.apt[map_grouping].data(det);

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
                Eigen::VectorXI pix_x = (xy.first.array() / pix_size_radians + fruit_maps.wcs.naxis[0] / 2.0).template cast<Eigen::Index>();
                Eigen::VectorXI pix_y = (xy.second.array() / pix_size_radians + fruit_maps.wcs.naxis[1] / 2.0).template cast<Eigen::Index>();

                // loop through data points
                for (int i = 0; i < n_pts; ++i) {
                    // don't run flagged detectors
                    if (tcdata.flag(i, det)) continue;

                    // if pixel in map
                    if (pix_x(i) >= 0 && pix_x(i) < fruit_maps.wcs.naxis[0] && pix_y(i) >= 0 && pix_y(i) < fruit_maps.wcs.naxis[1]) {
                        // check whether we should include pixel
                        bool run_pix_s2n_I = run_noise_maps && (fruit_maps.signal[sig_i_index].data(pix_y(i), pix_x(i)) / median_rms_I[array][group] >= fruit_sig2noise);
                        bool run_pix_flux_I = fruit_maps.signal[sig_i_index].data(pix_y(i), pix_x(i)) >= fruit_flux[array];

                        if (run_pix_s2n_I || run_pix_flux_I) {
                            tcdata.signal(i, det) += add_subtract_factor * fruit_maps.signal[sig_i_index].data(pix_y(i), pix_x(i));
                        }

                        if (tcdata.signal_q) {
                            // check whether we should include pixel
                            bool run_pix_s2n_Q = run_noise_maps && (fruit_maps.signal[sig_q_index].data(pix_y(i), pix_x(i)) / median_rms_Q[array][group] >= fruit_sig2noise);
                            bool run_pix_flux_Q = fruit_maps.signal[sig_q_index].data(pix_y(i), pix_x(i)) >= fruit_flux[array];
                            if (run_pix_s2n_Q || run_pix_flux_Q) {
                                tcdata.signal_q.value()(i, det) += add_subtract_factor * fruit_maps.signal[sig_q_index].data(pix_y(i), pix_x(i));
                            }
                        }

                        if (tcdata.signal_u) {
                            // check whether we should include pixel
                            bool run_pix_s2n_U = run_noise_maps && (fruit_maps.signal[sig_u_index].data(pix_y(i), pix_x(i)) / median_rms_U[array][group] >= fruit_sig2noise);
                            bool run_pix_flux_U = fruit_maps.signal[sig_u_index].data(pix_y(i), pix_x(i)) >= fruit_flux[array];
                            if (run_pix_s2n_U || run_pix_flux_U) {
                                tcdata.signal_u.value()(i, det) += add_subtract_factor * fruit_maps.signal[sig_u_index].data(pix_y(i), pix_x(i));
                            }
                        }
                    }
                }
            }
        }
    }
};
