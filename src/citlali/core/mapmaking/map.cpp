#include <citlali/core/mapmaking/map.h>
#include <citlali/core/utils/toltec_io.h>

namespace mapmaking {

// constructor
ObsMapBuffer::ObsMapBuffer() {}

// constructor
ObsMapBuffer::ObsMapBuffer(std::string _n): name(_n) {}

// get config file
void ObsMapBuffer::get_config(tula::config::YamlConfig &config, std::vector<std::vector<std::string>> &missing_keys,
                              std::vector<std::vector<std::string>> &invalid_keys, std::string pixel_axes,
                              std::string redu_type) {

    // coverage cut
    get_config_value(config, cov_cut, missing_keys, invalid_keys,
                     std::tuple{"mapmaking","coverage_cut"});

    // number of histogram bins
    get_config_value(config, hist_n_bins, missing_keys, invalid_keys,
                     std::tuple{"post_processing","map_histogram_n_bins"},{},{0});

    // pixel size
    get_config_value(config, pixel_size_rad, missing_keys, invalid_keys,
                     std::tuple{"mapmaking","pixel_size_arcsec"},{},{0});

    // map units
    get_config_value(config, sig_unit, missing_keys, invalid_keys,
                     std::tuple{"mapmaking","cunit"},{"mJy/beam","MJy/sr","uK/beam", "Jy/pixel"});

    // convert pixel size to to radians
    pixel_size_rad *= ASEC_TO_RAD;

    // set wcs cdelt for cols
    wcs.cdelt.push_back(-pixel_size_rad);
    // set wcs cdelt for rows
    wcs.cdelt.push_back(pixel_size_rad);

    // variable to get wcs config options
    double wcs_double;

    // get wcs naxis
    std::vector<std::string> naxis = {"x_size_pix","y_size_pix"};
    for (const auto &key: naxis) {
        get_config_value(config, wcs_double, missing_keys, invalid_keys,
                         std::tuple{"mapmaking",key});
        wcs.naxis.push_back(wcs_double);
    }

    // get wcs crpix
    std::vector<std::string> crpix = {"crpix1","crpix2"};
    for (const auto &key: crpix) {
        get_config_value(config, wcs_double, missing_keys, invalid_keys,
                         std::tuple{"mapmaking",key});
        wcs.crpix.push_back(wcs_double);
    }

    // get wcs crval
    std::vector<std::string> crval = {"crval1_J2000","crval2_J2000"};
    for (const auto &key: crval) {
        get_config_value(config, wcs_double, missing_keys, invalid_keys,
                         std::tuple{"mapmaking",key});
        crval_config.push_back(wcs_double);
    }

    // setup wcs for radec frame
    if (pixel_axes == "radec") {
        wcs.ctype.push_back("RA---TAN");
        wcs.ctype.push_back("DEC--TAN");

        wcs.cunit.push_back("deg");
        wcs.cunit.push_back("deg");

        wcs.cdelt[0] *= RAD_TO_DEG;
        wcs.cdelt[1] *= RAD_TO_DEG;
    }

    // setup wcs altaz frame
    else if (pixel_axes == "altaz") {
        wcs.ctype.push_back("AZOFFSET");
        wcs.ctype.push_back("ELOFFSET");

        // arcsec if pointing or beammap
        if (redu_type != "science") {
            wcs.cunit.push_back("arcsec");
            wcs.cunit.push_back("arcsec");
            wcs.cdelt[0] *= RAD_TO_ASEC;
            wcs.cdelt[1] *= RAD_TO_ASEC;
        }
        // degrees if science
        else {
            wcs.cunit.push_back("deg");
            wcs.cunit.push_back("deg");
            wcs.cdelt[0] *= RAD_TO_DEG;
            wcs.cdelt[1] *= RAD_TO_DEG;
        }
    }

    // set wcs cdelt for freq and stokes
    wcs.cdelt.insert(wcs.cdelt.end(),{1,1});

    // set wcs crpix for freq and stokes
    wcs.crpix.insert(wcs.crpix.end(),{0,0});

    // set wcs crval to initial defaults
    wcs.crval.resize(4,0.);

    // set wcs naxis for freq and stokes
    wcs.naxis.insert(wcs.naxis.end(),{1,1});

    // set wcs ctypes for freq and stokes
    wcs.ctype.insert(wcs.ctype.end(),{"FREQ","STOKES"});

    // set wcs cunits for freq and stokes
    wcs.cunit.insert(wcs.cunit.end(),{"Hz",""});
}

void ObsMapBuffer::normalize_maps() {
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // placeholder vectors for grppi map
    std::vector<int> map_in_vec, map_out_vec, pointing_in_vec, pointing_out_vec;

    // vectors for maps
    map_in_vec.resize(signal.size());
    std::iota(map_in_vec.begin(), map_in_vec.end(), 0);
    map_out_vec.resize(signal.size());

    // vectors for pointing
    pointing_in_vec.resize(pointing.size());
    std::iota(pointing_in_vec.begin(), pointing_in_vec.end(), 0);
    pointing_out_vec.resize(pointing.size());

    // if not in polarized mode or coadded map, use default normalization for signal and kernel
    if (pointing.empty() || obsnums.size() > 1) {
        // normalize science and kernel mpas
        grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), map_in_vec, map_out_vec, [&](auto i) {
            // loop through rows
            for (Eigen::Index j=0; j<n_rows; j++) {
                // loop through cols
                for (Eigen::Index k=0; k<n_cols; k++) {
                    // weight of current pixel
                    double sig_weight = weight[i](j,k);
                    // normalize if weight is larger than zero
                    if (sig_weight > 0.) {
                        signal[i](j,k) = signal[i](j,k) / sig_weight;

                        // normalize kernel
                        if (!kernel.empty()) {
                            kernel[i](j,k) = kernel[i](j,k) / sig_weight;
                        }
                    }
                    // otherwise set all to zero
                    else {
                        signal[i](j,k) = 0;
                        weight[i](j,k) = 0;

                        if (!kernel.empty()) {
                            kernel[i](j,k) = 0;
                        }
                    }
                }
            }

            return 0;
        });
    }

    // if pointing matrix is not empty normalize signal, kernel (obsnum only) and noise maps
    if (!pointing.empty()) {
        // calculate dimensions
        auto calc_stokes = [&](auto &map_vec, auto &m, int i, int j, int a, int step) {
            Eigen::VectorXd d(3);
            // I
            d(0) = map_vec[a](i,j);
            // Q
            d(1) = map_vec[a + step](i,j);
            // U
            d(2) = map_vec[a + 2*step](i,j);

            // solve the equation d = Mv for v
            Eigen::VectorXd v = m.colPivHouseholderQr().solve(d);

            // I
            map_vec[a](i,j) = v(0);
            // Q
            map_vec[a + step](i,j) = v(1);
            // U
            map_vec[a + 2*step](i,j) = v(2);
        };

        // number of maps to step over to get to next stokes param
        int step = pointing.size();

        // loop through pointing matrices
        grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), pointing_in_vec, pointing_out_vec, [&](auto a) {
            // pointing matrix for pixel
            Eigen::MatrixXd m(3,3);

            // loop through rows
            for (Eigen::Index i=0; i<n_rows; i++) {
                // loop through cols
                for (Eigen::Index j=0; j<n_cols; j++) {
                    // create pointing matrix for pixel
                    Eigen::Index n = 0;
                    for (Eigen::Index k=0; k<3; k++) {
                        for (Eigen::Index l=0; l<3; l++) {
                            m(k,l) = pointing[a](i,j,n);
                            n++;
                        }
                    }
                    // if m array is not zero and invertible
                    if ((m.array() != 0).all() && m.determinant() > 1e-20) {

                        // only run on signal and kernel of obsnum map
                        if (obsnums.size() == 1) {
                            // calc stokes values for signal map
                            calc_stokes(signal,m,i,j,a,step);

                            if (!kernel.empty()) {
                                // calc stokes values for kernel map
                                calc_stokes(kernel,m,i,j,a,step);
                            }
                        }

                        // if running noise maps
                        if (!noise.empty()) {
                            // loop through noise map
                            for (Eigen::Index nn=0; nn<n_noise; ++nn) {
                                // vector to hold noise map values
                                std::vector<Eigen::MatrixXd> noise_vec(noise.size());
                                // only store current pixel to save memory
                                Eigen::MatrixXd noise_map(1,1);
                                // I
                                noise_map(0,0) = noise[a](i,j,nn);
                                noise_vec[a] = noise_map;
                                // Q
                                noise_map(0,0) = noise[a + step](i,j,nn);
                                noise_vec[a + step] = noise_map;
                                // U
                                noise_map(0,0) = noise[a + 2*step](i,j,nn);
                                noise_vec[a + 2*step] = noise_map;

                                // calc stokes values for noise map
                                calc_stokes(noise_vec,m,0,0,a,step);

                                // repopulate noise vector
                                noise[a](i,j,nn) = noise_vec[a](0,0);
                                noise[a + step](i,j,nn) = noise_vec[a + step](0,0);
                                noise[a + 2*step](i,j,nn) = noise_vec[a + 2*step](0,0);
                            }
                        }
                    }
                    // otherwise set all stokes values to zero
                    else {
                        // only run on signal and kernel of obsnum map
                        if (obsnums.size() == 1) {
                            signal[a](i,j) = 0.;
                            signal[a + step](i,j) = 0.;
                            signal[a + 2*step](i,j) = 0.;

                            if (!kernel.empty()) {
                                kernel[a](i,j) = 0.;
                                kernel[a + step](i,j) = 0.;
                                kernel[a + 2*step](i,j) = 0.;
                            }
                        }

                        // if running noise maps
                        if (!noise.empty()) {
                            // loop through noise map
                            for (Eigen::Index nn=0; nn<n_noise; ++nn) {
                                // repopulate noise vector
                                noise[a](i,j,nn) = 0.;
                                noise[a + step](i,j,nn) = 0.;
                                noise[a + 2*step](i,j,nn) = 0.;
                            }
                        }
                    }
                }
            }
            // only run on signal and kernel of obsnum map
            if (obsnums.size() == 1) {
                // don't need to update weight maps
                weight[a + step] = weight[a];
                weight[a + 2*step] = weight[a];

                // don't need to update coverage map
                if (!coverage.empty()) {
                    coverage[a + step] = coverage[a];
                    coverage[a + 2*step] = coverage[a];
                }
            }
            return 0;
        });
    }
    // otherwise normalize noise maps normally
    else {
        // normalize noise maps
        if (!noise.empty()) {
            grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), map_in_vec, map_out_vec, [&](auto i) {
                // loop through rows
                for (Eigen::Index j=0; j<n_rows; j++) {
                    // loop through cols
                    for (Eigen::Index k=0; k<n_cols; k++) {
                        // weight of current pixel
                        double sig_weight = weight[i](j,k);
                        // normalize if weight is larger than zero
                        if (sig_weight > 0.) {
                            // loop through noise maps
                            for (Eigen::Index l=0; l<n_noise; l++) {
                                // normalize by weight
                                noise[i](j,k,l) = noise[i](j,k,l) / sig_weight;
                            }
                        }
                        // otherwise set all to zero
                        else {
                            for (Eigen::Index l=0; l<n_noise; l++) {
                                noise[i](j,k,l) = 0;
                            }
                        }
                    }
                }
                return 0;
            });
        }
    }
}

std::tuple<double, Eigen::MatrixXd, Eigen::Index, Eigen::Index> ObsMapBuffer::calc_cov_region(Eigen::Index i) {
    // calculate weight threshold
    double weight_threshold = engine_utils::find_weight_threshold(weight[i], cov_cut);

    // calculate coverage ranges
    Eigen::MatrixXd cov_ranges = engine_utils::set_cov_cov_ranges(weight[i], weight_threshold);

    // rows and cols of region above weight threshold
    Eigen::Index cov_n_rows = cov_ranges(1,0) - cov_ranges(0,0) + 1;
    Eigen::Index cov_n_cols = cov_ranges(1,1) - cov_ranges(0,1) + 1;

    return std::tuple<double, Eigen::MatrixXd, Eigen::Index, Eigen::Index>(weight_threshold, cov_ranges,
                                                                           cov_n_rows, cov_n_cols);
}

// loop through maps
void ObsMapBuffer::calc_map_psd() {
    // clear psd vectors
    psds.clear();
    psd_freqs.clear();
    psd_2ds.clear();
    psd_2d_freqs.clear();

    // clear noise psd vectors
    noise_psds.clear();
    noise_psd_freqs.clear();
    noise_psd_2ds.clear();
    noise_psd_2d_freqs.clear();

    // loop through maps
    for (Eigen::Index i=0; i<signal.size(); i++) {
        // calculate weight threshold
        auto [weight_threshold, cov_ranges, cov_n_rows, cov_n_cols] = calc_cov_region(i);

        // ensure even rows
        if (cov_n_rows % 2 == 1) {
            cov_ranges(1,0)--;
            cov_n_rows--;
        }

        // ensure even cols
        if (cov_n_cols % 2 == 1) {
            cov_ranges(1,1)--;
            cov_n_cols--;
        }

        // explicit copy signal map within coverage region
        Eigen::MatrixXd sig = signal[i].block(cov_ranges(0,0), cov_ranges(0,1), cov_n_rows, cov_n_cols);

        // calculate psds
        auto [p, pf, p_2d, pf_2d] = engine_utils::calc_2D_psd(sig, rows_tan_vec, cols_tan_vec, smooth_window, parallel_policy);
        // move current map psd values into vectors
        psds.push_back(std::move(p));
        psd_freqs.push_back(std::move(pf));

        psd_2ds.push_back(std::move(p_2d));
        psd_2d_freqs.push_back(std::move(pf_2d));

        // get average noise psd if noise maps are requested
        if (!noise.empty()) {
            for (Eigen::Index j=0; j<n_noise; j++) {
                // get noise map
                Eigen::Tensor<double, 2> noise_tensor = noise[i].chip(j, 2);
                // map to eigen matrix
                Eigen::Map<Eigen::MatrixXd> noise_matrix(noise_tensor.data(), noise_tensor.dimension(0), noise_tensor.dimension(1));
                sig = noise_matrix.block(cov_ranges(0,0), cov_ranges(0,1), cov_n_rows, cov_n_cols);

                // calculate psds
                auto [noise_p, noise_pf, noise_p_2d, noise_pf_2d] = engine_utils::calc_2D_psd(sig, rows_tan_vec, cols_tan_vec,
                                                                                              smooth_window, parallel_policy);

                // just copy if on first noise map
                if (j==0) {
                    noise_psds.push_back(std::move(noise_p));
                    noise_psd_freqs.push_back(std::move(noise_pf));

                    noise_psd_2ds.push_back(std::move(noise_p_2d));
                    noise_psd_2d_freqs.push_back(std::move(noise_pf_2d));
                }

                // otherwise add to existing vector
                else {
                    noise_psds.back() = noise_psds.back() + noise_p;
                    noise_psd_2ds.back() = noise_psd_2ds.back() + noise_p_2d;
                    noise_psd_2d_freqs.back() = noise_psd_2d_freqs.back() + noise_pf_2d;
                }
            }
            noise_psds.back() = noise_psds.back()/n_noise;
            noise_psd_2ds.back() = noise_psd_2ds.back()/n_noise;
            noise_psd_2d_freqs.back() = noise_psd_2d_freqs.back()/n_noise;
        }
    }
}

void ObsMapBuffer::calc_map_hist() {
    // clear storage vectors
    hists.clear();
    hist_bins.clear();

    noise_hists.clear();
    noise_hist_bins.clear();

    // loop through maps
    for (Eigen::Index i=0; i<signal.size(); i++) {
        // calculate weight threshold
        auto [weight_threshold, cov_ranges, cov_n_rows, cov_n_cols] = calc_cov_region(i);

        // setup input signal data
        Eigen::MatrixXd sig = signal[i].block(cov_ranges(0,0),cov_ranges(0,1), cov_n_rows, cov_n_cols);

        // calculate histogram and bins
        auto [h, h_bins] = engine_utils::calc_hist(sig, hist_n_bins);

        hists.push_back(std::move(h));
        hist_bins.push_back(std::move(h_bins));

        // get average noise psd if noise maps are requested
        if (!noise.empty()) {
            for (Eigen::Index j=0; j<n_noise; j++) {
                // get noise map
                Eigen::Tensor<double, 2> noise_tensor = noise[i].chip(j,2);
                // map to eigen matrix
                Eigen::Map<Eigen::MatrixXd> noise_matrix(noise_tensor.data(), noise_tensor.dimension(0), noise_tensor.dimension(1));
                sig = noise_matrix.block(cov_ranges(0,0), cov_ranges(0,1), cov_n_rows, cov_n_cols);

                // calculate histogram and bins
                auto [noise_h, noise_h_bins] = engine_utils::calc_hist(sig, hist_n_bins);

                // just copy if on first noise map
                if (j==0) {
                    noise_hists.push_back(std::move(noise_h));
                    noise_hist_bins.push_back(std::move(noise_h_bins));
                }
                // otherwise add to existing vector
                else {
                    noise_hists.back() = noise_hists.back() + noise_h;
                }
            }
            noise_hists.back() = noise_hists.back()/n_noise;
        }
    }
}

void ObsMapBuffer::calc_mean_err() {
    // resize mean errors
    mean_err.setZero(weight.size());
    for (Eigen::Index i=0; i<weight.size(); i++) {
        // calculate weight threshold
        auto [weight_threshold, cov_ranges, cov_n_rows, cov_n_cols] = calc_cov_region(i);

        int counter = (weight[i].array()>=weight_threshold).count();
        double mean_sqerr = ((weight[i].array()>=weight_threshold).select(1/weight[i].array(),0)).sum();

        // get mean square error
        mean_err(i) = mean_sqerr/counter;
    }
}

void ObsMapBuffer::calc_mean_rms() {
    // average filtered rms vector
    mean_rms.setZero(noise.size());

    // loop through arrays/polarizations
    for (Eigen::Index i=0; i<noise.size(); i++) {
        // vector of rms of noise maps
        Eigen::VectorXd noise_rms(n_noise);
        for (Eigen::Index j=0; j<n_noise; j++) {
            // calculate weight threshold
            auto [weight_threshold, cov_ranges, cov_n_rows, cov_n_cols] = calc_cov_region(i);

            // get matrix of current noise map
            Eigen::Tensor<double,2> out = noise[i].chip(j,2);
            auto out_matrix = Eigen::Map<Eigen::MatrixXd>(out.data(), out.dimension(0), out.dimension(1));

            int counter = (weight[i].array()>=weight_threshold).count();
            double rms = ((weight[i].array()>=weight_threshold).select(pow(out_matrix.array(),2),0)).sum();

            noise_rms(j) = sqrt(rms/counter);
        }
        // get mean rms
        mean_rms(i) = noise_rms.mean();
    }
}

bool ObsMapBuffer::find_sources(Eigen::Index map_index) {
    // calc coverage bool map
    Eigen::MatrixXd ones, zeros;

    ones.setOnes(n_rows, n_cols);
    zeros.setZero(n_rows, n_cols);

    // get weight threshold for current map
    auto [weight_threshold, cov_ranges, cov_n_rows, cov_n_cols] = calc_cov_region(map_index);
    // if weight is less than threshold, set to zero, otherwise set to one
    Eigen::MatrixXd cov_bool = (weight[map_index].array() < weight_threshold).select(zeros,ones);

    // swap the sign of the signal map
    if (source_finder_mode=="negative") {
        signal[map_index] = -signal[map_index];
    }

    // s/n map
    Eigen::MatrixXd sig2noise = sqrt(weight[map_index].array())*signal[map_index].array();

    // find pixels equal or above source sigma
    std::vector<int> row_index, col_index;

    // search both positive and negatives
    if (source_finder_mode=="both") {
        for (Eigen::Index i=0; i<n_rows; i++) {
            for (Eigen::Index j=0; j<n_cols; j++) {
                if (cov_bool(i,j) == 1) {
                    if (abs(sig2noise(i,j)) >= source_sigma) {
                        row_index.push_back(i);
                        col_index.push_back(j);
                    }
                }
            }
        }
    }
    else {
        for (Eigen::Index i=0; i<n_rows; i++) {
            for (Eigen::Index j=0; j<n_cols; j++) {
                if (cov_bool(i,j) == 1) {
                    if (sig2noise(i,j) >= source_sigma) {
                        row_index.push_back(i);
                        col_index.push_back(j);
                    }
                }
            }
        }
    }

    // if no sources found
    if (row_index.size()==0) {
        return false;
    }

    // make sure source extremum is within good coverage region by
    // searching in index boxes of +/- 1 pixel around hot pixels
    std::vector<int> row_source_index, col_source_index;
    for (unsigned int i=0; i<row_index.size(); i++) {
        double extremum;
        if (source_finder_mode=="both" && signal[map_index](row_index[i],col_index[i]) < 0.0) {
            extremum = signal[map_index](row_index[i],col_index[i]);
            // find minimum within index box
            for (Eigen::Index j=row_index[i]-1; j<row_index[i]+2; j++) {
                for (Eigen::Index k=col_index[i]-1; k<col_index[i]+2; k++) {
                    if (signal[map_index](j,k) < extremum) {
                        extremum = signal[map_index](j,k);
                    }
                }
            }
        }
        else {
            extremum = signal[map_index](row_index[i],col_index[i]);
            // find maximum within index box
            for (Eigen::Index j=row_index[i]-1; j<row_index[i]+2; j++) {
                for (Eigen::Index k=col_index[i]-1; k<col_index[i]+2; k++) {
                    if (signal[map_index](j,k) > extremum) {
                        extremum = signal[map_index](j,k);
                    }
                }
            }
        }
        // only keep the hot pixel if it is the extremum
        if (signal[map_index](row_index[i],col_index[i]) == extremum) {
            row_source_index.push_back(row_index[i]);
            col_source_index.push_back(col_index[i]);
        }
    }

    int n_raw_sources = row_source_index.size();
    // done with vectors of all hot pixels
    row_index.clear();
    col_index.clear();

    // if no sources found
    if (n_raw_sources == 0) {
        return false;
    }

    // find indices of hot pixels close together
    std::vector<int> row_dist_index, col_dist_index;

    for (Eigen::Index i=0; i<n_raw_sources; i++) {
        for (Eigen::Index j=0; j<n_raw_sources; j++) {
            unsigned int row_sep = pow(row_source_index[i] - row_source_index[j],2);
            unsigned int col_sep = pow(col_source_index[i] - col_source_index[j],2);
            double hot_dist = sqrt(row_sep + col_sep);
            if (hot_dist <= (source_window_rad/pixel_size_rad) && hot_dist != 0.0) {
                row_dist_index.push_back(i);
                col_dist_index.push_back(j);
            }
        }
    }

    // flag non-maximum hot pixel indices
    if (row_dist_index.size() != 0) {
        for (unsigned int i=0; i<row_dist_index.size(); i++) {
            if (row_source_index[row_dist_index[i]] == -1 || col_source_index[col_dist_index[i]] == -1) {
                continue;
            }
            double f1 = signal[map_index](row_source_index[row_dist_index[i]],col_source_index[row_dist_index[i]]);
            double f2 = signal[map_index](row_source_index[col_dist_index[i]],col_source_index[col_dist_index[i]]);
            // determine if same sign and which sign
            if (f1 < 0.0 && f2 < 0.0) {
                // negative case
                if (f1 <= f2) {
                    row_source_index[col_dist_index[i]] = -1;
                    col_source_index[col_dist_index[i]] = -1;
                }
                else {
                    row_source_index[row_dist_index[i]] = -1;
                    col_source_index[row_dist_index[i]] = -1;
                }
            }
            else{
                // positive case
                if (f1 >= f2) {
                    row_source_index[col_dist_index[i]] = -1;
                    col_source_index[col_dist_index[i]] = -1;
                }
                else{
                    row_source_index[row_dist_index[i]] = -1;
                    col_source_index[row_dist_index[i]] = -1;
                }
            }
        }
    }

    row_dist_index.clear();
    col_dist_index.clear();

    // get rows/cols of each source
    std::vector<int> row_source_loc, col_source_loc;
    for (Eigen::Index i=0; i<n_raw_sources; i++) {
        if ((row_source_index[i] != -1) && (col_source_index[i] != -1)) {
            row_source_loc.push_back(row_source_index[i]);
            col_source_loc.push_back(col_source_index[i]);
            n_sources[map_index]++;
        }
    }

    // done with flag filled source index vectors
    row_source_index.clear();
    col_source_index.clear();

    // copy locations for current map
    row_source_locs[map_index] = Eigen::Map<Eigen::VectorXi>(row_source_loc.data(),
                                                             row_source_loc.size());
    col_source_locs[map_index] = Eigen::Map<Eigen::VectorXi>(col_source_loc.data(),
                                                             col_source_loc.size());

    // if no sources found
    if (n_sources[map_index] == 0) {
        return false;
    }

    // flip signal map
    if (source_finder_mode=="negative") {
        signal[map_index] = -signal[map_index];
    }

    return true;
}

}  // namespace mapmaking
