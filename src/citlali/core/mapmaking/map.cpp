#include <citlali/core/mapmaking/map.h>
#include <citlali/core/utils/toltec_io.h>

namespace mapmaking {

/*template <class MapFitter, typename Derived>
void ObsMapBuffer::fit_maps(MapFitter &map_fitter, Eigen::DenseBase<Derived> &params,
                            Eigen::DenseBase<Derived> &perrors) {

    engine_utils::toltecIO toltec_io;

    // placeholder vectors for grppi map
    std::vector<int> map_in_vec, map_out_vec;

    map_in_vec.resize(signal.size());
    std::iota(map_in_vec.begin(), map_in_vec.end(), 0);
    map_out_vec.resize(signal.size());

    double init_row = -99;
    double init_col = -99;

    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), map_in_vec, map_out_vec, [&](auto i) {
        auto array = maps_to_arrays(i);
        // init fwhm in pixels
        double init_fwhm = toltec_io.array_fwhm_arcsec[array]*ASEC_TO_RAD/pixel_size_rad;
        auto [map_params, map_perror, good_fit] =
            map_fitter.fit_to_gaussian<engine_utils::mapFitter::pointing>(signal[i], weight[i], init_fwhm, init_row, init_col);
        params.row(i) = map_params;
        perrors.row(i) = map_perror;

        if (good_fit) {
            // rescale fit params from pixel to on-sky units
            params(i,1) = RAD_TO_ASEC*pixel_size_rad*(params(i,1) - (n_cols)/2);
            params(i,2) = RAD_TO_ASEC*pixel_size_rad*(params(i,2) - (n_rows)/2);
            params(i,3) = RAD_TO_ASEC*STD_TO_FWHM*pixel_size_rad*(params(i,3));
            params(i,4) = RAD_TO_ASEC*STD_TO_FWHM*pixel_size_rad*(params(i,4));

            // rescale fit errors from pixel to on-sky units
            perrors(i,1) = RAD_TO_ASEC*pixel_size_rad*(perrors(i,1));
            perrors(i,2) = RAD_TO_ASEC*pixel_size_rad*(perrors(i,2));
            perrors(i,3) = RAD_TO_ASEC*STD_TO_FWHM*pixel_size_rad*(perrors(i,3));
            perrors(i,4) = RAD_TO_ASEC*STD_TO_FWHM*pixel_size_rad*(perrors(i,4));
        }
        return 0;
    });
}*/

void ObsMapBuffer::normalize_maps() {
    // placeholder vectors for grppi map
    std::vector<int> map_in_vec, map_out_vec;

    map_in_vec.resize(signal.size());
    std::iota(map_in_vec.begin(), map_in_vec.end(), 0);
    map_out_vec.resize(signal.size());

    // normalize science and kernel mpas
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), map_in_vec, map_out_vec, [&](auto i) {
        for (Eigen::Index j=0; j<n_rows; j++) {
            for (Eigen::Index k=0; k<n_cols; k++) {
                double sig_weight = weight[i](j,k);
                if (sig_weight > 0.) {
                    signal[i](j,k) = signal[i](j,k) / sig_weight;

                    if (!kernel.empty()) {
                        kernel[i](j,k) = kernel[i](j,k) / sig_weight;
                    }
                }
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

    // normalize noise maps
    if (!noise.empty()) {
        grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), map_in_vec, map_out_vec, [&](auto i) {
            for (Eigen::Index j=0; j<n_rows; j++) {
                for (Eigen::Index k=0; k<n_cols; k++) {
                    double sig_weight = weight[i](j,k);
                    if (sig_weight > 0.) {
                        for (Eigen::Index l=0; l<n_noise; l++) {
                            noise[i](j,k,l) = noise[i](j,k,l) / sig_weight;
                        }
                    }
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

    /*for (Eigen::Index a=0; a<test0.size(); a++) {
        for (Eigen::Index i=0; i<nrows; i++) {
            for (Eigen::Index j=0; j<ncols; j++) {
                Eigen::MatrixXd m(3,3);
                Eigen::VectorXd d(3);
                d(0) = signal[3*a](i,j);
                d(1) = signal[3*a + 1](i,j);
                d(2) = signal[3*a + 2](i,j);

                Eigen::Index n = 0;
                for (k=0; k<3; k++) {
                    for (l=0; l<3; l++) {
                        m(k,l) = test0[a](i,j,n);
                        n++;
                    }
                }
                auto v = m.inverse()*d;
                signal[3*a](i,j) = v(0);
                signal[3*a + 1](i,j) = v(1);
                signal[3*a + 2](i,j) = v(2);
            }
        }
    }*/

}

std::tuple<double, Eigen::MatrixXd, Eigen::Index, Eigen::Index> ObsMapBuffer::calc_cov_region(Eigen::Index i) {
    // calculate weight threshold
    double weight_threshold = engine_utils::find_weight_threshold(weight[i], cov_cut);

    // calculate coverage ranges
    Eigen::MatrixXd cov_ranges = engine_utils::set_cov_cov_ranges(weight[i], weight_threshold);

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

        // explicit copy
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
                    noise_psd_2ds.back() = noise_psd_2ds.back() + noise_p_2d/n_noise;
                    noise_psd_2d_freqs.back() = noise_psd_2d_freqs.back() + noise_pf_2d/n_noise;
                }
            }
            noise_psds.back() = noise_psds.back()/n_noise;
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
    mean_rms.setZero(weight.size());

    // loop through arrays/polarizations
    for (Eigen::Index i=0; i<weight.size(); i++) {
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
        SPDLOG_INFO("mean rms {} ({})", static_cast<float>(mean_rms(i)), sig_unit);
    }
}

void ObsMapBuffer::renormalize_errors() {
    // get mean error from weight maps
    calc_mean_err();

    // get mean map rms from noise maps
    calc_mean_rms();

    // get rescaled normalization factor
    auto noise_factor = (1./pow(mean_rms.array(),2.))*mean_err.array();

    // loop through arrays/polarizations
    for (Eigen::Index i=0; i<weight.size(); i++) {
        // renormalize weights
        weight[i].noalias() = weight[i]*noise_factor(i);
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
