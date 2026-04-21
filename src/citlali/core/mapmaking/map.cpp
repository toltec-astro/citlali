#include <Eigen/Sparse>

#include <citlali/core/mapmaking/map.h>
#include <citlali/core/utils/toltec_io.h>

namespace mapmaking {

// constructor
MapBuffer::MapBuffer() {}

// constructor
MapBuffer::MapBuffer(std::string _n): name(_n) {}

// get config file
void MapBuffer::get_config(tula::config::YamlConfig &config, std::vector<std::vector<std::string>> &missing_keys,
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
                     std::tuple{"mapmaking","cunit"},{"mJy/beam","MJy/sr","uK", "Jy/pixel"});

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

    // setup wcs altaz frame
    else if (pixel_axes == "galactic") {
        wcs.ctype.push_back("GLON-TAN");
        wcs.ctype.push_back("GLAT-TAN");

        wcs.cunit.push_back("deg");
        wcs.cunit.push_back("deg");

        wcs.cdelt[0] *= RAD_TO_DEG;
        wcs.cdelt[1] *= RAD_TO_DEG;
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

void MapBuffer::normalize_maps() {
    // vectors for maps
    map_in_vec.resize(signal.size());
    std::iota(map_in_vec.begin(), map_in_vec.end(), 0);
    map_out_vec.resize(signal.size());

    const bool use_grid_weight = grid_weight.size() == signal.size();

    // normalize science and kernel mpas
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), map_in_vec, map_out_vec, [&](auto i) {
        Eigen::ArrayXXd mask(n_rows, n_cols);
        Eigen::MatrixXd finalized_weight = Eigen::MatrixXd::Zero(n_rows, n_cols);

        auto apply_weight_threshold = [&](Eigen::MatrixXd &weight_map) {
            mask = (weight_map.array() > 0.0).template cast<double>();
            if (cov_cut > 0.0) {
                // Use a softer support floor than the science coverage cut so
                // low-level kernel wings survive while unsupported edge pixels
                // are still removed before normalization.
                double support_cov_cut = cov_cut / 10.0;
                double weight_threshold = engine_utils::find_weight_threshold(weight_map, support_cov_cut);
                if (weight_threshold > 0.0) {
                    mask *= (weight_map.array() >= weight_threshold).template cast<double>();
                }
            }
        };

        if (use_grid_weight) {
            const auto &denom = grid_weight[i];
            const auto denom_valid = denom.array().isFinite() && (denom.array().abs() > 1e-8);
            const auto accum_weight_valid = weight[i].array().isFinite() && (weight[i].array() > 0.0);
            const auto valid_support = denom_valid && accum_weight_valid;
            const Eigen::ArrayXXd safe_denom = denom_valid.select(denom.array(), 1.0);

            finalized_weight =
                ((denom.array().square() / weight[i].array().max(1e-30)) *
                 valid_support.template cast<double>()).matrix();
            apply_weight_threshold(finalized_weight);

            signal[i] = ((signal[i].array() / safe_denom) * mask).matrix();

            if (!kernel.empty()) {
                kernel[i] = ((kernel[i].array() / safe_denom) * mask).matrix();
            }

            if (!noise.empty()) {
                for (Eigen::Index n = 0; n < n_noise; ++n) {
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(
                        noise[i].data() + n * n_rows * n_cols, n_rows, n_cols);
                    noise_matrix = ((noise_matrix.array() / safe_denom) * mask).matrix();
                }
            }

            weight[i] = (finalized_weight.array() * mask).matrix();
        }
        else {
            finalized_weight =
                (weight[i].array().isFinite() && (weight[i].array() > 0.0))
                    .select(weight[i].array(), 0.0).matrix();
            apply_weight_threshold(finalized_weight);
            const Eigen::ArrayXXd safe_weight = (finalized_weight.array() > 0.0)
                                                    .select(finalized_weight.array(), 1.0);

            signal[i] = ((signal[i].array() / safe_weight) * mask).matrix();

            if (!kernel.empty()) {
                kernel[i] = ((kernel[i].array() / safe_weight) * mask).matrix();
            }

            if (!noise.empty()) {
                for (Eigen::Index n = 0; n < n_noise; ++n) {
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(
                        noise[i].data() + n * n_rows * n_cols, n_rows, n_cols);
                    noise_matrix = ((noise_matrix.array() / safe_weight) * mask).matrix();
                }
            }

            weight[i] = (finalized_weight.array() * mask).matrix();
        }

        if (!coverage.empty()) {
            coverage[i] = (coverage[i].array() * mask).matrix();
        }
        return 0;
    });

    if (use_grid_weight) {
        std::vector<Eigen::MatrixXd>().swap(grid_weight);
    }
}

void MapBuffer::calculate_stokes(std::vector<Eigen::MatrixXd>& map_vec, const Eigen::MatrixXd& m, Eigen::Index i, Eigen::Index j,
                                 int index, int step) {
    Eigen::VectorXd d(3);
    d(0) = map_vec[index](i, j);
    d(1) = map_vec[index + step](i, j);
    d(2) = map_vec[index + 2 * step](i, j);
    Eigen::VectorXd v = m.ldlt().solve(d);//m.colPivHouseholderQr().solve(d);
    map_vec[index](i, j) = v(0);
    map_vec[index + step](i, j) = v(1);
    map_vec[index + 2 * step](i, j) = v(2);
}

void MapBuffer::calculate_stokes(std::vector<Eigen::Tensor<double,3>>& map_vec, const Eigen::MatrixXd& m, Eigen::Index i,
                                 Eigen::Index j, int index, int step) {
    Eigen::VectorXd d(3);
    for (Eigen::Index n = 0; n < n_noise; ++n) {
        d(0) = map_vec[index](i, j, n);
        d(1) = map_vec[index + step](i, j, n);
        d(2) = map_vec[index + 2 * step](i, j, n);
        Eigen::VectorXd v = m.colPivHouseholderQr().solve(d);
        map_vec[index](i, j, n) = v(0);
        map_vec[index + step](i, j, n) = v(1);
        map_vec[index + 2 * step](i, j, n) = v(2);
    }
}

void MapBuffer::process_maps_for_pixel(Eigen::Index i, Eigen::Index j, int a, int step, const Eigen::MatrixXd& m) {
    calculate_stokes(signal, m, i, j, a, step);
    if (!kernel.empty()) {
        calculate_stokes(kernel, m, i, j, a, step);
    }
    if (!noise.empty()) {
        calculate_stokes(noise, m, i, j, a, step);
    }
}

void MapBuffer::zero_out_maps(Eigen::Index i, Eigen::Index j, int index, int step) {
    signal[index](i, j) = 0;
    signal[index + step](i, j) = 0;
    signal[index + 2 * step](i, j) = 0;
    if (!kernel.empty()) {
        kernel[index](i, j) = 0;
        kernel[index + step](i, j) = 0;
        kernel[index + 2 * step](i, j) = 0;
    }
    if (!noise.empty()) {
        for (Eigen::Index n = 0; n < n_noise; ++n) {
            noise[index](i, j, n) = 0;
            noise[index + step](i, j, n) = 0;
            noise[index + 2 * step](i, j, n) = 0;
        }
    }
}

void MapBuffer::normalize_polarized_maps() {
    int step = pointing.size();
    for (Eigen::Index index = 0; index < pointing.size(); ++index) {
        Eigen::MatrixXd m(3, 3);
        for (Eigen::Index i = 0; i < n_rows; ++i) {
            for (Eigen::Index j = 0; j < n_cols; ++j) {
                int pix = n_rows * j + i;
                Eigen::VectorXd temp = pointing[index].row(pix);
                m = Eigen::Map<Eigen::MatrixXd>(temp.data(), 3, 3);
                Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(m);
                if ((m.array() != 0).all() && lu_decomp.isInvertible()) {
                    process_maps_for_pixel(i, j, index, step, m);
                }
                else {
                    zero_out_maps(i, j, index, step);
                }
            }
        }
        weight[index + step] = weight[index];
        weight[index + 2 * step] = weight[index];

        coverage[index + step] = coverage[index];
        coverage[index + 2 * step] = coverage[index];
    }

    if (!grid_weight.empty()) {
        std::vector<Eigen::MatrixXd>().swap(grid_weight);
    }
}

std::tuple<double, Eigen::MatrixXd, Eigen::Index, Eigen::Index> MapBuffer::calc_cov_region(Eigen::Index i) {
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
void MapBuffer::calc_map_psd() {
    // clear psd vectors
    std::vector<Eigen::VectorXd>().swap(psds);
    std::vector<Eigen::VectorXd>().swap(psd_freqs);
    std::vector<Eigen::MatrixXd>().swap(psd_2ds);
    std::vector<Eigen::MatrixXd>().swap(psd_2d_freqs);

    // clear noise psd vectors
    std::vector<Eigen::VectorXd>().swap(noise_psds);
    std::vector<Eigen::VectorXd>().swap(noise_psd_freqs);
    std::vector<Eigen::MatrixXd>().swap(noise_psd_2ds);
    std::vector<Eigen::MatrixXd>().swap(noise_psd_2d_freqs);

    // loop through maps
    for (Eigen::Index i=0; i<signal.size(); ++i) {
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
            for (Eigen::Index j=0; j<n_noise; ++j) {
                // get noise map
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(noise[i].data() + j * n_rows * n_cols,
                                                                                               n_rows, n_cols);
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

void MapBuffer::calc_map_hist() {
    // clear vectors
    std::vector<Eigen::VectorXd>().swap(hists);
    std::vector<Eigen::VectorXd>().swap(hist_bins);
    std::vector<Eigen::VectorXd>().swap(noise_hists);
    std::vector<Eigen::VectorXd>().swap(noise_hist_bins);

    // loop through maps
    for (Eigen::Index i=0; i<signal.size(); ++i) {
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
            for (Eigen::Index j=0; j<n_noise; ++j) {
                // get noise map
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(noise[i].data() + j * n_rows * n_cols,
                                                                                               n_rows, n_cols);
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

void MapBuffer::calc_median_err() {
    // resize mean errors
    median_err.setZero(weight.size());
    for (Eigen::Index i=0; i<weight.size(); ++i) {
        // calculate weight threshold
	if (weight[i].maxCoeff() == weight[i].minCoeff()) {
		median_err(i) = 0;
	}
	else {
        	auto [weight_threshold, cov_ranges, cov_n_rows, cov_n_cols] = calc_cov_region(i);
        	Eigen::MatrixXd mean_sqerr = ((weight[i].array()>=weight_threshold).select(1/weight[i].array(),0));
        	// construct a sparse matrix
        	Eigen::SparseMatrix<double> sparse_err(mean_sqerr.sparseView());
        	// construct a dense map
        	Eigen::Map<Eigen::VectorXd> dense_map(sparse_err.valuePtr(), sparse_err.nonZeros());
        	// construct a dense vector
        	Eigen::VectorXd dense_vector(dense_map);
        	// get mean square error
        	median_err(i) = tula::alg::median(dense_vector);
	}
    }
}

void MapBuffer::calc_median_rms() {
    // average filtered rms vector
    median_rms.setZero(noise.size());

    // loop through arrays/polarizations
    for (Eigen::Index i=0; i<noise.size(); ++i) {
        const double weight_threshold = std::get<0>(calc_cov_region(i));
        const auto valid_mask = (weight[i].array()>=weight_threshold);
        const int counter = valid_mask.count();
        if (counter <= 0) {
            median_rms(i) = 0.0;
            continue;
        }

        // vector of rms of noise maps
        Eigen::VectorXd noise_rms(n_noise);
        for (Eigen::Index j=0; j<n_noise; ++j) {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(noise[i].data() + j * n_rows * n_cols,
                                                                                           n_rows, n_cols);
            double rms = (valid_mask.select(noise_matrix.array().square(), 0.0)).sum();

            noise_rms(j) = sqrt(rms/counter);
        }
        // get mean rms
        median_rms(i) = tula::alg::median(noise_rms);
    }
}

void MapBuffer::calc_median_rms_annulus(double inner_radius_rad, double outer_radius_rad) {
    // average filtered rms vector
    median_rms.setZero(weight.size());

    // distance to each pixel
    Eigen::MatrixXd dist(n_rows,n_cols);

    // calculate distance to each pixel from center (same for all maps)
    for (Eigen::Index i=0; i<n_rows; ++i) {
        for (Eigen::Index j=0; j<n_cols; ++j) {
            dist(i,j) = sqrt(pow(rows_tan_vec(i),2) + pow(cols_tan_vec(j),2));
        }
    }

    // loop through maps
    for (Eigen::Index i=0; i<signal.size(); ++i) {
        int n_pts = 0;
        // loop through pixels
        for (Eigen::Index j=0; j<n_rows; ++j) {
            for (Eigen::Index k=0; k<n_cols; ++k) {
                if (dist(j,k) > inner_radius_rad && dist(j,k) <= outer_radius_rad) {
                    n_pts++;
                    median_rms(i) += signal[i](j,k);
                }
            }
        }
        // get mean
        median_rms(i) /= n_pts;
    }
}

bool MapBuffer::find_sources(Eigen::Index map_index) {
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
        for (Eigen::Index i=0; i<n_rows; ++i) {
            for (Eigen::Index j=0; j<n_cols; ++j) {
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
        for (Eigen::Index i=0; i<n_rows; ++i) {
            for (Eigen::Index j=0; j<n_cols; ++j) {
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
    for (unsigned int i=0; i<row_index.size(); ++i) {
        double extremum;
        if (source_finder_mode=="both" && signal[map_index](row_index[i],col_index[i]) < 0.0) {
            extremum = signal[map_index](row_index[i],col_index[i]);
            // find minimum within index box
            for (Eigen::Index j=row_index[i]-1; j<row_index[i]+2; ++j) {
                for (Eigen::Index k=col_index[i]-1; k<col_index[i]+2; ++k) {
                    if (signal[map_index](j,k) < extremum) {
                        extremum = signal[map_index](j,k);
                    }
                }
            }
        }
        else {
            extremum = signal[map_index](row_index[i],col_index[i]);
            // find maximum within index box
            for (Eigen::Index j=row_index[i]-1; j<row_index[i]+2; ++j) {
                for (Eigen::Index k=col_index[i]-1; k<col_index[i]+2; ++k) {
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

    for (Eigen::Index i=0; i<n_raw_sources; ++i) {
        for (Eigen::Index j=0; j<n_raw_sources; ++j) {
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
        for (unsigned int i=0; i<row_dist_index.size(); ++i) {
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
    for (Eigen::Index i=0; i<n_raw_sources; ++i) {
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
