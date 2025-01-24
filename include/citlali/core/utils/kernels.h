# pragma once


template <typename Derived,>
auto calc_dist_2d(Eigen::DenseBase<Derived> &rows, Eigen::DenseBase<Derived> &cols,
                   const double row_offset, const double col_offset) {

    int n_rows = rows.size();
    int n_cols = cols.size();

    Eigen::MatrixXd dist = ((rows - row_offset).array().square().replicate(1, n_cols - 1) +
                            (cols - col_offset).array().square().transpose().replicate(n_rows - 1, 1)).sqrt();

    return dist;
}

template <typename Derived>
void gaussian_template_2d(const Eigen::DenseBase<Derived> &rows, const Eigen::DenseBase<Derived> &cols,
                          const double fwhm_rad) {
    // assuming same x and y pixels
    double pix_size_rad = rows(1) - cols(0);
    double offset = -0.5 *  pix_size_rad;

    // distance to each pixel in radians
    auto dist = calc_dist_2d(rows, cols, offset, offset);
    // to hold minimum distance
    Eigen::Index row_index, col_index;
    // minimum distance
    double min_dist = dist.minCoeff(&row_index, &col_index);

    // calculate template
    filter_template = exp(-0.5 * pow(dist.array() / fwhm_rad * FWHM_TO_SIGMA, 2.));
    // shift template
    filter_template = shift(filter_template, row_indices, col_indices);

    return filter_template;
}

template <typename Derived>
void airy_template_2d(const Eigen::DenseBase<Derived> &rows, const Eigen::DenseBase<Derived> &cols,
                      const double fwhm_rad) {

    int n_rows = rows.size();
    int n_cols = cols.size();

    // assuming same x and y pixels
    double pix_size_rad = rows(1) - cols(0);
    double offset = -0.5 *  pix_size_rad;

    // distance to each pixel in radians
    auto dist = calc_dist_2d(rows, cols, offset, offset);
    // to hold minimum distance
    Eigen::Index row_index, col_index;
    // minimum distance
    double min_dist = dist.minCoeff(&row_index, &col_index);

    Eigen::MatrixXd filter_template(n_rows, n_cols);

    double airy_scale = pi * 1.028 / fwhm_rad;

    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            if (dist(i, j) == 0) {
                filter_template(i, j) = 1.0;
            } else {
                double kr = airy_scale * dist(i, j);
                return amplitude * std::pow(2.0 * boost::math::cyl_bessel_j(1, kr) / kr, 2);
            }
        }
    }

    // calculate template
    filter_template = exp(-0.5 * pow(dist.array() / fwhm_rad * FWHM_TO_SIGMA, 2.));
    // shift template
    filter_template = shift(filter_template, row_indices, col_indices);

    return filter_template;
}

template <typename DerivedA, typename DerivedB>
auto symmetric_kernel(const Eigen::DenseBase<DerivedA> &kernel, const Eigen::DenseBase<DerivedA> &weight,
                      const Eigen::DenseBase<DerivedB> &rows, const Eigen::DenseBase<DerivedB> &cols,
                      const double amp_lower_factor, const double amp_upper_factor,
                      const double fwhm_lower_factor, const double fwhm_upper_factor,
                      const double init_fwhm, const bool fit_theta) {

    int n_rows = rows.size();
    int n_cols = cols.size();
    // assuming same x and y pixels
    double pix_size_rad = rows(1) - cols(0);
    double offset = -0.5 *  pix_size_rad;

    // fit source
    auto [params, uncertainties] = fit_to_gaussian(kernel, weight, amp_lower_factor, amp_upper_factor,
                                                   fwhm_lower_factor, fwhm_upper_factor, init_fwhm, fit_theta);

    double x = params(1) * pix_size_rad - n_cols / 2;
    double y = params(2) * pix_size_rad - n_rows / 2;

    int shift_row = -std::round(y / pix_size_rad);
    int shift_col = -std::round(x / pix_size_rad);

    Eigen::MatrixXd temp_kernel = kernel;
    temp_kernel = shift(temp_kernel, shift_row, shift_col);

    // distance to each pixel in radians
    auto dist = calc_dist_2d(rows, cols, offset, offset);
    // to hold minimum distance
    Eigen::Index row_index, col_index;
    // minimum distance
    double min_dist = dist.minCoeff(&row_index, &col_index);

    // number of bins
    int n_bins = static_cast<int>(dist.maxCoeff() / bin_size) + 1;

    // left bin edges
    Eigen::VectorXd bin_edges = Eigen::VectorXd::LinSpaced(n_bins, 0, n_bins - 1) * pix_size_rad;
    Eigen::VectorXd binned_kernel(n_bins), binned_dist(n_bins);

    for (int bin = 0; bin < n_bins; ++bins) {
        int n_pts = 0;
        for (int row = 0; row < n_rows; ++row) {
            for (int col = 0; col < n_cols; ++col) {
                if (dist(row, col) >= bin_edges(bin) && dist(row, col) < bin_edges(i + 1)){
                    binned_kernel(bin) += temp_kernel(row, col);
                    binned_dist(bin) += dist(row, col);
                    n_pts++;
                }
            }
        }
        binned_kernel /= n_pts;
        binned_dist /= n_pts;
    }

    // get spline function
    SplineFunction spline(binned_dist, binned_kernel);

    for (int row = 0; row < n_rows; ++row) {
        for (int col = 0; col < n_cols; ++col) {
             if (dist(j,i) <= spline.x_max && dist(j,i) >= spline.x_min) {
                filter_template(row, col) = spline(dist(row, col));
             } else if (dist(row, col) > spline.x_max) {
                 filter_template(row, col) = kernel_interp(n_bins -1);
             } else {
                 filter_template(row, col) = kernel_interp(0);
             }
        }
    }

    filter_template = shift(filter_template, row_indices, col_indices);

    return filter_template;
}
