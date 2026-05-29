#pragma once

#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <boost/math/special_functions/bessel.hpp>

#include <unsupported/Eigen/Splines>
#include <Eigen/Sparse>

#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/utils.h>

#include <citlali/core/timestream/timestream.h>

#include <citlali/core/mapmaking/map.h>
#include <citlali/core/utils/pointing.h>


using timestream::TCData;

// selects the type of TCData
using timestream::TCDataKind;

namespace mapmaking {

struct JincDebugBreadcrumb {
    bool valid = false;
    const char *stage = "unset";
    long long det_col = -1;
    int det_uid = -1;
    long long sample = -1;
    long long map_index = -1;
    long long array_index = -1;
    int pixel_row = -1;
    int pixel_col = -1;
    int subpix_idx = -1;
    int lower_row = -1;
    int upper_row = -1;
    int lower_col = -1;
    int upper_col = -1;
    int jinc_lower_row = -1;
    int jinc_lower_col = -1;
    int size_rows = -1;
    int size_cols = -1;
};

inline thread_local JincDebugBreadcrumb jinc_debug_breadcrumb{};

inline void reset_jinc_debug_breadcrumb() {
    jinc_debug_breadcrumb = {};
}

inline void update_jinc_debug_breadcrumb(const char *stage,
                                         long long det_col,
                                         int det_uid,
                                         long long sample,
                                         long long map_index,
                                         long long array_index,
                                         int pixel_row,
                                         int pixel_col,
                                         int subpix_idx) {
    jinc_debug_breadcrumb.valid = true;
    jinc_debug_breadcrumb.stage = stage;
    jinc_debug_breadcrumb.det_col = det_col;
    jinc_debug_breadcrumb.det_uid = det_uid;
    jinc_debug_breadcrumb.sample = sample;
    jinc_debug_breadcrumb.map_index = map_index;
    jinc_debug_breadcrumb.array_index = array_index;
    jinc_debug_breadcrumb.pixel_row = pixel_row;
    jinc_debug_breadcrumb.pixel_col = pixel_col;
    jinc_debug_breadcrumb.subpix_idx = subpix_idx;
}

inline void update_jinc_debug_breadcrumb_block(const char *stage,
                                               int lower_row,
                                               int upper_row,
                                               int lower_col,
                                               int upper_col,
                                               int jinc_lower_row,
                                               int jinc_lower_col,
                                               int size_rows,
                                               int size_cols) {
    jinc_debug_breadcrumb.valid = true;
    jinc_debug_breadcrumb.stage = stage;
    jinc_debug_breadcrumb.lower_row = lower_row;
    jinc_debug_breadcrumb.upper_row = upper_row;
    jinc_debug_breadcrumb.lower_col = lower_col;
    jinc_debug_breadcrumb.upper_col = upper_col;
    jinc_debug_breadcrumb.jinc_lower_row = jinc_lower_row;
    jinc_debug_breadcrumb.jinc_lower_col = jinc_lower_col;
    jinc_debug_breadcrumb.size_rows = size_rows;
    jinc_debug_breadcrumb.size_cols = size_cols;
}

inline const JincDebugBreadcrumb &get_jinc_debug_breadcrumb() {
    return jinc_debug_breadcrumb;
}

class JincMapmaker {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");
    std::unique_ptr<std::mutex> jinc_mutex = std::make_unique<std::mutex>();

    // toltec array mounting angle
    std::map<int, double> install_ang = {
        {-1,-1},
        {0,pi/2},
        {1,-pi/2},
        {2,-pi/2},
        };

    // toltec detector orientation angles
    std::map<int, double> fgs = {
        {-1,-1},
        {0,0},
        {1,pi/4},
        {2,pi/2},
        {3,3*pi/4}
    };

    // run polarization?
    bool run_polarization;

    // parallel policy
    std::string parallel_policy;

    // method to calculate jinc weights
    std::string mode = "matrix";

    // lambda over diameter
    std::map<Eigen::Index,double> l_d;

    // maximum radius
    double r_max;

    // sub-pixel kernel sampling (1 = no sub-pixel shift)
    int subpixel_n = 1;

    // number of points for spline
    int n_pts_splines = 1000;

    // jinc filter shape parameters
    std::map<Eigen::Index,Eigen::VectorXd> shape_params;

    // matrices to hold precomputed jinc function
    std::map<Eigen::Index,Eigen::MatrixXd> jinc_weights_mat;
    // squared jinc weights for variance and coverage updates
    std::map<Eigen::Index,Eigen::MatrixXd> jinc_weights_sq_mat;
    // sub-pixel shifted jinc matrices (size = subpixel_n^2 per array)
    std::map<Eigen::Index,std::vector<Eigen::MatrixXd>> jinc_weights_mat_subpix;
    // squared sub-pixel shifted jinc matrices for variance and coverage updates
    std::map<Eigen::Index,std::vector<Eigen::MatrixXd>> jinc_weights_sq_mat_subpix;

    // splines for jinc function
    std::map<Eigen::Index, engine_utils::SplineFunction> jinc_splines;

    // calculate jinc weight at a given radius
    auto jinc_func(double, double, double, double, double, double);

    // precompute jinc weight matrix
    void allocate_jinc_matrix(double);

    // calculate spline function for jinc weights
    void calculate_jinc_splines();

    // allocate pointing matrix for polarization reduction
    template <class map_buffer_t>
    void allocate_pointing(map_buffer_t &, double, double, double, Eigen::Index, int, int);

    // populate maps with a time chunk (signal, kernel, coverage, and noise)
    template<class map_buffer_t, typename Derived, typename apt_t>
    void populate_maps_jinc(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, map_buffer_t &, map_buffer_t &,
                            Eigen::DenseBase<Derived> &, std::string &, apt_t &, double, bool, bool,
                            const Eigen::Matrix<bool, Eigen::Dynamic, 1> *active_maps = nullptr);

    // populate maps with a time chunk (signal, kernel, coverage, and noise)
    template<class map_buffer_t, typename Derived, typename apt_t>
    void populate_maps_jinc_parallel(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, map_buffer_t &, map_buffer_t &,
                                     Eigen::DenseBase<Derived> &, std::string &, apt_t &, double, bool, bool,
                                     const Eigen::Matrix<bool, Eigen::Dynamic, 1> *active_maps = nullptr);
};

auto JincMapmaker::jinc_func(double r, double a, double b, double c, double r_max, double l_d) {
    if (r!=0) {
        // unitless radius
        r = r/l_d;
        // first jinc function
        auto jinc_1 = 2.*boost::math::cyl_bessel_j(1,2.*pi*r/a)/(2.*pi*r/a);
        // exponential
        auto exp_func = exp(-pow(2.*r/b,c));
        // second jinc function
        auto jinc_2 = 2.*boost::math::cyl_bessel_j(1,3.831706*r/r_max)/(3.831706*r/r_max);
        // jinc1 x exp x jinc2
        return jinc_1*exp_func*jinc_2;
    }
    else {
        return 1.0;
    }
}

void JincMapmaker::allocate_jinc_matrix(double pixel_size_rad) {
    l_d[0] = (1.1/1000)/45;
    l_d[1] = (1.4/1000)/45;
    l_d[2] = (2.0/1000)/45;

    subpixel_n = std::max(1, subpixel_n);

    jinc_weights_mat.clear();
    jinc_weights_sq_mat.clear();
    jinc_weights_mat_subpix.clear();
    jinc_weights_sq_mat_subpix.clear();

    std::vector<double> subpixel_offsets;
    if (subpixel_n > 1) {
        subpixel_offsets.resize(subpixel_n);
        for (int i = 0; i < subpixel_n; ++i) {
            subpixel_offsets[i] = -0.5 + (static_cast<double>(i) + 0.5) / static_cast<double>(subpixel_n);
        }
    }

    // loop through lambda/diameters
    for (const auto &ld: l_d) {
        // get shape params
        auto a = shape_params[ld.first](0);
        auto b = shape_params[ld.first](1);
        auto c = shape_params[ld.first](2);

        // maximum radius in pixels
        int r_max_pix = std::floor(r_max*ld.second/pixel_size_rad);

        // pixel centers within max radius
        Eigen::VectorXd pixels = Eigen::VectorXd::LinSpaced(2*r_max_pix + 1,-r_max_pix, r_max_pix);

        // allocate jinc weights
        jinc_weights_mat[ld.first].setZero(2*r_max_pix + 1,2*r_max_pix + 1);
        jinc_weights_sq_mat[ld.first].setZero(2*r_max_pix + 1,2*r_max_pix + 1);

        // loop through matrix rows
        for (Eigen::Index i=0; i<pixels.size(); ++i) {
            // loop through matrix cols
            for (Eigen::Index j=0; j<pixels.size(); ++j) {
                // radius of current pixel in radians
                double r = pixel_size_rad*sqrt(pow(pixels(i),2) + pow(pixels(j),2));
                // calculate jinc weight at pixel
                auto w = jinc_func(r,a,b,c,r_max,ld.second);
                jinc_weights_mat[ld.first](i,j) = w;
                jinc_weights_sq_mat[ld.first](i,j) = w*w;
            }
        }

        if (subpixel_n > 1) {
            auto &subpix_vec = jinc_weights_mat_subpix[ld.first];
            auto &subpix_sq_vec = jinc_weights_sq_mat_subpix[ld.first];
            subpix_vec.resize(subpixel_n * subpixel_n);
            subpix_sq_vec.resize(subpixel_n * subpixel_n);
            for (int sr = 0; sr < subpixel_n; ++sr) {
                for (int sc = 0; sc < subpixel_n; ++sc) {
                    double drow = subpixel_offsets[sr];
                    double dcol = subpixel_offsets[sc];
                    auto mat_index = static_cast<size_t>(sr * subpixel_n + sc);
                    auto &mat = subpix_vec[mat_index];
                    auto &mat_sq = subpix_sq_vec[mat_index];
                    mat.setZero(2 * r_max_pix + 1, 2 * r_max_pix + 1);
                    mat_sq.setZero(2 * r_max_pix + 1, 2 * r_max_pix + 1);

                    for (Eigen::Index i=0; i<pixels.size(); ++i) {
                        for (Eigen::Index j=0; j<pixels.size(); ++j) {
                            double r = pixel_size_rad*sqrt(pow(pixels(i) - drow,2) + pow(pixels(j) - dcol,2));
                            auto w = jinc_func(r,a,b,c,r_max,ld.second);
                            mat(i,j) = w;
                            mat_sq(i,j) = w*w;
                        }
                    }
                }
            }
        }
    }
}

void JincMapmaker::calculate_jinc_splines() {
    l_d[0] = (1.1/1000)/45;
    l_d[1] = (1.4/1000)/45;
    l_d[2] = (2.0/1000)/45;

    // loop through lambda/diameters
    for (const auto &ld: l_d) {
        // get shape params
        auto a = shape_params[ld.first](0);
        auto b = shape_params[ld.first](1);
        auto c = shape_params[ld.first](2);

        // radius vector in radians
        auto radius = Eigen::VectorXd::LinSpaced(n_pts_splines, 0, r_max*ld.second);
        // jinc weights on dense vector
        Eigen::VectorXd jinc_weights(radius.size());

        Eigen::Index j = 0;

        for (const auto &r: radius) {
            // calculate jinc weights
            jinc_weights(j) = jinc_func(r,a,b,c,r_max,ld.second);
            ++j;
        }
        // create spline class
        engine_utils::SplineFunction s;
        // spline interpolate
        s.interpolate(radius, jinc_weights);
        // store jinc spline
        jinc_splines[ld.first] = s;
    }
}

template <class map_buffer_t>
void JincMapmaker::allocate_pointing(map_buffer_t &mb, double weight, double cos_2angle, double sin_2angle,
                                     Eigen::Index map_index, int ir, int ic) {
    int pix = mb.n_rows * ic + ir;
    Eigen::MatrixXd& matrix = mb.pointing[map_index];

    // calculate reused expressions
    double weight_cos_2angle = weight * cos_2angle;
    double weight_sin_2angle = weight * sin_2angle;
    double weight_cos2_2angle = weight * std::pow(cos_2angle, 2);
    double weight_sin2_2angle = weight * std::pow(sin_2angle, 2);
    double weight_cos_sin_2angle = weight * cos_2angle * sin_2angle;

    // update pointing matrix
    matrix(pix, 0) += weight;
    matrix(pix, 1) += weight_cos_2angle;
    matrix(pix, 2) += weight_sin_2angle;
    matrix(pix, 3) = matrix(pix, 1);  // previously set to += weight*cos_2angle, then directly assigned here
    matrix(pix, 4) += weight_cos2_2angle;
    matrix(pix, 5) += weight_cos_sin_2angle;
    matrix(pix, 6) = matrix(pix, 2);  // previously set to += weight*sin_2angle, then directly assigned here
    matrix(pix, 7) = matrix(pix, 5);  // reuse the result from matrix(pix,5)
    matrix(pix, 8) += weight_sin2_2angle;
}

template<class map_buffer_t, typename Derived, typename apt_t>
void JincMapmaker::populate_maps_jinc(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in,
                        map_buffer_t &omb, map_buffer_t &cmb, Eigen::DenseBase<Derived> &map_indices,
                        std::string &pixel_axes, apt_t &apt, double d_fsmp, bool run_omb, bool run_noise,
                        const Eigen::Matrix<bool, Eigen::Dynamic, 1> *active_maps) {

    const bool use_omb = !omb.noise.empty();
    const bool use_cmb = !cmb.noise.empty() && !use_omb;
    const bool run_kernel = !omb.kernel.empty();
    const bool run_coverage = !omb.coverage.empty();
    const bool run_hwpr = in.hwpr_angle.data.size()!=0;

    // dimensions of data
    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    // pointer to map buffer with noise maps
    map_buffer_t *nmb = nullptr;
    if (run_noise) {
        nmb = use_cmb ? &cmb : (use_omb ? &omb : nullptr);
    }

    double noise_cube_gb = 0.0;
    if (run_noise && nmb != nullptr) {
        noise_cube_gb = 8.0 * static_cast<double>(nmb->n_rows) * static_cast<double>(nmb->n_cols) *
                        static_cast<double>(nmb->noise.size()) * static_cast<double>(nmb->n_noise) / 1e9;
    }
    constexpr double direct_noise_accum_threshold_gb = 16.0;
    const bool direct_noise_accum =
        run_noise && nmb != nullptr && noise_cube_gb >= direct_noise_accum_threshold_gb;
    if (direct_noise_accum && logger) {
        logger->info("jinc noise cube is {} GB; using locked direct accumulation instead of full scratch copy",
                     static_cast<float>(noise_cube_gb));
    }

    struct TouchBounds {
        int row_min;
        int row_max;
        int col_min;
        int col_max;
        bool touched;

        TouchBounds(Eigen::Index n_rows, Eigen::Index n_cols)
            : row_min(static_cast<int>(n_rows)),
              row_max(-1),
              col_min(static_cast<int>(n_cols)),
              col_max(-1),
              touched(false) {}

        void update(int lower_row, int upper_row, int lower_col, int upper_col) {
            touched = true;
            row_min = std::min(row_min, lower_row);
            row_max = std::max(row_max, upper_row);
            col_min = std::min(col_min, lower_col);
            col_max = std::max(col_max, upper_col);
        }
    };

    struct ScratchBuffers {
        map_buffer_t omb_copy;
        map_buffer_t cmb_copy;
    };
    thread_local ScratchBuffers scratch;

    auto ensure_zero_matrix_vec = [](std::vector<Eigen::MatrixXd> &vec, Eigen::Index size, Eigen::Index n_rows,
                                     Eigen::Index n_cols) {
        if (static_cast<Eigen::Index>(vec.size()) != size) {
            vec.resize(static_cast<size_t>(size));
        }
        for (Eigen::Index ii = 0; ii < size; ++ii) {
            auto &m = vec[static_cast<size_t>(ii)];
            if (m.rows() != n_rows || m.cols() != n_cols) {
                m.resize(n_rows, n_cols);
            }
            m.setZero();
        }
    };

    auto ensure_zero_noise_vec = [](std::vector<Eigen::Tensor<double, 3>> &dst,
                                    const std::vector<Eigen::Tensor<double, 3>> &src) {
        if (dst.size() != src.size()) {
            dst.resize(src.size());
        }
        for (size_t ii = 0; ii < src.size(); ++ii) {
            const auto &ref = src[ii];
            auto &out = dst[ii];
            if (out.dimension(0) != ref.dimension(0) ||
                out.dimension(1) != ref.dimension(1) ||
                out.dimension(2) != ref.dimension(2)) {
                out.resize(ref.dimension(0), ref.dimension(1), ref.dimension(2));
            }
            out.setZero();
        }
    };

    auto &omb_copy = scratch.omb_copy;
    omb_copy.n_rows = omb.n_rows;
    omb_copy.n_cols = omb.n_cols;

    if (run_omb) {
        ensure_zero_matrix_vec(omb_copy.signal, static_cast<Eigen::Index>(omb.signal.size()), omb.n_rows, omb.n_cols);
        ensure_zero_matrix_vec(omb_copy.weight, static_cast<Eigen::Index>(omb.weight.size()), omb.n_rows, omb.n_cols);
        if (!omb.grid_weight.empty()) {
            ensure_zero_matrix_vec(omb_copy.grid_weight, static_cast<Eigen::Index>(omb.grid_weight.size()),
                                   omb.n_rows, omb.n_cols);
        }
        if (run_coverage) {
            ensure_zero_matrix_vec(omb_copy.coverage, static_cast<Eigen::Index>(omb.coverage.size()), omb.n_rows, omb.n_cols);
        }
        if (run_kernel) {
            ensure_zero_matrix_vec(omb_copy.kernel, static_cast<Eigen::Index>(omb.kernel.size()), omb.n_rows, omb.n_cols);
        }
    }
    if (use_omb && !direct_noise_accum) {
        ensure_zero_noise_vec(omb_copy.noise, omb.noise);
    }

    auto &cmb_copy = scratch.cmb_copy;
    if (use_cmb) {
        cmb_copy.n_rows = cmb.n_rows;
        cmb_copy.n_cols = cmb.n_cols;
        if (!direct_noise_accum) {
            ensure_zero_noise_vec(cmb_copy.noise, cmb.noise);
        }
    }
    if (direct_noise_accum) {
        std::vector<Eigen::Tensor<double, 3>>().swap(omb_copy.noise);
        std::vector<Eigen::Tensor<double, 3>>().swap(cmb_copy.noise);
    }

    map_buffer_t *nmb_copy = nullptr;

    if (run_noise && !direct_noise_accum) {
        nmb_copy = use_cmb ? &cmb_copy : (use_omb ? &omb_copy : nullptr);
    }

    std::vector<TouchBounds> omb_bounds;
    if (run_omb) {
        omb_bounds.reserve(omb.signal.size());
        for (Eigen::Index ii = 0; ii < static_cast<Eigen::Index>(omb.signal.size()); ++ii) {
            omb_bounds.emplace_back(omb.n_rows, omb.n_cols);
        }
    }

    std::vector<TouchBounds> nmb_bounds;
    if (run_noise && !direct_noise_accum && nmb != nullptr && nmb_copy != nullptr) {
        nmb_bounds.reserve(nmb->noise.size());
        for (Eigen::Index ii = 0; ii < static_cast<Eigen::Index>(nmb->noise.size()); ++ii) {
            nmb_bounds.emplace_back(nmb->n_rows, nmb->n_cols);
        }
    }

    // parallelize over detectors
    for (Eigen::Index i=0; i<n_dets; ++i) {
        // skip fg = -1 if in polarization mode
        const bool run_det = !(run_polarization && apt["fg"](i)==-1);
        if (!run_det) {
            continue;
        }

        // skip completely flagged detectors
        if (apt["flag"](i)==0 && (in.flags.data.col(i).array()==false).any()) {
            // get detector positions from apt table if not in detector mapmaking mode
            auto det_index = i;

            // which map to assign detector to
            Eigen::Index map_index = map_indices(i);
            if (active_maps != nullptr &&
                (map_index < 0 || map_index >= active_maps->size() || !(*active_maps)(map_index))) {
                continue;
            }
            Eigen::Index array_index = apt["array"](det_index);
            const bool use_subpix = (subpixel_n > 1) && (jinc_weights_mat_subpix.count(array_index) > 0);
            const auto *subpix_vec = use_subpix ? &jinc_weights_mat_subpix.at(array_index) : nullptr;
            const auto *subpix_sq_vec = use_subpix ? &jinc_weights_sq_mat_subpix.at(array_index) : nullptr;
            Eigen::Index mat_rows = jinc_weights_mat[array_index].rows();
            Eigen::Index mat_cols = jinc_weights_mat[array_index].cols();
            Eigen::Index mat_rows_center = (mat_rows - 1.)/2.;
            Eigen::Index mat_cols_center = (mat_cols - 1.)/2.;

            // get detector pointing
            auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, apt["x_t"](det_index), apt["y_t"](det_index),
                                                              pixel_axes, in.pointing_offsets_arcsec.data, omb.map_grouping);

            // get map buffer row and col indices for lat and lon vectors
            Eigen::VectorXd omb_irow = lat.array()/omb.pixel_size_rad + (omb.n_rows - 1)/2.;
            Eigen::VectorXd omb_icol = lon.array()/omb.pixel_size_rad + (omb.n_cols - 1)/2.;

            Eigen::VectorXd cmb_irow, cmb_icol;
            if (use_cmb) {
                // get coadded map buffer row and col indices for lat and lon vectors
                cmb_irow = lat.array()/cmb.pixel_size_rad + (cmb.n_rows - 1)/2.;
                cmb_icol = lon.array()/cmb.pixel_size_rad + (cmb.n_cols - 1)/2.;
            }

            // signal map value
            double signal;

            // noise map value
            double noise_v;

            // noise map indices
            Eigen::Index nmb_ir, nmb_ic;

            // cosine and sine of angles
            double angle, cos_2angle, sin_2angle;

            // loop through the samples
            for (Eigen::Index j=0; j<n_pts; ++j) {
                // check if sample is flagged, ignore if so
                if (!in.flags.data(j,i) &&
                    std::isfinite(in.scans.data(j,i)) &&
                    i < in.weights.data.size() &&
                    std::isfinite(in.weights.data(i)) &&
                    in.weights.data(i) > 0.0) {
                    Eigen::Index omb_ir = static_cast<Eigen::Index>(std::llround(omb_irow(j)));
                    Eigen::Index omb_ic = static_cast<Eigen::Index>(std::llround(omb_icol(j)));
                    int subpix_idx = 0;
                    if (use_subpix) {
                        auto subpix_index = [&](double d) {
                            int idx = static_cast<int>(std::floor((d + 0.5) * subpixel_n));
                            if (idx < 0) {
                                idx = 0;
                            }
                            else if (idx >= subpixel_n) {
                                idx = subpixel_n - 1;
                            }
                            return idx;
                        };
                        double drow = omb_irow(j) - static_cast<double>(omb_ir);
                        double dcol = omb_icol(j) - static_cast<double>(omb_ic);
                        int sr = subpix_index(drow);
                        int sc = subpix_index(dcol);
                        subpix_idx = sr * subpixel_n + sc;
                    }

                    if (run_polarization) {
                        auto fg_index = apt["fg"](det_index);
                        if (run_hwpr) {
                            angle = 2*in.hwpr_angle.data(j) - (in.angle.data(j) + fgs[fg_index] + install_ang[array_index]);
                        }
                        else {
                            angle = in.angle.data(j) + fgs[fg_index] + install_ang[array_index];
                        }

                        cos_2angle = cos(2.*angle);
                        sin_2angle = sin(2.*angle);
                    }

                    if (run_omb) {
                        // make sure the data point is within the map
                        if ((omb_ir >= 0) && (omb_ir < omb.n_rows) && (omb_ic >= 0) && (omb_ic < omb.n_cols)) {

                            int lower_row = omb_ir - mat_rows_center;
                            int upper_row = omb_ir + mat_rows - 1 - mat_rows_center;
                            int lower_col = omb_ic - mat_cols_center;
                            int upper_col = omb_ic + mat_cols - 1 - mat_cols_center;

                            int jinc_lower_row = abs(std::min(0, lower_row));
                            int jinc_lower_col = abs(std::min(0, lower_col));

                            lower_row = std::max(0,lower_row);
                            upper_row = std::min(static_cast<int>(omb.n_rows - 1),upper_row);
                            lower_col = std::max(0,lower_col);
                            upper_col = std::min(static_cast<int>(omb.n_cols - 1),upper_col);

                            int size_rows = upper_row - lower_row + 1;
                            int size_cols = upper_col - lower_col + 1;

                            const auto &jinc_mat = use_subpix ? subpix_vec->at(subpix_idx) : jinc_weights_mat[array_index];
                            const auto &jinc_sq_mat = use_subpix ? subpix_sq_vec->at(subpix_idx) : jinc_weights_sq_mat[array_index];
                            const auto mat_block = jinc_mat.block(jinc_lower_row,jinc_lower_col,size_rows,size_cols);
                            const auto mat_sq_block = jinc_sq_mat.block(jinc_lower_row,jinc_lower_col,size_rows,size_cols);

                            auto sig_block = omb_copy.signal[map_index].block(lower_row,lower_col,size_rows,size_cols);
                            auto grid_wt_block = omb_copy.grid_weight[map_index].block(lower_row,lower_col,size_rows,size_cols);
                            auto wt_block = omb_copy.weight[map_index].block(lower_row,lower_col,size_rows,size_cols);

                            // populate signal map
                            sig_block += (mat_block * in.weights.data(i) * in.scans.data(j,i)).eval();

                            // memo-style gridding denominator
                            grid_wt_block.array() += (mat_block.array() * in.weights.data(i));

                            // variance accumulator for final inverse-variance weights
                            wt_block.array() += (mat_sq_block.array() * in.weights.data(i));

                            // populate coverage map
                            if (run_coverage) {
                                auto cov_block = omb_copy.coverage[map_index].block(lower_row,lower_col,size_rows,size_cols);
                                cov_block.array() += (mat_sq_block.array() / d_fsmp);
                            }

                            // populate kernel map
                            if (run_kernel) {
                                auto ker_block = omb_copy.kernel[map_index].block(lower_row,lower_col,size_rows,size_cols);
                                ker_block += mat_block*in.weights.data(i)*in.kernel.data(j,i);
                            }

                            omb_bounds[static_cast<size_t>(map_index)].update(lower_row, upper_row, lower_col, upper_col);
                        }
                    }

                    // check if noise maps requested
                    if (run_noise) {
                        // if coaddition is enabled
                        if (use_cmb) {
                            nmb_ir = static_cast<Eigen::Index>(std::llround(cmb_irow(j)));
                            nmb_ic = static_cast<Eigen::Index>(std::llround(cmb_icol(j)));
                        }

                        // else make noise maps for obs
                        else {
                            nmb_ir = static_cast<Eigen::Index>(std::llround(omb_irow(j)));
                            nmb_ic = static_cast<Eigen::Index>(std::llround(omb_icol(j)));
                        }

                        // make sure pixel is in the map
                        if ((nmb_ir >= 0) && (nmb_ir < nmb->n_rows) && (nmb_ic >= 0) && (nmb_ic < nmb->n_cols)) {
                            int nmb_subpix_idx = 0;
                            if (use_subpix) {
                                auto subpix_index = [&](double d) {
                                    int idx = static_cast<int>(std::floor((d + 0.5) * subpixel_n));
                                    if (idx < 0) {
                                        idx = 0;
                                    }
                                    else if (idx >= subpixel_n) {
                                        idx = subpixel_n - 1;
                                    }
                                    return idx;
                                };
                                double drow = use_cmb ? (cmb_irow(j) - static_cast<double>(nmb_ir))
                                                      : (omb_irow(j) - static_cast<double>(nmb_ir));
                                double dcol = use_cmb ? (cmb_icol(j) - static_cast<double>(nmb_ic))
                                                      : (omb_icol(j) - static_cast<double>(nmb_ic));
                                int sr = subpix_index(drow);
                                int sc = subpix_index(dcol);
                                nmb_subpix_idx = sr * subpixel_n + sc;
                            }

                            int lower_row = nmb_ir - mat_rows_center;
                            int upper_row = nmb_ir + mat_rows - 1 - mat_rows_center;
                            int lower_col = nmb_ic - mat_cols_center;
                            int upper_col = nmb_ic + mat_cols - 1 - mat_cols_center;

                            int jinc_lower_row = abs(std::min(0, lower_row));
                            int jinc_lower_col = abs(std::min(0, lower_col));

                            lower_row = std::max(0,lower_row);
                            upper_row = std::min(static_cast<int>(nmb->n_rows - 1),upper_row);
                            lower_col = std::max(0,lower_col);
                            upper_col = std::min(static_cast<int>(nmb->n_cols - 1),upper_col);

                            int size_rows = upper_row - lower_row + 1;
                            int size_cols = upper_col - lower_col + 1;

                            const auto &jinc_mat = use_subpix ? subpix_vec->at(nmb_subpix_idx) : jinc_weights_mat[array_index];
                            const auto mat_block = jinc_mat.block(jinc_lower_row,jinc_lower_col,size_rows,size_cols);
                            signal = in.scans.data(j,i)*in.weights.data(i);

                            if (direct_noise_accum) {
                                std::scoped_lock<std::mutex> lk(*jinc_mutex);
                                for (Eigen::Index nn=0; nn<nmb->n_noise; ++nn) {
                                    // randomizing on dets
                                    if (nmb->randomize_dets) {
                                        noise_v = in.noise.data(nn,i)*signal;
                                    }
                                    else {
                                        noise_v = in.noise.data(nn)*signal;
                                    }
                                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(nmb->noise[map_index].data() + nn * nmb->n_rows * nmb->n_cols,
                                                                                                                   nmb->n_rows, nmb->n_cols);
                                    auto noise_block = noise_matrix.block(lower_row,lower_col,size_rows,size_cols);
                                    noise_block.array() += (mat_block.array() * noise_v);
                                }
                            }
                            else {
                                for (Eigen::Index nn=0; nn<nmb->n_noise; ++nn) {
                                    // randomizing on dets
                                    if (nmb->randomize_dets) {
                                        noise_v = in.noise.data(nn,i)*signal;
                                    }
                                    else {
                                        noise_v = in.noise.data(nn)*signal;
                                    }
                                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(nmb_copy->noise[map_index].data() + nn * nmb->n_rows * nmb->n_cols,
                                                                                                                   nmb->n_rows, nmb->n_cols);
                                    auto noise_block = noise_matrix.block(lower_row,lower_col,size_rows,size_cols);
                                    noise_block.array() += (mat_block.array() * noise_v);
                                }
                                nmb_bounds[static_cast<size_t>(map_index)].update(lower_row, upper_row, lower_col, upper_col);
                            }
                        }
                    }
                }
            }
        }
    }

    {
        std::scoped_lock<std::mutex> lk(*jinc_mutex);
        if (run_omb) {
            for (size_t i = 0; i < omb.signal.size(); ++i) {
                const auto &bounds = omb_bounds[i];
                if (!bounds.touched) {
                    continue;
                }
                int size_rows = bounds.row_max - bounds.row_min + 1;
                int size_cols = bounds.col_max - bounds.col_min + 1;

                omb.signal[i].block(bounds.row_min, bounds.col_min, size_rows, size_cols) +=
                    omb_copy.signal[i].block(bounds.row_min, bounds.col_min, size_rows, size_cols);
                omb.weight[i].block(bounds.row_min, bounds.col_min, size_rows, size_cols) +=
                    omb_copy.weight[i].block(bounds.row_min, bounds.col_min, size_rows, size_cols);
                omb.grid_weight[i].block(bounds.row_min, bounds.col_min, size_rows, size_cols) +=
                    omb_copy.grid_weight[i].block(bounds.row_min, bounds.col_min, size_rows, size_cols);

                if (run_coverage) {
                    omb.coverage[i].block(bounds.row_min, bounds.col_min, size_rows, size_cols) +=
                        omb_copy.coverage[i].block(bounds.row_min, bounds.col_min, size_rows, size_cols);
                }
                if (run_kernel) {
                    omb.kernel[i].block(bounds.row_min, bounds.col_min, size_rows, size_cols) +=
                        omb_copy.kernel[i].block(bounds.row_min, bounds.col_min, size_rows, size_cols);
                }
            }
        }

        if (run_noise && !direct_noise_accum) {
            for (size_t i = 0; i < nmb->noise.size(); ++i) {
                const auto &bounds = nmb_bounds[i];
                if (!bounds.touched) {
                    continue;
                }
                int size_rows = bounds.row_max - bounds.row_min + 1;
                int size_cols = bounds.col_max - bounds.col_min + 1;
                for (Eigen::Index nn = 0; nn < nmb->n_noise; ++nn) {
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_dst(
                        nmb->noise[i].data() + nn * nmb->n_rows * nmb->n_cols, nmb->n_rows, nmb->n_cols);
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_src(
                        nmb_copy->noise[i].data() + nn * nmb->n_rows * nmb->n_cols, nmb->n_rows, nmb->n_cols);
                    noise_dst.block(bounds.row_min, bounds.col_min, size_rows, size_cols) +=
                        noise_src.block(bounds.row_min, bounds.col_min, size_rows, size_cols);
                }
            }
        }
    }

    if (run_noise) {
        nmb = nullptr;
        nmb_copy = nullptr;
    }
}


template<class map_buffer_t, typename Derived, typename apt_t>
void JincMapmaker::populate_maps_jinc_parallel(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in,
                        map_buffer_t &omb, map_buffer_t &cmb, Eigen::DenseBase<Derived> &map_indices,
                        std::string &pixel_axes, apt_t &apt, double d_fsmp, bool run_omb, bool run_noise,
                        const Eigen::Matrix<bool, Eigen::Dynamic, 1> *active_maps) {

    const bool use_omb = !omb.noise.empty();
    const bool use_cmb = !cmb.noise.empty() && !use_omb;
    const bool run_kernel = !omb.kernel.empty();
    const bool run_coverage = !omb.coverage.empty();
    const bool run_hwpr = in.hwpr_angle.data.size()!=0;

    // dimensions of data
    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    // pointer to map buffer with noise maps
    map_buffer_t *nmb = nullptr;

    if (run_noise) {
        nmb = use_cmb ? &cmb : (use_omb ? &omb : nullptr);
    }

    auto fail_validation = [&](const std::string &stage, const std::string &message) -> void {
        std::ostringstream oss;
        oss << "populate_maps_jinc_parallel " << stage << ": " << message;
        throw std::runtime_error(oss.str());
    };

    auto matrix_dims = [](const auto &mat) {
        std::ostringstream oss;
        oss << mat.rows() << "x" << mat.cols();
        return oss.str();
    };

    auto validate_global_preflight = [&]() {
        if (run_omb) {
            if (omb.signal.empty()) {
                fail_validation("preflight", "run_omb=true but omb.signal is empty");
            }
            if (omb.weight.size() != omb.signal.size()) {
                std::ostringstream oss;
                oss << "omb.weight.size()=" << omb.weight.size()
                    << " does not match omb.signal.size()=" << omb.signal.size();
                fail_validation("preflight", oss.str());
            }
            if (omb.grid_weight.size() != omb.signal.size()) {
                std::ostringstream oss;
                oss << "omb.grid_weight.size()=" << omb.grid_weight.size()
                    << " does not match omb.signal.size()=" << omb.signal.size();
                fail_validation("preflight", oss.str());
            }
            if (run_coverage && omb.coverage.size() != omb.signal.size()) {
                std::ostringstream oss;
                oss << "omb.coverage.size()=" << omb.coverage.size()
                    << " does not match omb.signal.size()=" << omb.signal.size();
                fail_validation("preflight", oss.str());
            }
            if (run_kernel && omb.kernel.size() != omb.signal.size()) {
                std::ostringstream oss;
                oss << "omb.kernel.size()=" << omb.kernel.size()
                    << " does not match omb.signal.size()=" << omb.signal.size();
                fail_validation("preflight", oss.str());
            }
        }
        if (run_noise && nmb == nullptr) {
            fail_validation("preflight", "run_noise=true but no noise map buffer is available");
        }
    };

    auto validate_map_index = [&](const char *stage, Eigen::Index map_index, Eigen::Index det_index, int det_uid) {
        if (map_index < 0 || map_index >= static_cast<Eigen::Index>(omb.signal.size())) {
            std::ostringstream oss;
            oss << "det_col=" << det_index
                << " det_uid=" << det_uid
                << " map_index=" << map_index
                << " is outside [0, " << static_cast<Eigen::Index>(omb.signal.size()) - 1 << "]";
            fail_validation(stage, oss.str());
        }
        if (run_noise && nmb != nullptr &&
            (map_index < 0 || map_index >= static_cast<Eigen::Index>(nmb->noise.size()))) {
            std::ostringstream oss;
            oss << "det_col=" << det_index
                << " det_uid=" << det_uid
                << " noise map_index=" << map_index
                << " is outside [0, " << static_cast<Eigen::Index>(nmb->noise.size()) - 1 << "]";
            fail_validation(stage, oss.str());
        }
    };

    auto validate_kernel_cache = [&](const char *stage, Eigen::Index array_index, Eigen::Index det_index, int det_uid) {
        auto mat_it = jinc_weights_mat.find(array_index);
        auto mat_sq_it = jinc_weights_sq_mat.find(array_index);
        if (mat_it == jinc_weights_mat.end() || mat_sq_it == jinc_weights_sq_mat.end()) {
            std::ostringstream oss;
            oss << "det_col=" << det_index
                << " det_uid=" << det_uid
                << " array=" << array_index
                << " missing jinc kernel cache";
            fail_validation(stage, oss.str());
        }
        if (mat_it->second.rows() <= 0 || mat_it->second.cols() <= 0 ||
            mat_sq_it->second.rows() <= 0 || mat_sq_it->second.cols() <= 0) {
            std::ostringstream oss;
            oss << "det_col=" << det_index
                << " det_uid=" << det_uid
                << " array=" << array_index
                << " invalid kernel dims signal=" << matrix_dims(mat_it->second)
                << " weight=" << matrix_dims(mat_sq_it->second);
            fail_validation(stage, oss.str());
        }
        if (mat_it->second.rows() != mat_sq_it->second.rows() ||
            mat_it->second.cols() != mat_sq_it->second.cols()) {
            std::ostringstream oss;
            oss << "det_col=" << det_index
                << " det_uid=" << det_uid
                << " array=" << array_index
                << " mismatched kernel dims signal=" << matrix_dims(mat_it->second)
                << " weight=" << matrix_dims(mat_sq_it->second);
            fail_validation(stage, oss.str());
        }
        if (subpixel_n > 1) {
            auto subpix_it = jinc_weights_mat_subpix.find(array_index);
            auto subpix_sq_it = jinc_weights_sq_mat_subpix.find(array_index);
            const auto expected = static_cast<size_t>(subpixel_n * subpixel_n);
            if (subpix_it == jinc_weights_mat_subpix.end() || subpix_sq_it == jinc_weights_sq_mat_subpix.end()) {
                std::ostringstream oss;
                oss << "det_col=" << det_index
                    << " det_uid=" << det_uid
                    << " array=" << array_index
                    << " missing subpixel cache for subpixel_n=" << subpixel_n;
                fail_validation(stage, oss.str());
            }
            if (subpix_it->second.size() != expected || subpix_sq_it->second.size() != expected) {
                std::ostringstream oss;
                oss << "det_col=" << det_index
                    << " det_uid=" << det_uid
                    << " array=" << array_index
                    << " subpixel cache size mismatch signal=" << subpix_it->second.size()
                    << " weight=" << subpix_sq_it->second.size()
                    << " expected=" << expected;
                fail_validation(stage, oss.str());
            }
        }
    };

    auto validate_block_bounds = [&](const char *stage,
                                     const auto &target,
                                     int lower_row,
                                     int lower_col,
                                     int size_rows,
                                     int size_cols,
                                     Eigen::Index det_index,
                                     int det_uid,
                                     Eigen::Index sample_index,
                                     Eigen::Index map_index,
                                     Eigen::Index array_index) {
        if (lower_row < 0 || lower_col < 0 || size_rows <= 0 || size_cols <= 0 ||
            lower_row + size_rows > target.rows() ||
            lower_col + size_cols > target.cols()) {
            std::ostringstream oss;
            oss << "det_col=" << det_index
                << " det_uid=" << det_uid
                << " sample=" << sample_index
                << " map_index=" << map_index
                << " array=" << array_index
                << " block row=" << lower_row
                << " col=" << lower_col
                << " size=" << size_rows << "x" << size_cols
                << " target=" << matrix_dims(target);
            fail_validation(stage, oss.str());
        }
    };

    validate_global_preflight();

    // placeholder vectors of size ndet for grppi maps
    std::vector<int> map_in_vec, map_out_vec;
    map_in_vec.resize(static_cast<size_t>(n_dets));
    std::iota(map_in_vec.begin(), map_in_vec.end(), 0);
    map_out_vec.resize(map_in_vec.size());

    // parallelize over detectors
    //for (Eigen::Index i=0; i<n_dets; ++i) {
    grppi::map(tula::grppi_utils::dyn_ex(omb.parallel_policy), map_in_vec, map_out_vec, [&](auto i) {
        reset_jinc_debug_breadcrumb();
        // skip fg = -1 if in polarization mode
        const bool run_det = !(run_polarization && apt["fg"](i)==-1);
        if (!run_det) {
            return 0;
        }

        // skip completely flagged detectors
        if (apt["flag"](i)==0 && (in.flags.data.col(i).array()==false).any()) {
            // get detector positions from apt table if not in detector mapmaking mode
            auto det_index = i;
            const int det_uid = (apt["uid"].size() > det_index)
                ? static_cast<int>(std::llround(apt["uid"](det_index)))
                : static_cast<int>(det_index);

            // which map to assign detector to
            Eigen::Index map_index = map_indices(i);
            if (active_maps != nullptr &&
                (map_index < 0 || map_index >= active_maps->size() || !(*active_maps)(map_index))) {
                return 0;
            }
            Eigen::Index array_index = apt["array"](det_index);
            update_jinc_debug_breadcrumb("detector-preflight", det_index, det_uid, -1, map_index,
                                         array_index, -1, -1, -1);
            validate_map_index("detector-preflight", map_index, det_index, det_uid);
            validate_kernel_cache("detector-preflight", array_index, det_index, det_uid);
            const bool use_subpix = (subpixel_n > 1) && (jinc_weights_mat_subpix.count(array_index) > 0);
            const auto *subpix_vec = use_subpix ? &jinc_weights_mat_subpix.at(array_index) : nullptr;
            const auto *subpix_sq_vec = use_subpix ? &jinc_weights_sq_mat_subpix.at(array_index) : nullptr;
            Eigen::Index mat_rows = jinc_weights_mat[array_index].rows();
            Eigen::Index mat_cols = jinc_weights_mat[array_index].cols();
            Eigen::Index mat_rows_center = (mat_rows - 1.)/2.;
            Eigen::Index mat_cols_center = (mat_cols - 1.)/2.;

            // get detector pointing
            auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, apt["x_t"](det_index), apt["y_t"](det_index),
                                                              pixel_axes, in.pointing_offsets_arcsec.data, omb.map_grouping);

            // get map buffer row and col indices for lat and lon vectors
            Eigen::VectorXd omb_irow = lat.array()/omb.pixel_size_rad + (omb.n_rows - 1)/2.;
            Eigen::VectorXd omb_icol = lon.array()/omb.pixel_size_rad + (omb.n_cols - 1)/2.;

            Eigen::VectorXd cmb_irow, cmb_icol;
            if (use_cmb) {
                // get coadded map buffer row and col indices for lat and lon vectors
                cmb_irow = lat.array()/cmb.pixel_size_rad + (cmb.n_rows - 1)/2.;
                cmb_icol = lon.array()/cmb.pixel_size_rad + (cmb.n_cols - 1)/2.;
            }

            // signal map value
            double signal;

            // noise map value
            double noise_v;

            // noise map indices
            Eigen::Index nmb_ir, nmb_ic;

            // cosine and sine of angles
            double angle, cos_2angle, sin_2angle;

            // loop through the samples
            for (Eigen::Index j=0; j<n_pts; ++j) {
                // check if sample is flagged, ignore if so
                if (!in.flags.data(j,i) &&
                    std::isfinite(in.scans.data(j,i)) &&
                    i < in.weights.data.size() &&
                    std::isfinite(in.weights.data(i)) &&
                    in.weights.data(i) > 0.0) {
                    Eigen::Index omb_ir = static_cast<Eigen::Index>(std::llround(omb_irow(j)));
                    Eigen::Index omb_ic = static_cast<Eigen::Index>(std::llround(omb_icol(j)));
                    int subpix_idx = 0;
                    if (use_subpix) {
                        auto subpix_index = [&](double d) {
                            int idx = static_cast<int>(std::floor((d + 0.5) * subpixel_n));
                            if (idx < 0) {
                                idx = 0;
                            }
                            else if (idx >= subpixel_n) {
                                idx = subpixel_n - 1;
                            }
                            return idx;
                        };
                        double drow = omb_irow(j) - static_cast<double>(omb_ir);
                        double dcol = omb_icol(j) - static_cast<double>(omb_ic);
                        int sr = subpix_index(drow);
                        int sc = subpix_index(dcol);
                        subpix_idx = sr * subpixel_n + sc;
                    }
                    update_jinc_debug_breadcrumb("sample", det_index, det_uid, j, map_index,
                                                 array_index, static_cast<int>(omb_ir), static_cast<int>(omb_ic), subpix_idx);

                    if (run_polarization) {
                        auto fg_index = apt["fg"](det_index);
                        if (run_hwpr) {
                            angle = 2*in.hwpr_angle.data(j) - (in.angle.data(j) + fgs[fg_index] + install_ang[array_index]);
                        }
                        else {
                            angle = in.angle.data(j) + fgs[fg_index] + install_ang[array_index];
                        }

                        cos_2angle = cos(2.*angle);
                        sin_2angle = sin(2.*angle);
                    }

                    if (run_omb) {
                        // make sure the data point is within the map
                        if ((omb_ir >= 0) && (omb_ir < omb.n_rows) && (omb_ic >= 0) && (omb_ic < omb.n_cols)) {

                            int lower_row = omb_ir - mat_rows_center;
                            int upper_row = omb_ir + mat_rows - 1 - mat_rows_center;
                            int lower_col = omb_ic - mat_cols_center;
                            int upper_col = omb_ic + mat_cols - 1 - mat_cols_center;

                            int jinc_lower_row = abs(std::min(0, lower_row));
                            int jinc_lower_col = abs(std::min(0, lower_col));

                            lower_row = std::max(0,lower_row);
                            upper_row = std::min(static_cast<int>(omb.n_rows - 1),upper_row);
                            lower_col = std::max(0,lower_col);
                            upper_col = std::min(static_cast<int>(omb.n_cols - 1),upper_col);

                            int size_rows = upper_row - lower_row + 1;
                            int size_cols = upper_col - lower_col + 1;

                            const auto &jinc_mat = use_subpix ? subpix_vec->at(subpix_idx) : jinc_weights_mat[array_index];
                            const auto &jinc_sq_mat = use_subpix ? subpix_sq_vec->at(subpix_idx) : jinc_weights_sq_mat[array_index];
                            update_jinc_debug_breadcrumb_block("omb-block-bounds", lower_row, upper_row, lower_col, upper_col,
                                                               jinc_lower_row, jinc_lower_col, size_rows, size_cols);
                            validate_block_bounds("omb-jinc-kernel-block", jinc_mat, jinc_lower_row, jinc_lower_col,
                                                  size_rows, size_cols, det_index, det_uid, j, map_index, array_index);
                            validate_block_bounds("omb-jinc-weight-block", jinc_sq_mat, jinc_lower_row, jinc_lower_col,
                                                  size_rows, size_cols, det_index, det_uid, j, map_index, array_index);
                            validate_block_bounds("omb-signal-block", omb.signal[map_index], lower_row, lower_col,
                                                  size_rows, size_cols, det_index, det_uid, j, map_index, array_index);
                            validate_block_bounds("omb-grid-weight-block", omb.grid_weight[map_index], lower_row, lower_col,
                                                  size_rows, size_cols, det_index, det_uid, j, map_index, array_index);
                            validate_block_bounds("omb-weight-block", omb.weight[map_index], lower_row, lower_col,
                                                  size_rows, size_cols, det_index, det_uid, j, map_index, array_index);
                            const auto mat_block = jinc_mat.block(jinc_lower_row,jinc_lower_col,size_rows,size_cols);
                            const auto mat_sq_block = jinc_sq_mat.block(jinc_lower_row,jinc_lower_col,size_rows,size_cols);

                            auto sig_block = omb.signal[map_index].block(lower_row,lower_col,size_rows,size_cols);
                            auto grid_wt_block = omb.grid_weight[map_index].block(lower_row,lower_col,size_rows,size_cols);
                            auto wt_block = omb.weight[map_index].block(lower_row,lower_col,size_rows,size_cols);

                            // populate signal map
                            sig_block += (mat_block * in.weights.data(i) * in.scans.data(j,i)).eval();

                            // memo-style gridding denominator
                            grid_wt_block.array() += (mat_block.array() * in.weights.data(i));

                            // variance accumulator for final inverse-variance weights
                            wt_block.array() += (mat_sq_block.array() * in.weights.data(i));

                            // populate coverage map
                            if (run_coverage) {
                                validate_block_bounds("omb-coverage-block", omb.coverage[map_index], lower_row, lower_col,
                                                      size_rows, size_cols, det_index, det_uid, j, map_index, array_index);
                                auto cov_block = omb.coverage[map_index].block(lower_row,lower_col,size_rows,size_cols);
                                cov_block.array() += (mat_sq_block.array() / d_fsmp);
                            }

                            // populate kernel map
                            if (run_kernel) {
                                validate_block_bounds("omb-kernel-block", omb.kernel[map_index], lower_row, lower_col,
                                                      size_rows, size_cols, det_index, det_uid, j, map_index, array_index);
                                auto ker_block = omb.kernel[map_index].block(lower_row,lower_col,size_rows,size_cols);
                                ker_block += mat_block*in.weights.data(i)*in.kernel.data(j,i);
                            }
                        }
                    }

                    // check if noise maps requested
                    if (run_noise) {
                        // if coaddition is enabled
                        if (use_cmb) {
                            nmb_ir = static_cast<Eigen::Index>(std::llround(cmb_irow(j)));
                            nmb_ic = static_cast<Eigen::Index>(std::llround(cmb_icol(j)));
                        }

                        // else make noise maps for obs
                        else {
                            nmb_ir = static_cast<Eigen::Index>(std::llround(omb_irow(j)));
                            nmb_ic = static_cast<Eigen::Index>(std::llround(omb_icol(j)));
                        }

                        // make sure pixel is in the map
                        if ((nmb_ir >= 0) && (nmb_ir < nmb->n_rows) && (nmb_ic >= 0) && (nmb_ic < nmb->n_cols)) {
                            int nmb_subpix_idx = 0;
                            if (use_subpix) {
                                auto subpix_index = [&](double d) {
                                    int idx = static_cast<int>(std::floor((d + 0.5) * subpixel_n));
                                    if (idx < 0) {
                                        idx = 0;
                                    }
                                    else if (idx >= subpixel_n) {
                                        idx = subpixel_n - 1;
                                    }
                                    return idx;
                                };
                                double drow = use_cmb ? (cmb_irow(j) - static_cast<double>(nmb_ir))
                                                      : (omb_irow(j) - static_cast<double>(nmb_ir));
                                double dcol = use_cmb ? (cmb_icol(j) - static_cast<double>(nmb_ic))
                                                      : (omb_icol(j) - static_cast<double>(nmb_ic));
                                int sr = subpix_index(drow);
                                int sc = subpix_index(dcol);
                                nmb_subpix_idx = sr * subpixel_n + sc;
                            }

                            int lower_row = nmb_ir - mat_rows_center;
                            int upper_row = nmb_ir + mat_rows - 1 - mat_rows_center;
                            int lower_col = nmb_ic - mat_cols_center;
                            int upper_col = nmb_ic + mat_cols - 1 - mat_cols_center;

                            int jinc_lower_row = abs(std::min(0, lower_row));
                            int jinc_lower_col = abs(std::min(0, lower_col));

                            lower_row = std::max(0,lower_row);
                            upper_row = std::min(static_cast<int>(nmb->n_rows - 1),upper_row);
                            lower_col = std::max(0,lower_col);
                            upper_col = std::min(static_cast<int>(nmb->n_cols - 1),upper_col);

                            int size_rows = upper_row - lower_row + 1;
                            int size_cols = upper_col - lower_col + 1;

                            const auto &jinc_mat = use_subpix ? subpix_vec->at(nmb_subpix_idx) : jinc_weights_mat[array_index];
                            update_jinc_debug_breadcrumb("noise-sample", det_index, det_uid, j, map_index,
                                                         array_index, static_cast<int>(nmb_ir), static_cast<int>(nmb_ic),
                                                         nmb_subpix_idx);
                            update_jinc_debug_breadcrumb_block("noise-block-bounds", lower_row, upper_row, lower_col, upper_col,
                                                               jinc_lower_row, jinc_lower_col, size_rows, size_cols);
                            validate_block_bounds("noise-jinc-kernel-block", jinc_mat, jinc_lower_row, jinc_lower_col,
                                                  size_rows, size_cols, det_index, det_uid, j, map_index, array_index);
                            const auto mat_block = jinc_mat.block(jinc_lower_row,jinc_lower_col,size_rows,size_cols);
                            signal = in.scans.data(j,i)*in.weights.data(i);

                            for (Eigen::Index nn=0; nn<nmb->n_noise; ++nn) {
                                // randomizing on dets
                                if (nmb->randomize_dets) {
                                    noise_v = in.noise.data(nn,i)*signal;
                                }
                                else {
                                    noise_v = in.noise.data(nn)*signal;
                                }
                                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> noise_matrix(nmb->noise[map_index].data() + nn * nmb->n_rows * nmb->n_cols,
                                                                                                               nmb->n_rows, nmb->n_cols);
                                validate_block_bounds("noise-map-block", noise_matrix, lower_row, lower_col,
                                                      size_rows, size_cols, det_index, det_uid, j, map_index, array_index);
                                auto noise_block = noise_matrix.block(lower_row,lower_col,size_rows,size_cols);
                                noise_block.array() += (mat_block.array() * noise_v);
                            }
                        }
                    }
                }
            }
        }
        return 0;
    });

    if (run_noise) {
        nmb = nullptr;
    }
}
} // namespace mapmaking
