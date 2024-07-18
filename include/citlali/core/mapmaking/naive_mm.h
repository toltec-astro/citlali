#pragma once

#include <thread>
#include <mutex>

#include <Eigen/Sparse>

#include <citlali/core/timestream/timestream.h>

#include <citlali/core/mapmaking/map.h>
#include <citlali/core/utils/pointing.h>

using timestream::TCData;

// selects the type of TCData
using timestream::TCDataKind;

namespace mapmaking {

class NaiveMapmaker {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");
    std::unique_ptr<std::mutex> naive_mutex = std::make_unique<std::mutex>();

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

    template <typename Derived>
    void add_sparse_to_dense(std::vector<Eigen::Triplet<double>> &triplets, Eigen::DenseBase<Derived> &dense_matrix) {
        Eigen::SparseMatrix<double> sparse_matrix(dense_matrix.rows(),dense_matrix.cols());
        sparse_matrix.setFromTriplets(triplets.begin(), triplets.end());

        for (int k = 0; k < sparse_matrix.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(sparse_matrix, k); it; ++it) {
                dense_matrix(it.row(), it.col()) += it.value();
            }
        }
    }

    // run polarization?
    bool run_polarization;

    // allocate pointing matrix for polarization reduction
    template <class map_buffer_t>
    void allocate_pointing(map_buffer_t &, double, double, double, Eigen::Index, int, int);

    // populate maps with a time chunk (signal, kernel, coverage, and noise)
    template<class map_buffer_t, typename Derived, typename apt_t>
    void populate_maps_naive(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, map_buffer_t &, map_buffer_t &,
                             Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &,
                             std::string &, apt_t &, double, bool, bool);
};

template <class map_buffer_t>
void NaiveMapmaker::allocate_pointing(map_buffer_t &mb, double weight, double cos_2angle, double sin_2angle,
                                      Eigen::Index map_index, int ir, int ic) {

    int pix = mb.n_rows*ic + ir;
    // update pointing matrix
    mb.pointing[map_index](pix,0) += weight;
    mb.pointing[map_index](pix,1) += weight*cos_2angle;
    mb.pointing[map_index](pix,2) += weight*sin_2angle;
    mb.pointing[map_index](pix,3) = mb.pointing[map_index](pix,1);//weight*cos(2*angle);
    mb.pointing[map_index](pix,4) += weight*pow(cos_2angle,2.);
    mb.pointing[map_index](pix,5) += weight*cos_2angle*sin_2angle;
    mb.pointing[map_index](pix,6) = mb.pointing[map_index](pix,2);//weight*sin(2*angle);
    mb.pointing[map_index](pix,7) = mb.pointing[map_index](pix,5);//weight*cos(2*angle)*sin(2*angle);
    mb.pointing[map_index](pix,8) += weight*pow(sin_2angle,2.);
}

template<class map_buffer_t, typename Derived, typename apt_t>
void NaiveMapmaker::populate_maps_naive(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, map_buffer_t &omb,
                                        map_buffer_t &cmb, Eigen::DenseBase<Derived> &map_indices,
                                        Eigen::DenseBase<Derived> &det_indices, Eigen::DenseBase<Derived> &fg_indices,
                                        std::string &pixel_axes, apt_t &apt, double d_fsmp, bool run_omb, bool run_noise) {

    const bool use_cmb = !cmb.noise.empty();
    const bool use_omb = !omb.noise.empty();
    const bool run_kernel = !omb.kernel.empty();
    const bool run_coverage = !omb.coverage.empty();
    const bool run_hwpr = in.hwpr_angle.data.size()!=0;

    typedef Eigen::Triplet<double> T;
    std::vector<std::vector<T>> signals, weights, kernels, coverages;

    signals.resize(omb.signal.size());
    weights.resize(omb.signal.size());

    if (run_kernel) {
        kernels.resize(omb.signal.size());
    }
    if (run_coverage) {
        coverages.resize(omb.signal.size());
    }

    map_buffer_t omb_copy;

    if (use_omb) {
        omb_copy.noise = omb.noise;

        for (Eigen::Index i=0; i<omb.signal.size(); ++i) {
            omb_copy.noise[i].setZero();
        }
    }

    map_buffer_t cmb_copy;

    if (use_cmb) {
        cmb_copy.noise = cmb.noise;

        for (Eigen::Index i=0; i<cmb.noise.size(); ++i) {
            cmb_copy.noise[i].setZero();
        }
    }

    // dimensions of data
    Eigen::Index n_pts = in.scans.data.rows();
    Eigen::Index n_dets = in.scans.data.cols();

    // step to skip to reach next stokes param
    int step = omb.pointing.size();

    // pointer to map buffer with noise maps
    map_buffer_t *nmb, *nmb_copy;

    if (run_noise) {
        nmb = use_cmb ? &cmb : (use_omb ? &omb : nullptr);
        nmb_copy = use_cmb ? &cmb_copy : (use_omb ? &omb_copy : nullptr);
    }

    // signal and kernel map values
    double signal, kernel;

    // noise map value
    double noise_v;

    // noise map indices
    Eigen::Index nmb_ir, nmb_ic;

    // cosine and sine of angles
    double angle, cos_2angle, sin_2angle;

    for (Eigen::Index i=0; i<n_dets; ++i) {
        // skip completely flagged detectors
        if ((in.flags.data.col(i).array()==0).any()) {
            // which map to assign detector to
            Eigen::Index map_index = map_indices(i);

            // indices for Q and U maps
            int q_index = map_index + step;
            int u_index = map_index + 2 * step;

            // get detector positions from apt table
            auto det_index = det_indices(i);
            // array index
            Eigen::Index array_index = apt["array"](det_index);
            // get detector pointing
            auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, apt["x_t"](det_index), apt["y_t"](det_index),
                                                              pixel_axes, in.pointing_offsets_arcsec.data, omb.map_grouping);

            // get map buffer row and col indices for lat and lon vectors
            Eigen::VectorXd omb_irow = lat.array()/omb.pixel_size_rad + (omb.n_rows)/2.;
            Eigen::VectorXd omb_icol = lon.array()/omb.pixel_size_rad + (omb.n_cols)/2.;

            Eigen::VectorXd cmb_irow, cmb_icol;
            if (use_cmb) {
                // get coadded map buffer row and col indices for lat and lon vectors
                cmb_irow = lat.array()/cmb.pixel_size_rad + (cmb.n_rows)/2.;
                cmb_icol = lon.array()/cmb.pixel_size_rad + (cmb.n_cols)/2.;
            }

            // loop through the samples
            for (Eigen::Index j=0; j<n_pts; ++j) {
                // check if sample is flagged, ignore if so
                if (!in.flags.data(j,i)) {
                    Eigen::Index omb_ir = omb_irow(j);
                    Eigen::Index omb_ic = omb_icol(j);

                    if (run_polarization) {
                        if (run_hwpr) {
                            angle = 2*in.hwpr_angle.data(j) - (in.angle.data(j) + fgs[fg_indices(det_index)] + install_ang[array_index]);
                        }
                        else {
                            angle = in.angle.data(j) + fgs[fg_indices(det_index)] + install_ang[array_index];
                        }

                        cos_2angle = cos(2.*angle);
                        sin_2angle = sin(2.*angle);
                    }

                    if (run_omb) {
                        // make sure the data point is within the map
                        if ((omb_ir >= 0) && (omb_ir < omb.n_rows) && (omb_ic >= 0) && (omb_ic < omb.n_cols)) {
                            // populate signal map
                            signal = in.scans.data(j,i)*in.weights.data(i);
                            signals[map_index].push_back(T(omb_ir,omb_ic,signal));

                            // populate weight map
                            weights[map_index].push_back(T(omb_ir,omb_ic,in.weights.data(i)));

                            // populate kernel map
                            if (run_kernel) {
                                kernel = in.kernel.data(j,i)*in.weights.data(i);
                                kernels[map_index].push_back(T(omb_ir,omb_ic,kernel));
                            }

                            // populate coverage map
                            if (run_coverage) {
                                coverages[map_index].push_back(T(omb_ir,omb_ic,1./d_fsmp));
                            }

                            if (run_polarization) {
                                // calculate pointing matrix
                                allocate_pointing(omb, in.weights.data(i), cos_2angle, sin_2angle, map_index, omb_ir,omb_ic);

                                // update signal map Q and U
                                signals[q_index].push_back(T(omb_ir,omb_ic,signal*cos_2angle));
                                signals[u_index].push_back(T(omb_ir,omb_ic,signal*sin_2angle));

                                // update kernel map Q and U
                                if (run_kernel) {
                                    kernels[q_index].push_back(T(omb_ir,omb_ic,kernel*cos_2angle));
                                    kernels[u_index].push_back(T(omb_ir,omb_ic,kernel*sin_2angle));
                                }
                            }
                        }
                    }

                    // check if noise maps requested
                    if (run_noise) {
                        // if coaddition is enabled
                        if (use_cmb) {
                            nmb_ir = cmb_irow(j);
                            nmb_ic = cmb_icol(j);
                        }
                        // else make noise maps for obs
                        else if (use_omb) {
                            nmb_ir = omb_irow(j);
                            nmb_ic = omb_icol(j);
                        }

                        // coadd into current noise map
                        if ((nmb_ir >= 0) && (nmb_ir < nmb->n_rows) && (nmb_ic >= 0) && (nmb_ic < nmb->n_cols)) {
                            if (run_polarization) {
                                if (use_cmb) {
                                    // calculate pointing matrix for cmb
                                    allocate_pointing(cmb, in.weights.data(i), cos_2angle, sin_2angle, map_index, nmb_ir, nmb_ic);
                                }
                            }
                            // loop through noise maps
                            for (Eigen::Index nn=0; nn<nmb->n_noise; ++nn) {
                                // randomizing on dets
                                if (nmb->randomize_dets) {
                                    noise_v = in.noise.data(nn,i)*in.scans.data(j,i)*in.weights.data(i);
                                }
                                else {
                                    noise_v = in.noise.data(nn)*in.scans.data(j,i)*in.weights.data(i);
                                }
                                // add noise value to current noise map
                                nmb_copy->noise[map_index](nmb_ir,nmb_ic,nn) += noise_v;

                                if (run_polarization) {
                                    // update noise map Q
                                    nmb_copy->noise[q_index](nmb_ir,nmb_ic,nn) += noise_v*cos_2angle;
                                    // update noise map U
                                    nmb_copy->noise[u_index](nmb_ir,nmb_ic,nn) += noise_v*sin_2angle;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    {
        std::scoped_lock<std::mutex> lk(*naive_mutex);
        if (run_omb) {
            for (int i=0; i<omb.signal.size(); ++i) {
                add_sparse_to_dense(signals[i],omb.signal[i]);
                add_sparse_to_dense(weights[i],omb.weight[i]);

                if (run_kernel) {
                    add_sparse_to_dense(kernels[i],omb.kernel[i]);
                }

                if (run_coverage) {
                    add_sparse_to_dense(coverages[i],omb.coverage[i]);
                }
            }
        }

        if (run_noise) {
            for (int i=0; i<omb.noise.size(); ++i) {
                nmb->noise[i] += nmb_copy->noise[i];
            }
            //std::transform(nmb->noise.begin(), nmb->noise.end(), nmb_copy->noise.begin(), nmb->noise.begin(), std::plus<Eigen::Tensor<double,3>>());
        }
    }
}
} // namespace mapmaking
