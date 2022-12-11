#pragma once

#include <boost/random.hpp>
#include <boost/random/random_device.hpp>

#include <thread>

#include <citlali/core/utils/pointing.h>

using timestream::TCData;
using timestream::RTCProc;
using timestream::PTCProc;

// selects the type of TCData
using timestream::TCDataKind;

namespace mapmaking {

template<class map_buffer_t, typename Derived, typename apt_t, typename pointing_offset_t>
void populate_maps_naive(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in,
                         map_buffer_t &omb, map_buffer_t &cmb, Eigen::DenseBase<Derived> &map_indices, Eigen::DenseBase<Derived> &det_indices,
                         std::string &pixel_axes, std::string &redu_type, apt_t &apt,
                         pointing_offset_t &pointing_offsets_arcsec, double d_fsmp, bool run_noise) {


    Eigen::Index n_dets = in.scans.data.cols();
    Eigen::Index n_pts = in.scans.data.rows();

    // pointer to map buffer with noise maps
    ObsMapBuffer* nmb = NULL;

    // matrix to hold random noise value
    Eigen::MatrixXi noise;

    if (run_noise) {
        // set pointer to cmb if it has noise maps
        if (!cmb.noise.empty()) {
            nmb = &cmb;
        }

        // otherwise set pointer to omb if it has noise maps
        else if (!omb.noise.empty()) {
            nmb = &omb;
        }

        // declare random number generator
        thread_local boost::random::mt19937 eng;

        // boost random number generator (0,1)
        boost::random::uniform_int_distribution<> rands{0,1};

        // rescale random values to -1 or 1
        noise =
            Eigen::MatrixXi::Zero(nmb->n_noise, n_pts).unaryExpr([&](int dummy){return rands(eng);});
        noise = (2.*(noise.template cast<double>().array() - 0.5)).template cast<int>();
    }

    for (Eigen::Index i=0; i<n_dets; i++) {
        // skip completely flagged detectors
        //if ((in.flags.data.array() !=0).all()) {
            double az_off = 0;
            double el_off = 0;

            if (redu_type!="beammap") {
                auto det_index = det_indices(i);
                az_off = apt["x_t"](det_index);
                el_off = apt["y_t"](det_index);
            }

            Eigen::Index map_index = map_indices(i);

            auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, az_off, el_off,
                                                              pixel_axes, pointing_offsets_arcsec);

            // get map buffer row and col indices for lat and lon vectors
            Eigen::VectorXd omb_irow = lat.array()/omb.pixel_size_rad + (omb.n_rows)/2.;
            Eigen::VectorXd omb_icol = lon.array()/omb.pixel_size_rad + (omb.n_cols)/2.;

            Eigen::VectorXd cmb_irow, cmb_icol;
            if (!cmb.noise.empty()) {
                // get coadded map buffer row and col indices for lat and lon vectors
                cmb_irow = lat.array()/cmb.pixel_size_rad + (cmb.n_rows)/2.;
                cmb_icol = lon.array()/cmb.pixel_size_rad + (cmb.n_cols)/2.;
            }

            // loop through the samples
            for (Eigen::Index j=0; j<n_pts; j++) {
                // check if sample is flagged, ignore if so
                if (in.flags.data(j,i)) {
                    Eigen::Index omb_ir = omb_irow(j);
                    Eigen::Index omb_ic = omb_icol(j);

                    double signal;

                    // make sure the data point is within the map
                    if ((omb_ir >= 0) && (omb_ir < omb.n_rows) && (omb_ic >= 0) && (omb_ic < omb.n_cols)) {

                        // populate signal map
                        signal = in.scans.data(j,i)*in.weights.data(i);
                        omb.signal[map_index](omb_ir, omb_ic) += signal;

                        // populate weight map
                        omb.weight[map_index](omb_ir, omb_ic) += in.weights.data(i);

                        // populate kernel map
                        if (!omb.kernel.empty()) {
                            auto kernel = in.kernel.data(j,i)*in.weights.data(i);
                            omb.kernel[map_index](omb_ir, omb_ic) += kernel;
                        }

                        // populate coverage map
                        if (!omb.coverage.empty()) {
                            omb.coverage[map_index](omb_ir, omb_ic) += 1./d_fsmp;
                        }
                    }

                    // check if noise maps requested
                    if (run_noise) {
                        Eigen::Index nmb_ir, nmb_ic;

                        // if coaddition is enabled
                        if (!cmb.noise.empty()) {
                            nmb_ir = cmb_irow(j);
                            nmb_ic = cmb_icol(j);
                        }

                        // else make noise maps for obs
                        else if (!omb.noise.empty()) {
                            nmb_ir = omb_irow(j);
                            nmb_ic = omb_icol(j);
                        }

                        // loop through noise maps
                        for (Eigen::Index nn=0; nn<nmb->n_noise; nn++) {
                            // coadd into current noise map
                            if ((nmb_ir >= 0) && (nmb_ir < nmb->n_rows) && (nmb_ic >= 0) && (nmb_ic < nmb->n_cols)) {
                                nmb->noise[map_index](nmb_ir,nmb_ic,nn) += noise(nn,j)*signal;
                            }
                        }
                    }
                }
            }
        //}
    }
}
} // namespace mapmaking