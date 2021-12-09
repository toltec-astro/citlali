#pragma once
#include <string>
#include <vector>
#include <map>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <tula/logging.h>

#include <citlali/core/utils/pointing.h>
#include <citlali/core/map/psd.h>

using map_dims_t = std::tuple<int, int, Eigen::VectorXd, Eigen::VectorXd>;
using map_extent_t = std::vector<double>;
using map_coord_t = std::vector<Eigen::VectorXd>;
using map_count_t = std::size_t;


class CoaddedMapBuffer {
public:
    Eigen::Index nrows, ncols, nnoise;
    map_count_t map_count;
    double pixel_size;
    double cov_cut;

    // noise average filtered rms
    Eigen::VectorXd average_filtered_rms;

    // noise weight factor
    std::vector<Eigen::VectorXd> nfac;

    // for source fits in coadded maps
    Eigen::MatrixXd pfit;

    // tensor for noise map inclusion (nnoise,nobs,nmaps)
    Eigen::Tensor<double,3> noise_rand;

    // Physical coordinates for rows and cols (radians)
    Eigen::VectorXd rcphys, ccphys;

    // vectors for each map type
    std::vector<Eigen::MatrixXd> signal, weight, kernel, coverage;

    // noise maps (nrows, ncols, nnoise) of length nmaps
    std::vector<Eigen::Tensor<double,3>> noise;

    // vector of psd classes for signal maps
    std::vector<PSD> psd;

    // vector of vector of psd classes for noise maps
    std::vector<std::vector<PSD>> noise_psd;
    std::vector<PSD> noise_avg_psd;

    void setup_maps(std::vector<map_coord_t> map_coords, map_count_t _mc) {
        map_count = _mc;

        double min_row, max_row, min_col, max_col;

        // loop through physical coordinates and get min/max
        for (Eigen::Index i=0; i<map_coords.size(); i++) {
            auto rcp = map_coords.at(i).front();
            auto ccp = map_coords.at(i).back();

            auto rc_npts = rcp.size();
            auto cc_npts = ccp.size();

            // initialze to first obs map's min/max
            if (i == 0) {
                min_row = rcp(0);
                max_row = rcp(rc_npts-1);

                min_col = ccp(0);
                max_col = ccp(cc_npts-1);
            }            
            // see if current min/max is larger than previous
            // and replace if so
            else {
                if (rcp(0) < min_row) {
                    min_row = rcp(0);
                }

                if (rcp(rc_npts-1) > max_row) {
                    max_row = rcp(rc_npts-1);
                }

                if (ccp(0) < min_col) {
                    min_col = ccp(0);
                }

                if (ccp(cc_npts-1) > max_col) {
                    max_col = ccp(cc_npts-1);
                }
            }
        }

        // get number of rows
        Eigen::Index xminpix = ceil(abs(min_row/pixel_size));
        Eigen::Index xmaxpix = ceil(abs(max_row/pixel_size));
        xmaxpix = std::max(xminpix,xmaxpix);
        // always even
        nrows = 2.*xmaxpix + 4;

        // get number of cols
        Eigen::Index yminpix = ceil(abs(min_col/pixel_size));
        Eigen::Index ymaxpix = ceil(abs(max_col/pixel_size));
        ymaxpix = std::max(yminpix,ymaxpix);
        // always even
        ncols = 2.*ymaxpix + 4;

        // set coadded physical coordinate vectors
        rcphys = (Eigen::VectorXd::LinSpaced(nrows,0,nrows-1).array() -
                (nrows)/2.)*pixel_size;
        ccphys = (Eigen::VectorXd::LinSpaced(ncols,0,ncols-1).array() -
                (ncols)/2.)*pixel_size;

        SPDLOG_INFO("coadd map buffer nrows {} ncols {}", nrows, ncols);

        // exit if individual maps are too far apart
        if (nrows > 36000 || ncols > 36000 ||  nrows*ncols > 1.e9) {
            SPDLOG_INFO("map is too big: [{} {}]", nrows, ncols);
            std::exit(EXIT_FAILURE);
          }

        // resize the maps (nobs, [nrows, ncols])
        for (Eigen::Index i=0; i<map_count; i++) {
            signal.push_back(Eigen::MatrixXd::Zero(nrows, ncols));
            weight.push_back(Eigen::MatrixXd::Zero(nrows, ncols));
            kernel.push_back(Eigen::MatrixXd::Zero(nrows, ncols));
            coverage.push_back(Eigen::MatrixXd::Zero(nrows, ncols));
        }

        // resize noise maps (nobs, [nrows, ncols, nnoise])
        for (Eigen::Index i=0; i<map_count; i++) {
            noise.push_back(Eigen::Tensor<double,3>(nrows, ncols, nnoise));
            noise.at(i).setZero();
        }
    }

    template <class MB>
    void coadd(MB &mb, const double dfsmp, const bool run_kernel) {
        // offset between CMB and MB physical coordinates
        int deltai = (mb.rcphys(0) - rcphys(0))/pixel_size;
        int deltaj = (mb.ccphys(0) - ccphys(0))/pixel_size;

        // coadd the MB into the CMB
        for (Eigen::Index mi=0; mi<map_count; mi++) {
            // weight += weight
            weight.at(mi).block(deltai, deltaj, mb.nrows, mb.ncols) =
                    weight.at(mi).block(deltai, deltaj, mb.nrows, mb.ncols).array() +
                    mb.weight.at(mi).array();

            // signal += signal*weight
            signal.at(mi).block(deltai, deltaj, mb.nrows, mb.ncols) =
                    signal.at(mi).block(deltai, deltaj, mb.nrows, mb.ncols).array() +
                    (mb.weight.at(mi).array()*mb.signal.at(mi).array()).array();

            if (run_kernel) {
                // kernel += kernel*weight
                kernel.at(mi).block(deltai, deltaj, mb.nrows, mb.ncols) =
                        kernel.at(mi).block(deltai, deltaj, mb.nrows, mb.ncols).array() +
                        (mb.weight.at(mi).array()*mb.kernel.at(mi).array()).array();
            }

            // signal += fsmp*weight
            coverage.at(mi).block(deltai, deltaj, mb.nrows, mb.ncols) +=
                    mb.weight.at(mi)*1./dfsmp;
        }
    }

    void normalize_maps(const bool run_kernel) {
        /*for (Eigen::Index mc = 0; mc < map_count; mc++) {
            signal.at(mc) = (weight.at(mc).array() == 0).select(0, signal.at(mc).array() / weight.at(mc).array());
            if (run_kernel) {
                kernel.at(mc) = (weight.at(mc).array() == 0).select(0, kernel.at(mc).array() / weight.at(mc).array());
            }
        }*/

        // normalize signal and kernel
        for (Eigen::Index mc=0; mc<map_count; mc++) {
            for (Eigen::Index i=0; i<nrows; i++) {
                for (Eigen::Index j=0; j<ncols; j++) {
                    auto pixel_weight = weight.at(mc)(i,j);
                    if (pixel_weight != pixel_weight) {
                            SPDLOG_INFO("bad pixel weight {}", pixel_weight);
                    }
                    if (pixel_weight != 0. && pixel_weight == pixel_weight) {
                        signal.at(mc)(i,j) = (signal.at(mc)(i,j)) / pixel_weight;
                        if (run_kernel) {
                            kernel.at(mc)(i,j) = (kernel.at(mc)(i,j)) / pixel_weight;
                        }
                    }
                    else {
                        signal.at(mc)(i,j) = 0;
                        if (run_kernel) {
                            kernel.at(mc)(i,j) = 0;
                        }
                    }
                }
            }
        }

        // normalize noise maps
        for (Eigen::Index k=0;k<nnoise;k++) {
            for (Eigen::Index mc=0; mc<map_count; mc++) {
                for (Eigen::Index i=0; i<nrows; i++) {
                    for (Eigen::Index j=0; j<ncols; j++) {
                        auto pixel_weight = weight.at(mc)(i,j);
                        if (pixel_weight != pixel_weight) {
                                SPDLOG_INFO("bad pixel weight {}", pixel_weight);
                        }
                        if (pixel_weight != 0. && pixel_weight == pixel_weight) {
                            noise.at(mc)(i,j,k) = (noise.at(mc)(i,j,k)) / pixel_weight;
                        }
                        else {
                            noise.at(mc)(i,j,k) = 0;
                        }
                    }
                }
            }
        }
    }

    void normalize_noise_map_errors() {
        for (int m=0; m<map_count; m++) {
            Eigen::VectorXd nfacs;
            nfacs.setZero(nnoise);
            nfac.push_back(std::move(nfacs));

            for (int k=0; k<nnoise; k++) {
                Eigen::MatrixXd wt = weight.at(m);
                double weight_cut = engine_utils::find_weight_threshold(wt,cov_cut);

                double counter=0;
                double sig_of_map=0.;
                for (int i=0; i<nrows; i++) {
                      for (int j=0; j<ncols; j++) {
                          if (wt(i,j) >= weight_cut) {
                              counter++;
                              sig_of_map += pow(noise.at(m)(i,j,k),2);
                          }
                      }
                }
                sig_of_map /= (counter-1);
                sig_of_map = sqrt(sig_of_map);

                double mean_sqerr=0;
                counter=0.;
                for (int i=0; i<nrows; i++) {
                    for (int j=0; j<ncols; j++){
                        if (wt(i,j) >= weight_cut) {
                            counter++;
                            mean_sqerr += (1./wt(i,j));
                        }
                    }
                }
                mean_sqerr /= counter;
                nfac.back()(k) = (1./pow(sig_of_map,2.))*mean_sqerr;
            }
        }
    }

    void normalize_errors() {
        for (int m=0; m<map_count; m++) {
            Eigen::MatrixXd wt = weight.at(m);
            double weight_cut = engine_utils::find_weight_threshold(wt,cov_cut);

            double mean_sqerr=0.;
            int counter=0;
            for (int i=0; i<nrows; i++) {
                for (int j=0; j<ncols; j++) {
                    if (wt(i,j) >= weight_cut){
                        mean_sqerr += (1./wt(i,j));
                        counter++;
                    }
                }
            }

            mean_sqerr /= counter;
            double nfac = (1./pow(average_filtered_rms(m),2.))*mean_sqerr;
            SPDLOG_INFO("renormalization factor = {}",nfac);

            weight.at(m) = weight.at(m)*nfac;
        }
    }

    void calc_average_filtered_rms() {
        average_filtered_rms.resize(map_count);

        for (Eigen::Index m = 0; m<map_count; m++) {
            Eigen::VectorXd map_rms(nnoise);
            for (int k=0; k<nnoise; k++) {
                Eigen::MatrixXd wt = weight.at(m)*nfac.at(m)(k);
                double weight_cut = engine_utils::find_weight_threshold(wt,cov_cut);

                int counter = 0;
                double rms = 0.;
                for (int i=0; i<nrows; i++) {
                    for (int j=0; j<ncols; ++j) {
                        if (wt(i,j) > weight_cut) {
                            counter++;
                            rms += pow(noise.at(m)(i,j,k),2);
                        }
                    }
                }

                rms /= (counter);
                map_rms(k) = sqrt(rms);
                SPDLOG_INFO("Filtered noise rms {} from noise map {} map {}", map_rms(k), k, m);
            }
            average_filtered_rms(m) = map_rms.mean();
        }
    }
};
