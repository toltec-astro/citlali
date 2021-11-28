#pragma once

#include <Eigen/Core>

#include <citlali/core/utils/utils.h>

class PSD {
public:
    double cov_cut;

    Eigen::VectorXd rcphys, ccphys;
    Eigen::MatrixXd w;
    Eigen::VectorXd psd, psd_freq;
    Eigen::MatrixXd psd2d, psd2d_freq;

    template <typename Derived>
    void calc_map_psd(Eigen::DenseBase<Derived> &);
};

template <typename Derived>
void PSD::calc_map_psd(Eigen::DenseBase<Derived> &in) {

    auto weight_threshold = engine_utils::find_weight_threshold(in, cov_cut);
    auto [cut_x_range, cut_y_range] = engine_utils::set_coverage_cut_ranges(in, weight_threshold);

    // make sure coverage cut map has an even number
    // of rows and columns

    Eigen::Index nx = cut_x_range(1)-cut_x_range(0)+1;
    Eigen::Index ny = cut_y_range(1)-cut_y_range(0)+1;
    Eigen::Index cxr0 = cut_x_range(0);
    Eigen::Index cyr0 = cut_y_range(0);
    Eigen::Index cxr1 = cut_x_range(1);
    Eigen::Index cyr1 = cut_y_range(1);

    if (nx % 2 == 1) {
      cxr1 = cut_x_range(1) - 1;
      nx--;
    }

    if (ny % 2 == 1) {
      cyr1 = cut_y_range(1) - 1;
      ny--;
    }

    double diffx = rcphys(1) - rcphys(0);
    double diffy = ccphys(1) - ccphys(0);
    double xsize = diffx * nx;
    double ysize = diffy * ny;
    double diffqx = 1. / xsize;
    double diffqy = 1. / ysize;

    auto out = engine_utils::fft2w<engine_utils::forward>(in, nx,ny);
    out = out*diffx*diffy;

    Eigen::MatrixXd h = diffqx*diffqy*out.cwiseAbs2();

    w = Eigen::Map<Eigen::VectorXd>(h.data(),nx*ny);

    // vectors of frequencies
    Eigen::VectorXd qx(nx);
    Eigen::VectorXd qy(ny);

    int shift = nx/2-1;
    Eigen::Index index;
    for (int i=0; i<nx; i++) {
        index = i-shift;
        if (index < 0) {
            index += nx;
        }
        qx(index) = diffqx*(i-(nx/2-1));
    }

    shift = ny/2-1;
    for (int i=0; i<ny; i++) {
        index = i-shift;
        if (index < 0) {
            index += ny;
        }
        qy(index) = diffqy*(i-(ny/2-1));
    }

    // shed first row and column of h, qx, qy
    Eigen::MatrixXd pmfq(nx-1,ny-1);
    for (int i=1; i<nx; i++) {
        for (int j=1; j<ny; j++) {
            pmfq(i-1,j-1) = h(i,j);
        }
    }

    for (int i=0; i<nx-1; i++) {
        qx(i) = qx(i+1);
    }
    for (int j=0; j<ny-1; j++) {
        qy(j) = qy(j+1);
    }

    // matrices of frequencies and distances
    Eigen::MatrixXd qmap(nx-1,ny-1);
    Eigen::MatrixXd qsymm(nx-1,ny-1);

    for (int i=1; i<nx; i++) {
        for(int j=1; j<ny; j++) {
            qmap(i-1,j-1) = sqrt(pow(qx(i),2) + pow(qy(j),2));
            qsymm(i-1,j-1) = qx(i)*qy(j);
        }
    }

    // find max of nx and ny and correspoinding diffq
    int nn;
    double diffq;
    if (nx > ny) {
        nn = nx/2+1;
        diffq = diffqx;
    }
    else {
      nn = ny/2+1;
      diffq = diffqy;
    }

    // generate the final vector of frequencies
    psd_freq.resize(nn);
    for (int i=0; i<nn; i++) {
        psd_freq[i] = diffq*(i + 0.5);
    }

    // pack up the final vector of psd values
    psd.resize(nn);
    for (int i=0; i<nn; i++){
        int countS=0;
        int countA=0;
        double psdarrS=0.;
        double psdarrA=0.;
        for (int j=0;j<nx-1;j++)
            for (int k=0;k<ny-1;k++) {
                if((int) (qmap(j,k) / diffq) == i && qsymm(j,k) >= 0.){
                    countS++;
                    psdarrS += pmfq(j,k);
                }

                if((int) (qmap(j,k) / diffq) == i && qsymm(j,k) < 0.){
                    countA++;
                    psdarrA += pmfq(j,k);
                }
            }
            if(countS != 0) psdarrS /= countS;
            if(countA != 0) psdarrA /= countA;
            psd(i) = std::min(psdarrS,psdarrA);
    }

    // smooth the psd with a 10-element boxcar filter
    Eigen::VectorXd tmp(nn);
    engine_utils::smooth(psd, tmp, 10);
    psd = tmp;

    psd2d = pmfq;
    psd2d_freq = qmap;
}
