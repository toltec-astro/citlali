#pragma once

#include <Eigen/Core>

#include <citlali/core/utils/utils.h>

class PSD {
public:
    double cov_cut;

    Eigen::MatrixXd w;
    Eigen::VectorXd psd, psd_freq;
    Eigen::MatrixXd psd2d, psd2d_freq;

    template <typename DerivedA, typename DerivedB, typename DerivedC>
    void calc_map_psd(Eigen::DenseBase<DerivedA> &, Eigen::DenseBase<DerivedB> &,
                      Eigen::DenseBase<DerivedC> &, Eigen::DenseBase<DerivedC> &);
};

template <typename DerivedA, typename DerivedB, typename DerivedC>
void PSD::calc_map_psd(Eigen::DenseBase<DerivedA> &in, Eigen::DenseBase<DerivedB> &wt,
                       Eigen::DenseBase<DerivedC> &rcphys, Eigen::DenseBase<DerivedC> &ccphys) {

    auto weight_threshold = engine_utils::find_weight_threshold(wt, cov_cut);
    auto [cut_row_range, cut_col_range] = engine_utils::set_coverage_cut_ranges(wt, weight_threshold);

    // make sure coverage cut map has an even number of rows and cols
    Eigen::Index nr = cut_row_range(1) - cut_row_range(0) + 1;
    Eigen::Index nc = cut_col_range(1) - cut_col_range(0) + 1;
    Eigen::Index crr0 = cut_row_range(0);
    Eigen::Index ccr0 = cut_col_range(0);
    Eigen::Index crr1 = cut_row_range(1);
    Eigen::Index ccr1 = cut_col_range(1);

    if (nr % 2 == 1) {
      crr1 = cut_row_range(1) - 1;
      nr--;
    }

    if (nc % 2 == 1) {
      ccr1 = cut_col_range(1) - 1;
      nc--;
    }

    double diffr = rcphys(1) - rcphys(0);
    double diffc = ccphys(1) - ccphys(0);
    double rsize = diffr * nr;
    double csize = diffc * nc;
    double diffqr = 1. / rsize;
    double diffqc = 1. / csize;

    Eigen::MatrixXcd block(nr, nc);
    block.real() = in.block(crr0, ccr0, nr, nc);
    block.imag().setZero();

    block.real() = block.real().array() * engine_utils::hanning(nr, nc).array();

    auto out = engine_utils::fft2w<engine_utils::forward>(block, nr, nc);

    out = out*diffr*diffc;

    //Eigen::Map<Eigen::VectorXcd> out_vec(out.data(),nr*nc);

    //Eigen::VectorXd w = diffqr*diffqc*out.cwiseAbs2();
    //Eigen::MatrixXd h = Eigen::Map<Eigen::MatrixXd>(w.data(), nr, nc);

    Eigen::MatrixXd h = diffqr*diffqc*out.cwiseAbs2();

    // vectors of frequencies
    Eigen::VectorXd qr(nr);
    Eigen::VectorXd qc(nc);

    int shift = nr/2 - 1;
    Eigen::Index index;
    for (Eigen::Index i=0; i<nr; i++) {
        index = i-shift;
        if (index < 0) {
            index += nr;
        }
        qr(index) = diffqr*(i-(nr/2-1));
    }

    shift = nc/2 - 1;
    for (Eigen::Index i=0; i<nc; i++) {
        index = i-shift;
        if (index < 0) {
            index += nc;
        }
        qc(index) = diffqc*(i-(nc/2-1));
    }

    // shed first row and column of h, qr, qc
    Eigen::MatrixXd pmfq(nr-1,nc-1);
    for (Eigen::Index i=1; i<nc; i++) {
        for (int j=1; j<nr; j++) {
            pmfq(j-1,i-1) = h(j,i);
        }
    }

    for (Eigen::Index i=0; i<nr-1; i++) {
        qr(i) = qr(i+1);
    }
    for (int j=0; j<nc-1; j++) {
        qc(j) = qc(j+1);
    }

    // matrices of frequencies and distances
    Eigen::MatrixXd qmap(nr-1,nc-1);
    Eigen::MatrixXd qsymm(nr-1,nc-1);

    for (Eigen::Index i=1; i<nc; i++) {
        for(int j=1; j<nr; j++) {
            qmap(j-1,i-1) = sqrt(pow(qr(j),2) + pow(qc(i),2));
            qsymm(j-1,i-1) = qr(j)*qc(i);
        }
    }

    // find max of nr and nc and correspoinding diffq
    int nn;
    double diffq;
    if (nr > nc) {
        nn = nr/2+1;
        diffq = diffqr;
    }
    else {
      nn = nc/2+1;
      diffq = diffqc;
    }

    // generate the final vector of frequencies
    psd_freq.resize(nn);
    for (Eigen::Index i=0; i<nn; i++) {
        psd_freq(i) = diffq*(i + 0.5);
    }

    // pack up the final vector of psd values
    psd.setZero(nn);
    for (Eigen::Index i=0; i<nn; i++) {
        int count_s = 0;
        int count_a = 0;
        double psdarr_s = 0.;
        double psdarr_a = 0.;
        for (int j=0; j<nc-1; j++) {
            for (int k=0; k<nr-1; k++) {
                if ((int) (qmap(k,j) / diffq) == i && qsymm(k,j) >= 0.){
                    count_s++;
                    psdarr_s += pmfq(k,j);
                }

                if ((int) (qmap(k,j) / diffq) == i && qsymm(k,j) < 0.){
                    count_a++;
                    psdarr_a += pmfq(k,j);
                }
            }
        }
            if (count_s != 0) {
                psdarr_s /= count_s;
            }
            if (count_a != 0) {
                psdarr_a /= count_a;
            }
            psd(i) = std::min(psdarr_s,psdarr_a);
    }

    // smooth the psd with a 10-element boxcar filter
    Eigen::VectorXd tmp(nn);
    engine_utils::smooth(psd, tmp, 10);
    psd = tmp;

    psd2d = pmfq;
    psd2d_freq = qmap;
}
