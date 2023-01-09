#pragma once

#include <string>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/Splines>

#include <tula/algorithm/mlinterp/mlinterp.hpp>
#include <tula/logging.h>

#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/utils.h>

#include <citlali/core/utils/gauss_models.h>
#include <citlali/core/utils/fitting.h>

namespace mapmaking {

class WienerFilter {
public:
    std::string template_type;
    bool normalize_error, uniform_weight;
    bool run_lowpass;

    bool run_kernel;
    int nloops;

    std::map<std::string, double> gaussian_template_fwhm_rad;

    int nr, nc;
    double diffr, diffc;
    double denom_limit = 1.e-4;

    std::string exmode;

    Eigen::MatrixXd rr, vvq, denom, nume;
    Eigen::MatrixXd mflt;
    Eigen::MatrixXd tplate;

    template<class CMB>
    void make_gaussian_template(CMB &cmb, const double);

    template<class CMB, class CD>
    void make_symmetric_template(CMB &cmb, const int, CD &);

    template<class CMB, class CD>
    void make_template(CMB &cmb, CD &calib_data, const double gaussian_template_fwhm_rad, const int map_num) {
        // make sure new wiener filtered maps have even dimensions
        nr = 2*(cmb.n_rows/2);
        nc = 2*(cmb.n_cols/2);

        // x and y spacing should be equal
        diffr = abs(cmb.rows_tan_vec(1) - cmb.rows_tan_vec(0));
        diffc = abs(cmb.cols_tan_vec(1) - cmb.cols_tan_vec(0));

        if (template_type=="highpass") {
            SPDLOG_INFO("creating template with highpass only");
            tplate.setZero(nr,nc);
            tplate(0,0) = 1;
        }

        else if (template_type=="gaussian") {
            SPDLOG_INFO("creating gaussian template");
            make_gaussian_template(cmb, gaussian_template_fwhm_rad);
        }

        else {
            make_symmetric_template(cmb, map_num, calib_data);
        }
    }

    template<class CMB>
    void calc_rr(CMB &cmb, const int map_num) {
        if (uniform_weight) {
            rr = Eigen::MatrixXd::Ones(nr, nc);
        }
        else {
            rr = sqrt(cmb.weight.at(map_num).array());
        }
    }

    template <class CMB>
    void calc_vvq(CMB &, const int);
    void calc_numerator();
    void calc_denominator();

    template<class CMB>
    void run_filter(CMB &cmb, const int map_num) {
        SPDLOG_INFO("calculating rr");
        calc_rr(cmb, map_num);
        SPDLOG_INFO("calculating vvq");
        calc_vvq(cmb, map_num);
        SPDLOG_INFO("calculating denominator");
        calc_denominator();
        SPDLOG_INFO("calculating numerator");
        calc_numerator();
    }

    template<class CMB>
    void filter_coaddition(CMB &cmb, const int map_num) {
        SPDLOG_INFO("filtering coaddition");
        if (run_kernel) {
            SPDLOG_INFO("filtering kernel");
            mflt = cmb.kernel.at(map_num);
            uniform_weight = true;
            run_filter(cmb, map_num);

            for (int i=0; i<nc; i++) {
                for (int j=0; j<nr; j++) {
                    if (denom(j,i) != 0.0) {
                        cmb.kernel.at(map_num)(j,i)=nume(j,i)/denom(j,i);
                    }
                    else {
                        cmb.kernel.at(map_num)(j,i)= 0.0;
                    }
                }
            }
        }

        SPDLOG_INFO("filtering signal");
        mflt = cmb.signal.at(map_num);
        uniform_weight = false;
        run_filter(cmb, map_num);

        for (int i=0; i<nc; i++) {
            for (int j=0; j<nr; j ++) {
                if (denom(j,i) != 0.0) {
                    cmb.signal.at(map_num)(j,i) = nume(j,i)/denom(j,i);
                }
                else {
                    cmb.signal.at(map_num)(j,i)= 0.0;
                }
            }
        }
        cmb.weight.at(map_num) = denom;
    }

    template<class CMB>
    void filter_noise(CMB &cmb, const int map_num, const int noise_num) {
        SPDLOG_INFO("filtering noise {}/{}",noise_num+1, cmb.noise.at(map_num).dimension(2));
        Eigen::Tensor<double,2> out = cmb.noise.at(map_num).chip(noise_num,2);
        mflt = Eigen::Map<Eigen::MatrixXd>(out.data(),out.dimension(0),out.dimension(1));
        calc_numerator();

        //Eigen::MatrixXd ratio = (denom.array() == 0).select(0, nume.array() / denom.array());
        Eigen::MatrixXd ratio(nr,nc);

        for (int i=0; i<nc; i++) {
            for (int j=0; j<nr; j++) {
                if (denom(j,i) != 0.0) {
                    ratio(j,i) = nume(j,i)/denom(j,i);
                }
                else {
                    ratio(j,i)= 0.0;
                }
            }
        }

        Eigen::TensorMap<Eigen::Tensor<double, 2>> in_tensor(ratio.data(), ratio.rows(), ratio.cols());
        cmb.noise.at(map_num).chip(noise_num,2) = in_tensor;
    }
};

template<class CMB>
void WienerFilter::make_gaussian_template(CMB &cmb, const double gaussian_template_fwhm_rad) {

    tplate.setZero(nr,nc);

    Eigen::VectorXd rgcut = cmb.rows_tan_vec;
    Eigen::VectorXd cgcut = cmb.cols_tan_vec;

    Eigen::MatrixXd dist(nr, nc);
    /*dist = (xgcut.replicate(1, nc).array().pow(2.0).matrix()
            + ygcut.replicate(nr, 1).array().pow(2.0).matrix())
               .array()
               .sqrt();*/

    for (Eigen::Index i=0; i<nc; i++) {
        for (Eigen::Index j=0; j<nr; j++) {
            dist(j,i) = sqrt(pow(rgcut(j),2) + pow(cgcut(i),2));
        }
    }

    Eigen::Index rcind, ccind;

    double mindist = dist.minCoeff(&rcind, &ccind);
    double sigma = gaussian_template_fwhm_rad / STD_TO_FWHM;

    std::vector<Eigen::Index> shift_indices = {-rcind, -ccind};

    tplate = exp(-0.5 * pow(dist.array() / sigma, 2.));
    tplate = engine_utils::shift_2D(tplate, shift_indices);
}

template<class CMB, class CD>
void WienerFilter::make_symmetric_template(CMB &cmb, const int map_num, CD &calib_data) {
    // collect what we need
    Eigen::VectorXd rgcut = cmb.rows_tan_vec;
    Eigen::VectorXd cgcut = cmb.cols_tan_vec;
    Eigen::MatrixXd tem = cmb.kernel.at(map_num);

    // set nparams for fit
    Eigen::Index nparams = 6;
    Eigen::VectorXd pfit;
    pfit.setZero(nparams);

    // declare fitter class for detector
    engine_utils::mapFitter map_fitter;
    // size of region to fit in pixels
    map_fitter.bounding_box_pix = 15;//bounding_box_pix;


    double init_fwhm = 5;//toltec_io.array_fwhm_arcsec[array]*ASEC_TO_RAD/cmb.pixel_size_rad;
    auto [det_params, det_perror, good_fit] =
        map_fitter.fit_to_gaussian<engine_utils::mapFitter::peakValue>(cmb.signal[map_num], cmb.weight[map_num], init_fwhm);

    pfit(1) = cmb.pixel_size_rad*(pfit(1) - (nc)/2);
    pfit(2) = cmb.pixel_size_rad*(pfit(2) - (nr)/2);

    Eigen::Index r = -std::round(pfit(2)/diffr);
    Eigen::Index c = -std::round(pfit(1)/diffc);

    std::vector<Eigen::Index> shift_indices = {r,c};
    tem = engine_utils::shift_2D(tem, shift_indices);

    Eigen::MatrixXd dist(nr,nc);
    for (Eigen::Index i=0; i<nc; i++) {
        for(Eigen::Index j=0; j<nr; j++) {
            dist(j,i) = sqrt(pow(rgcut(j),2)+pow(cgcut(i),2));
        }
    }

    Eigen::Index rcind, ccind;
    auto mindist = dist.minCoeff(&rcind,&ccind);

    // create new bins based on diffr
    int nbins = rgcut(nr-1)/diffr;
    Eigen::VectorXd binlow(nbins);
    for(int i=0;i<nbins;i++) {
        binlow(i) = (double) (i*diffr);
    }

    Eigen::VectorXd kone(nbins-1);
    kone.setZero();
    Eigen::VectorXd done(nbins-1);
    done.setZero();
    for (int i=0;i<nbins-1;i++) {
        int c=0;
        for (int j=0;j<nc;j++) {
            for (int k=0;k<nr;k++) {
                if (dist(k,j) >= binlow(i) && dist(k,j) < binlow(i+1)){
                    c++;
                    kone(i) += tem(k,j);
                    done(i) += dist(k,j);
                }
            }
        }
        kone(i) /= c;
        done(i) /= c;
    }

    // now spline interpolate to generate new template array
    tplate.resize(nr,nc);

    engine_utils::SplineFunction s(done, kone);

    for (Eigen::Index i=0; i<nc; i++) {
        for (Eigen::Index j=0; j<nr; j++) {
            Eigen::Index tj = (j-rcind)%nr;
            Eigen::Index ti = (i-ccind)%nc;
            Eigen::Index shiftj = (tj < 0) ? nr+tj : tj;
            Eigen::Index shifti = (ti < 0) ? nc+ti : ti;

            if (dist(j,i) <= s.x_max && dist(j,i) >= s.x_min) {
                tplate(shiftj,shifti) = s(dist(j,i));
            }
            else if (dist(j,i) > s.x_max) {
                tplate(shiftj,shifti) = kone(kone.size()-1);
            }
            else if (dist(j,i) < s.x_min) {
                tplate(shiftj,shifti) = kone(0);
            }
        }
    }
}

template <class CMB>
void WienerFilter::calc_vvq(CMB &cmb, const int map_num) {
    // psd and psd freq vectors
    Eigen::VectorXd qf = cmb.noise_psds.at(map_num);
    Eigen::VectorXd hp = cmb.noise_psd_freqs.at(map_num);

    // size of psd and psd freq vectors
    Eigen::Index npsd = hp.size();

    //modify the psd array to take out lowpassing and highpassing
      double maxhp = -1.;
      int maxhpind = 0;
      double qfbreak = 0.;
      double hpbreak = 0.;
      for(int i=0;i<npsd;i++) if(hp[i] > maxhp){
          maxhp = hp[i];
          maxhpind = i;
        }
      for(int i=0;i<npsd;i++) if(hp[i]/maxhp < 1.e-4){
          qfbreak = qf[i];
          break;
        }
      //flatten the response above the lowpass break
      int count=0;
      for(int i=0;i<npsd;i++) if(qf[i] <= 0.8*qfbreak) count++;
      if(count > 0){
        for(int i=0;i<npsd;i++){
          if(qfbreak > 0){
            if(qf[i] <= 0.8*qfbreak) hpbreak = hp[i];
            if(qf[i] > 0.8*qfbreak) hp[i] = hpbreak;
          }
        }
      }
      //flatten highpass response if present
      if(maxhpind > 0) for(int i=0;i<maxhpind;i++) hp[i] = maxhp;

    // modify the psd array to take out lowpassing+highpassing
    /*Eigen::Index maxhpind;
    double qfbreak = 0.;
    double hpbreak = 0.;

    // get max value and index of hp
    auto maxhp = hp.maxCoeff(&maxhpind);

    if (maxhp < -1) {
        maxhp = -1;
        maxhpind = 0;
    }

    // find the frequency where hp falls to 1e-4 its max value
    for (int i = 0; i < npsd; i++)
        if (hp(i) / maxhp < 1.e-4) {
            qfbreak = qf(i);
            break;
        }
    // get the total number of points where the frequency is less than 0.8*qfbreak
    auto count = (qf.array() <= 0.8 * qfbreak).count();

    // set the region of hp corresponding to the frequency range above 0.8*qfbreak to
    // the value of hp at 0.8*qfbreak (i.e. flatten it).  We can do this as a vector
    // since the frequencies are monotonically increasing.
    if (count > 0) {
        if (qfbreak > 0) {
            Eigen::Index start = 0.8 * qfbreak;
            Eigen::Index size = hp.size() - 0.8 * qfbreak;
            hp.segment(start, size).setConstant(hp(start));
        }
    }
    */

    // do the same for the region below the maximum hp
    //hp.head(maxhpind).setConstant(maxhp);

    // set up q-space
    double rsize = nr * diffr;
    double csize = nc * diffc;
    double diffqr = 1. / rsize;
    double diffqc = 1. / csize;

    Eigen::VectorXd qr(nr), qc(nc);
    Eigen::MatrixXd qmap(nr, nc);

    /*shift qr */
    //qr = Eigen::VectorXd::LinSpaced(nr, -(nr - 1) / 2, (nr - 1) / 2) * diffqr;
    /*shift qc */
    //qc = Eigen::VectorXd::LinSpaced(nc, -(nc - 1) / 2, (nc - 1) / 2) * diffqc;

    for(int i=0;i<nr;i++) qr[i] = diffqr*(i-(nr-1)/2);
      for(int i=0;i<nc;i++) qc[i] = diffqc*(i-(nc-1)/2);


      std::vector<Eigen::Index> shift_1 = {-(nr-1)/2};
    engine_utils::shift_1D(qr, shift_1);
      std::vector<Eigen::Index> shift_2 = {-(nc-1)/2};
    engine_utils::shift_1D(qc, shift_2);

    // make qmap by replicating qc and qr ncols and nrows times respectively.  Faster than a for loop for expected
    // map dimensions
    //qmap = (qr.replicate(1, nc).array().pow(2.0).matrix() + qc.replicate(nr, 1).array().pow(2.0).matrix()).array().sqrt();

    for (int i=0; i<nc; i++) {
        for (int j=0; j<nr; j++) {
            qmap(j,i) = sqrt(pow(qr[j],2)+pow(qc[i],2));
        }
    }

    Eigen::MatrixXd psdq;
    psdq.setZero(nr, nc);

    if (run_lowpass) {
        psdq.setOnes();
    }

    else {
        Eigen::Matrix<Eigen::Index, 1, 1> nhp;
        nhp << hp.size();

        // interpolate onto psdq
        Eigen::Index interp_pts = 1;
        for (int i = 0; i < nc; i++) {
            for (int j = 0; j < nr; j++) {
                if ((qmap(j, i) <= qf(qf.size() - 1)) && (qmap(j, i) >= qf(0))) {
                    mlinterp::interp<mlinterp::rnatord>(nhp.data(), interp_pts,
                                     hp.data(), psdq.data() + nr * i + j,
                                     qf.data(), qmap.data() + nr * i + j);

                }

                else if (qmap(j, i) > qf(qf.size() - 1)) {
                    psdq(j, i) = hp(hp.size() - 1);
                }
                else if (qmap(j, i) < qf(0)) {
                    psdq(j, i) = hp(0);
                }
            }
        }

    // find the minimum value of hp
    auto lowval = 0;//hp.minCoeff();

    // set all the points in psdq smaller than lowval to lowval
    //(psdq.array() < lowval).select(lowval, psdq);

    for(int i=0;i<hp.size();i++) if(hp[i] < lowval) lowval=hp[i];
        for(int i=0;i<nc;i++)
          for(int j=0;j<nr;j++)
            if (psdq(j,i) < lowval) psdq(j,i)=lowval;
    }

    // normalize the power spectrum psdq and place into vvq
    //vvq = psdq/psdq.sum();
    vvq.setZero(nr,nc);
    double totpsdq=0.;
    for(int i=0;i<nc;i++) for(int j=0;j<nr;j++) totpsdq += psdq(j,i);
    for(int i=0;i<nc;i++) for(int j=0;j<nr;j++) vvq(j,i) = psdq(j,i)/totpsdq;
}

void WienerFilter::calc_numerator() {
    nume.setZero(nr,nc);
    // normalization for fft
    double fftnorm = 1. / nr / nc;

    Eigen::MatrixXcd in(nr, nc);
    Eigen::MatrixXcd out(nr, nc);

    //Eigen::Map<Eigen::VectorXd> rr_vec(rr.data(),rr.rows(),rr.cols());
    //Eigen::Map<Eigen::VectorXd> mflt_vec(mflt.data(),mflt.rows(),mflt.cols());

    in.real() = rr.array() * mflt.array();
    in.imag().setZero();

    out = engine_utils::fft<engine_utils::forward>(in, exmode);
    out = out * fftnorm;

    in.real() = out.real().array() / vvq.array();
    in.imag() = out.imag().array() / vvq.array();

    out = engine_utils::fft<engine_utils::inverse>(in, exmode);

    in.real() = out.real().array() * rr.array();
    in.imag().setZero();

    out = engine_utils::fft<engine_utils::forward>(in, exmode);
    out = out * fftnorm;

    // copy of out
    Eigen::MatrixXcd qqq = out;

    in.real() = tplate;
    in.imag().setZero();

    out = engine_utils::fft<engine_utils::forward>(in, exmode);
    out = out * fftnorm;

    in.real() = out.real().array() * qqq.real().array() + out.imag().array() * qqq.imag().array();
    in.imag() = -out.imag().array() * qqq.real().array() + out.real().array() * qqq.imag().array();

    out = engine_utils::fft<engine_utils::inverse>(in, exmode);

    // populate numerator
    nume = out.real();
}

void WienerFilter::calc_denominator() {
    double fftnorm = 1. / nr / nc;
    denom.setZero(nr, nc);
    //Eigen::VectorXcd in(nr * nc);
    //Eigen::VectorXcd out(nr * nc);

    Eigen::MatrixXcd in(nr, nc);
    Eigen::MatrixXcd out(nr, nc);

    if (uniform_weight) {
        in.real() = tplate;
        in.imag().setZero();

        out = engine_utils::fft<engine_utils::forward>(in, exmode);
        out = out * fftnorm;

       // auto d = ((out.real().array() * out.real().array() + out.imag().array() * out.imag().array()) / vvq.array()).sum();

        double d=0;
        for(int i=0;i<nc;i++)
              for(int j=0;j<nr;j++){
                d += (out.real()(j,i)*out.real()(j,i) + out.imag()(j,i)*out.imag()(j,i))/vvq(j,i);
              }
        denom.setConstant(d);
    }

    else {

        //Eigen::Map<Eigen::VectorXd> vvq_vec(vvq.data(), vvq.rows(),vvq.cols());
        //in.real() = pow(vvq_vec.array(),-1.);
        in.real() = pow(vvq.array(), -1);
        in.imag().setZero();

        out = engine_utils::fft<engine_utils::inverse>(in, exmode);

        Eigen::VectorXd zz2d(nr * nc);
        //Eigen::Map<Eigen::VectorXd> out_vec(out.real().data(), nr*nc);

        for(int i=0;i<nc;i++)
            for(int j=0;j<nr;j++){
              int ii = nr*i+j;
              zz2d(ii) = (out.real()(j,i));
            }

        //zz2d = abs(out_vec.array());

        Eigen::VectorXd ss_ord = zz2d;
        auto sorted = engine_utils::sorter(ss_ord);

        for(int i=0;i<nc;i++)
            for(int j=0;j<nr;j++){
              int ii = nr*i+j;
              zz2d(ii) = (out.real()(j,i));
            }

        // number of iterations for convergence
        nloops = nr * nc / 100;

        ////SPDLOG_INFO("denom g");

        // flag for convergence
        bool done = false;

        tula::logging::progressbar pb0(
            [](const auto &msg) { SPDLOG_INFO("{}", msg); }, 60,
            "denominator progress ");

        for (int k = 0; k < nc; k++) {
            for (int l = 0; l < nr; l++) {
                if (!done) {
                    Eigen::MatrixXcd in2(nr, nc);
                    Eigen::MatrixXcd out2(nr, nc);

                    /*May need to flip directions due to order of matrix storage order*/
                    int kk = nr * k + l;
                    if (kk >= nloops) {
                        continue;
                    }

                    auto shifti = std::get<1>(sorted[nr * nc - kk - 1]);

                    // changed nr
                    double r_shift_n = shifti / nr;
                    double c_shift_n = shifti % nr;

                    Eigen::Index shift_1 = -r_shift_n;
                    Eigen::Index shift_2 = -c_shift_n;

                    std::vector<Eigen::Index> shift_indices = {shift_1, shift_2};


                    Eigen::MatrixXd in_prod = tplate.array() * engine_utils::shift_2D(tplate, shift_indices).array();

                    /*in2.real() = Eigen::Map<Eigen::VectorXd>(
                        (in_prod).data(),
                        nr * nc);
                    in2.imag().setZero();
                    */

                    in2.real() = in_prod;
                    in2.imag().setZero();

                    out2 = engine_utils::fft<engine_utils::forward>(in2, exmode);
                    out2 = out2 * fftnorm;

                    Eigen::MatrixXcd ffdq(nr,nc);

                    ffdq.real() = out2.real();
                    ffdq.imag() = out2.imag();


                    in_prod = rr.array() * engine_utils::shift_2D(rr, shift_indices).array();

                    //in2.real() = Eigen::Map<Eigen::VectorXd>(in_prod.data(), nr * nc);
                    //in2.imag().setZero();

                    in2.real() = in_prod;
                    in2.imag().setZero();

                    out2 = engine_utils::fft<engine_utils::forward>(in2, exmode);
                    out2 = out2 * fftnorm;

                    in2.real() = ffdq.real().array() * out2.real().array() + ffdq.imag().array() * out2.imag().array();
                    in2.imag() = -ffdq.imag().array() * out2.real().array() + ffdq.real().array() * out2.imag().array();

                    out2 = engine_utils::fft<engine_utils::inverse>(in2, exmode);

                    Eigen::MatrixXd updater = zz2d(shifti) * out2.real() * fftnorm;

                    denom = denom + updater;

                    pb0.count(nloops, nloops / 100);

                    if ((kk % 100) == 1) {
                        double max_ratio = -1;
                        double maxdenom;

                        maxdenom = denom.maxCoeff();
                        for (int i = 0; i < nr; i++) {
                            for (int j = 0; j < nc; j++) {
                                if (denom(i, j) > 0.01 * maxdenom) {
                                    if (abs(updater(i, j) / denom(i, j)) > max_ratio)
                                        max_ratio = abs(updater(i, j) / denom(i, j));
                                }
                            }
                        }

                        if (((kk >= 500) && (max_ratio < 0.0002)) || max_ratio < 1e-10) {
                            SPDLOG_INFO("done.  max_ratio={} {} iterations", max_ratio, kk);
                            done = true;
                        }
                        else {
                            //SPDLOG_INFO("completed iteration {} of {}. max_ratio={}",
                                        //kk, nloops, max_ratio);
                        }
                    }
                }
            }
        }

        SPDLOG_INFO("zeroing out any values < {} in denominator", denom_limit);
        SPDLOG_INFO("min denom {}",denom.minCoeff());
        //(denom.array() < denom_limit).select(0, denom);
        for (Eigen::Index i=0;i<nr;i++) {
            for (Eigen::Index j=0;j<nc;j++) {
                if (denom(i,j) < 1.e-4) denom(i,j) = 0;
            }
        }
        SPDLOG_INFO("min denom after {}",denom.minCoeff());

    }
}

} // namespace mapmaking
