#pragma once
#include <eigen3/unsupported/Eigen/FFT>
#include "../../common_utils/src/utils/algorithm/mlinterp/mlinterp.hpp"

namespace mapmaking {

template<typename Derived>
std::vector<std::tuple<double, int>> sorter(Eigen::DenseBase<Derived> &vec){
    std::vector<std::tuple<double, int>> vis;
    Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(vec.size(),0,vec.size()-1);

    for(int i=0; i<vec.size(); i++){
        std::tuple<double, double> vec_and_val(vec[i], indices[i]);
        vis.push_back(vec_and_val);
    }

    std::sort(vis.begin(), vis.end(),
              [&](const std::tuple<double, int>& a, const std::tuple<double, int>& b) -> bool{
                  return std::get<0>(a) < std::get<0>(b);
              });

    return vis;
}

enum FFTdirection {
    forward = 0,
    backward = 1
};

template<FFTdirection direc, typename DerivedA>
Eigen::VectorXcd fft2w(Eigen::DenseBase<DerivedA> &vecIn, int nx, int ny){
    const int nRows = nx;//matIn.rows();
    const int nCols = ny;//matIn.cols();

    //Eigen::Map<Eigen::MatrixXd> matIn2(vecIn.derived().data(),nx,ny);

    Eigen::MatrixXcd matIn(nRows,nCols);

    for(int i =0;i<nx;i++){
        for(int j=0;j<ny;j++)
            matIn(i,j) = vecIn(ny*i+j);
    }

    Eigen::FFT<double> fft;
    fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
    //fft.SetFlag(Eigen::FFT<double>::Unscaled);
    Eigen::MatrixXcd matOut(nRows, nCols);

    for (int k = 0; k < nRows; ++k) {
        Eigen::VectorXcd tmpOut(nCols);
        if constexpr(direc == forward){
            fft.fwd(tmpOut, matIn.row(k));
        }
        else{
            fft.inv(tmpOut, matIn.row(k));
        }
        matOut.row(k) = tmpOut;
    }

    for (int k = 0; k < nCols; ++k) {
        Eigen::VectorXcd tmpOut(nRows);
        if constexpr(direc == forward){
            fft.fwd(tmpOut, matOut.col(k));
        }
        else{
            fft.inv(tmpOut, matOut.col(k));
        }
        matOut.col(k) = tmpOut;
    }

    Eigen::VectorXcd vec3(nx*ny);

    for(int i =0;i<nx;i++){
        for(int j=0;j<ny;j++)
            vec3(ny*i+j) = matOut(i,j);
    }

    return vec3;
}

bool shift(Eigen::VectorXd &vec, int n){
    int nx = vec.size();
    Eigen::VectorXd vec2(nx);
    for(int i=0;i<nx;i++){
        int ti = (i+n)%nx;
        int shifti = (ti < 0) ? nx+ti : ti;
        vec2[shifti] = vec[i];
    }
    for(int i=0;i<nx;i++) vec[i] = vec2[i];
    return 1;
}

//same yet again but returns just a single value of the shifted
//matrix at location (i,j)
double shift2(Eigen::MatrixXd &mat, int n1, int n2, int i, int j){
    int nx = mat.rows();
    int ny = mat.cols();
    int ti = (i+n1)%nx;
    int tj = (j+n2)%ny;
    int shifti = (ti < 0) ? nx+ti : ti;
    int shiftj = (tj < 0) ? ny+tj : tj;
    return mat(shifti,shiftj);
}

void shift3(Eigen::MatrixXd &mat, int n1, int n2){
    int nx = mat.rows();
    int ny = mat.cols();
    Eigen::MatrixXd mat2(nx, ny);
    for(int i=0;i<nx;i++)
        for(int j=0;j<ny;j++){
            int ti = (i+n1)%nx;
            int tj = (j+n2)%ny;
            int shifti = (ti < 0) ? nx+ti : ti;
            int shiftj = (tj < 0) ? ny+tj : tj;
            mat2(shifti,shiftj) = mat(i,j);
        }
    for(int i=0;i<nx;i++) for(int j=0;j<ny;j++) mat(i,j) = mat2(i,j);
}

class wiener {
public:
    ///storage for current and future wf
    Eigen::MatrixXd Denom;               ///<denominator to filter (calc once)
    Eigen::MatrixXd Nume;                ///<numerator to the filter (changes)
    Eigen::MatrixXd rr;
    Eigen::MatrixXd vvq;
    Eigen::MatrixXd tplate;

    psdclass psd;

    bool uniformWeight;          ///<set to force uniform weighting
                    ///<number of columns in coadded maps
    int nx;                      ///<xdim of wiener filtered maps
    int ny;                      ///<ydim of wiener filtered maps
    double diffx;
    double diffy;
    double GaussianTemplateFWHM = 3.88e-5;///<set to force uniform weighting
    bool getLowpassOnly = 0;

    ///methods
    wiener(MapStruct &cmap, std::shared_ptr<lali::YamlConfig> config_): config(std::move(config_)){
        nx = 2*(cmap.nrows/2);
        ny = 2*(cmap.ncols/2);

        bool getHighpassOnly = 0;
        bool getGaussianTemplate = 1;
        if(getHighpassOnly){
            tplate.resize(nx,ny);
            tplate.setZero();
            tplate(0,0)=1.;
        }
        else if(getGaussianTemplate){
            prepareGaussianTemplate(cmap);
        } else prepareTemplate(cmap);
    };

    std::shared_ptr<lali::YamlConfig> config;

    void prepareTemplate(MapStruct &cmap);
    void prepareGaussianTemplate(MapStruct &cmap);
    void filterCoaddition(MapStruct &cmap, psdclass &psd);
    void filterNoiseMaps(MapStruct &cmap);
    void calcRr(MapStruct &cmap);
    void calcVvq(psdclass &psd);
    void calcNumerator(Eigen::MatrixXd &mflt);
    bool calcDenominator();
    void simplewienerFilter2d();
    //  void gaussFilter(Coaddition *cmap);
    //~wiener();

};

void wiener::filterCoaddition(MapStruct &cmap, psdclass &psd){

    cmap.filteredkernel.resize(nx,ny);
    cmap.filteredsignal.resize(nx,ny);
    cmap.filteredweight.resize(nx,ny);

    //do the kernel first since it's simpler to calculate
    //the kernel calculation requires uniform weighting
    uniformWeight=1;
    calcRr(cmap);
    calcVvq(psd);
    calcDenominator();
    //for(int i=0;i<nx;i++) for(int j=0;j<ny;j++)
    //Eigen::Tensor<double,2> mflt= cmap.kernel;

    Eigen::MatrixXd mflt = Eigen::Map<Eigen::MatrixXd> (cmap.kernel.data(),cmap.kernel.dimension(0),cmap.kernel.dimension(1));

    calcNumerator(mflt);
    for(int i=0;i<nx;i++)
        for(int j=0;j<ny;j++){
            if (Denom(i,j) != 0.0)
                cmap.filteredkernel(i,j)=Nume(i,j)/Denom(i,j);
            else {
                cmap.filteredkernel(i,j)= 0.0;
                SPDLOG_INFO("Nan prevented");
            }
        }



    //Do the more complex precomputing for the signal map
    uniformWeight=0;
    calcRr(cmap);
    calcVvq(psd);

    calcDenominator();

    //Here is the signal map to be filtered
    //for(int i=0;i<nx;i++) for(int j=0;j<ny;j++)
    //mflt = cmap.signal;
    mflt = Eigen::Map<Eigen::MatrixXd> (cmap.signal.data(),cmap.signal.dimension(0),cmap.signal.dimension(1));

    //calculate the associated numerator
    calcNumerator(mflt);

    //replace the original images with the filtered images
    //note that the filtered weight map = Denom
    for(int i=0;i<nx;i++) for(int j=0;j<ny;j++){
            if (Denom(i,j) !=0.0)
                cmap.filteredsignal(i,j)=Nume(i,j)/Denom(i,j);
            else{
                //cerr<<"Nan avoided"<<endl;
                cmap.filteredsignal(i,j)=0.0;
            }
            cmap.filteredweight(i,j)=Denom(i,j);
        }
}

///Main driver code to filter a set of noise realizations.
/** This is the main driver for rapidly filtering the images from
    a complete set of coadded noise realizations.  Unlike
    the coaddition filtering above, this code directly writes
    the filtered noise map into the appropriate netcdf file.
    It requires that vvq, rr, tplate, and Denom are already
    calculated.
 **/
void wiener::filterNoiseMaps(MapStruct &cmap){
    //The strategy here is to loop through the noise files one
    //by one, pulling in the noise image, filtering it, and then
    //writing it back into the file.
    for(int k=0;k<cmap.NNoiseMapsPerObs;k++){
        // print out what's happening
        SPDLOG_INFO("WienerFilter(): Filtering noise map {}", k);

        //the input array dims
        int nrows = cmap.filterednoisemaps.dimension(0);
        int ncols = cmap.filterednoisemaps.dimension(1);

        //the actual noise matrix
        Eigen::MatrixXd noise(nrows, ncols);

        //do the filtering
        //Eigen::MatrixXd filterednoisemaps(nrows,ncols,0.);
        //Eigen::MatrixXd filteredweight(nrows,ncols,0.);
        calcNumerator(noise);
        for(int i=0;i<nx;i++) {
            for(int j=0;j<ny;j++){
                if(Denom(i,j) > 0){
                    cmap.filterednoisemaps(i,j,k)=Nume(i,j)/Denom(i,j);
                    cmap.filteredweight(i,j)=Denom(i,j);
                }
            }
        }
    }
}

///Calculation of the RR matrix
void wiener::calcRr(MapStruct &cmap){
    rr.resize(nx,ny);

    if(uniformWeight){
        rr.setOnes();//(nx,ny,1.);
    }
    else {
        //for(int i=0;i<nx;i++) for(int j=0;j<ny;j++)
        rr = Eigen::Map<Eigen::MatrixXd> (cmap.wtt.data(),cmap.wtt.dimension(0),cmap.wtt.dimension(1));
        rr = rr.array().sqrt();
    }
}

void wiener::calcVvq(psdclass &psd){
    //start by fetching the noise psd
    //string psdfile = ap->getAvgNoisePsdFile();
    //NcFile ncfid = NcFile(psdfile.c_str(), NcFile::ReadOnly);
    int npsd = psd.psd.size();
    Eigen::VectorXd qf(npsd);
    Eigen::VectorXd hp(npsd);
    //NcVar* psdv = ncfid.get_var("psd");
    //NcVar* psdfv = ncfid.get_var("psdFreq");
    //for(int i=0;i<npsd;i++){
      //  hp[i] = psdv->as_double(i);
        //qf[i] = psdfv->as_double(i);
    //}

    //modify the psd array to take out lowpassing and highpassing
    double maxhp=-1.;
    int maxhpind=0;
    double qfbreak=0.;
    double hpbreak=0.;
    //for(int i=0;i<npsd;i++) if(hp[i] > maxhp){
      //      maxhp = hp[i];
        //    maxhpind = i;
        //}
    maxhp = hp.maxCoeff();
    for(int i=0;i<npsd;i++) if(hp[i]/maxhp < 1.e-4){
            qfbreak = qf[i];
            break;
        }
    //flatten the response above the lowpass break
    int count=0;
    //for(int i=0;i<npsd;i++){
    //if(qf[i] <= 0.8*qfbreak) count++;
    count = (qf.array()<= 0.8*qfbreak).count();
    //}
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

    //set up the Q-space
    double xsize = nx*diffx;
    double ysize = ny*diffy;
    double diffqx = 1./xsize;
    double diffqy = 1./ysize;
    Eigen::VectorXd qx(nx);
    for(int i=0;i<nx;i++) qx[i] = diffqx*(i-(nx-1)/2);
    mapmaking::shift(qx,-(nx-1)/2);
    Eigen::VectorXd qy(ny);
    for(int i=0;i<ny;i++) qy[i] = diffqy*(i-(ny-1)/2);
    mapmaking::shift(qy,-(ny-1)/2);
    Eigen::MatrixXd qmap(nx,ny);
    for(int i=0;i<nx;i++)
        for(int j=0;j<ny;j++) qmap(i,j) = sqrt(pow(qx[i],2)+pow(qy[j],2));

    //making psd array which will give us vvq
    Eigen::MatrixXd psdq(nx,ny);

    if(getLowpassOnly){
        //set psdq=1 for all elements
        //for(int i=0;i<nx;i++) for(int j=0;j<ny;j++) psdq(i,j)=1.;
        psdq.setOnes();
    } else {
        //int nhp = hp.size();

        Eigen::Matrix<Eigen::Index,1,1> nhp;
        nhp << hp.size();

        Eigen::Index blah = 1;

        //Need to test limits here
        for(int i=0;i<nx;i++){
            for(int j=0;j<ny;j++){
                if(qmap(i,j) <= qf[qf.size()-1] && qmap(i,j)>=qf[0])
                    //psdq(i,j) = gsl_interp_eval(interp, &qf[0], &hp[0], qmap(i,j), acc);

                    mlinterp::interp(nhp.data(), blah,
                                 hp.data(), psdq.data() + nx*j + i,
                                 qf.data(), qmap.data() + nx*j + i);

                 else if (qmap(i,j) > qf[qf.size()-1])//interp->xmax)
                    psdq(i,j) = hp[hp.size()-1];
                 else if (qmap(i,j) <qf[0])//<interp->xmin)
                    psdq(i,j) = hp[0];
            }
        }
        double lowval=hp[0];
        for(Eigen::Index i=0;i<hp.size();i++){
            if(hp[i] < lowval) lowval=hp[i];
        }
        for(int i=0;i<nx;i++){
            for(int j=0;j<ny;j++){
                if(psdq(i,j)<lowval) psdq(i,j)=lowval;
            }
        }
    }

    //normalize the noise power spectrum and calc vvq
    vvq.resize(nx,ny);
    double totpsdq=0.;
    //for(int i=0;i<nx;i++) for(int j=0;j<ny;j++) totpsdq += psdq(i,j);
    //totpsdq = psdq.sum();
    vvq = psdq/psdq.sum();
    //for(int i=0;i<nx;i++) for(int j=0;j<ny;j++) vvq(i,j) = psdq(i,j)/totpsdq;
}


///calculates the WienerFilter numerator
/** This code calculates the WienerFilter numerator given a
    particular set of input matrices that are members of the class
    and the input image to be filtered.
    Assumptions:
      - rr, VVq, and template all have been precomputed
 **/
void wiener::calcNumerator(Eigen::MatrixXd &mflt){
    Eigen::VectorXcd in(nx*ny);
    Eigen::VectorXcd out(nx*ny);

    //calculate the numerator
    Nume.resize(nx,ny);
    double fftnorm=1./nx/ny;
    {
        int ii;
        for(int i=0;i<nx;i++)
            for(int j=0;j<ny;j++){
                ii = ny*i+j;
                //in.real()(ii) = rr(i,j)*mflt(i,j);
                //in.imag()(ii) = 0.;
                in.real()(ii) = rr(i,j)*mflt(i,j);
                in.imag()(ii) = 0.;
            }
        out = mapmaking::fft2w<forward>(in,nx,ny);// fftw_execute(pf);
        //for(int i=0;i<nx*ny;i++){
          //  out(i,0) *= fftnorm;
            //out(i,1) *= fftnorm;
        //}
        out = out*fftnorm;
        for(int i=0;i<nx;i++)
            for(int j=0;j<ny;j++){
                ii = ny*i+j;
                in.real()(ii) = out.real()(ii)/vvq(i,j);
                in.imag()(ii) = out.imag()(ii)/vvq(i,j);
            }

        in = mapmaking::fft2w<backward>(out,nx,ny);// fftw_execute(pf);

        for(int i=0;i<nx;i++)
            for(int j=0;j<ny;j++){
                ii = ny*i+j;
                in.real()(ii) = out.real()(ii)*rr(i,j);
                in.imag()(ii) = 0.;
            }

        out = mapmaking::fft2w<forward>(in,nx,ny);// fftw_execute(pf);

        for(int i=0;i<nx*ny;i++){
            out.real()[i] *= fftnorm;
            out.imag()[i] *= fftnorm;
        }
        Eigen::VectorXcd qqq(nx*ny);

        qqq = out;

        //for(int i=0;i<nx*ny;i++){
            //qqq(i,0) = out(i,0);
            //qqq(i,1) = out(i,1);
        //}
        for(int i=0;i<nx;i++)
            for(int j=0;j<ny;j++){
                ii = ny*i+j;
                in.real()(ii) = tplate(i,j);
                in.imag()(ii) = 0.;
            }

        out = mapmaking::fft2w<forward>(in,nx,ny);// fftw_execute(pf);
        //for(int i=0;i<nx*ny;i++){
          //  out(i,0) *= fftnorm;
            //out(i,1) *= fftnorm;
        //}
        out = out*fftnorm;
        for(int i=0;i<nx*ny;i++){
            in.real()(i) = out.real()(i)*qqq.real()(i)+ out.imag()(i)*qqq.imag()(i);
            in.imag()(i) = -out.imag()(i)*qqq.real()(i) + out.real()(i)*qqq.imag()(i);
        }
        in = mapmaking::fft2w<backward>(out,nx,ny);// fftw_execute(pf);
        for(int i=0;i<nx;i++)
            for(int j=0;j<ny;j++){
                ii = ny*i+j;
                Nume(i,j) = out.real()(ii);
            }
    }
}

//----------------------------- o ---------------------------------------

///calculates the WienerFilter denominator
/** This code calculates the WienerFilter denominator
    Assumptions:
      - rr, VVq, and tPlate all have been precomputed
 **/
bool wiener::calcDenominator(){
    double fftnorm=1./nx/ny;
    Eigen::VectorXcd in(nx*ny);
    Eigen::VectorXcd out(nx*ny);

    //calculate the denominator
    Denom.resize(nx,ny);
    int ii;

    //if uniformWeight is set then the denominator calc is simple
    if(uniformWeight){
        double d=0.;
        for(int i=0;i<nx;i++)
            for(int j=0;j<ny;j++){
                ii = ny*i+j;
                in.real()(ii) = tplate(i,j);
                in.imag()(ii) = 0.;
            }
        out = mapmaking::fft2w<forward>(in,nx,ny);// fftw_execute(pf);
        //for(int i=0;i<nx*ny;i++){
          //  out.real()(i) *= fftnorm;
            //out.imag()(i) *= fftnorm;
        //}
        out = out*fftnorm;
        for(int i=0;i<nx;i++)
            for(int j=0;j<ny;j++){
                ii = ny*i+j;
                d += (out.real()(ii)*out.real()(ii) + out.imag()(ii)*out.imag()(ii))/vvq(i,j);
            }
        Denom.setConstant(d);
        return 1;
    }

    //here's where the involved calculation is done
    for(int i=0;i<nx;i++)
        for(int j=0;j<ny;j++){
            ii = ny*i+j;
            in.real()(ii) = 1./vvq(i,j);
            in.imag()(ii) = 0.;
        }
    in = mapmaking::fft2w<backward>(out,nx,ny);// fftw_execute(pf);

    //using gsl vector sort routines
    //remember this is the forward sort but we want the reverse
    //also, the idl code sorts on the absolute value of out but
    //then adds in the components with the correct sign.
    //gsl_vector* zz2d = gsl_vector_alloc(nx*ny);
    Eigen::VectorXd zz2d(nx*ny);
    for(int i=0;i<nx;i++)
        for(int j=0;j<ny;j++){
            ii = ny*i+j;
            //gsl_vector_set(zz2d,ii,abs(out.real()(ii)));
            zz2d(ii) = abs(out.real()(ii));
        }

    //gsl_permutation* ss_ord = gsl_permutation_alloc(nx*ny);
    //gsl_sort_vector_index(ss_ord,zz2d);

    Eigen::VectorXd ss_ord;
    ss_ord = zz2d;

    auto sorted = mapmaking::sorter(ss_ord);

    for(int i=0;i<nx;i++)
        for(int j=0;j<ny;j++){
            ii = ny*i+j;
            //gsl_vector_set(zz2d,ii,out.real()(ii));
            zz2d(ii) = out.real()(ii);
        }

    //number of iterations for convergence (hopefully)
    int nloop = nx*ny/100;

    //the loop
    //for(int i=0;i<nx;i++)
      //  for(int j=0;j<ny;j++)
        //    Denom(i,j)=0.;

    Denom.setZero();

    //flag to say we're done
    bool done=false;

    for(int k=0;k<nx;k++){
//#pragma omp parallel for schedule (dynamic) ordered shared (fftnorm, ss_ord, nloop, zz2d, k, cerr, done,sorted) private (ii) default (none)
        for(int l=0;l<ny;l++){

//#pragma omp flush (done)
            if(!done){

                Eigen::VectorXcd in2;
                Eigen::VectorXcd out2;

                int kk = ny*k+l;
                if(kk >= nloop) continue;
//#pragma omp critical (wfFFTW)
                {
                    in2.resize(nx*ny);
                    out2.resize(nx*ny);
                }

                //int shifti = gsl_permutation_get(ss_ord,nx*ny-kk-1);
                int shifti = std::get<1>(sorted[nx*ny-kk-1]);

                double xshiftN = shifti / ny;
                double yshiftN = shifti % ny;

                //the ffdq fft
                for(int i=0;i<nx;i++)
                    for(int j=0;j<ny;j++){
                        ii = ny*i+j;
                        in2.real()(ii) = tplate(i,j)*shift2(tplate,-xshiftN,-yshiftN,i,j);
                        in2.imag()(ii) = 0.;
                    }
                out2 = mapmaking::fft2w<forward>(in2,nx,ny);// fftw_execute(pf);
                for(int i=0;i<nx*ny;i++){
                    out2.real()(i) *= fftnorm;
                    out2.imag()(i) *= fftnorm;
                }
                Eigen::MatrixXcd ffdq(nx,ny);
                for(int i=0;i<nx*ny;i++){
                    ffdq.real()(i)=out2.real()(i);
                    ffdq.imag()(i)=out2.imag()(i);
                }

                //the rrdq fft, this is in "out" in the next step
                for(int i=0;i<nx;i++)
                    for(int j=0;j<ny;j++){
                        ii = ny*i+j;
                        in2.real()(ii) = rr(i,j)*shift2(rr,-xshiftN,-yshiftN,i,j);
                        in2.imag()(ii) = 0.;
                    }
                out2 = mapmaking::fft2w<forward>(in2,nx,ny);// fftw_execute(pf);
                for(int i=0;i<nx*ny;i++){
                    out2.real()(i) *= fftnorm;
                    out2.imag()(i) *= fftnorm;
                }

                //the convolution: conj(ffdq)*rr
                double ar, br, ai, bi;
                for(int i=0;i<nx*ny;i++){
                    ar = ffdq.real()(i);
                    ai = ffdq.imag()(i);
                    br = out2.real()(i);
                    bi = out2.imag()(i);
                    in2.real()(i) = ar*br + ai*bi;
                    in2.imag()(i) = -ai*br + ar*bi;
                }
                in2 = mapmaking::fft2w<backward>(out2,nx,ny);// fftw_execute(pf);
                //update Denom
//#pragma omp ordered
                {
                    //storage
                    Eigen::MatrixXd updater(nx,ny);
                    for(int i=0;i<nx;i++)
                        for(int j=0;j<ny;j++){
                            ii = ny*i+j;
                            //updater(i,j) = gsl_vector_get(zz2d,shifti)*out2.real()(ii)*fftnorm;
                            updater(i,j) = zz2d[shifti]*out2.real()(ii)*fftnorm;

        }
                    for(int i=0;i<nx;i++)
                        for(int j=0;j<ny;j++){
                            Denom(i,j) += updater(i,j);
                        }

/*#pragma omp critical (wfFFTW)
                    {
                        fftw_free(in2);
                        fftw_free(out2);
                        fftw_destroy_plan(pf2);
                        fftw_destroy_plan(pr2);
                    }*/

                    //check to see if we're done every 100 iterations.  The criteria for
                    //finishing is that either we are at nx*ny/100 iterations or that
                    //the significant elements Denom are growing by less than 0.1%

                    if((kk % 100) == 1){
                        double maxRatio=-1;
                        double maxDenom=-999.;
                        for(int i=0;i<nx;i++)
                          for(int j=0;j<ny;j++)
                    if(Denom(i,j) > maxDenom) maxDenom = Denom(i,j);
                        for(int i=0;i<nx;i++)
                            for(int j=0;j<ny;j++){
                                if(Denom(i,j) > 0.01*maxDenom){
                                    if(abs(updater(i,j)/Denom(i,j)) > maxRatio)
            maxRatio = abs(updater(i,j)/Denom(i,j));
                                }
                            }
                        if(((kk >= 500) && (maxRatio < 0.0002)) || maxRatio < 1e-10){
                            SPDLOG_INFO("Seems like we're done.  maxRatio={}", maxRatio);
                            SPDLOG_INFO("The Denom calcualtion required {} iterations",kk);
                            done=true;
//#pragma omp flush(done)
                        } else {
                            //update the console with where we are
                            SPDLOG_INFO("Completed iteration {} of {}. maxRatio={}",kk,nloop,maxRatio);
                        }
      }
                }
            }
        }
    }
//);
    //not sure this is correct but need to avoid small negative values
    SPDLOG_INFO("Zeroing out any small values in Denom");
    for(int i=0;i<nx;i++)
        for(int j=0;j<ny;j++)
            if(Denom(i,j) < 1.e-4) Denom(i,j)=0;

    //}
    return 1;

}

void wiener::prepareTemplate(MapStruct &cmap){
/*
    //collect what we need
    Eigen::VectorXd xgcut(nx);
    for(int i=0;i<nx;i++) xgcut[i] = cmap->getRowCoordsPhys(i);
    Eigen::VectorXd ygcut(ny);
    for(int i=0;i<ny;i++) ygcut[i] = cmap->getColCoordsPhys(i);
    Eigen::MatrixXd tem(nx, ny);
    for(int i=0;i<nx;i++) for(int j=0;j<ny;j++)
            tem(i,j) = cmap->kernel->image(i,j);

    //rotationally symmeterize the kernel map the brute force way
    //but note that we need to center the kernel image in the
    //array before proceeding
    Eigen::VectorXd pp(6);
    cmap->kernel->fitToGaussian(pp);

    double beamRatio = 30.0/8.0;

    beamRatio *=8.0;


    //bail out if this is a bad fit, this means the kernel is not
    //properly constructed
    if(pp[2]/TWO_PI*360.*3600.*2.3548 < 0 || pp[2]/TWO_PI*360.*3600.*2.3548 > beamRatio ||
        pp[3]/TWO_PI*360.*3600.*2.3548 < 0 || pp[3]/TWO_PI*360.*3600.*2.3548 > beamRatio ||
        abs(pp[4]/TWO_PI*360.*3600.) > beamRatio || abs(pp[5]/TWO_PI*360.*3600.) > beamRatio){
        cerr << "Something's terribly wrong with the coadded kernel." << endl;
        cerr << "Wiener Filtering your map with this kernel would result " << endl;
        cerr << "in nonsense.  Instead, consider using the highpass only " << endl;
        cerr << "option or using a synthesized gaussian kernel. " << endl;
        cerr << "Exiting...";
        exit(1);
    }

    //shift the kernel appropriately
    shift(tem,-round(pp[4]/diffx),-round(pp[5]/diffy));

    Eigen::MatrixXd dist(nx,ny);
    for(int i=0;i<nx;i++)
        for(int j=0;j<ny;j++)
            dist(i,j) = sqrt(pow(xgcut[i],2)+pow(ygcut[j],2));

    //find location of minimum dist
    int xcind = nx/2;
    int ycind = ny/2;
    double mindist = 99.;
    for(int i=0;i<nx;i++)
        for(int j=0;j<ny;j++)
            if(dist(i,j) < mindist) mindist = dist(i,j);
    for(int i=0;i<nx;i++)
        for(int j=0;j<ny;j++)
            if(dist(i,j) == mindist){
                xcind = i;
                ycind = j;
            }

    //create new bins based on diffx
    int nbins = xgcut[nx-1]/diffx;
    Eigen::VectorXd binlow(nbins);
    for(int i=0;i<nbins;i++) binlow[i] = (double) (i*diffx);

    Eigen::VectorXd kone(nbins-1,0.);
    Eigen::VectorXd done(nbins-1,0.);
    for(int i=0;i<nbins-1;i++){
        int c=0;
        for(int j=0;j<nx;j++){
            for(int k=0;k<ny;k++){
                if(dist(j,k)  >= binlow[i] && dist(j,k) < binlow[i+1]){
                    c++;
                    kone[i] += tem(j,k);
                    done[i] += dist(j,k);
                }
            }
        }
        kone[i] /= c;
        done[i] /= c;
    }

    //now spline interpolate to generate new template array
    tplate.resize(nx,ny);
    {
        gsl_interp_accel *acc = gsl_interp_accel_alloc();
        gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, nbins-1);
        gsl_spline_init(spline, &done[0], &kone[0], nbins-1);

        for(int i=0;i<nx;i++){
            for(int j=0;j<ny;j++){
                int ti = (i-xcind)%nx;
                int tj = (j-ycind)%ny;
                int shifti = (ti < 0) ? nx+ti : ti;
                int shiftj = (tj < 0) ? ny+tj : tj;
                if(dist(i,j) <= spline->interp->xmax && dist(i,j) >= spline->interp->xmin)
                    tplate(shifti,shiftj) = gsl_spline_eval(spline, dist(i,j), acc);
                else if (dist(i,j) >spline->interp->xmax)
                    tplate(shifti,shiftj) = kone[kone.size()-1];
                else if (dist(i,j) <spline->interp->xmin)
                    tplate(shifti,shiftj) = kone[0];


            }
        }
        //gsl_spline_free(spline);
        //gsl_interp_accel_free(acc);
    }
    */

}


//----------------------------- o ---------------------------------------

///Produce a gaussian template in place of the kernel map
void wiener::prepareGaussianTemplate(MapStruct &cmap){
    //collect what we need
    Eigen::VectorXd xgcut(nx);
    for(int i=0;i<nx;i++) xgcut[i] = cmap.rowcoordphys(i);
    Eigen::VectorXd  ygcut(ny);
    for(int i=0;i<ny;i++) ygcut[i] = cmap.colcoordphys(i);
    Eigen::MatrixXd tem(nx, ny);
    for(int i=0;i<nx;i++) for(int j=0;j<ny;j++)
            tem(i,j) = 0;

    Eigen::MatrixXd dist(nx,ny);
    for(int i=0;i<nx;i++)
        for(int j=0;j<ny;j++)
            dist(i,j) = sqrt(pow(xgcut[i],2)+pow(ygcut[j],2));

    //find location of minimum dist
    int xcind = nx/2;
    int ycind = ny/2;
    double mindist = 99.;
    for(int i=0;i<nx;i++)
        for(int j=0;j<ny;j++)
            if(dist(i,j) < mindist) mindist = dist(i,j);
    for(int i=0;i<nx;i++)
        for(int j=0;j<ny;j++)
            if(dist(i,j) == mindist){
                xcind = i;
                ycind = j;
            }

    //generate gaussian with preset fwhm
    tplate.resize(nx,ny);
    double fwhm = GaussianTemplateFWHM;
    double sig = fwhm/2.3548;
    for(int i=0;i<nx;i++)
        for(int j=0;j<ny;j++)
            tplate(i,j) = exp(-0.5*pow(dist(i,j)/sig,2.));

    //and shift the gaussian to peak at the origin
    shift3(tplate,-xcind, -ycind);
}

} //namespace mapmaking
