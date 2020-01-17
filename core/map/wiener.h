#pragma once

#include <unsupported/Eigen/Splines>

namespace mapmaking{
template <typename DerivedA>
void wienerfilter(DerivedA &mapstruct){

    bool getHighpassOnly = 0;
    bool getGaussianTemplate = 0;

        //save typing
        pixelSize = mapstruct.pixelsize/3600./360.*TWO_PI;
        nrows = mapstruct.nrows;
        ncols = mapstruct.ncols;

        //make sure new wiener filtered maps are even dimensioned
        nx = 2*(nrows/2);
        ny = 2*(ncols/2);

        //The following is a preliminary in the IDL utilities
        int zcount=0;
        for(int i=0;i<nrows;i++)
                for(int j=0;j<ncols;j++)
                        if(mapstruct.wtt(i,j) == 0.) zcount++;
        Eigen::Matrix<int,Eigen::Dynamic,1> whz;
        if(zcount > 0){
                whz.resize(zcount);
                zcount = 0;
                for(int i=0;i<nrows;i++)
                        for(int j=0;j<ncols;j++)
                                if(mapstruct.wtt(i,j) == 0.){
                                        whz(zcount) = ncols*i+j;
                                        zcount++;
                                }
        }

        //----- o -----

        //x and y spacing should be equal
        diffx = abs(mapstruct.rowcoordphys(1)-mapstruct.rowcoordphys(0));
        diffy = abs(mapstruct.colcoordphys(1)-mapstruct.colcoordphys(0));

        //----- o -----
        //prepare the template
        if(getHighpassOnly()){
          tplate.assign(nx,ny,0.);
          tplate(0,0)=1.;
        } else if(getGaussianTemplate()){
          prepareGaussianTemplate(cmap);
        } else prepareTemplate(cmap);
}


//----------------------------- o ---------------------------------------

///Main driver code to filter a set of maps from a coaddition.
/** This is the main driver for rapidly filtering the images from
    a coadded data set.
    It requires that vvq, rr, tplate, and Denom are already
    calculated.  This is done in the constructor.  The
    steps to Wiener filtering are then:
     1) calculate the numerator for the image you wish to filter
     2) calculate the filtered image as numerator/denominator
     3) don't forget that the denominator is the filtered weight map.
 **/
template <typename DerivedA>
void filterCoaddition(DerivedA &mapstruct){
        //here's storage for the filtering
        Eigen::MatrixXd mflt(nx, ny);

        //do the kernel first since it's simpler to calculate
        //the kernel calculation requires uniform weighting
        uniformWeight=1;
        calcRr(cmap);
        calcVvq();
        calcDenominator();
        for(int i=0;i<nx;i++) for(int j=0;j<ny;j++)
                mflt(i,j) = mapstruct.kernel(i,j);
        calcNumerator(mflt);
        for(int i=0;i<nx;i++)
                for(int j=0;j<ny;j++){
                        if (Denom(i,j) != 0.0)
                                cmap->filteredKernel->image(i,j)=Nume(i,j)/Denom(i,j);
                        else {
                                cmap->filteredKernel->image(i,j)= 0.0;
                                cerr<<"Nan prevented "<<endl;
                        }
                }

        //Do the more complex precomputing for the signal map
        uniformWeight=0;
        calcRr(cmap);
        calcVvq();
        calcDenominator();

        //Here is the signal map to be filtered
        for(int i=0;i<nx;i++) for(int j=0;j<ny;j++)
                mflt(i,j) = mapstruct.signal(i,j);

        //calculate the associated numerator
        calcNumerator(mflt);

        //replace the original images with the filtered images
        //note that the filtered weight map = Denom
        for(int i=0;i<nx;i++) for(int j=0;j<ny;j++){
                if (Denom(i,j) !=0.0)
                        cmap->filteredSignal->image(i,j)=Nume(i,j)/Denom(i,j);
                else{
                        //cerr<<"Nan avoided"<<endl;
                        cmap->filteredSignal->image(i,j)=0.0;
                }
                cmap->filteredWeight->image(i,j)=Denom(i,j);
        }

        return 1;
}


//----------------------------- o ---------------------------------------

///Main driver code to filter a set of noise realizations.
/** This is the main driver for rapidly filtering the images from
    a complete set of coadded noise realizations.  Unlike
    the coaddition filtering above, this code directly writes
    the filtered noise map into the appropriate netcdf file.
    It requires that vvq, rr, tplate, and Denom are already
    calculated.
 **/
template <typename DerivedA>
void filterNoiseMaps(NoiseRealizations *nr){

        //The strategy here is to loop through the noise files one
        //by one, pulling in the noise image, filtering it, and then
        //writing it back into the file.
        for(int k=0;k<nr->nNoiseFiles;k++){
          // print out what's happening
          cerr << "WienerFilter(): Filtering noise map " << k << ".\r";

                //open the noise file
                NcFile ncfid = NcFile(nr->noiseFiles(k).c_str(), NcFile::Write);

                //the input array dims
                nrows = ncfid.get_dim("nrows")->size();
                ncols = ncfid.get_dim("ncols")->size();

                //the actual noise matrix
                Eigen::MatrixXd noise(nrows, ncols);
                NcVar* noisev = ncfid.get_var("noise");
                noisev->get(&noise(0,0), nrows, ncols);

                //do the filtering
                Eigen::MatrixXd filteredNoise(nrows,ncols,0.);
                Eigen::MatrixXd filteredWeight(nrows,ncols,0.);
                calcNumerator(noise);
                for(int i=0;i<nx;i++) for(int j=0;j<ny;j++){
                    if(Denom(i,j) > 0){
                      filteredNoise(i,j)=Nume(i,j)/Denom(i,j);
                      filteredWeight(i,j)=Denom(i,j);
                    }
                  }

                //write the result back into the netcdf file
                NcDim* rowDim = ncfid.get_dim("nrows");
                NcDim* colDim = ncfid.get_dim("ncols");
                NcVar *fnVar = ncfid.add_var("filteredNoise", ncDouble, rowDim, colDim);
                fnVar->put(&filteredNoise(0,0), nrows, ncols);
                NcVar *fwVar = ncfid.add_var("filteredWeight", ncDouble, rowDim, colDim);
                fwVar->put(&filteredWeight(0,0), nrows, ncols);
        }
        cerr << endl;
        return 1;
}

//----------------------------- o ---------------------------------------

///Calculation of the RR matrix
template <typename DerivedA>
void calcRr(DerivedA &mapstruct){

        if(uniformWeight){
                rr.assign(nx,ny,1.);
        } else {
                rr.resize(nx,ny);
                for(int i=0;i<nx;i++) for(int j=0;j<ny;j++)
                        rr(i,j) = sqrt(mapstruct.wtt(i,j));
        }
        return 1;
}

//----------------------------- o ---------------------------------------

///Calculation of the VVQ matrix
/** This set of code calculates two matrices that are members
    of the WienerFilter Class.  Both are essential for calculating
    the numerator and denominator.
 **/
void calcVvq(){
  //start by fetching the noise psd
  string psdfile = ap->getAvgNoisePsdFile();
  NcFile ncfid = NcFile(psdfile.c_str(), NcFile::ReadOnly);
  int npsd = ncfid.get_dim("npsd")->size();
  Eigen::VectorXd qf(npsd);
  Eigen::VectorXd hp(npsd);
  NcVar* psdv = ncfid.get_var("psd");
  NcVar* psdfv = ncfid.get_var("psdFreq");
  for(int i=0;i<npsd;i++){
    hp(i) = psdv->as_double(i);
    qf(i) = psdfv->as_double(i);
  }

  //modify the psd array to take out lowpassing and highpassing
  double maxhp=-1.;
  int maxhpind=0;
  double qfbreak=0.;
  double hpbreak=0.;
  for(int i=0;i<npsd;i++) if(hp(i) > maxhp){
      maxhp = hp(i);
      maxhpind = i;
    }
  for(int i=0;i<npsd;i++) if(hp(i)/maxhp < 1.e-4){
      qfbreak = qf(i);
      break;
    }
  //flatten the response above the lowpass break
  int count=0;
  for(int i=0;i<npsd;i++) if(qf(i) <= 0.8*qfbreak) count++;
  if(count > 0){
    for(int i=0;i<npsd;i++){
      if(qfbreak > 0){
        if(qf(i) <= 0.8*qfbreak) hpbreak = hp(i);
        if(qf(i) > 0.8*qfbreak) hp(i) = hpbreak;
      }
    }
  }
  //flatten highpass response if present
  if(maxhpind > 0) for(int i=0;i<maxhpind;i++) hp(i) = maxhp;

  //set up the Q-space
  double xsize = nx*diffx;
  double ysize = ny*diffy;
  double diffqx = 1./xsize;
  double diffqy = 1./ysize;
  Eigen::VectorXd qx(nx);
  for(int i=0;i<nx;i++) qx(i) = diffqx*(i-(nx-1)/2);
  shift(qx,-(nx-1)/2);
  Eigen::VectorXd qy(ny);
  for(int i=0;i<ny;i++) qy(i) = diffqy*(i-(ny-1)/2);
  shift(qy,-(ny-1)/2);
  Eigen::MatrixXd qmap(nx,ny);
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++) qmap(i,j) = sqrt(pow(qx(i),2)+pow(qy(j),2));


  //making psd array which will give us vvq
  Eigen::MatrixXd psdq(nx,ny);

  if(ap->getLowpassOnly()){
    //set psdq=1 for all elements
    for(int i=0;i<nx;i++) for(int j=0;j<ny;j++) psdq(i,j)=1.;
  } else {
    int nhp = hp.size();
    gsl_interp *interp = gsl_interp_alloc(gsl_interp_linear, nhp);
    gsl_interp_init(interp, &qf(0), &hp(0), nhp);
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    for(int i=0;i<nx;i++){
      for(int j=0;j<ny;j++){
        if(qmap(i,j) <= qf(qf.size()-1) && qmap(i,j)>=qf(0))
          psdq(i,j) = gsl_interp_eval(interp, &qf(0), &hp(0), qmap(i,j), acc);
        else if (qmap(i,j) >interp->xmax)
          psdq(i,j) = hp(hp.size()-1);
        else if (qmap(i,j) <interp->xmin)
          psdq(i,j) = hp(0);
      }
    }
    double lowval=hp(0);
    for(int i=0;i<nhp;i++) if(hp(i) < lowval) lowval=hp(i);
    for(int i=0;i<nx;i++)
      for(int j=0;j<ny;j++)
        if(psdq(i,j)<lowval) psdq(i,j)=lowval;
    gsl_interp_accel_free(acc);
  }


  //normalize the noise power spectrum and calc vvq
  vvq.resize(nx,ny);
  double totpsdq=0.;
  for(int i=0;i<nx;i++) for(int j=0;j<ny;j++) totpsdq += psdq(i,j);
  for(int i=0;i<nx;i++) for(int j=0;j<ny;j++) vvq(i,j) = psdq(i,j)/totpsdq;

  return 1;
}

//----------------------------- o ---------------------------------------

///calculates the WienerFilter numerator
/** This code calculates the WienerFilter numerator given a
    particular set of input matrices that are members of the class
    and the input image to be filtered.
    Assumptions:
      - rr, VVq, and template all have been precomputed
 **/
template<typename DerivedA>
void calcNumerator(Eigen::DenseBase<DerivedA> &mflt)
{

  //here is the memory allocation and the plan setup
  fftw_complex *in;
  fftw_complex *out;
  in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*nx*ny);
  out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*nx*ny);
  fftw_plan pf, pr;
  pf = fftw_plan_dft_2d(nx, ny, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  pr = fftw_plan_dft_2d(nx, ny, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);

  //calculate the numerator
  Nume.resize(nx,ny);
  double fftnorm=1./nx/ny;
  {
    int ii;
    for(int i=0;i<nx;i++)
      for(int j=0;j<ny;j++){
        ii = ny*i+j;
        in(ii,0) = rr(i,j)*mflt(i,j);
        in(ii,1) = 0.;
      }
    fftw_execute(pf);
    for(int i=0;i<nx*ny;i++){
      out(i,0) *= fftnorm;
      out(i,1) *= fftnorm;
    }
    for(int i=0;i<nx;i++)
      for(int j=0;j<ny;j++){
        ii = ny*i+j;
        in(ii,0) = out(ii,0)/vvq(i,j);
        in(ii,1) = out(ii,1)/vvq(i,j);
      }
    fftw_execute(pr);
    for(int i=0;i<nx;i++)
      for(int j=0;j<ny;j++){
        ii = ny*i+j;
        in(ii,0) = out(ii,0)*rr(i,j);
        in(ii,1) = 0.;
      }
    fftw_execute(pf);
    for(int i=0;i<nx*ny;i++){
      out(i,0) *= fftnorm;
      out(i,1) *= fftnorm;
    }
    Eigen::MatrixXd qqq(nx*ny,2);
    for(int i=0;i<nx*ny;i++){
      qqq(i,0) = out(i,0);
      qqq(i,1) = out(i,1);
    }
    for(int i=0;i<nx;i++)
      for(int j=0;j<ny;j++){
        ii = ny*i+j;
        in(ii,0) = tplate(i,j);
        in(ii,1) = 0.;
      }
    fftw_execute(pf);
    for(int i=0;i<nx*ny;i++){
      out(i,0) *= fftnorm;
      out(i,1) *= fftnorm;
    }
    for(int i=0;i<nx*ny;i++){
      in(i,0) = out(i,0)*qqq(i,0) + out(i,1)*qqq(i,1);
      in(i,1) = -out(i,1)*qqq(i,0) + out(i,0)*qqq(i,1);
    }
    fftw_execute(pr);
    for(int i=0;i<nx;i++)
      for(int j=0;j<ny;j++){
        ii = ny*i+j;
        Nume(i,j) = out(ii,0);
      }
  }

  //cleanup
  fftw_free(in);
  fftw_free(out);
  fftw_destroy_plan(pf);
  fftw_destroy_plan(pr);

  return 1;
}

//----------------------------- o ---------------------------------------

///calculates the WienerFilter denominator
/** This code calculates the WienerFilter denominator
    Assumptions:
      - rr, VVq, and tPlate all have been precomputed
 **/
void calcDenominator()
{

  //here is the memory allocation and the plan setup
  fftw_complex *in;
  fftw_complex *out;
  in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*nx*ny);
  out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*nx*ny);
  fftw_plan pf, pr;
  pf = fftw_plan_dft_2d(nx, ny, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  pr = fftw_plan_dft_2d(nx, ny, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
  double fftnorm=1./nx/ny;

  //calculate the denominator
  Denom.resize(nx,ny);
  int ii;

  //if uniformWeight is set then the denominator calc is simple
  if(uniformWeight){
    double d=0.;
    for(int i=0;i<nx;i++)
      for(int j=0;j<ny;j++){
        ii = ny*i+j;
        in(ii,0) = tplate(i,j);
        in(ii,1) = 0.;
      }
    fftw_execute(pf);
    for(int i=0;i<nx*ny;i++){
      out(i,0) *= fftnorm;
      out(i,1) *= fftnorm;
    }
    for(int i=0;i<nx;i++)
      for(int j=0;j<ny;j++){
        ii = ny*i+j;
        d += (out(ii,0)*out(ii,0) + out(ii,1)*out(ii,1))/vvq(i,j);
      }
    Denom.assign(nx,ny,d);
    //cleanup
    fftw_free(in);
    fftw_free(out);
    fftw_destroy_plan(pf);
    fftw_destroy_plan(pr);
    return 1;
  }

  //here's where the involved calculation is done
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++){
      ii = ny*i+j;
      in(ii,0) = 1./vvq(i,j);
      in(ii,1) = 0.;
    }
  fftw_execute(pr);

  //using gsl vector sort routines
  //remember this is the forward sort but we want the reverse
  //also, the idl code sorts on the absolute value of out but
  //then adds in the components with the correct sign.
  gsl_vector* zz2d = gsl_vector_alloc(nx*ny);
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++){
      ii = ny*i+j;
      gsl_vector_set(zz2d,ii,abs(out(ii,0)));
    }
  gsl_permutation* ss_ord = gsl_permutation_alloc(nx*ny);
  gsl_sort_vector_index(ss_ord,zz2d);
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++){
      ii = ny*i+j;
      gsl_vector_set(zz2d,ii,out(ii,0));
    }

  //number of iterations for convergence (hopefully)
  int nloop = nx*ny/100;

  //the loop
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
      Denom(i,j)=0.;

  //flag to say we're done
  bool done=false;

  for(int k=0;k<nx;k++){
#pragma omp parallel for schedule (dynamic) ordered shared (fftnorm, ss_ord, nloop, zz2d, k, cerr, done) private (ii) default (none)
    for(int l=0;l<ny;l++){

#pragma omp flush (done)
      if(!done){
        fftw_complex *in2,*out2;
        fftw_plan pf2, pr2;
        int kk = ny*k+l;
        if(kk >= nloop) continue;
#pragma omp critical (wfFFTW)
        {
          in2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*nx*ny);
          out2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*nx*ny);
          pf2 = fftw_plan_dft_2d(nx, ny, in2, out2, FFTW_FORWARD, FFTW_ESTIMATE);
          pr2 = fftw_plan_dft_2d(nx, ny, in2, out2, FFTW_BACKWARD, FFTW_ESTIMATE);
        }

        int shifti = gsl_permutation_get(ss_ord,nx*ny-kk-1);
        double xshiftN = shifti / ny;
        double yshiftN = shifti % ny;

        //the ffdq fft
        for(int i=0;i<nx;i++)
          for(int j=0;j<ny;j++){
            ii = ny*i+j;
            in2(ii,0) = tplate(i,j)*shift(tplate,-xshiftN,-yshiftN,i,j);
            in2(ii,1) = 0.;
          }
        fftw_execute(pf2);
        for(int i=0;i<nx*ny;i++){
          out2(i,0) *= fftnorm;
          out2(i,1) *= fftnorm;
        }
        Eigen::MatrixXd ffdq(nx*ny,2);
        for(int i=0;i<nx*ny;i++){
          ffdq(i,0)=out2(i,0);
          ffdq(i,1)=out2(i,1);
        }

        //the rrdq fft, this is in "out" in the next step
        for(int i=0;i<nx;i++)
          for(int j=0;j<ny;j++){
            ii = ny*i+j;
            in2(ii,0) = rr(i,j)*shift(rr,-xshiftN,-yshiftN,i,j);
            in2(ii,1) = 0.;
          }
        fftw_execute(pf2);
        for(int i=0;i<nx*ny;i++){
          out2(i,0) *= fftnorm;
          out2(i,1) *= fftnorm;
        }

        //the convolution: conj(ffdq)*rr
        double ar, br, ai, bi;
        for(int i=0;i<nx*ny;i++){
          ar = ffdq(i,0);
          ai = ffdq(i,1);
          br = out2(i,0);
          bi = out2(i,1);
          in2(i,0) = ar*br + ai*bi;
          in2(i,1) = -ai*br + ar*bi;
        }
        fftw_execute(pr2);
        //update Denom
#pragma omp ordered
        {
          //storage
          Eigen::MatrixXd updater(nx,ny);
          for(int i=0;i<nx;i++)
            for(int j=0;j<ny;j++){
              ii = ny*i+j;
              updater(i,j) = gsl_vector_get(zz2d,shifti)*out2(ii,0)*fftnorm;
            }

          for(int i=0;i<nx;i++)
            for(int j=0;j<ny;j++){
              Denom(i,j) += updater(i,j);
            }

#pragma omp critical (wfFFTW)
          {
            fftw_free(in2);
            fftw_free(out2);
            fftw_destroy_plan(pf2);
            fftw_destroy_plan(pr2);
          }

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
              cerr << endl;
              cerr << endl;
              cerr << "Seems like we're done.  maxRatio=" << maxRatio << endl;
              cerr << "The Denom calcualtion required " << kk << " iterations." << endl;
              done=true;
#pragma omp flush(done)
            } else {
              //update the console with where we are
              cerr << "Completed iteration " << kk << " of " << nloop
                   << ".  maxRatio = " << maxRatio
                   << ".....\r";
            }
          }
        }
      }
    }
  }
  cerr << endl;

  //not sure this is correct but need to avoid small negative values
  cerr << "Zeroing out any small values in Denom" << endl;
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
      if(Denom(i,j) < 1.e-4) Denom(i,j)=0;


  //}
  gsl_vector_free(zz2d);
  gsl_permutation_free(ss_ord);

  //cleanup
  fftw_free(in);
  fftw_free(out);
  fftw_destroy_plan(pf);
  fftw_destroy_plan(pr);

  return 1;
}


//----------------------------- o ---------------------------------------

///Rotationally Symmeterizes the template
template <typename DerivedA>
void prepareTemplate(DerivedA &mapstruct)
{

  //collect what we need
  Eigen::VectorXd xgcut(nx);
  for(int i=0;i<nx;i++) xgcut(i) = mapstruct.rowcoordphys(i);
  Eigen::VectorXd ygcut(ny);
  for(int i=0;i<ny;i++) ygcut(i) = mapstruct.colcoordphys(i);
  Eigen::MatrixXd tem(nx, ny);
  for(int i=0;i<nx;i++) for(int j=0;j<ny;j++)
                          tem(i,j) = mapstruct.kernel(i,j);

  //rotationally symmeterize the kernel map the brute force way
  //but note that we need to center the kernel image in the
  //array before proceeding
  Eigen::VectorXd pp(6);
  cmap->kernel->fitToGaussian(pp);

  double beamRatio = 30.0/8.0;

  if (ap->getObservatory().compare("ASTE"))
          beamRatio *= 30.0;
  else if (ap->getObservatory().compare("JCMT"))
          beamRatio *= 18.0;
  else //Default is LMT data
          beamRatio *=8.0;


  //bail out if this is a bad fit, this means the kernel is not
  //properly constructed
  if(pp(2)/TWO_PI*360.*3600.*2.3548 < 0 || pp(2)/TWO_PI*360.*3600.*2.3548 > beamRatio ||
     pp(3)/TWO_PI*360.*3600.*2.3548 < 0 || pp(3)/TWO_PI*360.*3600.*2.3548 > beamRatio ||
     abs(pp(4)/TWO_PI*360.*3600.) > beamRatio || abs(pp(5)/TWO_PI*360.*3600.) > beamRatio){
    cerr << "Something's terribly wrong with the coadded kernel." << endl;
    cerr << "Wiener Filtering your map with this kernel would result " << endl;
    cerr << "in nonsense.  Instead, consider using the highpass only " << endl;
    cerr << "option or using a synthesized gaussian kernel. " << endl;
    cerr << "Exiting...";
    exit(1);
  }

  //shift the kernel appropriately
  shift(tem,-round(pp(4)/diffx),-round(pp(5)/diffy));

  Eigen::MatrixXd dist(nx,ny);
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
      dist(i,j) = sqrt(pow(xgcut(i),2)+pow(ygcut(j),2));

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
  int nbins = xgcut(nx-1)/diffx;
  Eigen::VectorXd binlow(nbins);
  for(int i=0;i<nbins;i++) binlow(i) = (double) (i*diffx);

  Eigen::VectorXd kone(nbins-1,0.);
  Eigen::VectorXd done(nbins-1,0.);
  for(int i=0;i<nbins-1;i++){
    int c=0;
    for(int j=0;j<nx;j++){
      for(int k=0;k<ny;k++){
        if(dist(j,k) >= binlow(i) && dist(j,k) < binlow(i+1)){
          c++;
          kone(i) += tem(j,k);
          done(i) += dist(j,k);
        }
      }
    }
    kone(i) /= c;
    done(i) /= c;
  }

  //now spline interpolate to generate new template array
  tplate.resize(nx,ny);
  {
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, nbins-1);
    gsl_spline_init(spline, &done(0), &kone(0), nbins-1);

    for(int i=0;i<nx;i++){
      for(int j=0;j<ny;j++){
        int ti = (i-xcind)%nx;
        int tj = (j-ycind)%ny;
        int shifti = (ti < 0) ? nx+ti : ti;
        int shiftj = (tj < 0) ? ny+tj : tj;
        if(dist(i,j) <= spline->interp->xmax && dist(i,j) >= spline->interp->xmin)
          tplate(shifti,shiftj) = gsl_spline_eval(spline, dist(i,j), acc);
        else if (dist(i,j) >spline->interp->xmax)
          tplate(shifti,shiftj) = kone(kone.size()-1);
        else if (dist(i,j) <spline->interp->xmin)
          tplate(shifti,shiftj) = kone(0);


      }
    }
    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);
  }

  return 1;

}


//----------------------------- o ---------------------------------------

///Produce a gaussian template in place of the kernel map
template <typename DerivedA>
void prepareGaussianTemplate(DerivedA &mapstruct){

  //collect what we need
  Eigen::VectorXd xgcut(nx);
  for(int i=0;i<nx;i++) xgcut(i) = mapstruct.rowcoordphys(i);
  Eigen::VectorXd ygcut(ny);
  for(int i=0;i<ny;i++) ygcut(i) = mapstruct.colcoordphys(i);
  Eigen::MatrixXd tem(nx, ny);
  for(int i=0;i<nx;i++) for(int j=0;j<ny;j++)
                          tem(i,j) = 0;

  Eigen::MatrixXd dist(nx,ny);
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
      dist(i,j) = sqrt(pow(xgcut(i),2)+pow(ygcut(j),2));

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
  double fwhm = ap->getGaussianTemplateFWHM();
  double sig = fwhm/2.3548;
  for(int i=0;i<nx;i++)
    for(int j=0;j<ny;j++)
      tplate(i,j) = exp(-0.5*pow(dist(i,j)/sig,2.));

  //and shift the gaussian to peak at the origin
  shift(tplate,-xcind, -ycind);

  return 1;

}

//void WienerFilter::gaussFilter(Coaddition *cmap){
//
//	prepareGaussianTemplate(cmap);
//
//	//Get the FFT of the template
//
//	size_t nx = cmap->signal->getNrows();
//	size_t ny = cmap->signal->getNcols();
//
//	fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*nx*ny);
//	fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*nx*ny);
//	fftw_complex *ftplate = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*nx*ny);
//	fftw_plan pf = fftw_plan_dft_2d(nx, ny, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
//	fftw_plan pr = fftw_plan_dft_2d(nx, ny, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
//
//	for (int i=0; i<nx; i++)
//		for (int j=0; j<ny; j++){
//			in(i*ny +j,0)= tplate(i,j);
//			in(i*ny +j,1)= 0.0;
//		}
//	fftw_execute(pf);
//
//	for (int i=0; i<nx; i++)
//		for (int j=0; j<ny; j++){
//			ftplate (i*ny+j,0) = out(i*ny+j,0);
//			ftplate (i*ny+j,1) = out(i*ny+j,1);
//		}
//
//	//Now we get the signal map
//
//	for (int i=0; i<nx; i++)
//		for (int j=0; j<ny; j++){
//			in(i*ny +j,0)= mapstruct.signal(i,j);
//			in(i*ny +j,1)= 0.0;
//		}
//	fftw_execute(pf);
//
//	//Apply filter
//	for (int i=0; i<nx; i++)
//		for (int j=0; j<ny; j++){
//			out (i*ny+j,0) *= ftplate(i*ny+j,0);
//			out (i*ny+j,1) *= ftplate(i*ny+j,1);
//		}
//
//	fftw_execute(pr);
//
//	for (int i=0; i<nx; i++)
//		for (int j=0; j<ny; j++){
//			mapstruct.signal(i,j) = out(i*ny+j,0);
//		}
//
//	fftw_free(in);
//	fftw_free(out);
//	fftw_free(ftplate);
//	fftw_destroy_plan(pf);
//	fftw_destroy_plan(pr);
//
//}

template <typename DerivedA>
void simpleWienerFilter2d(DerivedA mapstruct){

        Eigen::MatrixXd mapPsd = cmap->signal->psd2d;
        string noisePsdFile = ap->getAvgNoisePsdFile();

        NcFile ncfid = NcFile(noisePsdFile.c_str(), NcFile::ReadOnly);

        int nx2dpsd = ncfid.get_dim("nxpsd_2d")->size();
        int ny2dpsd = ncfid.get_dim("nypsd_2d")->size();

        if (nx2dpsd != mapPsd.nrows() || ny2dpsd != mapPsd.ncols()){
                cerr<<"Signal and Noise maps dimension does not agree"<<endl;
                cerr<<"Please check your pixel size parameter in Analysis xml file"<<endl;
                cerr<<"X size: "<<nx2dpsd << "vs"<< mapPsd.nrows()<<endl;
                cerr<<"Y size: "<<ny2dpsd << "vs"<< mapPsd.ncols()<<endl;
                exit (-1);
        }
        Eigen::MatrixXd noisePsd(nx2dpsd, ny2dpsd);

        NcVar *vNoisePsd = ncfid.get_var("psd_2d");
        if (!vNoisePsd->get(&noisePsd(0,0),nx2dpsd,ny2dpsd)){
                cerr<<"No 2d psd in noise average psd file."<<endl;
                exit(-1);
        }

        Eigen::MatrixXd psdFilter (nx2dpsd,ny2dpsd,0.0);
        double sn = 0;
        for (int i=0; i<nx2dpsd; i++)
                for (int j=0; j<ny2dpsd; j++){
                        sn = noisePsd(i,j)+mapPsd(i,j);
                        if (sn == 0){
                                psdFilter(i,j)=0.0;
                        }else{
                                psdFilter(i,j)= mapPsd(i,j)/sn;
                        }

                }
        //Now get the FFT from the map
        fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*nx2dpsd*ny2dpsd);
        fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*nx2dpsd*ny2dpsd);
        fftw_plan pf = fftw_plan_dft_2d(nx2dpsd, ny2dpsd, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_plan pr = fftw_plan_dft_2d(nx2dpsd, ny2dpsd, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);

        Eigen::MatrixXd hann = mapmaking::internal::hanning(nx2dpsd,ny2dpsd);

        int mx = mapstruct.signal.nrows();
        int my = mapstruct.signal.ncols();

        int dx = (mx-nx2dpsd)/2.0;
        int dy = (my-nx2dpsd)/2.0;

        for (int i=0; i<nx2dpsd; i++)
                for (int j=0; j<ny2dpsd; j++){
                        in(i*ny2dpsd +j,0)= mapstruct.signal(dx+i,dy+j);
                        in(i*ny2dpsd +j,1)= 0.0;
                }
        fftw_execute(pf);
        //Apply filter to signal
        for (int i=0; i<nx2dpsd; i++)
                        for (int j=0; j<ny2dpsd; j++){
                                /*in(i*ny2dpsd+j,0) =*/ out(i*ny2dpsd +j,0)*=psdFilter(i,j);
                                /*in(i*ny2dpsd+j,0) =*/ out(i*ny2dpsd +j,1)*=psdFilter(i,j);
                        }
        fftw_execute(pr);

        cerr<<"Making zero area outside the coverage region:"<<dx<<dy<<endl;
        mapstruct.signal.assign(mx,my, 0.0);
        double norm = nx2dpsd*ny2dpsd;
        for (int i=0; i<nx2dpsd; i++)
                        for (int j=0; j<ny2dpsd; j++){
                                mapstruct.signal(dx+i,dy+j)=in(i*ny2dpsd +j,0) /norm;
                        }
}
} //namespace
