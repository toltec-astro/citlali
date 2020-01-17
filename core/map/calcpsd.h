#pragma once
#include "map_utils.h"

namespace mapmaking {

enum FFTdirection2 {
    forward2 = 0,
    backward2 = 1
};


//This function does a 2D FFT since its not supported in Eigen
template<FFTdirection2 direc, typename DerivedA>
Eigen::VectorXcd fft2w2(Eigen::DenseBase<DerivedA> &vecIn, int nx, int ny){
    const int nRows = nx;//matIn.rows();
    const int nCols = ny;//matIn.cols();

    //Eigen::Map<Eigen::MatrixXd> matIn2(vecIn.derived().data(),nx,ny);

    Eigen::MatrixXcd matIn(nRows,nCols);

    //Keep in for loop for now to ensure matrix is filled in correct order.
    for(int i =0;i<nx;i++){
        for(int j=0;j<ny;j++)
            matIn(i,j) = vecIn(ny*i+j);
    }

    Eigen::FFT<double> fft;
    //fft.SetFlag(Eigen::FFT<double>::HalfSpectrum);
    fft.SetFlag(Eigen::FFT<double>::Unscaled);
    Eigen::MatrixXcd matOut(nRows, nCols);

    //Loop through rows and do 1D FFT
    for (int k = 0; k < nRows; ++k) {
        Eigen::VectorXcd tmpOut(nCols);
        if constexpr(direc == forward2){
            fft.fwd(tmpOut, matIn.row(k));
        }
        else{
            fft.inv(tmpOut, matIn.row(k));
        }
        matOut.row(k) = tmpOut;
    }

    //Loop through columns and do 1D FFT
    for (int k = 0; k < nCols; ++k) {
        Eigen::VectorXcd tmpOut(nRows);
        if constexpr(direc == forward2){
            fft.fwd(tmpOut, matOut.col(k));
        }
        else{
            fft.inv(tmpOut, matOut.col(k));
        }
        matOut.col(k) = tmpOut;
    }


    Eigen::VectorXcd vec3(nx*ny);

    //Fill up 1D vector by a for loop to ensure order is correct.
    for(int i =0;i<nx;i++){
        for(int j=0;j<ny;j++)
            vec3(ny*i+j) = matOut(i,j);
    }

    return vec3;
}

//This class holds the psd results and outputs them to a netcdf file
class psdclass {
public:
    Eigen::VectorXd psd;
    Eigen::VectorXd psdFreq;
    Eigen::MatrixXd psd2d;
    Eigen::MatrixXd psd2dFreq;

    psdclass() {}

    int toNcFile(const std::string &filepath) {

      try{
          //Create NetCDF file
          netCDF::NcFile fo(filepath, netCDF::NcFile::replace);

          netCDF::NcDim nrows = fo.addDim("nrows", psd2d.rows());
          netCDF::NcDim ncols = fo.addDim("ncols", psd2d.cols());
          netCDF::NcDim nn = fo.addDim("nn", psd.size());

          std::vector<netCDF::NcDim> dims;
          dims.push_back(nrows);
          dims.push_back(ncols);

          auto psd2d_var = "psd2d";
          auto psd2dFreq_var = "psd2dFreq";
          auto psd_var = "psd";
          auto psdFreq_var = "psdFreq";

          netCDF::NcVar psd2d_data = fo.addVar(psd2d_var, netCDF::ncDouble, dims);
          netCDF::NcVar psd2dFreq_data = fo.addVar(psd2dFreq_var, netCDF::ncDouble, dims);
          netCDF::NcVar psd_data = fo.addVar(psd_var, netCDF::ncDouble, nn);
          netCDF::NcVar psdFreq_data = fo.addVar(psdFreq_var, netCDF::ncDouble, nn);

          psd2d.transposeInPlace();
          psd2dFreq.transposeInPlace();

          psd2d_data.putVar(psd2d.data());
          psd2dFreq_data.putVar(psd2dFreq.data());
          psd_data.putVar(psd.data());
          psdFreq_data.putVar(psdFreq.data());

          fo.close();

    }
      catch(netCDF::exceptions::NcException& e)
    {e.what();
        return NC_ERR;
    }
    return 0;
    }
};

namespace internal {

//Hanning window function.  Move to map_utils?
Eigen::MatrixXd hanning(int n1in, int n2in){
  double a = 2.*pi/n1in;
  double b = 2.*pi/n2in;

  Eigen::VectorXd index = Eigen::VectorXd::LinSpaced(n1in,0,n1in-1);
  Eigen::VectorXd row = -0.5 * (index*a).array().cos() + 0.5;

  index.resize(n2in);
  index = Eigen::VectorXd::LinSpaced(n2in,0,n2in-1);
  Eigen::VectorXd col = -0.5 * (index*b).array().cos() + 0.5;

  Eigen::MatrixXd han(n1in,n2in);
  for(Eigen::Index i=0;i<n1in;i++)
    for(Eigen::Index j=0;j<n2in;j++)
      han(i,j) = row[i]*col[j];

  return han;
}

double select(vector<double> input, int index){
  unsigned int pivotIndex = rand() % input.size();
  double pivotValue = input[pivotIndex];
  vector<double> left;
  vector<double> right;
  for (unsigned int x = 0; x < input.size(); x++) {
    if (x != pivotIndex) {
      if (input[x] > pivotValue) {
        right.push_back(input[x]);
      } else {
        left.push_back(input[x]);
      }
    }
  }
  if ((int)left.size() == index) {
    return pivotValue;
  } else if ((int)left.size() < index) {
    return select(right, index - left.size() - 1);
  } else {
    return select(left, index);
  }
}


template <typename DerivedA>
double findWeightThresh(DerivedA &mapstruct, double coverageCut) {
  // number of elements in map
  vector<double> og;
  for (Eigen::Index x = 0; x < mapstruct.nrows; x++) {
    for (Eigen::Index y = 0; y < mapstruct.ncols; y++) {
      if (mapstruct.wtt(x,y) > 0.) {
        og.push_back(mapstruct.wtt(x,y));
      }
    }
  }

  double covlim;
  int covlimi;
  covlimi = 0.75 * og.size();
  covlim = select(og, covlimi);

  double mval;
  double mvali;
  mvali = floor((covlimi + og.size()) / 2.);
  mval = select(og, mvali);
  double weightCut = coverageCut * mval;

  return weightCut;

}

//Sets the coverage range for a given cut in map weight values.
template <typename DerivedA>
std::tuple<Eigen::VectorXd, Eigen::VectorXd> setCoverageCutRanges(DerivedA &mapstruct, double weightCut) {

  Eigen::VectorXd cutXRange(2);
  Eigen::VectorXd cutYRange(2);

  cutXRange[0] = 0;
  cutXRange[1] = mapstruct.nrows - 1;
  cutYRange[0] = 0;
  cutYRange[1] = mapstruct.ncols - 1;

  // find lower row bound
  bool flag = false;
  for (int i = 0; i < mapstruct.nrows; i++) {
    for (int j = 0; j < mapstruct.ncols; j++) {
      if (mapstruct.wtt(i,j) >= weightCut) {
        cutXRange[0] = i;
        flag = true;
        break;
      }
    }
    if (flag == true) {
      break;
    }
  }

  // find upper row bound
  flag = false;
  for (int i = mapstruct.nrows - 1; i > -1; i--) {
    for (int j = 0; j < mapstruct.ncols; j++) {
      if (mapstruct.wtt(i,j) >= weightCut) {
        cutXRange[1] = i;
        flag = true;
        break;
      }
    }
    if (flag == true) {
      break;
    }
  }

  // find lower column bound
  flag = false;
  for (int i = 0; i < mapstruct.ncols; i++) {
    for (int j = cutXRange[0]; j < cutXRange[1] + 1; j++) {
      if (mapstruct.wtt(j,i) >= weightCut) {
        cutYRange[0] = i;
        flag = true;
        break;
      }
    }
    if (flag == true) {
      break;
    }
  }

  // find upper column bound
  flag = false;
  for (int i = mapstruct.ncols - 1; i > -1; i--) {
    for (int j = cutXRange[0]; j < cutXRange[1] + 1; j++) {
      if (mapstruct.wtt(j,i) >= weightCut) {
        cutYRange[1] = i;
        flag = true;
        break;
      }
    }
    if (flag == true) {
      break;
    }
  }

  return std::make_tuple(cutXRange,cutYRange);

}

//Function to smooth edges.  Move to map_utils?
template <typename Derived>
void smooth_edge_truncate(Eigen::DenseBase<Derived> &inArr, Eigen::DenseBase<Derived> &outArr, int w){
  int nIn = inArr.size();
  int nOut = outArr.size();

  //as with idl, if w is even then add 1
  if(w % 2 == 0) w++;

  //do this all at once
  double winv = 1./w;
  int wm1d2 = (w-1)/2;
  double tmpsum;
  for(int i=0;i<nIn;i++){
    tmpsum=0;
    for(int j=0;j<w;j++){
      int addindex = i+j-wm1d2;
      if(addindex < 0) addindex=0;
      if(addindex > nIn-1) addindex=nIn-1;
      tmpsum += inArr[addindex];
    }
    outArr[i] = winv*tmpsum;
  }
}

} //internal


template <typename DerivedA>
/// calculates the 2D map psd
std::tuple<Eigen::VectorXd,Eigen::VectorXd,Eigen::MatrixXd,Eigen::MatrixXd> calcMapPsd(DerivedA &mapstruct, double coverageCut) {
  // make sure we've got up to date coverage cut indices
  double weightCut = mapmaking::internal::findWeightThresh(mapstruct,coverageCut);
  auto [cutXRange, cutYRange] = mapmaking::internal::setCoverageCutRanges(mapstruct,weightCut);

  // make sure our coverage cut map has an even number
  // of rows and columns
  Eigen::Index nx = cutXRange[1] - cutXRange[0] + 1;
  Eigen::Index ny = cutYRange[1] - cutYRange[0] + 1;
  Eigen::Index cxr0 = cutXRange[0];
  Eigen::Index cyr0 = cutYRange[0];
  Eigen::Index cxr1 = cutXRange[1];
  Eigen::Index cyr1 = cutYRange[1];
  if (nx % 2 == 1) {
    cxr1 = cutXRange[1] - 1;
    nx--;
  }
  if (ny % 2 == 1) {
    cyr1 = cutYRange[1] - 1;
    ny--;
  }

  Eigen::VectorXcd in(nx*ny);
  Eigen::VectorXcd out(nx*ny);

  // the matrix to get fft'd is cast into vector form in *in;
  Eigen::Index ii, jj, stride, index;
  for (Eigen::Index i = cxr0; i <= cxr1; i++)
    for (Eigen::Index j = cyr0; j <= cyr1; j++) {
      ii = i - cxr0;
      jj = j - cyr0;
      stride = cyr1 - cyr0 + 1;
      index = stride * ii + jj;
      in.real()(index) = mapstruct.signal(i,j);
      in.imag()(index) = 0.;
    }

  // apply a hanning window
  Eigen::MatrixXd h(nx,ny);

  h = internal::hanning(nx, ny);
  //for (Eigen::Index i = 0; i < nx; i++)
    //for (Eigen::Index j = 0; j < ny; j++)
      //in(ny * i + j,0) *= h(i,j);


  for(int i=0;i<nx;i++) for(int j=0;j<ny;j++) in.real()[ny*i+j] *= h(i,j);

  //in = in.array()*h.array();

  // calculate frequencies
  double diffx = mapstruct.rowcoordphys[1] - mapstruct.rowcoordphys[0];
  double diffy = mapstruct.colcoordphys[1] - mapstruct.colcoordphys[0];
  double xsize = diffx * nx;
  double ysize = diffy * ny;
  double diffqx = 1. / xsize;
  double diffqy = 1. / ysize;

  out = fft2w2<forward2>(in, nx,ny);
  out = out* xsize * ysize / nx / ny;

  //h = diffqx * diffqy * out.cwiseAbs2();

  for(int i=0;i<nx;i++)
      for(int j=0;j<ny;j++)
          h(i,j) = diffqx*diffqy*(pow(out.real()[ny*i+j],2)+pow(out.imag()[ny*i+j],2));


  // vectors of frequencies
  Eigen::VectorXd qx(nx);
  Eigen::VectorXd qy(ny);
  Eigen::Index shift = nx / 2 - 1;
  for (Eigen::Index i = 0; i < nx; i++) {
    index = i - shift;
    if (index < 0)
        index += nx;

    qx[index] = diffqx * (i - (nx / 2 - 1));
  }
  shift = ny / 2 - 1;
  for (Eigen::Index i = 0; i < ny; i++) {
    index = i - shift;
    if (index < 0)
      index += ny;
    qy[index] = diffqy * (i - (ny / 2 - 1));
  }

  // shed first row and column of h, qx, qy
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> pmfq = h.block(1,1,nx-1,ny-1);

  for (Eigen::Index i = 0; i < nx - 1; i++)
    qx[i] = qx[i + 1];
  for (Eigen::Index j = 0; j < ny - 1; j++)
    qy[j] = qy[j + 1];

  // matrices of frequencies and distances
  //Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> qmap(nx - 1, ny - 1);

  Eigen::MatrixXd qmap = sqrt(qx.tail(nx-1).replicate(1,ny-1).array().pow(2).rowwise() + qy.tail(ny-1).transpose().array().pow(2));
  Eigen::MatrixXd qsymm = qx.tail(nx-1).replicate(1,ny-1).array().rowwise()* qy.tail(ny-1).transpose().array();

  // find max of nx and ny and correspoinding diffq
  Eigen::Index nn;
  double diffq;
  if (nx > ny) {
    nn = nx / 2 + 1;
    diffq = diffqx;
  } else {
    nn = ny / 2 + 1;
    diffq = diffqy;
  }

  // generate the final vector of frequencies
  Eigen::VectorXd psdFreq;
  psdFreq.resize(nn);
  for (Eigen::Index i = 0; i < nn; i++)
    psdFreq[i] = diffq * (i + 0.5);

  // pack up the final vector of psd values
  Eigen::VectorXd psd(nn);

  for (Eigen::Index i = 0; i < nn; i++) {
    int countS = 0;
    int countA = 0;
    double psdarrS = 0.;
    double psdarrA = 0.;
    for (Eigen::Index j = 0; j < nx - 1; j++)
      for (Eigen::Index k = 0; k < ny - 1; k++) {
        if ((Eigen::Index)(qmap(j,k) / diffq) == i && qsymm(j,k) >= 0.) {
          countS++;
          psdarrS += pmfq(j,k);
        }
        if ((Eigen::Index)(qmap(j,k) / diffq) == i && qsymm(j,k) < 0.) {
          countA++;
          psdarrA += pmfq(j,k);
        }
      }
    if (countS != 0)
      psdarrS /= countS;
    if (countA != 0)
      psdarrA /= countA;
    psd[i] = min(psdarrS, psdarrA);
  }

  // smooth the psd with a 10-element boxcar filter
  Eigen::VectorXd tmp(nn);
  internal::smooth_edge_truncate(psd, tmp, 10);
  psd = tmp;

  return std::make_tuple(psd,psdFreq,pmfq,qmap);
}
} //namespace
