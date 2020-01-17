#pragma once
#include "map_utils.h"

//Needed for config file input

#include "../timestream/timestream.h"

//Eigen Includes
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream> // cout

#include <boost/math/constants/constants.hpp>
#include <cmath>

#include "../../common_utils/src/utils/algorithm/ei_stats.h"

// Arcseconds in 360 degrees
#define ASEC_CIRC 1296000.0
// rad per arcsecond
#define RAD_ASEC (2.0*pi / ASEC_CIRC)

typedef std::map<string,Eigen::Matrix<double,Eigen::Dynamic,1>> pointing;

using namespace std;
using timestream::TCData;
using timestream::LaliDataKind;


//This structure holds the Maps as Eigen tensors.
namespace mapmaking{

class MapStruct
{
public:
    MapStruct() {}

    Eigen::Tensor<double, 2> signal;
    Eigen::Tensor<double, 2> wtt;
    Eigen::Tensor<double, 2> sigma;
    Eigen::Tensor<double, 2> kernel;
    Eigen::Tensor<double, 2> inttime;
    Eigen::Tensor<double,3> noisemaps;

    Eigen::Tensor<double, 2> filteredsignal;
    Eigen::Tensor<double, 2> filteredweight;
    Eigen::Tensor<double, 2> filteredsigma;
    Eigen::Tensor<double, 2> filteredkernel;
    Eigen::Tensor<double,3> filterednoisemaps;

    double pixelsize, mgrid_0, mgrid_1;
    Eigen::Index nrows, ncols, npixels, NNoiseMapsPerObs;
    Eigen::VectorXd rowcoordphys, colcoordphys;

    void resize(Eigen::Index ndet);
};


//Simple function to resize each matrix after the dimensions and number of rows/cols has
//been determined
void MapStruct::resize(Eigen::Index ndet){

    signal.resize(nrows,ncols);
    wtt.resize(nrows,ncols);
    sigma.resize(nrows,ncols);
    kernel.resize(nrows,ncols);
    inttime.resize(nrows,ncols);

    npixels = nrows*ncols;

    signal.setZero();
    wtt.setZero();
    sigma.setZero();
    kernel.setZero();
    inttime.setZero();
}

namespace internal {

//Convert from physical coordinates to absolute coordinates.
void physToAbs(double &pra, double &pdec, double &cra, double &cdec,
           double &ara, double &adec){
    double rho = sqrt(pow(pra,2)+pow(pdec,2));
    double c = atan(rho);
    if(c == 0.){
      ara = cra;
      adec = cdec;
    } else {
      double ccwhn0 = cos(c);
      double scwhn0 = sin(c);
      double ccdec = cos(cdec);
      double scdec = sin(cdec);
      double a1;
      double a2;
      a1 = ccwhn0*scdec + pdec*scwhn0*ccdec/rho;
      adec = asin(a1);
      a2 = pra*scwhn0/(rho*ccdec*ccwhn0 - pdec*scdec*scwhn0);
      ara = cra + atan(a2);
    }
}

//Get the number of rows and columns from map dimensions
template <typename DerivedA>
void getRowCol(DerivedA &mapstruct, double mgrid_0, double mgrid_1,
                       double x_max_act, double x_min_act, double y_max_act, double y_min_act){
    // start with the map dimensions but copy IDL utilities
    // for aligning the pixels in various maps

    double ps = mapstruct.pixelsize*RAD_ASEC;

    // explicitly center on mastergrid
    // require that maps are even dimensioned with a few extra pixels.
    int xminpix = ceil(abs(x_min_act / ps));
    int xmaxpix = ceil(abs(x_max_act / ps));
    xmaxpix = max(xminpix, xmaxpix);

    mapstruct.nrows = 2 * xmaxpix + 4;
    int yminpix = ceil(abs(y_min_act / ps));
    int ymaxpix = ceil(abs(y_max_act / ps));
    ymaxpix = max(yminpix, ymaxpix);
    mapstruct.ncols = 2 * ymaxpix + 4;
    mapstruct.npixels = mapstruct.nrows * mapstruct.ncols;

    SPDLOG_INFO("nrows {}", mapstruct.nrows);
    SPDLOG_INFO("ncols {}", mapstruct.ncols);
    SPDLOG_INFO("npixels {}", mapstruct.npixels);

    // physical coordinates: this grid is set up so that physical
    // coordinates are 0 at center of pixel near center of map

    mapstruct.rowcoordphys.resize(mapstruct.nrows);
    mapstruct.colcoordphys.resize(mapstruct.ncols);

    for (int i = 0; i < mapstruct.nrows; i++)
        mapstruct.rowcoordphys[i] = (i - (mapstruct.nrows + 1.) / 2.) * ps;
    for (int i = 0; i < mapstruct.ncols; i++)
        mapstruct.colcoordphys[i] = (i - (mapstruct.ncols + 1.) / 2.) * ps;

    // matrices of absolute coordinates
   /* Eigen::MatrixXd xCoordsAbs(nrows, ncols);
    Eigen::MatrixXd yCoordsAbs(nrows, ncols);

    for (int i = 0; i < nrows; i++)
        for (int j = 0; j < ncols; j++) {
            physToAbs(rowCoordsPhys[i], colCoordsPhys[j], mgrid_0,
                      mgrid_1, xCoordsAbs(i,j), yCoordsAbs(i,j));
        }*/
}


//Determine the weight of a scan ignoring bad flags
template <typename DerivedA, typename DerivedB>
double calcScanWeight(const Eigen::DenseBase<DerivedA> &scans, const Eigen::DenseBase<DerivedB> &flags, const double samplerate){

  //Get number of good samples in scan
  Eigen::Index numgood;
  numgood = (flags.derived().array()==1).count();

  //calculate the standard deviation
  double tmp = std::sqrt(((scans.derived().array() * flags.derived().template cast<double>().array())
                   -  (scans.derived().array() * flags.derived().template cast<double>().array()).sum()/numgood).square().sum()/(numgood - 1.0));


  //insist on at least 1s of good samples
  if (tmp!=tmp || numgood < samplerate)
      return 0.0;
  else
      return 1.0/pow(tmp,2.0);
}


//loop through all scans and get the weights.
template<typename DerivedA>
DerivedA calculateWeights(std::vector<TCData<LaliDataKind::PTC>> &ptcs, const int samplerate){
    Eigen::Index ndet = ptcs[0].scans.data.cols();
    Eigen::Index nscans = ptcs.size();

    Eigen::MatrixXd tmpwt(ndet, nscans);

    for (Eigen::Index i = 0;i<nscans;i++) {
        for (Eigen::Index j = 0;j<ndet;j++) {
            Eigen::Map<Eigen::VectorXd> scns(ptcs[i].scans.data.col(j).data(),ptcs[i].scans.data.rows());
            Eigen::Map<Eigen::Matrix<bool,Eigen::Dynamic,1>> flgs(ptcs[i].flags.data.col(j).data(),ptcs[i].flags.data.rows());

            tmpwt(j,i) = internal::calcScanWeight(scns,flgs,samplerate);
        }
    }

      Eigen::Map<Eigen::VectorXd> tmpwts (tmpwt.data(),nscans*ndet);
      double medweight = alg::median(tmpwts);

      for(Eigen::Index i=0;i<nscans;i++){
          for(Eigen::Index j=0;j<ndet;j++){
            if(tmpwt(j,i) > 2.* medweight){
                tmpwt(j,i) = medweight / 2.0;
            }
          }
      }
    return tmpwt;
}
} //namespace internal


//Beammap generation code
template <typename DerivedA, typename DerivedC>//, typename DerivedB>
void mapgen(std::vector<TCData<LaliDataKind::PTC>> &ptcs,
                 pointing &telescope_data,
                 double mgrid_0, double mgrid_1, const int samplerate,
                 DerivedC &mapstruct, Eigen::DenseBase<DerivedA> &offsets, Eigen::Index det,
                 Eigen::DenseBase<DerivedA> &tmpwts){

    //MapStruct map;

    //pixel size
    double ps = mapstruct.pixelsize*RAD_ASEC;

    Eigen::Index nscans = ptcs.size();
    Eigen::VectorXd lat, lon;

    //Matrix for how many times each pixel was visited
    Eigen::MatrixXi nValues(mapstruct.nrows,mapstruct.ncols);
    nValues.setZero();

    //get the pointing
    getPointing(telescope_data, lat, lon, offsets);

    //loop through scans
    for (Eigen::Index j = 0; j < nscans; j++) {
        Eigen::Index s = 0;
        Eigen::Index si = ptcs[j].scanindex.data(0);
        Eigen::Index ei = ptcs[j].scanindex.data(1);

        //Loop through each element in scan
        for (Eigen::Index k=si;k<ei;k++) {
            if(ptcs[j].flags.data(s,det)){
                double hx = ptcs[j].scans.data(s,det)*tmpwts(det,j);
                Eigen::Index irow = 0;
                Eigen::Index icol = 0;

                //get row and col index
                latlonPhysToIndex(lat[k], lon[k], irow, icol, mapstruct);

                //Populate that pixel
                mapstruct.signal(det,irow,icol) += hx;
                mapstruct.wtt(det,irow,icol) += tmpwts(det,j);

                nValues(irow,icol)++;
            } //if flags is True
            s++;
        } //per scan loop
    } //scan loop


    //Now we loop through again and normalize by the weight matrix.
    for(Eigen::Index irow = 0; irow < mapstruct.nrows; irow++)
        for(Eigen::Index icol = 0; icol < mapstruct.ncols; icol++){
            if(nValues(irow,icol) != 0)
            {
                mapstruct.signal(det,irow,icol) = -mapstruct.signal(det,irow,icol)/mapstruct.wtt(det,irow,icol);
                mapstruct.sigma(det,irow,icol) = 1./std::sqrt(mapstruct.wtt(det,irow,icol));
            }
            else
            {
                mapstruct.signal(det,irow,icol) = 0.;//std::nan("");
                mapstruct.sigma(det,irow,icol) = 0.;//std::numeric_limits<double>::infinity();
            }
        }
}
} //namespace
