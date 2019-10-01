#pragma once

#include <Eigen/Dense>
#include <iostream>

//#include <lali/dataset.h>
//#include <optional>

#include <boost/math/constants/constants.hpp>

#include "../../common_utils/src/utils/algorithm/mlinterp/mlinterp.hpp"

static double pi = boost::math::constants::pi<double>();

using namespace std;

namespace  observation {

typedef std::map<string,Eigen::Matrix<double,Eigen::Dynamic,1>> pointing;

namespace internal {

template<typename DerivedA>
void findUniqueTelUtc(DerivedA &telescope_data){

    Eigen::Index npts = telescope_data["TelUtc"].size();

    //telUtc should be increasing or constant, so let's test for this first
    for(Eigen::Index i=1;i<npts;i++){
        if(telescope_data["TelUtc"](i) < telescope_data["TelUtc"](i-1)){
            SPDLOG_INFO("findUniqueTelUtc(): telUtc is not constant or increasing.");
            SPDLOG_INFO("Measuring glitch ...");

            //measure the glitch.  If it's less than 32 samples we will ignore it for now
            Eigen::Index bad=1;
            Eigen::Index nglitch=0;
            for(Eigen::Index j=0;j<32;j++){
                if(telescope_data["TelUtc"](i+j) > telescope_data["TelUtc"](i-1)){
                    bad=0;
                    nglitch=j+1;
                    break;
                }
            }

            if(bad){
                SPDLOG_WARN("Warning, there may be large glitches in telUtc in this data file.");
            }

            else{
                SPDLOG_WARN("UTC glitch is {} samples long.  This is less than 32 samples.  "
                            "Ignoring it.  Consider throwing out this data file.");
            }
        }
    }

    //count up the unique values
    int count=1;
    for(Eigen::Index i=1;i<npts;i++){
        if(telescope_data["TelUtc"](i) > telescope_data["TelUtc"](i-1)){
            count++;
        }
    }

    Eigen::Index nUnique;

    //build the new arrays
    telescope_data["telUtcUnique"].resize(count,1);
    telescope_data["locUnique"].resize(count,1);
    telescope_data["telUtcUnique"](0) = telescope_data["TelUtc"](0);
    telescope_data["locUnique"](0) = 0;

    int counter=1;
    for(Eigen::Index i=1;i<npts;i++){
        if(telescope_data["TelUtc"](i) > telescope_data["TelUtc"](i-1)){
            telescope_data["telUtcUnique"](counter) = telescope_data["TelUtc"](i);
            telescope_data["locUnique"](counter) = i;
            counter++;
        }
    }

    nUnique = count;

    SPDLOG_INFO("findUniqueTelUtc(): {} unique samples", nUnique);
}

template <typename DerivedA>
void removeDropouts(Eigen::DenseBase<DerivedA> &sig, Eigen::Index npts){
    //sometimes the LMT signals drop out.  Replace these with average of
    //adjacent signals
    for(Eigen::Index i=1;i<npts-1;i++){
        if(sig(i) <= 1.e-9){
            //this is a dropout
            sig(i) = (sig(i-1)+sig(i+1))/2.;
        }
    }
    if(sig(0) != sig(0)){
        SPDLOG_INFO("removeDropouts(): First data point is corrupted as a dropout.  "
                    "Setting equal to adjacent data point.");
        sig(0) = sig(1);
    }
    if(sig(npts-1) != sig(npts-1)){
        SPDLOG_INFO("removeDropouts(): Last data point is corrupted as a dropout.  "
                    "Setting equal to adjacent data point.");
    sig(npts-1) = sig(npts-2);
  }
}

template <typename DerivedA>
void deNanSignal(Eigen::DenseBase<DerivedA> &sig, Eigen::Index npts){
    //need to find single point NaNs and replace with average of
    //bracketing values
    for(Eigen::Index i=1;i<npts-1;i++){
        if(sig(i) != sig(i)){
            //this is a NaN
            sig(i) = (sig(i-1)+sig(i+1))/2.;
        }
    }
    if(sig(0) != sig(0)){
        SPDLOG_INFO("deNanSignal(): First data point is corrupted as a NaN. "
                    "Setting equal to adjacent data point.");
        sig(0) = sig(1);
    }
    if(sig(npts-1) != sig(npts-1)){
        SPDLOG_INFO("deNanSignal(): Last data point is corrupted as a NaN. "
                    "Setting equal to adjacent data point.");
        sig(npts-1) = sig(npts-2);
  }
}

//turn wrapped signals to monotonically increasing
template <typename DerivedA>
void correctRollover(Eigen::DenseBase<DerivedA> &sig, double low, double high,
                     double ulim, Eigen::Index npts)
{
  double mx=sig(0);
  double mn=sig(0);
  for(Eigen::Index i=1;i<npts;i++){
    mx = (mx > sig(i)) ? mx : sig(i);
    mn = (mn < sig(i)) ? mn : sig(i);
  }
  if(mx > high && mn < low){
    for(Eigen::Index i=0;i<npts;i++){
      if(sig(i)<low) sig(i)+=ulim;
    }
  }
}

template <typename DerivedA>
void correctOutOfBounds(Eigen::DenseBase<DerivedA> &sig, double low, double high, Eigen::Index npts)
{
  for(Eigen::Index i=1;i<npts-1;i++){
    if(sig(i)<low || sig(i)>high) sig(i) = (sig(i-1)+sig(i+1))/2.;
  }

  if(sig(0)<low || sig(0)>high){
      SPDLOG_INFO("correctOutOfBounds(): First data point is out of bounds."
                  "Setting equal to adjacent data point.");
      sig(0) = sig(1);
  }

  if(sig(npts-1)<low || sig(npts-1)>high){
      SPDLOG_INFO("correctOutOfBounds(): Last data point is out of bounds."
                   "Setting equal to adjacent data point.");
      sig(npts-1) = sig(npts-2);
  }
}

///interpolate the pointing signals
template <typename DerivedA>
void alignWithDetectors(DerivedA &telescope_data,
                        const double timeoffset){

    Eigen::Index nunique = telescope_data["locUnique"].size();

    SPDLOG_INFO("alignWithDetectors() with {} samples", nunique);

    Eigen::VectorXd xx =  telescope_data["telUtcUnique"].array() + timeoffset/3600.;
    Eigen::VectorXd yy(nunique);

    Eigen::Matrix<Eigen::Index,1,1> nu;
    nu << nunique;

    for (const auto &it : telescope_data){
        if (it.first!="Hold" && it.first!="AztecUtc" && it.first!="locUnique" && it.first!="telUtcUnique"){

            for(Eigen::Index i=0;i<nunique;i++){
                yy[i]=telescope_data[it.first](telescope_data["locUnique"].template cast<Eigen::Index>()(i));
            }

            Eigen::Index npts = telescope_data[it.first].size();
            mlinterp::interp(nu.data(), npts,
                             yy.data(), telescope_data[it.first].data(),
                             xx.data(), telescope_data["AztecUtc"].data());
        }
    }

  SPDLOG_INFO("Aligned with Detectors");
}

template <typename DerivedA>
void absToPhysHorPointing(DerivedA &telescope_data){
    Eigen::Index npts = telescope_data["TelAzAct"].rows();

    for(Eigen::Index i=0;i<npts;i++){
        if((telescope_data["TelAzAct"](i)-telescope_data["SourceAz"](i)) > 0.9*2.*pi){
            telescope_data["TelAzAct"](i) -= 2.*pi;
        }
    }

    telescope_data["TelAzPhys"] = (telescope_data["TelAzAct"].array() - telescope_data["SourceAz"].array())*(telescope_data["TelElDes"].array().cos()) - telescope_data["TelAzCor"].array();
    telescope_data["TelElPhys"] = telescope_data["TelElAct"] - telescope_data["SourceEl"] - telescope_data["TelElCor"];
}

} //namespace internal

template<typename DerivedA>
void obs(Eigen::DenseBase<DerivedA> &scanindex, pointing &telescope_data,
                 const bool timeChunk, const int samplerate, const double timeoffset){

    Eigen::Index npts = telescope_data["Hold"].rows();
    Eigen::Index nscans;

    for (const auto &it : telescope_data){
        if (it.first!="Hold"){
            if(it.first!="TelUtc" && it.first!="AztecUtc"){
                internal::removeDropouts(telescope_data[it.first], npts);
                internal::deNanSignal(telescope_data[it.first], npts);
                internal::correctOutOfBounds(telescope_data[it.first], 1.e-9,5.*pi, npts);
                internal::correctRollover(telescope_data[it.first], pi, 1.99*pi, 2.0*pi, npts);
            }
            else if (it.first=="TelUtc" || it.first=="AztecUtc") {
                internal::removeDropouts(telescope_data[it.first], npts);
                internal::deNanSignal(telescope_data[it.first], npts);
                internal::correctOutOfBounds(telescope_data[it.first], 1.e-9,30, npts);
            }
        }
    }

    //LMT's UTC is in radians so convert to hours
    telescope_data["TelUtc"] = telescope_data["TelUtc"]*24./2./pi;
    telescope_data["AztecUtc"] = telescope_data["AztecUtc"].array() - (telescope_data["AztecUtc"]-telescope_data["TelUtc"]).mean();

    internal::correctRollover(telescope_data["TelUtc"], 10., 23., 24., npts);
    internal::correctRollover(telescope_data["AztecUtc"], 10., 23., 24., npts);

    internal::findUniqueTelUtc(telescope_data);
    internal::alignWithDetectors(telescope_data, timeoffset);

    internal::absToPhysHorPointing<pointing>(telescope_data);

    //the goal is to pack up the turning array
    Eigen::Matrix<bool, Eigen::Dynamic, 1> turning(npts);
    Eigen::Matrix<bool, Eigen::Dynamic, 1> flagT2(npts);
    flagT2.setOnes();

    //recast hold from double to int
    Eigen::Matrix<int,Eigen::Dynamic,1> holdint(npts);
    holdint = telescope_data["Hold"].cast<int>();

    SPDLOG_INFO("hold {}", logging::pprint(telescope_data["Hold"]));

    //the telescope is turning at end of scan when hold&8=1
    for(Eigen::Index i=0;i<npts;i++) {
          turning[i] = (holdint[i]&8);
    }

    // Raster scan mode for timeChunk=0.
    if (timeChunk == 0){

      SPDLOG_INFO("timeChunk is zero. Raster scan mode enabled.");

      //count up the number of scans and keep track of the scan number
      nscans=0;
      for(Eigen::Index i=1;i<npts;i++){
        if(turning[i]-turning[i-1]==1){
            nscans++;
        }
      }

      if (turning(npts-1) == 0){
          nscans++;
      }

      //Make scanindex array
      scanindex.derived().resize(4,nscans);
      int counter=-1;
      if(!turning[0]){
        scanindex(0,0) = 1;
        counter++;
      }

      for(Eigen::Index i=1;i<npts;i++){
          if(turning[i]-turning[i-1] < 0){
              //this is the beginning of the next scan intentionally sacrificing a sample to keep
              counter++;
              scanindex(0,counter) = i+1;
          }
          if(turning[i]-turning[i-1] > 0){
              //one sample ago was the end of the scan
              scanindex(1,counter) = i-1;
          }
      }

      //the last sample is the end of the last scan
      scanindex(1,nscans-1) = npts-1;
    }

    //Here we set up the 3rd and 4th scanindex so that we don't lose data during lowpassing
    scanindex.row(2) = scanindex.row(0).array() - 32;
    scanindex.row(3) = scanindex.row(1).array() + 32;

    scanindex(2,0) = scanindex(0,0);
    scanindex(0,0) = scanindex(0,0) + 32;
    scanindex(3,nscans-1) = scanindex(1,nscans-1);
    scanindex(1,nscans-1) = scanindex(1,nscans-1) - 32;

    Eigen::Matrix<Eigen::Index,Eigen::Dynamic,Eigen::Dynamic> tmpSI(4,nscans);
    tmpSI = scanindex;


    /*
    //do a final check of scan length.  If a scan is less than 2s of data then delete it
    int nBadScans=0;
    int sum=0;
    Eigen::Matrix<bool,Eigen::Dynamic,1> isBadScan(nscans);
    for(Eigen::Index i=0;i<nscans;i++)
    {
        sum=0;
        for(Eigen::Index j=tmpSI(0,i);j<(tmpSI(1,i)+1);j++) sum+=1;//obsFlags[j];

        if(sum < 2.*samplerate){
        nBadScans++;
        isBadScan[i]=1;
        }
        else {
            isBadScan[i]=0;
        }
    }

    if(nBadScans > 0){
        SPDLOG_ERROR("{} scans with duration less than 2 seconds detected.  Ignoring", nBadScans);
    }

    //pack up the scan indices
    int c=0;
    scanindex.resize(4,nscans-nBadScans);
    for(Eigen::Index i=0;i<nscans;i++)
    {
        if(!isBadScan[i]){
            scanindex(0,c) = tmpSI(0,i);
            scanindex(1,c) = tmpSI(1,i);
            c++;
        }
    }
    nscans = nscans-nBadScans;
    */
}
} //namespace
