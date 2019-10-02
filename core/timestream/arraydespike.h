#pragma once

#include <Eigen/Dense>

#include "../../common_utils/src/utils/algorithm/mlinterp/mlinterp.hpp"

using namespace std;

namespace timestream {
namespace internal {

template <typename DerivedA>
void condition_flags(Eigen::DenseBase<DerivedA> &flags, Eigen::Index npts, Eigen::Index i){
    //condition flags so that if there is a spike we can make
    //one long flagged or unflagged region.first do spikes from 0 to 1
    for(Eigen::Index j=1;j<npts;j++){
      if(flags(j,i) == 1 &&
         flags(j-1,i) == 0 &&
         flags(j+1,i) == 0)
            flags(j,i) = 0;
    }
    //now spikes from 1 to 0
    for(Eigen::Index j=1;j<npts;j++){
      if(flags(j,i) == 0 &&
         flags(j-1,i) == 1 &&
         flags(j+1,i) == 1)
            flags(j,i) = 1;
    }
    //and the first and last samples
    flags(0,i) = flags(1,i);
    flags(npts-1,i) = flags(npts-2,i);
}

template <typename DerivedA, typename DerivedB>
int get_regions(Eigen::DenseBase<DerivedA> &flags, Eigen::DenseBase<DerivedB> &siFlags,
                 Eigen::DenseBase<DerivedB> &eiFlags, Eigen::Index npts, Eigen::Index i){
    Eigen::Index j = 0;
    int count = 0;
    while (j<npts){
      if(flags(j,i) == 0){
        int jstart=j;
        int sampcount=0;
        while(flags(j,i) == 0 && j<=npts){
          sampcount++;
          j++;
        }
        if(sampcount > 1){
                siFlags(count)=jstart;
                eiFlags(count) = j-1;
                count++;
        } else {
          j++;
        }
      } else {
        j++;
      }
    }
    return count;
}


} //namespace internal

template <typename DerivedA, typename DerivedB, typename DerivedC>
void fakeFlaggedData(Eigen::DenseBase<DerivedA> &scans, Eigen::DenseBase<DerivedB> &flags,
                     Eigen::DenseBase<DerivedC> &responsivity){
    //create fake data with following signal/noise sources:
    // 1) random noise based on calculated variance
    // 2) The linear baseline of spikey detector
    // 3) sky model (average of spike free detectors

    Eigen::Index ndet = flags.cols();
    Eigen::Index npts = flags.rows();

    Eigen::Index si=0;
    Eigen::Index ei=npts-1;
    bool useAllDet=0; //use all detectors regardless of spikes
    Eigen::Matrix<bool,Eigen::Dynamic,1> hasSpikes(ndet);
    hasSpikes.setZero();

    //figure out if there are any flag-free detectors
    Eigen::Index nFlagged=0;
    for(Eigen::Index i=0;i<ndet;i++){
        if ((flags.block(0,i,npts,1).array() == 0).any()){
            nFlagged++;
            hasSpikes(i) = 1;
        }
    }

    if(nFlagged > 0.5*ndet){
      SPDLOG_INFO("has fewer than 50% of the detectors flag-free.");
      SPDLOG_INFO("Using entire array (with spikes included) as sky model.");
      useAllDet=1;
    }

    //go through detectors with flags=0 and fake data
    for(Eigen::Index i=0;i<ndet;i++){
        if(hasSpikes(i)){
        //must assume there are multiple flagged regions
        //flags at beginning of scans are special cases
        internal::condition_flags(flags,npts,i);

        //count up the number of flagged regions of data in the scan
        Eigen::Index nflaggedregions=0;

        if(flags(ei,i) == 0){
          nflaggedregions++;
        }

        nflaggedregions += ((flags.block(1,i,npts-1,1) - flags.block(0,i,npts-1,1)).array() > 0).count()/2;

        //sanity check: nflaggedregions should not equal zero unless above
        //fixes to sampleflags did all that needed to be done.
        if(nflaggedregions==0) break;

        //find the start and end index for each flagged region
        Eigen::VectorXi siFlags(nflaggedregions);
        siFlags.setConstant(-1);
        Eigen::VectorXi eiFlags(nflaggedregions);
        eiFlags.setConstant(-1);

        int count;
        count = internal::get_regions(flags,siFlags,eiFlags,npts,i);

        Eigen::Index j;

        if(count != nflaggedregions){
          SPDLOG_INFO("Array::fakeFlaggedData(): count = {}",count);
          SPDLOG_INFO(" but it should be equal to nflaggedregions= {}",nflaggedregions);
          Eigen::VectorXd mysf(ei-si+1);
          for(j=si;j<=ei;j++) mysf(j-si) = (double) flags(j,i);
          exit(1);
        }

        //now loop on the number of flagged regions for the fix
        Eigen::VectorXd xx(2);
        Eigen::VectorXd yy(2);
        Eigen::Matrix<Eigen::Index, 1,1> tnpts;
        tnpts << 2;
        for(Eigen::Index j=0;j<nflaggedregions;j++){
            //FLAGGED DETECTOR
            //determine linear baseline for flagged region but use flat dc level if flagged at endpoints
            Eigen::Index nFlags = eiFlags(j)-siFlags(j);
            Eigen::VectorXd linOffset(nFlags);
            if(siFlags(j) == si){
                //first sample in scan is flagged so offset is flat
                //with the value of the last sample in the flagged region
                linOffset = scans.block(eiFlags(j)+1,i,nFlags,1);
            }

            else if(eiFlags(j) == ei){
                //last sample in scan is flagged so offset is flat
                //with the value of the first sample in the flagged region
                linOffset = scans.block(siFlags(j)-1,i,nFlags,1);
            }

            else {
                //in this case we linearly interpolate between the before and after good samples
                xx(0) = siFlags(j)-1;
                xx(1) = eiFlags(j)+1;
                yy(0) = scans(siFlags(j)-1,i);
                yy(1) = scans(eiFlags(j)+1,i);
                Eigen::VectorXd xLinOffset = Eigen::VectorXd::LinSpaced(nFlags,siFlags(j),siFlags(j) + nFlags);

                mlinterp::interp(tnpts.data(), nFlags,
                                 yy.data(), linOffset.data(),
                                 xx.data(), xLinOffset.data());
            }

            //ALL NONFLAGGED DETECTORS
            //do the same thing but for all detectors without spikes
            //count up spike-free detectors and store their values
            Eigen::Index detCount=0;
            detCount = (hasSpikes.array() == 0).count();
            if(useAllDet){
                detCount=ndet;
            }
            //storage
            Eigen::MatrixXd det(detCount,nFlags);     //detector values
            Eigen::VectorXd res(detCount);            //detector responsivities
            int c=0;
            for(Eigen::Index ii=0;ii<ndet;ii++){
                if(!hasSpikes(ii) || useAllDet){
                    //det.block(c,0,1,nFlags) = scans.block(siFlags(j),ii,nFlags,1);
                    for(Eigen::Index l=0;l<nFlags;l++){
                        det(c,l) = scans(siFlags(j)+l,ii);
                    }
                    res(c) = responsivity(ii);
                    c++;
                }
            }
              //for each of these go through and redo the offset bit
              Eigen::MatrixXd linOffsetOthers(detCount,nFlags);
              if(siFlags(j) == si){
                  //first sample in scan is flagged so offset is flat
                  //with the value of the last sample in the flagged region
                  for(Eigen::Index ii=0;ii<detCount;ii++){
                      linOffsetOthers.block(ii,0,1,nFlags).setConstant(det(ii,0));
                  }
                }
              else if(eiFlags(j) == ei){
                  //last sample in scan is flagged so offset is flat
                  //with the value of the first sample in the flagged region
                  for(Eigen::Index ii=0;ii<detCount;ii++){
                    linOffsetOthers.block(ii,0,1,nFlags).setConstant(det(ii,nFlags-1));
                  }

              }

              else {
                  //in this case we linearly interpolate between the before
                  //and after good samples
                  Eigen::VectorXd xLinOffset = Eigen::VectorXd::LinSpaced(nFlags,siFlags(j),siFlags(j) + nFlags);
                  Eigen::VectorXd tmpVec(nFlags);
                  xx(0) = siFlags(j)-1;
                  xx(1) = eiFlags(j)+1;
                  for(Eigen::Index ii=0;ii<detCount;ii++){
                      yy(0) = det(ii,0);
                      yy(1) = det(ii,nFlags-1);

                      mlinterp::interp(tnpts.data(), nFlags,
                                       yy.data(), tmpVec.data(),
                                       xx.data(), linOffset.data());
                      linOffsetOthers.row(ii) = tmpVec;
                    }
                }

                //subtract off the linear offset from the spike free dets
                det.block(0,0,detCount,nFlags) -= linOffsetOthers.block(0,0,detCount,nFlags);

                //scale det by responsivities and average to make sky model
                Eigen::VectorXd skyModel(nFlags);
                skyModel.setZero();

                for(Eigen::Index ii=0;ii<detCount;ii++){
                    skyModel += det.row(ii)/res(ii);
                }

                skyModel /= detCount;

              //find mean std dev of sky-model subtracted detectors
              Eigen::VectorXd stdDevFF(detCount);
              stdDevFF.setZero();
              double tmpMean;
              Eigen::VectorXd tmpVec(nFlags);
              for(Eigen::Index ii=0;ii<detCount;ii++){
                  tmpMean=0;
                  tmpVec = det.row(ii)/res(ii);//-skyModel;
                  tmpMean = tmpVec.mean();

                  stdDevFF(ii) = (tmpVec.array()-tmpMean).cwiseAbs2().sum();
                  stdDevFF(ii) = (nFlags == 1.) ? stdDevFF(ii)/nFlags : stdDevFF(ii)/(nFlags-1.);
                }

              double meanStdDev;
              meanStdDev = stdDevFF.cwiseSqrt().mean();

              //the noiseless fake data is then the sky model plus the
              //flagged detectors linear offset
              Eigen::VectorXd fake(nFlags);
              fake = linOffset;

              //for(Eigen::Index l=0;l<nFlags;l++) {
                //fake[l] = skyModel[l]*responsivity(i) + linOffset[l];
                //fake[l] = linOffset[l];
                //}

              //add noise to the fake signal
              meanStdDev *= responsivity(i);   //put back in volts

              //replace detector values with fake signal
              //std::mt19937 generator;
              //std::normal_distribution<double> dist(0., meanStdDev);

              scans.block(siFlags(j),i,nFlags,1) = fake;

              //for(Eigen::Index l=0;l<nFlags;l++){
                //scans(siFlags(j)+l,i) = fake[l];// + dist(generator);
              //}
        }
        }
    }
  //}//index k, looping on scans
}
} //namespace timestream
