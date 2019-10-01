#pragma once

#include <Eigen/Dense>
#include <iostream>

#include "../../common_utils/src/utils/algorithm/mlinterp/mlinterp.hpp"

using namespace std;

namespace timestream {
namespace internal {


template <typename DerivedA>
void condition_flags(Eigen::DenseBase<DerivedA> &flags, Eigen::Index npts, Eigen::Index i){
    //condition sampleflags so that if there is a spike we can make
    //one long flagged or unflagged region.
    //first do spikes from 0 to 1
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
                siFlags[count]=jstart;
                eiFlags[count] = j-1;
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
                     Eigen::DenseBase<DerivedC> &responsivity)
{
  //create fake data with following signal/noise sources:
  // 1) random noise based on calculated variance
  // 2) The linear baseline of spikey bolometer
  // 3) sky model (average of spike free bolometers

  //our pointer to the vector of good detectors

  //do this scan by scan
  //int nscans = scanindex.cols();
  Eigen::Index ndet = flags.cols();
  Eigen::Index npts = flags.rows();

  //for(int k=0;k<nscans;k++){
    Eigen::Index si=0;//scanindex(0);
    Eigen::Index ei=npts-1;//scanindex(1);
    bool useAllDet=0;      //use all detectors regardless of spikes
    Eigen::Matrix<bool,Eigen::Dynamic,1> hasSpikes(ndet);
    hasSpikes.setZero();

    //for(int i=0;i<ndet;i++) hasSpikes[i]=0;

    //figure out if there are any flag-free detectors
    Eigen::Index nFlagged=0;
    for(Eigen::Index i=0;i<ndet;i++){
        if ((flags.block(0,i,npts,1).array() == 0).any()){
            nFlagged++;
            hasSpikes[i] = 1;
        }


      //for(Eigen::Index j=si;j<=ei;j++){
        //if (flags(j,i)==0){
          //nFlagged++;
          //hasSpikes[i]=1;
          //break;
        //}
      //}
    }

    if(nFlagged > 0.5*ndet){
      //cerr << "Array::fakeFlaggedData(): scan #" << k;
      cerr << " has fewer than 50% of the detectors flag-free." << endl;
      cerr << "Using entire array (with spikes included) as sky model.";
      cerr << endl;
      useAllDet=1;
    }

    //go through detectors with flags=0 and fake data
    for(Eigen::Index i=0;i<ndet;i++){
      if(hasSpikes[i]){
        //must assume there are multiple flagged regions
        //flags at beginning of scans are special cases

          internal::condition_flags(flags,npts,i);

        //condition sampleflags so that if there is a spike we can make
        //one long flagged or unflagged region.
        //first do spikes from 0 to 1
        /*for(Eigen::Index j=si+1;j<ei;j++){
          if(flags(j,i) == 1 &&
             flags(j-1,i) == 0 &&
             flags(j+1,i) == 0)
                flags(j,i) = 0;
        }
        //now spikes from 1 to 0
        for(Eigen::Index j=si+1;j<ei;j++){
          if(flags(j,i) == 0 &&
             flags(j-1,i) == 1 &&
             flags(j+1,i) == 1)
                flags(j,i) = 1;
    }
        //and the first and last samples
        flags(si,i) = flags(si+1,i);
        flags(ei,i) = flags(ei-1,i);
        */

        //count up the number of flagged regions of data in the scan
        Eigen::Index nFlaggedRegions=0;

        if(flags(ei,i) == 0){
          nFlaggedRegions++;
        }

        nFlaggedRegions += ((flags.block(1,i,npts-1,1) - flags.block(0,i,npts-1,1)).array() > 0).count()/2;

        //for(Eigen::Index j=si+1;j<=ei;j++)
          //if(flags(j,i)-flags(j-1,i) > 0)
            //nFlaggedRegions++;

        //sanity check: nFlaggedRegions should not equal zero unless above
        //fixes to sampleflags did all that needed to be done.
        if(nFlaggedRegions==0) break;

        //find the start and end index for each flagged region
        Eigen::VectorXi siFlags(nFlaggedRegions);
        siFlags.setConstant(-1);
        Eigen::VectorXi eiFlags(nFlaggedRegions);
        eiFlags.setConstant(-1);

        int count;
        count = internal::get_regions(flags,siFlags,eiFlags,npts,i);

       /* int count=0;
        Eigen::Index j=si;
        while (j<ei){
          if(flags(j,i) == 0){
            int jstart=j;
            int sampcount=0;
            while(flags(j,i) == 0 && j<=ei){
              sampcount++;
              j++;
            }
            if(sampcount > 1){
                    siFlags[count]=jstart;
                    eiFlags[count] = j-1;
                    count++;
            } else {
              j++;
            }
          } else {
            j++;
          }
        }
        */

        Eigen::Index j;

        if(count != nFlaggedRegions){
          //cerr << "Scan k=" << k << ", Det i=" << i << endl;
          cerr << "Array::fakeFlaggedData(): count=" << count;
          cerr << " but it should be equal to nFlaggedRegions=";
          cerr << nFlaggedRegions << endl;
          Eigen::VectorXd mysf(ei-si+1);
          for(j=si;j<=ei;j++) mysf[j-si] = (double) flags(j,i);
          exit(1);
        }

        //now loop on the number of flagged regions for the fix
        Eigen::VectorXd xx(2);
        Eigen::VectorXd yy(2);
        Eigen::Matrix<Eigen::Index, 1,1> tnpts;
        tnpts << 2;
        for(Eigen::Index j=0;j<nFlaggedRegions;j++){

        //FLAGGED DETECTOR
        //determine linear baseline for flagged region
        //but use flat dc level if flagged at endpoints

          Eigen::Index nFlags = eiFlags[j]-siFlags[j];
          Eigen::VectorXd linOffset(nFlags);
          if(siFlags[j] == si){
            //first sample in scan is flagged so offset is flat
            //with the value of the last sample in the flagged region
            //for(int l=0;l<nFlags;l++)
              //linOffset[l] = scans(eiFlags[j]+1,i);

            linOffset = scans.block(eiFlags[j]+1,i,nFlags,1);

          } else if(eiFlags[j] == ei){
            //last sample in scan is flagged so offset is flat
            //with the value of the first sample in the flagged region
            //for(int l=0;l<nFlags;l++)
              //linOffset[l] = scans(siFlags[j]-1,i);

            linOffset = scans.block(siFlags[j]-1,i,nFlags,1);

          } else {
            //in this case we linearly interpolate between the before
            //and after good samples
            xx[0] = siFlags[j]-1;
            xx[1] = eiFlags[j]+1;
            yy[0] = scans(siFlags[j]-1,i);
            yy[1] = scans(eiFlags[j]+1,i);
            Eigen::VectorXd xLinOffset = Eigen::VectorXd::LinSpaced(nFlags,siFlags[j],siFlags[j] + nFlags);
            //for(int l=0;l<nFlags;l++) xLinOffset[l]=siFlags[j]+l;

            mlinterp::interp(tnpts.data(), nFlags,
                             yy.data(), linOffset.data(),
                             xx.data(), xLinOffset.data());
          }

          //ALL NONFLAGGED DETECTORS
          //do the same thing but for all detectors without spikes
          //count up spike-free detectors and store their values
          Eigen::Index detCount=0;
          detCount = (hasSpikes.array() == 0).count();
          //for(int ii=0;ii<ndet;ii++) if(!hasSpikes[ii]) detCount++;
          if(useAllDet) detCount=ndet;
          //storage
          Eigen::MatrixXd det(detCount,nFlags);     //detector values
          Eigen::VectorXd res(detCount);            //detector responsivities
          int c=0;
          for(Eigen::Index ii=0;ii<ndet;ii++){
            if(!hasSpikes[ii] || useAllDet){
                //det.block(c,0,1,nFlags) = scans.block(siFlags[j],ii,nFlags,1);
              for(Eigen::Index l=0;l<nFlags;l++)
                det(c,l) = scans(siFlags[j]+l,ii);
              res[c] = responsivity(ii);
              c++;
            }
          }
          //for each of these go through and redo the offset bit
          Eigen::MatrixXd linOffsetOthers(detCount,nFlags);
          if(siFlags[j] == si){
            //first sample in scan is flagged so offset is flat
            //with the value of the last sample in the flagged region
            for(Eigen::Index ii=0;ii<detCount;ii++)
                linOffsetOthers.block(ii,0,1,nFlags).setConstant(det(ii,0));
                //for(int l=0;l<nFlags;l++)
                    //linOffsetOthers(ii,l) = det(ii,0);
          } else if(eiFlags[j] == ei){
            //last sample in scan is flagged so offset is flat
            //with the value of the first sample in the flagged region
            for(Eigen::Index ii=0;ii<detCount;ii++)
                linOffsetOthers.block(ii,0,1,nFlags).setConstant(det(ii,nFlags-1));
                //for(int l=0;l<nFlags;l++)
                    //linOffsetOthers(ii,l) = det(ii,nFlags-1);
          } else {
            //in this case we linearly interpolate between the before
            //and after good samples
            Eigen::VectorXd xLinOffset = Eigen::VectorXd::LinSpaced(nFlags,siFlags[j],siFlags[j] + nFlags);
            Eigen::VectorXd tmpVec(nFlags);
            //for(int l=0;l<nFlags;l++) xLinOffset[l]=siFlags[j]+l;
            xx[0] = siFlags[j]-1;
            xx[1] = eiFlags[j]+1;
            for(int ii=0;ii<detCount;ii++){
              yy[0] = det(ii,0);
              yy[1] = det(ii,nFlags-1);

              mlinterp::interp(tnpts.data(), nFlags,
                               yy.data(), tmpVec.data(),
                               xx.data(), linOffset.data());

              //for(int l=0;l<nFlags;l++) linOffsetOthers(ii,l)=tmpVec[l];
              linOffsetOthers.row(ii) = tmpVec;
            }
          }

          //subtract off the linear offset from the spike free bolos

          det.block(0,0,detCount,nFlags) -= linOffsetOthers.block(0,0,detCount,nFlags);
          //for(Eigen::Index ii=0;ii<detCount;ii++)
            //for(Eigen::Index l=0;l<nFlags;l++)
              //det(ii,l) -= linOffsetOthers(ii,l);

          //scale det by responsivities and average to make sky model
          Eigen::VectorXd skyModel(nFlags);
          skyModel.setZero();

          for(Eigen::Index ii=0;ii<detCount;ii++){
              skyModel += det.row(ii)/res[ii];
          }

            //for(Eigen::Index l=0;l<nFlags;l++)

          skyModel /= detCount;


          //for(Eigen::Index l=0;l<nFlags;l++) skyModel[l] /= detCount;

          //find mean standard deviation of sky-model subtracted detectors
          //this is a different approach than that taken in IDL pipeline
          //but I think it's a good one considering the PCA to come later.
          //This is stored in the standard deviation of flag free detectors.
          Eigen::VectorXd stdDevFF(detCount);
          stdDevFF.setZero();
          double tmpMean;
          Eigen::VectorXd tmpVec(nFlags);
          for(Eigen::Index ii=0;ii<detCount;ii++){
            tmpMean=0;

            tmpVec = det.row(ii)/res(ii);//-skyModel;

            //for(Eigen::Index l=0;l<nFlags;l++) tmpVec[l] = det(ii,l)/res[ii]-skyModel[l];

            tmpMean = tmpVec.mean();

            //for(Eigen::Index l=0;l<nFlags;l++) tmpMean += tmpVec[l];
            //tmpMean /= nFlags;

            stdDevFF[ii] = (tmpVec.array()-tmpMean).cwiseAbs2().sum();

            //for(Eigen::Index l=0;l<nFlags;l++) stdDevFF[ii] += pow((tmpVec[l]-tmpMean),2);
            stdDevFF[ii] = (nFlags == 1.) ?
              stdDevFF[ii]/nFlags : stdDevFF[ii]/(nFlags-1.);
          }

          double meanStdDev;
          meanStdDev = stdDevFF.cwiseSqrt().mean();

          //for(Eigen::Index ii=0;ii<detCount;ii++) meanStdDev += sqrt(stdDevFF[ii]);
          //meanStdDev /= detCount;

          //the noiseless fake data is then the sky model plus the
          //flagged detectors linear offset
          Eigen::VectorXd fake(nFlags);
          fake = linOffset;

      //for(Eigen::Index l=0;l<nFlags;l++) {
        //fake[l] = skyModel[l]*responsivity[i] + linOffset[l];
        //fake[l] = linOffset[l];
        //}

          //add noise to the fake signal
      meanStdDev *= responsivity[i];   //put back in volts

      //replace detector values with fake signal
      //std::mt19937 generator;
      //std::normal_distribution<double> dist(0., meanStdDev);

      scans.block(siFlags[j],i,nFlags,1) = fake;

      //for(Eigen::Index l=0;l<nFlags;l++){
        //scans(siFlags[j]+l,i) = fake[l];// + dist(generator);
      //}
    }
      }
    }
  //}//index k, looping on scans
}
} //namespace timestream
