#pragma once
//Eigen includes
#include <Eigen/Dense>

#include <iostream>

namespace timestream {

namespace internal {

template <typename DerivedA,typename DerivedB>
std::tuple<int, bool> findspikes(Eigen::DenseBase<DerivedA> &delta, Eigen::DenseBase<DerivedB> &flgs,
                                 const double sigma){
    int nSpikes=0;

    double deltamean=0;
    double deltastddev=0;

    //Get mean and standard deviation.  Move to function.
    deltamean = delta.mean();
    deltastddev = std::sqrt((delta.derived().array() - deltamean).square().sum()/(delta.rows()-1));

    bool newSpikes = 0;

    //Loop through delta function, find spikes, and flag.  newSpikes is for the recursive search.
    for(Eigen::Index i=1;i<delta.rows();i++){
        if(abs(delta[i]-deltamean) > sigma*deltastddev){
            flgs(i)=0;
            delta[i]=0.;
            nSpikes++;
            newSpikes=1;
        }
    }

    return std::tuple<int,bool> (nSpikes, newSpikes);
}

template <typename DerivedA>
std::tuple<Eigen::Index, Eigen::Index, Eigen::Index> windowing(Eigen::DenseBase<DerivedA> &spikeLoc,
                                                               int nSpikes, const int npts, const int samplerate){

    Eigen::Index winIndex0;
    Eigen::Index winIndex1;
    Eigen::Index winSize;

    if(nSpikes > 1){
        Eigen::VectorXi deltaSpikeLoc(nSpikes+1);
        deltaSpikeLoc[0] = spikeLoc[0]-0;  //distance to first spike
        deltaSpikeLoc[nSpikes] = npts-spikeLoc[nSpikes-1];
        deltaSpikeLoc.tail(nSpikes) = spikeLoc.tail(nSpikes) - spikeLoc.head(nSpikes);

        Eigen::Index mxWindow=deltaSpikeLoc[0];
        Eigen::Index mxWindowIndex=0;
        for(Eigen::Index i=1;i<=nSpikes;i++){
            if(deltaSpikeLoc[i] > mxWindow){
                mxWindow = deltaSpikeLoc[i];
                mxWindowIndex = i;
            }
        }
        //set the starting and ending indices for the window
        //leave a 2 second pad after the spike beginning the
        //window and a 1 second pad before the spike ending the window
        if(mxWindowIndex == 0){
            winIndex0=0;
            winIndex1=spikeLoc[1] - samplerate;
        }
        else{
            if(mxWindowIndex == nSpikes){
                winIndex0 = spikeLoc[nSpikes-1];
                winIndex1 = npts;
            } else {
                winIndex0=spikeLoc[mxWindowIndex-1] + 2*samplerate;
                winIndex1=spikeLoc[mxWindowIndex] - 1*samplerate;
            }
        }
        //Error checking
        if(winIndex0 > winIndex1 || winIndex0 < 0 || winIndex0 > npts || winIndex1 < 0 ||  winIndex1 > npts){
            std::cerr << "Either spikes everywhere or something else is terribly wrong." << std::endl;

            for(int i=0;i<nSpikes;i++) std::cerr << "spikeLoc["<<i<<"] = " << spikeLoc[i] << std::endl;

            std::cerr << "Overall there are " << nSpikes << " spikes." << std::endl;
            exit(1);
        }
    }
    else {
        //in this case there is only one spike
        if(npts-spikeLoc[0] > spikeLoc[0]){
            winIndex0 = spikeLoc[0] + 2*samplerate;
            winIndex1 = npts-1;
        }
        else{
            winIndex0 = 0;
            winIndex1 = spikeLoc[0] - 1*samplerate;
        }
    }
    //and after all that, let's limit the maximum window size to 10s
    if((winIndex1-winIndex0-1)/samplerate > 10.){
        winIndex1 = winIndex0+10*samplerate+1;
    }
    winSize = winIndex1-winIndex0-1;

    return std::tuple<Eigen::Index, Eigen::Index, Eigen::Index> (winIndex0, winIndex1, winSize);
}


template <typename DerivedA, typename DerivedB>
void smooth(Eigen::DenseBase<DerivedA> &inArr, Eigen::DenseBase<DerivedB> &outArr, const int winSize, int w)
{
    //Ensure w is even
    if(w % 2 == 0) w++;

    //first deal with the end cases where the output is the input
    outArr.head((w-1)/2) = inArr.head((w-1)/2);
    outArr.segment(winSize-(w+1)/2.+1,(w+1)/2.-1) = inArr.segment(winSize-(w+1)/2.+1,(w+1)/2.-1);

    double winv = 1./w;
    int wm1d2 = (w-1)/2.;
    int wp1d2 = (w+1)/2.;
    for(Eigen::Index i=wm1d2;i<=winSize-wp1d2;i++){
        outArr[i]=winv*inArr.segment(i-wm1d2,w).sum();
    }
}

} //namespace internal


template <typename DerivedA, typename DerivedB>
void despike(Eigen::DenseBase<DerivedA> const &scans, Eigen::DenseBase<DerivedB> const &flags, const double sigma, const int samplerate, const int despikewindow,
             const double timeconstant, bool isLowpassed){

    auto& s = const_cast<Eigen::DenseBase<DerivedA>&>(scans).derived();
    auto& f = const_cast<Eigen::DenseBase<DerivedB>&>(flags).derived();

    Eigen::Index npts = s.rows();
    Eigen::Index ndetectors = s.cols();

    //Loop through detectors
    for (Eigen::Index det=0; det<ndetectors; det++) {
        //Map the detector column of the scans and flag matrices to vectors for simplicity
        auto scns = s.col(det);
        auto flgs = f.col(det);

        //Generate array of differences from scans
        Eigen::Index nDelta = npts-1;
        Eigen::VectorXd delta(nDelta);

        delta = scns.tail(nDelta) - scns.head(nDelta);

        //Run the spike finder function
        auto [nSpikes,newSpikes] = internal::findspikes(delta, flgs, sigma);

        //Search for spikes recursively due to large spikes skewing the mean and stddev
        int newfound=1;
        while(newfound == 1){
            auto [nSpikes_new,newSpikes] = internal::findspikes(delta, flgs, sigma);
            nSpikes += nSpikes_new;
            if(!newSpikes) newfound=0;
        }


        //Check if more than 100 spikes for the detector
        if(nSpikes > 100){
                std::cerr << "More than 100 spikes found for detector: " << std::to_string(det) << std::endl;
                //std::cerr << "Setting this detector's goodflag to 0" << std::endl;
                //goodFlag=0;
          }

          //if there are multiple spikes in a window of 10 samples then
          //call it a single spike at the location of the middle of the window

          //Need to double check edges!!!!
          Eigen::Index c;
          for(Eigen::Index i=0;i<npts;i++){
                if(flgs[i] == 0){
                    c=0;
                    if (npts - i + 1 > 10){
                        c = (flgs.block(i+1,0,10,1).array() == 0).count();
                    }
                    else {
                        c = (flgs.block(i+1,0,npts-i,1).array() == 0).count();
                    }
                    if(c>0){
                        //reduce the number nSpikes c times
                        nSpikes = nSpikes - c;

                        if (npts - i + 1 > 10){
                            flgs.block(i+1,0,10,1).setOnes();
                        }
                        else {
                            flgs.block(i+1,0,npts-i,1).setOnes();
                        }

                        flgs[i+5] = 0;
                    }
                    i = i + 9;
                }
            }

            //Flag samples around spikes
            if(nSpikes > 0){
                //flag the samples around the spikes assuming a fixed time constant for the detectors
                //recount the spikes to avoid pathologies
                nSpikes = (flgs.array() == 0).count();

                //gMake vectors of the spike locations and det values
                Eigen::VectorXi spikeLoc(nSpikes);
                Eigen::VectorXd spikeVals(nSpikes);
                int count=0;

                for(Eigen::Index i=0;i<npts-1;i++){
                    if(flgs[i] == 0){
                        spikeLoc[count]=i+1;
                        spikeVals[count]=scns[i+1]-scns[i];
                        count++;
                    }
                }

                //Find the largest spike-free window.  First deal with the harder case of multiple spikes (nSpikes+1 possible windows)
                auto [winIndex0,winIndex1,winSize] = internal::windowing(spikeLoc,nSpikes,npts,samplerate);

                //make a sub-array with values from largest spike-free window
                Eigen::VectorXd subVals(winSize);
                subVals.head(winIndex1 - 1 - winIndex0) = scns.segment(winIndex0,winIndex1 - 1 - winIndex0);

                //make a boxcar smoothed copy of the subarray
                Eigen::VectorXd smoothedSubVals(winSize);
                internal::smooth(subVals,smoothedSubVals,winSize,10);

                //here is the estimate of the stardard deviation
                double sigest;
                subVals.head(winSize) =  subVals.head(winSize) - smoothedSubVals.head(winSize);

                sigest = std::sqrt((subVals.array() - subVals.mean()).square().sum()/(subVals.size()-1));
                if(sigest < 1.e-8) sigest=1.e-8;

                //calculate for each spike the time it takes to decay to sigest
                Eigen::VectorXd decayLength(nSpikes);
                decayLength = -samplerate*timeconstant*log(abs(sigest/spikeVals.array()));
                for(Eigen::Index i=0;i<nSpikes;i++){
                    if(decayLength[i] < 6) decayLength[i]=6;
                    if(decayLength[i] > samplerate*10.){
                        std::cerr << "Decay length is too long.";
                        exit(1);
                    }
                }

                //now flag samples, 1 decayLength before and 2 decayLengths after
                //if not lowpassed, otherwise extend window by length of lowpass
                //filter/2 on either side of spike
                for(Eigen::Index i=0;i<nSpikes;i++){
                    if (isLowpassed){
                        /*if (npts - i > 15*decayLength[i]){
                            flgs.block(spikeLoc[i]-15*decayLength[i],0,2.*15*decayLength[i],1).setZero();
                        }
                        else {
                            flgs.block(spikeLoc[i]-15*decayLength[i],0,15*decayLength[i]+npts-i,1).setZero();
                        }*/
                    }

                    else {
                        if (npts - i > (despikewindow-1)/2){
                            flgs.block(spikeLoc[i]-(despikewindow-1)/2,0,2.*(despikewindow-1)/2,1).setZero();
                        }
                        else {
                            flgs.block(spikeLoc[i]-(despikewindow-1)/2,0,(despikewindow-1)/2+npts-i,1).setZero();
                        }
                    }
                }

            } //if nSpikes gt 0 then flag samples around spikes

            //flags.col(det) = flgs;
        }
}
} //namespace
