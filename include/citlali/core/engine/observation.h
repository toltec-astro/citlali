#pragma once

#include <string>
#include <Eigen/Core>

#include <tula/eigen.h>
#include <tula/logging.h>

#include <citlali/core/utils/constants.h>
#include <tula/algorithm/mlinterp/mlinterp.hpp>

class Observation {
public:
    using scanindicies_t = Eigen::MatrixXI;
    scanindicies_t scanindices;
    double time_chunk;
    int obsnum;

    template <typename tel_meta_data_t, typename C>
    void fix_tel_meta(tel_meta_data_t &, C &);

    template <typename tel_meta_data_t, typename C>
    void get_phys_pointing(tel_meta_data_t &, C &, std::string);

    template <typename tel_meta_data_t, typename C>
    void get_phys_icrs(tel_meta_data_t &, C &);

    template <typename tel_meta_data_t, typename C>
    void get_phys_altaz(tel_meta_data_t &, C &);

    template <typename tel_meta_data_t, typename C>
    void get_scanindices(tel_meta_data_t &, C &, std::string, const double, const double,
                         const int);
private:
    /* old code.  may not apply to TolTEC*/
    template<typename Derived>
    void remove_dropouts(Eigen::DenseBase<Derived> &);

    template<typename Derived>
    void remove_nan(Eigen::DenseBase<Derived> &);

    template <typename Derived>
    void fix_out_of_bounds(Eigen::DenseBase<Derived> &, double, double);

    template <typename Derived>
    void fix_rollover(Eigen::DenseBase<Derived> &, double, double, double);

    template <typename tel_meta_data_t>
    void find_unique_tel_utc(tel_meta_data_t &);

    //template <typename tel_meta_data_t>
    //void align_with_dets(tel_meta_data_t &);

};

template<typename Derived>
void Observation::remove_dropouts(Eigen::DenseBase<Derived> &data) {
    for (Eigen::Index i=1; i<data.size() - 1; i++) {
        if (data(i) <= 1e-9) {
            data(i) = (data(i-1) + data(i+1))/2.;
        }
    }

    if (data(0) != data(0)) {
        SPDLOG_WARN("First data point is nan.  Replaced with adjacent");
        data(0) = data(1);
    }

    if (data(data.size()-1) != data(data.size()-1)) {
        SPDLOG_WARN("Last data point is nan.  Replaced with adjacent");
        data(data.size()-1) = data(data.size()-2);
    }
}

template<typename Derived>
void Observation::remove_nan(Eigen::DenseBase<Derived> &data) {
    for (Eigen::Index i=1; i<data.size() - 1; i++) {
        if (data(i) != data(i)) {
            data(i) = (data(i-1) + data(i+1))/2.;
        }
    }

    if (data(0) != data(0)) {
        SPDLOG_WARN("First data point is nan.  Replaced with adjacent");
        data(0) = data(1);
    }

    if (data(data.size()-1) != data(data.size()-1)) {
        SPDLOG_WARN("Last data point is nan.  Replaced with adjacent");
        data(data.size()-1) = data(data.size()-2);
    }
}


template <typename Derived>
void Observation::fix_out_of_bounds(Eigen::DenseBase<Derived> &data, double low, double high) {

    for (Eigen::Index i=1; i<data.size()-1; i++) {
        if (data(i) < low || data(i) > high) {
            data(i) = (data(i-1) + data(i+1))/2.;
        }
  }

  if(data(0) < low || data(0) > high) {
      SPDLOG_INFO("First point is out of bounds. Replacing with adjacent.");
      data(0) = data(1);
  }

  if(data(data.size() - 1) < low || data(data.size() - 1) > high) {
      SPDLOG_INFO("Last point is out of bounds. Replacing with adjacent.");
      data(data.size()-1) = data(data.size()-2);
  }
}

template <typename Derived>
void Observation::fix_rollover(Eigen::DenseBase<Derived> &data,
                               double low, double high, double ulim) {

    if (data.maxCoeff() > high && data.minCoeff() < low) {
        for (Eigen::Index i=0; i<data.size(); i++) {
            if (data(i) < low) {
                data(i) += ulim;
            }
        }
    }
}

template <typename tel_meta_data_t>
void find_unique_tel_utc(tel_meta_data_t &tel_meta_data, const double glitch_limit) {

    Eigen::Index npts = tel_meta_data["TelUtc"].size();

    // check if TelUTC is constant or increasing
    for (Eigen::Index i=1; i<npts; i++) {
        if (tel_meta_data["TelUtc"](i) < tel_meta_data["TelUtc"](i-1)) {
            // find the size of the glitch.  If its smaller than the limit, it
            // is ignored
            Eigen::Index bad = 1;
            Eigen::Index nglitch = 0;
            for (Eigen::Index j=0; j < glitch_limit; j++) {
                if (tel_meta_data["TelUtc"](i+j) > tel_meta_data["TelUtc"](i-1)) {
                    bad = 0;
                    nglitch = j + 1;
                    break;
                }
            }
            if (bad) {
                SPDLOG_WARN("There may be large glitches in TelUtc.");
            }
            else {
                SPDLOG_WARN("UTC glitch is {} samples long.  Consider throwing out this data file.", nglitch);
            }
        }
    }

    // count up the unique values
    int count = 1;
    for (Eigen::Index i=1; i<npts; i++){
        if (tel_meta_data["TelUtc"](i) > tel_meta_data["TelUtc"](i-1)) {
            count++;
        }
    }

    tel_meta_data["TelUtcUnique"].resize(count);
    tel_meta_data["locUnique"].resize(count);
    tel_meta_data["TelUtcUnique"](0) = tel_meta_data["TelUtc"](0);
    tel_meta_data["locUnique"](0) = 0;

    int counter = 1;
    for (Eigen::Index i=1; i<npts; i++) {
        if (tel_meta_data["TelUtc"](i) > tel_meta_data["TelUtc"](i-1)) {
            tel_meta_data["telUtcUnique"](counter) = tel_meta_data["TelUtc"](i);
            tel_meta_data["locUnique"](counter) = i;
            counter++;
        }
    }
}

/*template <typename tel_meta_data_t>
void align_with_dets(tel_meta_data_t &tel_meta_data, const double time_offset) {

    Eigen::Index nunique = tel_meta_data["locUnique"].size();

    Eigen::VectorXd xx =  tel_meta_data["TelUtcUnique"].array() + time_offset/3600.;
    Eigen::VectorXd yy(nunique);

    Eigen::Matrix<Eigen::Index,1,1> nu;
    nu << nunique;

    for (const auto &it : tel_meta_data) {
        if (it.first!="Hold" && it.first!="AztecUtc" && it.first!="locUnique" && it.first!="TelUtcUnique") {
            for (Eigen::Index i=0; i<nunique; i++) {
                yy(i) = tel_meta_data[it.first](tel_meta_data["locUnique"].template cast<Eigen::Index>()(i));
            }

            Eigen::Index npts = tel_meta_data[it.first].size();
            mlinterp::interp(nu.data(), npts,
                             yy.data(), tel_meta_data[it.first].data(),
                             xx.data(), tel_meta_data["AztecUtc"].data());
        }
    }
}
*/

template <typename tel_meta_data_t, typename C>
void Observation::fix_tel_meta(tel_meta_data_t &tel_meta_data, C &center) {

    for (const auto &it : tel_meta_data) {
        if ((it.first != "Hold") || (it.first != " TelTime")) {
            remove_dropouts(tel_meta_data[it.first]);
            remove_nans(tel_meta_data[it.first]);
            fix_out_of_bounds(tel_meta_data[it.first], 1.e-9, 5.*pi);
            if (it.first != "ParAng") {
                fix_rollover(tel_meta_data[it.first], pi, 1.99*pi, 2.0*pi);
            }
        }

        else if (it.first == "TelUtc") {// || it.first == "AztecUtc") {
            remove_dropouts(tel_meta_data[it.first]);
            remove_nan(tel_meta_data[it.first]);
            fix_rollover(tel_meta_data[it.first], 10., 23., 24.);
            fix_out_of_bounds(tel_meta_data[it.first], 1.e-9, 30);
        }
    }

    // LMT's UTC is in radians so convert to hours
    tel_meta_data["TelUtc"].array() *= 24./2./pi;

    // find unique TelUTC values
    find_unique_tel_utc(tel_meta_data, 32);

    // align tel meta data with detector time
    // align_with_dets(tel_meta_data, 0);
}

template <typename tel_meta_data_t, typename C>
void Observation::get_phys_pointing(tel_meta_data_t &tel_meta_data, C &center, std::string map_type) {

    // get icrs physical pointing
    if (std::strcmp("icrs", map_type.c_str()) == 0) {
        get_phys_icrs(tel_meta_data, center);
    }

    // get altaz physical pointing
    else if (std::strcmp("altaz", map_type.c_str()) == 0) {
        SPDLOG_INFO("getting altaz map");
        get_phys_altaz(tel_meta_data, center);
    }
}

template <typename tel_meta_data_t, typename C>
void Observation::get_phys_icrs(tel_meta_data_t &tel_meta_data, C &center) {

    // copy of absolute ra
    Eigen::VectorXd temp_ra = tel_meta_data["TelRa"];
    //(temp_ra.array() > pi).select(tel_meta_data["TelRa"].array() - 2.0*pi, tel_meta_data["TelRa"]);


   // temp ra must range from -pi to pi
    for(Eigen::Index i=0;i<temp_ra.size();i++)
      temp_ra(i) = (tel_meta_data["TelRa"](i) > pi) ? tel_meta_data["TelRa"](i)-(2*pi) : tel_meta_data["TelRa"](i);


    // copy of absolute dec
    Eigen::VectorXd temp_dec = tel_meta_data["TelDec"];

    // copy of center ra
    double temp_center_ra = center["Ra"](0);
    temp_center_ra = (temp_center_ra > pi) ? temp_center_ra-(2.0*pi) : temp_center_ra;

    // copy of center dec
    double tempCenterDec = center["Dec"](0);

    auto cosc = sin(tempCenterDec)*sin(tel_meta_data["TelDec"].array()) +
            cos(tempCenterDec)*cos(tel_meta_data["TelDec"].array())*cos(temp_ra.array()-temp_center_ra);

    tel_meta_data["TelLatPhys"].resize(temp_ra.size());
    tel_meta_data["TelLonPhys"].resize(temp_ra.size());

    // get physical Ra/Dec
    for (Eigen::Index i = 0; i < temp_ra.size(); i++) {
        if (cosc(i) == 0.) {
            tel_meta_data["TelLatPhys"](i) = 0.;
            tel_meta_data["TelLonPhys"](i) = 0.;
        }
        else {
            tel_meta_data["TelLatPhys"](i) = (cos(tempCenterDec)*sin(tel_meta_data["TelDec"](i)) -
                    sin(tempCenterDec)*cos(tel_meta_data["TelDec"](i))*cos(temp_ra(i)-temp_center_ra))/cosc(i);

            tel_meta_data["TelLonPhys"](i) = cos(tel_meta_data["TelDec"](i))*sin(temp_ra(i)-temp_center_ra)/cosc(i);
        }
    }
}

template <typename tel_meta_data_t, typename C>
void Observation::get_phys_altaz(tel_meta_data_t &tel_meta_data, C &center) {
    for (Eigen::Index i = 0; i < tel_meta_data["TelAzAct"].size(); i++) {
        if ((tel_meta_data["TelAzAct"](i) - tel_meta_data["SourceAz"](i)) > 0.9*2.0*pi) {
            tel_meta_data["TelAzAct"](i) = tel_meta_data["TelAzAct"](i) - 2.0*pi;
        }
    }

    // calculate physical altaz
    tel_meta_data["TelLatPhys"] = (tel_meta_data["TelElAct"] - tel_meta_data["SourceEl"]) - tel_meta_data["TelElCor"];
    tel_meta_data["TelLonPhys"] = cos(tel_meta_data["TelElDes"].array())*(tel_meta_data["TelAzAct"].array() - tel_meta_data["SourceAz"].array())
            - tel_meta_data["TelAzCor"].array();
}

template <typename tel_meta_data_t, typename C>
void Observation::get_scanindices(tel_meta_data_t &tel_meta_data, C &center, std::string ObsPgm,
                                  const double fsmp, const double time_chunk, const int filter_nterms) {

    Eigen::Index nscans = 0;

    SPDLOG_INFO("OBSPGM {}", ObsPgm);
    SPDLOG_INFO("OBSPGM bool{}", std::strcmp("Map", ObsPgm.c_str()) == 0);

    // get scan indices for Raster pattern
    if (1){//std::strcmp("Map", ObsPgm.c_str()) == 0) {
        SPDLOG_INFO("Calculating scans for Raster mode");

        // cast Hold signal to bool
        Eigen::Matrix<bool,Eigen::Dynamic,1> turning = tel_meta_data["Hold"].template cast<bool>();

        //for(int i=0;i<turning.size();i++) turning[i] = (tel_meta_data["Hold"].template cast<int>()(i)&8);

        SPDLOG_INFO("how many zeros {}",(turning.array()==0).count());

        SPDLOG_INFO("turning {}", turning);

        // this doesn't work for some reason
        //nscans = ((turning.tail(turning.size() - 1) - turning.head(turning.size() - 1)).array() == 1).count();

        for (Eigen::Index i=1; i<turning.size(); i++) {
            if (turning(i) - turning(i-1) == 1) {
                nscans++;
            }
        }

        SPDLOG_INFO("nscans {}", nscans);


        if (turning(turning.size()-1) == 0){
            nscans++;
        }

        SPDLOG_INFO("nscans {}", nscans);

        scanindices.resize(4,nscans);

        int counter = -1;
        if (!turning(0)) {
          scanindices(0,0) = 1;
          counter++;
        }

        for (Eigen::Index i=1; i<turning.size(); i++) {
            if (turning(i) - turning(i-1) < 0) {
                counter++;
                scanindices(0,counter) = i + 1;
            }

            else if (turning(i) - turning(i-1) > 0) {
                scanindices(1,counter) = i - 1;
            }
        }
              scanindices(1,nscans - 1) = turning.size() - 1;
    }

    // get scanindices for Lissajous/Rastajous pattern
    else if (std::strcmp("Lissajous", ObsPgm.c_str()) == 0) {
        SPDLOG_INFO("Calculating scans for Lissajous/Rastajous mode");

        // index of first scan
        Eigen::Index first_scan_i = 0;
        // index of last scan
        Eigen::Index last_scan_i = tel_meta_data["Hold"].size() - 1;
        // period (time_chunk/fsmp in seconds/Hz)
        Eigen::Index period_i = floor(time_chunk*fsmp);

        double period = floor(time_chunk*fsmp);
        // calculate number of scans
        nscans = floor((last_scan_i - first_scan_i + 1)*1./period);

        // assign scans to scanindices matrix
        scanindices.resize(4,nscans);
        scanindices.row(0) =
                Eigen::Vector<Eigen::Index,Eigen::Dynamic>::LinSpaced(nscans,0,nscans-1).array()*period_i + first_scan_i;
        scanindices.row(1) = scanindices.row(0).array() + period_i - 1;

    }

    Eigen::Matrix<Eigen::Index,Eigen::Dynamic, Eigen::Dynamic> scanindices_temp(4,nscans); 
    scanindices_temp = scanindices; 

    // do a final check of scan length.  If a scan is
    // less than 2s of data then delete it
    int n_bad_scans = 0;
    int sum = 0;
    
    Eigen::Matrix<bool, Eigen::Dynamic, 1> is_bad_scan(nscans);
    for (Eigen::Index i=0; i<nscans; i++) {
        sum=0;
        for (Eigen::Index j=scanindices_temp(0,i); j<(scanindices_temp(1,i)+1); j++) {
            sum += 1;
        }
        if(sum < 2.*fsmp) {
            n_bad_scans++;
            is_bad_scan(i)=1;
        } 
        else {
            is_bad_scan(i) = 0;
        }
    }

    if (n_bad_scans > 0) {
      SPDLOG_INFO("n_bad_scans {} scans with duration less than 2 seconds detected", n_bad_scans);
    }

    int c = 0;
    scanindices.resize(4,nscans-n_bad_scans);
    for (Eigen::Index i=0; i<nscans; i++) {
        if (!is_bad_scan(i)) {
            scanindices(0,c) = scanindices_temp(0,i);
            scanindices(1,c) = scanindices_temp(1,i);
            c++;
        }
    }
    nscans = nscans - n_bad_scans;

    // set up the 3rd and 4th scanindices rows so that we don't lose data during lowpassing
    // filter_nterms is zero if lowpassing is not enabled
    scanindices.row(2) = scanindices.row(0).array() - filter_nterms;
    scanindices.row(3) = scanindices.row(1).array() + filter_nterms;

    // set first and last outer scan positions to the same as inner scans since there's no
    // data on either side
    scanindices(2,0) = scanindices(0,0) + filter_nterms;
    scanindices(3,nscans-1) = scanindices(1,nscans-1) - filter_nterms;
}
