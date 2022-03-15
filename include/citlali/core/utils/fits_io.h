#pragma once

#include <Eigen/Core>

#include <CCfits/CCfits>
#include <chrono>

#include <tula/logging.h>
#include <citlali/core/utils/constants.h>

enum fileType {
    read_fits = 0,
    write_fits = 1
};

enum UnitsType {
    deg = 0,
    arcsec = 1,
    pixel = 3,
};

//template <typename ext_hdu_t>
template<fileType file_type, typename ext_hdu_t>
class FitsIO {
public:

    // file path
    std::string filepath;

    // pointer to FITS file
    std::unique_ptr<CCfits::FITS> pfits;

    // vector of hdu's for easy access
    std::vector<ext_hdu_t> hdus;

    // vector of tables for easy access
    std::vector<CCfits::Table*> tables;

    // number of rows for each table
    std::vector<Eigen::Index> rows;

    // wcs keys
    std::vector<std::string> wcs_keys {
        "CTYPE1","CTYPE2",
        "CRVAL1","CRVAL2",
        "CRPIX1","CRPIX2",
        "CDELT1","CDELT2",
        "CD1_1","CD1_2",
        "CD2_1","CD2_2",
    };

    // constructor
    FitsIO<file_type,ext_hdu_t>(std::string _filepath) {

        filepath = _filepath;
        // read in file
        if constexpr (file_type==fileType::read_fits) {
            try {
            pfits.reset( new CCfits::FITS(filepath, CCfits::Read));
            }
            catch (CCfits::FITS::CantOpen) {
                SPDLOG_ERROR("unable to open file {}", filepath);
                std::exit(EXIT_FAILURE);
            }
        }

        // create file
        else if constexpr (file_type==fileType::write_fits) {
            try {
                pfits.reset( new CCfits::FITS(filepath + ".fits", CCfits::Write));
                // write date
                pfits->pHDU().writeDate();
                SPDLOG_INFO("created file {}", filepath + ".fits");
            }
            catch (CCfits::FITS::CantCreate) {
                SPDLOG_ERROR("unable to create file {}", filepath);
                std::exit(EXIT_FAILURE);
            }
        }
    }

    template <typename Derived>
    void add_hdu(std::string hdu_name, Eigen::DenseBase<Derived> &data) {
        // axes dimensions (note the reversed order)
        std::vector naxes{data.cols(), data.rows()};

        // add an extension hdu to vector
        hdus.push_back((pfits->addImage(hdu_name,DOUBLE_IMG,naxes)));

        // valarray to copy data into (seems to be necessary)
        std::valarray<double> tmp(data.size());

        // copy the data (flip in x direction)
        int k = 0;
        for (int i=0; i<data.rows(); i++){
            for (int j=0; j<data.cols(); j++) {
                tmp[k] = data(i, data.cols() - j - 1);
                k++;
            }
        }

        // no clue why we need this
        long fpixel = 1;

        // write to the hdu
        hdus.back()->write(fpixel, tmp.size(), tmp);
    }

    auto get_hdu(std::string hdu_name) {
        // add extension hdu to vector
        CCfits::ExtHDU& hdu = pfits->extension(hdu_name);
        std::valarray<double> contents;

        // read all user-specifed, coordinate, and checksum keys in the image
        hdu.readAllKeys();
        hdu.read(contents);

        // this doesn't print the data, just header info.
        long ax1(hdu.axis(0));
        long ax2(hdu.axis(1));

        // holds the image data
        Eigen::MatrixXd data(ax2,ax1);

        // loop through and copy into eigen matrix
        Eigen::Index k = 0;
        for (Eigen::Index i=0; i<ax2; i++) {
            for (Eigen::Index j=0; j<ax1; j++) {
                data(i,j) = contents[k];
                k++;
            }
        }

        return data;
    }

    template <UnitsType units, typename hdu_t, typename map_type_t, typename center_t>
    void add_wcs(hdu_t *hdu, map_type_t map_type, const int nrows, const int ncols,
                 const double pixel_size, center_t &source_center) {

        // get units
        double unit_scale;

        // if degrees requested
        if constexpr (units == UnitsType::deg) {
            unit_scale = DEG_TO_RAD;
            hdu->addKey("CUNIT1", "deg", "");
            hdu->addKey("CUNIT2", "deg", "");
        }

        // if arcseconds requested
        else if constexpr (units == UnitsType::arcsec) {
            unit_scale = RAD_ASEC;
            hdu->addKey("CUNIT1", "arcsec", "");
            hdu->addKey("CUNIT2", "arcsec", "");
        }

        // get reference value
        double CRVAL1, CRVAL2;

        // if icrs map, set the center to the (RA, Dec)
        if (std::strcmp("icrs", map_type.c_str()) == 0) {
            hdu->addKey("CTYPE1", "RA---TAN", "");
            hdu->addKey("CTYPE2", "DEC--TAN", "");

            CRVAL1 = source_center["Ra"](0)/unit_scale;
            CRVAL2 = source_center["Dec"](0)/unit_scale;
        }

        // else set it to (0,0) for offset maps
        else if (std::strcmp("altaz", map_type.c_str()) == 0) {
            hdu->addKey("CTYPE1", "AZOFFSET", "");
            hdu->addKey("CTYPE2", "ELOFFSET", "");

            CRVAL1 = 0.0;
            CRVAL2 = 0.0;
        }

        // add CRVAL values
        hdu->addKey("CRVAL1", CRVAL1, "");
        hdu->addKey("CRVAL2", CRVAL2, "");

        // pixel corresponding to reference value
        double ref_pix_c1 = ncols/2;
        double ref_pix_c2 = nrows/2;

        // add 0.5 for even sided maps (all maps are currently even sided)
        if ((int)ref_pix_c1 == ref_pix_c1) {
            ref_pix_c1 += 0.5;
        }

        if ((int)ref_pix_c2 == ref_pix_c2) {
            ref_pix_c2 += 0.5;
        }

        // add CRPIX values
        hdu->addKey("CRPIX1", ref_pix_c1, "");
        hdu->addKey("CRPIX2", ref_pix_c2, "");

        // add CD matrix
        hdu->addKey("CD1_1", -pixel_size/unit_scale, "");
        hdu->addKey("CD1_2", 0, "");
        hdu->addKey("CD2_1", 0, "");
        hdu->addKey("CD2_2", pixel_size/unit_scale, "");

        // add source ra
        hdu->addKey("SRC_RA_RAD", source_center["Ra"][0], "Source RA (radians)");
        // add source dec
        hdu->addKey("SRC_DEC_RAD", source_center["Dec"][0], "Source DEC (radians)");
    }

    void add_ascii_table(std::string hdu_name, unsigned long nrows, std::vector<std::string> &colnames,
                   std::vector<std::string> &colform, std::vector<std::string> &colunits) {

        tables.push_back(pfits->addTable(hdu_name,nrows,colnames,colform,colunits,CCfits::AsciiTbl));
        rows.push_back(nrows);
    }

    template<typename Derived, typename dtype>
    void write_to_table(CCfits::Table* table, Eigen::DenseBase<Derived> &data, std::string colname, unsigned long rows,
                        unsigned long start_index) {
        long ignore_val(12112);
        long null_number(-999);

        std::vector<dtype> data_vec;
        for (Eigen::Index i=0; i<data.size(); i++) {
            data_vec.push_back(data(i));
        }

        try {
            table->column(colname).addNullValue(null_number);
            table->column(colname).write(data,data.size(),start_index);
        }

        catch (CCfits::FitsException&) {
          // do something
        }

    }
};
