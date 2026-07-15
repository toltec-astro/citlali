#pragma once

#include <citlali/core/error/error.h>

#include <CCfits/CCfits>
#include <stdexcept>

enum file_type_enum {
    read_fits = 0,
    write_fits = 1
};
template<file_type_enum file_type, typename ext_hdu_t>
class fitsIO {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // filepath
    std::string filepath;

    // pointer to FITS file
    std::unique_ptr<CCfits::FITS> pfits;

    // vector of hdu's for easy access
    std::vector<ext_hdu_t> hdus;

    fitsIO() {}

    // constructor
    fitsIO(std::string _f) : filepath(_f) {
        // read in file
        if constexpr (file_type==file_type_enum::read_fits) {
            try {
                pfits.reset( new CCfits::FITS(filepath, CCfits::Read));
                logger->info("opened FITS file {}", filepath);
            }
            catch (const CCfits::FitsException &error) {
                throw citlali::error::io(
                    "unable to open FITS input file " + filepath + ": " +
                    error.message());
            }
        }

        // create file
        else if constexpr (file_type==file_type_enum::write_fits) {
            try {
                // ! is the overwrite flag
                pfits.reset( new CCfits::FITS("!" + filepath + ".fits", CCfits::Write));
                // write date
                pfits->pHDU().writeDate();
            }
            catch (const CCfits::FitsException &error) {
                throw citlali::error::output(
                    "unable to create required FITS output file " + filepath +
                    ".fits: " + error.message());
            }
        }
    }

    template <typename Derived>
    void add_hdu(std::string hdu_name, Eigen::DenseBase<Derived> &data) {
        try {
            // axes in reverse order (cols, rows, pol, freq)
            std::vector<long> naxes{data.cols(), data.rows(), 1, 1};

            // add an extension hdu to vector
            hdus.push_back((pfits->addImage(hdu_name,DOUBLE_IMG,naxes)));

            // valarray to copy data into (seems to be necessary)
            std::valarray<double> temp_data(data.size());

            // copy the data (flip in x direction)
            int k = 0;
            for (int i=0; i<data.rows(); ++i){
                for (int j=0; j<data.cols(); ++j) {
                    temp_data[k] = data(i, data.cols() - j - 1);
                    k++;
                }
            }

            // first pixel (starts at 1 I think)
            long first_pixel = 1;

            // write to the hdu
            hdus.back()->write(first_pixel, temp_data.size(), temp_data);
        } catch (const CCfits::FitsException &e) {
            throw citlali::error::output(
                "failed to add/write FITS HDU '" + hdu_name + "' in " + filepath + ": " + e.message());
        }
    }

    auto get_hdu(std::string hdu_name) {
        try {
            // get extension
            CCfits::ExtHDU& hdu = pfits->extension(hdu_name);

            // hold image data
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
            for (Eigen::Index i=0; i<ax2; ++i) {
                for (Eigen::Index j=0; j<ax1; ++j) {
                    data(i,j) = contents[k];
                    k++;
                }
            }

            // flip to match internal orientation
            data.rowwise().reverseInPlace();

            return data;

        } catch (const CCfits::FitsException &error) {
            throw citlali::error::io(
                "cannot read FITS HDU '" + hdu_name + "' in " + filepath +
                ": " + error.message());
        }
    }

    template <typename hdu_t, class wcs_t, typename epoch_t>
    void add_wcs(hdu_t *hdu, wcs_t &wcs, const epoch_t epoch) {
        try {
            // add equinox
            hdu->addKey("EQUINOX", epoch, "WCS: Equinox");

            for (Eigen::Index i=0; i<wcs.ctype.size(); ++i) {
                hdu->addKey("CTYPE"+std::to_string(i+1), wcs.ctype[i], "WCS: Projection Type " +std::to_string(i+1));
                hdu->addKey("CUNIT"+std::to_string(i+1), wcs.cunit[i], "WCS: Axis Unit " +std::to_string(i+1));
                hdu->addKey("CRVAL"+std::to_string(i+1), wcs.crval[i], "WCS: Ref Pixel Value " +std::to_string(i+1));
                hdu->addKey("CDELT"+std::to_string(i+1), wcs.cdelt[i], "WCS: Pixel Scale " +std::to_string(i+1));
                // add one to crpix due to FITS convention
                hdu->addKey("CRPIX"+std::to_string(i+1), wcs.crpix[i] + 1, "WCS: Ref Pixel " +std::to_string(i+1));
            }
        } catch (const CCfits::FitsException &e) {
            throw citlali::error::output(
                "failed to add WCS keywords in " + filepath + ": " + e.message());
        }
    }
};
