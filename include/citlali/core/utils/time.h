# pragma once

#include <time.h>

namespace citlali::utils::timing {

// get current date/time, format is YYYY-MM-DD.HH:mm:ss
static const std::string current_date_time() {
    time_t now = time(0);
    struct tm  tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}

// convert unixt time to utc, format is YYYY-MM-DD.HH:mm:ss
static const std::string unix_to_utc(double &t) {
    time_t     now = t;
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}

// convert utc to unix
template <typename DerivedA, typename DerivedB>
void utc_to_unix(Eigen::DenseBase<DerivedA> &tel_utc, Eigen::DenseBase<DerivedB> &ut_date) {
    // size of time vector
    Eigen::Index n_pts = tel_utc.size();

    time_t utc_time;

    // who would have guessed?
    auto days_in_year = 365.0;

    int year = std::floor(ut_date(0));
    int days = int(((ut_date(0)-year)*days_in_year)+1);

    auto ut_time = 180/15/pi * tel_utc.derived().array();

    Eigen::VectorXd tel_unix(n_pts);

    // loop through points
    for (Eigen::Index i=0; i<n_pts; ++i) {
        auto h =(int)ut_time(i);
        auto m = (int)((ut_time(i) - h)*60);
        auto s = (((ut_time(i) - h)*60 - m)*60);

        struct tm tm_time;
        tm_time.tm_isdst = -1;
        tm_time.tm_mon = 0;
        tm_time.tm_mday = days;
        tm_time.tm_year = year - 1900;

        tm_time.tm_sec = s;
        tm_time.tm_min = m;
        tm_time.tm_hour = h;

        // get UTC time
        utc_time = timegm(&tm_time);

        // convert UTC time to Unix timestamp
        time_t unix_time = (time_t) utc_time;
        // add Unix time to vector
        tel_unix(i) = unix_time;
    }
    // overrwite UTC time with Unix time
    tel_utc = tel_unix;
}

// mjd to unix
static long long mjd_to_unix(double jd) {
    // unix epoch in julian date
    const double UNIX_EPOCH_JD = 2440587.5;

    // difference in days and convert from modified julian to julian date
    double diff = jd - UNIX_EPOCH_JD + 2400000.5;

    // convert days to seconds
    long long unix_time = static_cast<long long>(diff * 86400.0);

    return unix_time;
}

// unix to mjd
static double unix_to_mjd(double unix_time) {
    const double seconds_per_day = 86400.0;
    const double mjd_offset = 40587.0;
    return unix_time / seconds_per_day + mjd_offset;
}
} // namespace
