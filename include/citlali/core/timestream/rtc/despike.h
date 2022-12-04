#pragma once

namespace timestream {

class Despiker {
public:
    double min_spike_sigma, time_constant_sec, window_size;
    void despiker();

private:
    void spike_finder();
    void make_window();

};

} // namespace timestream
