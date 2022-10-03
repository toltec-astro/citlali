#pragma once

namespace timestream {

class Despiker {
public:
    double min_spike_sigma, time_constant_sec, window_size;

};

} // namespace timestream
