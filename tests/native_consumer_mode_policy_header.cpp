#include <citlali/core/pipeline/native_consumer_mode_policy.h>

static_assert(!citlali::pipeline::native_consumer_lineage_required(
    citlali::pipeline::NativeConsumerRoute::beammap_raw_apt_producer));
