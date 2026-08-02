#include <citlali/core/pipeline/telescope_header_snapshot.h>

namespace {

[[maybe_unused]] void telescope_header_snapshot_header_compiles_in_isolation() {
    citlali::pipeline::sci_align::TelescopeHeaderSnapshot snapshot;
    (void)snapshot;
}

}  // namespace
