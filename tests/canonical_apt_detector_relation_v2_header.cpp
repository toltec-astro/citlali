#include <citlali/core/pipeline/canonical_apt_detector_relation_v2.h>

#include <type_traits>

void canonical_apt_detector_relation_v2_header_compiles_in_isolation() {
    static_assert(std::is_move_constructible_v<
                  citlali::pipeline::CanonicalAptDetectorRelationV2>);
}
