#include <citlali/core/mapmaking/oof_transfer_reference.h>

#include <type_traits>

static_assert(std::is_enum_v<mapmaking::DiscreteGFieldGroup>);
static_assert(mapmaking::discrete_g_field_groups.size() == 14);
