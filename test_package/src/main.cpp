#include <citlali/core/engine/beammap.h>
#include <citlali/core/engine/lali.h>
#include <citlali/core/engine/pointing.h>

int main()
{
    return sizeof(Lali) > 0 && sizeof(Pointing) > 0 && sizeof(Beammap) > 0
               ? 0
               : 1;
}
