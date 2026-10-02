// Actual LMTOOF reader/window, no fitting or model evaluation.
#include "DataGrid.h"
#include <cstdio>
int main(int argc,char **argv) {
 if(argc!=3)return 2;
 DataGrid g(argv[1],12,12);
 FILE *f=std::fopen(argv[2],"wb");if(!f)return 3;
 for(int y=0;y<g.N;++y)for(int x=0;x<g.N;++x) {
   double z=g.power_pattern[x*g.N+y][0];if(std::fwrite(&z,sizeof(z),1,f)!=1)return 4;
 }
 std::fclose(f);
 std::printf("CHECK %d %d %.17g %.17g %.17g\n",g.N,g.n_valid_points(),g.sky_cell,g.wavelength,g.subreflector_position[0]);
 return 0;
}
