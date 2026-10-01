// Exercise the existing LMTOOF reader and focus convention; no fit or model run.
#include "DataGrid.h"
#include "Subref.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
int main(int argc,char **argv) {
  if(argc!=4)return 2;
  DataGrid g(argv[1],std::atoi(argv[2])/2-1,std::atoi(argv[2])/2-1);
  FILE *f=std::fopen(argv[3],"wb");if(!f)return 3;
  for(int y=0;y<g.N;++y)for(int x=0;x<g.N;++x) {
    double z=g.power_pattern[x*g.N+y][0];if(std::fwrite(&z,sizeof(z),1,f)!=1)return 4;
  }
  std::fclose(f);DishParams dish(1100);Subref sub(dish);
  std::printf("CHECK %d %d %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",g.N,g.n_valid_points(),g.sky_cell,g.sky_axis[g.N0],g.sky_axis[g.N0+1],g.wavelength,g.subreflector_position[0],dish.rstop,sub.focus(0,0),sub.focus(20,0));
  return 0;
}
