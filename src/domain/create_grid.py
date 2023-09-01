#!/usr/bin/env python

import xarray as xr
from xnemogcm import open_domain_cfg

# The "AGRIF_FixedGrids.in" file is where the grid hierarchy is defined:
#
# 1) The first line indicates the number of zooms
# 2) The second line contains the start and end indices of the child grid in both directions 
#    on the parent grid, followed by the space (rx and ry) and time (rt) refinement factors:  
# 
#                   imin imax jmin jmax rx ry rt
#
#    The locations of the edges of the zoom, as defined by imin, imax, jmin and jmax are set 
#    excluding the parent ghost cells. By default (set in par_oce.F90), AGRIF grids have 
#                 nbghost_n = nbghost_s = nbghost_w = nbghost_e = 4
#    This number comes from the maximum order of the spatial schemes in the code. One of these 
#    ghost points is masked as required in NEMO.
# 3) The last line is the number of child grids nested in the refined region 
#
# The nested grid size can be computed as
#
#               Ni0glo = (imax-imin)*rx + nbghost_w + nbghost_e
#
#               Nj0glo = (jmax-jmin)*ry + nbghost_s + nbghost_n 
#
#

coord = '/data/users/dbruciaf/NATL/orca025/domain/r4.2_halo/ORCA025_coordinates_isf_filled_scalefactorfix_r42.nc'

nbghost = 4

# indexes in python convention
imin = 819
imax = 958
jmin = 813
jmax = 868 

IMIN = imin - nbghost
IMAX = imax + nbghost
JMIN = jmin - nbghost
JMAX = jmax + nbghost 

ds_cor  = xr.open_dataset(coord)

# Extracting only the part of the domain we need
ds_cor = ds_cor.isel(x=slice(IMIN,IMAX), y=slice(JMIN,JMAX))

Vars = []
for grd in ['t','u','v','f']:
    Vars.append('glam'+grd)
    Vars.append('gphi'+grd) 
    Vars.append('e1'+grd)
    Vars.append('e2'+grd) 

ds_cor = ds_cor[Vars]
outdir = '/data/users/dbruciaf/NATL/orca025/domain/r4.2_halo/'
outfile = '1_coordinates_'+str(IMIN)+'_'+str(IMAX)+'_'+str(JMIN)+'_'+str(JMAX)+'.nc'
ds_cor.to_netcdf(outdir+outfile)

