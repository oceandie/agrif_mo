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

domcfg = '/data/users/dbruciaf/AGRIF-NAtl/gs/orca04/domain_cfg_large.nc'

# Global grids will have nghost_w = 0 and nghost_s = 1 
# (the western boundary is cyclic, the southern boundary 
# over Antarctica is closed).
root_nbghost_s = 1
root_nbghost_w = 0
# AGRIF grids have, by default, nghost_w = nbghost_s = 4. 
# One of these ghost points is masked as required in NEMO.
zoom_nbghost_s = 4
zoom_nbghost_w = 4
rx = 1
ry = 1

# indexes in python convention
imin =  ( 820 + root_nbghost_w) - 1
imax =  (1080 + root_nbghost_w  - 1) - 1
jmin =  ( 779 + root_nbghost_s) - 1
jmax =  ( 920 + root_nbghost_s  - 1) - 1

IMIN = imin - int(zoom_nbghost_w / rx)
IMAX = imax + int(zoom_nbghost_w / rx)
JMIN = jmin - int(zoom_nbghost_s / ry)
JMAX = jmax + int(zoom_nbghost_s / ry) 

ds_dom  = xr.open_dataset(domcfg)

# Extracting only the part of the domain we need
ds_dom = ds_dom.isel(x=slice(IMIN,IMAX+1), y=slice(JMIN,JMAX+1))

#Vars = []
#for grd in ['t','u','v','f']:
#    Vars.append('glam'+grd)
#    Vars.append('gphi'+grd) 
#    Vars.append('e1'+grd)
#    Vars.append('e2'+grd) 

#ds_cor = ds_cor[Vars]
outdir = './'
outfile = '1_domain_cfg_'+str(IMIN)+'_'+str(IMAX)+'_'+str(JMIN)+'_'+str(JMAX)+'.nc'
ds_dom.to_netcdf(outdir+outfile)

