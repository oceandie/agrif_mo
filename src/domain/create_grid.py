#!/usr/bin/env python

import xarray as xr
from xnemogcm import open_domain_cfg

coord = ''

ds_cor  = xr.open_dataset(coord)

Vars = []
for grd in ['t','u','v','f']:
    Vars.append('glam'+grd)
    Vars.append('gphi'+grd) 
    Vars.append('e1'+grd)
    Vars.append('e2'+grd) 

ds_cor = ds_cor[Vars]
outdir = '/data/users/dbruciaf/AGRIF-NAtl/gs/orca08/'
outfile = '1_coordinates.nc'
ds_cor.to_netcdf(outdir+outfile)

