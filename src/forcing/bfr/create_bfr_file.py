#!/usr/bin/env python

import xarray as xr

mesh = '/data/users/dbruciaf/AGRIF-NAtl/gs/orca08/1_mesh_mask.nc'

ds_msh = xr.open_dataset(mesh)
ds_bfr = ds_msh[['nav_lon','nav_lat','time_counter','glamt']]
ds_bfr = ds_bfr.rename({'glamt':'bfr_coef'})
ds_bfr.bfr_coef[:,:,:] = 0. 

outdir = '/data/users/dbruciaf/AGRIF-NAtl/gs/orca08/bfr/'
outfile = 'bfr_coef_gs-orca08.nc'
ds_bfr.to_netcdf(outdir+outfile)

