#!/usr/bin/env python

import xarray as xr

#mesh = '/data/users/dbruciaf/AGRIF-NAtl/gosi10-gs_r12/1_mesh_mask-tmask_as_gosi10-025.nc'
mesh = '/data/users/dbruciaf/AGRIF-NAtl/gosi10-ls_r20/1_mesh_mask-tmask_as_gosi10-025.nc'
#mesh = '/data/users/dbruciaf/AGRIF-NAtl/gosi10na+/3_gb_r20/3_domain_cfg_zps.nc'

ds_msh = xr.open_dataset(mesh)
ds_bfr = ds_msh[['nav_lon','nav_lat','time_counter','glamt']]
ds_bfr = ds_bfr.rename({'glamt':'bfr_cdmin'})
ds_bfr.bfr_cdmin[:,:,:] = 0.001

#outdir = '/data/users/dbruciaf/AGRIF-NAtl/gosi10-gs_r12/bfr/'
outdir = '/data/users/dbruciaf/AGRIF-NAtl/gosi10-ls_r20/bfr/'
#outdir = '/data/users/dbruciaf/AGRIF-NAtl/gosi10na+/3_gb_r20/bfr/'

#outfile = 'bfr_cdmin_2d_001-0025_r42-gs_r12.nc'
outfile = 'bfr_cdmin_2d_001-0025_r42-ls_r20.nc'
#outfile = 'bfr_cdmin_2d_001-0025_r42-gb_r20.nc'

ds_bfr.to_netcdf(outdir+outfile)

