#!/usr/bin/env python

import xarray as xr

#mesh = '/data/users/dbruciaf/AGRIF-NAtl/gs/orca04/zps/1_domain_cfg_zps.nc'
#mesh = '/data/users/dbruciaf/AGRIF-NAtl/ls/orca20/1_domain_cfg.nc'
#mesh = '/data/users/dbruciaf/AGRIF-NAtl/oliver_gs/orca04/1_domain_cfg.nc'
#mesh = '/data/users/dbruciaf/AGRIF-NAtl/oliver_gs/orca08/1_domain_cfg.nc'
#mesh = '/data/users/dbruciaf/AGRIF-NAtl/oliver_gs/orca12/1_domain_cfg.nc'
#mesh = '/data/users/dbruciaf/AGRIF-NAtl/movf/orca20/zps/1_domain_cfg.nc'
#mesh = '/data/users/dbruciaf/AGRIF-NAtl/movf/orca12/zps/1_domain_cfg.nc'
#mesh = '/data/users/dbruciaf/AGRIF-NAtl/gosi10na+/1_gs_r12/1_domain_cfg.nc'
#mesh = '/data/users/dbruciaf/AGRIF-NAtl/gosi10na+/2_ls_r20/2_domain_cfg.nc'
mesh = '/data/users/dbruciaf/AGRIF-NAtl/gosi10na+/3_gb_r20/3_domain_cfg.nc'

ds_msh = xr.open_dataset(mesh)
ds_bfr = ds_msh[['nav_lon','nav_lat','time_counter','glamt']]
ds_bfr = ds_bfr.rename({'glamt':'bfr_coef'})
ds_bfr.bfr_coef[:,:,:] = 0. 

#outdir = '/data/users/dbruciaf/AGRIF-NAtl/ls/orca20/bfr/'
#outdir = '/data/users/dbruciaf/AGRIF-NAtl/oliver_gs/orca04/bfr/'
#outdir = '/data/users/dbruciaf/AGRIF-NAtl/oliver_gs/orca08/bfr/'
#outdir = '/data/users/dbruciaf/AGRIF-NAtl/oliver_gs/orca12/bfr/'
#outdir = '/data/users/dbruciaf/AGRIF-NAtl/movf/orca20/bfr/'
#outdir = '/data/users/dbruciaf/AGRIF-NAtl/movf/orca12/bfr/'
#outdir = '/data/users/dbruciaf/AGRIF-NAtl/gosi10na+/1_gs_r12/bfr/'
#outdir = '/data/users/dbruciaf/AGRIF-NAtl/gosi10na+/2_ls_r20/bfr/'
outdir = '/data/users/dbruciaf/AGRIF-NAtl/gosi10na+/3_gb_r20/bfr/'

#outfile = 'bfr_coef_ls-orca20.nc'
#outfile = 'bfr_coef_movf-orca12.nc'
#outfile = 'bfr_coef_movf-orca20.nc'
#outfile = 'bfr_coef_gs-orca08.nc'
#outfile = 'bfr_coef_gs_r12.nc'
#outfile = 'bfr_coef_ls_r20.nc'
outfile = 'bfr_coef_gb_r20.nc'

ds_bfr.to_netcdf(outdir+outfile)

