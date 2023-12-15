#!/usr/bin/env python

import xarray as xr

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

#domcfg0 = '/data/users/dbruciaf/AGRIF-NAtl/gs/orca04/domain_cfg_large.nc'
#domcfg1 = '/data/users/dbruciaf/AGRIF-NAtl/gs/orca04/1_domain_cfg_large.nc'
#clomsk  = '/data/users/dbruciaf/NATL/orca025/domain/r4.2_halo/domcfg_eORCA025_v2_r42_closea.nc'
rivers  = '/data/users/dbruciaf/AGRIF-NAtl/gs/orca04/runoff/eORCA025_runoff_GO6_icb.nc'
M2tmx   = '/data/users/dbruciaf/AGRIF-NAtl/gs/orca04/tmx/M2rowdrg_R025_modif_nonpositive_r42.nc'
K1tmx   = '/data/users/dbruciaf/AGRIF-NAtl/gs/orca04/tmx/K1rowdrg_R025_modif_nonpositive_r42.nc'

imin =  820
imax = 1080
jmin =  779
jmax =  920

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
imin =  (imin + root_nbghost_w) - 1
imax =  (imax + root_nbghost_w  - 1) - 1
jmin =  (jmin + root_nbghost_s) - 1
jmax =  (jmax + root_nbghost_s  - 1) - 1

IMIN = imin - int(zoom_nbghost_w / rx)
IMAX = imax + int(zoom_nbghost_w / rx)
JMIN = jmin - int(zoom_nbghost_s / ry)
JMAX = jmax + int(zoom_nbghost_s / ry) 

#ds_dom0 = xr.open_dataset(domcfg0)
#ds_dom1 = xr.open_dataset(domcfg1)
#ds_clomsk  = xr.open_dataset(clomsk)
ds_riv  = xr.open_dataset(rivers)
ds_m2   = xr.open_dataset(M2tmx)
ds_k1   = xr.open_dataset(K1tmx)

#Vars = ['mask_csemp'   , 'mask_csglo', 'mask_csgrpemp', 'mask_csgrpglo',
#        'mask_csgrprnf', 'mask_csrnf', 'mask_csundef' , 'mask_opensea'
#       ]
#ds_clomsk = ds_clomsk[Vars]

#DOM0 = xr.merge([ds_dom0, ds_clomsk])

# Extracting only the part of the domain we need
#ds_clomsk = ds_clomsk.isel(x=slice(IMIN,IMAX+1), y=slice(JMIN,JMAX+1))
ds_riv = ds_riv.isel(x=slice(IMIN,IMAX+1), y=slice(JMIN,JMAX+1))
ds_m2 = ds_m2.isel(x=slice(IMIN,IMAX+1), y=slice(JMIN,JMAX+1))
ds_k1 = ds_k1.isel(x=slice(IMIN,IMAX+1), y=slice(JMIN,JMAX+1))
#DOM1 = xr.merge([ds_dom1, ds_clomsk]) 

outdir = '/data/users/dbruciaf/AGRIF-NAtl/gs/orca04'

#outfile0 = outdir + '/domain_cfg_large_closea.nc'
#enc_v = {x: {"_FillValue": None } for x in DOM0}
#enc_c = {x: {"_FillValue": None } for x in DOM0.coords}
#enc = enc_c | enc_v
#DOM0.to_netcdf(outfile0, encoding=enc, unlimited_dims={'time_counter':True})

#outfile1 = outdir + '/1_domain_cfg_large_closea.nc'
#enc_v = {x: {"_FillValue": None } for x in DOM1}
#enc_c = {x: {"_FillValue": None } for x in DOM1.coords}
#enc = enc_c | enc_v
#DOM1.to_netcdf(outfile1, encoding=enc, unlimited_dims={'time_counter':True})

outriv1 = outdir + '/runoff/eORCA025_runoff_GO6_icb_gs-orca04.nc'
enc = {"icbrnftemper"        : {"_FillValue": None },
       "icbtemper"           : {"_FillValue": None },
       "nav_lat"             : {"_FillValue": None },
       "nav_lat_grid_T"      : {"_FillValue": None },
       "nav_lon"             : {"_FillValue": None },
       "nav_lon_grid_T"      : {"_FillValue": None },
       "rnftemper"           : {"_FillValue": -999. },
       "socoefr"             : {"_FillValue": -999. },
       "sofwficb"            : {"_FillValue": 1.e+20 },
       "sofwfisf"            : {"_FillValue": -999. },
       "sornficb"            : {"_FillValue": 1.00000002004088e+20 },
       "sornfisf"            : {"_FillValue": -999. },
       "sorunoff"            : {"_FillValue": -999. },
       "sozisfmax"           : {"_FillValue": -999. },
       "sozisfmin"           : {"_FillValue": -999. },
       "time_counter"        : {"_FillValue": None },
       "time_counter_bounds" : {"_FillValue": None },
      } 
ds_riv.to_netcdf(outriv1, encoding=enc, unlimited_dims={'time_counter':True})

outm2tmx = outdir + '/tmx/M2rowdrg_R025_modif_nonpositive_r42_gs-orca04.nc'
enc_v = {x: {"_FillValue": None } for x in ds_m2}
enc_c = {x: {"_FillValue": None } for x in ds_m2.coords}
enc = enc_c | enc_v
ds_m2.to_netcdf(outm2tmx, encoding=enc, unlimited_dims={'time_counter':True})

outk1tmx = outdir + '/tmx/K1rowdrg_R025_modif_nonpositive_r42_gs-orca04.nc'
enc_v = {x: {"_FillValue": None } for x in ds_k1}
enc_c = {x: {"_FillValue": None } for x in ds_k1.coords}
enc = enc_c | enc_v
ds_k1.to_netcdf(outk1tmx, encoding=enc, unlimited_dims={'time_counter':True})
