#!/usr/bin/env python

import xarray as xr
import numpy as np

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

def round_up(a,b):
    return int(a/b) + (a % b > 0)

river_data = '/data/users/dbruciaf/GOSI10_input_files/p1.0/eORCA025_runoff_GO6_icb.nc'

#agrif_conf = "/data/users/dbruciaf/AGRIF-NAtl/ls/orca20/AGRIF_FixedGrids.in"
#coord_zoom = ["/data/users/dbruciaf/AGRIF-NAtl/ls/orca20/1_mesh_mask.nc"]
#outdir = '/data/users/dbruciaf/AGRIF-NAtl/ls/orca20/runoff'

#agrif_conf = "/data/users/dbruciaf/AGRIF-NAtl/gs/orca12/AGRIF_FixedGrids.in"
#coord_zoom = ["/data/users/dbruciaf/AGRIF-NAtl/gs/orca12/1_mesh_mask.nc"]
#outdir = '/data/users/dbruciaf/AGRIF-NAtl/gs/orca12/runoff'

#agrif_conf = "/data/users/dbruciaf/AGRIF-NAtl/oliver_gs/orca04/AGRIF_FixedGrids.in"
#coord_zoom = ["/data/users/dbruciaf/AGRIF-NAtl/oliver_gs/orca04/1_mesh_mask.nc"]
#outdir = '/data/users/dbruciaf/AGRIF-NAtl/oliver_gs/orca04/runoff'

#agrif_conf = "/data/users/dbruciaf/AGRIF-NAtl/oliver_gs/orca08/AGRIF_FixedGrids.in"
#coord_zoom = ["/data/users/dbruciaf/AGRIF-NAtl/oliver_gs/orca08/1_mesh_mask.nc"]
#outdir = '/data/users/dbruciaf/AGRIF-NAtl/oliver_gs/orca08/runoff'

#agrif_conf = "/data/users/dbruciaf/AGRIF-NAtl/oliver_gs/orca12/AGRIF_FixedGrids.in"
#coord_zoom = ["/data/users/dbruciaf/AGRIF-NAtl/oliver_gs/orca12/1_mesh_mask.nc"]
#outdir = '/data/users/dbruciaf/AGRIF-NAtl/oliver_gs/orca12/runoff'

#agrif_conf = "/data/users/dbruciaf/AGRIF-NAtl/movf/orca20/AGRIF_FixedGrids.in"
#coord_zoom = ["/data/users/dbruciaf/AGRIF-NAtl/movf/orca20/zps/1_mesh_mask.nc"]
#outdir = '/data/users/dbruciaf/AGRIF-NAtl/movf/orca20/runoff'

#agrif_conf = "/data/users/dbruciaf/AGRIF-NAtl/movf/orca12/AGRIF_FixedGrids.in"
#coord_zoom = ["/data/users/dbruciaf/AGRIF-NAtl/movf/orca12/zps/1_mesh_mask.nc"]
#outdir = '/data/users/dbruciaf/AGRIF-NAtl/movf/orca12/runoff'

#agrif_conf = "/data/users/dbruciaf/AGRIF-NAtl/gosi10-gs_r12/AGRIF_FixedGrids.in"
agrif_conf = "/data/users/dbruciaf/AGRIF-NAtl/gosi10-ls_r20/AGRIF_FixedGrids.in"
coord_zoom = [#"/data/users/dbruciaf/AGRIF-NAtl/gosi10-gs_r12/1_mesh_mask-tmask_as_gosi10-025.nc",
              "/data/users/dbruciaf/AGRIF-NAtl/gosi10-ls_r20/1_mesh_mask-tmask_as_gosi10-025.nc",
              #"/data/users/dbruciaf/AGRIF-NAtl/gosi10na+/3_gb_r20/3_mesh_mask.nc"
             ]
outfile = [#'/data/users/dbruciaf/AGRIF-NAtl/gosi10-gs_r12/runoff/1_eORCA025_runoff_GO6_icb-gs_r12.nc',
           '/data/users/dbruciaf/AGRIF-NAtl/gosi10-ls_r20/runoff/1_eORCA025_runoff_GO6_icb-ls_r20.nc',
           #'/data/users/dbruciaf/AGRIF-NAtl/gosi10na+/3_gb_r20/3_eORCA025_runoff_GO6_icb-gb_r20.nc'
           ]

# Global grids will have nghost_w = 0 and nghost_s = 1 
# (the western boundary is cyclic, the southern boundary 
# over Antarctica is closed).
root_nbghost_s = 1
root_nbghost_w = 0
# AGRIF grids have, by default, nghost_w = nbghost_s = 4. 
# One of these ghost points is masked as required in NEMO.
zoom_nbghost_s = 4
zoom_nbghost_w = 4

# 1) Reading AGRIF_FixedGrids.in and extracting indexes of the zooms
zooms=[]
with open(agrif_conf) as fp:
     line = fp.readline()
     cnt = 1
     while line:
           if len(line.split()) >=4 :
              zooms.append(line.split()[0:6])
           line = fp.readline()
           cnt += 1

# 3) Reading runoff dataset

ds_Priv = xr.open_dataset(river_data)
Vars = ['sornficb'    , 'icbrnftemper', 
        'socoefr'     , 'sofwfisf'    , 
        'sozisfmin'   , 'sozisfmax'   ,
        'time_counter'                ]
#ds_Priv = ds_Priv[Vars]

# 4) Looping over each zoom:

for n in range(len(zooms)):

    # Read zoom coordinates
    ds_zoom = xr.open_dataset(coord_zoom[n]).squeeze()
    lon = ds_zoom.glamt.data
    lat = ds_zoom.gphit.data

    imin = int(zooms[n][0])
    imax = int(zooms[n][1])
    jmin = int(zooms[n][2])
    jmax = int(zooms[n][3])
    refx = int(zooms[n][4])
    refy = int(zooms[n][5])

    nizoom = (imax-imin)*refx + zoom_nbghost_w + zoom_nbghost_w
    njzoom = (jmax-jmin)*refy + zoom_nbghost_s + zoom_nbghost_s
    reffac = 1. / (refx * refy)

    ds_Zriv = xr.Dataset()
    for v in Vars:
        if   len(ds_Priv[v].dims) == 3:
             Data = np.zeros(shape=(ds_Priv[v].shape[0],njzoom,nizoom))
             Coor = dict(
                      time_counter=ds_Priv['time_counter'],
                      nav_lat=(["y","x"], lat),
                      nav_lon=(["y","x"], lon)
                    )
        elif len(ds_Priv[v].dims) == 2:
             Data = np.zeros(shape=(njzoom,nizoom))
             Coor = dict(
                      nav_lat=(["y","x"], lat),
                      nav_lon=(["y","x"], lon)
                    )
        elif len(ds_Priv[v].dims) == 1:
             Data = ds_Priv[v].data
             Coor = {'time_counter':ds_Priv['time_counter']}
        ds_Zriv[v] = xr.DataArray(data=Data, coords=Coor, dims=ds_Priv[v].dims)

    # Recovering the first/last parent tracer cells inside 
    # the zoom (indexes are in python convention)
    imin =  (imin + root_nbghost_w) - 1
    imax =  (imax + root_nbghost_w  - 1) - 1
    jmin =  (jmin + root_nbghost_s) - 1
    jmax =  (jmax + root_nbghost_s  - 1) - 1
    #IMIN = imin - round_up(zoom_nbghost_w, refx)
    #IMAX = imax + round_up(zoom_nbghost_w, refx)
    #JMIN = jmin - round_up(zoom_nbghost_s, refy)
    #JMAX = jmax + round_up(zoom_nbghost_s, refy) 

    # Computing river runoff values for the child grid
    
    jzoom_b = zoom_nbghost_s

    Vars = ['sornficb', 'icbrnftemper', 'socoefr']
    # 'sofwfisf', 'sozisfmin', 'sozisfmax'
    da_msk = ds_Priv['socoefr']*0 + 1

    for jp in range(jmin, jmax+1):
        jzoom_e = jzoom_b + refy
        izoom_b = zoom_nbghost_w
        for ip in range(imin, imax+1):
            if ds_Priv['socoefr'][jp,ip] > 0.: da_msk[jp,ip] = 0
            izoom_e = izoom_b + refx          
            
            for v in Vars:
                if v in ["sornficb"]: #,"sofwfisf"]:      
                   for jt in range(12): 
                       ds_Zriv[v][jt,jzoom_b:jzoom_e, izoom_b:izoom_e] =  reffac * ds_Priv[v][jt,jp,ip]
                else:
                   ds_Zriv[v][jzoom_b:jzoom_e, izoom_b:izoom_e] = ds_Priv[v][jp,ip].copy(deep=True)
            izoom_b = izoom_e
        jzoom_b = jzoom_e

    # setting no runoff in the parent since it is taken into account in the zoom
    #ds_Priv["sornficb"] = ds_Priv.sornficb.where(da_msk == 1, 0.)
    #ds_Priv['socoefr']  = ds_Priv.socoefr.where(da_msk == 1, 0.)

    #outdir = '/data/users/dbruciaf/AGRIF-NAtl/ls/orca12'
    #outrivP = outdir + '/eORCA025_runoff_GO6_icb_zoom_mod.nc'
    #outrivZ = outdir + '/eORCA025_runoff_GO6_icb_ls-orca20.nc'
    
    #outrivZ = outdir + '/eORCA025_runoff_GO6_icb_gs-orca04.nc'
    #outrivZ = outdir + '/eORCA025_runoff_GO6_icb_gs-orca08.nc'
    #outrivZ = outdir + '/eORCA025_runoff_GO6_icb_gs-orca12.nc'
    #outrivZ = outdir + '/eORCA025_runoff_GO6_icb_movf-orca20.nc'
    #outrivZ = outdir + '/eORCA025_runoff_GO6_icb_movf-orca12.nc'
    outrivZ = outfile[n]

    encP = {"icbrnftemper"        : {"_FillValue": None },
           "nav_lat"             : {"_FillValue": None },
           "nav_lon"             : {"_FillValue": None },
           "socoefr"             : {"_FillValue": -999. },
           "sofwfisf"            : {"_FillValue": -999. },
           "sornficb"            : {"_FillValue": 1.00000002004088e+20 },
           "sozisfmax"           : {"_FillValue": -999. },
           "sozisfmin"           : {"_FillValue": -999. },
           "time_counter"        : {"_FillValue": None },
           }
    encZ = {"icbrnftemper"        : {"_FillValue": None },
           "nav_lat"             : {"_FillValue": None },
           "nav_lon"             : {"_FillValue": None },
           "socoefr"             : {"_FillValue": -999. },
           "sornficb"            : {"_FillValue": 1.00000002004088e+20 },
           "time_counter"        : {"_FillValue": None },
           }  
    #ds_Priv.to_netcdf(outrivP, encoding=encP, unlimited_dims={'time_counter':True})
    ds_Zriv.to_netcdf(outrivZ, encoding=encZ, unlimited_dims={'time_counter':True})
    
