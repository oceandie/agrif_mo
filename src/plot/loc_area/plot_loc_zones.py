#!/usr/bin/env python

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import cartopy.feature as feature
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==============================================================================
# Input parameters

# 1. INPUT FILES

# Change this to match your local paths set-up
#base_dir = "/data/users/dbruciaf/AGRIF-NAtl/gs/orca04"
#base_dir = "/data/users/dbruciaf/AGRIF-NAtl/movf/orca12"
base_dir = "/data/users/dbruciaf/AGRIF-NAtl/movf/orca20"

#loc_file = base_dir + '/bathymetry.loc_area.dep0.0015_pol3_sig1_itr2.nc'
#loc_file = base_dir + '/loc_MEs/bathymetry.loc_area.dep0.002_pol3_sig1_itr3.nc'
#glo_file = base_dir + '/bathy_as_parent/domain_cfg_large_noclosea.nc'
#loc_file = base_dir + '/loc_MEs/bathymetry.loc_area.dep0.002_pol4_sig1_itr2.nc'
#loc_file = base_dir + '/loc_MEs/bathymetry.loc_area.dep0.001_pol4_sig1_itr2.nc'
#loc_file = base_dir + '/loc_MEs/bathymetry.loc_area.dep1e-05_pol4_sig1_itr2.nc'
#loc_file = base_dir + '/loc_MEs/bathymetry.loc_area.dep3000_pol3_sig1_itr2.nc'
#loc_file = base_dir + '/loc_MEs/bathymetry.loc_area.dep4000_pol3_sig1_itr2.nc'
#loc_file = base_dir + '/loc_MEs/bathymetry.loc_area.dep4000_pol3_sig1_itr4.nc'
#loc_file = base_dir + '/loc_MEs/bathymetry.loc_area.dep4000_pol3_sig1_itr8.nc'
#loc_file = base_dir + '/loc_MEs/bathymetry.loc_area.dep4000_pol3_sig3_itr2.nc'
loc_file = base_dir + '/loc_MEs/bathymetry.loc_area.dep4000_pol3_sig3_itr3.nc'

glo_file = '/data/users/dbruciaf/GOSI10_input_files/p1.0/domcfg_eORCA025_v3.1_r42.nc'

# AGRIF domains

# Gulf Stream region
#i = [815, 1080]  
#j = [779,  910]

# MED OVF
i = [1097, 1145]  
j = [820,  860]

# 2. PLOT
proj = ccrs.Mercator() #ccrs.Robinson()

# ==============================================================================

# COMPUTING AGRIF INDRXES:
# The locations of the edges of the zoom, as defined by imin, imax, jmin and jmax 
# are set excluding the parent ghost cells.
# Global grids will have nghost_w = 0 and nghost_s = 1 (the western boundary is 
# cyclic, the southern boundary over Antarctica is closed).
root_nbghost_s = 1
root_nbghost_w = 0
# AGRIF grids have, by default, nghost_w = nbghost_s = 4 one of these ghost points 
# is masked as required in NEMO.
zoom_nbghost_s = 4
zoom_nbghost_w = 4
#rx = 1
#ry = 1
rx = 5
ry = 5

# indexes in python convention
imin =  ( i[0] + root_nbghost_w) - 1
imax =  ( i[1] + root_nbghost_w  - 1) - 1
jmin =  ( j[0] + root_nbghost_s) - 1
jmax =  ( j[1] + root_nbghost_s  - 1) - 1

IMIN = imin - int(zoom_nbghost_w / rx)
IMAX = imax + int(zoom_nbghost_w / rx)
JMIN = jmin - int(zoom_nbghost_s / ry)
JMAX = jmax + int(zoom_nbghost_s / ry)

# Printing in FORTRAN convention
print('i=',i)
print('j=',j)
print('I=',[IMIN+1,IMAX+1])
print('J=',[JMIN+1,JMAX+1])

# Load localisation file

ds_loc = xr.open_dataset(loc_file)
ds_glo = xr.open_dataset(glo_file)

# Extract only the part of the domain we need 
#ds_loc = ds_loc.isel(x=slice(880,1150),y=slice(880,1140))

# Extracting variables
bat = ds_loc.Bathymetry
lat = ds_loc.nav_lat
lon = ds_loc.nav_lon
loc_msk = ds_loc.s2z_msk
LAT = ds_glo.nav_lat
LON = ds_glo.nav_lon

#loc_msk = np.ma.array(loc_msk)
loc_s = loc_msk.where(loc_msk==2)
loc_t = loc_msk.where(loc_msk==1)

print(np.nanmax(bat.where(loc_s==2)))

#LLcrnrlon = -87. 
#LLcrnrlat =  20. 
#URcrnrlon = -12.
#URcrnrlat =  45.

LLcrnrlon = -15. 
LLcrnrlat =  30. 
URcrnrlon =   0.
URcrnrlat =  42.

map_lims = [LLcrnrlon, URcrnrlon, LLcrnrlat, URcrnrlat]

fig = plt.figure(figsize=(25, 25), dpi=100)
spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
ax = fig.add_subplot(spec[:1], projection=proj)
ax.coastlines(linewidth=4, zorder=6)
ax.add_feature(feature.LAND, color='gray',edgecolor='black',zorder=1)

# Drawing settings
transform = ccrs.PlateCarree()

# Grid settings
gl_kwargs = dict()
gl = ax.gridlines(**gl_kwargs)
gl.xlines = False
gl.ylines = False
gl.top_labels = True
gl.right_labels = True
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 50, 'color': 'k'}
gl.ylabel_style = {'size': 50, 'color': 'k'}

cn_lev = [0., 250., 500., 1000., 2000., 3000., 4000., 5000.]
ax.contour(lon, lat, bat, levels=cn_lev,colors='black',linewidths=1., transform=transform, zorder=4)
#ax.contour(lon, lat, bat, levels=[0],colors='green',linewidths=5., transform=transform, zorder=4)

ax.pcolormesh(lon, lat, loc_s, cmap = 'autumn', transform=transform, zorder=3, alpha=0.6, antialiased=True)
ax.pcolormesh(lon, lat, loc_t, cmap = 'winter_r',transform=transform, zorder=2, alpha=0.9, antialiased=True)

#ax.plot(LON[j[0]:j[1],i[0]], LAT[j[0]:j[1],i[0]], '-',color='red', linewidth=4, transform=transform)
#ax.plot(LON[j[1],i[0]:i[1]], LAT[j[1],i[0]:i[1]], '-',color='red', linewidth=4, transform=transform)
#ax.plot(LON[j[0]:j[1],i[1]], LAT[j[0]:j[1],i[1]], '-',color='red', linewidth=4, transform=transform)
#ax.plot(LON[j[0],i[0]:i[1]], LAT[j[0],i[0]:i[1]], '-',color='red', linewidth=4, transform=transform)

lw = 5

ax.plot(LON[jmin:jmax,imin     ],
        LAT[jmin:jmax,imin     ],
        '-',
        color='red',
        linewidth=lw,
        transform=transform
       )
ax.plot(LON[jmax     ,imin:imax],
        LAT[jmax     ,imin:imax],
        '-',
        color='red',
        linewidth=lw,
        transform=transform
       )
ax.plot(LON[jmin:jmax,imax     ],
        LAT[jmin:jmax,imax     ],
        '-',
        color='red',
        linewidth=lw,
        transform=transform
       )
ax.plot(LON[jmin     ,imin:imax],
        LAT[jmin     ,imin:imax],
        '-',
        color='red',
        linewidth=lw,
        transform=transform
       )


ax.plot(LON[JMIN:JMAX,IMIN     ], 
        LAT[JMIN:JMAX,IMIN     ], 
        '-',
        color='black', 
        linewidth=lw, 
        transform=transform
       )
ax.plot(LON[JMAX     ,IMIN:IMAX], 
        LAT[JMAX     ,IMIN:IMAX], 
        '-',
        color='black', 
        linewidth=lw, 
        transform=transform
       )
ax.plot(LON[JMIN:JMAX,IMAX     ],
        LAT[JMIN:JMAX,IMAX     ], 
        '-',
        color='black', 
        linewidth=lw, 
        transform=transform
       )
ax.plot(LON[JMIN     ,IMIN:IMAX], 
        LAT[JMIN     ,IMIN:IMAX], 
        '-',
        color='black', 
        linewidth=lw, 
        transform=transform
       )

ax.set_extent(map_lims)

out_name ='loc_areas.png'
plt.savefig(out_name,bbox_inches="tight", pad_inches=0.1)
plt.close()

