import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib as mpl   
import matplotlib.lines as mlines
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import cartopy.mpl.ticker as ctk
import cartopy.feature as feature

# Input settings
# a) T files
yyyy = "1998"
cnt_T = "/scratch/dbruciaf/GOSI10p0/u-dc531/nemo_dc531o_1m_"+yyyy+"0301-"+yyyy+"0401_grid-T.nc"
agr_T = "/scratch/dbruciaf/GOSI10p0/u-df907/nemo_df907o_1m_"+yyyy+"0301-"+yyyy+"0401_grid-T.nc"

# c) GOSI10p0 grid coordinates
domf_0 = '/data/users/dbruciaf/AGRIF-NAtl/gs/orca12/domain_cfg.nc'

# d) PLOT limits
lon0 = -75. #-85
lon1 = -7.
lat0 = 45. #20.
lat1 = 67.
map_lims = [lon0, lon1, lat0, lat1]
proj = ccrs.Mercator()

###################################################

# Reading GOSI10p0 grid coordinates
ds_d0 = xr.open_dataset(domf_0)
nav_lon0 = ds_d0.nav_lon
nav_lat0 = ds_d0.nav_lat
bathy = ds_d0.bathy_metry.squeeze()
del ds_d0

# BATHY
bathy = bathy.isel({'x':slice(700,1120), 'y':slice(750,1120)})
nav_lon = nav_lon0.isel({'x':slice(700,1120), 'y':slice(750,1120)})
nav_lat = nav_lat0.isel({'x':slice(700,1120), 'y':slice(750,1120)})

# GOSI10p0
ds_cnt = xr.open_dataset(cnt_T)
ds_cnt = ds_cnt.assign_coords(nav_lon=nav_lon0, nav_lat=nav_lat0)
ds_cnt = ds_cnt.rename_dims({'y':'j','x':'i','deptht':'k'})
ds_cnt.coords["i"] = range(ds_cnt.dims["i"])
ds_cnt.coords["j"] = range(ds_cnt.dims["j"])
cntT = ds_cnt['somxzint1'].squeeze()
# Get rid of discontinuity on lon grid 
# (from https://gist.github.com/willirath/fbfd21f90d7f2a466a9e47041a0cee64)
# see also https://docs.xarray.dev/en/stable/examples/multidimensional-coords.html
after_discont = ~(cntT.coords["nav_lon"].diff("i", label="upper") > 0).cumprod("i").astype(bool)
cntT.coords["nav_lon"] = (cntT.coords["nav_lon"] + 360 * after_discont)


cntT  = cntT.isel(i=slice(1, -1), j=slice(None, -1))
cntT  = cntT.isel(i=slice(700, 1120), j=slice(750, 1120))
land  = cntT.isel(i=slice(700,1120), j=slice(750,1120))

# GOSI10p0-AGRIF
ds_agr = xr.open_dataset(agr_T)
ds_agr = ds_agr.assign_coords(nav_lon=nav_lon0, nav_lat=nav_lat0)
ds_agr = ds_agr.rename_dims({'y':'j','x':'i','deptht':'k'})
ds_agr.coords["i"] = range(ds_agr.dims["i"])
ds_agr.coords["j"] = range(ds_agr.dims["j"])
agrT = ds_agr['somxzint1'].squeeze()
# Get rid of discontinuity on lon grid 
# (from https://gist.github.com/willirath/fbfd21f90d7f2a466a9e47041a0cee64)
# see also https://docs.xarray.dev/en/stable/examples/multidimensional-coords.html
after_discont = ~(agrT.coords["nav_lon"].diff("i", label="upper") > 0).cumprod("i").astype(bool)
agrT.coords["nav_lon"] = (agrT.coords["nav_lon"] + 360 * after_discont)

agrT = agrT.isel(i=slice(1, -1), j=slice(None, -1))
agrT = agrT.isel(i=slice(700,1120), j=slice(750,1120))

# PLOTTING # ----------------------------------------------------------------------

# Model land
land = bathy.where(bathy==0)

cmap = 'jet'
levs = np.arange(0.,2400.,10)

# figaspect(0.5) makes the figure twice as wide as it is tall. 
# Then the *1.5 increases the size of the figure.
fig1 = plt.figure(figsize=plt.figaspect(0.5)*4.)
ax1 = fig1.add_subplot((111), projection=proj)
fig2 = plt.figure(figsize=plt.figaspect(0.5)*4.)
ax2 = fig2.add_subplot((111), projection=proj)

transform = ccrs.PlateCarree()

gl1 = ax1.gridlines(draw_labels=True, x_inline=False, y_inline=False)
gl1.xlines = False
gl1.ylines = False
gl1.top_labels = False
gl1.bottom_labels = True
gl1.right_labels = True
gl1.left_labels = False
#gl.xformatter = LONGITUDE_FORMATTER
#gl.yformatter = LATITUDE_FORMATTER
gl1.xlabel_style = {'size': 25, 'color': 'k'}
gl1.ylabel_style = {'size': 25, 'color': 'k'}
gl1.rotate_labels=False
gl1.xlocator=ctk.LongitudeLocator(6)
gl1.ylocator=ctk.LatitudeLocator(4)
gl1.xformatter=ctk.LongitudeFormatter(zero_direction_label=False)
gl1.yformatter=ctk.LatitudeFormatter()

gl2 = ax2.gridlines(draw_labels=True, x_inline=False, y_inline=False)
gl2.xlines = False
gl2.ylines = False
gl2.top_labels = False
gl2.bottom_labels = True
gl2.right_labels = True
gl2.left_labels = False
#gl.xformatter = LONGITUDE_FORMATTER
#gl.yformatter = LATITUDE_FORMATTER
gl2.xlabel_style = {'size': 25, 'color': 'k'}
gl2.ylabel_style = {'size': 25, 'color': 'k'}
gl2.rotate_labels=False
gl2.xlocator=ctk.LongitudeLocator(6)
gl2.ylocator=ctk.LatitudeLocator(4)
gl2.xformatter=ctk.LongitudeFormatter(zero_direction_label=False)
gl2.yformatter=ctk.LatitudeFormatter()
 

# LAND
ax1.contourf(nav_lon, nav_lat, land, transform = transform, colors='black')
ax2.contourf(nav_lon, nav_lat, land, transform = transform, colors='black')

lev = [500., 1000., 2000., 3000.]
ax1.contour(nav_lon, nav_lat, bathy, levels=lev, colors='k', linewidths=1, transform = transform)
ax1.contour(nav_lon, nav_lat, bathy, levels=[0], colors='k', linewidths=3, transform = transform)
ax2.contour(nav_lon, nav_lat, bathy, levels=lev, colors='k', linewidths=1, transform = transform)
ax2.contour(nav_lon, nav_lat, bathy, levels=[0], colors='k', linewidths=3, transform = transform)

# GOSI10
p1 = ax1.contourf(cntT.nav_lon, cntT.nav_lat, cntT, levels=levs, cmap=cmap, transform=transform, extend='max')

# GOSI10-AGRIF
p2 = ax2.contourf(agrT.nav_lon, agrT.nav_lat, agrT, levels=levs, cmap=cmap, transform=transform, extend='max')

# AGRIF domain
# Labrador Sea region
i = [905,  976]
j = [930, 1030]
ax2.plot(nav_lon0[j[0]:j[1],i[0]], nav_lat0[j[0]:j[1],i[0]], '-',color='lime', linewidth=6, transform=transform)
ax2.plot(nav_lon0[j[1],i[0]:i[1]], nav_lat0[j[1],i[0]:i[1]], '-',color='lime', linewidth=6, transform=transform)
ax2.plot(nav_lon0[j[0]:j[1],i[1]], nav_lat0[j[0]:j[1],i[1]], '-',color='lime', linewidth=6, transform=transform)
ax2.plot(nav_lon0[j[0],i[0]:i[1]], nav_lat0[j[0],i[0]:i[1]], '-',color='lime', linewidth=6, transform=transform)

ax1.set_extent(map_lims, crs=ccrs.PlateCarree())
ax2.set_extent(map_lims, crs=ccrs.PlateCarree())

cbar1 = plt.colorbar(p1)
cbar1.ax.tick_params(labelsize=40)
cbar1.set_label(label='Mixed Layer Depth $[m]$',size=40,weight='bold')

cbar2 = plt.colorbar(p2)
cbar2.ax.tick_params(labelsize=40)
cbar2.set_label(label='Mixed Layer Depth $[m]$',size=40,weight='bold')


# Legend

fig1.savefig("labrador_sea_mld_CNTRL_"+yyyy+"-03.png", bbox_inches="tight")
fig2.savefig("labrador_sea_mld_AGRIF_"+yyyy+"-03.png", bbox_inches="tight")
print("done")
plt.close()

