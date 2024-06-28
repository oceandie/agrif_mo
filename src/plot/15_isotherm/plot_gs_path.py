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
woa_T = "/data/users/dbruciaf/NOAA_WOA18/2005-2017/temperature/0.25/woa18_A5B7_t00_04.nc"
cnt_T = "/data/users/dbruciaf/AGRIF-NAtl/parent/GOSI10p0-025/r4.2_halo/output/u-dd982/nemo_dd982o_1y_average_2000-2004_grid_T.nc"
agr_T = "/data/users/dbruciaf/AGRIF-NAtl/gs/orca12/output/u-dd748/nemo_dd748o_1y_average_2000-2004_grid_T.nc"
O25_T = '/data/users/dbruciaf/AGRIF-NAtl/parent/GOSI9p8.0/output/u-cm028/nemo_cm028o_1y_average_2000-2004_grid_T.nc'
O12_T = '/data/users/dbruciaf/AGRIF-NAtl/parent/GOSI9p8.0/output/u-cm491/nemo_cm491o_1y_average_2000-2004_grid_T.nc'

# b) Level at 200m
lev200 = [24, 29, 29, 29, 29] 

# c) GOSI10p0 grid coordinates
domf_0 = '/data/users/dbruciaf/AGRIF-NAtl/gs/orca12/domain_cfg.nc'
domf_1 = '/data/users/dbruciaf/OVF/GOSI9-eORCA025/domcfg_eORCA025_v3.nc'
domf_2 = '/data/users/dbruciaf/GS/orca12/GO8_domain/coordinates.nc'

# d) PLOT limits
lon0 = -82. #-85
lon1 = -32. #-8.
lat0 = 27. #20.
lat1 = 50.
map_lims = [lon0, lon1, lat0, lat1]
proj = ccrs.Mercator()

###################################################

# Reading GOSI10p0 grid coordinates
ds_d0 = xr.open_dataset(domf_0)
nav_lon0 = ds_d0.nav_lon
nav_lat0 = ds_d0.nav_lat
bathy = ds_d0.bathy_metry.squeeze()
del ds_d0

ds_d1 = xr.open_dataset(domf_1)
nav_lon1 = ds_d1.nav_lon
nav_lat1 = ds_d1.nav_lat
del ds_d1

ds_d2 = xr.open_dataset(domf_2)
nav_lon2 = ds_d2.nav_lon
nav_lat2 = ds_d2.nav_lat
del ds_d2

# WOA T
ds_woa = xr.open_dataset(woa_T, decode_times=False)
ds_woa = ds_woa.rename_dims({'lat':'j','lon':'i','depth':'k'})
woa15  = ds_woa['t_an'].squeeze().isel({'i':slice(310,730),'j':slice(420,640),'k':lev200[0]})
woa15['lon'] = woa15.lon + 360.
woa15 = woa15.where(woa15.lat>30.)
woa15 = woa15.where(woa15.lon<-38.)

# BATHY
bathy = bathy.isel({'x':slice(700,1120), 'y':slice(750,1020)})
nav_lon = nav_lon0.isel({'x':slice(700,1120), 'y':slice(750,1020)})
nav_lat = nav_lat0.isel({'x':slice(700,1120), 'y':slice(750,1020)})
# GOSI10p0
ds_cnt = xr.open_dataset(cnt_T)
ds_cnt = ds_cnt.assign_coords(nav_lon=nav_lon0, nav_lat=nav_lat0)
ds_cnt = ds_cnt.rename_dims({'y':'j','x':'i','deptht':'k'})
ds_cnt.coords["i"] = range(ds_cnt.dims["i"])
ds_cnt.coords["j"] = range(ds_cnt.dims["j"])
cntT = ds_cnt['thetao_con'].squeeze()
# Get rid of discontinuity on lon grid 
# (from https://gist.github.com/willirath/fbfd21f90d7f2a466a9e47041a0cee64)
# see also https://docs.xarray.dev/en/stable/examples/multidimensional-coords.html
after_discont = ~(cntT.coords["nav_lon"].diff("i", label="upper") > 0).cumprod("i").astype(bool)
cntT.coords["nav_lon"] = (cntT.coords["nav_lon"] + 360 * after_discont)

cntT  = cntT.isel(i=slice(1, -1), j=slice(None, -1))
cnt15 = cntT.isel({'i':slice(700,1120), 'j':slice(750,1020), 'k':lev200[1]})
cnt15 = cnt15.where(cnt15.nav_lat>30.)
cnt15 = cnt15.where(cnt15.nav_lon<(-38.+360.))
land  = cntT.isel({'i':slice(700,1120), 'j':slice(750,1020), 'k':0})

# GOSI10p0-AGRIF
ds_agr = xr.open_dataset(agr_T)
ds_agr = ds_agr.assign_coords(nav_lon=nav_lon0, nav_lat=nav_lat0)
ds_agr = ds_agr.rename_dims({'y':'j','x':'i','deptht':'k'})
ds_agr.coords["i"] = range(ds_agr.dims["i"])
ds_agr.coords["j"] = range(ds_agr.dims["j"])
agrT = ds_agr['thetao_con'].squeeze()
# Get rid of discontinuity on lon grid 
# (from https://gist.github.com/willirath/fbfd21f90d7f2a466a9e47041a0cee64)
# see also https://docs.xarray.dev/en/stable/examples/multidimensional-coords.html
after_discont = ~(agrT.coords["nav_lon"].diff("i", label="upper") > 0).cumprod("i").astype(bool)
agrT.coords["nav_lon"] = (agrT.coords["nav_lon"] + 360 * after_discont)

agrT  = agrT.isel(i=slice(1, -1), j=slice(None, -1))
agr15 = agrT.isel({'i':slice(700,1120), 'j':slice(750,1020), 'k':lev200[2]})
agr15 = agr15.where(agr15.nav_lat>30.)
agr15 = agr15.where(agr15.nav_lon<(-38.+360.))

# GOSI9-025
ds_025 = xr.open_dataset(O25_T)
ds_025 = ds_025.assign_coords(nav_lon=nav_lon1, nav_lat=nav_lat1)
ds_025 = ds_025.rename_dims({'y':'j','x':'i','deptht':'k'})
ds_025.coords["i"] = range(ds_025.dims["i"])
ds_025.coords["j"] = range(ds_025.dims["j"])
O25T = ds_025['thetao_con'].squeeze()
# Get rid of discontinuity on lon grid 
# (from https://gist.github.com/willirath/fbfd21f90d7f2a466a9e47041a0cee64)
# see also https://docs.xarray.dev/en/stable/examples/multidimensional-coords.html
after_discont = ~(O25T.coords["nav_lon"].diff("i", label="upper") > 0).cumprod("i").astype(bool)
O25T.coords["nav_lon"] = (O25T.coords["nav_lon"] + 360 * after_discont)

O25T  = O25T.isel(i=slice(1, -1), j=slice(None, -1))
o2515 = O25T.isel({'i':slice(700,1120), 'j':slice(750,1020), 'k':lev200[3]})
o2515 = o2515.where(o2515.nav_lat>30.)
o2515 = o2515.where(o2515.nav_lon<(-38.+360.))

# GOSI9-12
ds_012 = xr.open_dataset(O12_T)
ds_012 = ds_012.assign_coords(nav_lon=nav_lon2, nav_lat=nav_lat2)
ds_012 = ds_012.rename_dims({'y':'j','x':'i','deptht':'k'})
ds_012.coords["i"] = range(ds_012.dims["i"])
ds_012.coords["j"] = range(ds_012.dims["j"])
O12T = ds_012['thetao_con'].squeeze()
# Get rid of discontinuity on lon grid 
# (from https://gist.github.com/willirath/fbfd21f90d7f2a466a9e47041a0cee64)
# see also https://docs.xarray.dev/en/stable/examples/multidimensional-coords.html
after_discont = ~(O12T.coords["nav_lon"].diff("i", label="upper") > 0).cumprod("i").astype(bool)
O12T.coords["nav_lon"] = (O12T.coords["nav_lon"] + 360 * after_discont)

O12T  = O12T.isel(i=slice(1, -1), j=slice(None, -1))
o1215 = O12T.isel({'i':slice(2220,3440), 'j':slice(2300,3000), 'k':lev200[4]})
o1215 = o1215.where(o1215.nav_lat>30.)
o1215 = o1215.where(o1215.nav_lon<(-38.+360.))

# PLOTTING # ----------------------------------------------------------------------

# Model land
land = land.fillna(-999)
land = land.where(land==-999)

# figaspect(0.5) makes the figure twice as wide as it is tall. 
# Then the *1.5 increases the size of the figure.
fig = plt.figure(figsize=plt.figaspect(0.5)*4.)
ax = fig.add_subplot((111), projection=proj)

transform = ccrs.PlateCarree()

gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
gl.xlines = False
gl.ylines = False
gl.top_labels = False
gl.bottom_labels = True
gl.right_labels = True
gl.left_labels = False
#gl.xformatter = LONGITUDE_FORMATTER
#gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 40, 'color': 'k'}
gl.ylabel_style = {'size': 40, 'color': 'k'}
gl.rotate_labels=False
gl.xlocator=ctk.LongitudeLocator(6)
gl.ylocator=ctk.LatitudeLocator(4)
gl.xformatter=ctk.LongitudeFormatter(zero_direction_label=False)
gl.yformatter=ctk.LatitudeFormatter()


# LAND
ax.contourf(land.nav_lon, land.nav_lat, land, colors='gray', transform = transform)

lev = [500., 1000., 1500., 2000., 2500., 3000., 3500., 4000., 4500., 5000.]
ax.contour(nav_lon, nav_lat, bathy, levels=lev, colors='k', linewidths=1, transform = transform)
ax.contour(nav_lon, nav_lat, bathy, levels=[0], colors='k', linewidths=3, transform = transform)

# WOA
p = ax.contour(woa15.lon, woa15.lat, woa15, levels=[15], colors='k', linewidths=8, transform=transform)

# GOSI10
ax.contour(cnt15.nav_lon, cnt15.nav_lat, cnt15, levels=[15], colors='r', linewidths=8, transform=transform)

# GOSI10-AGRIF
ax.contour(agr15.nav_lon, agr15.nav_lat, agr15, levels=[15], colors='blue', linewidths=8, transform=transform)

# GOSI9-025
#ax.contour(o2515.nav_lon, o2515.nav_lat, o2515, levels=[15], colors='deepskyblue', linewidths=8, transform=transform)

# GOSI9-12
#ax.contour(o1215.nav_lon, o1215.nav_lat, o1215, levels=[15], colors='magenta', linewidths=8, transform=transform)

# AGRIF domain
# Gulf Stream region
i = [815, 1080]
j = [779,  910]
#ax.plot(nav_lon0[j[0]:j[1],i[0]], nav_lat0[j[0]:j[1],i[0]], '-',color='g', linewidth=3, transform=transform)
#ax.plot(nav_lon0[j[1],i[0]:i[1]], nav_lat0[j[1],i[0]:i[1]], '-',color='g', linewidth=3, transform=transform)
#ax.plot(nav_lon0[j[0]:j[1],i[1]], nav_lat0[j[0]:j[1],i[1]], '-',color='g', linewidth=3, transform=transform)
#ax.plot(nav_lon0[j[0],i[0]:i[1]], nav_lat0[j[0],i[0]:i[1]], '-',color='g', linewidth=3, transform=transform)

ax.set_extent(map_lims, crs=ccrs.PlateCarree())

# Legend
obs  = mlines.Line2D([], [], linewidth=6, color='black', marker=None,
                    label='WOA18')
mod1 = mlines.Line2D([], [], linewidth=6, color='red', marker=None,
                    label='GOSI10p0-4')
mod2 = mlines.Line2D([], [], linewidth=6, color='blue', marker=None,
                    label='GOSI10p0-4 + AGRIF-12 (parent)')
#mod3 = mlines.Line2D([], [], linewidth=6, color='deepskyblue', marker=None,
#                    label='GOSI9-4')
#mod4 = mlines.Line2D([], [], linewidth=6, color='magenta', marker=None,
#                    label='GOSI9-12')

#plt.legend(handles=[obs,mod1, mod2, mod3, mod4], fontsize="35", loc="lower right", fancybox=True, framealpha=1.)
plt.legend(handles=[obs,mod1, mod2], fontsize="35", loc="lower right", fancybox=True, framealpha=1.)


plt.savefig("gulf_stream_15deg_woa-gosi10p0-agrif.png", bbox_inches="tight")
print("done")
plt.close()

