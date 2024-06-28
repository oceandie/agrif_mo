import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib as mpl   
import cmocean

# Input settings
# a) grid coordinates
domf_0 = '/data/users/dbruciaf/AGRIF-NAtl/gs/orca12/domain_cfg.nc'
domf_1 = '/data/users/dbruciaf/AGRIF-NAtl/gs/orca12/1_domain_cfg.nc'

# Reading GOSI10p0 grid coordinates
ds_d0 = xr.open_dataset(domf_0)
ds_d1 = xr.open_dataset(domf_1)

nav_lon0 = ds_d0.nav_lon
nav_lat0 = ds_d0.nav_lat
nav_lon1 = ds_d1.nav_lon
nav_lat1 = ds_d1.nav_lat

bathy_0 = ds_d0.bathy_metry.squeeze()
bathy_1 = ds_d1.bathy_metry.squeeze()

# Taking care of longitude discontinuity
bathy_0 = bathy_0.assign_coords(nav_lon=nav_lon0, nav_lat=nav_lat0)
bathy_1 = bathy_1.assign_coords(nav_lon=nav_lon1, nav_lat=nav_lat1)
bathy_0.coords["x"] = range(ds_d0.dims["x"])
bathy_0.coords["y"] = range(ds_d0.dims["y"])
bathy_1.coords["x"] = range(ds_d1.dims["x"])
bathy_1.coords["y"] = range(ds_d1.dims["y"])

# Get rid of discontinuity on lon grid 
#(from https://gist.github.com/willirath/fbfd21f90d7f2a466a9e47041a0cee64)
# see also https://docs.xarray.dev/en/stable/examples/multidimensional-coords.html
after_discont = ~(bathy_0.coords["nav_lon"].diff("x", label="upper") > 0).cumprod("x").astype(bool)
bathy_0.coords["nav_lon"] = (bathy_0.coords["nav_lon"] + 360 * after_discont)
bathy_0 = bathy_0.isel(x=slice(1, -1), y=slice(None, -1))
# Marking the boundary of AGRIF domain on the parent model
# read from the AGRIF_FixedGrids,in file
bathy_0[779:910,815:816] = np.nan
bathy_0[779:910,1080:1081] = np.nan
bathy_0[779:780,815:1080] = np.nan
bathy_0[910:911,815:1080] = np.nan
bathy_0 = bathy_0.isel({'x':slice(700,1120), 'y':slice(750,1020)})

# Get rid of discontinuity on lon grid 
#(from https://gist.github.com/willirath/fbfd21f90d7f2a466a9e47041a0cee64)
# see also https://docs.xarray.dev/en/stable/examples/multidimensional-coords.html
after_discont = ~(bathy_1["nav_lon"].diff("x", label="upper") > 0).cumprod("x").astype(bool)
bathy_1.coords["nav_lon"] = (bathy_1.coords["nav_lon"] + 360 * after_discont)
bathy_1 = bathy_1.isel(x=slice(1, -1), y=slice(None, -1))

# PLOTTING # --------------------------------------------------------------------------------

# Model land
#land0 = bathy_0.fillna(-999)
land0 = bathy_0.where(bathy_0==0)
#land1 = bathy_1.fillna(-999)
land1 = bathy_1.where(bathy_1==0)
bathy_0 = bathy_0.where(bathy_0>0)
bathy_1 = bathy_1.where(bathy_1>0)

cmap = 'nipy_spectral_r'#'terrain_r' #cmocean.cm.deep #'terrain_r'
# figaspect(0.5) makes the figure twice as wide as it is tall. 
# Then the *1.5 increases the size of the figure.
fig = plt.figure(figsize=plt.figaspect(0.5)*4.)
ax = fig.add_subplot((111), projection='3d')

lev = np.arange(0.,6200.,50)

# AGRIF ZOOM
ax.contourf(land1.nav_lon, land1.nav_lat, land1,             zdir='z', offset=0., colors='black')#, zorder=10)
ax.contourf(bathy_1.nav_lon, bathy_1.nav_lat, bathy_1, levels=lev, zdir='z', offset=0., cmap=cmap, vmin=0, vmax=6200)

# PARENT MODEL
ax.contourf(land0.nav_lon, land0.nav_lat, land0,                 zdir='z', offset=0.5, colors='black')#, zorder=10)
p = ax.contourf(bathy_0.nav_lon, bathy_0.nav_lat, bathy_0, levels=lev, zdir='z', offset=0.5, cmap=cmap, vmin=0, vmax=6200)#, zorder=1)

plt.gca().invert_zaxis()
ax.set_zlim(1, 0)
ax.set_axis_off()

cax = fig.add_axes([ax.get_position().x1-0.04,ax.get_position().y0+0.25,0.006,ax.get_position().height/2])
cbar = plt.colorbar(p, cax=cax)
cbar.ax.tick_params(labelsize=20)
cbar.set_label(label='$Depth [m]$',size=20,weight='bold')

ax.text(298, 42, 0.3, "AGRIF 1/12", color='w', size='15', zdir=(1,-0.01,0.), zorder=301)
ax.text(315, 22, 0.61, "GLOBAL 1/4", color='w', size='15', zdir=(1,-0.01,0), zorder=301)

#print('ax.azim {}'.format(ax.azim))
#print('ax.elev {}'.format(ax.elev))
ax.view_init(elev=31, azim=-65)

print(f"Saving", end=": ")
plt.savefig("bathy_agrif-gosi10p0.png", bbox_inches="tight")
print("done")
plt.close()
