import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib as mpl   

# Input settings
# a) grid coordinates
domf_0 = '/data/users/dbruciaf/AGRIF-NAtl/gs/orca12/domain_cfg.nc'
domf_1 = '/data/users/dbruciaf/AGRIF-NAtl/gs/orca12/1_domain_cfg.nc'

# b) main output directories
mod_dir = '/scratch/dbruciaf/AGRIF/gs_orca12/u-dd748/'

# c) output stems
mod_stm = 'nemo_dd748o_1m_'
agr_stm = 'nemo_1-gs-orca12o_1m_'

# Reading GOSI10p0 grid coordinates
ds_d0 = xr.open_dataset(domf_0)
ds_d1 = xr.open_dataset(domf_1)
nav_lon0 = ds_d0.nav_lon
nav_lat0 = ds_d0.nav_lat
nav_lon1 = ds_d1.nav_lon
nav_lat1 = ds_d1.nav_lat

for yyyy in range(2005,2008):
    yy1 = str(yyyy)
    for mm in range(1,13):

        if mm < 10:
           mm1 = '0'+str(mm)
        else:
           mm1 = str(mm)
        if mm+1 < 10:
           mm2 = '0'+str(mm+1)
        else:
           mm2 = str(mm+1)

        if mm == 12:
           mm2 = '01'
           yy2 = str(yyyy+1)
        else:
           yy2 = str(yyyy)

        date_avi = yy1+mm1
        date_mod = yy1+mm1+'01-'+ yy2+mm2+'01'

        Ugrd_0 = mod_dir + mod_stm + date_mod + '_grid-U.nc'
        Vgrd_0 = mod_dir + mod_stm + date_mod + '_grid-V.nc'
        Ugrd_1 = mod_dir + agr_stm + date_mod + '_grid-U.nc'
        Vgrd_1 = mod_dir + agr_stm + date_mod + '_grid-V.nc'
        ds_U0 = xr.open_dataset(Ugrd_0).rename_dims({'depthu':'z'}).squeeze()
        ds_V0 = xr.open_dataset(Vgrd_0).rename_dims({'depthv':'z'}).squeeze()
        ds_U1 = xr.open_dataset(Ugrd_1, chunks={}).rename_dims({'depthu':'z'}).squeeze()
        ds_V1 = xr.open_dataset(Vgrd_1, chunks={}).rename_dims({'depthv':'z'}).squeeze()

        # Taking care of longitude discontinuity and computing spped

        ds_U0 = ds_U0.assign_coords(nav_lon=nav_lon0, nav_lat=nav_lat0)
        ds_V0 = ds_V0.assign_coords(nav_lon=nav_lon0, nav_lat=nav_lat0)
        ds_U1 = ds_U1.assign_coords(nav_lon=nav_lon1, nav_lat=nav_lat1)
        ds_V1 = ds_V1.assign_coords(nav_lon=nav_lon1, nav_lat=nav_lat1)
        ds_U0.coords["x"] = range(ds_U0.dims["x"])
        ds_U0.coords["y"] = range(ds_U0.dims["y"])
        ds_V0.coords["x"] = range(ds_V0.dims["x"])
        ds_V0.coords["y"] = range(ds_V0.dims["y"])
        ds_U1.coords["x"] = range(ds_U1.dims["x"])
        ds_U1.coords["y"] = range(ds_U1.dims["y"])
        ds_V1.coords["x"] = range(ds_V1.dims["x"])
        ds_V1.coords["y"] = range(ds_V1.dims["y"])

        # 1) PARENT model
        u0 = ds_U0['uo'].squeeze()
        v0 = ds_V0['vo'].squeeze()
        # Interpolating from U,V to T
        U0 = u0.rolling({'x':2}).mean().fillna(0.)
        V0 = v0.rolling({'y':2}).mean().fillna(0.)
        # Calc speed
        spd0 = np.sqrt(np.power(U0,2) + np.power(V0,2))
        spd0 = spd0.where(spd0>0)

        # Get rid of discontinuity on lon grid 
        #(from https://gist.github.com/willirath/fbfd21f90d7f2a466a9e47041a0cee64)
        # see also https://docs.xarray.dev/en/stable/examples/multidimensional-coords.html
        after_discont = ~(spd0.coords["nav_lon"].diff("x", label="upper") > 0).cumprod("x").astype(bool)
        spd0.coords["nav_lon"] = (spd0.coords["nav_lon"] + 360 * after_discont)
        spd0 = spd0.isel(x=slice(1, -1), y=slice(None, -1))
        # Marking the boundary of AGRIF domain on the parent model
        # read from the AGRIF_FixedGrids,in file
        spd0[:,779:910,815:816] = np.nan
        spd0[:,779:910,1080:1081] = np.nan
        spd0[:,779:780,815:1080] = np.nan
        spd0[:,910:911,815:1080] = np.nan
        spd0 = spd0.isel({'x':slice(700,1120), 'y':slice(750,1020)})

        # 2) AGRIF ZOOM
        u1 = ds_U1['uo'].squeeze()
        v1 = ds_V1['vo'].squeeze()
        # Interpolating from U,V to T
        U1 = u1.rolling({'x':2}).mean().fillna(0.)
        V1 = v1.rolling({'y':2}).mean().fillna(0.)
        # Calc speed
        spd1 = np.sqrt(np.power(U1,2) + np.power(V1,2))
        spd1 = spd1.where(spd1>0)

        # Get rid of discontinuity on lon grid 
        #(from https://gist.github.com/willirath/fbfd21f90d7f2a466a9e47041a0cee64)
        # see also https://docs.xarray.dev/en/stable/examples/multidimensional-coords.html
        after_discont = ~(spd1.coords["nav_lon"].diff("x", label="upper") > 0).cumprod("x").astype(bool)
        spd1.coords["nav_lon"] = (spd1.coords["nav_lon"] + 360 * after_discont)
        spd1 = spd1.isel(x=slice(1, -1), y=slice(None, -1))

        # PLOTTING # --------------------------------------------------------------------------------

        # Model land
        land0 = spd0.fillna(-999)
        land0 = land0.where(land0==-999)
        land1 = spd1.fillna(-999)
        land1 = land1.where(land1==-999)

        cmap = 'hot' #'Blues_r'
        # figaspect(0.5) makes the figure twice as wide as it is tall. 
        # Then the *1.5 increases the size of the figure.
        fig = plt.figure(figsize=plt.figaspect(0.5)*4.)
        ax = fig.add_subplot((111), projection='3d')

        lev = np.arange(0.,1.,0.01)

        # AGRIF ZOOM
        ax.contourf(land1.nav_lon, land1.nav_lat, land1.isel(z=0),          zdir='z', offset=0., colors='gray')
        ax.contourf(spd1.nav_lon, spd1.nav_lat, spd1.isel(z=0), levels=lev, zdir='z', offset=0., cmap=cmap, vmin=0, vmax=1)

        # PARENT MODEL
        ax.contourf(land0.nav_lon, land0.nav_lat, land0.isel(z=0),                   zdir='z', offset=0.5, colors='gray')
        p = ax.contourf(spd0.nav_lon, spd0.nav_lat, spd0.isel(z=0), levels=lev, zdir='z', offset=0.5, cmap=cmap, vmin=0, vmax=1)

        plt.gca().invert_zaxis()
        ax.set_zlim(1, 0)
        ax.set_axis_off()

        cax = fig.add_axes([ax.get_position().x1-0.04,ax.get_position().y0+0.25,0.006,ax.get_position().height/2])
        cbar = plt.colorbar(p, cax=cax)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label(label='$m\;s^{-1}$',size=20,weight='bold')

        ax.text(298, 42, 0.3, "AGRIF 1/12", color='w', size='15', zdir=(1,-0.01,0.), zorder=301)
        ax.text(315, 22, 0.61, "GLOBAL 1/4", color='w', size='15', zdir=(1,-0.01,0), zorder=301)

        #print('ax.azim {}'.format(ax.azim))
        #print('ax.elev {}'.format(ax.elev))
        ax.view_init(elev=31, azim=-65)

        print(f"Saving {date_avi}", end=": ")
        plt.savefig(date_avi + "_agrif-gosi10p0.png", bbox_inches="tight")
        print("done")
        plt.close()
