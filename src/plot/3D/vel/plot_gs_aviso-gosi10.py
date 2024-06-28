import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib as mpl   

# Input settings
# a) GOSI10p0 grid coordinates
domf_0 = '/data/users/dbruciaf/AGRIF-NAtl/gs/orca12/domain_cfg.nc'

# b) main output directories
avi_dir = '/data/users/dbruciaf/GS/ssh_aviso/sla_cmems/monthly/'
mod_dir = '/scratch/dbruciaf/AGRIF/gs_orca12/u-dd982/'

# c) output stems
avi_stm = 'dt_global_allsat_phy_l4_'
mod_stm = 'nemo_dd982o_1m_' 

# Reading GOSI10p0 grid coordinates
ds_d0 = xr.open_dataset(domf_0)
nav_lon0 = ds_d0.nav_lon
nav_lat0 = ds_d0.nav_lat

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

        # AVISO geostrophic currents
        aviso = avi_dir + avi_stm + date_avi + '.nc' 
        ds_avi = xr.open_dataset(aviso)
        ds_avi = ds_avi.rename_dims({'latitude':'y','longitude':'x'})
        avi_u = ds_avi['ugos'].squeeze()
        avi_v = ds_avi['vgos'].squeeze()

        avi_u = avi_u.isel({'x':slice(290,690),'y':slice(420,625)})
        avi_v = avi_v.isel({'x':slice(290,690),'y':slice(420,625)})

        # Interpolating from U,V to T 
        U = avi_u.rolling({'x':2}).mean().fillna(0.)
        V = avi_v.rolling({'y':2}).mean().fillna(0.)
        # Calc speed
        speed = np.sqrt(np.power(U,2) + np.power(V,2))
        speed = speed.where(speed>0)
        speed['longitude'] = speed.longitude + 360.

        # GOSI10p0 surface current 
        Ugrd = mod_dir + mod_stm + date_mod + '_grid-U.nc'
        Vgrd = mod_dir + mod_stm + date_mod + '_grid-V.nc'
        ds_U = xr.open_dataset(Ugrd).rename_dims({'depthu':'z'}).squeeze()
        ds_V = xr.open_dataset(Vgrd).rename_dims({'depthv':'z'}).squeeze()
        ds_U = ds_U.assign_coords(nav_lon=nav_lon0, nav_lat=nav_lat0)
        ds_V = ds_V.assign_coords(nav_lon=nav_lon0, nav_lat=nav_lat0)
        ds_U.coords["x"] = range(ds_U.dims["x"])
        ds_U.coords["y"] = range(ds_U.dims["y"])
        ds_V.coords["x"] = range(ds_V.dims["x"])
        ds_V.coords["y"] = range(ds_V.dims["y"])

        u = ds_U['uo'].squeeze()
        v = ds_V['vo'].squeeze()
        # Interpolating from U,V to T 
        U = u.rolling({'x':2}).mean().fillna(0.)
        V = v.rolling({'y':2}).mean().fillna(0.)
        # Calc speed
        spd = np.sqrt(np.power(U,2) + np.power(V,2))
        spd = spd.where(spd>0)

        # Get rid of discontinuity on lon grid 
        # (from https://gist.github.com/willirath/fbfd21f90d7f2a466a9e47041a0cee64)
        # see also https://docs.xarray.dev/en/stable/examples/multidimensional-coords.html
        after_discont = ~(spd.coords["nav_lon"].diff("x", label="upper") > 0).cumprod("x").astype(bool)
        spd.coords["nav_lon"] = (spd.coords["nav_lon"] + 360 * after_discont)

        spd = spd.isel(x=slice(1, -1), y=slice(None, -1))
        spd = spd.isel({'x':slice(700,1120), 'y':slice(750,1020)})

        # PLOTTING #

        # Model land
        landa = speed.fillna(-999)
        landa = landa.where(landa==-999)
        land = spd.fillna(-999)
        land = land.where(land==-999)

        cmap = 'hot' #'Blues_r'
        # figaspect(0.5) makes the figure twice as wide as it is tall. 
        # Then the *1.5 increases the size of the figure.
        fig = plt.figure(figsize=plt.figaspect(0.5)*4.)
        ax = fig.add_subplot((111), projection='3d')

        lev = np.arange(0.,1.,0.01)

        # AVISO 
        ax.contourf(landa.longitude, landa.latitude, landa, zdir='z', offset=0., colors='gray')
        p = ax.contourf(speed.longitude, speed.latitude, speed, levels=lev, zdir='z', offset=0., cmap=cmap, vmin=0, vmax=1.)

        # GOSI10
        ax.contourf(land.nav_lon, land.nav_lat, land.isel(z=0), zdir='z', offset=1, colors='gray')
        ax.contourf(spd.nav_lon, spd.nav_lat, spd.isel(z=0), levels=lev, zdir='z', offset=1, cmap=cmap, vmin=0, vmax=1.)

        plt.gca().invert_zaxis()
        ax.set_zlim(1, 0)
        ax.set_axis_off()

        #cax = fig.add_axes([ax.get_position().x1-0.04,ax.get_position().y0+0.2,0.006,ax.get_position().height/2])
        #cbar = plt.colorbar(p, cax=cax, extend='max')
        #cbar.ax.tick_params(labelsize=20)


        ax.text(305, 35, 0.3, "AVISO", color='w', size='15', zdir=(1,-0.01,0.), zorder=301)
        ax.text(300, 32, 1.3, "GLOBAL 1/4", color='w', size='15', zdir=(1,-0.01,0), zorder=301)

        ax.view_init(elev=31, azim=-65)

        print(f"Saving {date_avi}", end=": ")
        plt.savefig(date_avi + "_aviso-gosi10p0.png", bbox_inches="tight")
        print("done")
        plt.close()

