#!/usr/bin/env python

#     |------------------------------------------------------------|
#     | Author: Diego Bruciaferri                                  |
#     | Date and place: 28-02-2022, Met Office, UK                 |
#     |------------------------------------------------------------|


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import xarray as xr
from utils import calc_r0, e3_to_dep, compute_masks

# ==============================================================================
# Input parameters

# 1. INPUT FILES

domcfg = ['/data/users/dbruciaf/AGRIF-NAtl/movf/orca20/loc_MEs/domain_cfg.nc',
          '/data/users/dbruciaf/AGRIF-NAtl/movf/orca20/zps/1_domain_cfg.nc']
fbathy = [['/data/users/dbruciaf/AGRIF-NAtl/movf/orca20/loc_MEs/bathymetry.loc_area.dep4000_pol3_sig3_itr3.MEs_4env_4000_r10_r10_r12.nc'],[]]

fig_path = './' 

# Define the section we want to plot:
# TO DO: extend to general section

#j_sec = 99 # we plot a zonal cross section
#j_sec = 97
#j_sec = 112
j_sec = 116

# ==============================================================================

for exp in range(len(domcfg)):

    # Loading domain geometry
    ds_dom = xr.open_dataset(domcfg[exp], drop_variables=("x", "y","nav_lev")).squeeze()
    ds_dom = ds_dom.rename({"nav_lev": "z"})
    # Computing land-sea masks
    ds_dom = compute_masks(ds_dom, merge=True)

    # Extracting variables
    e3t = ds_dom.e3t_0.squeeze()
    e3w = ds_dom.e3w_0.squeeze()
    e3u = ds_dom.e3u_0.squeeze()
    e3uw = ds_dom.e3uw_0.squeeze()  
    glamt = ds_dom.glamt.squeeze()
    glamu = ds_dom.glamu.squeeze()
    tmask = ds_dom.tmask.squeeze()   
    e3t_1d = ds_dom.e3t_1d.squeeze()
    e3w_1d = ds_dom.e3w_1d.squeeze()

    # Computing depths
    gdepw, gdept  = e3_to_dep(e3w,  e3t)
    gdepuw, gdepu = e3_to_dep(e3uw, e3u)
    gdepw_1d, gdept_1d  = e3_to_dep(e3w_1d,  e3t_1d) 

    # Adding 3rd dimension for plotting
    glamt = glamt.expand_dims({"z": len(ds_dom.z)})
    glamu = glamu.expand_dims({"z": len(ds_dom.z)})

    # Determining the type of vertical coordinate
    for vcr in ["zco", "zps", "sco"]:
        if ds_dom["ln_"+vcr] == 1: vcoor = vcr

    # Checking if localisation has been applied
    loc_msk = ds_dom.bathy_metry * 0.
    if not fbathy[exp]:
       if vcoor == "sco":
          loc_msk[:,:] = 1.
       else:
          loc_msk[:,:] = 0.
    else:
       ds_bat  = xr.open_dataset(fbathy[exp][0], drop_variables=("x", "y")).squeeze()
       loc_msk = ds_bat.s2z_msk
       #loc_msk = loc_msk.where(loc_msk==0,1)

    # Checking if we are using a ME system
    hbatt = []
    if vcoor == "sco" and fbathy[exp]:
       ds_bat  = xr.open_dataset(fbathy[exp][0], drop_variables=("x", "y")).squeeze()
       nenv = 1
       while nenv > 0:
         name_env = "hbatt_"+str(nenv)
         if name_env in ds_bat.variables:
            envlp = ds_bat[name_env]
            envlp = envlp.where(loc_msk>1)
            hbatt.append(envlp)
            nenv+=1
         else:
            nenv=0

    # Computing slope paramter of model levels
    r0_3D = e3u * 0.0
    r0_3D = r0_3D.rename("r0")
    for k in range(r0_3D.shape[0]):
        r0 = calc_r0(gdept.isel(z=k))
        r0_3D[k,:,:] = r0

    # Extracting arrays of the section
    gdept  = gdept.isel(y=j_sec).values
    gdepw  = gdepw.isel(y=j_sec).values
    gdepu  = gdepu.isel(y=j_sec).values
    gdepuw = gdepuw.isel(y=j_sec).values
    gdept_1d = gdept_1d.values
    gdepw_1d = gdepw_1d.values
    e3t    = e3t.isel(y=j_sec).values
    e3w    = e3w.isel(y=j_sec).values 
    e3u    = e3u.isel(y=j_sec).values
    e3uw   = e3uw.isel(y=j_sec).values
    glamt  = glamt.isel(y=j_sec).values
    glamu  = glamu.isel(y=j_sec).values
    tmask  = tmask.isel(y=j_sec).values    
    r0     = r0_3D.isel(y=j_sec).values
    loc_msk = loc_msk.isel(y=j_sec).values
    if hbatt:
       for env in range(len(hbatt)): hbatt[env] = hbatt[env].isel(y=j_sec).values

    r0[tmask==0] = np.nan

    # Plotting ----------------------------------------------------------

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(54, 34))
    plt.sca(ax) # Set the current Axes to ax and the current Figure to the parent of ax.
    ax.invert_yaxis()
    #ax.set_facecolor('gray')
    ax.set_facecolor('black')

    # RMAX -----------------------------------------------
    # We create a polygon patch for each T-cell of the
    # section and we colour it based on the value of the rmax
    #
    ni = r0.shape[1]
    nk = r0.shape[0]
    patches = []
    colors = []
    depw = np.zeros((nk,ni))

    for k in range(0,nk-1):             
        for i in range(1,ni-1):
            x = [0.5*(glamt[k  ,i-1]+glamt[k   ,i ]), # U_(k,i-1)
                      glamt[k  ,i  ]                , # T_(k,i)
                 0.5*(glamt[k  ,i  ]+glamt[k  ,i+1]), # U_(k,i)
                 0.5*(glamt[k+1,i  ]+glamt[k+1,i+1]), # U_(k+1,i)
                      glamt[k+1,i  ]                , # T_(k+1,i)
                 0.5*(glamt[k+1,i-1]+glamt[k+1,i  ]), # U_(k+1,i-1)
                 0.5*(glamt[k  ,i-1]+glamt[k  ,i  ])] # U_(k  ,i-1)
                 
            if loc_msk[i] == 0:
               y = [gdepuw[k,i-1],
                    gdepw [k,i  ],
                    gdepuw[k,i  ],
                    np.max([gdepuw[k+1,i  ], gdepw [k+1,i  ]]),
                    gdepw [k+1,i  ],
                    np.max([gdepw [k+1,i  ], gdepuw[k+1,i-1]]),
                    gdepuw[k  ,i-1]]
               depw[k,i] = gdepw_1d[k]
            else:         
               y = [gdepuw[k,i-1],
                    gdepw [k,i  ],
                    gdepuw[k,i  ],
                    gdepuw[k+1,i  ],
                    gdepw [k+1,i  ],
                    gdepuw[k+1,i-1],
                    gdepuw[k  ,i-1]]
               depw[k,i] = gdepw[k,i]

            polygon = Polygon(np.vstack((x,y)).T, closed=True)
            patches.append(polygon)  
            colors = np.append(colors,r0[k,i])

    # MODEL W-levels and U-points ----------------------------
    for k in range(len(ds_dom.z)):
        x = glamt[k,:]
        z = depw[k,:]
        ax.plot(
            x,
            z,
            color="k",
            linewidth=0.5,
            zorder=5
        )

    # MODEL ENVELOPES
    if hbatt:
       for env in range(len(hbatt)):
           x = glamt[0,:]
           z = hbatt[env]
           ax.plot(
               x,
               z,
               color="red",
               #linewidth=2.,
               linewidth=4.,
               zorder=5
        )
         
    #for i in range(len(ds_dom.x)):
    #    ax.plot(
    #       [glamu[0,i], glamu[-1,i]],
    #       [0., gdepuw[-1,i]],
    #       color="k",
    #       linestyle='--',
    #       zorder=5
    #    )

    # MODEL T-points ----------------------------
    ax.scatter(np.ravel(glamt[:-1,:]),
               np.ravel(gdept[:-1,:]),
               s=1,
               color='black',
               zorder=5
               )

    # PLOT setting ----------------------------
    ax.set_ylim(np.amax(gdepw), 0.0)
    ax.set_xlim(glamt[0,0], glamt[0,-1])

    p = PatchCollection(patches, alpha=0.9) #edgecolor='black', alpha=0.8)
    p.set_array(np.array(colors))
    p.set_clim((0,0.2))
    #cmap = cm.get_cmap("hot_r").copy()
    # get discrete colormap
    cmap = plt.get_cmap('hot_r', 8)
    p.set_cmap(cmap)
    ax.add_collection(p)

    plt.xticks(fontsize=50.)
    #plt.xlabel('Domain extension [km]', fontsize=50.)
    plt.xlabel('Domain extension [$^{\circ} East$]', fontsize=50.)
    plt.yticks(fontsize=50.)
    plt.ylabel('Depth [m]' ,fontsize=50.)

    cbaxes = inset_axes(ax, width="22%", height="3.5%", loc=4, borderpad=15.)
    cb = plt.colorbar(p, cax=cbaxes, ticks=[0., 0.1, 0.2], 
                      orientation='horizontal', extend="max",drawedges=True) 
    cb.set_label("Slope parameter", size=50, color='w')
    cb.ax.tick_params(labelsize=50, labelcolor='w')
    cb.outline.set_color('white')
    cb.outline.set_linewidth(4)
    cb.dividers.set_color('black')
    cb.dividers.set_linewidth(2)

    fig_name = vcoor+'_j'+ str(j_sec)+'_section_rmax.png'
    plt.savefig(fig_path+fig_name, bbox_inches="tight")
    plt.close()
