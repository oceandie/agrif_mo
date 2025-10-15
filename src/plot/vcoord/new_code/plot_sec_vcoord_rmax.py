#!/usr/bin/env python

#     |------------------------------------------------------------|
#     | Author: Diego Bruciaferri                                  |
#     | Date and place: 13-08-2024, Met Office, UK                 |
#     |------------------------------------------------------------|


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import xarray as xr
from utils import calc_r0, prepare_domcfg, plot_sec

# ==============================================================================
# Input parameters

config = 'orca20' # 'orca12'
fig_path = './'

# 1. INPUT FILES

vvar   = "r0x"
fig_path = './'

if config == 'orca12':

   expdir = '/data/users/dbruciaf/AGRIF-NAtl/movf/orca12/'
   domcfg = [expdir + 'agrif-loc_MEs/4env_4000_r065_r045/1_domain_cfg.nc',
             expdir + 'zps/1_domain_cfg.nc']
   fbathy = [expdir + 'loc_MEs/bathymetry.loc_area.dep4000_pol3_sig3_itr3.MEs_4env_4000_r065_r045.nc',
             None]

elif config == 'orca20':

   expdir = '/data/users/dbruciaf/AGRIF-NAtl/movf/orca20/'
   domcfg = [expdir + 'agrif-loc_MEs/4env_4000_r045_r045/1_domain_cfg.nc',
             expdir + 'zps/1_domain_cfg.nc']
   fbathy = [expdir + 'loc_MEs/bathymetry.loc_area.dep4000_pol3_sig3_itr6.MEs_4env_4000_r045_r045.nc',
             None]

# Define the section we want to plot:
list_sec = [
            {'lon':[-13.29 , -5.60, -1.17] , 'lat':[36.03, 35.90, 36.11]}, # cross GSt
            {'lon':[-8.3333, -8.3333] , 'lat':[36.8632, 33.798]}, # SEMANE 2002 I (10 -> 400)
            {'lon':[-8.1675, -6.332 ] , 'lat':[35.8335, 35.8335]}, # SEMANE 2002 II (410 -> 520)
            {'lon':[-6.2513, -6.254 ] , 'lat':[35.468 , 36.0332]}, # SEMANE 2002 III (530 -> 700)
           ]

# ==============================================================================

for exp in range(len(domcfg)):

    # Loading domain geometry
    ds_dom, hbatt, vcoor = prepare_domcfg(domcfg[exp], fbathy[exp])

    hbatt = [] # TODO: use realistic envelopes

    # Computing slope paramter of model levels
    r0_3D = ds_dom.gdept_0 * 0.0
    r0_3D = r0_3D.rename("r0x")
    for k in range(r0_3D.shape[0]):
        r0 = calc_r0(ds_dom.gdept_0.isel(z=k))
        r0_3D[k,:,:] = r0
    r0_3D = r0_3D.where(ds_dom.tmask > 0)
    ds_dom[vvar] = r0_3D

    # Extracting variables for the specific type of section
    var_list = ["gdepu_0" , "gdepuw_0", "gdepv_0" , "gdepvw_0",
                "gdept_0" , "gdepw_0" , "gdepf_0" , "gdepfw_0",
                "glamt"   , "glamu"   , "glamv"   , "glamf"   ,
                "gphit"   , "gphiu"   , "gphiv"   , "gphif"   ,
                "gdepw_1d", "loc_msk" , "r0x"]

    for coord_sec in list_sec:

        sec_lon = coord_sec['lon']
        sec_lat = coord_sec['lat']

        print ('section through lon:', sec_lon)
        print ('                lat:', sec_lat)

        ds_sec = ds_dom[var_list]
        ds_var = ds_dom[var_list]

        sec_name = str(sec_lon[0])+'-'+str(sec_lat[0])+'_'+str(sec_lon[-1])+'-'+str(sec_lat[-1])

        fig_name = vcoor+'_section_'+vvar+'_'+sec_name+'.png'
        plot_sec(fig_name, fig_path, ds_sec, ds_var, vvar, sec_lon, sec_lat, hbatt, imap=True)

