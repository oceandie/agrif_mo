#!/usr/bin/env python

import os
import sys
import subprocess
import numpy as np
import xarray
from xnemogcm import open_domain_cfg
from plot_section import mpl_sec_loop
from utils import compute_masks

# ========================================================================
# INPUT PARAMETERS

orca='025' # "025"

if orca == '025':
   DOMCFG_zps = '/data/users/dbruciaf/GS/orca025/zps_GO8/domain_cfg_zps_new_bathy.nc'
elif orca == "12":
   DOMCFG_zps = '/data/users/dbruciaf/GS/orca12/zps_GO8/domain_cfg_zps_new_bathy.nc'

# 2. ANALYSIS
sec_lon1 = [-81.11, -70.38]
sec_lat1 = [ 28.15,  27.80]
sec_lon2 = [-81.85, -73.46]
sec_lat2 = [ 29.92,  30.56]
sec_lon3 = [-81.33, -73.28]
sec_lat3 = [ 32.06,  31.65]
sec_lon4 = [-82.56, -66.82]
sec_lat4 = [ 32.82,  26.69]
sec_lon5 = [-80.01, -69.55]
sec_lat5 = [ 33.14,  32.44]
sec_lon6 = [-78.02, -67.90]
sec_lat6 = [ 34.62,  32.25]
# Cape Hatteras
sec_lon7 = [-75.91, -67.38]
sec_lat7 = [ 35.69,  34.33]
#
sec_lon8 = [-75.58, -67.38]
sec_lat8 = [ 38.56,  35.42]
# Ezer 2016 sec
sec_lon9 = [-77., -60.]
sec_lat9 = [ 35.,  35.]
# North Cape Hatteras
sec_lon10 = [-78., -60.]
sec_lat10 = [ 37.,  37.]
#
sec_lon11 = [-70.11, -65.54]
sec_lat11 = [ 44.40,  38.83]
#
sec_lon12 = [-58.99, -54.93, -47.96]
sec_lat12 = [ 51.75,  48.18,  39.15]

sec_I_indx_1b_L  = [sec_lon1, sec_lon2, sec_lon3, sec_lon4, sec_lon5, sec_lon6, sec_lon7, sec_lon8, sec_lon9, sec_lon10, sec_lon11, sec_lon12]
sec_J_indx_1b_L  = [sec_lat1, sec_lat2, sec_lat3, sec_lat4, sec_lat5, sec_lat6, sec_lat7, sec_lat8, sec_lat9, sec_lat10, sec_lat11, sec_lat12]
coord_type_1b_L  = "dist"
rbat2_fill_1b_L  = "false"
xlim_1b_L        = "maxmin" #[0., 1600.]
ylim_1b_L        = [0., 5900.] #5900.]
vlevel_1b_L      = 'Z_ps'
xgrid_1b_L       = "false"

# ========================================================================
if orca == '025':
   x1 = 730
   x2 = 1110
   y1 = 760
   y2 = 1000
elif orca == '12':
   x1 = 2230
   x2 = 3100
   y1 = 2065
   y2 = 3300

# Loading domain geometry
ds_dom  = open_domain_cfg(files=[DOMCFG_zps])
for i in ['bathymetry','bathy_meter']:
    for dim in ['x','y']:
        ds_dom[i] = ds_dom[i].rename({dim: dim+"_c"})

# Computing masks
ds_dom = compute_masks(ds_dom, merge=True)

# Extracting only the part of the domain we need
ds_dom = ds_dom.isel(x_c=slice(x1,x2),x_f=slice(x1,x2),
                     y_c=slice(y1,y2),y_f=slice(y1,y2))

tlon2 = ds_dom["glamt"].values
tlat2 = ds_dom["gphit"].values
e3t_3 = ds_dom["e3t_0"].values
e3w_3 = ds_dom["e3w_0"].values
tmsk3 = ds_dom["tmask"].values
bathy = ds_dom["bathymetry"].values

nk = e3t_3.shape[0]
nj = e3t_3.shape[1]
ni = e3t_3.shape[2]

tlon3 = np.repeat(tlon2[np.newaxis, :, :], nk, axis=0)
tlat3 = np.repeat(tlat2[np.newaxis, :, :], nk, axis=0)

# Computing model levels' depth
tdep3 = np.zeros(shape=(nk,nj,ni))
wdep3 = np.zeros(shape=(nk,nj,ni))
wdep3[0,:,:] = 0.
tdep3[0,:,:] = 0.5 * e3w_3[0,:,:]
for k in range(1, nk):
    wdep3[k,:,:] = wdep3[k-1,:,:] + e3t_3[k-1,:,:]
    tdep3[k,:,:] = tdep3[k-1,:,:] + e3w_3[k,:,:]

proj = []

# PLOTTING VERTICAL DOMAIN

var_strng  = ""
unit_strng = ""
date       = ""
timeres_dm = ""
timestep   = []
PlotType   = ""
var4       = []
hbatt      = []
mbat_ln    = "false"
mbat_fill  = "true"
varlim     = "no"
check      = 'true'
check_val  = 'false'

mpl_sec_loop('ORCA025-zps mesh', '.png', var_strng, unit_strng, date, timeres_dm, timestep, PlotType,
              sec_I_indx_1b_L, sec_J_indx_1b_L, tlon3, tlat3, tdep3, wdep3, tmsk3, var4, proj,
              coord_type_1b_L, vlevel_1b_L, bathy, hbatt, rbat2_fill_1b_L, mbat_ln, mbat_fill,
              xlim_1b_L, ylim_1b_L, varlim, check, check_val, xgrid_1b_L)

