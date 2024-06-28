#!/usr/bin/env python

from typing import Tuple
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

#=======================================================================================
def e3_to_dep(e3W: xr.DataArray, e3T: xr.DataArray) -> Tuple[xr.DataArray, ...]:

    gdepT = xr.full_like(e3T, None, dtype=np.double).rename('gdepT')
    gdepW = xr.full_like(e3W, None, dtype=np.double).rename('gdepW')

    gdepW[{"z":0}] = 0.0
    gdepT[{"z":0}] = 0.5 * e3W[{"z":0}]
    for k in range(1, e3W.sizes["z"]):
        gdepW[{"z":k}] = gdepW[{"z":k-1}] + e3T[{"z":k-1}]
        gdepT[{"z":k}] = gdepT[{"z":k-1}] + e3W[{"z":k}]

    return tuple([gdepW, gdepT])

max_dep = 6000

# PATHS of domain_cfg.nc or mesh_mask.nc files of the z-level model
cfg_file = "/data/users/dbruciaf/AGRIF-NAtl/parent/GOSI10p0-025/r4.2_halo/domcfg_eORCA025_v2_r42_closea.nc"
dsz = xr.open_dataset(cfg_file)
dsz = dsz.rename_dims({'nav_lev':'z'})

e3T = dsz.e3t_1d.squeeze()
e3W = dsz.e3w_1d.squeeze()

# Computing vertical levels depth
gdepw, gdept = e3_to_dep(e3W, e3T)

print('k', 'gdepw')
n = 0
for k in range(gdepw.shape[0]):
    if n == 0: print(k+1, gdepw.data[k])
    if gdepw[k] > max_dep: n = 1

gdepw.plot(marker="o", yincrease=False)
plt.show()

