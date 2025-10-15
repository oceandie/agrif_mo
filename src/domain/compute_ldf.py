#!/usr/bin/env python

import numpy as np
import xarray as xr

def calc_ud(A0, e1, e2, ldf):

    emax = np.nanmax(np.maximum(e1,e2))
    if ldf == 'lap':
       ud = (2. * A0) / emax
    elif ldf == 'bilap':
       ud = (12. * A0) / (emax**3)

    return ud

# input

AT4 = 300.   # m2/s, lap
AM4 = 1.5e11 # m4/s, bilap

AT20 = 60.  # m2/s, lap
AM20 = 6.e9 # m4/s, bilap

# configuration at 1/4

coord = '/data/users/dbruciaf/AGRIF-NAtl/gs/orca04/zps/domain_cfg_zps.nc'
ds_cor  = xr.open_dataset(coord)
e1 = ds_cor.e1t.squeeze()
e2 = ds_cor.e2t.squeeze()

print('rn_Ud for diffusivity at 1/4: ', calc_ud(AT4, e1, e2, 'lap'))
print('rn_Ud for viscosity at 1/4: ', calc_ud(AM4, e1, e2, 'bilap'))

print('rn_Ud for diffusivity at 1/20: ', calc_ud(AT20, e1/5., e2/5., 'lap'))
print('rn_Ud for viscosity at 1/20: ', calc_ud(AM20, e1/5., e2/5., 'bilap'))

