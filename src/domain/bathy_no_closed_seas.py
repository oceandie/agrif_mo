#!/usr/bin/env python

from os.path import isfile, basename, splitext
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr

#=======================================================================================
def floodfill(field,j,i,checkValue,newValue):
    '''
    This is a modified version of the original algorithm:

    1) checkValue is the value we do not want to change,
       i.e. is the value identifying the boundaries of the 
       region we want to flood.
    2) newValue is the new value we want for points whose initial value
       is not checkValue and is not newValue.
       N.B. if a point with initial value = to newValue is met, then the
            flooding stops. 

    Example:

    a = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 3, 2, 1, 5, 6, 9, 0],
                  [0, 0, 8, 9, 0, 0, 0, 4, 0],
                  [0, 0, 8, 9, 7, 2, 3, 0, 0],
                  [0, 0, 4, 4, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0]])
   
    j_start = 3
    i_start = 4
    b = com.floodfill(a,j_start,i_start,0,2)
 
    b = array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 2, 2, 1, 5, 6, 9, 0],
               [0, 0, 2, 2, 0, 0, 0, 4, 0],
               [0, 0, 2, 2, 2, 2, 3, 0, 0],
               [0, 0, 2, 2, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0]])

    '''
    Field = np.copy(field)

    theStack = [ (j, i) ]

    while len(theStack) > 0:
          try:
              j, i = theStack.pop()
              if Field[j,i] == checkValue:
                 continue
              if Field[j,i] == newValue:
                 continue
              Field[j,i] = newValue
              theStack.append( (j, i + 1) )  # right
              theStack.append( (j, i - 1) )  # left
              theStack.append( (j + 1, i) )  # down
              theStack.append( (j - 1, i) )  # up
          except IndexError:
              continue # bounds reached

    return Field

# ==============================================================================
# 1. Checking for input files
# ==============================================================================

# Load GOSI bathymetry
GOSI_bat = "/data/users/dbruciaf/AGRIF-NAtl/parent/GOSI10p0-025/r4.2_halo/bathy_noisf_ORCA025_IBCSO_ETOPO1_mask75L_v2.2_r42.nc"
ds_bathy = xr.open_dataset(GOSI_bat).squeeze()
#ds_bathy = ds_bathy.set_coords(["nav_lon","nav_lat"])
bathy = ds_bathy["Bathymetry"].squeeze()

wrk = bathy.data.copy()

# Closing Caspian Sea and Aral Sea
ii = 1347
ij = 898
wrk = floodfill(wrk, ij, ii, 0., -1.)
ii = 1386
ij = 931
wrk = floodfill(wrk, ij, ii, 0., -1.)
# Closing Lakes in USA
ii = 805 
ij = 913
wrk = floodfill(wrk, ij, ii, 0., -1.)
ii = 822 
ij = 894
wrk = floodfill(wrk, ij, ii, 0., -1.)
ii = 824
ij = 874
wrk = floodfill(wrk, ij, ii, 0., -1.)
ii = 839
ij = 884
wrk = floodfill(wrk, ij, ii, 0., -1.)
# Lake Victoria
ii = 1280
ij = 680
wrk = floodfill(wrk, ij, ii, 0., -1.)

wrk[wrk==-1]=0.
ds_bathy["Bathymetry"].data = wrk
ds_bathy["Bathymetry"].plot()
plt.show()

# -------------------------------------------------------------------------------------   
# Writing the bathy_meter.nc file

out_file = "/data/users/dbruciaf/AGRIF-NAtl/parent/GOSI10p0-025/r4.2_halo/bathy_noisf_ORCA025_IBCSO_ETOPO1_mask75L_v2.2_r42_noclosea.nc"
ds_bathy.to_netcdf(out_file)

