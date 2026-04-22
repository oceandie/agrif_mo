#!/usr/bin/env python

#################################################
# Code to extrapolate sea values onto the land. #
#                                               #
# Author: Diego Bruciaferri                     #
# Date  :  21/04/2026                           #
#################################################


import numpy as np
import xarray as xr
import sys

def eoshift(array,shift,boundary,dim):

    a = np.roll(array, shift, axis=dim)
    if dim == 1:
       b = np.reshape(boundary,(boundary.shape[0],1))
    else:
       b = boundary

    if shift > 0:
       if dim == 0:
          a[0:shift,:] = b
       if dim == 1:
          a[:,0:shift] = b
    elif  shift < 0:
       if dim == 0:
          a[shift:,:] = b
       if dim == 1:
          a[:,shift:] = b

    return a

def seaoverland(c):
    """
    c = 2D field with np.nan land values
    """
    a = np.copy(c)
    b = np.copy(c)

    [nj, ni] = a.shape

    mat8 = eoshift(a   ,  1, boundary = a[:,0]      , dim = 1)
    mat1 = eoshift(mat8,  1, boundary = mat8[0,:]   , dim = 0)
    mat2 = eoshift(a   ,  1, boundary = a[0,:]      , dim = 0)
    mat4 = eoshift(a   , -1, boundary = a[:,ni-1]   , dim = 1)
    mat3 = eoshift(mat4,  1, boundary = mat4[0,:]   , dim = 0)
    mat5 = eoshift(mat4, -1, boundary = mat4[nj-1,:], dim = 0)
    mat6 = eoshift(a   , -1, boundary = a[nj-1,:]   , dim = 0)
    mat7 = eoshift(mat8, -1, boundary = mat8[nj-1,:], dim = 0)

    S = np.array([mat1, mat2, mat3, mat4, mat5, mat6, mat7, mat8])

    mask_logic = np.isnan(a) # TRUE for land point

    try:
       SS = np.nanmean(S, axis=0)
    except AttributeError: # added this fix to make it compatible with numpy < 1.8
       from scipy.stats import nanmean
       SS = nanmean(S, axis=0)

    a[mask_logic] = SS[mask_logic]
    a[~mask_logic] = b[~mask_logic]

    return a

def apply_seaoverland(inp_file, msh_file, niter, inpvars, out_file):

   nvar = len(inpvars)
   ds_var = xr.open_dataset(inp_file, decode_times=False).squeeze()

   ds_msh = xr.open_dataset(msh_file, decode_times=False).squeeze()
   da_msk = ds_msh["tmask"]
   msk    = da_msk.data

   nk = msk.shape[0]
   nj = msk.shape[1]
   ni = msk.shape[2]

   for nn in range(nvar):
       inp_var = inpvars[nn]
       da_var  = ds_var[inp_var]
       var     = da_var.data

       ds_seaol  = ds_var.copy()
       seaol_var = ds_seaol[inp_var].data

       if niter > 0:

          print('*** SEA-OVER-LAND ***')
          print('var.shape: ', var.shape)
          print('msk.shape: ', msk.shape)

          # var is 4D
          if len(var.shape) == 4:
             nt = var.shape[0]
             for t in range(nt):
                 var3 = np.copy(var[t,:,:,:])
                 var3[msk==0] = np.nan
                 for k in range(nk):
                     var2 = np.copy(var3[k,:,:])
                     var2 = seaoverland(var2)
                     for i in range(niter-1):
                         var2 = seaoverland(var2)
                     seaol_var[t,k,:,:] = var2

          # var is 3D
          elif len(var.shape) == 3:
             if var.shape[0] == nk:
                var[msk==0] = np.nan
                for k in range(nk):
                    var2 = np.copy(var[k,:,:])
                    var2 = seaoverland(var2)
                    for i in range(niter-1):
                        var2 = seaoverland(var2)
                    seaol_var[k,:,:] = var2
             else:
                nt = var.shape[0]
                for t in range(nt):
                    var2 = np.copy(var[t,:,:])
                    var2 = seaoverland(var2)
                    for i in range(niter-1):
                        var2 = seaoverland(var2)
                    seaol_var[t,:,:] = var2

          # var is 2D
          elif len(var.shape) == 2:
             msk2 = msk[0,:,:]
             var[msk2==0] = np.nan
             var2 = seaoverland(var)
             for i in range(niter-1):
                 var2 = seaoverland(var2)
             seaol_var[:,:] = var2

          # Removing NaNs
          seaol_var[np.isnan(seaol_var)] = 0.

   # WRITE out tracer file after sea-over-land

   ds_seaol[inp_var].data = seaol_var
   enc = {}
   for i in ds_seaol.coords: enc[i] = {"_FillValue": None }
   for i in ds_seaol.data_vars: enc[i] = {"_FillValue": None }

   ds_seaol.to_netcdf(out_file, encoding=enc)
