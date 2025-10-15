#!/usr/bin/env python

import xarray as xr
import numpy as np

def e3_to_dep(e3W, e3T, zdim, nameT, nameW):

    gdepT = xr.full_like(e3T, None, dtype=np.double).rename(nameT)
    gdepW = xr.full_like(e3W, None, dtype=np.double).rename(nameW)

    gdepW[{zdim:0}] = 0.0
    gdepT[{zdim:0}] = 0.5 * e3W[{zdim:0}]
    for k in range(1, e3W.sizes[zdim]):
        gdepW[{zdim:k}] = gdepW[{zdim:k-1}] + e3T[{zdim:k-1}]
        gdepT[{zdim:k}] = gdepT[{zdim:k-1}] + e3W[{zdim:k}]

    return tuple([gdepW, gdepT])

def compute_masks(ds_domain, zdim, merge=False):
    """
    Compute masks from domain_cfg Dataset.
    If merge=True, merge with the input dataset.
    Parameters
    ----------
    ds_domain: xr.Dataset
        domain_cfg datatset
    add: bool
        if True, merge with ds_domain
    Returns
    -------
    ds_mask: xr.Dataset
        dataset with masks
    """

    # Extract variables
    k = ds_domain[zdim] + 1
    top_level = ds_domain["top_level"]
    bottom_level = ds_domain["bottom_level"]

    # Page 27 NEMO book.
    # I think there's a typo though.
    # It should be:
    #                  | 0 if k < top_level(i, j)
    # tmask(i, j, k) = | 1 if top_level(i, j) ≤ k ≤ bottom_level(i, j)
    #                  | 0 if k > bottom_level(i, j)
    tmask = xr.where(np.logical_or(k < top_level, k > bottom_level), 0, np.nan)
    tmask = xr.where(np.logical_and(bottom_level >= k, top_level <= k), 1, tmask)
    tmask = tmask.rename("tmask")

    tmask = tmask.transpose("time_counter", zdim, "y", "x")

    # Need to shift and replace last row/colum with tmask
    # umask(i, j, k) = tmask(i, j, k) ∗ tmask(i + 1, j, k)
    umask = tmask.rolling(x=2).prod().shift(x=-1)
    umask = umask.where(umask.notnull(), tmask)
    umask = umask.rename("umask")

    # vmask(i, j, k) = tmask(i, j, k) ∗ tmask(i, j + 1, k)
    vmask = tmask.rolling(y=2).prod().shift(y=-1)
    vmask = vmask.where(vmask.notnull(), tmask)
    vmask = vmask.rename("vmask")

    # Return
    masks = xr.merge([tmask, umask, vmask])
    if merge:
        return xr.merge([ds_domain, masks])
    else:
        return masks

# =================================================================================================

#domcfg = '/data/users/diego.bruciaferri/Model_Config/GOSI/AGRIF-NAtl/gosi10na+/v2/domain_cfg.nc'
#domcfg = '/data/users/diego.bruciaferri/Model_Config/GOSI/AGRIF-NAtl/gosi10-ls_r20/closea_gosi10p3/domain_cfg.nc'
#domcfg = '/data/users/diego.bruciaferri/Model_Config/GOSI/AGRIF-NAtl/gosi10-gs_r12/closea_gosi10p3/domain_cfg.nc'
#domcfg = '/data/users/diego.bruciaferri/Model_Config/GOSI/AGRIF-NAtl/gosi10-gb_r20/closea_gosi10p3/loc_MEs/domain_cfg.nc'
domcfg = '/data/users/diego.bruciaferri/Model_Config/GOSI/AGRIF-NAtl/gosi10-gb_r20/closea_gosi10p3/zps/domain_cfg.nc'

ds_dom = xr.open_dataset(domcfg, 
                         drop_variables=['mbku',
                                         'mbkv',
                                         'mbkf',
                                         'mask_opensea',
                                         'mask_csundef',
                                         'mask_csglo',
                                         'mask_csemp',
                                         'mask_csrnf',
                                         'mask_csgrpglo',
                                         'mask_csgrpemp',
                                         'mask_csgrprnf',
                                         'e3f_0',
                                         'e3uw_0',
                                         'e3vw_0'
                         ]
         )
#ds_dom = ds_dom.rename_dims({"nav_lev":"z"})
ds_dom = ds_dom.rename_vars({"nav_lev":"z"})
ds_dom = compute_masks(ds_dom, "nav_lev", merge=True)
gdepw, gdept = e3_to_dep(ds_dom.e3w_0, ds_dom.e3t_0, "nav_lev", "gdept_0", "gdepw_0")
gdepw_1d, gdept_1d = e3_to_dep(ds_dom.e3w_1d, ds_dom.e3t_1d, "nav_lev", "gdept_1d", "gdepw_1d")
gdep = xr.merge([gdepw, gdept, gdepw_1d, gdept_1d])
ds_dom = xr.merge([ds_dom, gdep])
ds_dom = ds_dom.drop_vars(['top_level',
                           'e3t_1d',
                           'e3w_1d'
                ]
         )
ds_dom = ds_dom.rename_vars({"bottom_level":"mbathy",
                             "z":"nav_lev"
                }
         )

outdir = '/data/users/diego.bruciaferri/'
outfile = 'mesh_mask.nc'
ds_dom.to_netcdf(outdir+outfile, unlimited_dims={'time_counter':True})

