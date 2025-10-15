#!/usr/bin/env python

#     |------------------------------------------------------------|
#     | Author: Diego Bruciaferri                                  |
#     | Date and place: 28-02-2022, Met Office, UK                 |
#     |------------------------------------------------------------|


import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
#import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import xarray as xr
from nsv import SectionFinder
from geopy.distance import great_circle
from typing import Tuple
import cartopy.crs as ccrs
import cartopy.feature as feature

def calc_r0(depth: xr.DataArray) -> xr.DataArray:
    """
    Calculate slope parameter r0: measure of steepness
    This function returns the slope paramater field

    r = abs(Hb - Ha) / (Ha + Hb)

    where Ha and Hb are the depths of adjacent grid cells (Mellor et al 1998).

    Reference:
    *) Mellor, Oey & Ezer, J Atm. Oce. Tech. 15(5):1122-1131, 1998.

    Parameters
    ----------
    depth: DataArray
        Bottom depth (units: m).

    Returns
    -------
    DataArray
        2D slope parameter (units: None)

    Notes
    -----
    This function uses a "conservative approach" and rmax is overestimated.
    rmax at T points is the maximum rmax estimated at any adjacent U/V point.
    """
    # Mask land
    depth = depth.where(depth > 0)

    # Loop over x and y
    both_rmax = []
    for dim in depth.dims:

        # Compute rmax
        rolled = depth.rolling({dim: 2}).construct("window_dim")
        diff = rolled.diff("window_dim").squeeze("window_dim")
        rmax = np.abs(diff) / rolled.sum("window_dim")

        # Construct dimension with velocity points adjacent to any T point
        # We need to shift as we rolled twice
        rmax = rmax.rolling({dim: 2}).construct("vel_points")
        rmax = rmax.shift({dim: -1})

        both_rmax.append(rmax)

    # Find maximum rmax at adjacent U/V points
    rmax = xr.concat(both_rmax, "vel_points")
    rmax = rmax.max("vel_points", skipna=True)

    # Mask halo points
    for dim in rmax.dims:
        rmax[{dim: [0, -1]}] = 0

    return rmax.fillna(0)

def velocity_points_along_zigzag_section(finder, lons, lats, grd='f'):
        """
        Given the coordinates defining a section, find the corrisponding velocity points
        along a zigzag section.

        Args:
            finder (obj)        : object created with nsv.SectionFinder() class
            lons (1D array-like): Longitudes defining a section
            lats (1D array-like): Latitudes defining a section
            grd (string)       : 't' or 'f'

        Returns:
            dict: Dictionary mapping u/v grids to their coordinates and indexes.
        """

        # ZigZag path along the specified grid
        ds = finder.zigzag_section(lons, lats, grd)

        # Find dimension name
        dim = list(ds.dims)[0]
        ds[dim] = ds[dim]

        # Compute diff and find max index of each pair
        ds_diff = ds.diff(dim)
        if grd == 'f':
           ds_roll = ds.rolling({dim: 2}).max().dropna(dim)
        else:
           ds_roll = ds.rolling({dim: 2}).min().dropna(dim)

        # Fill dictionary
        ds_dict = {}
        for grid in ("u", "v"):

            # Apply mask
            if grd == 'f':
               # Meridional: u - Zonal: v
               ds = ds_roll.where(
                       ds_diff[f"{'y' if grid == 'u' else 'x'}_index"], drop=True
               )
            else:
               ds = ds_roll.where(
                       ds_diff[f"{'x' if grid == 'u' else 'y'}_index"], drop=True
               )
            da_dim = ds[dim]

            if not ds.sizes[dim]:
                # Empty: either zonal or meridional
                continue

            # Find new lat/lon
            ds = finder.grids[grid].isel(
                x=xr.DataArray(ds["x_index"].astype(int), dims=dim),
                y=xr.DataArray(ds["y_index"].astype(int), dims=dim),
            )
            ds = finder.nearest_neighbor(ds["lon"], ds["lat"], grid)

            # Assign coordinate - useful to concatenate u and v after extraction
            ds_dict[grid] = ds.assign_coords({dim: da_dim - 1})

        return ds_dict

def e3_to_dep(e3W: xr.DataArray, e3T: xr.DataArray) -> Tuple[xr.DataArray, ...]:

    gdepT = xr.full_like(e3T, None, dtype=np.double).rename('gdepT')
    gdepW = xr.full_like(e3W, None, dtype=np.double).rename('gdepW')

    gdepW[{"z":0}] = 0.0
    gdepT[{"z":0}] = 0.5 * e3W[{"z":0}]
    for k in range(1, e3W.sizes["z"]):
        gdepW[{"z":k}] = gdepW[{"z":k-1}] + e3T[{"z":k-1}]
        gdepT[{"z":k}] = gdepT[{"z":k-1}] + e3W[{"z":k}]

    return tuple([gdepW, gdepT])

def compute_e3fw(ds_domain, merge=False):
    """
    Compute e3fw scale factors from domain_cfg Dataset.
    If merge=True, merge with the input dataset.
    Parameters
    ----------
    ds_domain: xr.Dataset
        domain_cfg datatset
    merge: bool
        if True, merge with ds_domain
    Returns
    -------
    ds_e3fw: xr.Dataset
        dataset with e3fw
    """

    # Extract variables
    e3w = ds_domain["e3w_0"]

    e3fw = e3w.rolling(x=2).prod().shift(x=-1)
    e3fw = e3fw.rolling(y=2).prod().shift(y=-1)
    e3fw = e3fw.where(e3fw.notnull(), e3w)
    e3fw = e3fw.rename("e3fw_0")

    # Return
    if merge:
        return xr.merge([ds_domain, e3fw])
    else:
        return e3fw


def compute_masks(ds_domain, merge=False):
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
    k = ds_domain["z"] + 1
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

    tmask = tmask.transpose("z","y","x")

    # Need to shift and replace last row/colum with tmask
    # umask(i, j, k) = tmask(i, j, k) ∗ tmask(i+1, j, k)
    umask = tmask.rolling(x=2).prod().shift(x=-1)
    umask = umask.where(umask.notnull(), tmask)
    umask = umask.rename("umask")

    # vmask(i, j, k) = tmask(i, j, k) ∗ tmask(i, j+1, k)
    vmask = tmask.rolling(y=2).prod().shift(y=-1)
    vmask = vmask.where(vmask.notnull(), tmask)
    vmask = vmask.rename("vmask")

    # Return
    masks = xr.merge([tmask, umask, vmask])
    if merge:
        return xr.merge([ds_domain, masks])
    else:
        return masks

def regrid_UV_to_T(daU, daV):
    U = daU.rolling({'x':2}).mean().fillna(0.)
    V = daV.rolling({'y':2}).mean().fillna(0.)
    return U, V

def calc_speed(daU, daV):
    if "depthu" in daU.dims:
       daU = daU.rename({"depthu": "z"})
    if "depthv" in daV.dims:
       daV = daV.rename({"depthv": "z"})
    return np.sqrt(daU**2 + daV**2)

def calc_max_vel(daU, daV):
    return np.maximum(np.nanmax(np.absolute(daU)),np.nanmax(np.absolute(daV)))

def calc_KE(daU, daV):
    if "depthu" in daU.dims:
       daU = daU.rename({"depthu": "z"})
    if "depthv" in daV.dims:
       daV = daV.rename({"depthv": "z"})
    return 0.5 * (daU**2 + daV**2)

def calc_vol_avg(da, e1, e2, e3):
    cel_vol = e1 * e2 * e3
    dom_vol = cel_vol.sum(skipna=True)
    if "z" not in da:
       cel_vol = cel_vol.sum(dim="z",skipna=True)
    return (cel_vol*da).sum(skipna=True) / dom_vol

def calc_SM03_rhd(dep):
    '''
    Shchepetkin & McWilliams (2003) initial 
    rhd profile.    
    rhd = (rho - rho0)/rho0
    '''
    rho0  = 1026.
    rn_a0 = 0.1655
    drho  = 3.0
    delta = 500.
    rhd   = -(drho/rho0)*np.exp( -dep/delta )
    return rhd

def calc_press(dep):
    drho  = 3.0
    delta = 500.
    grav  = 9.80665
    rho0  = 1026.
    p0    = (grav * drho * delta) / rho0
    press =  p0 * (np.exp(-dep/delta)-1.0) 
    return press

def calc_Fx_from_surf(dep):
    drho  = 3.0
    delta = 500.
    grav  = 9.80665
    rho0  = 1026.
    p0    = (grav * drho * delta) / rho0
    Fx    = - p0 * (dep + delta*(np.exp(-dep/delta)-1.0))
    return Fx

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def combine_arrays_alternating_columns(array1, array2):

    # Check if both arrays have the same number of rows
    if array1.shape[0] != array2.shape[0]:
        raise ValueError("Both arrays must have the same number of rows")

    # Check if both arrays have the same number of columns
    eqcol = False
    if array1.shape[1] == array2.shape[1]: eqcol = True
       
    # Get the number of rows and columns
    rows, cols = array1.shape
    if eqcol:
       cols = cols * 2
    else:
       cols = (max(array1.shape[1], array2.shape[1]) * 2) - 1

    # Initialize the combined array with the appropriate shape
    combined_array = np.empty((rows, cols), dtype=array1.dtype)

    # Fill the combined array by interleaving columns from array1 and array2
    combined_array[:, ::2] = array1
    if eqcol:
       combined_array[:, 1::2] = array2
    else:
       combined_array[:, 1:-1:2] = array2

    return combined_array

def extract_section_from_array(ds_domcfg, ds_scalar, vvar, target_lons, target_lats):

    upnt = False
    vpnt = False

    finder = SectionFinder(ds_domcfg)
    pnts_UV = finder.velocity_points_along_zigzag_section(
                                    lons=target_lons,
                                    lats=target_lats,
              )
    if "u" in pnts_UV: upnt = True
    if "v" in pnts_UV: vpnt = True

    pnts_F = finder.zigzag_section(lons=target_lons,
                                   lats=target_lats,
                                   grid='f'
             )

    # a) scalar variable interpolated at U- and V-points
    if upnt:
       var3D_TU = ds_scalar[vvar].rolling(x=2).mean().shift(x=-1)
       var2D_TU = var3D_TU.isel(x=pnts_UV['u'].x_index,
                                y=pnts_UV['u'].y_index
                               )
    if vpnt:
       var3D_TV = ds_scalar[vvar].rolling(y=2).mean().shift(y=-1)
       var2D_TV = var3D_TV.isel(x=pnts_UV['v'].x_index,
                                y=pnts_UV['v'].y_index
                               )
    if upnt and vpnt:
       var2D_TUV = xr.concat([var2D_TU, var2D_TV], dim="dim_0")
    elif upnt and not vpnt:
       var2D_TUV = var2D_TU.copy()
    elif not upnt and vpnt:
       var2D_TUV = var2D_TV.copy()  

    var2D_TUV = var2D_TUV.reindex({'dim_0':sorted(var2D_TUV.dim_0)})
    var2D_F = ds_scalar[vvar].isel(x=pnts_F.x_index,y=pnts_F.y_index)*0.0

    var2D = combine_arrays_alternating_columns(var2D_F, var2D_TUV)
  
    # b) depths of T- and W-levels
    gdepf  = ds_domcfg["gdepf_0"].isel(x=pnts_F.x_index, y=pnts_F.y_index)
    gdepfw = ds_domcfg["gdepfw_0"].isel(x=pnts_F.x_index, y=pnts_F.y_index)

    if upnt:
       gdepu  = ds_domcfg["gdepu_0"].isel(x=pnts_UV['u'].x_index,
                                          y=pnts_UV['u'].y_index
                                         )
       gdepuw = ds_domcfg["gdepuw_0"].isel(x=pnts_UV['u'].x_index,
                                           y=pnts_UV['u'].y_index
                                          )
    if vpnt:
       gdepv  = ds_domcfg["gdepv_0"].isel(x=pnts_UV['v'].x_index,
                                          y=pnts_UV['v'].y_index
                                         )
       gdepvw = ds_domcfg["gdepvw_0"].isel(x=pnts_UV['v'].x_index,
                                           y=pnts_UV['v'].y_index
                                          )
    
    if upnt and vpnt:
       gdepuv = xr.concat([gdepu, gdepv], dim="dim_0")
       gdepuvw = xr.concat([gdepuw, gdepvw], dim="dim_0")
    elif upnt and not vpnt:
       gdepuv = gdepu.copy()
       gdepuvw = gdepuw.copy()
    elif not upnt and vpnt:
       gdepuv = gdepv.copy()
       gdepuvw = gdepvw.copy()

    gdepuv = gdepuv.reindex({'dim_0':sorted(gdepuv.dim_0)})
    gdepuvw = gdepuvw.reindex({'dim_0':sorted(gdepuvw.dim_0)})

    gdepcw = combine_arrays_alternating_columns(gdepfw, gdepuvw)
    gdepc  = combine_arrays_alternating_columns(gdepf, gdepuv)

    gdepw_1d = ds_domcfg["gdepw_1d"]

    # c) distance along the section
    glamf = ds_domcfg["glamf"].isel(x=pnts_F.x_index, y=pnts_F.y_index)
    gphif = ds_domcfg["gphif"].isel(x=pnts_F.x_index, y=pnts_F.y_index)

    if upnt:
       glamu = ds_domcfg["glamu"].isel(x=pnts_UV['u'].x_index,
                                       y=pnts_UV['u'].y_index
                                      )
       gphiu = ds_domcfg["gphiu"].isel(x=pnts_UV['u'].x_index,
                                       y=pnts_UV['u'].y_index
                                      )
    if vpnt:
       glamv = ds_domcfg["glamv"].isel(x=pnts_UV['v'].x_index,
                                       y=pnts_UV['v'].y_index
                                      )
       gphiv = ds_domcfg["gphiv"].isel(x=pnts_UV['v'].x_index,
                                       y=pnts_UV['v'].y_index
                                      )
    if upnt and vpnt:
       glamuv = xr.concat([glamu, glamv], dim="dim_0")
       gphiuv = xr.concat([gphiu, gphiv], dim="dim_0")
    elif upnt and not vpnt:
       glamuv = glamu.copy()
       gphiuv = gphiu.copy()
    elif not upnt and vpnt:
       glamuv = glamv.copy()
       gphiuv = gphiv.copy()

    glamuv = glamuv.reindex({'dim_0':sorted(glamuv.dim_0)})
    gphiuv = gphiuv.reindex({'dim_0':sorted(gphiuv.dim_0)})

    min_length = min(len(glamf), len(glamuv))
    glamc = [val for pair in zip(glamf, glamuv) for val in pair]
    glamc.extend(glamf[min_length:])
    glamc.extend(glamuv[min_length:])

    gphic = [val for pair in zip(gphif, gphiuv) for val in pair]
    gphic.extend(gphif[min_length:])
    gphic.extend(gphiuv[min_length:])

    distc = [0]
    for i in range(1,len(glamc)):
        coords = ((gphic[ind], glamc[ind]) for ind in (i - 1, i))
        distc.append(distc[-1] + great_circle(*coords).km)
    distc = xr.DataArray(
         distc,
         dims={'dim_0':range(len(distc))},
         attrs={"long_name": "distance along transect", "units": "km"},
    )
    distc = distc.expand_dims({"z": gdepuv.shape[0]})

    # e) localisation mask if needed
    if "loc_msk" in ds_domcfg.variables:
       loc_msk_F = ds_domcfg["loc_msk"].isel(x=pnts_F.x_index, y=pnts_F.y_index)*0.0
       if upnt:
          loc_msk_U = ds_domcfg["loc_msk"].rolling(x=2).mean().shift(x=-1)
          loc_msk_U = loc_msk_U.where(loc_msk_U==0.,1.)
          loc_msk_U = loc_msk_U.isel(x=pnts_UV['u'].x_index,
                                     y=pnts_UV['u'].y_index
                                    )
       if vpnt:
          loc_msk_V = ds_domcfg["loc_msk"].rolling(y=2).mean().shift(y=-1)
          loc_msk_V = loc_msk_V.where(loc_msk_V==0.,1.)
          loc_msk_V = loc_msk_V.isel(x=pnts_UV['v'].x_index,
                                     y=pnts_UV['v'].y_index
                                    )
       if upnt and vpnt:
          loc_msk_UV = xr.concat([loc_msk_U, loc_msk_V], dim="dim_0")
       elif upnt and not vpnt:
          loc_msk_UV = loc_msk_U.copy()
       elif not upnt and vpnt:
          loc_msk_UV = loc_msk_V.copy()

       loc_msk_UV = loc_msk_UV.reindex({'dim_0':sorted(loc_msk_UV.dim_0)})
       
       min_length = min(len(loc_msk_F), len(loc_msk_UV))
       loc_msk = [val for pair in zip(loc_msk_F, loc_msk_UV) for val in pair]
       loc_msk.extend(loc_msk_F[min_length:])
       loc_msk.extend(loc_msk_UV[min_length:])

    # g) envelopes if needed
    # TODO
    hbatt = []
    #if len(envelopes) > 0:
    #   for e in range(len(envelopes)): hbatt.append(envelopes[e].isel({vcor:indx}).values)

    return var2D, distc, gdepcw, gdepc, gdepw_1d, loc_msk, hbatt

def prepare_domcfg(domcfg,fbathy=None):

    # Loading domain geometry
    ds_dom = xr.open_dataset(domcfg, drop_variables=("x", "y","nav_lev")).squeeze()
    ds_dom = ds_dom.rename({"nav_lev": "z"})
    # Computing land-sea masks
    ds_dom = compute_masks(ds_dom, merge=True)
    # cCompute e3fw scale factors
    ds_dom = compute_e3fw(ds_dom, merge=True)

    # Computing depths
    ds_dom["gdepw_0"], ds_dom["gdept_0"]  = e3_to_dep(ds_dom.e3w_0, ds_dom.e3t_0)
    ds_dom["gdepuw_0"], ds_dom["gdepu_0"] = e3_to_dep(ds_dom.e3uw_0, ds_dom.e3u_0)
    ds_dom["gdepvw_0"], ds_dom["gdepv_0"] = e3_to_dep(ds_dom.e3vw_0, ds_dom.e3v_0)
    ds_dom["gdepfw_0"], ds_dom["gdepf_0"] = e3_to_dep(ds_dom.e3fw_0, ds_dom.e3f_0)
    ds_dom["gdepw_1d"], ds_dom["gdept_1d"] = e3_to_dep(ds_dom.e3w_1d, ds_dom.e3t_1d)

    # Determining the type of vertical coordinate
    for vcr in ["zco", "zps", "sco"]:
        if ds_dom["ln_"+vcr] == 1: vcoor = vcr

    # Checking if localisation has been applied
    ds_dom["loc_msk"] = ds_dom.bathy_metry * 0.
    if fbathy:
       ds_bat  = xr.open_dataset(fbathy, drop_variables=("x", "y")).squeeze()
       ds_dom["loc_msk"].values = ds_bat.s2z_msk.values
    else:
       if vcoor == "sco":
          ds_dom["loc_msk"][:,:] = 1.
       else:
          ds_dom["loc_msk"][:,:] = 0.

    # Checking if we are using a ME system
    hbatt = []
    if vcoor == "sco" and fbathy:
       ds_bat  = xr.open_dataset(fbathy, drop_variables=("x", "y")).squeeze()
       nenv = 1
       while nenv > 0:
         name_env = "hbatt_"+str(nenv)
         if name_env in ds_bat.variables:
            envlp = ds_bat[name_env]
            envlp = envlp.where(ds_dom.loc_msk > 1)
            hbatt.append(envlp)
            nenv+=1
         else:
            nenv=0

    return ds_dom, hbatt, vcoor

def plot_sec(fig_name, fig_path, ds_domcfg, ds_scalar, vvar, target_lons, target_lats, envelopes=[], imap=False):

    # Extracting arrays of the section
    var2D, distc, gdepcw, gdepc, gdepw_1d, loc_msk, hbatt = extract_section_from_array(
                          ds_domcfg, 
                          ds_scalar, 
                          vvar, 
                          target_lons, 
                          target_lats
    )
 
    # Plotting ----------------------------------------------------------

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(54, 34), dpi=200)
    plt.sca(ax) # Set the current Axes to ax and the current Figure to the parent of ax.
    ax.invert_yaxis()
    ax.set_facecolor('saddlebrown')
    #ax.set_facecolor('black')

    # SCALAR VARIABLE -----------------------------------------------
    # We create a polygon patch for each U- V- cell of the section and 
    # we colour it based on the magnitude of the scalar variable
    #
    ni = var2D.shape[1]
    nk = var2D.shape[0]
    patches = []
    colors = []
    depw = np.zeros((nk,ni))

    for k in range(0,nk-1):
        for i in range(1, ni-1, 2):
            x = [distc[k  , i-1], # F_(k,i-1)
                 distc[k  , i  ], # UV_(k,i)
                 distc[k  , i+1], # F_(k,i+1)
                 distc[k+1, i+1], # F_(k+1,i+1)
                 distc[k+1, i  ], # UV_(k+1,i)
                 distc[k+1, i-1], # F_(k+1,i)
                 distc[k  , i-1]] # F_(k  ,i)

            if loc_msk[i] == 0:
               #y = [np.max([gdepcw[k  ,i-1], gdepcw[k  ,i  ]]),
               #     gdepcw[k  ,i  ],
               #     np.max([gdepcw[k  ,i  ], gdepcw[k  ,i+1]]),
               #     np.max([gdepcw[k+1,i  ], gdepcw[k+1,i+1]]),
               #     gdepcw[k+1,i  ],
               #     np.max([gdepcw[k+1,i-1], gdepcw[k+1,i  ]]),
               #     np.max([gdepcw[k  ,i-1], gdepcw[k  ,i  ]])]
               y = [gdepcw[k  ,i  ],
                    gdepcw[k  ,i  ],
                    gdepcw[k  ,i  ],
                    gdepcw[k+1,i  ],
                    gdepcw[k+1,i  ],
                    gdepcw[k+1,i  ],
                    gdepcw[k  ,i  ]]
               depw[k,i-1:i+2] = gdepw_1d[k]
            else:
                y = [gdepcw[k  ,i-1],
                     gdepcw[k  ,i  ],
                     gdepcw[k  ,i+1],
                     gdepcw[k+1,i+1],
                     gdepcw[k+1,i  ],
                     gdepcw[k+1,i-1],
                     gdepcw[k  ,i-1]]
                depw[k,i-1:i+2] = gdepcw[k,i-1:i+2]

            polygon = Polygon(np.vstack((x,y)).T, closed=True)
            patches.append(polygon)
            colors = np.append(colors,var2D[k,i])
 
    # MODEL W-levels and U-points ----------------------------
    for k in range(nk):
        x = distc[k,:]
        #z = gdepcw[k,:]
        z = depw[k,:]
        ax.plot(
            x,
            z,
            color="k",
            linewidth=0.5,
            zorder=5
        )

    # MODEL ENVELOPES
    if len(hbatt) > 0:
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

    # MODEL T-points ----------------------------
    ax.scatter(np.ravel(distc[:-1,1:-1:2]),
               np.ravel(gdepc[:-1,1:-1:2]),
               s=1,
               color='black',
               zorder=5
              )
    # PLOT setting ----------------------------
    ax.set_ylim(np.amax(gdepcw), 0.0)
    ax.set_xlim(distc[0,1], distc[0,-1])

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
    plt.xlabel('Domain extension [$km$]', fontsize=50.)
    plt.yticks(fontsize=50.)
    plt.ylabel('Depth [m]' ,fontsize=50.)

    cbaxes = inset_axes(ax, width="22%", height="3.5%", loc=4, borderpad=15.)
    cb = plt.colorbar(p, cax=cbaxes, ticks=[0., 0.1, 0.2],
                      orientation='horizontal', extend="both",drawedges=True)
    cb.set_label("Slope Parameter", size=50, color='w')
    cb.ax.tick_params(labelsize=50, labelcolor='w')
    cb.outline.set_color('white')
    cb.outline.set_linewidth(4)
    cb.dividers.set_color('black')
    cb.dividers.set_linewidth(2)

    # Inset map
    if imap:
       transform = ccrs.PlateCarree()
       proj = ccrs.Mercator()
       pos1 = ax.get_position()

       lon0 = np.nanmin(target_lons)
       lon1 = np.nanmax(target_lons)
       lat0 = np.nanmin(target_lats)
       lat1 = np.nanmax(target_lats)

       if np.absolute(lon0-lon1) < 1. or np.absolute(lat0-lat1) < 1.:
          stp = 5.
       else:
          stp = 2.
       lon0 += -stp
       lon1 +=  stp
       lat0 += -stp
       lat1 +=  stp
       map_lims = [lon0, lon1, lat0, lat1]
       a = plt.axes([pos1.x0+0.005, pos1.y0+0.005, .2, .2], projection=proj)
       a.coastlines()
       a.add_feature(feature.LAND, color='gray',edgecolor='gray',zorder=1)
       MAP = plt.plot(target_lons, target_lats, c="red", transform=transform)
       plt.setp(MAP, 'linewidth', 5.)
       a.set_extent(map_lims)
       a.patch.set_edgecolor('black')  
       a.patch.set_linewidth(4.5) 

    plt.savefig(fig_path+fig_name, bbox_inches="tight")
    plt.close()

