#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import f90nml as fn90

#########################################################################
# Code to create remapping weights for NEMO on-the-fly interpolation of # 
# JRA fluxes onto an AGRIF target grid.                                 #
#                                                                       #
# Author: Diego Bruciaferri                                             #
# Date  :  21/04/2026                                                   #
#########################################################################

#
# BEGIN USER MODIFICATIONS
#

# Directory with domccfg target file
DOMCFG_DIR="/data/users/adam.blaker.ext/PROMOTE/DOMAINcfg_exp/GOSI10na+P/v3/"
# Suffix of domcfg files
RAD='domain_cfg.nc'
# Directory with original forcing on native grid
FORCING_DIR='/common/share/monsoon_ancils_nemo/SBC/JRA55/'

# List of lists containing 
# 1) forcing file name
# 2) interpolation method
# 3) stem for input grid
# 4) stem for target grid
# 5) name of lon variable in input forcing file
# 6) name of lat variable in input forcing file
# 7) name of variable in input forcing file that we want to remap (for testing)
# 8) name of time variable in input forcing file
FILES=[
['t_10_y1988.nc','bilin','jra','nor20','lon','lat','tas','time'],
['t_10_y1988.nc','bicub','jra','nor20','lon','lat','tas','time'],
]

#
# END USER MODIFICATIONS
#
mycmd="ls "+DOMCFG_DIR+"/?_"+RAD
returned_output = subprocess.check_output(mycmd, shell=True)
listcfg = (returned_output.decode("utf-8")).split()

for i in range(len(listcfg)):
    print ('Computing weights for cfg file %s :' % listcfg[i])
    print()

    for myfile in FILES:
        print('Input file is %s with %s interpolation' % (myfile[0],  myfile[1]))
        print('   Performing weights computation ...')
        if myfile[1]=='bicub':
           namelist='namelist_bicub'
        else :
           namelist='namelist_bilin'

        print(namelist)
        myfilename=FORCING_DIR+'/'+myfile[0]
        mylon=myfile[4]
        mylat=myfile[5]
        print('   Interpolation based on longitude %s ...' % mylon)
        print('   Interpolation based on latitude  %s ...' % mylat)

        nml_out = 'scrip_in'
        nml = fn90.read(namelist)

        nml['grid_inputs']['input_file'] = myfilename
        nml['grid_inputs']['nemo_file'] = listcfg[i]
        nml['grid_inputs']['datagrid_file'] = "remap_"+myfile[2]+"_grid.nc"
        nml['grid_inputs']['nemogrid_file'] = "remap_"+myfile[3]+"_grid.nc"
        nml['grid_inputs']['input_lon'] = mylon
        nml['grid_inputs']['input_lat'] = mylat

        nml['remap_inputs']['grid1_file'] = "remap_"+myfile[2]+"_grid.nc"
        nml['remap_inputs']['grid2_file'] = "remap_"+myfile[3]+"_grid.nc"
        nml['remap_inputs']['interp_file1'] = myfile[2]+"_"+myfile[3]+"_"+myfile[1]+".nc"
        nml['remap_inputs']['interp_file2'] = myfile[3]+"_"+myfile[2]+"_"+myfile[1]+".nc"
        nml['remap_inputs']['map1_name'] = myfile[2]+" to "+myfile[3]+" "+myfile[1]+" mapping"
        nml['remap_inputs']['map2_name'] = myfile[3]+" to "+myfile[2]+" "+myfile[1]+" mapping"

        nml['interp_inputs']['input_file'] = myfilename
        nml['interp_inputs']['interp_file'] = myfile[2]+"_"+myfile[3]+"_"+myfile[1]+".nc"
        nml['interp_inputs']['input_name'] = myfile[6]
        nml['interp_inputs']['input_vars'] = myfile[7]

        nml['interp_outputs']['output_file'] = myfile[3]+"_"+myfile[0]
        nml['interp_outputs']['output_name'] = myfile[6]

        nml['shape_inputs']['interp_file'] = myfile[2]+"_"+myfile[3]+"_"+myfile[1]+".nc"
        nml['shape_inputs']['output_file'] = "weights_" + myfile[2]+"_"+myfile[3]+"_"+myfile[1]+".nc"

        nml.write(nml_out)

        mycheck=subprocess.check_output('scripgrid.exe '  + nml_out, shell=True)
        mycheck=subprocess.check_output('scrip.exe'                , shell=True)
        mycheck=subprocess.check_output('scripshape.exe ' + nml_out, shell=True)
        mycheck=subprocess.check_output('scripinterp.exe ' + nml_out, shell=True)

        print('   Success ...')
        print('   => weight file is %s' % "weights_" + myfile[2]+"_"+myfile[3]+"_"+myfile[1]+".nc")

        os.rename(nml_out, "namelist_"+myfile[2]+"_"+myfile[3]+"_"+myfile[1])

    print()

