#!/usr/bin/env python3

########################################################################
# Code to remap ancillary fields defined on the native NEMO parent grid#
# onto an AGRIF target grid.                                           #
#                                                                      #
# Author: Diego Bruciaferri                                            #
# Date  :  21/04/2026                                                  #
########################################################################

import os
import subprocess
import f90nml as fn90
from sea_over_land import apply_seaoverland

#
# BEGIN USER MODIFICATIONS
#

# Path of GOSI10p3-eORCA025 native domain_cfg.nc input file
inp_domcfg='/common/share/monsoon_ancils_nemo/GOSIdev/GOSI10p3/eORCA025/mesh_mask_MEs_novf_4env_2930_r12_r16-r075-r040-r035_it2-r030.nc'
# Path of AGRIF domain_cfg.nc target file
trg_domcfg="/data/users/adam.blaker.ext/PROMOTE/DOMAINcfg_exp/GOSI10na+P/v3/1_domain_cfg.nc"
# Stem for input and target models
inp_stem='gosi025'
trg_stem='nor20'

# Directory with original forcing on native eORCA025 grid
inp_dir='/common/share/monsoon_ancils_nemo/GOSIdev/GOSI10p3/eORCA025/'

niter=10

# Forcing file name, list of the name of the variables to be remapped, name of the time variable  
FILES=[ [ 'new_chlorophyll_min0p01_ORCA025_r42.nc',
          ['CHL'],
          'time_counter'
        ],
        [ 'geothermal_heating_orca025ext_extrap40_r42.nc',
          ['heatflow'],
          'time'
        ],
        [ 'sss_1m_TEOS10_woa13v2.nc',
          ['so_abs'],
          'time'
        ],
        [ 'zdfiwm_forcing_eORCA025_r42.nc',
          ['power_bot','power_cri','power_nsq','power_sho','scale_cri','scale_bot','bathymetry'],
          None
        ]
]

#
# END USER MODIFICATIONS
#

# 1) Create remapping weights from GOSI10p3 to AGRIF zoom

namelist='namelist_bilin'
nml_out = 'scrip_in'

if not os.path.isfile(inp_stem+"_"+trg_stem+"_bilin.nc"):

   print('Computing weights for cfg file %s :' % trg_domcfg)
   print('Input file is %s with %s interpolation' % (inp_domcfg,  'bilinear'))
   print('   Performing weights computation ...')
   print('   Interpolation based on longitude %s ...' % 'nav_lon')
   print('   Interpolation based on latitude  %s ...' % 'nav_lat')

   nml = fn90.read(namelist)

   nml['grid_inputs']['input_file'] = inp_domcfg
   nml['grid_inputs']['nemo_file'] = trg_domcfg
   nml['grid_inputs']['datagrid_file'] = "remap_"+inp_stem+"_grid.nc"
   nml['grid_inputs']['nemogrid_file'] = "remap_"+trg_stem+"_grid.nc"
   nml['grid_inputs']['input_lon'] = "nav_lon"
   nml['grid_inputs']['input_lat'] = "nav_lat"

   nml['remap_inputs']['grid1_file'] = "remap_"+inp_stem+"_grid.nc"
   nml['remap_inputs']['grid2_file'] = "remap_"+trg_stem+"_grid.nc"
   nml['remap_inputs']['interp_file1'] = inp_stem+"_"+trg_stem+"_bilin.nc"
   nml['remap_inputs']['interp_file2'] = trg_stem+"_"+inp_stem+"_bilin.nc"
   nml['remap_inputs']['map1_name'] = inp_stem+" to "+trg_stem+" bilinear mapping"
   nml['remap_inputs']['map2_name'] = trg_stem+" to "+inp_stem+" bilinear mapping"

   nml.write(nml_out)

   mycheck=subprocess.check_output('scripgrid.exe '  + nml_out, shell=True)
   mycheck=subprocess.check_output('scrip.exe'                , shell=True)
   print('   Success scrip.exe ...')

# 2) Remap necessary ancilllary files

for ancil in FILES:

    inp_file = inp_dir + ancil[0]
    inp_fvar = ancil[1]
    inp_tvar = ancil[2]
    out_file = "seaoverland_"+ancil[0]

    apply_seaoverland(inp_file, inp_domcfg, niter, inp_fvar, out_file)

    for n in range(len(inp_fvar)):

        var = inp_fvar[n]
        nml_out = 'nml_interp_'+var

        nml = fn90.read(namelist)

        nml['interp_inputs']['input_file'] = out_file
        nml['interp_inputs']['interp_file'] = inp_stem+"_"+trg_stem+"_bilin.nc"
        nml['interp_inputs']['input_name'] = var

        nml['interp_outputs']['output_file'] = trg_stem+"_"+var+"_"+ancil[0]
        nml['interp_outputs']['output_name'] = var

        if inp_tvar:
           nml['interp_inputs']['input_vars'] = inp_tvar
        else:
           nml['interp_inputs']['input_vars'] = None
           nml['interp_outputs']['output_dims'] = ['x', 'y']
           nml['interp_outputs']['output_vars'] = None
           nml['interp_outputs']['output_attributes'] = None

        nml.write(nml_out)

        mycheck=subprocess.check_output('scripinterp.exe ' + nml_out, shell=True)
        if n == 0:
           outname = trg_stem+"_"+ancil[0]
           subprocess.check_output("cp "+trg_stem+"_"+var+"_"+ancil[0]+" "+outname, shell=True)
        else:
           subprocess.check_output("ncks -h -A "+trg_stem+"_"+var+"_"+ancil[0]+" "+outname, shell=True)
        os.remove(trg_stem+"_"+var+"_"+ancil[0])

    print('   Success scripinterp.exe ...')
