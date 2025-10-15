#!/usr/bin/bash

cmd="/data/users/dbruciaf/NEMO/v_4.2/nemo/tools/WEIGHTS/scripshape.exe"
#nam_bil="gosi10na+/namelist_bilin_gs_r12"
#nam_bic="gosi10na+/namelist_bicub_gs_r12"
nam_bil="gosi10na+/namelist_bilin_ls_r20"
nam_bic="gosi10na+/namelist_bicub_ls_r20"
#nam_bil="gosi10na+/namelist_bilin_gb_r20"
#nam_bic="gosi10na+/namelist_bicub_gb_r20"

source /data/users/dbruciaf/NEMO/ukmo_utils/use_vdi.sh

${cmd} ${nam_bil}
${cmd} ${nam_bic}
