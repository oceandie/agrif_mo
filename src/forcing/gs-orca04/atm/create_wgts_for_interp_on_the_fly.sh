#!/usr/bin/bash

cmd="/data/users/dbruciaf/NEMO/v_4.2/nemo/tools/WEIGHTS/scripshape.exe"
nam_bil="namelist_bilin"
nam_bic="namelist_bicub"

source /data/users/dbruciaf/NEMO/ukmo_utils/use_vdi.sh

${cmd} ${nam_bil}
${cmd} ${nam_bic}
