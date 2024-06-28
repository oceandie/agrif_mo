#!/bin/bash -l

#SBATCH --qos=normal                
#SBATCH --mem=100000                    
#SBATCH --ntasks=8                   
#SBATCH --output=gen_env.out       
#SBATCH --time=360                    

conda activate pyogcm

export MPLBACKEND='Agg'
python plot_vlevels_MEs.py
#python plot_vlevels_zps.py
unset MPLBACKEND

