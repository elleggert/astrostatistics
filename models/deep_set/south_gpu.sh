#!/bin/bash
#SBATCH --job-name=DeepSet
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=ele20 # required to send email notifcations - please replace <your_username> with your college login name or email address
#SBATCH --output=south_1.out
#SBATCH --error=south_1.out

export PATH=/vol/bitbucket/${USER}/myvenv/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime
for gal in lrg elg qso
    do
      python hp_optim.py -g $gal -a south -t 40
    done



