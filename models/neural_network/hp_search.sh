#!/bin/bash

for area in des
do
  for gal in lrg elg qso glbg rlbg
    do
      python hp_optim.py -g $gal -a $area -t 25
    done
done


