#!/bin/bash

for area in des north south
do
  for gal in lrg elg qso
    do
      python hp_optim.py -g $gal  -a $area -t 60 
    done
done


