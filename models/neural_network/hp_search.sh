#!/bin/bash

for area in des
do
  for gal in lrg
    do
      python hp_optim.py -g $gal  -a $area -t 30 
    done
done


