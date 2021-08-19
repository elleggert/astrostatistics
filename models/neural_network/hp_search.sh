#!/bin/bash

for area in north south des
do
  for gal in lrg elg qso
    do
      python hp_optim.py -g $gal  -a $area -t 10 -n 400
    done
done


