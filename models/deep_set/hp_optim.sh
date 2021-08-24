#!/bin/bash

#for area in north south des
for area in south
do
  for gal in elg qso lrg
    do
      python hp_optim.py -g $gal -a $area -t 50
    done
done


