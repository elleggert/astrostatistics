#!/bin/bash

#for area in north south des
for area in south
do
  for gal in elg qso
    do
      python hp_optim.py -g $gal -a $area -t 15
    done
done


