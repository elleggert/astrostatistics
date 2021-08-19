#!/bin/bash

#for area in north south des
for area in south
do
  for gal in qso
    do
      python hp_optim.py -g $gal -a $area -t 30
    done
done


