#!/bin/bash

#for area in north south des
for area in north
do
  for gal in elg
    do
      python hp_optim.py -g $gal -a $area -t 30
    done
done


