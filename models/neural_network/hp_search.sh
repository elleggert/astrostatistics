#!/bin/bash

for area in north
do
  for gal in qso
    do
      python hp_optim.py -g $gal -a $area -n 1000 -t 10
    done
done


