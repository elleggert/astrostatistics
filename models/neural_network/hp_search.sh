#!/bin/bash

for area in south des
do
  for gal in elg qso
    do
      python hp_optim.py -g $gal  -a $area -t 20 
    done
done


