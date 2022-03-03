
0;95;0c#!/bin/bash

for area in north south des
do
  for gal in lrg elg qso glbg rlbg
    do
      python fine_tune.py -g $gal -a $area -t 10
    done
done
