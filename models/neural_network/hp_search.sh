#!/bin/bash

for area in north south elg
do
  for gal in lrg elg qso glbg rlbg
    do
      python final_run.py -g $gal -a $area
    done
done


