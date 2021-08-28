#!/bin/bash

#for area in north south des
for area in north
do
  for gal in  lrg elg qso
    do
      python final_run.py -g $gal -a $area 
    done
done


