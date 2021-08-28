#!/bin/bash

for area in north
do
  for gal in lrg elg qso
    do
      python final_run.py -g $gal -a $area -n 400
    done
done


