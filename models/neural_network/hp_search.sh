#!/bin/bash

for area in south
do
  for gal in qso 
    do
      python final_run.py -g $gal -a $area
    done
done


