#!/bin/bash

for area in south
do
  for gal in qso qso qso qso
    do
      python final_run.py -g $gal -a $area
    done
done


