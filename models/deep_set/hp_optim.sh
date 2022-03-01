
#!/bin/bash

for area in  des
do
  for gal in lrg elg qso glbg rlbg
    do
      python final_run.py -g $gal -a $area
    done
done
