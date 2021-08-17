
#!/bin/bash

#for area in north south des
for area in north
do
  for gal in lrg elg qso
    do
      python hp_optim.py -g $gal -n 1000 -a $area -t 4
    done
done


