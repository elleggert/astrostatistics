
#!/bin/bash

for area in north south des
do
  for gal in lrg elg qso
    do
      python hp_optim.py -g $gal -n 400 -a $area -t 10
    done
done


