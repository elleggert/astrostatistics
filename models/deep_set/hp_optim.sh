
#!/bin/bash

#for area in north south des
for area in north south des
do
  for gal in lrg elg qso
    do
      python hp_optim.py -g $gal -a $area -t 10
    done
done


