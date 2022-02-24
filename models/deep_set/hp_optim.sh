
#!/bin/bash

for area in north
do
  for gal in lrg
    do
      python hp_optim.py -g $gal -a $area -t 20
    done
done


