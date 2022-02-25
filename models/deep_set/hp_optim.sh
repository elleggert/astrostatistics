
#!/bin/bash

for area in south
do
  for gal in lrg
    do
      python hp_optim.py -g $gal -a $area -t 15
    done
done