
#!/bin/bash

for area in north
do
  for gal in  glbg
    do
      python hp_optim.py -g $gal -a $area -t 1
    done
done


