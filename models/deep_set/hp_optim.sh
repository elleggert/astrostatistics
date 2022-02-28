
#!/bin/bash

for area in south
do
  for gal in elg
    do
      python hp_optim.py -g $gal -a $area -t 20
    done
done
