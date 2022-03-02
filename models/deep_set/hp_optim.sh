
#!/bin/bash

for area in north south des
do
  for gal in glbg
    do
      python hp_optim.py -g $gal -a $area -t 10
    done
done
