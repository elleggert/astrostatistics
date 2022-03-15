
#!/bin/bash

for area in south
do
  for gal in rlbg
    do
	python hp_optim.py -g $gal -a $area -t 20
	python fine_tune.py -g $gal -a $area -t 20
    done
done
