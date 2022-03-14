
#!/bin/bash

for area in south north des
do
  for gal in rlbg
    do
	python hp_optim.py -g $gal -a $area -t 15
	python fine_tune.py -g $gal -a $area -t 15
    done
done
