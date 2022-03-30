
#!/bin/bash

for area in north south des
do
  for gal in lrg
    do
	python hp_optim.py -g $gal -a $area -t 20
	python fine_tune.py -g $gal -a $area -t 20
    done
done
