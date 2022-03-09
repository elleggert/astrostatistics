
#!/bin/bash

for area in north
do
  for gal in elg
    do
	python hp_optim.py -g $gal -a $area -t 15
	python fine_tune.py -g $gal -a $area -t 15
    done
done
