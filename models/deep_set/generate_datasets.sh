
#!/bin/bash

for area in north south des
  do
    python deepset_dataset_generator.py -a $area
  done
