
#!/bin/bash

for area in north south des
  do
    python deepset_dataset_creator.py -a $area
  done
