import os
from desitarget.cuts import select_targets
import pandas as pd


filenames = []

for filename in os.listdir(f'../../bricks_data/tractor/'):
    if '.fits' not in filename:
        continue
    filenames.append(f'../../bricks_data/tractor/{filename}')

print(filenames)


res = select_targets(
    infiles=filenames, numproc=1, qso_selection='colorcuts', nside=None, gaiasub=False,
    tcnames=['LRG', 'ELG', 'QSO'], backup=False)


df = pd.DataFrame(data=res)

print(df.head())