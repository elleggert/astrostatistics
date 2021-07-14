from astropy.io import fits
import os
import wget
import numpy as np
import time
import pandas as pd
from brick import Brick

# 2772p520 Last exported
# 2776p527 Last downloaded

area = 'north'
device = 'Astrodisk'
bricks_to_classify = 5

hdulistBricksSouthSummary = fits.open('../../bricks_data/survey-bricks-dr9-south.fits')
data_south = hdulistBricksSouthSummary[1].data
brickname_south = data_south.field('brickname')
brickid_south = data_south.field('brickid')
south_survey_is_south = data_south.field('survey_primary')

hdulistBricksNorthSummary = fits.open('../../bricks_data/survey-bricks-dr9-north.fits')
data_north = hdulistBricksNorthSummary[1].data
brickname_north = data_north.field('brickname')
brickid_north = data_north.field('brickid')
survey_north = data_north.field('survey_primary')
north_survey_is_south = np.invert(survey_north)

start = time.time()

print()
print(f"=============================== Download {area} ..... ==================================")
print()

hdulistBricks = fits.open(f'../../bricks_data/survey-bricks-dr9-{area}.fits')
data = hdulistBricks[1].data

bricknames = list(data.field('brickname'))

downloaded_bricks = []

# Getting already downloaded files:
for filename in os.listdir(f'/Volumes/{device}/bricks_data/{area}/'):
    brickn = filename.replace("tractor-", "")
    brickn = brickn.replace(".fits", "")
    downloaded_bricks.append(brickn)

# Getting a random sample of bricknames without replacement and deleting all that are already downloaded
bricknames_sample = [x for x in bricknames if x not in downloaded_bricks]
df = pd.read_csv(f'../../bricks_data/galaxy_catalogue_{area}.csv',
                 dtype={'BrickID': 'int32', 'LRG': 'int8', 'ELG': 'int8', 'QSO': 'int8'})

brickids_processed = list(df.BrickID.unique())

bricknames_processed = []
for i, id in enumerate(brickids_processed):
    bricknames_processed.append(brickname_north[np.where(brickid_north == id)][0])

# Deleted all those already downloaded
bricknames_sample = [x for x in bricknames_sample if x not in bricknames_processed]

df_galaxy = pd.DataFrame(columns=['BrickID', 'RA', 'DEC', 'LRG', 'ELG', 'QSO'])
df_stars = pd.DataFrame(columns=['RA', 'DEC', 'GMAG', 'RMAG', 'ZMAG'])

print(df_galaxy.head())
print(df_stars.head())

print(f"No of bricks left for area {area}: {len(bricknames_sample)} ")

for i, brickname in enumerate(bricknames_sample):
    folder = brickname[:3]
    url = f'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/{area}/tractor/{folder}/tractor-{brickname}.fits'
    wget.download(url, f'/Volumes/{device}/bricks_data/{area}/')

    # brickid = brickid_south[np.where(brickname_south == brickname)]

    # North Bricks
    brickid = brickid_north[np.where(brickname_north == brickname)]

    if len(brickid > 0):
        brickid = brickid[0]
    else:
        brickid = 0

    hdu = fits.open(f'/Volumes/{device}/bricks_data/{area}/tractor-{brickname}.fits')
    data = hdu[1].data
    brick = Brick(data)

    south = north_survey_is_south[np.where(brickid_north == brickid)]

    # south = south_survey_is_south[np.where(brickid_south == brickid)]
    if len(south) > 0:
        south = south[0]
    else:
        south = True

    ## Enable this is classifying North Objects
    # south = north_survey_is_south[np.where(brickid_north == brickid)][0]

    brick.initialise_brick_for_galaxy_classification(south)
    target_objects = brick.classify_galaxies()

    # Appending one empty line per brick to be sure that all bricks are extracted
    df_galaxy = df_galaxy.append({'BrickID': brickid, 'RA': np.nan, 'DEC': np.nan, 'LRG': 0, 'ELG': 0, 'QSO': 0},
                                 ignore_index=True)

    support_df = pd.DataFrame(target_objects,
                              columns=['BrickID', 'RA', 'DEC', 'LRG', 'ELG', 'QSO'])

    df_galaxy = df_galaxy.append(support_df)

    brick.initialise_brick_for_stellar_density()

    stars = brick.get_stellar_objects()

    support_df = pd.DataFrame(stars, columns=['RA', 'DEC', 'GMAG', 'RMAG', 'ZMAG'])
    df_stars = df_stars.append(support_df)


    if i % 488 == 0:
        print()
        print(i / 488, '%')
        df_galaxy = df_galaxy.astype(
            {'BrickID': 'int32', 'LRG': 'int8', 'ELG': 'int8', 'QSO': 'int8'})
        df_galaxy.to_csv(f'../../bricks_data/galaxy_catalogue_{area}.csv', mode='a', index=False, header=False)
        df_stars.to_csv(f'../../bricks_data/stellar_catalogue_{area}.csv', mode='a', index=False, header=False)
        # df_galaxy.to_csv('../../bricks_data/galaxy_catalogue_sample_profiling.csv', index=False, header=False)
        # df_stars.to_csv('../../bricks_data/stellar_catalogue_sample_profiling.csv', index=False, header=False)
        df_galaxy = df_galaxy[0:0]
        df_stars = df_stars[0:0]

    # Remove Downloaded Brick
    os.remove(f'/Volumes/{device}/bricks_data/{area}/tractor-{brickname}.fits')

    if i > bricks_to_classify:
        break

    print(f" Brick {area} processed: ", brickname, ", Brick ", i, " of ", len(bricknames_sample))



df_galaxy = df_galaxy.astype(
            {'BrickID': 'int32', 'LRG': 'int8', 'ELG': 'int8', 'QSO': 'int8'})
df_galaxy.to_csv(f'../../bricks_data/galaxy_catalogue_{area}.csv', mode='a', index=False, header=False)
df_stars.to_csv(f'../../bricks_data/stellar_catalogue_{area}.csv', mode='a', index=False, header=False)
df_galaxy = df_galaxy[0:0]
df_stars = df_stars[0:0]
print()
print(f"=============================== Download {area} completed ==================================")
print()

print("Time taken for: ", bricks_to_classify, " bricks: ", round(((time.time() - start) / 60), 2))
