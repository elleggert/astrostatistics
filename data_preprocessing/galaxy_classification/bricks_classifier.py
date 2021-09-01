import time

from astropy.io import fits
import numpy as np
import os
import pandas as pd

from brick import Brick

area = 'south'
device = 'Astrodisk'

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

print()
print(f"=============================== Classification {area}... ==================================")
print()

start = time.time()

bricknames_south_sample = []

for filename in os.listdir(f'/Volumes/{device}/bricks_data/{area}/'):
    brickn = filename.replace("tractor-", "")
    brickn = brickn.replace(".fits", "")
    bricknames_south_sample.append(brickn)

if device == "Astrostick":
    bricknames_south_sample.pop()

# print(bricknames_south_sample[0])
df_galaxy = pd.DataFrame(columns=['BrickID', 'RA', 'DEC', 'LRG', 'ELG', 'QSO'])
df_stars = pd.DataFrame(columns=['RA', 'DEC', 'GMAG', 'RMAG', 'ZMAG'])
# df_galaxy.to_csv(f'../../bricks_data/galaxy_catalogue_{area}.csv', index=False)
# df_stars.to_csv(f'../../bricks_data/stellar_catalogue_{area}.csv', index=False)

print(df_galaxy.head())
print(df_stars.head())

for no, brickname in enumerate(bricknames_south_sample):

    brickid = brickid_south[np.where(brickname_south == brickname)]
    # North Bricks
    # brickid = brickid_north[np.where(brickname_north == brickname)]

    if len(brickid > 0):
        brickid = brickid[0]
    else:
        brickid = 0

    hdu = fits.open(f'/Volumes/{device}/bricks_data/{area}/tractor-{brickname}.fits')
    data = hdu[1].data
    brick = Brick(data)

    # south = north_survey_is_south[np.where(brickid_north == brickid)]

    south = south_survey_is_south[np.where(brickid_south == brickid)]
    if len(south) > 0:
        south = south[0]
    else:
        south = True

    ## Enable this is classifying North Objects
    # south = north_survey_is_south[np.where(brickid_north == brickid)][0]

    brick.initialise_brick_for_galaxy_classification(south)
    target_objects = brick.classify_galaxies()

    support_df = pd.DataFrame(target_objects,
                              columns=['BrickID', 'RA', 'DEC', 'LRG', 'ELG', 'QSO'])

    df_galaxy = df_galaxy.append(support_df)

    brick.initialise_brick_for_stellar_density()

    stars = brick.get_stellar_objects()

    support_df = pd.DataFrame(stars, columns=['RA', 'DEC', 'GMAG', 'RMAG', 'ZMAG'])
    df_stars = df_stars.append(support_df)

    # print("Brick progression ", time.time() - start)

    # Do not forget to check this clause
    # ['BrickID', 'ObjectID','RA', 'DEC', 'South', 'Target_type']

    # df.to_csv('../bricks_data/galaxy_catalogue_sample_profiling.csv', index=False)

    if no % 447 == 0:
        print(no / 447)
        df_galaxy = df_galaxy.astype(
            {'BrickID': 'int32', 'LRG': 'int8', 'ELG': 'int8', 'QSO': 'int8'})
        df_galaxy.to_csv(f'../../bricks_data/galaxy_catalogue_{area}.csv', mode='a', index=False, header=False)
        df_stars.to_csv(f'../../bricks_data/stellar_catalogue_{area}.csv', mode='a', index=False, header=False)
        # df_galaxy.to_csv('../../bricks_data/galaxy_catalogue_sample_profiling.csv', index=False, header=False)
        # df_stars.to_csv('../../bricks_data/stellar_catalogue_sample_profiling.csv', index=False, header=False)
        df_galaxy = df_galaxy[0:0]
        df_stars = df_stars[0:0]

    # print(" ===================== Brick", brickname, " complete=====================")

# df_galaxy = df_galaxy.astype({'BrickID': 'int32', 'LRG': 'int8', 'Fitbits': 'int16', 'Maskbits': 'int16'})

# df_galaxy.to_csv('../../bricks_data/galaxy_catalogue.csv', mode='a', index=False, header=False)
# df_stars.to_csv('../../bricks_data/stellar_catalogue.csv', mode='a', index=False, header=False)

# df_galaxy.to_csv('../../bricks_data/galaxy_catalogue_sample_profiling.csv', index=False)
# df_stars.to_csv('../../bricks_data/stellar_catalogue_sample_profiling.csv', index=False)


print()
print(f"=============================== Classification {area} Completed ==================================")
print()
# df_galaxy = df_galaxy[df_galaxy['Target_type'] > 0]
# df_galaxy.to_csv('../bricks_data/galaxy_catalogue_sample_profiling.csv', index=False)
# df_stars.to_csv('../bricks_data/stellar_catalogue_sample_profiling.csv', index=False)


# df_galaxy = df_galaxy.astype({'BrickID': 'int32', 'LRG': 'int8', 'Fitbits': 'int16', 'Maskbits': 'int16'})

# df_galaxy.to_csv('../../bricks_data/galaxy_catalogue_sample_profiling.csv', mode='a', index=False, header=False)
# df_stars.to_csv('../../bricks_data/stellar_catalogue_sample_profiling.csv',  mode='a', index=False, header=False)
# print(df_galaxy.groupby('Target_type').count())


print(f"Time taken for {len(bricknames_south_sample)} bricks: ", (time.time() - start) / 60)

'''


print()
print("=============================== Removing South ... ==================================")
print()


for filename in os.listdir('/Volumes/Astrostick/bricks_data/south/'):
    print("Removing file:", filename)
    os.remove(f'/Volumes/Astrostick/bricks_data/south/{filename}')

print()
print("=============================== Removing South Completed ==================================")
print()

'''

'''
print()
print("=============================== Removing North ... ==================================")
print()

for filename in os.listdir('/Volumes/Astrostick/bricks_data/north/'):
    print("Removing file:", filename)
    os.remove(f'/Volumes/Astrostick/bricks_data/north/{filename}')

print()
print("=============================== Removing North Completed ==================================")
print()

'''
