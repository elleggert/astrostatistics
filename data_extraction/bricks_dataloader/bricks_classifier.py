import time

from astropy.io import fits
from astropy.table import Table, vstack
import numpy as np
import os
import pandas as pd

from brick import Brick

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
print("=============================== Classification South... ==================================")
print()

start = time.time()

bricknames_south_sample = []

for filename in os.listdir('/Volumes/Astrodisk/bricks_data/north/'):
    brickn = filename.replace("tractor-", "")
    brickn = brickn.replace(".fits", "")
    bricknames_south_sample.append(brickn)

bricknames_south_sample.pop()

#print(bricknames_south_sample[0])
df_galaxy = pd.DataFrame(columns=['BrickID', 'RA', 'DEC', 'Target_type', 'Fitbits', 'Maskbits'])
df_stars = pd.DataFrame(columns=['RA', 'DEC', 'GMAG', 'RMAG', 'ZMAG'])
#df_galaxy.to_csv('../../bricks_data/galaxy_catalogue.csv', index=False)
#df_stars.to_csv('../../bricks_data/stellar_catalogue.csv', index=False)

print(df_galaxy.head())
print(df_stars.head())

#new = Table.read('tractor-0053m395.fits', format='fits')


for no, brickname in enumerate(bricknames_south_sample):

    # df = pd.read_csv('../bricks_data/galaxy_catalogue_sample.csv')

        # Check how there can be a brickname without corresponding id

    """
    if no == 0:
    

        south = south_survey_is_south[np.where(brickid_south == brickid)]
        if len(south) > 0:
            south = south[0]
        else:
            south = True

        brick.initialise_brick_for_galaxy_classification(south)
        brick.initialise_brick_for_stellar_density()

        stars = brick.get_stellar_objects()

        support_df = pd.DataFrame(stars, columns=['RA', 'DEC', 'GMAG', 'RMAG', 'ZMAG'])
        df_stars = df_stars.append(support_df)
        continue
    

    #t2 = Table.read(f'/Volumes/Astrostick/bricks_data/south_test/tractor-{brickname}.fits', format='fits')
    #new = vstack([new, t2])

    
    nrows2 = t2[1].data.shape[0]
    nrows1 = t1[1].data.shape[0]
    nrows = nrows1 + nrows2
    hdu = fits.BinTableHDU.from_columns(t1[1].columns, nrows=nrows)
    for colname in t1[1].columns.names:
        hdu.data[colname][nrows1:] = t2[1].data[colname]
    fits.update('tractor-0053m395.fits', hdu.data, 1)
    #header = fits.getheader(f'/Volumes/Astrostick/bricks_data/south_test/tractor-{brickname}.fits')
    
    print("Brick progression ", time.time() - start)
    """

    brickid = brickid_north[np.where(brickname_north == brickname)]

    if len(brickid > 0):
        brickid = brickid[0]
    else:
        brickid = 0

    hdu = fits.open(f'/Volumes/Astrodisk/bricks_data/north/tractor-{brickname}.fits')
    data = hdu[1].data
    brick = Brick(data)
    south = north_survey_is_south[np.where(brickid_north == brickid)]
    if len(south) > 0:
        south = south[0]
    else:
        south = True

    ## Enable this is classifying North Objects
    # south = north_survey_is_south[np.where(brickid_north == brickid)][0]

    #south_array = np.full(shape=brick.no_of_objects, fill_value=south)


    brick.initialise_brick_for_galaxy_classification(south)
    target_type = brick.classify_galaxies()


    # Process array
    stacked_array = np.stack((brick.ids, brick.ra, brick.dec, target_type, brick.maskbits, brick.fitbits), axis=1)
    support_df = pd.DataFrame(stacked_array, columns=['BrickID', 'RA', 'DEC', 'Target_type', 'Fitbits', 'Maskbits'])
    support_df.drop(support_df[support_df.Target_type == 0].index, inplace=True)

    df_galaxy = df_galaxy.append(support_df)
    


    brick.initialise_brick_for_stellar_density()

    stars = brick.get_stellar_objects()


    support_df = pd.DataFrame(stars, columns=['RA', 'DEC', 'GMAG', 'RMAG', 'ZMAG'])
    df_stars = df_stars.append(support_df)



    #print("Brick progression ", time.time() - start)

    # Do not forget to check this clause
    # ['BrickID', 'ObjectID','RA', 'DEC', 'South', 'Target_type']

    # df.to_csv('../bricks_data/galaxy_catalogue_sample_profiling.csv', index=False)



    if no % 500 == 0:
        print(no)
        df_galaxy = df_galaxy.astype(
            {'BrickID': 'int32', 'Target_type': 'int8', 'Fitbits': 'int16', 'Maskbits': 'int16'})
        df_galaxy.to_csv('../../bricks_data/galaxy_catalogue.csv', mode='a', index=False, header=False)
        df_stars.to_csv('../../bricks_data/stellar_catalogue.csv', mode='a', index=False, header=False)

        df_galaxy = df_galaxy[0:0]
        df_stars = df_stars[0:0]

    """
    print(df_galaxy.head())
    print(df_galaxy.Maskbits.unique())
    print(df_stars.head())

    
    print(df_galaxy.shape)
    print(df_stars.shape)
    """

    # print(" ===================== Brick", brickname, " complete=====================")

"""
hdu = fits.table_to_hdu(new)
fits.update('tractor-0053m395.fits', hdu.data, 1)


hdu = fits.open('tractor-0053m395.fits')
print(hdu.info())
data = hdu[1].data
print(type(data))

ra = data.field('ra')
print(len(ra))
brick = Brick(data)
brick.initialise_brick_for_galaxy_classification(south=True)
target_type = brick.classify_galaxies()



# Process array
stacked_array = np.stack((brick.ids, brick.ra, brick.dec, target_type, brick.maskbits, brick.fitbits), axis=1)
support_df = pd.DataFrame(stacked_array, columns=['BrickID', 'RA', 'DEC', 'Target_type', 'Fitbits', 'Maskbits'])
support_df.drop(support_df[support_df.Target_type == 0].index, inplace=True)
df_galaxy = df_galaxy.append(support_df)
print(df_galaxy.shape)

"""

df_galaxy = df_galaxy.astype({'BrickID': 'int32', 'Target_type': 'int8', 'Fitbits': 'int16', 'Maskbits': 'int16'})

df_galaxy.to_csv('../../bricks_data/galaxy_catalogue.csv', mode='a', index=False, header=False)
df_stars.to_csv('../../bricks_data/stellar_catalogue.csv', mode='a', index=False, header=False)

#print(df_galaxy.groupby('Target_type').count())


"""

data = hdul1[1].data
brick = Brick(data)
brick.initialise_brick_for_galaxy_classification(south=True)
target_type = brick.classify_galaxies()

# Process array
stacked_array = np.stack((brick.ids, brick.ra, brick.dec, target_type, brick.maskbits, brick.fitbits), axis=1)
support_df = pd.DataFrame(stacked_array, columns=['BrickID', 'RA', 'DEC', 'Target_type', 'Fitbits', 'Maskbits'])
support_df.drop(support_df[support_df.Target_type == 0].index, inplace=True)
df_galaxy = df_galaxy.append(support_df)
"""
print()
print("=============================== Classification South Completed ==================================")
print()
#df_galaxy = df_galaxy[df_galaxy['Target_type'] > 0]
#df_galaxy.to_csv('../bricks_data/galaxy_catalogue_sample_profiling.csv', index=False)
#df_stars.to_csv('../bricks_data/stellar_catalogue_sample_profiling.csv', index=False)


df_galaxy = df_galaxy.astype({'BrickID': 'int32', 'Target_type': 'int8', 'Fitbits': 'int16', 'Maskbits': 'int16'})

df_galaxy.to_csv('../../bricks_data/galaxy_catalogue_sample_profiling.csv', mode='a', index=False, header=False)
df_stars.to_csv('../../bricks_data/stellar_catalogue_sample_profiling.csv',  mode='a', index=False, header=False)
print(df_galaxy.groupby('Target_type').count())



print(f"Time taken for{len(bricknames_south_sample)} bricks: ", time.time() - start)

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
