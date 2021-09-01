from astropy.io import fits
import os
import wget
import numpy.random
import random
import time

start = time.time()

bricks_to_download = 16000
# Sampling from area with probability 1:3, since this is the real distribution of bricks
area = "south"
rand = numpy.random.uniform(low=0.0, high=1.0, size=None)
print(rand)
if rand <= 0.25:
    area = "north"

print()
print(f"=============================== Download {area} ..... ==================================")
print()

hdulistBricks = fits.open(f'../../bricks_data/survey-bricks-dr9-{area}.fits')
data = hdulistBricks[1].data

bricknames = list(data.field('brickname'))

downloaded_bricks = []

# Getting already downloaded files:
for filename in os.listdir(f'/Volumes/Astrodisk/bricks_data/{area}/'):
    brickn = filename.replace("tractor-", "")
    brickn = brickn.replace(".fits", "")
    downloaded_bricks.append(brickn)

# Getting a random sample of bricknames without replacement and deleting all that are already downloaded
bricknames_sample = random.sample(bricknames, bricks_to_download)
bricknames_sample = [x for x in bricknames_sample if x not in downloaded_bricks]

for i, brickname in enumerate(bricknames_sample):
    folder = brickname[:3]
    url = f'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/{area}/tractor/{folder}/tractor-{brickname}.fits'
    wget.download(url, f'/Volumes/Astrodisk/bricks_data/{area}/')

    print(f" Brick {area} downloaded: ", brickname, ", Brick ", i, " of ", bricks_to_download)

print()
print(f"=============================== Download {area} completed ==================================")
print()

print("Time taken for: ", bricks_to_download, " bricks: ", round(((time.time() - start) / 60), 2))

print(f"Number of bricks in {area}:", len(os.listdir(f'/Volumes/Astrodisk/bricks_data/{area}/')))
