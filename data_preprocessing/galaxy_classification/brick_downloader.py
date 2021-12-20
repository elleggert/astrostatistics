from astropy.io import fits
import os
import wget
import random
import time
import argparse


def main():
    parser = argparse.ArgumentParser(description='Script to download bricks from DESI DR9',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--num_bricks', default=None, metavar='', type=int, help='number of bricks to download')
    parser.add_argument('-a', '--area', default='south', metavar='', type=str,
                        help='The catalogue that ought to be downloaded, by default extracts southern bricks. '
                             'Accepted values: south or north')

    args = vars(parser.parse_args())

    # Timing information
    # start = time.time()

    area = args['area']
    bricks_to_download = args['num_bricks']
    path = f'/Volumes/Astrostick/bricks_data/{area}/'

    if area != 'south' and area != 'north':
        print(area)
        print("Invalid Area argument: must be either 'north' or 'south'")
        exit()

    print()
    print(f"=============================== Download {area} ..... ==================================")
    print()

    # Utilises the summary files to get a list of all bricks in the catalogue --> these need to be stored somewhere
    hdulistBricks = fits.open(f'../../bricks_data/survey-bricks-dr9-{area}.fits')
    data = hdulistBricks[1].data

    bricknames = list(data.field('brickname'))

    downloaded_bricks = []

    # Getting already downloaded files:
    for filename in os.listdir(path):
        brick = filename.replace("tractor-", "")
        brick = brick.replace(".fits", "")
        downloaded_bricks.append(brick)

    print(f'Number of bricks left in {area}: {len(bricknames) - len(downloaded_bricks)}')

    """If a numerical limit was provided, getting a random sample of bricks without replacement 
    and deleting all that are already downloaded, else simply iterate through all bricks"""

    print(len(bricknames))

    bricknames = [x for x in bricknames if x not in downloaded_bricks]
    print(len(bricknames))

    if bricks_to_download:
        bricknames = random.sample(bricknames, bricks_to_download)
    print(len(bricknames))


    print(len(downloaded_bricks))
    for i, brickname in enumerate(bricknames):
        break
        folder = brickname[:3]
        url = f'https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/{area}/tractor/{folder}/tractor-{brickname}.fits'
        wget.download(url, path)

    print()
    print(f"=============================== Download {area} completed ==================================")
    print()

    # print("Time taken for: ", i, " bricks: ", round(((time.time() - start) / 60), 2))

    print(f"Number of bricks in downloaded in {area}:", len(os.listdir(path)))


if __name__ == "__main__":
    main()
