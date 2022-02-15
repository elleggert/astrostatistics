import numpy as np
import pickle
import set_dataloader
import time
import pandas as pd
import random
from collections import defaultdict
import argparse


# Importing Dicts and Defining Utilities

def main():
    parser = argparse.ArgumentParser(description='MultiSetSequence Dataset Generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--num_pixels', default=None, metavar='', type=int,
                        help='number of training examples to generate')
    #parser.add_argument('-a', '--area', default='north', metavar='', type=str,
                        #help='The area of the sky that should be prepared')

    args = vars(parser.parse_args())

    NSIDE = 512
    tests = [False, True]
    export_to_disk = True
    #area = args['area']
    num_pixels = args['num_pixels']

    areas = ['north', 'south', 'des']

    ccd, pixel2subpixel_dict, subpixel2ccd_dict = import_pixel_mappings(NSIDE)

    # Generate Train and Set

    for area in areas:
        print(f'Area {area} started')
        # Make this variable depenndent on the Area Size
        if area == 'north':
            max_ccds = 30
        elif area == 'south':
            max_ccds = 25
        else:
            max_ccds = 50

        for test in tests:

            if test:
                import_path = f'../../bricks_data/{area}_test_{NSIDE}.csv'
                if export_to_disk:
                    export_path = f'/Volumes/Astrodisk/bricks_data/{area}_test_{NSIDE}.pickle'
                else:
                    export_path = f'data/north/{area}_test_{NSIDE}.pickle'


            else:
                import_path = f'../../bricks_data/{area}_{NSIDE}.csv'
                if export_to_disk:
                    # If stored on Astrodisk Volume
                    export_path = f'/Volumes/Astrodisk/bricks_data/{area}_{NSIDE}.pickle'
                else:
                    export_path = f'data/north/{area}_{NSIDE}.pickle'

            df = pd.read_csv(import_path)
            # Randomly Sampling Pixel Indices from Dataframe
            pix_ids = df.pixel_id.to_numpy()

            if num_pixels is None:
                num_pixels = len(df)
            else:
                pix_ids = pix_ids[:num_pixels]
            num_subpixels = 16
            num_features = ccd.num_features

            input, lengths = generate_length_inputs(ccd, max_ccds, num_features, num_pixels, num_subpixels, pix_ids,
                                                    pixel2subpixel_dict, subpixel2ccd_dict)

            # Generate Dict

            mini_multiset = generate_dict(df, input, lengths, pix_ids)

            with open(export_path, 'wb') as f:
                pickle.dump(mini_multiset, f)
                f.close()

            """with open(export_path, 'wb') as f:
                pickle.dump(mini_multiset, f)
                f.close()"""


def generate_length_inputs(ccd, max_ccds, num_features, num_pixels, num_subpixels, pix_ids, pixel2subpixel_dict,
                           subpixel2ccd_dict):
    input = np.zeros((num_pixels, num_subpixels, max_ccds, num_features))
    # Iterate through the pixels
    for pix_no, pix in enumerate(pix_ids):

        subpix_ids = pixel2subpixel_dict[pix]
        subpix_ids = subpix_ids[:num_subpixels]

        for subpix_no, subpix in enumerate(subpix_ids):
            if subpix not in subpixel2ccd_dict:
                continue
            subpix_ccds = subpixel2ccd_dict[subpix]
            random.shuffle(subpix_ccds)
            subpix_ccds = subpix_ccds[:max_ccds]
            x = ccd.get_ccds(subpix_ccds)

            # Iterate through the CCDs for every pixel
            for ccd_no in range(len(subpix_ccds)):
                input[pix_no, subpix_no, ccd_no] = x[ccd_no]
    var_set_len = True
    lengths = np.zeros((num_pixels, num_subpixels), dtype=int)
    if var_set_len:
        for pix_no, pix in enumerate(pix_ids):
            subpix_ids = pixel2subpixel_dict[pix]
            subpix_ids = subpix_ids[:num_subpixels]

            for subpix_no, subpix in enumerate(subpix_ids):
                if subpix not in subpixel2ccd_dict:
                    lengths[pix_no, subpix_no] = 0
                    continue
                c = len(subpixel2ccd_dict[subpix])
                if c < max_ccds:
                    lengths[pix_no, subpix_no] = c
                else:
                    lengths[pix_no, subpix_no] = max_ccds

    else:
        lengths.fill(max_ccds)
    return input, lengths


def generate_dict(df, input, lengths, pix_ids):
    lrg = df.lrg.to_numpy()
    elg = df.elg.to_numpy()
    qso = df.qso.to_numpy()
    glbg = df.glbg.to_numpy()
    rlbg = df.rlbg.to_numpy()
    stellar = df.stellar.to_numpy().flatten()
    ebv = df.EBV.to_numpy().flatten()
    hinh = df.hinh.to_numpy().flatten()
    gaia = df.gaia.to_numpy().flatten()
    gaia12 = df.gaia12.to_numpy().flatten()
    sagitarius = df.sagitarius.to_numpy().flatten()
    mini_multiset = defaultdict(list)
    for i, pix in enumerate(pix_ids):
        mini_multiset[pix].append(input[i])
        mini_multiset[pix].append(lengths[i])
        mini_multiset[pix].append(lrg[i])
        mini_multiset[pix].append(elg[i])
        mini_multiset[pix].append(qso[i])
        mini_multiset[pix].append(glbg[i])
        mini_multiset[pix].append(rlbg[i])
        mini_multiset[pix].append(stellar[i])
        mini_multiset[pix].append(hinh[i])
        mini_multiset[pix].append(gaia[i])
        mini_multiset[pix].append(gaia12[i])
        mini_multiset[pix].append(sagitarius[i])
        mini_multiset[pix].append(ebv[i])
        mini_multiset[pix].append(pix)
    return mini_multiset


def import_pixel_mappings(NSIDE):
    time_start = time.time()
    # Importing Pixel Mappings
    # For DECAM, BASS, MzLS
    with open('../../bricks_data/pixel2ccd_2048_non_inclusive.pickle', 'rb') as f:
        subpixel2ccd_dict = pickle.load(f)
        f.close()
    time_end = time.time()
    time_passed = time_end - time_start
    print()
    print(f"{time_passed / 60:.5} minutes ({time_passed:.3} seconds) taken to import the dict")
    print()
    time_start = time.time()
    # For DECAM, BASS, MzLS
    with open(f'../../bricks_data/pixel2subpixel_{NSIDE}_2048.pickle', 'rb') as f:
        pixel2subpixel_dict = pickle.load(f)
        f.close()
    time_end = time.time()
    time_passed = time_end - time_start
    print()
    print(f"{time_passed / 60:.5} minutes ({time_passed:.3} seconds) taken to import the dict")
    print()
    ccd = set_dataloader.CCD()
    ccd.initialise_for_deepset()
    print(f'Number of Features : {ccd.num_features}')
    return ccd, pixel2subpixel_dict, subpixel2ccd_dict


if __name__ == "__main__":
    main()
