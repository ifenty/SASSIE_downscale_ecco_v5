#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 19:00:14 2020

@author: ifenty
"""
import argparse
import time
import numpy as np
from pathlib import Path
import netCDF4 as nc4
import datetime
from pprint import pprint

def update_metadata(nc_dir, key_value_dict, dry_run):

    print('\n\n===========================')
    print('APPLY_VALID_MINMAX_FOR_DATASET')

    print('directory ', nc_dir)
    pprint(key_value_dict)

    files = np.sort(list(nc_dir.glob('*nc')))

    # loop through files
    for file_i, file in enumerate(files):
        st = time.time()
        print(file)
        # open netcdf
        tmp_ds = nc4.Dataset(nc_dir / file, 'a')

        for attrs in tmp_ds.ncattrs(): 
            if attrs in key_value_dict.keys():
              if not dry_run:
                tmp_ds.setncattr(attrs, key_value_dict[attrs])
                current_time = datetime.datetime.now().isoformat()[0:19]
                tmp_ds.setncattr('date_metadata_modified', current_time)

        tmp_ds.close()
        print('took ', time.time()-st)

#%%

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else: #default value is True
        return True


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--nc_dir', type=str, required=True,\
                        help='directory containing nc files')

    parser.add_argument('--dry_run', type=str2bool, nargs='?', const=True,\
                        default = False,
                        help='do not apply new metadta')

    return parser


#%%
if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()
    print(args)

    nc_dir = Path(args.nc_dir)

    dry_run = args.dry_run

    print('\n\n===================================')
    print('starting update metadata')
    print('\n')
    print('nc_dir', nc_dir)
    print('dry_run', dry_run)
    print('\n')

    key_value_dict = {'history':'Initial release of the ECCO N1 Sassie Ocean-Sea Ice Simulation',
                      'metadata_link':'https://cmr.earthdata.nasa.gov/search/collections.umm_json?ShortName=TBD'}
    
    update_metadata(nc_dir,key_value_dict, dry_run)
    
