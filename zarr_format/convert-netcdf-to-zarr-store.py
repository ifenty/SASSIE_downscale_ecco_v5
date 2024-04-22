#!/usr/bin/env python
# coding: utf-8

# # Process SASSIE ocean model granules on s3

## import required packages
import numpy as np
import xarray as xr
import s3fs
import zarr
import argparse
from pathlib import Path
import os
from contextlib import contextmanager
import time 
import netCDF4 as nc4


## Define functions


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def time_it(func):
    """
    Decorator that reports the execution time.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Capture the start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Capture the end time
        print(f"{func.__name__} took {end_time-start_time:.4f} seconds to execute")
        return result
    return wrapper


def create_encoding(ecco_ds, output_array_precision = np.float32):
    
    # Create NetCDF encoding directives
    # ---------------------------------------------
    # print('\n... creating variable encodings')
    # ... data variable encoding directives
    
    # Define fill values for NaN
    if output_array_precision == np.float32:
        netcdf_fill_value = nc4.default_fillvals['f4']

    elif output_array_precision == np.float64:
        netcdf_fill_value = nc4.default_fillvals['f8']
    
    dv_encoding = dict()
    for dv in ecco_ds.data_vars:
        dv_encoding[dv] =  {'compressor': zarr.Blosc(cname="zlib", clevel=5, shuffle=False)}

    # ... coordinate encoding directives
    coord_encoding = dict()
    
    for coord in ecco_ds.coords:
        # set default no fill value for coordinate
        if output_array_precision == np.float32:
            coord_encoding[coord] = {'_FillValue':None, 'dtype':'float32'}
        elif output_array_precision == np.float64:
            coord_encoding[coord] = {'_FillValue':None, 'dtype':'float64'}

        # force 64 bit ints to be 32 bit ints
        if (ecco_ds[coord].values.dtype == np.int32) or \
           (ecco_ds[coord].values.dtype == np.int64) :
            coord_encoding[coord]['dtype'] ='int32'

        # fix encoding of time
        if coord == 'time' or coord == 'time_bnds':
            coord_encoding[coord]['dtype'] ='int32'

            if 'units' in ecco_ds[coord].attrs:
                # apply units as encoding for time
                coord_encoding[coord]['units'] = ecco_ds[coord].attrs['units']
                # delete from the attributes list
                del ecco_ds[coord].attrs['units']

        elif coord == 'time_step':
            coord_encoding[coord]['dtype'] ='int32'

    # ... combined data variable and coordinate encoding directives
    encoding = {**dv_encoding, **coord_encoding}

    return encoding


########### Create final routine to process files ########### 

## Specify root directory and process all variables in that dataset
@time_it
def convert_sassie_ecco_netcdfs_to_zarr(var_name, sassie_s3_netcdf_dir, zarr_s3_bucket,\
                                        files_to_process, sassie_key, sassie_secret):

    ## initialize s3 system for sassie bucket
    s3 = s3fs.S3FileSystem(anon=False, key=sassie_key, secret=sassie_secret) 
    
    ## list all files
    nc_file_list = np.sort(s3.glob(f'{sassie_s3_netcdf_dir}{var_name}_AVG_DAILY/*.nc'))
    
    ## append "s3://" to create url in order to open the dataset
    nc_file_list_urls = []
    for file in nc_file_list:
        file_url_tmp = f"s3://{file}"
        nc_file_list_urls.append(file_url_tmp)

    ## specify start and end indices or process all files   
    if len(files_to_process) == 2: # two numbers indicates a range (two indices)
        data_urls_select = nc_file_list_urls[files_to_process[0]:files_to_process[1]]
        print(f'... first file to process : {data_urls_select[0]}')
        print(f'... last file to process  : {data_urls_select[-1]}')
    
    elif len(files_to_process) == 1 and files_to_process[0] == -1: # process all files
        data_urls_select = nc_file_list_urls
        print(f'... first file to process : {data_urls_select[0]}')
        print(f'... last file to process  : {data_urls_select[-1]}')
    
    elif len(files_to_process) == 1 and files_to_process[0] >= 0: # process one file using number as index
        # wrap in list
        data_urls_select = [nc_file_list_urls[files_to_process[0]]]
        print(f'... 1 file to process : {data_urls_select}')
    
    else:
        print("ERROR: invalid entry for `files_to_process` argument")
        
    ## define s3 zarr bucket directory
    zarr_s3_bucket_dir = f"{zarr_s3_bucket}{var_name}_AVG_DAILY.ZARR/"
    s3_store = s3fs.S3Map(root=zarr_s3_bucket_dir, s3=s3, check=False)
    
    ## loop through each file and save new zarr format
    for i in range(len(data_urls_select)):
        file_url = data_urls_select[i]
        
        print(f"\n... opening {file_url}")
        s3_file = s3.open(file_url)
        s3_file_ec2 = xr.open_dataset(s3_file)
        s3_file_ec2.close()
    
        ## get filename
        filename_i = file_url.split("/")[-1]
    
        ## write the first netCDF to establish the zarr store, then we will append to that one
        if i == 0:
            ## create encoding for saving file
            enc = create_encoding(s3_file_ec2)
            s3_file_ec2.to_zarr(store=s3_store, mode='w', encoding=enc, consolidated=True)
            print(f"\n... saved first timestep {filename_i} to {zarr_s3_bucket_dir}")
        if i > 0:
            ## append with remaining netCDFs
            s3_file_ec2.to_zarr(store=s3_store, mode='a', append_dim='time')
            print(f"\n... saved timestep {filename_i} to {zarr_s3_bucket_dir}")
        
    print(f"* * * * * saved all netCDF files to zarr store: {zarr_s3_bucket_dir} * * * * *")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    parser.add_argument("-v", "--var_name", action="store",
                        help="The variable to be processed (e.g. SALT).",
                        dest="var_name", type=str, required=True)

    parser.add_argument("-f", "--s3_netcdf_dir", action="store",
                        help="The s3 bucket name where all netCDF data files are stored (e.g. s3://podaac-dev-sassie/ECCO_model/N1/V1/HH/NETCDF/).", 
                        dest="s3_netcdf_dir", type=str, required=True)

    parser.add_argument("-d", "--zarr_dest_s3_name", action="store",
                        help="The destination s3 bucket where processed zarr stores will be created (e.g., s3://podaac-dev-sassie/ECCO_model/N1/V1/HH/ZARR/).", 
                        dest="zarr_dest_s3_name", type=str, required=True)

    parser.add_argument("-p", "--files_to_process", action="store",
                        help="String specifying whether to process all files (-1), one file (one number as index), or range (start end).",
                        dest="files_to_process", nargs="*", type=int, required=False, default = [-1])

    parser.add_argument("-k", "--sassie_key", action="store",   
                        help="SASSIE s3 bucket key.", 
                        dest="sassie_key", type=str, required=True)

    parser.add_argument("-s", "--sassie_secret", action="store",   
                        help="SASSIE s3 bucket secret key.", 
                        dest="sassie_secret", type=str, required=True)
    
    args = parser.parse_args()

    var_name = args.var_name
    s3_netcdf_dir = args.s3_netcdf_dir
    zarr_dest_s3_name = args.zarr_dest_s3_name
    files_to_process = args.files_to_process
    sassie_key = args.sassie_key
    sassie_secret = args.sassie_secret

    print('WELCOME TO THE NETCDF TO ZARR CONVERTER')
    print('----------------------------------------------')
    print('\nARGUMENTS: ')
    print('var_name ', var_name)
    print('s3_netcdf_dir ' , s3_netcdf_dir)
    print('zarr_dest_s3_name ', zarr_dest_s3_name)
    print('files_to_process ', files_to_process)
    print('sassie_key ', sassie_key)
    print('sassie_secret ', sassie_secret)

    print('\n>>>> BEGIN EXECUTION')

    convert_sassie_ecco_netcdfs_to_zarr(var_name, s3_netcdf_dir, zarr_dest_s3_name,\
                                        files_to_process, sassie_key, sassie_secret)
    print('>>>> END EXECUTION\n')


