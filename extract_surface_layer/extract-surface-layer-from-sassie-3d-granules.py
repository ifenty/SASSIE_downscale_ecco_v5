#!/usr/bin/env python
# coding: utf-8

# # Process 3D SASSIE ocean model granules on s3 to extract just the surface layer

## import required packages
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import s3fs
import pandas as pd
from datetime import datetime
import json
from contextlib import contextmanager
import argparse
from pathlib import Path
import os
import time 
import netCDF4 as nc4

## import ECCO utils
import sys
sys.path.append('/home/jpluser/git_repos/ECCOv4-py')
import ecco_v4_py as ecco

## define functions

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


def show_me_the_ds(ds):
    print('\n>>show_me_the_ds')
    print('dims: ', list(ds.dims))
    print('coords: ', list(ds.coords))
    print('data_vars: ', list(ds.data_vars))


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
        dv_encoding[dv] =  {'compression':'zlib',\
                            'complevel':5,\
                            'shuffle':False,\
                            'fletcher32': False,\
                            '_FillValue':netcdf_fill_value}

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

@time_it
def save_surface_layer_from_3d(var_3d, sassie_s3_netcdf_dir, ec2_scratch_dir, sassie_key, sassie_secret, files_to_process):
    """
    Generates new netCDFs of surface layer from 3D fields.

    Returns:
        None
    """
    
    ## initialize s3 system
    s3 = s3fs.S3FileSystem(anon=False, key=sassie_key, secret=sassie_secret) 
    
    ## list all files
    nc_file_list = np.sort(s3.glob(f'{sassie_s3_netcdf_dir}{var_3d}_AVG_DAILY/*.nc'))

    print(f'\n> Looking for files on {sassie_s3_netcdf_dir}{var_3d}_AVG_DAILY/')
    print(f'... num files  : {len(nc_file_list)}')
    print(f'... first file : {nc_file_list[0]}')
    print(f'... last file  : {nc_file_list[-1]}')
    
    ## append "s3://" to create url in order to open the dataset
    nc_file_list_urls = []
    for file in nc_file_list:
        file_url_tmp = f"s3://{file}"
        nc_file_list_urls.append(file_url_tmp)

    print(f'\n> Preparing list of files to process')
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
        return 
    
    ## loop through each file, extract surface, and save new netCDF
    for file_url in data_urls_select:
        
        print(f"\n... opening {file_url}")
        s3_file = s3.open(file_url)
        s3_file_ec2 = xr.open_dataset(s3_file)
        s3_file_ec2.close()
     
        ## isolate the surface layer
        print(f"... extracting surface layer")
        tmp_surface = s3_file_ec2.isel(k=[0], k_u=[0], k_l=[0], k_p1=slice(0,2))
        
        ## edit typo in metadata and fix time_bnds
        tmp_surface.k_p1.attrs['comment'] = "Top and bottom of model tracer cell."
        tmp_surface.coords['time'].attrs['units'] = "hours since 1992-01-01T12:00:00"
        tmp_surface.coords['time_bnds'].attrs['units'] = "hours since 1992-01-01T12:00:00"
        
        ## print(s3_file_ec2)
        show_me_the_ds(tmp_surface)
    
        ## save newly generated 2D surface layer dataset to scratch directory
        print(f"... saving surface netCDF dataset to scratch directory {ec2_scratch_dir}")
    
        ## edit filename to indicate that it is a surface layer file (not 3D)
        filename_split = file_url.split("/")[-1].split("day")
        netcdf_filename_new = f"{filename_split[0]}SURFACE_day{filename_split[1]}"

        ## create encoding
        encoding_var = create_encoding(tmp_surface, output_array_precision = np.float32)
        
        tmp_surface.to_netcdf(f"{ec2_scratch_dir}/{netcdf_filename_new}", encoding = encoding_var)
        tmp_surface.close()
        print(f"* * * * saved netcdf to {ec2_scratch_dir}/{netcdf_filename_new} * * * *\n")


@time_it
def push_nc_dir_from_ec2(ec2_scratch_dir, dest_s3_name, var_name):
    """
    Pushes the netcdf files from a directory to an S3 bucket.

    Args:
        ec2_scratch_dir (str): The path to the directory containing the netcdf files on the EC2 instance.
        root_dest_s3_name (str): The root name of the S3 bucket where the files will be pushed.
        var_name (str): The name of the variable used to create the S3 bucket.

    Returns:
        None
    """
    ## push file to s3 bucket
    mybucket = dest_s3_name + var_name + "_AVG_DAILY_SURF"
    nc_files = list(ec2_scratch_dir.glob('*.nc'))

    print(f'\n>pushing netcdf files in {ec2_scratch_dir} to s3 bucket : {mybucket}')
    print(f'... looking for *.nc files in {ec2_scratch_dir}')
    print(f'... found {len(nc_files)} nc files to upload')

    if len(nc_files)>0:
        cmd=f"aws s3 cp {ec2_scratch_dir} {mybucket}/ --recursive --include '*.nc' --no-progress --profile sassie> /dev/null 2>&1"
        print(f'... aws command: {cmd}')
        with suppress_stdout():
           os.system(cmd)
    else:
        print("... nothing to upload!") 


########### Create final routine to process files ############
def extract_surface_layer(var_3d, sassie_s3_netcdf_dir, ec2_nvme_scratch_dir, sassie_key, sassie_secret, dest_s3_name, keep_local_files, push_to_s3, files_to_process):
    """
    Create 2D netCDF files for the surface layer of all netCDfs for a given 3d variable.

    Args:
        var_3d (str): The 3D variable to be processed.
        sassie_s3_netcdf_dir (str): The directory for the s3 bucket where all the datasets to be processed are stored.
        ec2_nvme_scratch_dir (str): The scratch directory on the local EC2 NVMe storage system where newly generated datasets are stored.
        sassie_key (str): s3 bucket key for sassie team.
        sassie_secret (str): s3 bucket secret key for sassie team.
        dest_s3_name (str): The s3 bucket where 2D netCDFs will be saved.

    Returns:
        None
    """
    
    ## process variable
    print(f"= = = = = processing {var_3d} = = = = =\n")

    ## create temporary scratch directory on ec2
    nc_root_dir_ec2 =  Path(f"{ec2_nvme_scratch_dir}/tmp_nc/{var_3d}_AVG_DAILY_SURF")
    print(f'... temporary nc directory {nc_root_dir_ec2}\n')
    nc_root_dir_ec2.mkdir(exist_ok=True, parents=True)

    ## extract surface layer from 3D netcdfs and save to scratch directory
    save_surface_layer_from_3d(var_3d, sassie_s3_netcdf_dir, nc_root_dir_ec2, sassie_key, sassie_secret, files_to_process)

    ## clean up local scratch disk
    ## push nc files to aws s3
    if push_to_s3:
        push_nc_dir_from_ec2(nc_root_dir_ec2, dest_s3_name, var_3d)
    else:
        print('> not pushing files to s3')

    print('\n> cleaning up local nc files') 
    if keep_local_files:
        print('... keeping local nc  directories')
    else:
        ## remove tmp nc var directory and all of its contents
        print("... removing tmp nc root dir ", nc_root_dir_ec2)
        os.system(f"rm -rf {str(nc_root_dir_ec2)}")
    
    print(f'\n==== done processing 3d variable: {var_3d} ====\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--var_3d", action="store",
                        help="The 3D variable to be processed (e.g., THETA).",
                        dest="var_3d", type=str, required=True)

    parser.add_argument("-b", "--sassie_s3_netcdf_dir", action="store",
                        help="The s3 bucket name where all data files are stored (e.g. s3://ecco-model-granules/SASSIE/N1/).", 
                        dest="sassie_s3_netcdf_dir", type=str, required=True)

    parser.add_argument("-l", "--ec2_nvme_scratch_dir", action="store",   
                        help="The local scratch directory on the ec2 instance where files will be stored temporarily.", 
                        dest="ec2_nvme_scratch_dir", type=str, required=True)

    parser.add_argument("-k", "--sassie_key", action="store",   
                        help="SASSIE s3 bucket key.", 
                        dest="sassie_key", type=str, required=True)

    parser.add_argument("-s", "--sassie_secret", action="store",   
                        help="SASSIE s3 bucket secret key.", 
                        dest="sassie_secret", type=str, required=True)

    parser.add_argument("-d", "--dest_s3_name", action="store",   
                        help="The s3 bucket name where the newly generated 2D netcdfs will be saved.", 
                        dest="dest_s3_name", type=str, required=True)
    
    parser.add_argument("--save_nc_to_disk", action="store_true",
                        help="Boolean to indicate whether to keep local files on ec2 instance after processing.")
    
    parser.add_argument("--push_to_s3", action="store_true",
                        help="Boolean to indicate whether to keep send netcdf files to s3 bucket after processing.")

    parser.add_argument("-p", "--files_to_process", action="store",
                        help="String specifying whether to process all files (-1), one file (one number as index), or range (start end).",
                        dest="files_to_process", nargs="*", type=int, required=False, default = [-1])


    args = parser.parse_args()

    var_3d = args.var_3d
    sassie_s3_netcdf_dir = args.sassie_s3_netcdf_dir
    ec2_nvme_scratch_dir = args.ec2_nvme_scratch_dir
    sassie_key = args.sassie_key
    sassie_secret = args.sassie_secret
    dest_s3_name = args.dest_s3_name
    save_nc_to_disk = args.save_nc_to_disk
    push_to_s3 = args.push_to_s3
    files_to_process = args.files_to_process
    

    if save_nc_to_disk == False:
       push_to_s3 = False
       print('save_nc_to_disk is not set, so push_to_s3 set to false') 

    print('PROCESSING 3D SASSIE-ECCO NETCDF GRANULES TO SURFACE 2D NETCDFS')
    print('----------------------------------------------')
    print('\nARGUMENTS: \n')
    print('var_3d ', var_3d)
    print('sassie_s3_netcdf_dir ', sassie_s3_netcdf_dir)
    print('ec2_nvme_scratch_dir ', ec2_nvme_scratch_dir)
    print('sassie_key ', sassie_key)
    print('sassie_secret ', sassie_secret)
    print('dest_s3_name ', dest_s3_name)
    print('save_nc_to_disk ', save_nc_to_disk)
    print('push_to_s3 ', push_to_s3)
    print('files_to_process ', files_to_process)

    print('\n>>>> BEGIN EXECUTION')
    extract_surface_layer(var_3d, sassie_s3_netcdf_dir, ec2_nvme_scratch_dir, \
                          sassie_key, sassie_secret, dest_s3_name, \
                          save_nc_to_disk, push_to_s3, files_to_process)

    print('>>>> END EXECUTION\n')



    