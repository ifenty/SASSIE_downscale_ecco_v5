#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import s3fs
import logging
import fsspec
import ujson
from tqdm import tqdm
from glob import glob

from kerchunk.hdf import SingleHdf5ToZarr 
from kerchunk.combine import MultiZarrToZarr

import matplotlib.pyplot as plt
from datetime import datetime
import os
import subprocess
import requests
import boto3
import s3fs
import time;
from pprint import pprint
from pathlib import Path

import argparse


# In[2]:


def get_urls_for_variable(s3, s3_root, var_name):
    # s3_root: s3://ecco-processed-data/SASSIE/N1/V1/HH/NETCDF/
    # var_name 'SIheff_AVG_DAILY 
    s3_urls = s3.glob(f'{s3_root}/{var_name}/*nc')
    full_urls = ['s3://' + f for f in s3_urls ]

    proc_info = {'s3_urls':s3_urls, 'full_urls':full_urls, 'var_name': var_name}
    return proc_info
    


# In[3]:


def gen_json(v):
    url=v[0]
    var_name = v[1]
    output_base_dir = v[2]
    key = v[3]
    aws_secret_access_key = v[4]
    so = dict(
        mode="rb", anon=False, default_fill_cache=False,
        default_cache_type="none",
        key=key, secret=aws_secret_access_key
    )
    output_dir = Path(f'{output_base_dir}/tmp/{var_name}')

    if not output_dir.exists():
        try:
            output_dir.mkdir(exist_ok=True, parents=True)
    
        except:
            print(f'could not make {output_dir} directory')
            return
            
    with fsspec.open(u, **so) as inf:
        h5chunks = SingleHdf5ToZarr(inf, url, inline_threshold=3000)
        with open(f"{output_dir}/{url.split('/')[-1]}.json", 'wb') as outf:
           outf.write(ujson.dumps(h5chunks.translate()).encode())

    return output_dir


# In[4]:


def get_var_names(s3, s3_root):
    s3_urls = s3.glob(f'{s3_root}/*AVG*')
    var_names = [x.split('/')[-1] for x in s3_urls]
    return var_names



if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", type=int, help="an integer number")
    parser.add_argument("--key", type=str, help="aws key")
    parser.add_argument("--secret", type=str, help="aws key")    

    args = parser.parse_args()
    var_num = args.number
    key = args.key
    secret = aws_secret_access_key

    output_base_dir = '/home/jpluser/sassie_jsons_podaac/'

    s3=None
    #s3_root = 's3://ecco-processed-data/SASSIE/N1/V1/HH/NETCDF'
    #s3 = s3fs.S3FileSystem(anon=False)

    s3_root = 's3://podaac-dev-sassie/ECCO_model/N1/V1/HH/NETCDF'
    
    s3 = s3fs.S3FileSystem(anon=False, key=key, secret=aws_secret_access_key)

    var_names = get_var_names(s3,s3_root)

    print('TOTAL VARIABLES: ', len(var_names))
    print(var_names[0])
    print(var_names[-1])
    
    print(f'\n... processing {var_names[var_num]}')
    
    # process variable number i
    proc_info = get_urls_for_variable(s3,s3_root, var_names[var_num])

    print("\n... proc_info ")
    print(f"var_name: {proc_info['var_name']}")
    pprint(proc_info['s3_urls'][:3])
    pprint(proc_info['full_urls'][:3])
    print(f"\n... total {len(proc_info['full_urls'])} records for {proc_info['var_name']}")

    print('\n... begin making individual jsons')

    debug = True
    if debug:
        urls_to_process = proc_info['full_urls'][:5]
        for u in tqdm(urls_to_process):
            tmp_json_output_dir = gen_json([u, proc_info['var_name'], \
                                            output_base_dir, \
                                            key, \
                                            aws_secret_access_key])
        
    else:
        urls_to_process = proc_info['full_urls']
        for u in urls_to_process:
            print(u)
            tmp_json_output_dir = gen_json([u, proc_info['var_name'], \
                                            output_base_dir, \
                                            key, \
                                            aws_secret_access_key])
        
    print('\n... end making individual jsons')
    
    
    print(f'\n... tmp_output_dir {tmp_json_output_dir}')
    
    json_list = sorted(glob(f'{tmp_json_output_dir}/*.json'))

    
    final_json_fname = f'{output_base_dir}{proc_info["var_name"]}.json'
    print(f'\n... reference output json: {final_json_fname}')

    print('\n... begin MultiZarrToZarr')
    start_time = time.time()
    mzz = MultiZarrToZarr(
        json_list,
        remote_protocol="s3",
        remote_options={'anon':False, 'key':key, 'secret':aws_secret_access_key},
        concat_dims='time',
        inline_threshold=3000
    )
    
    mzz.translate(final_json_fname)
    end_time = time.time()
    print('\n... end MultiZarrToZarr')
    
    print(f'\nMZZ {len(json_list)} json files in: {(end_time - start_time):0.3f}s')
    print(f'MZZ {len(json_list)} time per file: {(end_time - start_time)/len(json_list):0.3f}s')
    

    print('\n... begin open MZZ json')
    print(final_json_fname)
    fs = fsspec.filesystem(
        "reference", 
        fo=final_json_fname, 
        remote_protocol="s3", 
        remote_options={"anon":False, 'key':key, 'secret':aws_secret_access_key},
        skip_instance_cache=False
    )
    
    # open and dump contents of the final json fname
    m = fs.get_mapper("")

    print('\n\n dump of ds')
    ds = xr.open_dataset(m, engine='zarr', consolidated=False)
    print('\n... end open MZZ json')

    ds.close()
    pprint(ds)
