#!/bin/bash

# there are ~2600 files in a full dataset

declare -a arr=("THETA")

## now loop through the above array
for x in "${arr[@]}"
do
    echo $x
    ## do not keep files saved to disk - need to manually add secret keys
        cmd="python convert-netcdf-to-zarr-store.py --var_name $x --s3_netcdf_dir s3://podaac-dev-sassie/ECCO_model/N1/V1/HH/NETCDF/ --zarr_dest_s3_name s3://podaac-dev-sassie/ECCO_model/N1/V1/HH/ZARR/ --sassie_key --sassie_secret --files_to_process 0 2 1> ${x}.log 2> ${x}.err.log &"
    
    echo $cmd
    eval $cmd

done