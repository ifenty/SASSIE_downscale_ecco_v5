#!/bin/bash

# takes two arguments, the first is the gz file index to start, the second is the gz file index to end
declare -a arr=("ocean_state_2D_day_mean"
                "KPP_hbl_day_mean"
                "EXF_day_mean"
                "oce_flux_day_mean"
                "seaice_flux_day_mean"
                "seaice_state_day_mean"
                "seaice_vel_day_mean"
                "tr_adv_x_2D_day_mean")


## now loop through the above array
for i in "${arr[@]}"
do
   echo "$i $1 $2"
   python generate-sassie-ecco-netcdfs-s3.py --root_filenames $i --root_s3_name s3://ecco-model-granules/SASSIE/N1/ --root_dest_s3_name s3://ecco-processed-data/SASSIE/N1/V1/HH/NETCDF/  --files_to_process $1 $2 --push_to_s3
done

