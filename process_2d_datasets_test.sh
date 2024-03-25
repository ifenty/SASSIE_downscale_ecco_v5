#!/bin/bash

# takes two arguments, the first is the gz file index to start, the second is the gz file index to end
declare -a arr=("ocean_state_2D_day_mean"
                "KPP_hbl_day_mean")


## now loop through the above array
for i in "${arr[@]}"
do
   echo "$i $1 $2"
   python generate-sassie-ecco-netcdfs-s3.py --root_filenames $i --root_s3_name s3://ecco-model-granules/SASSIE/N1/ --root_dest_s3_name s3://ecco-processed-data/tmp/ --push_to_s3 --save_nc_to_disk --files_to_process $1 $2 -l /tmp/sassie
done

