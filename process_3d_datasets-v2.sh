#!/bin/bash

# takes two arguments, the first is the gz file index to start, the second is the gz file index to end
declare -a arr=("tr_diff_r_day_mean")
# declare -a arr=("tr_diff_r_day_mean")
# declare -a arr=("phi_3D_day_mean")
# declare -a arr=("KPP_mix_day_mean")

## now loop through the above array
for i in "${arr[@]}"
do
   echo "$i"
   python generate-sassie-ecco-netcdfs-s3-v2.py  --root_filenames $i  --root_s3_name s3://ecco-model-granules/SASSIE/N1/  --root_dest_s3_name s3://ecco-processed-data/SASSIE/N1/V1/HH/NETCDF/   --files_to_process 0 1  -l /nvme_data11  --push_to_s3  --save_nc_to_disk
done