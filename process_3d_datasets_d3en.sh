#!/bin/bash

# takes two arguments, the first is the gz file index to start, the second is the gz file index to end
#declare -a arr=("ocean_vel_day_mean" "KPP_mix_day_mean" "tr_adv_r_day_mean")
declare -a arr=("KPP_mix_day_mean")

## now loop through the above array
for x in "${arr[@]}"
do

  for i in `seq 0 28`
  do
    sn=$((i * 7))
    en=$((sn+7))
    fsn=$(printf "%0*d" 3 $sn)
    fen=$(printf "%0*d" 3 $en)
    disk=$((sn / 14 +1))
    echo $sn $en $fsn $fen $disk
    #s3://ecco-processed-data/SASSIE/N1/V1/HH/NETCDF/ 
    python generate-sassie-ecco-netcdfs-s3.py --root_filenames $x --root_s3_name s3://ecco-model-granules/SASSIE/N1/ --root_dest_s3_name s3://ecco-processed-data/SASSIE/N1/V1/HH/NETCDF/ --files_to_process $sn $en  -l /data${disk} --push_to_s3 --save_nc_to_disk 1> ${x}_${fsn}_${fen}.log 2> ${x}_${fsn}_${fen}.err.log &

  done

  # If wait is called without any arguments, it waits for all currently active child processes to complete.
  wait
done


