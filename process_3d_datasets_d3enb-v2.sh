#!/bin/bash

# takes two arguments, the first is the gz file index to start, the second is the gz file index to end
#declare -a arr=("ocean_vel_day_mean" "KPP_mix_day_mean" "tr_adv_r_day_mean")
#declare -a arr=("KPP_mix_day_mean")
#declare -a arr=("vol_adv_day_mean" "phi_3D_day_mean")
#declare -a arr=("phi_3D_day_mean")
declare -a arr=("vol_adv_day_mean")
# declare -a arr=("tr_diff_x_day_mean")

## now loop through the above array
for x in "${arr[@]}"
do

  for i in `seq 0 19`
  do
    sn=$((i))
    en=$((sn+1))
    fsn=$(printf "%0*d" 3 $sn)
    fen=$(printf "%0*d" 3 $en)
    disk=$((sn / 2 +1))
    echo $sn $en $fsn $fen $disk
    #s3://ecco-processed-data/SASSIE/N1/V1/HH/NETCDF/ 
    cmd="python generate-sassie-ecco-netcdfs-s3-v2.py --root_filenames $x --root_s3_name s3://ecco-model-granules/SASSIE/N1/ --root_dest_s3_name s3://podaac-dev-sassie/ECCO_model/N1/V1/HH/NETCDF/  --files_to_process $sn $en -l data$3 --push_to_s3 --save_nc_to_disk 1> ${x}_${fsn}_${fen}.log 2> ${x}_${fsn}_${fen}.err.log &"
    echo $cmd
    eval $cmd

  done

  # If wait is called without any arguments, it waits for all currently active child processes to complete.
  wait
done


