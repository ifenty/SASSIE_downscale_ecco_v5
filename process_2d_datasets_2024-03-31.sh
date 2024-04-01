#!/bin/bash

declare -a arr=("KPP_hbl_day_mean") 

## now loop through the above array
for x in "${arr[@]}"
do

  for i in `seq 0 16`
  do
    #sn=$((i*12))
    #en=$((sn+12))
    sn=$((180+i))
    en=$((180+i))
    fsn=$(printf "%0*d" 3 $sn)
    fen=$(printf "%0*d" 3 $en)
    disk=$((i / 2 +1))
    echo $sn $en $fsn $fen $disk
    export es3="s3://podaac-dev-sassie/ECCO_model/N1/V1/HH/NETCDF/"
    cmd="python generate-sassie-ecco-netcdfs-s3.py  --root_filenames $x  --root_s3_name s3://ecco-model-granules/SASSIE/N1/  --root_dest_s3_name ${es3}   --files_to_process $sn  -l /nvme_data${disk}  --push_to_s3 --keep_local_files  --save_nc_to_disk 1> ${x}_${fsn}_${fen}.log 2> ${x}_${fsn}_${fen}.err.log &"
    echo $cmd
    eval $cmd

  done

  # If wait is called without any arguments, it waits for all currently active child processes to complete.
  wait
done


