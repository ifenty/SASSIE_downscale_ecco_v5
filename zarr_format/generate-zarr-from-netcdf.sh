#!/bin/bash

# there are ~2600 files in a full dataset

declare -a arr=("THETA")

## now loop through the above array
for x in "${arr[@]}"
do

  # for i in `seq 0 25`
  for i in `seq 1 25`
  do
    sn=$((i*100))
    en=$((sn+100))
    fsn=$(printf "%0*d" 3 $sn)
    fen=$(printf "%0*d" 3 $en)
    disk=$((sn / 160 +1))
    echo $sn $en $fsn $fen
    
    ## do not keep files saved to disk - need to manually add secret keys
    cmd="python convert-netcdf-to-zarr-store.py --var_name $x --s3_netcdf_dir s3://podaac-dev-sassie/ECCO_model/N1/V1/HH/NETCDF/ --zarr_dest_s3_name s3://podaac-dev-sassie/ECCO_model/N1/V1/HH/ZARR/ --sassie_key --sassie_secret --files_to_process $sn $en 1> ${x}_${fsn}_${fen}.log 2> ${x}_${fsn}_${fen}.err.log &"
        
    echo $cmd
    eval $cmd
    
  done
  
  wait
done
