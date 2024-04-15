#!/bin/bash

# there are ~2600 files in a full dataset

# declare -a arr=("THETA" "SALT" "PHIHYD" "RHOAnoma" "DFrI_SLT" "DFrI_TH")
# declare -a arr=("UVEL" "VVEL" "WVEL" "KPPdiffS" "KPPviscA" "ADVr_SLT" "ADVr_TH")
# declare -a arr=("ADVx_SLT" "ADVx_TH" "ADVy_SLT" "ADVy_TH")

# declare -a arr=("UVELMASS" "VVELMASS"  "WVELMASS") # change seq from 0 to 24
# declare -a arr=("KPPdiffT")
# declare -a arr=("PHIHYDcR")
# declare -a arr=("DFrE_SLT" "DFrE_TH")
declare -a arr=("DFxE_SLT" "DFxE_TH" "DFyE_SLT" "DFyE_TH")

## now loop through the above array
for x in "${arr[@]}"
do

  # for i in `seq 0 25` # for complete dataset
  for i in `seq 0 4`
  do
    sn=$((i*100))
    en=$((sn+100))
    fsn=$(printf "%0*d" 3 $sn)
    fen=$(printf "%0*d" 3 $en)
    disk=$((sn / 160 +1))
    
    echo $sn $en $fsn $fen $disk
    ## keep files saved to disk
    #     cmd="python extract-surface-layer-from-sassie-3d-granules.py --var_3d $x --sassie_s3_netcdf_dir s3://podaac-dev-sassie/ECCO_model/N1/V1/HH/NETCDF/ --ec2_nvme_scratch_dir /nvme_data${disk} --sassie_key --sassie_secret --dest_s3_name s3://podaac-dev-sassie/ECCO_model/N1/V1/HH/NETCDF_3D_SURF/  --files_to_process $sn $en  --push_to_s3 --save_nc_to_disk 1> ${x}_${fsn}_${fen}.log 2> ${x}_${fsn}_${fen}.err.log &"

    ## do not keep files saved to disk - need to manually add secret keys
        cmd="python extract-surface-layer-from-sassie-3d-granules.py --var_3d $x --sassie_s3_netcdf_dir s3://podaac-dev-sassie/ECCO_model/N1/V1/HH/NETCDF/ --ec2_nvme_scratch_dir /nvme_data${disk} --sassie_key --sassie_secret --dest_s3_name s3://podaac-dev-sassie/ECCO_model/N1/V1/HH/NETCDF_3D_SURF/  --files_to_process $sn $en  --push_to_s3 --save_nc_to_disk 1> ${x}_${fsn}_${fen}.log 2> ${x}_${fsn}_${fen}.err.log &"
    
    echo $cmd
    eval $cmd

  done

  # If wait is called without any arguments, it waits for all currently active child processes to complete.
  wait
done
