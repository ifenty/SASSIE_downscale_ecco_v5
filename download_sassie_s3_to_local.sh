#!/bin/sh

## list all variable folders that need to be transferred
declare -a arr=('ADVr_SLT_AVG_DAILY'
 'ADVr_TH_AVG_DAILY'
 'ADVxHEFF_AVG_DAILY'
 'ADVxSNOW_AVG_DAILY'
 'ADVx_SLT_AVG_DAILY'
 'ADVyHEFF_AVG_DAILY'
 'ADVySNOW_AVG_DAILY'
 'ADVy_SLT_AVG_DAILY'
 'ADVy_TH_AVG_DAILY'
 'DFrE_SLT_AVG_DAILY'
 'DFrE_TH_AVG_DAILY'
 'DFrI_SLT_AVG_DAILY'
 'DFrI_TH_AVG_DAILY'
 'ETAN_AVG_DAILY'
 'EXFaqh_AVG_DAILY'
 'EXFatemp_AVG_DAILY'
 'EXFempmr_AVG_DAILY'
 'EXFevap_AVG_DAILY'
 'EXFhl_AVG_DAILY'
 'EXFhs_AVG_DAILY'
 'EXFlwdn_AVG_DAILY'
 'EXFlwnet_AVG_DAILY'
 'EXFpreci_AVG_DAILY'
 'EXFqnet_AVG_DAILY'
 'EXFroff_AVG_DAILY'
 'EXFswdn_AVG_DAILY'
 'EXFswnet_AVG_DAILY'
 'EXFtaux_AVG_DAILY'
 'EXFtauy_AVG_DAILY'
 'EXFvwind_AVG_DAILY'
 'KPPdiffS_AVG_DAILY'
 'KPPviscA_AVG_DAILY')


## now loop through the above array
for i in "${arr[@]}"
do
   echo "$i"
   cd /Volumes/SASSIE_1
   mkdir $i
   aws s3 sync s3://podaac-dev-sassie/ECCO_model/N1/V1/HH/NETCDF/${i}/ /Volumes/SASSIE_1/${i} --profile sassie
done

