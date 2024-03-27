#!/bin/bash
aws s3 ls s3://ecco-processed-data/SASSIE/N1/V1/HH/NETCDF/ > s3_dirs.txt
awk '/AVG_DAILY/ {print $2}' s3_dirs.txt | sed 's/.$//' > s3_dirs.clean
