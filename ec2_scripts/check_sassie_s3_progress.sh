#!/bin/bash

echo ""
date
aws s3 ls s3://ecco-processed-data/SASSIE/N1/V1/HH/NETCDF/ --recursive | awk '{print $4}' | awk -F'/' 'NF>1{print $(NF-1)}' | sort | uniq -c
echo ""
