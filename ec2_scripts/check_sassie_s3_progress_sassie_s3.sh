#!/bin/bash

echo ""
date
aws s3 ls s3://podaac-dev-sassie/ECCO_model/N1/V1/HH/NETCDF/ --recursive --profile sassie | awk '{print $4}' | awk -F'/' 'NF>1{print $(NF-1)}' | sort | uniq -c
echo ""
