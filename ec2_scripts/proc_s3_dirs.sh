#!/bin/bash

# Check if the filename was provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 filename"
    exit 1
fi

# Assign the first argument to a variable
filename="$1"

# Check if the file exists
if [[ -f "$filename" ]]; then
    # Read the file into an array
    mapfile -t lines < "$filename"
    
    # Now you can access the lines from the 'lines' array
    for i in "${!lines[@]}"; do
        cur_var=${lines[$i]}
        echo "Line $((i+1)): ${lines[$i]} $cur_var"
        rm -fr ${cur_var}
        aws s3 sync s3://ecco-processed-data/SASSIE/N1/V1/HH/NETCDF/${cur_var}/ ${cur_var} 
        ~/git_repos/SASSIE_downscale_ecco_v5/ec2_scripts/fix_metadata.sh /nvme_data1/${cur_var}/
        last_file=$(ls -1 ${cur_var} | tail -n 1)
        ncdump -h ${cur_var}/${last_file} |tail -5
        aws s3 sync ${cur_var} s3://podaac-dev-sassie/ECCO_model/N1/V1/HH/NETCDF/${cur_var}/ --profile sassie
        rm -fr ${cur_var}
    done
else
    echo "Error: File '$filename' not found."
fi


