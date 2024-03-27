#!/bin/bash

# The file to split
filename=$1

# Calculate the total number of lines in the file
total_lines=$(wc -l < "$filename")
# Calculate lines per split; using 'bc' to get ceiling value if not divisible evenly
lines_per_split=$(echo "($total_lines+3)/4" | bc)

# Use 'split' to divide the file into 4 parts
# -l specifies the number of lines per split file
# --numeric-suffixes starts the suffixes at 0
# The last argument is the prefix for the output files
split -l "$lines_per_split" --numeric-suffixes=1 --additional-suffix=".txt" "$filename" "part_"

echo "File '$filename' has been split into 4 parts."
