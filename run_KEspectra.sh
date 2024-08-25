#!/bin/bash

# Array of plane indices you want to process
indices=(83 85 87 89 91 93 95 97 99)  # Add or modify indices as needed

# Input file path
input_file="/media/user/Extreme SSD/SIMULATION_DATA/XYZ_SS.nc"

# Base output directory
output_dir="/media/user/Extreme SSD/SIMULATION_DATA/KEresults"

# Loop through each index
for index in "${indices[@]}"
do
    # Create the output filename
    output_file="${output_dir}"
    
    # Run the Python script with the current index
    python3 KEspectra.py "$input_file" "$output_file" --plane_index "$index"
    
    echo "Processed plane index $index"
done

echo "All plane indices processed"