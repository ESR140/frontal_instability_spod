#!/bin/bash

# Check if the required environment variables are set
if [ -z "$INPUT_PATH" ] || [ -z "$BASE_OUTPUT_PATH" ]; then
    echo "Error: INPUT_PATH and BASE_OUTPUT_PATH must be set in config.sh."
    exit 1
fi

# Use the environment variables
input_path="${INPUT_PATH}"
output_dir="${BASE_OUTPUT_PATH}"

# Array of plane indices you want to process
indices=(83 85 87 89 91 93 95 97 99)  # Add or modify indices as needed

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