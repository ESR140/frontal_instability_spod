#!/bin/bash

# Source the config file
if [ -f "./config.sh" ]; then
    source ./config.sh
else
    echo "Error: config.sh not found. Please create it based on config.sample.sh."
    exit 1
fi

# Check if the required environment variables are set
if [ -z "$INPUT_PATH" ] || [ -z "$BASE_OUTPUT_PATH" ]; then
    echo "Error: INPUT_PATH and BASE_OUTPUT_PATH must be set in config.sh."
    exit 1
fi

# Use the environment variables
input_path="${INPUT_PATH}"
base_output_path="${BASE_OUTPUT_PATH}"

# Define an array of plane types and indices
declare -a planes=(
    "XY 97"
    "XY 95"
    "YZ 128"
    "YZ 64"
    "XZ 512"
)

# Loop through the planes array and run the Python scripts for each
for plane in "${planes[@]}"; do
    # Split the plane string into type and index
    plane_type=$(echo $plane | cut -d' ' -f1)
    plane_index=$(echo $plane | cut -d' ' -f2)
    
    # Create the specific output folder
    output_folder="${plane_type}${plane_index}"
    full_output_path="${base_output_path}/${output_folder}"
    
    # Create the output directory if it doesn't exist
    mkdir -p "$full_output_path"
    
    # Run the vorticity_calc.py script
    python3 vorticity_calc.py "$input_path" "$full_output_path" --plane_type "$plane_type" --plane_index "$plane_index"
    
    # Check if vorticity_calc.py was successful
    if [ $? -eq 0 ]; then
        echo "Completed vorticity calculation for $plane_type plane, index $plane_index. Output in $full_output_path"
        
        # Construct the vorticity output filename
        vorticity_file="vorticity_${plane_type}_${plane_index}"
        
        # Run the pod.py script with the output of vorticity_calc.py as input
        python3 pod.py "${full_output_path}/${vorticity_file}" "$full_output_path" --plane_type "$plane_type"
        
        if [ $? -eq 0 ]; then
            echo "Completed POD analysis for $plane_type plane, index $plane_index"
        else
            echo "Error: POD analysis failed for $plane_type plane, index $plane_index"
        fi
    else
        echo "Error: Vorticity calculation failed for $plane_type plane, index $plane_index"
    fi
done

echo "All calculations and analyses completed."