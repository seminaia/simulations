#!/bin/bash

# Submit all elastic constant calculations with directory-specific job names
MAIN_DIR="elastic_inputs"
JOB_SCRIPT="job.sh"

# Validate job script exists
if [ ! -f "$JOB_SCRIPT" ]; then
    echo "Error: $JOB_SCRIPT not found in current directory!"
    exit 1
fi

# Submit jobs for each deformation directory
for dir in "${MAIN_DIR}"/deformation_*; do
    if [ -d "$dir" ]; then
        # Get directory name without path
        dir_name=$(basename "$dir")
        
        # Copy job script to directory
        cp "$JOB_SCRIPT" "$dir/"
        
        # Submit job with directory-specific name
        echo "Submitting job: $dir_name"
        (
            cd "$dir" || exit
            sbatch -J "$dir_name" job.sh
        )
    fi
done

echo "All jobs submitted with directory-based names!"
# Validate job script exists
if [ ! -f "$JOB_SCRIPT" ]; then
    echo "Error: $JOB_SCRIPT not found in current directory!"
    exit 1
fi

# Submit jobs for each deformation directory
for dir in "${MAIN_DIR}"/deformation_*; do
    if [ -d "$dir" ]; then
        # Get directory name without path
        dir_name=$(basename "$dir")
        
        # Copy job script to directory
        cp "$JOB_SCRIPT" "$dir/"
        
        # Submit job with directory-specific name
        echo "Submitting job: $dir_name"
        (
            cd "$dir" || exit
            sbatch -J "$dir_name" job.sh
        )
    fi
done

echo "All jobs submitted with directory-based names!"
