#!/bin/bash

# Configuration
HPC_USER="ssem"
HPC_ADDRESS="stampede3.tacc.utexas.edu"
REMOTE_BASE_DIR="/work2/10291/ssem/stampede3/scratch/NNO/Nd2NiO4_I4_mmm/Nd2NiO4_vacancies"  # Base directory on HPC
LOCAL_DIR="./"  # Local directory to sync to (current dir by default)

# Safety checks
[ -z "$HPC_USER" ] && { echo "Error: HPC_USER not configured"; exit 1; }
[ -z "$HPC_ADDRESS" ] && { echo "Error: HPC_ADDRESS not configured"; exit 1; }
[ -z "$REMOTE_BASE_DIR" ] && { echo "Error: REMOTE_BASE_DIR not configured"; exit 1; }

echo "Starting sync from HPC server..."
echo "Remote base: $HPC_USER@$HPC_ADDRESS:$REMOTE_BASE_DIR"
echo "Local destination: $LOCAL_DIR"
echo "----------------------------------------"

# Find all v_*/vasp_gam directories and sync them
for dir in $(ssh $HPC_USER@$HPC_ADDRESS "ls -d $REMOTE_BASE_DIR/v_*/vasp_gam/vasprun.xml.gz 2>/dev/null"); do
    echo "Syncing: $dir"
    
    # Dry-run first (remove --dry-run when ready for real sync)
    rsync -avzP --relative --dry-run \
        $HPC_USER@$HPC_ADDRESS:"$REMOTE_BASE_DIR/./${dir#$REMOTE_BASE_DIR/}" \
        "$LOCAL_DIR"
    
    # Uncomment for actual sync
    # rsync -avzP --relative \
    #     $HPC_USER@$HPC_ADDRESS:"$REMOTE_BASE_DIR/./${dir#$REMOTE_BASE_DIR/}" \
    #     "$LOCAL_DIR"
    
    echo "----------------------------------------"
done

echo "Sync complete. Remember to remove --dry-run for actual transfer!"