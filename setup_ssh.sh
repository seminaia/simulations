#!/bin/bash

# Setup SSH public key authentication for HPC access
# This enables passwordless SSH/rsync for syncing.sh

HPC_USER="ssem"
HPC_ADDRESS="stampede3.tacc.utexas.edu"
KEY_FILE="$HOME/.ssh/id_ed25519"

echo "Setting up SSH public key authentication for $HPC_USER@$HPC_ADDRESS"
echo "----------------------------------------------------------------------"

# Generate SSH key pair if it doesn't already exist
if [ ! -f "$KEY_FILE" ]; then
    echo "Generating new ED25519 SSH key pair at $KEY_FILE ..."
    ssh-keygen -t ed25519 -f "$KEY_FILE" -C "$HPC_USER@$HPC_ADDRESS"
else
    echo "SSH key already exists at $KEY_FILE — skipping key generation."
fi

# Copy the public key to the remote HPC server
echo ""
echo "Copying public key to $HPC_USER@$HPC_ADDRESS ..."
echo "(You will be prompted for your HPC password one last time.)"
ssh-copy-id -i "${KEY_FILE}.pub" "$HPC_USER@$HPC_ADDRESS"

echo ""
echo "Done! Test the passwordless connection with:"
echo "  ssh $HPC_USER@$HPC_ADDRESS"
