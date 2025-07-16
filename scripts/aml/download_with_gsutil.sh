#!/bin/bash

# Define target directory
TARGET_DIR="/mnt/azure-storage/open-x-embodiment"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# List of datasets to download with their GCS paths
declare -A datasets=(
    ["stanford_kuka_multimodal_dataset_converted_externally_to_rlds"]="gs://gresearch/robotics/stanford_kuka_multimodal_dataset_converted_externally_to_rlds/0.1.0"
)

# Check if gsutil is installed
if ! command -v gsutil &> /dev/null; then
    echo "gsutil is not installed. Installing Google Cloud SDK..."
    # Install Google Cloud SDK
    curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-latest-linux-x86_64.tar.gz
    tar -xf google-cloud-cli-latest-linux-x86_64.tar.gz
    ./google-cloud-sdk/install.sh --quiet
    source ./google-cloud-sdk/path.bash.inc
    gcloud init
fi

for dataset in "${!datasets[@]}"; do
    echo "Downloading $dataset dataset from ${datasets[$dataset]}..."
    
    # Create dataset directory with proper nested structure
    mkdir -p "$TARGET_DIR/$dataset/0.1.0"
    
    # Download using gsutil with multithreading and progress
    echo "Starting download of $dataset (this may take a while)..."
    gsutil -m cp -r "${datasets[$dataset]}/*" "$TARGET_DIR/$dataset/0.1.0/"
    
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded $dataset to $TARGET_DIR/$dataset/0.1.0"
        # Show download summary
        echo "Files in $dataset:"
        ls -lah "$TARGET_DIR/$dataset/0.1.0/" | head -10
        echo "Total size of $dataset:"
        du -sh "$TARGET_DIR/$dataset/0.1.0"
    else
        echo "Failed to download $dataset"
    fi
    echo "----------------------------------------"
done

echo "Download process completed."
