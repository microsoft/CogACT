# CogACT Azure ML integration

In what follows we provide instructions to set up the CogACT environment in Azure ML (AML), and to run CogACT inference and fine-tuning jobs using AML clusters as compute target.

## Setting up the environment

In [docker/Dockerfile.train.multistage](../../docker/Dockerfile.train.multistage) we provide a Docker file to build an environment which can be used for both inference and training. The environment is based on the [pyproject.toml](../../pyproject.toml) file and mirrors the instructions for installing the [train] conda environment as provided in the main [README](../../README.md).

The installation of the flash attention library is lengthy and complex and may lead to time-out errors if one attemps to build the corresponding Azure ML environment directly. For this reason, we recommend following these steps instead:

1. Build the image locally, preferrably on a Virtual Machine (VM) with the same architecture as the target compute cluster:

    ```bash
    docker build -f docker/Dockerfile.train.multistage -t cogact-train .
    ```

2. Push the image to the Azure Container Registry (ACR):

    Set your Azure details

    ```bash
    RESOURCE_GROUP=<your resource group name>
    WORKSPACE_NAME=<your AML workspace name>
    ```

    Get the ACR details from the AML workspace

    ```bash
    ACR_INFO=$(az ml workspace show --resource-group $RESOURCE_GROUP --name $WORKSPACE_NAME --query container_registry -o tsv)
    ACR_NAME=$(echo $ACR_INFO | cut -d'/' -f9 | cut -d'.' -f1)
    ```

    List available Docker images (optional)

    ```console
    docker images | grep cogact
    ```

    Choose your local image (substitute your actual image name here)

    ```console
    LOCAL_IMAGE=cogact-train:latest
    ```

    Tag the image for ACR

    ```console
    ACR_IMAGE="${ACR_NAME}.azurecr.io/cogact-train:latest"
    docker tag $LOCAL_IMAGE $ACR_IMAGE
    ```

    Login to ACR

    ```console
    az acr login --name $ACR_NAME
    ```

    Push the image to ACR

    ```console
    docker push $ACR_IMAGE
    ```

3. Create AML environment YAML config using the ACR image:

    ```bash
    cat > acr-environment.yml << EOF
    \$schema: https://azuremlschemas.azureedge.net/latest/environment.schema.json
    name: <your chosen environment name>
    version: 1
    image: $ACR_IMAGE
    description: CogACT environment using pre-built image from ACR
    EOF
    ```

4. Register the environment in AML:

    ```bash
    az ml environment create --file acr-environment.yml --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME
    ```

## Getting access to gated repos

Running e.g. inference requires access to gated repos such as Llama 2. Follow these steps to set up the access:

1. Get a Hugging Face account (if you don't have one already)

2. Request access to Llama 2:

    2.1 Go to <https://huggingface.co/meta-llama/Llama-2-7b-hf>

    2.2 Click "Access repository"

    2.3 Fill out Meta's form with your details

    2.4 Wait for approval (usually quick)

3. Create an access token:

    3.1 Go to <https://huggingface.co/settings/tokens>

    3.2 Create a new token with read permissions

    3.3 Copy the token

## Optional (but recommended): downloading and cache-ing CogACT base checkpoints

For faster execution, we recommend to download and cache the CogACT base checkpoints. If not found locally, the code will attempt to download the chosen base checkpoint from Hugging Face, which might lead to timeout errors depending on network speed.

Here is an example of how to download and cache the base checkpoints using the Docker image built in step 1:

```bash
sudo mkdir -p /mnt/data/cogact_checkpoints
sudo chmod -R 777 /mnt/data
HF_TOKEN=<your_huggingface_token_here>
docker run --rm -it \
  -e HUGGINGFACE_TOKEN=$HF_TOKEN \
  -e HF_TOKEN=$HF_TOKEN \
  -v /mnt/data:/mnt/data \
  cogact-train \
  python -c "from vla import load_vla; load_vla('CogACT/CogACT-Base', cache_dir='/mnt/data/cogact_checkpoints', hf_token='$HF_TOKEN')"
```

For later use, we recommend uploading these checkpoints to a folder in an Azure Data Lake Gen 2 container. In what follows, and in the accompanying YAML job configuration files, we will use the placeholder `<hf_cache_folder_name>` for this folder's name. One convenient way to upload the checkpoint is to create a container inside the Data Lake Gen 2 storage account and mount it in a local directory (e.g. /mnt/azure-storage/) using the blobfuse configuration file we provide [here](blofuse_config.yaml) as follows:

```bash
sudo mkdir -p /mnt/azure-storage/

sudo chown $(whoami) /mnt/azure-storage/

blobfuse2 mount /mnt/azure-storage/ --config-file=blobfuse_config.yaml --allow-other
```

after which we can create the `<hf_cache_folder_name>` folder and move the checkpoint files there. **Note:** the first time the blobfuse command is run, one may additionally need to modify the `/etc/fuse.conf` file as follows: run `sudo nano /etc/fuse.conf`, uncomment this line: `#user_allow_other`, save and exit (e.g. Ctrl+O, Enter, then Ctrl+X in nano).

Once the data has been uploaded, register the container as a [Datastore](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-connect-data-ui?view=azureml-api-1&tabs=credential) in AML.

## Running a minimal inference example in AML

A minimal inference example is provided in the "Getting Started" section of the main [README](../../README.md). In [inference_aml.py](inference_aml.py) we have slightly adapted the code to enable logging and running in AML.

**Notes:**

1. The minimal inference example requires an image as input, preferrably consistent with the "move sponge near apple" task. We have provided one such image in [test_image.png](test_image.png).
2. To run the minimal inference example locally (instead of in an AML cluster), one may simply point the `cache_dir` variable in the inference script to the local folder containing the cached checkpoints (if available), and run it as follows `docker run --gpus all -e HF_TOKEN=<your_hf_token_here> -v /CogACT:/app -v <checkpoint_directory>:<checkpoint_directory> -w /app cogact-train:latest python inference.py
`. Warning: this may take a long time (~1 hour) the first time the command is run.

The [inference-job-config.yml](inference-job-config.yml) file contains the configuration for running the inference job in an AML cluster (fill in the placeholders according to your setup). One can then submit the job as follows:

```bash
export HF_TOKEN=<your HF token>
az ml job create --file inference-job-config.yml --name <job name> --resource-group <resource group name> --workspace-name <workspace name> --set environment_variables.HF_TOKEN="$HF_TOKEN"
```

## Running CogACT fine-tuning in AML

The CogACT fine-tuning job utilizes the [train.py](../train.py) script, and can be triggered locally following the instructions in the main [README](../../README.md). Here we provide two files which enable running CogACT fine-tuning as an AML job using the Azure CLI:

1. [finetune-job-config.yaml](finetune-job-config.yaml): basic configuration for the AML job.
2. [train_aml.py](train_aml.py): a wrapper for the original [train.py](../train.py) script which implements the necessary integration with AML.

The AML environment for the training job is created according to the instructions in the section [Setting up the environment](#setting-up-the-environment) above.

The [finetune-job-config.yaml](finetune-job-config.yaml) file specifies the following input/output mount points:

| Component | Datastore | Path  | Mode | Purpose |
|-----------|-----------|------|------|---------|
| **Finetuning Dataset** | `<datastore_name>` | `<dataset_folder_name>` | `ro_mount` | Fine-tuning data, e.g. one or more datasets from the open-x embodiment collection |
| **CogACT checkpoints and HF models (Input)** | `<datastore_name>` | `<hf_cache_folder_name>` | `rw_mount` | Pre-downloaded CogACT base checkpoints. Also HF models (including Llama2, etc.) which will be downloaded and cached the first time the fine-tuning job is executed |
| **Training Outputs** |  `<datastore_name>` | `<outputs_folder_name>` | `rw_mount` | Fine-tuned model checkpoints saved at regular intervals during training|

Below we describe the main contents of the [finetune-job-config.yaml](finetune-job-config.yaml) job configuration file.

### AML Setup and Environment Variables

- `environment` → the name and version of the AML environment registered as described in [Setting up the environment](#setting-up-the-environment), e.g. `azureml:<aml env_name>:<env version>`
- `compute`  → name of AML compute cluster where the job will be run, e.g. `azureml:<your_compute_name>`
- `HF_TOKEN` → HuggingFace token
- `WANDB_API_KEY` → Weights & Biases API key

### Input Configuration

#### 1. The fine-tuning dataset (Read-Only mount)

```yaml
inputs:
  data_root_dir:
    type: uri_folder
    path: azureml://datastores/<datastore_name>/paths/<dataset_folder_name>
    mode: ro_mount
```

Section [Open X-Embodiment Dataset Download Guide](#optional-open-x-embodiment-dataset-download-guide) below provides detailed instructions to download one or more datasets from the Open X-Embodiment collection, which can be used to finne-tune CogACT.

**Expected Structure:**

The hierarchy must be TFDS-compliant, e.g.

```
dataset_dir/
└── stanford_kuka_multimodal_dataset_converted_externally_to_rlds/
    └── 0.1.0/
        ├── dataset_info.json
        ├── features.json
        └── *.tfrecord files
```

#### 2. CogACT Checkpoints and cached HF models (Read-Write mount)

Here we point to pre-downloaded base CogACT checkpoints if the steps in section [downloading and cache-ing CogACT checkpoints
](#optional-but-recommended-downloading-and-cache-ing-cogact-checkpoints) were followed. For simplicity, we use the same datastore to cache the rest of the base models which will be downloaded from Hugging Face the first time the fine-tuning job is run:

```yaml
inputs:
  hf_home:
    type: uri_folder
    path: azureml://datastores/<datastore_name>/paths/<hf_cache_folder_name>
    mode: ro_mount
```

### Output Configuration

#### 1. Training Checkpoints (Read-Write mount)

This is where the fine-tuned checkpoints and other metadata and ouputs from the training job will be written to:

```yaml
outputs:
  run_root_dir:
    type: uri_folder
    path: azureml://datastores/<datastore_name>/paths/<outputs_folder_name>
    mode: rw_mount
```

### Job submission

Once the necessary datastores have been created and the placeholders in [finetune-job-config.yaml](finetune-job-config.yaml) have been filled according to your environment, the fine-tuning job can be submitted as follows:

```bash
# Set required environment variables
export HF_TOKEN=<your_huggingface_token>
export WANDB_API_KEY=<your_wandb_api_key>  # Optional

# Submit job to AzureML
az ml job create \
  --file scripts/aml/finetune-job-config.yml \
  --resource-group <your-resource-group> \
  --workspace-name <your-workspace-name> \
  --set environment_variables.HF_TOKEN="$HF_TOKEN" \
  --set environment_variables.WANDB_API_KEY="$WANDB_API_KEY"
```

and the training metrics can be tracked in wandb (integration with mlflow, the native Azure ML logging library, will be provided elsewhere). **Note:** the first time the job run the various required models (beyond the CogACT base checkpoints, such as Llama 2, etc.) will be downloaded and cached in the specified AML datastore, which can take of the order of hours depending on network speed. Subsequent runs will take advantange of the cached models and will proceed much faster.

## [Optional] Open X-Embodiment Dataset Download Guide

This section describes how to download robotics datasets from the Open X-Embodiment collection to an Azure storage account mounted on an Azure ML compute instance.

### Prerequisites

- Azure VM with managed identity or Azure CLI access
- Sufficient storage space (datasets range from MB to hundreds of GB)

### Step 1: Mount Azure Storage

#### 1.1 Configure blobfuse2

Create a configuration file for blobfuse2:

```yaml
# blobfuse_config.yaml
version: 2
logging:
  type: syslog
components:
  - libfuse
  - file_cache
  - azstorage
libfuse:
  allow-other: true
  uid: 1000 # Replace with your user id if different
  gid: 1000 # Replace with your group id if different
  umask: 0002
file_cache:
  path: /tmp/blobfuse_cache
  timeout-sec: 120
  max-size-mb: 4000
azstorage:
  type: adls
  account-name: <your-storage-account-name>
  container: <your-container-name>
  auth-type: azcli  # or 'msi' if using managed identity
```

**Configuration parameters:**

- `file_cache`: acts as a local buffer to enable tools like `gsutil` to write temporary files during downloads
- `path`: Local directory for cache (ensure sufficient space)
- `timeout-sec`: How long to keep files in cache before sync
- `max-size-mb`: Maximum cache size (adjust based on available disk space)

#### 1.2 Create mount directory and cache

```bash
# Create mount point
sudo mkdir -p /mnt/azure-storage

# Create cache directory
mkdir -p /tmp/blobfuse_cache
```

#### 1.3 Mount the storage

```bash
# Mount using Azure CLI authentication
sudo -E -u $USER blobfuse2 mount /mnt/azure-storage --config-file=./blobfuse_config.yaml

# Verify mount
mount | grep azure-storage
ls -la /mnt/azure-storage/
```

#### 1.4 Test write permissions

```bash
# Test writing to the mount
echo "test" > /mnt/azure-storage/test.txt
cat /mnt/azure-storage/test.txt
rm /mnt/azure-storage/test.txt
```

### Step 2: Download Datasets

#### 2.1 Available Datasets

The Open X-Embodiment collection includes 55+ robotics datasets. Here are some popular ones:

| Dataset Name | Size | Description |
|--------------|------|-------------|
| stanford_kuka_multimodal | 31.97 GB | Kuka iiwa peg insertion with force feedback |
| jaco_play | 9.23 GB | Jaco arm manipulation tasks |
| berkeley_cable_routing | 4.66 GB | Cable routing tasks |
| columbia_cairlab_pusht_real | 2.80 GB | Pushing tasks |
| nyu_franka_play_dataset | 5.18 GB | Franka arm play data |

For a complete list with sizes, run:

```bash
./check_dataset_sizes.sh
```

#### 2.2 Download Script

Use the provided download script to get datasets:

```bash
# Make script executable
chmod +x download_with_gsutil.sh

# Run the download
./download_with_gsutil.sh
```

#### 2.3 Customize Downloads

Edit `download_with_gsutil.sh` to specify which datasets to download:

```bash
# List of datasets to download with their GCS paths
declare -A datasets=(
    ["stanford_kuka_multimodal_dataset_converted_externally_to_rlds"]="gs://gresearch/robotics/stanford_kuka_multimodal_dataset_converted_externally_to_rlds/0.1.0"
    ["jaco_play"]="gs://gresearch/robotics/jaco_play/0.1.0"
    ["bridge"]="gs://gresearch/robotics/bridge/0.1.0"
    # Add more datasets as needed
)
```

**Note**: Use the full dataset name as the key to preserve the original GCS bucket structure.

### Step 3: Dataset Structure

Downloaded datasets will be organized in the original GCS bucket structure:

```
/mnt/azure-storage/
└── open-x-embodiment/
    ├── stanford_kuka_multimodal_dataset_converted_externally_to_rlds/
    │   └── 0.1.0/
    │       ├── dataset_info.json
    │       ├── features.json
    │       └── *.tfrecord files
    ├── jaco_play/
    │   └── 0.1.0/
    │       ├── dataset_info.json
    │       ├── features.json
    │       └── *.tfrecord files
    └── other_datasets/
        └── 0.1.0/
            └── ...
```

This structure preserves the original GCS bucket layout, including version numbers, making it easier to work with existing tools and scripts that expect this format.

### Clean Up

#### Unmount storage

```bash
sudo umount /mnt/azure-storage
```

#### Remove cache

```bash
rm -rf /tmp/blobfuse_cache
```
