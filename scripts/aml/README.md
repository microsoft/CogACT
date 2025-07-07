# CogACT Azure ML integration

In what follows we provide instructions to set up the CogACT environment in Azure ML (AML), and to run CogACT inference and fine-tuning jobs using AML clusters as compute target.

## Setting up the environment
In [docker/Dockerfile.train.multistage](../../docker/Dockerfile.train.multistage) we provide a Docker file to build an environment which can be used for both inference and training. The environment is based on the [pyproject.toml](../../pyproject.toml) file and mirrors the instructions for installing the [train] conda environment as provided in the main [README](../../README.md).

The installation of the flash attention library is lengthy and complex and may lead to time-out errors if one attemps to build the corresponding Azure ML environment directly. For this reason, we recommend following these steps instead:


1. Build the image locally, preferrably on a Virtual Machine (VM) with the same architecture as the target compute cluster: 

    ```console
    docker build -f docker/Dockerfile.train.multistage -t cogact-train .
    ```

2. Push the image to the Azure Container Registry (ACR):

    Set your Azure details

    ```console
    RESOURCE_GROUP=<your resource group name>
    WORKSPACE_NAME=<your AML workspace name>
    ```

    Get the ACR details from the AML workspace
    ```console
    ACR_INFO=$(az ml workspace show --resource-group $RESOURCE_GROUP --name $WORKSPACE_NAME --query container_registry -o tsv) ACR_NAME=$(echo $ACR_INFO | cut -d'/' -f9 | cut -d'.' -f1)
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
    ```console
    cat > acr-environment.yml << EOF
    \$schema: https://azuremlschemas.azureedge.net/latest/environment.schema.json
    name: <your chosen environment name>
    version: 1
    image: $ACR_IMAGE
    description: CogACT environment using pre-built image from ACR
    EOF
    ```

4. Register the environment in AML:
    ```console
    az ml environment create --file acr-environment.yml --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME
    ```

## Getting access to gated repos
Running e.g. inference requires access to gated repos such as Llama 2. Follow these steps to set up the access:

1. Get a Hugging Face account (if you don't have one already)

2. Request access to Llama 2:

    2.1 Go to https://huggingface.co/meta-llama/Llama-2-7b-hf
    
    2.2 Click "Access repository"
    
    2.3 Fill out Meta's form with your details
    
    2.4 Wait for approval (usually quick)

3. Create an access token:

    3.1 Go to https://huggingface.co/settings/tokens
    
    3.2 Create a new token with read permissions
    
    3.3 Copy the token

## Optional (but recommended): downloading and cache-ing CogACT checkpoints
For faster execution, it is recommended to download and cache the CogACT checkpoints. Here is an example of how this can be achieved by using the Docker image built in step 1:
```console
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

For later use, upload these checkpoints to an Azure Data Lake Gen 2 container. This can be achieved by creating a container and mounting it in a local directory (e.g. /mnt/azure-storage/)

```console
sudo mkdir -p /mnt/azure-storage/

sudo chown $(whoami) /mnt/azure-storage/

blobfuse2 mount /mnt/azure-storage/ --config-file=blobfuse_config.yaml --allow-other
````

and copying the files to the mounted volume. Note this uses the [blobfuse_config.yaml](blofuse_config.yaml) config file we have provided. Note: the first time this command is run, one may
additionally need to modify the /etc/fuse.conf file as follows: run `sudo nano /etc/fuse.conf`, 
uncomment this line: `#user_allow_other`, save and exit (Ctrl+O, Enter, then Ctrl+X in nano).
 
Once the data has been uploaded to the container, register the latter as a Datastore in AML.

## Running a minimal inference example in AML
A minimal inference example is provided in the "Getting Started" section of the main [README](../../README.md). In [inference_aml.py](inference_aml.py) we have slightly adapted the code to enable logging and running in AML. 

Notes:

1. The minimal inference example requires an image as input, preferrably consistent with the "move sponge near apple" task. We have provided one such image in [test_image.png](test_image.png)
2. To run the minimal inference example locally (instead of in an AML cluster), one may simply point the `cache_dir` variable in the inference script to the local folder containing the cached checkpoints (if available), and run it as follows `docker run --gpus all -e HF_TOKEN=<your_hf_token_here> -v /CogACT:/app -v <checkpoint_directory>:<checkpoint_directory> -w /app cogact-train:latest python inference.py
`. Warning: this may take a very long time (~1 hour) the first time the command is run.

The [inference-job-config.yml](inference-job-config.yml) file contains the configuration for running the inference job in an AML cluster (fill in the placeholders according to your setup). One can then submit the job as follows:

```console
export HF_TOKEN=<your HF token>
az ml job create --file inference-job-config.yml --name <job name> --resource-group <resource group name> --workspace-name <workspace name> --set environment_variables.HF_TOKEN="$HF_TOKEN"
```