# CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation
### 🚩[Project Page](https://cogact.github.io/) | 📑[Paper](https://arxiv.org/abs/2411.19650) | 🤗[Models](https://huggingface.co/CogACT)


This is the code for CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation.
## News

- 🔥 [2024-12-23] Update new setions: [Deployment in The Real World](#deployment-in-the-real-world) and [Inference Speed](#inference-speed). It demonstrates the advantage of our approach in inference speed.
- 🔥 [2024-12-01] Initial release.

## Contents
 * [**Installation**](#installation)
 * [**Getting Started**](#getting-started)
 * [**Fully Fine-Tuning**](#fully-fine-tuning)
 * [**Training CogACT from Scratch**](#training-cogact-from-scratch)
 * [**Evaluation in SIMPLER**](#evaluation-in-simpler)
 * [**Deployment in The Real World**](#deployment-in-the-real-world)
 * [**Inference Speed**](#inference-speed)
## Installation
The code is built using Python 3.10, and can be run under any environment with Python 3.8 and above. We require PyTorch >= 2.2.0 and CUDA >= 12.0 (It may run with lower versions, but we have not tested it).

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and setting up an environment:
```bash
conda create --name cogact python=3.10
```
Next, clone our repo and install the required packages:
```bash
git clone https://github.com/microsoft/CogACT
cd CogACT
pip install -e .
```
If you need to use the traning code, please also install the [Flash Attention](https://github.com/Dao-AILab/flash-attention). You can simply run (The first run might take a bit longer):
```bash
pip install -e .[train]
```
or install it manually:
```bash
# [Optional]
# Training additionally requires Flash-Attention 2 (https://github.com/Dao-AILab/flash-attention)
pip install packaging ninja

# Verify Ninja --> should return exit code "0"
ninja --version; echo $?

# Install Flash Attention 2
# =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install "flash-attn==2.5.5" --no-build-isolation
```
### How to run the same on Intel's XPU
Install the XPU build of PyTorch instead of a CUDA wheel, and skip the Flash Attention step (inference does not use it):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
pip install -e . --no-deps    # then install the other packages listed in pyproject.toml (transformers==4.40.1, timm==0.9.10, ...)
```
Run the minimal inference example from "Getting Started" with `model.to('xpu:0')` instead of `model.to('cuda:0')`. To pick the card, use `ZE_AFFINITY_MASK=0` in place of `CUDA_VISIBLE_DEVICES=0`. `vla.device.resolve_device()` returns the first XPU, else the first CUDA device, else the CPU:
```python
from vla.device import resolve_device
model.vlm = model.vlm.to(torch.bfloat16)
model.to(resolve_device()).eval()
```
The mixed precision context in `predict_action` follows the device the model is on. The SIMPLER evaluation needs its CUDA simulator stack and is not covered.

## Getting Started
We release three CogACT models with different model sizes, including [Small](https://huggingface.co/CogACT/CogACT-Small), [Base](https://huggingface.co/CogACT/CogACT-Base) and [Large](https://huggingface.co/CogACT/CogACT-Large). Checkpoints, configs, and model cards are availabel on [Hugging Face page](https://huggingface.co/CogACT). Refer to the code below for the minimal inference:
```python
from PIL import Image
from vla import load_vla
import torch

model = load_vla(
        'CogACT/CogACT-Base',                   # choose from [CogACT-Small, CogACT-Base, CogACT-Large] or the local path
        load_for_training=False, 
        action_model_type='DiT-B',              # choose from ['DiT-S', 'DiT-B', 'DiT-L'] to match the model weight
        future_action_window_size=15,
    )                                 
# about 30G Memory in fp32; 

# (Optional) use "model.vlm = model.vlm.to(torch.bfloat16)" to load vlm in bf16

model.to('cuda:0').eval()

image: Image.Image = <input_your_image>     
prompt = "move sponge near apple"               # input your prompt

# Predict Action (7-DoF; un-normalize for RT-1 google robot data, i.e., fractal20220817_data)
actions, _ = model.predict_action(
            image,
            prompt,
            unnorm_key='fractal20220817_data',  # input your unnorm_key of the dataset
            cfg_scale = 1.5,                    # cfg from 1.5 to 7 also performs well
            use_ddim = True,                    # use DDIM sampling
            num_ddim_steps = 10,                # number of steps for DDIM sampling
        )

    # results in 7-DoF actions of 16 steps with shape [16, 7]
```

Alternatively, you can use batch inference function ``predict_action_batch`` from [vla/cogactvla.py](./vla/cogactvla.py) to accelerate inference in the simulator. For our ``Adaptive Action Ensemble`` strategy, please refer to [adaptive_ensemble.py](./evaluation/adaptive_ensemble.py).

## Fully Fine-Tuning
To fully fine-tune the pretrained models, we use PyTorch Fully Sharded Data Parallel ([FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)). The training script used is from [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms).
We recommend using fully finetune on your dataset instead of LoRA, because the model with fully finetuning performs better in a shorter training time. Empirically. Fully finetuning the pretrained model for around 30 epochs already yields good results. Pretrained models can be download from our [Hugging Face page](https://huggingface.co/CogACT/CogACT-Base) or by passing the model_id to the training scripts for automatic download.

Download from our [Hugging Face page](https://huggingface.co/CogACT/CogACT-Base), using CogACT-Base for an example. (Optional)
```bash
# Change directory to your base model PATH
cd <your_base_model_path>

# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

# Download checkpoint (30 GB)
git clone https://huggingface.co/CogACT/CogACT-Base
```
You can also pass the model_id (e.g., `CogACT/CogACT-Base`) to the training scripts for automatic download. (Seeing below)

Next, create a [Hugging Face user access token](https://huggingface.co/docs/hub/en/security-tokens) and export the token value.

```bash
# export the HuggingFace user access token token
export HF_TOKEN = hf_..
```

Then launch the training script. We use one node with 8 A100 GPUs as an example.
```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/train.py \
  --pretrained_checkpoint <model_id/local_path_to_model,e.g,"CogACT/CogACT-Base"> \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix <data_mix_option,e.g,"bridge"> \
  --vla.expected_world_size 8 \
  --vla.global_batch_size 256 \
  --vla.per_device_batch_size 32 \
  --vla.learning_rate 2e-5 \
  --data_root_dir <path_to_dataset_dir> \
  --run_root_dir <path_to_log/checkpoint_dir> \                 
  --run_id <optional_run_id_for_wandb> \
  --image_aug <True_or_False> \
  --wandb_project <your_wandb_project> \
  --wandb_entity <your_wandb_entity> \
  --save_interval <num_of_steps_to_save_checkpoint> \
  --repeated_diffusion_steps 8 \
  --future_action_window_size 15 \
  --action_model_type DiT-B \
  --is_resume False
```
More customized training settings and changes can be made in [`conf/vla.py`](conf/vla.py) by modifying and registering a new VLA type. If you want to resume from a checkpoint instead of starting training from scratch, please set `is_resume=True`. Note that you also need to set `--resume_step` and `--resume_epoch` to match the checkpoint, and the optimizer in the checkpoint also needs to be loaded.

To finetune on datasets belong to [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/), you can download them from [OXE](https://robotics-transformer-x.github.io/) and change the ``vla.data_mix`` to the corresponding name. To finetune on your own customized data, please follow the instruction [(rlds_dataset_builder)](https://github.com/kpertsch/rlds_dataset_builder) for converting your data to RLDS format. The actions should be the deltas of end effector ``EEF Delta XYZ (3) + Roll-Pitch-Yaw (3) + Gripper Open/Close (1)``. Once your customized data is ready, place the customized data directly under the ``<data_root_dir>/custom_finetuning/1.0.0`` directory. Then set ``vla.data_mix="custom_finetuning"``.

## Training CogACT from Scratch
You can start the trainging from the weights of [OpenVLA](https://github.com/openvla/openvla) for greater efficiency. Please follow the instruction of [OpenVLA](https://github.com/openvla/openvla) to download their weights:
```bash
# From OpenVLA repo
# Change directory to your base model checkpoints folder
cd <PATH TO BASE MODEL CHECKPOINTS DIR>

# Download checkpoint (30 GB) -- may take a few minutes
git clone git@hf.co:openvla/openvla-7b-prismatic

# If the command above did not download the full checkpoint,
# manually fetch it via git Large File Storage (LFS)
# Note: You may have to configure an SSH key for this to work
cd openvla-7b-prismatic
git lfs fetch --all
```
The data of [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/) can be download following [OXE](https://robotics-transformer-x.github.io/) and [OpenVLA](https://github.com/openvla/openvla). Then launch the training script. We use one node with 8 A100 GPUs as an example.

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/train.py \
  --pretrained_checkpoint openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix oxe_magic_soup_plus_minus \
  --vla.expected_world_size 8 \
  --vla.global_batch_size 256 \
  --vla.per_device_batch_size 32 \
  --vla.learning_rate 2e-5 \
  --data_root_dir <path_to_dataset_dir> \
  --run_root_dir <path_to_log/checkpoint_dir> \                 
  --run_id <optional_run_id_for_wandb> \
  --image_aug <True_or_False> \
  --wandb_project <your_wandb_project> \
  --wandb_entity <your_wandb_entity> \
  --save_interval <num_of_steps_to_save_checkpoint> \
  --repeated_diffusion_steps 8 \
  --future_action_window_size 15 \
  --action_model_type DiT-B \
  --is_resume False
```
You can also start training from PrismaticVLM and simply ignore the ``--pretrained_checkpoint``. However, it will take longer to converge.

## Evaluation in SIMPLER
In this section, we provide a minimal evaluation for our models in [SIMPLER](https://simpler-env.github.io/). First, please follow the instruction of [SimplerEnv](https://github.com/simpler-env/SimplerEnv) to install the simulation environment. Next, add our [./sim_cogact](./sim_cogact) to [SimplerEnv/simpler_env/policies](https://github.com/simpler-env/SimplerEnv/tree/main/simpler_env/policies).
```bash
cp ./sim_cogact <your_path_to_simpler>/simpler_env/policies -r
```
Then add a new policy model in [SimplerEnv/simpler_env/main_inference.py](https://github.com/simpler-env/SimplerEnv/blob/main/simpler_env/main_inference.py) as below:
```python
elif args.policy_model == "cogact":
    from simpler_env.policies.sim_cogact import CogACTInference
    assert args.ckpt_path is not None
    model = CogACTInference(
        saved_model_path=args.ckpt_path,  # e.g., CogACT/CogACT-Base
        policy_setup=args.policy_setup,
        action_scale=args.action_scale,
        action_model_type='DiT-B',
        cfg_scale=1.5                     # cfg from 1.5 to 7 also performs well
    )
```
After that, you can modify and launch the scripts in [`sim_cogact/scripts`](sim_cogact/scripts) like:
```bash
cd <your_path_to_simpler>
bash simpler_env/policies/sim_cogact/scripts/cogact_put_in_drawer_visual_matching.sh
```

## Deployment in The Real World

For your own environment or robot, please first collect the corresponding real-world operation data (e.g., using teleoperation). Then, use the data to fine-tune the pretrained model we provided, following the instructions in the section [Fully Fine-Tuning](#fully-fine-tuning).

Next, you can set up the server and client as instructed in the [`scripts/deploy.py`](scripts/deploy.py) and deploy it on the real robot according to the hardware you are using. Please run the following line to serve the fine-tuned model: (Using 'fractal20220817_data' as an example, please replace "unnorm_key" with the value from your fine-tuned dataset in actual use.)
```bash
python scripts/deploy.py --saved_model_path <your_model_path> --unnorm_key fractal20220817_data --action_ensemble --use_bf16 --action_ensemble_horizon 2 --adaptive_ensemble_alpha 0.1 --cfg_scale 1.5 --port 5500
```
You can also use other inference strategies modifying the parameters in [`scripts/deploy.py`](scripts/deploy.py) such as the action chunking (output multiple acitons without ensembling).

As for the client, only a Python environment and the ``requests`` library (``pip install requests``) are required; 
no other dependencies need to be installed.

A simple client (standalone) usage (assuming a server running on 127.0.0.1:5500):

```python
import requests
import json

# Define the API endpoint
url = 'http://127.0.0.1:5500/api/inference'

# Define the parameters you want to send
data = {
    'task_description': "Pick up the red can.",
}
image = "image/google_robot.png"

json.dump(data, open("data.json", "w"))

with open ("data.json", "r") as query_file:
    with open(image, "rb") as image_file:
        file = [
            ('images', (image, image_file, 'image/png')),
            ('json', ("data.json", query_file, 'application/json'))
        ]
        
        response = requests.post(url, files=file)
        # print(response)
if response.status_code == 200:
    pass
else:
    print("Failed to get a response from the API")
    print(response.text)
```

## Inference Speed
We serve the ``CogACT-Base`` on a single A6000 GPU in bfloat16 format and invoke it 100 times repeatedly (see [Deployment in The Real World](#deployment-in-the-real-world) for deployment details). It takes about 181ms for each inference in average. Therefore, the action generation frequency is approximately 5.5Hz on a single A6000 GPU using our ``Adaptive Action Ensemble`` strategy. If the action chunking strategy is used and *k* actions (*k* is at most 16) are output each time, the frequency will become *k* times the original. However, the accuracy of the actions will gradually decrease as *k* increases due to the longer open-loop prediction.

We also deploy OpenVLA in bfloat16 format on the same device for comparison, test the average time for model inference, and list the number of actions the model can generate in a single inference in the following table.

|                             | CogACT-Base | OpenVLA |  
|-----------------------------|-------------|---------|  
| Inference time (ms)              | 181       | 307   |  
| Number of generated actions | 16          | 1       |  

As shown in the table, our method has a faster inference speed because we use a single cognition token to generate an entire action sequence. In contrast, an OpenVLA-style model needs to generate 7 tokens to represent a 7-dimensional action. Even when considering the time taken for our DiT inference, our model still achieves a significant speedup compared to OpenVLA. Additionally, our approach can utilize action chunking to generate multiple actions in a single inference.

## Citing
If you find our work useful, please consider citing [our paper](https://cogact.github.io/CogACT_paper.pdf):

```bibtex
@article{li2024cogact,
  title={CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation},
  author={Li, Qixiu and Liang, Yaobo and Wang, Zeyu and Luo, Lin and Chen, Xi and Liao, Mozheng and Wei, Fangyun and Deng, Yu and Xu, Sicheng and Zhang, Yizhong and others},
  journal={arXiv preprint arXiv:2411.19650},
  year={2024}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## License

All the code, model weights, and data are licensed under [MIT license](./LICENSE).
