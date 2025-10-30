from PIL import Image
from vla import load_vla
import torch
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# AzureML configuration: Use mounted datastore if available, otherwise fallback to HuggingFace
COGACT_CHECKPOINTS = os.environ.get('COGACT_CHECKPOINTS')
if COGACT_CHECKPOINTS and os.path.exists(COGACT_CHECKPOINTS):
    cache_dir = COGACT_CHECKPOINTS
    logging.info(f"Using mounted datastore cache: {cache_dir}")
else:
    cache_dir = '/tmp/hf_cache'
    os.makedirs(cache_dir, exist_ok=True)
    logging.info(f"Using HuggingFace cache: {cache_dir}")

# HuggingFace token for gated models
hf_token = os.environ.get('HF_TOKEN')
if not hf_token:
    logging.error("HF_TOKEN environment variable not set")
    raise ValueError("HF_TOKEN required for accessing gated models")

# Load model - will use local cache if available
model = load_vla(
    'CogACT/CogACT-Base',                   # choose from [CogACT-Small, CogACT-Base, CogACT-Large]
    hf_token=hf_token,                      # HuggingFace token for authentication
    cache_dir=cache_dir,                    # Use mounted storage or local cache
    load_for_training=False, 
    action_model_type='DiT-B',              # choose from ['DiT-S', 'DiT-B', 'DiT-L'] to match the model weight
    future_action_window_size=15,
)                                 
# about 30G Memory in fp32; 

# (Optional) use "model.vlm = model.vlm.to(torch.bfloat16)" to load vlm in bf16

model.to('cuda:0').eval()

image: Image.Image = Image.open('test_image.png').convert('RGB')  # input your image path     
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

# Log results
logging.info(f"Actions shape: {actions.shape}")  # should log torch.Size([16, 7])
logging.info(f"Actions: {actions}")
# results in 7-DoF actions of 16 steps with shape [16, 7]
