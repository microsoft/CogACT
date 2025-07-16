#!/usr/bin/env python3
"""
Azure ML wrapper for CogACT fine-tuning.

This script sets up the job for fine-tuning the CogACT model in Azure ML by:
1. Configuring HuggingFace authentication and caching
2. Setting up input/output mount resolution
3. Implementing a local checkpoint search with download as fallback strategy
4. Launching the training script with appropriate parameters

Checkpoint Resolution Strategy:
- Step 1: Check inputs.cogact_checkpoints_dir for local checkpoint files
- Step 2: Check outputs.hf_cache_dir for cached checkpoint files
- Step 3: Download from HuggingFace if neither found

Environment Variables:
- HF_TOKEN: Required for HuggingFace authentication
- GPU_COUNT: Number of GPUs to use (default: 4)
- AZURE_ML_INPUT_*: AzureML input mount environment variables
- AZURE_ML_OUTPUT_*: AzureML output mount environment variables
"""

import logging
import os
import subprocess
import sys

# Constants
DEFAULT_HF_MODEL_ID = "CogACT/CogACT-Base"
CHECKPOINT_FILENAMES = ["CogACT-Base.pt", "pytorch_model.bin", "model.safetensors"]
DEFAULT_GPU_COUNT = 4

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_hf_token():
    """Set up HuggingFace authentication."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable required for accessing gated models")

    # Write token to HF token file
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    os.makedirs(hf_home, exist_ok=True)

    token_path = os.path.join(hf_home, "token")
    with open(token_path, "w", encoding="utf-8") as f:
        f.write(hf_token)

    logger.info("HuggingFace authentication configured")


def resolve_input_mount(env_var_names, description):
    """Resolve AzureML input mount from environment variables."""
    for env_var in env_var_names:
        path = os.environ.get(env_var)
        if path and not path.startswith("${{") and os.path.exists(path):
            logger.info("Resolved %s from %s: %s", description, env_var, path)
            return path

    logger.warning("Could not resolve %s from variables: %s", description, env_var_names)
    return None


def setup_model_cache(hf_cache_dir, output_dir):
    """Set up HuggingFace model cache using dedicated cache mount or output directory."""
    if hf_cache_dir:
        # Use dedicated HF cache mount (environment variables already configured by AzureML)
        logger.info("Using dedicated HuggingFace cache mount: %s", hf_cache_dir)
        return hf_cache_dir

    elif output_dir:
        # Fallback to output directory
        cache_dir = os.path.join(output_dir, "hf_cache")
        os.makedirs(cache_dir, exist_ok=True)
        logger.info("Using HuggingFace cache in output directory: %s", cache_dir)

        # Set HF environment variables for fallback cache
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache_dir, "hub")
        return cache_dir

    else:
        raise ValueError("No cache directory available")


def find_checkpoint_in_directory(base_dir, description="checkpoint"):
    """
    Find CogACT checkpoint files in a directory structure.

    This function searches for checkpoint files in multiple locations:
    1. Direct file in the base directory (CogACT-Base.pt)
    2. HuggingFace cache structure: models--CogACT--CogACT-Base/snapshots/*/
    3. Checkpoints subdirectory within snapshots

    Args:
        base_dir (str): Base directory to search in
        description (str): Description for logging purposes

    Returns:
        str or None: Path to the first checkpoint file found, or None if not found
    """
    if not base_dir or not os.path.exists(base_dir):
        return None

    # Common checkpoint file patterns (in order of preference)
    checkpoint_files = CHECKPOINT_FILENAMES

    possible_checkpoint_paths = []

    # Check for direct file in root
    for filename in checkpoint_files:
        direct_path = os.path.join(base_dir, filename)
        possible_checkpoint_paths.append(direct_path)

    # Check HuggingFace cache structure: models--CogACT--CogACT-Base/snapshots/*/
    cogact_model_dir = os.path.join(base_dir, "models--CogACT--CogACT-Base")
    if os.path.exists(cogact_model_dir):
        snapshots_dir = os.path.join(cogact_model_dir, "snapshots")
        if os.path.exists(snapshots_dir):
            try:
                for snapshot_id in os.listdir(snapshots_dir):
                    snapshot_path = os.path.join(snapshots_dir, snapshot_id)
                    if os.path.isdir(snapshot_path):
                        # Check for checkpoint files in snapshot root
                        for filename in checkpoint_files:
                            checkpoint_path = os.path.join(snapshot_path, filename)
                            possible_checkpoint_paths.append(checkpoint_path)

                        # Also check in checkpoints/ subdirectory within snapshot
                        checkpoints_subdir = os.path.join(snapshot_path, "checkpoints")
                        if os.path.exists(checkpoints_subdir):
                            for filename in checkpoint_files:
                                checkpoint_path = os.path.join(checkpoints_subdir, filename)
                                possible_checkpoint_paths.append(checkpoint_path)
            except OSError:
                pass

    # Log what we're checking
    logger.info("Checking for %s checkpoint in: %s", description, base_dir)
    for path in possible_checkpoint_paths:
        exists_status = "EXISTS" if os.path.exists(path) else "NOT FOUND"
        logger.info("  Checking: %s -> %s", path, exists_status)

    # Return the first existing checkpoint
    for path in possible_checkpoint_paths:
        if os.path.exists(path):
            return path

    return None


def resolve_pretrained_checkpoint(model_checkpoints_dir, hf_cache_dir, pretrained_checkpoint):
    """
    Resolve pretrained checkpoint using a fallback strategy.

    Strategy:
    1. If user specifies a HuggingFace model ID (CogACT/*), use it directly
    2. Check for local checkpoint in cogact_checkpoints_dir (inputs mount)
    3. Check for cached checkpoint in hf_cache_dir (HF cache, output mount)
    4. Fall back to downloading from HuggingFace using model ID

    Args:
        model_checkpoints_dir (str): Path to cogact checkpoints directory
        hf_cache_dir (str): Path to HuggingFace cache directory
        pretrained_checkpoint (str): User-specified checkpoint argument

    Returns:
        str: Resolved checkpoint path or HuggingFace model ID
    """
    # Step 1: Honor user-specified HuggingFace model ID
    if pretrained_checkpoint and pretrained_checkpoint.startswith("CogACT/"):
        logger.info("Using user-specified HuggingFace model ID: %s", pretrained_checkpoint)
        return pretrained_checkpoint

    # Step 2: Check for local checkpoint in cogact_checkpoints_dir
    if model_checkpoints_dir:
        local_checkpoint = find_checkpoint_in_directory(model_checkpoints_dir, "local COGACT")
        if local_checkpoint:
            logger.info("Found local COGACT checkpoint: %s", local_checkpoint)
            logger.info("Using local checkpoint file instead of downloading from HuggingFace")
            return local_checkpoint

    # Step 3: Check for cached checkpoint in hf_cache_dir
    if hf_cache_dir:
        logger.info("Local checkpoint not found, checking HuggingFace cache...")
        cached_checkpoint = find_checkpoint_in_directory(hf_cache_dir, "cached COGACT")
        if cached_checkpoint:
            logger.info("SUCCESS: Found cached COGACT checkpoint in HF cache: %s", cached_checkpoint)
            logger.info("Using cached checkpoint file instead of downloading from HuggingFace")
            return cached_checkpoint

    # Step 4: Fall back to downloading from HuggingFace
    if model_checkpoints_dir:
        logger.info(
            "No local or cached COGACT checkpoint found, downloading from HuggingFace ID: %s",
            DEFAULT_HF_MODEL_ID,
        )
    else:
        logger.info("No model checkpoints directory available, downloading from HuggingFace ID: %s", DEFAULT_HF_MODEL_ID)

    logger.info("This will download the model from HuggingFace (may take several minutes)")
    return DEFAULT_HF_MODEL_ID


def process_command_line_args():
    """
    Process and transform command-line arguments for the training script.

    This function implements argument interception and transformation to:
    1. Filter out arguments that need special handling (--pretrained_checkpoint, --hf_token)
    2. Replace HF token with controlled environment variable reference
    3. Extract user's checkpoint preference for fallback strategy
    4. Preserve all other user arguments unchanged

    Returns:
        tuple: (filtered_args, user_checkpoint_preference)
            - filtered_args: List of arguments with special ones removed
            - user_checkpoint_preference: User's original --pretrained_checkpoint value or None
    """
    # Filter out arguments that need special handling
    filtered_args = []
    skip_next = False

    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue

        if arg in ["--pretrained_checkpoint", "--hf_token"]:
            skip_next = True  # Skip the next argument (the value)
            continue

        filtered_args.append(arg)

    # Extract user's checkpoint preference from original command line
    user_checkpoint_preference = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--pretrained_checkpoint" and i + 1 < len(sys.argv[1:]):
            user_checkpoint_preference = sys.argv[1:][i + 1]
            break

    return filtered_args, user_checkpoint_preference


def main():
    """
    Main entry point for CogACT fine-tuning in AzureML.

    This function orchestrates the fine-tuning job setup:
    1. Sets up HuggingFace authentication
    2. Resolves Azure ML input/output mounts
    3. Configures model caching
    4. Resolves pretrained checkpoint using fallback strategy
    5. Constructs and executes the training command

    The checkpoint resolution follows a three-step fallback strategy:
    - First: Check cogact_checkpoints_dir for local checkpoints
    - Second: Check hf_cache_dir for cached checkpoints
    - Third: Download from HuggingFace if neither found

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    logger.info("Starting CogACT fine-tuning in AzureML")

    # Set up HuggingFace authentication
    setup_hf_token()

    # Resolve input/output mounts
    model_checkpoints_dir = resolve_input_mount(
        ["AZURE_ML_INPUT_COGACT_CHECKPOINTS_DIR", "COGACT_CHECKPOINTS_DIR", "cogact_checkpoints_dir"],
        "model checkpoints",
    )
    dataset_dir = resolve_input_mount(["AZURE_ML_INPUT_DATASET_DIR", "DATASET_DIR", "dataset_dir"], "dataset")
    output_dir = resolve_input_mount(
        ["AZURE_ML_OUTPUT_CHECKPOINTS_DIR", "CHECKPOINTS_DIR", "checkpoints_dir"], "output directory"
    )
    hf_cache_dir = resolve_input_mount(
        ["AZURE_ML_OUTPUT_HF_CACHE_DIR", "HF_CACHE_DIR", "hf_cache_dir"], "HuggingFace cache directory"
    )

    # Set up model cache
    setup_model_cache(hf_cache_dir, output_dir)

    # Debug: Show what's available in both potential checkpoint locations
    if model_checkpoints_dir:
        logger.info("Contents of cogact_checkpoints_dir (%s):", model_checkpoints_dir)
        try:
            for item in os.listdir(model_checkpoints_dir):
                item_path = os.path.join(model_checkpoints_dir, item)
                if os.path.isdir(item_path):
                    logger.info("  [DIR]  %s", item)
                else:
                    logger.info("  [FILE] %s", item)
        except OSError as e:
            logger.warning("Could not list contents of %s: %s", model_checkpoints_dir, e)

    if hf_cache_dir:
        hf_hub_dir = os.path.join(hf_cache_dir, "hub")
        if os.path.exists(hf_hub_dir):
            logger.info("Contents of hf_cache_dir/hub (%s):", hf_hub_dir)
            try:
                for item in os.listdir(hf_hub_dir):
                    if "CogACT" in item:
                        item_path = os.path.join(hf_hub_dir, item)
                        logger.info("  [DIR]  %s", item)
            except OSError as e:
                logger.warning("Could not list contents of %s: %s", hf_hub_dir, e)

    # Determine distributed training parameters
    gpu_count = int(os.environ.get("GPU_COUNT", DEFAULT_GPU_COUNT))
    logger.info("Using single-node distributed training with %d GPUs", gpu_count)

    # Construct the training command
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc-per-node",
        str(gpu_count),
        "scripts/train.py",
    ]

    # ============================================================================
    # ARGUMENT PROCESSING AND TRANSFORMATION
    # ============================================================================
    # This section intercepts and transforms command-line arguments before passing
    # them to the actual training script. We need to handle certain arguments
    # specially to implement our checkpoint fallback strategy and security controls.
    #
    # EXAMPLE TRANSFORMATION:
    # User runs: python train_aml.py --pretrained_checkpoint CogACT/CogACT-Base --hf_token abc123 --batch_size 32
    #
    # Step 1: Extract user preferences and filter args
    #   - pretrained_checkpoint = "CogACT/CogACT-Base"
    #   - filtered_args = ["--batch_size", "32"]
    #
    # Step 2: Add controlled token
    #   - args = ["--batch_size", "32", "--hf_token", "HF_TOKEN"]
    #
    # Step 3: Resolve checkpoint via fallback strategy
    #   - resolved_checkpoint = "/path/to/local/checkpoint.pt" (or HF model ID)
    #   - final_args = ["--batch_size", "32", "--hf_token", "HF_TOKEN",
    #                   "--pretrained_checkpoint", "/path/to/local/checkpoint.pt"]

    # Step 1: Process and transform command-line arguments
    args, pretrained_checkpoint = process_command_line_args()

    # Step 2: Add controlled HuggingFace token reference
    # Instead of passing through the user's token directly, we reference the
    # environment variable that was set up earlier by setup_hf_token()
    args.extend(["--hf_token", "HF_TOKEN"])

    # Step 3: Resolve checkpoint using our intelligent fallback strategy
    # This implements the three-step process:
    # 1. Check local cogact_checkpoints_dir
    # 2. Check HF cache directory
    # 3. Download from HuggingFace if neither found
    resolved_checkpoint = resolve_pretrained_checkpoint(model_checkpoints_dir, hf_cache_dir, pretrained_checkpoint)
    args.extend(["--pretrained_checkpoint", resolved_checkpoint])

    # At this point, 'args' contains all the arguments that will be passed to the
    # actual training script, with special handling completed for checkpoint resolution
    # and token security.

    # Add dataset and output directories
    if dataset_dir:
        args.extend(["--data_root_dir", dataset_dir])
    else:
        logger.error("Dataset directory not found")

    if output_dir:
        args.extend(["--run_root_dir", output_dir])

    cmd.extend(args)

    # Execute training
    logger.info("Running training command: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, check=False, text=True)
        return result.returncode
    except (subprocess.SubprocessError, OSError) as e:
        logger.error("Training failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
