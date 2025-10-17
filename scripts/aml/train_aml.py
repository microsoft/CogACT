#!/usr/bin/env python3
"""
Azure ML wrapper for CogACT fine-tuning.

This script wraps the CogACT training script by:
1. Parsing command-line arguments to extract --pretrained_checkpoint
2. Resolving checkpoint using fallback strategy (HF cache -> HuggingFace download)
3. Launching torchrun with the resolved checkpoint and user arguments

Environment Variables:
- HF_HOME: HuggingFace cache directory for checkpoint resolution
- GPU_COUNT: Number of GPUs to use (default: 4)
"""

import argparse
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


def resolve_pretrained_checkpoint(hf_cache_dir, pretrained_checkpoint):
    """
    Resolve pretrained checkpoint using fallback strategy.

    Args:
        hf_cache_dir (str): Path to HuggingFace cache directory
        pretrained_checkpoint (str): User-specified checkpoint argument

    Returns:
        str: Resolved checkpoint path or HuggingFace model ID
    """
    # Use user-specified HuggingFace model ID if provided
    if pretrained_checkpoint and pretrained_checkpoint.startswith("CogACT/"):
        logger.info("Using user-specified HuggingFace model ID: %s", pretrained_checkpoint)
        return pretrained_checkpoint

    # Check for cached checkpoint in HF cache directory
    if hf_cache_dir:
        logger.info("Local checkpoint not found, checking HuggingFace cache...")
        cached_checkpoint = find_checkpoint_in_directory(hf_cache_dir, "cached COGACT")
        if cached_checkpoint:
            logger.info("SUCCESS: Found cached COGACT checkpoint in HF cache: %s", cached_checkpoint)
            logger.info("Using cached checkpoint file instead of downloading from HuggingFace")
            return cached_checkpoint

    # Fall back to downloading from HuggingFace
    logger.info("No model checkpoints directory available, downloading from HuggingFace ID: %s", DEFAULT_HF_MODEL_ID)
    logger.info("This will download the model from HuggingFace (may take several minutes)")
    return DEFAULT_HF_MODEL_ID


def process_command_line_args():
    """
    Extract --pretrained_checkpoint argument and return remaining args.

    Returns:
        tuple: (filtered_args, user_checkpoint_preference)
    """
    # Parse arguments to extract --pretrained_checkpoint
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--pretrained_checkpoint", type=str, default=None, help="Pretrained checkpoint path or HuggingFace model ID"
    )
    known_args, unknown_args = parser.parse_known_args()

    return unknown_args, known_args.pretrained_checkpoint


def main():
    """
    Main entry point that resolves checkpoint and runs training command.

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    logger.info("Starting CogACT fine-tuning in AzureML")
    logger.info("HF_HOME: %s", os.environ.get("HF_HOME"))

    # Extract --pretrained_checkpoint argument from command line
    args, pretrained_checkpoint = process_command_line_args()

    # Build torchrun command
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc-per-node",
        str(os.environ.get("GPU_COUNT", DEFAULT_GPU_COUNT)),
        "scripts/train.py",
    ]

    hf_home = os.environ.get("HF_HOME")
    resolved_checkpoint = resolve_pretrained_checkpoint(hf_home, pretrained_checkpoint)

    logger.info("Resolved pretrained checkpoint: %s", resolved_checkpoint)

    # Add resolved checkpoint back to arguments
    args.extend(["--pretrained_checkpoint", resolved_checkpoint])

    # Build final command with all arguments
    cmd.extend(args)

    # Run training command
    logger.info("Running training command: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, check=False, text=True)
        return result.returncode
    except (subprocess.SubprocessError, OSError) as e:
        logger.error("Training failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
