"""
cogactvla.py

"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaTokenizerFast

from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.vlms.base_vlm import VLM
from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector

from action_model.action_model import ActionModel
from action_model.models import DiT

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class CogACT(nn.Module):
    def __init__(
        self,
        vlm: PrismaticVLM,
        action_model_type: str = 'DiT-B',
        token_size: int = 4096,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        use_ema: bool = False,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.action_model = ActionModel(model_type = action_model_type, 
                                            token_size = token_size, 
                                            in_channels = action_dim, 
                                            future_action_window_size = future_action_window_size, 
                                            past_action_window_size = past_action_window_size)
        self.vlm = vlm
        self.future_action_window_size = future_action_window_size
        self.past_action_window_size = past_action_window_size
        self.use_ema = use_ema
        if self.use_ema:
            self.ema_diffusion = deepcopy(self.action_model)
            self.ema_diffusion.requires_grad_(False)
            self.all_module_keys = ['action_model', 'ema_diffusion']
        else:
            self.all_module_keys = ['action_model']
        for module_keys in self.vlm.all_module_keys:
            self.all_module_keys.append("vlm." + module_keys)

        # Diffusion head is always trainable
        self._trainable_module_keys = ['action_model']
        self.norm_stats = norm_stats

    @property
    def trainable_module_keys(self) -> List[str]:
        keys = []
        for module_keys in self.vlm.trainable_module_keys:
            keys.append("vlm." + module_keys)
        keys += self._trainable_module_keys
        return keys
    
    @property
    def llm_backbone(self) -> LLMBackbone:
        return self.vlm.llm_backbone
    
    @property
    def vision_backbone(self) -> VisionBackbone:
        return self.vlm.vision_backbone
    
    def freeze_backbones(self, stage):
        self.vlm.freeze_backbones(stage)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        repeated_diffusion_steps: int = 4,
        action_masks = None,
    ) -> Tuple:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""
        
        output: CausalLMOutputWithPast = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # extract the last hidden state and the learnable EOS token feature
        last_hidden = output.hidden_states[-1]

        # extract the visual token number
        if self.vlm.vision_backbone.featurizer is not None:
            num_patch = self.vlm.vision_backbone.featurizer.patch_embed.num_patches
        elif hasattr(self.vlm.vision_backbone, 'siglip_featurizer') and self.vlm.vision_backbone.siglip_featurizer is not None:
            num_patch = self.vlm.vision_backbone.siglip_featurizer.patch_embed.num_patches
        else:
            raise ValueError("No vision backbone found")
        
        last_hidden = last_hidden[:, num_patch :]

        # extract the cognition feature
        cumulative_sum = attention_mask.cumsum(dim=1)
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)  
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, last_hidden.size(-1))  
        cognition_features = last_hidden.gather(1, expanded_indices.unsqueeze(1))  # [B, 1, D]

        actions_history = actions[:,0:self.past_action_window_size,:]
        actions_future = actions[:, -(self.future_action_window_size+1):, :]
        
        # Repeat 'actions' 'repeated_diffusion_steps' times, resulting in [repeated_diffusion_steps*B, T, D]
        actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)
        actions_history_repeated = actions_history.repeat(repeated_diffusion_steps, 1, 1)
        cognition_features_repeated = cognition_features.repeat(repeated_diffusion_steps, 1, 1) # [repeated_diffusion_steps*B, 1, D]

        # Action model forward and compute loss
        loss = self.action_model.loss(actions_repeated, cognition_features_repeated)
        return loss, output

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vlm.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.vlm.llm_backbone.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector` and DiT
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector, MLPProjector, FusedMLPProjector, DiT},
        )

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )

    def load_ema_to_weights(self):
        """Load the EMA state dict to the weights."""
        if self.use_ema:
            self.action_model.load_state_dict(self.ema_diffusion.state_dict())
            del self.ema_diffusion

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        freeze_weights: bool = True,
        action_dim: int = 7,
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        action_model_type: str = 'DiT-B',
        use_ema: bool = False,
        norm_stats = None,
        **kwargs,
    ) -> CogACT:

        # Load VLM backbone, borrowed from PrismaticVLM
        vlm = PrismaticVLM(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            **kwargs,
        )

        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
        assert (
            "projector" in model_state_dict and "llm_backbone" in model_state_dict
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"

        vlm.projector.load_state_dict(model_state_dict["projector"])
        vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])
        if "vision_backbone" in model_state_dict.keys():
            vlm.vision_backbone.load_state_dict(model_state_dict["vision_backbone"])

        # Freeze Weights
        if freeze_weights:
            vlm.requires_grad_(False)
            vlm.eval()

        # Initialize CogACT
        cogact = CogACT(vlm,
                        token_size = vlm.llm_backbone.llm.lm_head.in_features,
                        action_dim = action_dim,
                        future_action_window_size = future_action_window_size,
                        past_action_window_size = past_action_window_size,
                        action_model_type = action_model_type,
                        use_ema = use_ema,
                        norm_stats = norm_stats,
                        )

        # Load ActionModel from Checkpoint
        if "action_model" in model_state_dict:
            cogact.action_model.load_state_dict(model_state_dict["action_model"])
            if "ema_diffusion" in model_state_dict and use_ema:
                cogact.ema_diffusion.load_state_dict(model_state_dict["ema_diffusion"])
            elif use_ema:
                cogact.ema_diffusion.load_state_dict(model_state_dict["action_model"])
        else:
            overwatch.warning("No ActionModel found in the pretrained checkpoint. Initializing a new one.")
        return cogact        

    def _autocast(self, dtype: torch.dtype):
        """Mixed precision context on the backend the model runs on (CPU stays without autocast, as before)."""
        device_type = self.vlm.device.type
        enabled = self.vlm.enable_mixed_precision_training and device_type != "cpu"
        return torch.autocast(device_type, dtype=dtype, enabled=enabled)

    @torch.inference_mode()
    def predict_action(
        self, image: Image, 
        instruction: str, 
        unnorm_key: Optional[str] = None, 
        cfg_scale: float = 1.5, 
        use_ddim: bool = False,
        num_ddim_steps: int = 5,
        **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action.

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.
        @param cfg_scale: Scaling factor for classifier-free guidance (CFG); if == 1.0, CFG is disabled.
        @param use_ddim: Use DDIM sampling instead of DDPM sampling.
        @param num_ddim_steps: Number of DDIM steps to use for sampling.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        image_transform, tokenizer = self.vlm.vision_backbone.image_transform, self.vlm.llm_backbone.tokenizer

        # Build VLA Prompt
        prompt_builder = self.vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()
        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vlm.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            # Note: We need to add this special empty token ('') after the colon (':') token in "ASSISTANT:"
            #       insert it to match the inputs seen at training time. The empty token is at index 29871.
            #       We also need to add the special cognition token at index 2 (i.e. the EOS token).
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871, 2]).long(), dim=0).to(self.vlm.device)), dim=1
            )
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # Preprocess Image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.vlm.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.vlm.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")
        
        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.vlm.llm_backbone.half_precision_dtype

        # Generate cognition feature through vlm
        with self._autocast(autocast_dtype):
            # fmt: off
            output = super(PrismaticVLM, self.vlm).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
                max_new_tokens=1,
                output_hidden_states=True, 
                return_dict_in_generate=True,
                **kwargs
            )
            # fmt: on

        # Extract cognition feature
        cognition_features = output.hidden_states[0][-1][:,-1,:]
        assert (cognition_features.shape[0], cognition_features.shape[1]) == (1,4096), "Batch size must be 1 for action prediction"
        using_cfg = cfg_scale > 1.0

        model_dtype = next(self.action_model.net.parameters()).dtype
        B = cognition_features.shape[0]

        cognition_features = cognition_features.unsqueeze(1).to(model_dtype)  # [B, 1, D]

        # Sample random noise
        noise = torch.randn(B, self.future_action_window_size+1, self.action_model.in_channels, device=cognition_features.device).to(model_dtype)  #[B, T, D]
    
        # Setup classifier-free guidance:
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition = self.action_model.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0)  #[1, D]
            uncondition = uncondition.expand(B, 1, -1) #[B, 1, D]
            z = torch.cat([cognition_features, uncondition], 0)
            cfg_scale = cfg_scale
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.action_model.net.forward_with_cfg
        else:
            model_kwargs = dict(z=cognition_features)
            sample_fn = self.action_model.net.forward

        # DDIM Sampling
        if use_ddim and num_ddim_steps is not None:
            if self.action_model.ddim_diffusion is None:
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(sample_fn, 
                                                                noise.shape, 
                                                                noise, 
                                                                clip_denoised=False,
                                                                model_kwargs=model_kwargs,
                                                                progress=False,
                                                                device=cognition_features.device,
                                                                eta=0.0
                                                                )
        else:
            # DDPM Sampling
            samples = self.action_model.diffusion.p_sample_loop(sample_fn, 
                                                                    noise.shape, 
                                                                    noise, 
                                                                    clip_denoised=False,
                                                                    model_kwargs=model_kwargs,
                                                                    progress=False,
                                                                    device=cognition_features.device
                                                                    )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        normalized_actions = samples[0].cpu().numpy()

        # Un-normalize Actions        
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions, normalized_actions

    @torch.inference_mode()
    def predict_action_batch(
        self, image: List[Image], 
        instruction: List[str], 
        unnorm_key: Optional[str] = None, 
        cfg_scale: float = 1.5, 
        use_ddim: bool = False,
        num_ddim_steps: int = 10,
        **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference in batch; maps input image and task instruction to continuous action.
        This function is used for batch inference in the simulators.
        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.
        @param cfg_scale: Scaling factor for classifier-free guidance (CFG); if == 1.0, CFG is disabled.
        @param use_ddim: Use DDIM sampling instead of DDPM sampling.
        @param num_ddim_steps: Number of DDIM steps to use for sampling.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        image_transform, tokenizer = self.vlm.vision_backbone.image_transform, self.vlm.llm_backbone.tokenizer
        
        input_ids = []
        pixel_values = []

        # Build VLA Prompt
        B = len(image)

        if isinstance(tokenizer, LlamaTokenizerFast):
            pass
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        for id in range(B):
            prompt_builder = self.vlm.get_prompt_builder()
            prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction[id].lower()}?")
            prompt_text = prompt_builder.get_prompt()
            # Prepare Inputs
            single_input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vlm.device).squeeze(0)
            # Note: We need to add this special empty token ('') after the colon (':') token in "ASSISTANT:"
            #       insert it to match the inputs seen at training time. The empty token is at index 29871.
            #       We also need to add the special cognition token at index 2 (i.e. the EOS token).
            single_input_ids = torch.cat(
                (single_input_ids, torch.Tensor([29871, 2]).long().to(self.vlm.device)), dim=0
            ) # [seq]

            input_ids.append(single_input_ids)
            # Preprocess Image
            pixel_values.append(image_transform(image[id]))

        # Padding
        padding_side = "right"
        # For now, we only support Tokenizers with `padding_side = "right"`
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert padding_side == "right", f"Invalid Tokenizer `{padding_side = }`"

        model_max_length = tokenizer.model_max_length
        pad_token_id = tokenizer.pad_token_id
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)

        # Truncate (if necessary)
        input_ids = input_ids[:, : model_max_length]
        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(pad_token_id)

        # Preprocess Image
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values).to(self.vlm.device)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]).to(self.vlm.device) for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.vlm.llm_backbone.half_precision_dtype
        with self._autocast(autocast_dtype):
            # fmt: off
            output = super(PrismaticVLM, self.vlm).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
                max_new_tokens=1,
                output_hidden_states=True, 
                return_dict_in_generate=True,
                attention_mask = attention_mask,
                **kwargs
            )
            # fmt: on

        # Extract cognition feature
        if self.vlm.vision_backbone.featurizer is not None:
            num_patch = self.vlm.vision_backbone.featurizer.patch_embed.num_patches
        elif hasattr(self.vlm.vision_backbone, 'siglip_featurizer') and self.vlm.vision_backbone.siglip_featurizer is not None:
            num_patch = self.vlm.vision_backbone.siglip_featurizer.patch_embed.num_patches
        else:
            raise ValueError("No vision backbone found")

        last_hidden = output.hidden_states[0][-1]
        last_hidden = last_hidden[:, num_patch :]

        cumulative_sum = attention_mask.cumsum(dim=1)  
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)  
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, last_hidden.size(-1))  
        cognition_features = last_hidden.gather(1, expanded_indices.unsqueeze(1)).squeeze(1) #[B, D]

        assert (cognition_features.shape[0], cognition_features.shape[1]) == (B, 4096), "Batch size must be B for action prediction"
        using_cfg = cfg_scale > 1.0


        model_dtype = next(self.action_model.net.parameters()).dtype

        B = cognition_features.shape[0]
        
        cognition_features = cognition_features.unsqueeze(1).to(model_dtype)  # [B, 1, D]

        # Sample random noise
        noise = torch.randn(B, self.future_action_window_size+1, self.action_model.in_channels, device=cognition_features.device).to(model_dtype)  #[B, T, D]
        # Setup classifier-free guidance:
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition = self.action_model.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0)  #[1, D]
            uncondition = uncondition.expand(B, 1, -1) #[B, 1, D]
            z = torch.cat([cognition_features, uncondition], 0)
            cfg_scale = cfg_scale
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.action_model.net.forward_with_cfg
        else:
            model_kwargs = dict(z=cognition_features)
            sample_fn = self.action_model.net.forward

        # DDIM Sampling
        if use_ddim and num_ddim_steps is not None:
            if self.action_model.ddim_diffusion is None:
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(sample_fn, 
                                                                noise.shape, 
                                                                noise, 
                                                                clip_denoised=False,#False, try to set True 
                                                                model_kwargs=model_kwargs,
                                                                progress=False,
                                                                device=cognition_features.device,
                                                                eta=0.0)
        else:
            # DDPM Sampling
            samples = self.action_model.diffusion.p_sample_loop(sample_fn, 
                                                                    noise.shape, 
                                                                    noise, 
                                                                    clip_denoised=False,#False, try to set True 
                                                                    model_kwargs=model_kwargs,
                                                                    progress=False,
                                                                    device=cognition_features.device)
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        normalized_actions = samples.cpu().numpy()

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, :, 6] = np.where(normalized_actions[:, :, 6] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions, normalized_actions

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]