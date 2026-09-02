# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Ported from the FastWAM-RL project (https://github.com/.../FastWAM-RL,
# Apache-2.0) at commit e269771 "define the interfaces"; the module content is
# the FastWAM model / flow-matching scheduler definition reused verbatim (only
# imports were made relative and the loguru logger replaced).

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .dit import ActionDiT, VideoDiT
from .flow_matching import FlowMatchingScheduler, FlowStates
from .llm import WanTextEncoder, WantTextEncoderTokenizer
from .mot import ExpertInputs, FastWAMMoT
from .vae import WanVideoVAE


@dataclass
class StepOutputs:
    video_loss: torch.Tensor
    action_loss: Optional[torch.Tensor] = None


@dataclass
class ModelInputs:
    video_latents: torch.Tensor
    action_latents: torch.Tensor
    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor
    frame_is_pad: torch.Tensor
    action_is_pad: torch.Tensor
    first_frame_latents: torch.Tensor


class ProprioEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.proj(x)


class FastWAM(nn.Module):
    use_gradient_checkpointing: bool = False

    def __init__(
        self,
        vae: WanVideoVAE,
        tokenizer: WantTextEncoderTokenizer,
        text_encoder: WanTextEncoder,
        proprio_encoder: ProprioEncoder,
        video_expert: VideoDiT,
        action_expert: ActionDiT,
        video_scheduler: FlowMatchingScheduler,
        action_scheduler: FlowMatchingScheduler,
    ):
        super().__init__()

        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.mot: FastWAMMoT = FastWAMMoT(video_expert, action_expert)
        self.mot.use_gradient_checkpointing = self.use_gradient_checkpointing
        self.proprio_encoder = proprio_encoder
        self.video_scheduler = video_scheduler
        self.action_scheduler = action_scheduler

    @property
    def video_expert(self) -> VideoDiT:
        return self.mot.mixtures["video"]

    @property
    def action_expert(self) -> ActionDiT:
        return self.mot.mixtures["action"]

    @classmethod
    def from_pretrained(
        cls: "FastWAM",
        pretrained_model_name_or_path: str,
        dtype: Optional[torch.dtype] = None,
    ) -> Self:
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        vae = WanVideoVAE.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae", dtype=dtype
        )
        tokenizer = WantTextEncoderTokenizer.from_pretrained(
            pretrained_model_name_or_path / "tokenizer"
        )
        text_encoder = WanTextEncoder.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder", dtype=dtype
        )
        proprio_encoder = ProprioEncoder.from_pretrained(
            pretrained_model_name_or_path, subfolder="proprio_encoder", dtype=dtype
        )
        video_expert = VideoDiT.from_pretrained(
            pretrained_model_name_or_path, subfolder="video_expert", dtype=dtype
        )
        action_expert = ActionDiT.from_pretrained(
            pretrained_model_name_or_path, subfolder="action_expert", dtype=dtype
        )
        video_scheduler = FlowMatchingScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="video_scheduler"
        )
        action_scheduler = FlowMatchingScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="action_scheduler"
        )
        return cls(
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            proprio_encoder=proprio_encoder,
            video_expert=video_expert,
            action_expert=action_expert,
            video_scheduler=video_scheduler,
            action_scheduler=action_scheduler,
        )

    def save_pretrained(self, save_dir: str):
        save_dir = Path(save_dir)
        self.vae.save_pretrained(save_dir / "vae")
        self.text_encoder.save_pretrained(save_dir / "text_encoder")
        self.proprio_encoder.save_pretrained(save_dir / "proprio_encoder")
        self.video_expert.save_pretrained(save_dir / "video_expert")
        self.action_expert.save_pretrained(save_dir / "action_expert")
        self.tokenizer.save_pretrained(save_dir / "tokenizer")
        self.video_scheduler.save_pretrained(save_dir / "video_scheduler")
        self.action_scheduler.save_pretrained(save_dir / "action_scheduler")

    def build_shared_attn_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros(
            (total_seq_len, total_seq_len), dtype=torch.bool, device=device
        )
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_self_attn_mask(
            video_seq_len, video_tokens_per_frame, device
        )
        mask[video_seq_len:, video_seq_len:] = True
        first_frame_len = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_len] = True
        return mask

    def encode_prompts(
        self, prompts: list[str] | str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Tokenize and encode prompts into cross-attention conditioning.
        """
        device = next(self.text_encoder.parameters()).device
        ids, mask = self.tokenizer(prompts, return_mask=True, add_special_tokens=True)
        ids: torch.Tensor = ids.to(device)
        mask: torch.Tensor = mask.to(device=device, dtype=torch.bool)

        prompt_embeds: torch.Tensor = self.text_encoder(ids, mask)
        prompt_embeds = prompt_embeds.masked_fill(~mask.unsqueeze(-1), 0)
        return prompt_embeds, mask

    def encode_proprios(
        self, proprios: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = proprios.shape[0]
        device = next(self.proprio_encoder.parameters()).device

        initial_proprio = proprios[:, 0, :] if proprios.ndim == 3 else proprios
        initial_proprio = initial_proprio.to(device=device, dtype=proprios.dtype)

        proprio_embeds = self.proprio_encoder(initial_proprio.unsqueeze(1))
        proprio_embeds_mask = torch.ones(
            (batch_size, 1), device=device, dtype=torch.bool
        )
        return proprio_embeds, proprio_embeds_mask

    def encode_videos(self, videos: torch.Tensor, device: torch.device) -> torch.Tensor:
        return self.vae.encode(videos=videos, device=device)

    def prepare_inputs(
        self, sample: dict[str, list[str] | torch.Tensor]
    ) -> ModelInputs:
        """
        Preprocess inputs, put to device and convert dtype.

        prompt (str):
        video: [B,3,T,H,W] The video size should be dividisible by 16 and the frames should be T=(F-1)*4+1
        action: [B,T,C]
        proprio_states: [B,C]
        video_pad: [B,T]
        action_pad: [B,T]
        """
        prompts: list[str] = sample["prompt"]
        videos: torch.Tensor = sample["video"]
        actions: torch.Tensor = sample["action"]
        proprios: torch.Tensor = sample["proprio_states"]
        video_pad: torch.Tensor = sample["video_pad"]
        action_pad: torch.Tensor = sample["action_pad"]

        device = next(self.text_encoder.parameters()).device
        prompt_embeds, prompt_mask = self.encode_prompts(prompts)
        proprio_embeds, proprio_mask = self.encode_proprios(proprios)
        embeds = torch.cat([prompt_embeds, proprio_embeds], dim=1)
        mask = torch.cat([prompt_mask, proprio_mask], dim=1)

        video_latents = self.encode_videos(videos, device)
        first_frame_latents = video_latents[:, :, 0:1]

        # The prompt_embeds can be encoded text prompt or the concatenated text and state
        return ModelInputs(
            video_latents=video_latents,
            action_latents=actions.to(device=device, dtype=video_latents.dtype),
            prompt_embeds=embeds,
            prompt_embeds_mask=mask,
            first_frame_latents=first_frame_latents,
            frame_is_pad=video_pad.to(device=device, dtype=torch.bool),
            action_is_pad=action_pad.to(device=device, dtype=torch.bool),
        )

    def prepare_video_cache(
        self,
        first_frame_latents: torch.Tensor,
        action_horizon: int,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
    ) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
        r"""
        prompt_embeds: The encoded text prompts or the concatenated text and proprio states embeds.
        """
        zero_sigma = torch.zeros(
            first_frame_latents.shape[0], device=first_frame_latents.device
        )
        video_info = self.video_expert.preprocess(
            video_tokens=first_frame_latents,
            timestep=self.video_scheduler.convert_to_model_timesteps(zero_sigma),
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
        )

        video_length = video_info.hidden_states.shape[1]
        attention_mask = self.build_shared_attn_mask(
            video_seq_len=video_length,
            action_seq_len=action_horizon,
            video_tokens_per_frame=video_info.num_tokens_per_frame,
            device=first_frame_latents.device,
        )

        key_values = self.mot.prefill_video_cache(
            ExpertInputs(
                hidden_states=video_info.hidden_states,
                freqs=video_info.freqs,
                prompt_embeds=video_info.prompt_embeds,
                prompt_embeds_mask=video_info.prompt_embeds_mask,
                time_embeds=video_info.time_projs,
            ),
            attention_mask[:video_length, :video_length],
        )
        return key_values, attention_mask

    def predict_action(
        self,
        action_latents: torch.Tensor,
        action_sigma: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        action_info = self.action_expert.preprocess(
            action_tokens=action_latents,
            timestep=self.action_scheduler.convert_to_model_timesteps(action_sigma),
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
        )
        hidden_states = self.mot.forward_with_video_cache(
            ExpertInputs(
                action_info.hidden_states,
                action_info.freqs,
                action_info.prompt_embeds,
                action_info.prompt_embeds_mask,
                action_info.time_projs,
            ),
            video_kv_cache=video_kv_cache,
            attn_mask=attention_mask,
        )
        return self.action_expert.postprocess(hidden_states["action"])

    def predict_video_action(
        self,
        video_latents: torch.Tensor,
        action_latents: torch.Tensor,
        video_sigma: torch.Tensor,
        action_sigma: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_info = self.video_expert.preprocess(
            video_tokens=video_latents,
            timestep=self.video_scheduler.convert_to_model_timesteps(video_sigma),
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
        )
        action_info = self.action_expert.preprocess(
            action_tokens=action_latents,
            timestep=self.action_scheduler.convert_to_model_timesteps(action_sigma),
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
        )
        attention_mask = self.build_shared_attn_mask(
            video_info.hidden_states.shape[1],
            action_info.hidden_states.shape[1],
            video_info.num_tokens_per_frame,
            video_latents.device,
        )
        hidden_states = self.mot(
            video_inputs=ExpertInputs(
                video_info.hidden_states,
                video_info.freqs,
                video_info.prompt_embeds,
                video_info.prompt_embeds_mask,
                video_info.time_projs,
            ),
            action_inputs=ExpertInputs(
                action_info.hidden_states,
                action_info.freqs,
                action_info.prompt_embeds,
                action_info.prompt_embeds_mask,
                action_info.time_projs,
            ),
            attn_mask=attention_mask,
        )
        video_velocity = self.video_expert.postprocess(
            hidden_states["video"], video_info.time_embeds, video_info.grid_size
        )
        action_velocity = self.action_expert.postprocess(hidden_states["action"])
        return video_velocity, action_velocity

    def compute_video_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        sigma: torch.Tensor,
        video_pad: torch.Tensor,
    ) -> torch.Tensor:
        loss_field = F.mse_loss(pred.float(), gt.float(), reduction="none").mean(
            dim=(1, 3, 4)
        )  # [B,C,T,H,W]->[B,T]

        temporal_factor = self.vae.temporal_downsample_factor
        latent_is_pad = (
            video_pad[:, 1:].view(video_pad.shape[0], -1, temporal_factor).all(dim=-1)
        )
        valid_latents = ~latent_is_pad
        loss = (loss_field * valid_latents).sum(dim=-1) / valid_latents.sum(
            dim=-1
        ).clamp(min=1.0)

        weight = self.video_scheduler.training_weight(sigma)
        return (weight * loss).mean()

    def compute_action_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        sigma: torch.Tensor,
        action_pad: torch.Tensor,
    ) -> torch.Tensor:
        loss_field = F.mse_loss(pred.float(), gt.float(), reduction="none").mean(
            dim=-1
        )  # [B,T,C]->[B,T]

        valid_latents = ~action_pad
        loss = (loss_field * valid_latents).sum(dim=-1) / valid_latents.sum(
            dim=-1
        ).clamp(min=1.0)

        weight = self.action_scheduler.training_weight(sigma)
        return (weight * loss).mean()

    def stochastic_action_sampling_step(
        self,
        prompt_embeds: torch.Tensor,
        first_frame: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        sigma: torch.Tensor,
        next_sigma: torch.Tensor,
        noise_level: float,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        next_action: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> FlowStates:
        r"""
        One step sampling with differentiable action transition used by rollout replay.

        WARNING: This method return the predicted action via noised SDE based on the
            noise level. The generated action chunk contains `action_horizon` frames,
            however, it will be replaned with some first frames rather than all frames.
            So this method do not generate a complete spisode of a given task, it just
            generate next action chunk based on the current observation.
        """
        device, dtype = action.device, action.dtype
        first_frame = first_frame.to(device=device, dtype=dtype).unsqueeze(2)

        first_frame_latents = self.encode_videos(first_frame, device)
        proprio_embeds, proprio_embeds_mask = self.encode_proprios(proprio)

        # Concat
        cond_embeds = torch.cat([prompt_embeds, proprio_embeds], dim=1)
        cond_embeds_mask = torch.cat([prompt_embeds_mask, proprio_embeds_mask], dim=1)

        video_kv_cache, attention_mask = self.prepare_video_cache(
            first_frame_latents=first_frame_latents,
            action_horizon=action.shape[1],
            prompt_embeds=cond_embeds,
            prompt_embeds_mask=cond_embeds_mask,
        )

        action = action.to(device=device, dtype=dtype)
        sigma = sigma.to(device)
        next_sigma = next_sigma.to(device)

        velocity = self.predict_action(
            action_latents=action,
            action_sigma=sigma,
            prompt_embeds=cond_embeds,
            prompt_embeds_mask=cond_embeds_mask,
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
        )
        return self.action_scheduler.stochastic_step(
            sample=action,
            velocity=velocity,
            sigma=sigma,
            next_sigma=next_sigma,
            noise_level=noise_level,
            next_sample=None
            if next_action is None
            else next_action.to(device=device, dtype=dtype),
            generator=generator,
        )

    def training_step(self, sample: dict[str, str | torch.Tensor]) -> StepOutputs:
        r"""
        training_step will conduct supervised flow-matching training on both videos and actions.
        """
        inputs = self.prepare_inputs(sample)
        device = inputs.video_latents.device

        # Video
        video_latents = inputs.video_latents
        batch_size = video_latents.shape[0]
        video_noise = torch.randn_like(video_latents)
        video_sigma = self.video_scheduler.sample_training_sigmas(batch_size, device)
        video_xt = self.video_scheduler.add_noise(
            video_latents, video_noise, video_sigma
        )
        video_xt[:, :, 0:1] = inputs.first_frame_latents
        video_gt = self.video_scheduler.training_target(video_latents, video_noise)
        video_gt = video_gt[
            :, :, 1:
        ]  # The first latent frame is provided as conditioning.

        # Action
        action_latents = inputs.action_latents
        action_noise = torch.randn_like(action_latents)
        action_sigma = self.action_scheduler.sample_training_sigmas(batch_size, device)
        action_xt = self.action_scheduler.add_noise(
            action_latents, action_noise, action_sigma
        )
        action_gt = self.action_scheduler.training_target(action_latents, action_noise)

        pred_video, pred_action = self.predict_video_action(
            video_latents=video_xt,
            action_latents=action_xt,
            video_sigma=video_sigma,
            action_sigma=action_sigma,
            prompt_embeds=inputs.prompt_embeds,
            prompt_embeds_mask=inputs.prompt_embeds_mask,
        )
        pred_video = pred_video[:, :, 1:]  # The first latent frame is ground truth.

        # Loss
        video_loss = self.compute_video_loss(
            pred_video, video_gt, video_sigma, inputs.frame_is_pad
        )
        action_loss = self.compute_action_loss(
            pred_action, action_gt, action_sigma, inputs.action_is_pad
        )

        return StepOutputs(video_loss=video_loss, action_loss=action_loss)

    def forward(self, sample: dict[str, str | torch.Tensor]) -> StepOutputs:
        r"""
        A standard torch forward entrypoint for SFT training.
        """
        return self.training_step(sample)
