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

"""RLinf RL wrapper around the FastWAM flow-matching action policy.

The wrapper exposes the FastWAM model (a video-conditioned action flow) through
the embodied-policy interface RLinf expects:

* :meth:`predict_action_batch` rolls an action chunk out through the action
  flow. In ``mode="train"`` each row picks one *uniformly random* denoising
  step that is taken as a stochastic reverse-SDE transition (the ``window=1``
  Flow-GRPO / MixGRPO scheme, identical in spirit to the pi0-family RL port in
  ``rlinf.models.embodiment.openpi_rlinf``); all other steps are deterministic
  Euler-ODE steps. The per-element Gaussian log-probability of the stochastic
  transition is recorded as ``prev_logprobs`` (shape ``[B, T, C]``) and the
  transition itself (``x_t``, ``x_next``, ``denoise_inds`` plus the frozen
  conditioning latents) is cached in ``forward_inputs``.
* :meth:`default_forward` recomputes that log-probability under the *current*
  weights, so the generic GRPO advantage + clipped-ratio actor loss
  (``algorithm.adv_type: grpo`` / ``algorithm.loss_type: actor`` with
  ``logprob_type: chunk_level``) implements the Flow-GRPO update.

Only the FastWAM ``action_expert`` is trained. The video expert (KV prefill for
the conditioning frame), the video VAE and the text encoder stay frozen; no
critic/value head is attached (GRPO is critic-free).

Normalization
-------------
FastWAM was trained on min-max normalized actions/states described by
``dataset_stats.json`` (``action`` / ``state`` entries). The wrapper expects the
same file (config key ``model.stats_path``) so sampled chunks can be
denormalized to environment commands and env proprio states normalized into the
conditioning space.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType

from .flow_likelihood import (
    denormalize_minmax,
    elementwise_gaussian_log_prob,
    normalize_minmax,
    transition_mean_std,
)
from .wam import FastWAM

_PROMPT_TEMPLATE = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)


class FastWAMActionPolicy(nn.Module, BasePolicy):
    """RLinf embodied-policy interface over the FastWAM action flow."""

    def __init__(
        self,
        model: FastWAM,
        *,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
        proprio_min: torch.Tensor,
        proprio_max: torch.Tensor,
        num_flow_steps: int = 10,
        noise_level: float = 1.0,
        seed: Optional[int] = None,
        cond_max_len: int = 512,
    ):
        super().__init__()
        self.model = model

        self.action_dim: int = int(action_min.numel())
        # Chunk length the env expects (cfg ``actor.model.num_action_chunks``);
        # assigned by the builder since it comes from the experiment YAML.
        self.action_horizon: int = 32
        self.num_flow_steps = int(num_flow_steps)
        self.noise_level = float(noise_level)
        self.seed = seed
        # Fixed prompt-conditioning length (padded, see _pad_prompt_conditioning)
        # so cached forward_inputs of different rollout calls can be stacked.
        self.cond_max_len = int(cond_max_len)

        # Statistics travel with the model (registered buffers follow .to()).
        self.register_buffer("action_min", action_min, persistent=False)
        self.register_buffer("action_max", action_max, persistent=False)
        self.register_buffer("proprio_min", proprio_min, persistent=False)
        self.register_buffer("proprio_max", proprio_max, persistent=False)

        self._freeze_conditioning_networks()

    # ------------------------------------------------------------------ setup

    def _freeze_conditioning_networks(self) -> None:
        """Freeze every FastWAM component except the action expert."""
        for module in (
            self.model.vae,
            self.model.text_encoder,
            self.model.proprio_encoder,
            self.model.video_expert,
        ):
            for param in module.parameters():
                param.requires_grad = False

    def set_global_step(self, global_step: int) -> None:
        """Interface-parity hook; Flow-GRPO keeps a constant ``noise_level``."""
        del global_step

    # ------------------------------------------------------------ conditioning

    @torch.no_grad()
    def _encode_condition(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Encode prompt + proprio + current first-frame for one obs batch.

        The returned dict feeds both rollout and training recompute:
        ``first_frame_latents`` (video-VAE latent of the concatenated current
        frames) and ``cond_embeds``/``cond_mask`` (text + proprio embeds).
        """
        device = next(self.model.text_encoder.parameters()).device
        main_images = env_obs["main_images"]  # [B, H, W, 3]
        wrist_images = env_obs.get("wrist_images", None)
        states = env_obs["states"]  # [B, C_p]
        task_descriptions = env_obs["task_descriptions"]  # list[str]

        prompts = [_PROMPT_TEMPLATE.format(task=t) for t in task_descriptions]
        prompt_embeds, prompt_mask = self.model.encode_prompts(prompts)
        if prompt_embeds.shape[0] != main_images.shape[0]:
            # encode_prompts may collapse equal prompts; re-expand to rows.
            if len(prompts) == 1 and main_images.shape[0] > 1:
                prompt_embeds = prompt_embeds.expand(main_images.shape[0], -1, -1)
                prompt_mask = prompt_mask.expand(main_images.shape[0], -1)
        prompt_embeds, prompt_mask = self._pad_prompt_conditioning(
            prompt_embeds, prompt_mask
        )

        proprio = normalize_minmax(
            states.to(device=device, dtype=torch.float32),
            self.proprio_min.to(device=device),
            self.proprio_max.to(device=device),
        )
        proprio_embeds, proprio_mask = self.model.encode_proprios(proprio)

        cond_embeds = torch.cat([prompt_embeds, proprio_embeds], dim=1)
        cond_mask = torch.cat([prompt_mask, proprio_mask], dim=1)

        video = self._stack_current_frames(main_images, wrist_images, device)
        first_frame_latents = self.model.encode_videos(video, device)

        return {
            "first_frame_latents": first_frame_latents,
            "cond_embeds": cond_embeds,
            "cond_mask": cond_mask,
        }

    def _pad_prompt_conditioning(
        self,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad prompt embeddings to a fixed sequence length.

        Different rollout calls (different tasks / chunk boundaries) may encode
        prompts of different lengths; RLinf stacks cached ``forward_inputs``
        rows of a micro batch, so all cached conditioning must share one
        sequence length. Padded positions carry a ``False`` mask and are never
        attended to by the experts.
        """
        seq_len = prompt_embeds.shape[1]
        if seq_len > self.cond_max_len:
            raise ValueError(
                f"Prompt encoding length {seq_len} exceeds cond_max_len "
                f"({self.cond_max_len}); increase actor.model.cond_max_len."
            )
        if seq_len == self.cond_max_len:
            return prompt_embeds, prompt_mask
        pad_len = self.cond_max_len - seq_len
        prompt_embeds = torch.nn.functional.pad(prompt_embeds, (0, 0, 0, pad_len))
        prompt_mask = torch.nn.functional.pad(prompt_mask, (0, pad_len), value=False)
        return prompt_embeds, prompt_mask

    def _stack_current_frames(
        self,
        main_images: torch.Tensor,
        wrist_images: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Build the ``[B, 3, 1, 224, 448]`` video tensor of the current frame.

        Images may arrive as ``uint8`` HWC tensors (LIBERO default); they are
        normalized to floats in ``[0, 1]``, resized to 224 px and concatenated
        horizontally (``image | wrist_image``), matching the FastWAM SFT data
        contract (``video_size: [224, 224]``, cameras ``image, wrist_image``).
        """
        frames = []
        for img in (main_images, wrist_images):
            if img is None:
                continue
            img = img.to(device=device)
            if img.ndim == 4:  # [B, H, W, 3] -> [B, 3, H, W]
                img = img.permute(0, 3, 1, 2)
            if img.dtype in (torch.uint8, torch.int32, torch.int64):
                img = img.to(dtype=torch.float32) / 255.0
            elif img.is_floating_point() and img.max() > 1.5:
                # Some env wrappers deliver float tensors that still carry
                # 0-255 pixel values; normalize them as well.
                img = img / 255.0
            img = F.interpolate(
                img, size=(224, 224), mode="bilinear", align_corners=False
            ).clamp(0.0, 1.0)
            frames.append(img)
        video = torch.cat(frames, dim=-1)  # [B, 3, H, 2W]
        return video.unsqueeze(2)  # single frame: [B, 3, 1, H, 2W]

    # ------------------------------------------------------------------ rollout

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        compute_values: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample a full action chunk through the FastWAM action flow.

        Returns:
            actions: denormalized chunk ``[B, action_horizon, C]`` ready for
                the environment (LIBERO performs no post-processing for this
                model type, see ``rlinf.envs.action_utils``).
            result: ``prev_logprobs`` (``None`` in eval mode), ``prev_values``
                (zeros; GRPO is critic-free) and the cached ``forward_inputs``.
        """
        del compute_values, kwargs
        device = next(self.model.text_encoder.parameters()).device
        batch_size = int(env_obs["main_images"].shape[0])
        chunk_len = int(self.action_horizon)
        action_dim = int(self.action_dim)

        cond = self._encode_condition(env_obs)

        num_steps = self.num_flow_steps
        sigmas = self.model.action_scheduler.sample_inference_sigmas(num_steps, device)

        if self.seed is not None:
            rng = torch.Generator(device=device).manual_seed(self.seed)
        else:
            rng = None

        # One uniformly random stochastic (SDE) denoise step per row, i.e. the
        # Flow-GRPO "window=1" scheme; every other step is deterministic Euler.
        if mode == "train":
            denoise_inds = torch.randint(0, num_steps, (batch_size,), device=device)
        else:
            denoise_inds = torch.zeros(batch_size, dtype=torch.long, device=device)

        x_t = torch.randn(
            (batch_size, chunk_len, action_dim),
            device=device,
            dtype=torch.float32,
            generator=rng,
        )
        log_probs = torch.zeros(
            (batch_size, chunk_len, action_dim),
            device=device,
            dtype=torch.float32,
        )
        x_pre = torch.zeros_like(x_t)
        x_next = torch.zeros_like(x_t)
        stochastic_rows = (
            torch.ones(batch_size, dtype=torch.bool, device=device)
            if mode == "train"
            else torch.zeros(batch_size, dtype=torch.bool, device=device)
        )

        video_kv_cache, attention_mask = self.model.prepare_video_cache(
            first_frame_latents=cond["first_frame_latents"],
            action_horizon=chunk_len,
            prompt_embeds=cond["cond_embeds"],
            prompt_embeds_mask=cond["cond_mask"],
        )

        for step in range(num_steps):
            sigma = sigmas[step].expand(batch_size)
            next_sigma = sigmas[step + 1].expand(batch_size)
            velocity = self.model.predict_action(
                action_latents=x_t,
                action_sigma=sigma,
                prompt_embeds=cond["cond_embeds"],
                prompt_embeds_mask=cond["cond_mask"],
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
            )
            # Every row advances from ``sigma`` to ``next_sigma`` in this step:
            # rows whose denoise index was chosen take a stochastic reverse-SDE
            # transition (recording the per-element Gaussian log-probability and
            # the transition pair for the actor-side recompute), all other rows
            # take a deterministic Euler-ODE transition.
            sde_rows = (denoise_inds == step) & stochastic_rows
            if sde_rows.any():
                mean, std = transition_mean_std(
                    x_t, velocity, sigma, next_sigma, self.noise_level
                )
                noise = torch.randn(
                    x_t.shape, device=device, dtype=torch.float32, generator=rng
                )
                sampled = mean + std * noise
                if mode == "train":
                    log_probs[sde_rows] = elementwise_gaussian_log_prob(
                        sampled[sde_rows], mean[sde_rows], std[sde_rows]
                    )
                x_pre[sde_rows] = x_t[sde_rows]
                x_next[sde_rows] = sampled[sde_rows]
                if sde_rows.all():
                    x_t = sampled
                else:
                    # ODE rows still advance on the same (sigma, next_sigma).
                    ode_rows = ~sde_rows
                    ode_next = self.model.action_scheduler.step(
                        x_t[ode_rows],
                        velocity[ode_rows],
                        sigma[ode_rows],
                        next_sigma[ode_rows],
                    )
                    x_t = torch.empty_like(x_t)
                    x_t[sde_rows] = sampled[sde_rows]
                    x_t[ode_rows] = ode_next
            else:
                # Deterministic Euler-ODE transition for all rows.
                x_t = self.model.action_scheduler.step(x_t, velocity, sigma, next_sigma)

        # ``x_t`` is now the clean (denoised) action chunk.
        actions = denormalize_minmax(
            x_t,
            self.action_min.to(device=device),
            self.action_max.to(device=device),
        ).to(dtype=torch.float32)

        forward_inputs = {
            "x_t": x_pre,
            "x_next": x_next,
            "denoise_inds": denoise_inds,
            "first_frame_latents": cond["first_frame_latents"],
            "cond_embeds": cond["cond_embeds"],
            "cond_mask": cond["cond_mask"],
        }
        result = {
            "prev_logprobs": log_probs if mode == "train" else None,
            "prev_values": torch.zeros(
                (batch_size, 1), device=device, dtype=torch.float32
            ),
            "forward_inputs": forward_inputs,
        }
        return actions, result

    # ---------------------------------------------------------------- training

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        """RLinf actor entry: dispatch to :meth:`default_forward`."""
        if forward_type != ForwardType.DEFAULT:
            raise NotImplementedError(
                f"FastWAMActionPolicy only supports ForwardType.DEFAULT, got "
                f"{forward_type!r}."
            )
        return self.default_forward(**kwargs)

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        compute_logprobs: bool = True,
        compute_entropy: bool = False,
        compute_values: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Recompute the cached transition log-probability under current weights.

        The frozen conditioning is re-encoded once for the whole micro batch
        (gradients disabled), then the velocity network is applied per denoise
        index on the recorded ``x_t`` to rebuild the transition mean/std and
        the elementwise Gaussian log-probability of the recorded ``x_next``.
        """
        del compute_logprobs, compute_entropy, compute_values, kwargs
        device = next(self.model.text_encoder.parameters()).device

        x_t = forward_inputs["x_t"].to(device=device, dtype=torch.float32)
        x_next = forward_inputs["x_next"].to(device=device, dtype=torch.float32)
        denoise_inds = forward_inputs["denoise_inds"].to(device=device).long()
        batch_size = x_t.shape[0]
        chunk_len, action_dim = x_t.shape[1], x_t.shape[2]

        cond = {
            "first_frame_latents": forward_inputs["first_frame_latents"].to(device),
            "cond_embeds": forward_inputs["cond_embeds"].to(device),
            "cond_mask": forward_inputs["cond_mask"].to(device),
        }
        num_steps = self.num_flow_steps
        sigmas = self.model.action_scheduler.sample_inference_sigmas(num_steps, device)

        with torch.no_grad():
            video_kv_cache, attention_mask = self.model.prepare_video_cache(
                first_frame_latents=cond["first_frame_latents"],
                action_horizon=chunk_len,
                prompt_embeds=cond["cond_embeds"],
                prompt_embeds_mask=cond["cond_mask"],
            )

        log_probs = torch.zeros(
            (batch_size, chunk_len, action_dim),
            device=device,
            dtype=torch.float32,
        )
        # Group rows by their recorded denoise index to keep velocity forwards
        # batched while recomputing only the single stochastic transition.
        for step in range(num_steps):
            rows = torch.nonzero(denoise_inds == step, as_tuple=False).squeeze(-1)
            if rows.numel() == 0:
                continue
            sigma = sigmas[step].expand(rows.numel())
            next_sigma = sigmas[step + 1].expand(rows.numel())
            velocity = self.model.predict_action(
                action_latents=x_t[rows],
                action_sigma=sigma,
                prompt_embeds=cond["cond_embeds"][rows],
                prompt_embeds_mask=cond["cond_mask"][rows],
                video_kv_cache=_select_batch(video_kv_cache, rows),
                attention_mask=attention_mask,
            )
            mean, std = transition_mean_std(
                x_t[rows], velocity, sigma, next_sigma, self.noise_level
            )
            log_probs[rows] = elementwise_gaussian_log_prob(x_next[rows], mean, std)

        values = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)
        entropy = torch.zeros((batch_size,), device=device, dtype=torch.float32)
        return {
            "logprobs": log_probs,
            "values": values,
            "entropy": entropy,
        }


def _select_batch(
    kv_cache: list[dict[str, torch.Tensor]], rows: torch.Tensor
) -> list[dict[str, torch.Tensor]]:
    """Sub-select the batch dimension of a (nested) video KV cache."""
    selected = []
    for layer in kv_cache:
        selected.append(
            {key: value.index_select(0, rows) for key, value in layer.items()}
        )
    return selected
