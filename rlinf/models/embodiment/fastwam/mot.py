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

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .dit import ActionDiT, DiTBlock, VideoDiT, apply_rope

logger = logging.getLogger(__name__)


@dataclass
class ExpertInputs:
    hidden_states: torch.Tensor
    freqs: torch.Tensor
    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor
    time_embeds: torch.Tensor


@dataclass
class AttentionInputs:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    msa_gate: torch.Tensor
    mlp_shift: torch.Tensor
    mlp_scale: torch.Tensor
    mlp_gate: torch.Tensor


class FastWAMMoT(nn.Module):
    experts = ["video", "action"]
    use_gradient_checkpointing: bool = False

    def __init__(
        self,
        video_dit: VideoDiT,
        action_dit: ActionDiT,
    ):
        """
        Args:
            video_dit and action_dit are derived from the same DiTBlocks with same num of layers.
        """
        super().__init__()

        self.mixtures = nn.ModuleDict({"video": video_dit, "action": action_dit})
        for name, expert in self.mixtures.items():
            logger.info(
                f"  Expert `{name}`: num_params={sum(p.numel() for p in expert.parameters()) / 1e9:.2f} B"
            )

    def preprocess_expert_attention_inputs(
        self,
        block: DiTBlock,
        hidden_states: torch.Tensor,
        freqs: torch.Tensor,
        time_embeds: torch.Tensor,
    ) -> AttentionInputs:
        msa_shift, msa_scale, msa_gate, mlp_shift, mlp_scale, mlp_gate = (
            block.split_modulation(time_embeds)
        )

        b, l, _ = hidden_states.shape
        attn_inputs = (1.0 + msa_scale) * block.norm1(hidden_states) + msa_shift
        q = block.self_attn.norm_q(block.self_attn.q(attn_inputs))
        k = block.self_attn.norm_k(block.self_attn.k(attn_inputs))
        v = block.self_attn.v(attn_inputs)

        num_heads = block.self_attn.num_heads
        head_dim = block.self_attn.head_dim
        q = q.view(b, l, num_heads, head_dim).transpose(1, 2)
        k = k.view(b, l, num_heads, head_dim).transpose(1, 2)
        v = v.view(b, l, num_heads, head_dim).transpose(1, 2)

        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        return AttentionInputs(
            query=q,
            key=k,
            value=v,
            msa_gate=msa_gate,
            mlp_shift=mlp_shift,
            mlp_scale=mlp_scale,
            mlp_gate=mlp_gate,
        )

    def postprocess_expert_attention_outputs(
        self,
        block: DiTBlock,
        hidden_states: torch.Tensor,
        attn_outputs: torch.Tensor,
        msa_gate: torch.Tensor,
        mlp_shift: torch.Tensor,
        mlp_scale: torch.Tensor,
        mlp_gate: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + msa_gate * block.self_attn.o(attn_outputs)
        hidden_states = hidden_states + block.cross_attn(
            block.norm3(hidden_states),
            prompt_embeds,
            attn_mask=prompt_embeds_mask,
        )

        mlp = block.ffn((1.0 + mlp_scale) * block.norm2(hidden_states) + mlp_shift)
        hidden_states = hidden_states + mlp_gate * mlp
        return hidden_states

    def prefill_video_cache(
        self,
        video_inputs: ExpertInputs,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> list[dict[str, torch.Tensor]]:
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_index, video_block in enumerate(self.mixtures["video"].blocks):
            # [B,L,C] -> [B,H,L,C]
            attn_inputs = self.preprocess_expert_attention_inputs(
                block=video_block,
                hidden_states=video_inputs.hidden_states,
                freqs=video_inputs.freqs,
                time_embeds=video_inputs.time_embeds,
            )
            kv_cache.append({"key": attn_inputs.key, "value": attn_inputs.value})
            attn = F.scaled_dot_product_attention(
                query=attn_inputs.query,
                key=attn_inputs.key,
                value=attn_inputs.value,
                attn_mask=attn_mask,
            )
            # [B,H,L,C] -> [B,L,C]
            attn = attn.transpose(1, 2).reshape(
                attn.shape[0], attn.shape[2], video_block.self_attn.attn_dim
            )
            video_inputs.hidden_states = self.postprocess_expert_attention_outputs(
                block=video_block,
                hidden_states=video_inputs.hidden_states,
                attn_outputs=attn,
                msa_gate=attn_inputs.msa_gate,
                mlp_shift=attn_inputs.mlp_shift,
                mlp_scale=attn_inputs.mlp_scale,
                mlp_gate=attn_inputs.mlp_gate,
                prompt_embeds=video_inputs.prompt_embeds,
                prompt_embeds_mask=video_inputs.prompt_embeds_mask,
            )
        return kv_cache

    def forward_with_video_cache(
        self,
        action_inputs: ExpertInputs,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attn_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        video_seq_len = video_kv_cache[0]["key"].shape[2]
        action_seq_len = action_inputs.hidden_states.shape[1]
        seq_len = video_seq_len + action_seq_len

        action_attn_mask = (
            None if attn_mask is None else attn_mask[video_seq_len:seq_len, :seq_len]
        )

        for layer_index, action_block in enumerate(self.mixtures["action"].blocks):
            # [B,L,C] -> [B,H,L,C]
            attn_inputs = self.preprocess_expert_attention_inputs(
                block=action_block,
                hidden_states=action_inputs.hidden_states,
                freqs=action_inputs.freqs,
                time_embeds=action_inputs.time_embeds,
            )
            k_cache = video_kv_cache[layer_index]["key"]
            v_cache = video_kv_cache[layer_index]["value"]

            key = torch.cat([k_cache, attn_inputs.key], dim=2)
            value = torch.cat([v_cache, attn_inputs.value], dim=2)
            attn = F.scaled_dot_product_attention(
                query=attn_inputs.query,
                key=key,
                value=value,
                attn_mask=action_attn_mask,
            )
            # [B,H,L,C] -> [B,L,C]
            attn = attn.transpose(1, 2).reshape(
                attn.shape[0], action_seq_len, action_block.self_attn.attn_dim
            )
            action_inputs.hidden_states = self.postprocess_expert_attention_outputs(
                block=action_block,
                hidden_states=action_inputs.hidden_states,
                attn_outputs=attn,
                msa_gate=attn_inputs.msa_gate,
                mlp_shift=attn_inputs.mlp_shift,
                mlp_scale=attn_inputs.mlp_scale,
                mlp_gate=attn_inputs.mlp_gate,
                prompt_embeds=action_inputs.prompt_embeds,
                prompt_embeds_mask=action_inputs.prompt_embeds_mask,
            )
        return {
            "video": None,
            "action": action_inputs.hidden_states,
        }

    def forward(
        self,
        video_inputs: ExpertInputs,
        action_inputs: ExpertInputs,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        for layer_index, (video_block, action_block) in enumerate(
            zip(self.mixtures["video"].blocks, self.mixtures["action"].blocks)
        ):
            # [B,L,C] -> [B,H,L,C]
            video_attn_inputs = self.preprocess_expert_attention_inputs(
                block=video_block,
                hidden_states=video_inputs.hidden_states,
                freqs=video_inputs.freqs,
                time_embeds=video_inputs.time_embeds,
            )

            action_attn_inputs = self.preprocess_expert_attention_inputs(
                block=action_block,
                hidden_states=action_inputs.hidden_states,
                freqs=action_inputs.freqs,
                time_embeds=action_inputs.time_embeds,
            )

            batch_size = video_attn_inputs.query.shape[0]
            video_seq_len = video_attn_inputs.query.shape[2]
            action_seq_len = action_attn_inputs.query.shape[2]
            attn_dim = video_block.self_attn.attn_dim

            query = torch.cat(
                [video_attn_inputs.query, action_attn_inputs.query], dim=2
            )
            key = torch.cat([video_attn_inputs.key, action_attn_inputs.key], dim=2)
            value = torch.cat(
                [video_attn_inputs.value, action_attn_inputs.value], dim=2
            )

            if self.use_gradient_checkpointing:
                attn = checkpoint(
                    F.scaled_dot_product_attention,
                    use_reentrant=False,
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=attn_mask,
                )
            else:
                attn = F.scaled_dot_product_attention(
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=attn_mask,
                )
            # [B,H,L,C] -> [B,L,C]
            attn = attn.transpose(1, 2).reshape(
                batch_size, video_seq_len + action_seq_len, attn_dim
            )
            video_attn = attn[:, :video_seq_len]
            action_attn = attn[:, video_seq_len : video_seq_len + action_seq_len]

            video_attn = self.postprocess_expert_attention_outputs(
                video_block,
                video_inputs.hidden_states,
                video_attn,
                msa_gate=video_attn_inputs.msa_gate,
                mlp_shift=video_attn_inputs.mlp_shift,
                mlp_scale=video_attn_inputs.mlp_scale,
                mlp_gate=video_attn_inputs.mlp_gate,
                prompt_embeds=video_inputs.prompt_embeds,
                prompt_embeds_mask=video_inputs.prompt_embeds_mask,
            )
            video_inputs.hidden_states = video_attn

            action_attn = self.postprocess_expert_attention_outputs(
                action_block,
                action_inputs.hidden_states,
                action_attn,
                msa_gate=action_attn_inputs.msa_gate,
                mlp_shift=action_attn_inputs.mlp_shift,
                mlp_scale=action_attn_inputs.mlp_scale,
                mlp_gate=action_attn_inputs.mlp_gate,
                prompt_embeds=action_inputs.prompt_embeds,
                prompt_embeds_mask=action_inputs.prompt_embeds_mask,
            )
            action_inputs.hidden_states = action_attn
        return {
            "video": video_inputs.hidden_states,
            "action": action_inputs.hidden_states,
        }
