:orphan:

.. _fastwam_flow_grpo:

FastWAM + Flow-GRPO on LIBERO-Plus (embodied RL)
==================================================

This page documents the RLinf-native support for

* the **FastWAM** world-action model (model definition ported from the
  FastWAM-RL project, Apache-2.0);
* the **Flow-GRPO** online-RL algorithm for flow-matching policies
  (arXiv:2505.05470 and the ``flow_grpo`` reference implementation);
* the **LIBERO-Plus** robustness benchmark (sylvestf/LIBERO-plus fork) as the
  simulation environment with sparse success rewards.

Code layout
-----------

* ``rlinf/models/embodiment/fastwam/`` — vendored FastWAM model files
  (``llm.py``/``dit.py``/``mot.py``/``vae.py``/``wam.py`` and the
  ``FlowMatchingScheduler`` in ``flow_matching.py``), the RL wrapper
  ``FastWAMActionPolicy`` (``policy.py``), the torch-only math helpers
  (``flow_likelihood.py``) and the ``get_model`` builder (``__init__.py``).
* ``rlinf/envs/libero/libero_splits.py`` — deterministic train/test task splits
  of the LIBERO-Plus suites (used through the env ``suite_split`` block).
* Example configs: ``examples/embodiment/config/model/fastwam.yaml``,
  ``examples/embodiment/config/env/libero_plus_spatial_{train,eval}.yaml`` and
  ``examples/embodiment/config/libero_plus_fastwam_flow_grpo.yaml``.
* E2E smoke config: ``tests/e2e_tests/embodied/libero_plus_fastwam_flow_grpo.yaml``.

Model interface (RLinf contracts)
---------------------------------

The wrapper implements the embodied-policy contract of RLinf
(``BasePolicy``-style ``predict_action_batch`` / ``default_forward``; the model
is registered as ``model_type: fastwam``):

* ``predict_action_batch(env_obs, mode)`` encodes the current two-camera frame
  (224x224 each, concatenated horizontally), the instruction and the proprio
  state, then denoises an action chunk ``[B, num_action_chunks, 7]`` through
  the action flow. In ``mode="train"`` each row performs exactly one uniformly
  random stochastic reverse-SDE transition (the ``window=1`` Flow-GRPO scheme);
  all other steps are deterministic Euler-ODE steps. The per-element Gaussian
  log-probability of that transition is returned as ``prev_logprobs``
  (``[B, T, C]``) and the transition (``x_t``/``x_next``/``denoise_inds``)
  together with the conditioning latents is cached in ``forward_inputs``.
* ``default_forward(forward_inputs)`` re-encodes the frozen conditioning once
  and recomputes the recorded transition likelihood under the *current* action
  expert, grouped by ``denoise_inds``.
* Prompt conditioning is padded to a fixed ``actor.model.cond_max_len`` so the
  cached ``forward_inputs`` of different rollout calls (different task
  instructions / chunk boundaries) stack into one actor micro batch.

Only the FastWAM **action expert** is trained; the video expert (used only for
KV prefill of the conditioning frame), the video VAE and the text encoder stay
frozen. There is no critic/value head (GRPO is critic-free); an optional
``add_value_head`` is intentionally not provided. Configuration-time validation
(``validate_embodied_cfg``) enforces the FastWAM structural requirements
(``stats_path``, chunk/flow-step/noise settings, no LoRA) and warns when the
algorithm block deviates from the Flow-GRPO combination.

Flow-GRPO formulation
---------------------

With the log-probabilities above, the Flow-GRPO update is exactly RLinf's
generic GRPO + clipped-ratio policy surrogate:

* advantage: GRPO group normalization of the episode return over
  ``algorithm.group_size`` envs that share one ``(task, initial state)`` group
  (``algorithm.adv_type: grpo``);
* loss: ``loss_type: actor`` with ``logprob_type: chunk_level`` computes the
  clipped importance ratio ``exp(sum(log p_new - log p_old))`` over the chunk;
* rewards: the simulator returns a sparse ``1.0`` exactly on the terminal step
  of a successful episode (LIBERO ``done == _check_success()``), ``0``
  otherwise; episode returns are accumulated per chunk-step and grouped like in
  any RLinf embodied GRPO run.

This matches the reference Flow-GRPO formulas (per-transition Gaussian
likelihood, group-normalized advantage, ratio clipping; the optional
KL-to-base term is left disabled with ``kl_beta: 0`` for full action-expert
fine-tuning).

Prerequisites and data
----------------------

1. Install the LIBERO-Plus fork as the ``libero`` package
   (``pip install -e .`` in ``FastWAM-RL/submodules/LIBERO-plus``) and its
   assets (see its README); keep ``LIBERO_TYPE=standard`` (the fork is a drop-in
   replacement of vanilla LIBERO).
2. Provide a FastWAM checkpoint directory with the diffusers-style subfolders
   ``vae``, ``text_encoder``, ``proprio_encoder``, ``video_expert``,
   ``action_expert``, ``tokenizer``, ``video_scheduler``, ``action_scheduler``
   (weights supplied separately).
3. Provide ``dataset_stats.json`` (keys ``action``/``state``) — required to
   denormalize action chunks into environment commands; point
   ``actor.model.stats_path`` at it.
4. ``diffusers``/``transformers`` are imported lazily by the model builder.

Train/test split
----------------

LIBERO-Plus inflates four vanilla suites to ~10,000 perturbed tasks from 10
base stems per suite. ``suite_split`` in the env config splits *base stems*
deterministically (fixed seed) so perturbed variants of one origin task never
leak across train and test:

.. code-block:: yaml

   task_suite_name: libero_spatial
   suite_split:
     seed: 0
     subset: train        # or test
     train_fraction: 0.8

Resolved indices feed RLinf's existing ``task_id_filter`` mechanism. You can
inspect a split with ``python -m rlinf.envs.libero.libero_splits``.

Run
---

.. code-block:: bash

   LIBERO_TYPE=standard bash examples/embodiment/run_embodiment.sh \
       libero_plus_fastwam_flow_grpo
