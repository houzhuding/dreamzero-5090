# DreamZero Training (Flow Matching): Algorithm ↔ Code Mapping

This document maps your Algorithm 1 (chunk-wise flow matching training) to the DreamZero codebase implementation.

---

## 0) TL;DR: where training actually happens

- **Launch / entrypoint**
  - `scripts/train/droid_training_wan22.sh`
  - `scripts/train/droid_training_full_finetune.sh`
  - Both call: `groot/vla/experiment/experiment.py`

- **Experiment + Trainer orchestration**
  - `groot/vla/experiment/experiment.py`
  - `groot/vla/experiment/base.py`

- **Model forward path**
  - `groot/vla/model/dreamzero/base_vla.py` (`VLA.forward`)
  - `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py` (`WANPolicyHead.forward`)
  - `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py` (`CausalWanModel._forward_train`)

- **Flow matching noise/interpolation target math**
  - `groot/vla/model/dreamzero/modules/flow_match_scheduler.py`

- **Chunked data assembly and language-consistent sampling**
  - `groot/vla/data/dataset/lerobot_sharded.py`
  - `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py`

---

## 1) Your Algorithm 1 ↔ Code, step by step

Below I mirror your pseudocode numbering.

### Step 1: Input dataset $\mathcal D$, text condition $c$

**Algorithm**
- Input: trajectory dataset + text condition.

**Code**
- Dataset creation in experiment bootstrap:
  - `BaseExperiment.create_train_dataset()` in `groot/vla/experiment/base.py`
  - `cfg.train_dataset` usually points to `ShardedLeRobotMixtureDataset.from_mixture_spec`
- DROID config wiring:
  - `groot/vla/configs/data/dreamzero/droid_relative.yaml`
- Text condition generation/tokenization:
  - `DreamTransform._prepare_language()` in `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py`
  - `DefaultDataCollator` / `collate()` in same file
  - `WANPolicyHead.encode_prompt()` in `wan_flow_matching_action_tf.py`

**Important detail**
- `c` is not a single scalar token; it is full prompt embeddings from T5 (`text_encoder`), plus optional negative prompt text used in inference paths.

---

### Step 2: Hyperparam number of chunks $M$

**Algorithm**
- Training loops chunk-wise over $k=1..M$.

**Code mapping**
- There are two “chunk” layers in code:
  1. **Data chunking in dataset** (episode sampling windows)
     - `max_chunk_size` from dataset config (`droid_relative.yaml`)
     - Action chunk size is 24 steps (`get_action(...), _convert_to_relative_action(..., chunk_size=24)`)
  2. **Model-time token blocks**
     - `num_frame_per_block`, `num_action_per_block`, `num_state_per_block`
     - set in scripts and action-head configs

**Where this appears**
- `lerobot_sharded.py`:
  - state chunk-aligned anchor sampling (language consistent)
  - action chunk expansion per anchor (`24` steps per anchor block)
- `wan_video_dit_action_casual_chunk.py`:
  - blockwise multimodal causal attention (image/action/state blocks)

**Interpretation**
- Your paper-style explicit `for k in 1..M` is implemented mostly as **vectorized blockwise operations** + **structured causal masking/teacher-forcing**, rather than a Python `for k` in high-level training loop.

---

### Step 3: Model $u_\theta$ (joint video-action DiT)

**Algorithm**
- Joint model predicts velocity/noise for video and action.

**Code**
- Main class: `CausalWanModel` in `wan_video_dit_action_casual_chunk.py`
- Wrapped by action head:
  - `WANPolicyHead` in `wan_flow_matching_action_tf.py`
- Top-level VLA wrapper:
  - `VLA` in `base_vla.py`

**Structure**
- Video path: VAE latents + Wan DiT blocks + unpatchify to video-noise prediction.
- Action path: action encoder/register + action decoder for action-noise prediction.
- Joint attention uses blockwise causal rules across image/action/state token regions.

---

### Step 4–5: while not converged, sample trajectory $\tau \sim \mathcal D$

**Code**
- `self.trainer.train(...)` in `BaseExperiment.train()` (`base.py`)
- HF Trainer’s training loop drives batch sampling.
- Dataloader built in `BaseTrainer.get_train_dataloader` pathway (`base.py`).
- Sampler: `BaseSampler` with epoch-seeded shuffle.

**Convergence criterion**
- practical stopping is `max_steps` / epochs from `TrainingArguments`.

---

### Step 6: encode video to clean latents $z^1$, normalize actions $a^1$

**Video clean latent $z^1$**
- `WANPolicyHead.forward()`:
  - rearrange + normalize input videos to `[-1, 1]`
  - `latents = self.encode_video(...)` via VAE encoder

**Action normalization / canonicalization**
- happens before model forward in data transforms:
  - `StateActionTransform` in dataset transform pipeline (configured in `base_48_wan_fine_aug_relative.yaml`)
  - `DreamTransform._prepare_action()` pads to `max_action_dim` and builds `action_mask`
  - optional relative-action conversion in dataset (`lerobot_sharded.py`, `_convert_to_relative_action`)

---

### Step 7: split $\tau$ into $M$ chunks

**Code**
- Dataset sampling in `lerobot_sharded.py` does chunk-consistent selection:
  - state anchors sampled in strides of 24
  - action windows sampled in 24-step chunks per anchor
  - language consistency enforced while expanding around anchor
  - number of chunks aligned with video chunk count via `_current_num_chunks`

**Key point**
- Chunk split is mostly a **data-loader responsibility** + model block layout; not a simple explicit list split inside trainer.

---

### Step 8–9: per-chunk training with clean context $\mathcal C_k$ (teacher-forcing history)

**Algorithm intent**
- each chunk sees clean history context from previous chunks.

**Code realization (very important)**
- In training, `WANPolicyHead.forward()` calls model with:
  - `clean_x=latents.transpose(1, 2)`
- In `CausalWanModel._forward_train(...)`:
  - if `clean_x is not None`, it concatenates clean token stream before noisy stream
  - sets `is_tf=True`
- In `CausalWanSelfAttention.forward(..., is_tf=True)`:
  - split clean and noisy halves
  - apply dedicated teacher-forcing attention pattern where noisy blocks attend to clean prior context and aligned current action/state blocks

So your $\mathcal C_k$ is implemented by **teacher-forcing clean-token half + blockwise causal attention policy**, not by a literal Python dictionary/set of previous chunk tensors.

---

### Step 10: sample timestep $t_k \sim \mathcal U(0,1)$

**Standard mode**
- `WANPolicyHead.forward()` samples integer timestep ids uniformly:
  - `torch.randint(0, num_train_timesteps, ...)`
- Then mapped to scheduler’s continuous sigma/timestep values:
  - `timestep = self.scheduler.timesteps[timestep_id]`

**Equivalent interpretation**
- Uniform over discrete bucketed timesteps approximates the $\mathcal U(0,1)$ draw.

---

### Step 11–16: optional DreamZero-Flash decoupling ($t_{vid}$ vs $t_{act}$)

**Code support exists directly**
- Config flags in `WANPolicyHeadConfig`:
  - `decouple_video_action_noise`
  - `video_noise_beta_alpha`, `video_noise_beta_beta`
  - `use_high_noise_emphasis` (coupled high-noise mode)
- In `WANPolicyHead.forward()`:
  - if decoupled: video timestep from Beta, action timestep from independent uniform
  - else coupled: action timestep derived from video timestep blocks

**Important nuance vs your pseudocode**
- Your pseudocode says Beta(7,1) in Flash mode.
- Current default YAML (`wan_flow_matching_action_tf.yaml`) uses `video_noise_beta_alpha: 3.0` when enabled.
- So **mechanism matches**, but default parameter may differ from the algorithm text you pasted.

---

### Step 17: sample noise $z_0^k, a_0^k \sim \mathcal N(0,I)$

**Code**
- Video noise: `noise = torch.randn_like(latents)`
- Action noise: `noise_action = torch.randn_like(actions)` (if action exists)

---

### Step 18–20: interpolation (flow matching)

Your Eq.2 form:
$$
z_t = t z_1 + (1-t) z_0,\quad a_t = t a_1 + (1-t) a_0
$$

**Code form via scheduler**
- `FlowMatchScheduler.add_noise(original, noise, timestep)` computes:
  - `sample = (1 - sigma) * original + sigma * noise`

This is the same interpolation form with $t \leftrightarrow (1-\sigma)$ mapping.

---

### Step 21–22: predict velocity / noise with joint model

**Code**
- In `WANPolicyHead.forward()` model call:
  - `video_noise_pred, action_noise_pred = self.model(...)`
  - conditioned on prompt embeddings, image condition features, state, embodiment id, timesteps, noisy action, plus clean context (`clean_x`) in training.

---

### Step 23: target velocity $v^k := [z_1,a_1]-[z_0,a_0]$

**Code equivalent**
- `FlowMatchScheduler.training_target(sample, noise, timestep)` returns:
  - `target = noise - sample`

Given scheduler parameterization, model predicts this flow target representation (`noise - sample`) rather than explicitly forming `[x1 - x0]` tensor in main loop. Functionally this is the FM target under this scheduler’s variable choice.

---

### Step 24: loss $\mathcal L = ||v_{pred} - v||^2$

**Code**
- Dynamics loss:
  - MSE between `video_noise_pred` and `training_target`
  - then weighted by `scheduler.training_weight(timestep)`
- Action loss (if action present):
  - MSE between `action_noise_pred` and `training_target_action`
  - masked by `action_mask`
  - masked by `has_real_action`
  - weighted by `training_weight(timestep_action)`
- Final:
  - `loss = weighted_dynamics_loss + weighted_action_loss`

**Extra behavior**
- If no action exists, action loss is zero and only dynamics loss contributes.

---

### Step 25: update $\theta \leftarrow \theta - \eta \nabla\mathcal L$

**Code**
- `BaseTrainer.compute_loss()` returns `outputs["loss"]`
- HF `Trainer.training_step()` performs backward + optimizer step + scheduler step.
- Optimizer created in `BaseTrainer.create_optimizer()`.

---

### Step 26–27: end chunk loop / end while loop

**Code interpretation**
- Chunk-level semantics are vectorized by token/block organization and teacher-forcing attention masks.
- Outer while loop corresponds to trainer loop over steps/epochs until configured stop condition.

---

## 2) Symbol table: paper notation ↔ runtime tensors

- $\tau$ (trajectory): sampled trajectory data row group in `ShardedLeRobot...Dataset`.
- $c$ (text condition): tokenized prompt embeddings from `encode_prompt()`.
- $M$ (chunks): implied by sampled chunk count (`max_chunk_size`, 24-step chunkization, and block counts).
- $z_1^k$ (clean video latent): `latents` before noising.
- $a_1^k$ (clean action): `actions` from transformed batch.
- $z_0^k, a_0^k$ (noise): `noise`, `noise_action`.
- $t_k, t_{vid}, t_{act}$: `timestep_id`, `timestep`, `timestep_action_id`, `timestep_action`.
- $z_t^k, a_t^k$: `noisy_latents`, `noisy_actions`.
- $u_\theta(\cdot)$: `self.model(...)` (`CausalWanModel`).
- $v_{pred}$: `video_noise_pred`, `action_noise_pred`.
- $v$ target: `training_target`, `training_target_action`.

---

## 3) Critical implementation nuance: explicit chunk loop vs vectorized teacher forcing

Your pseudocode is naturally written as:
- for chunk `k`, build context from previous clean chunks, then predict.

DreamZero code does this mostly by:
- concatenating **clean context stream** + **noisy stream** during training (`clean_x` path),
- using specialized **blockwise causal attention** that enforces “previous clean context + current aligned action/state block” access pattern.

So if you were searching for a literal:
```python
for k in range(M):
    Ck = ...
```
you won’t find it at top-level training. It is encoded in token layout and attention rules inside `wan_video_dit_action_casual_chunk.py`.

---

## 4) Where each hyperparameter in Algorithm 1 lives

- Number of sampled chunks / chunk window:
  - data config `max_chunk_size`
  - dataset logic in `lerobot_sharded.py`
- Action horizon:
  - script arg `action_horizon=24` (common training scripts)
  - propagated into action head config
- Block granularity:
  - `num_frame_per_block`, `num_action_per_block`, `num_state_per_block`
- Noise schedule mode:
  - `decouple_video_action_noise`
  - `video_noise_beta_alpha/beta`
  - `use_high_noise_emphasis`, `high_noise_beta_alpha`

---

## 5) Suggested reading order (fastest path to understanding)

1. `scripts/train/droid_training_wan22.sh` (runtime overrides)
2. `groot/vla/experiment/experiment.py`
3. `groot/vla/experiment/base.py` (`create_*`, `compute_loss`)
4. `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py`
5. `groot/vla/data/dataset/lerobot_sharded.py` (chunk sampling)
6. `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py` (noise + loss)
7. `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py` (teacher-forcing causal attention core)
8. `groot/vla/model/dreamzero/modules/flow_match_scheduler.py`

---

## 6) If you want exact one-to-one “Step X → function call trace”

I can generate a second document with:
- exact call graph arrows,
- tensor shapes per stage for your specific config (e.g., Wan2.2, `num_frames=33`, `action_horizon=24`, `num_frame_per_block=2`),
- and a concrete mini worked example for one batch showing how many chunk blocks are formed and how `action_register_length` is computed.
