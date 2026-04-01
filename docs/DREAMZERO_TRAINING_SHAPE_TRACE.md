# DreamZero Training Shape Trace (One-Batch, End-to-End)

This companion doc focuses on **exact tensor shapes** through training so you can connect Algorithm 1 to runtime tensors.

It complements:
- `docs/DREAMZERO_TRAINING_FLOW_MATCHING_CODE_MAP.md`

---

## 1) What controls shape in practice

### 1.1 Data/chunk controls
- `max_chunk_size` from data config (`droid_relative*.yaml`)
- Video sampling in `lerobot_sharded.py` returns `8M+1` frames (where `M = max_chunk_size` unless overridden).
- Action sampling returns `24M` steps.
- State sampling returns `M` anchor states.

### 1.2 Model/block controls
- `num_frame_per_block`
- `num_action_per_block`
- `num_state_per_block`
- `frame_seqlen` (tokens per latent frame expected by DiT)

### 1.3 Common script overrides (important)
Training scripts often override defaults to:
- `max_chunk_size=4`
- `num_frames=33`
- `action_horizon=24`
- `num_frame_per_block=2`
- `num_action_per_block=24`
- `num_state_per_block=1`

That gives a very clean alignment:
- $M=4$
- video frames $=8M+1=33$
- action steps $=24M=96$
- state tokens $=M=4$

---

## 2) Shape formulas (quick reference)

Let:
- $B$: batch size
- $M$: chunk count (effective video chunks for sample)
- $T_v = 8M + 1$: sampled raw video frames
- $T_a = 24M$: sampled action horizon in training batch
- $T_s = M$: sampled state anchors
- $C_{img}=3$

After transform/collate (`dreamzero_cotrain.py`):
- `images`: `[B, T_v, H, W, C_img]`
- `action`: `[B, T_a, max_action_dim]`
- `state`: `[B, T_s, max_state_dim]`
- `action_mask`: same as `action`

Latent temporal compression in Wan path is effectively $F = 1 + (T_v-1)/4$.

Latent token sequence length in action head:
$$
\text{tokens\_per\_frame} = (H_{lat}//2) \cdot (W_{lat}//2)
$$
$$
\text{seq\_len} = F \cdot \text{tokens\_per\_frame}
$$

Action/state register length:
$$
L_{reg} = T_a + T_s
$$

Block consistency during teacher forcing requires:
$$
\text{num\_image\_blocks} = (F-1) / \text{num\_frame\_per\_block}
$$
$$
T_a = \text{num\_image\_blocks} \cdot \text{num\_action\_per\_block}
$$
$$
T_s = \text{num\_image\_blocks} \cdot \text{num\_state\_per\_block}
$$

---

## 3) Worked example A (the common 33-frame training setup)

Use the common script-style setup:
- `B=1`
- `max_chunk_size=4` → $M=4$
- `T_v=33`, `T_a=96`, `T_s=4`
- `num_frame_per_block=2`, `num_action_per_block=24`, `num_state_per_block=1`

### 3.1 Before model forward

From transform + collate:
- `images`: `[1, 33, H, W, 3]`
- `action`: `[1, 96, 32]` (assuming `max_action_dim=32`)
- `state`: `[1, 4, 64]` (assuming `max_state_dim=64`)
- `action_mask`: `[1, 96, 32]`

### 3.2 In `WANPolicyHead.forward()`

1) Video reorder/normalize:
- `images -> videos`: `[1, 3, 33, H, W]`

2) VAE encode:
- `latents` (before transpose): `[1, C_lat, F, H_lat, W_lat]`
- After transpose used by scheduler path:
  - `latents`: `[1, F, C_lat, H_lat, W_lat]`
  - `noise`: same shape

3) Time ids:
- `timestep_id`: `[1, F]`
- Block-adjusted `timestep_id` remains `[1, F]`
- Coupled mode `timestep_action_id`: `[1, 96]`
- Decoupled mode `timestep_action_id`: independent random `[1, 96]`

4) Noised tensors:
- `noisy_latents`: `[1, F, C_lat, H_lat, W_lat]`
- `training_target` (after transpose): `[1, C_lat, F, H_lat, W_lat]`
- `noisy_actions`: `[1, 96, 32]`
- `training_target_action`: `[1, 96, 32]`

### 3.3 In `CausalWanModel._forward_train()`

1) Patch embedding output:
- `x` flattened tokens: `[1, seq_len, dim]`

2) Add action/state register:
- action features length = `96`
- state features length = `4`
- register length `L_reg = 100`
- `x` becomes `[1, seq_len + 100, dim]`

3) Teacher-forcing concat (`clean_x` is passed during training):
- clean stream adds another `seq_len`
- total token length into blocks:
  - `[1, 2*seq_len + 100, dim]`

4) Output split:
- action part extracted from noisy stream: `[1, 96, dim]` then decoded to `[1, 96, 32]`
- video part unpatchified to `[1, C_lat, F, H_lat, W_lat]`

### 3.4 Loss shapes

In action head:
- `video_noise_pred`: `[1, C_lat, F, H_lat, W_lat]`
- `training_target`: same (possibly cropped in spatial dims if needed)
- per-sample dynamics loss after reduction: `[1, F]`

- `action_noise_pred`: `[1, 96, 32]`
- `training_target_action`: `[1, 96, 32]`
- masked/reduced action loss before final mean: `[1, 96]`

Final scalars:
- `weighted_dynamics_loss`: `[]`
- `weighted_action_loss`: `[]`
- `loss = weighted_dynamics_loss + weighted_action_loss`

---

## 4) Worked example B (Wan2.2 / `droid_relative_wan22`)

With:
- target resize: `160x320`
- VAE38 latent downscale gives approximately `H_lat=10`, `W_lat=20`
- DiT patch stride `(1,2,2)` gives tokens/frame:
  - $(10//2)\cdot(20//2)=50$
- config sets `frame_seqlen=50` to match.

If `T_v=33`, then latent frames `F=9` and:
$$
\text{seq\_len} = 9 \cdot 50 = 450
$$

With `num_frame_per_block=2`:
- image blocks = $(9-1)/2 = 4$
- expected action length = $4\cdot24=96$
- expected state length = $4\cdot1=4$

This exactly matches the register layout and avoids block mismatch errors.

---

## 5) Why the explicit chunk loop is not obvious in Python

Algorithm 1 writes:
- loop chunks `k=1..M`, construct clean context $\mathcal C_k$.

Code does it as **token layout + attention policy**:
- `clean_x` adds clean context stream.
- blockwise causal teacher-forcing logic in
  - `CausalWanSelfAttention.forward(..., is_tf=True)`
  - helper functions `_process_clean_image_only`, `_process_noisy_image_blocks`, `_process_noisy_action_blocks`, `_process_state_blocks`.

So chunk semantics are implemented inside attention blocks, not as a top-level `for k` in trainer code.

---

## 6) Practical sanity checks for shape mismatches

If training crashes with sequence/register mismatch, verify in this order:

1. `max_chunk_size`, `num_frames`, and actual dataset sample lengths agree (`8M+1`, `24M`, `M`).
2. `num_frame_per_block`, `num_action_per_block`, `num_state_per_block` satisfy:
   - action length = image blocks × actions/block
   - state length = image blocks × states/block
3. `frame_seqlen` matches latent tokens per frame for your backbone+resolution.
4. For Wan2.2, use `160x320` (or another resolution yielding correct even latent/token geometry) if you want no spatial cropping in dynamics loss.

---

## 7) Fast call path with shape checkpoints

1. `experiment.py` → `BaseExperiment.train()`
2. `BaseTrainer.compute_loss()`
3. `VLA.forward()`
4. `WANPolicyHead.forward()`
   - input prep, VAE encode, scheduler noising, target build
5. `CausalWanModel._forward_train()`
   - patch, register concat, teacher forcing, decode
6. back to action head
   - dynamics/action losses, weighting, scalar total loss

---

## 8) Notes on default config vs script overrides

Base data config (`base_48_wan_fine_aug_relative.yaml`) has:
- `num_frames: 49`, `action_horizon: 48`

But common training scripts override to:
- `num_frames: 33`, `action_horizon: 24`, `max_chunk_size: 4`

So when tracing real runs, always trust the **effective merged Hydra config** from the run (`experiment_cfg/conf.yaml`) over base defaults.
