# DreamZero Training: Exact Call Graph Arrows (Algorithm-Matched)

This doc gives **exact caller → callee arrows** (with `file:line`) and maps them to your Algorithm 1 steps.

Companion docs:
- `docs/DREAMZERO_TRAINING_FLOW_MATCHING_CODE_MAP.md`
- `docs/DREAMZERO_TRAINING_SHAPE_TRACE.md`

---

## 1) End-to-end training call graph (top level)

```text
scripts/train/*.sh
  -> torchrun groot/vla/experiment/experiment.py

groot/vla/experiment/experiment.py:126  main(cfg)
  -> groot/vla/experiment/experiment.py:130  VLAExperiment(cfg)
     -> groot/vla/experiment/base.py:585  BaseExperiment.__init__
        -> groot/vla/experiment/base.py:700  create_model
        -> groot/vla/experiment/base.py:742  create_train_dataset
        -> groot/vla/experiment/base.py:750  create_data_collator
        -> groot/vla/experiment/base.py:753  create_trainer
  -> groot/vla/experiment/experiment.py:131  experiment.train()
     -> groot/vla/experiment/base.py:868  BaseExperiment.train
        -> groot/vla/experiment/base.py:870  self.trainer.train(...)
```

---

## 2) Per-step train call graph (forward/loss/update path)

```text
HF Trainer loop
  -> BaseTrainer.training_step(...)                [base.py:375, super call at 389]
     -> BaseTrainer.compute_loss(...)              [base.py:408]
        -> outputs = model(inputs)                 [base.py:410]

model == VLA
  -> VLA.forward(inputs)                           [base_vla.py:139]
     -> VLA.prepare_input(inputs)                  [base_vla.py:242]
        -> backbone.prepare_input(...)             [base_vla.py:244]
        -> action_head.prepare_input(...)          [base_vla.py:245]
     -> backbone(backbone_inputs)                  [base_vla.py:145]
     -> action_head(backbone_outputs, action_inputs)
                                                 [base_vla.py:146]

action_head == WANPolicyHead
  -> WANPolicyHead.forward(...)                    [wan_flow_matching_action_tf.py:603]
     -> encode_prompt(...)                         [wan_flow_matching_action_tf.py:637 -> 542]
     -> encode_video(...)                          [wan_flow_matching_action_tf.py:659 -> 557]
     -> scheduler.add_noise(...)                   [wan_flow_matching_action_tf.py:744]
     -> scheduler.training_target(...)             [wan_flow_matching_action_tf.py:745]
     -> (if action) scheduler.add_noise(...)       [wan_flow_matching_action_tf.py:749]
     -> (if action) scheduler.training_target(...) [wan_flow_matching_action_tf.py:754]
     -> self.model(...)                            [wan_flow_matching_action_tf.py:763/770]

self.model == CausalWanModel
  -> CausalWanModel.forward(*args, **kwargs)       [wan_video_dit_action_casual_chunk.py:2164]
     -> (no kv_cache) _forward_train(...)          [wan_video_dit_action_casual_chunk.py:2172]
        -> patch_embedding(x)                      [wan_video_dit_action_casual_chunk.py:2042]
        -> action_encoder(...)                     [wan_video_dit_action_casual_chunk.py:2059]
        -> state_encoder(...)                      [wan_video_dit_action_casual_chunk.py:2061]
        -> if clean_x: patch_embedding(clean_x)    [wan_video_dit_action_casual_chunk.py:2097-2100]
        -> for block in self.blocks                [wan_video_dit_action_casual_chunk.py:2135]
        -> head(...)                               [wan_video_dit_action_casual_chunk.py:2159]
        -> unpatchify(...)                         [wan_video_dit_action_casual_chunk.py:2160]
        -> return video_noise_pred, action_noise_pred
                                                 [wan_video_dit_action_casual_chunk.py:2162]

back in WANPolicyHead.forward
  -> mse_loss(video_noise_pred, training_target)   [wan_flow_matching_action_tf.py:784]
  -> scheduler.training_weight(timestep)           [wan_flow_matching_action_tf.py:788]
  -> (if action) mse_loss(action_noise_pred, target_action)
                                                 [wan_flow_matching_action_tf.py:792]
  -> (if action) scheduler.training_weight(...)    [wan_flow_matching_action_tf.py:796]
  -> loss = weighted_dynamics_loss + weighted_action_loss
                                                 [wan_flow_matching_action_tf.py:800]
  -> return BatchFeature({loss, dynamics_loss, action_loss})
                                                 [wan_flow_matching_action_tf.py:814]

HF Trainer (outside repo file)
  -> backward(loss) -> optimizer.step() -> lr_scheduler.step()
```

---

## 3) Data path call graph (trajectory → transformed batch)

```text
ShardedLeRobotMixtureDataset iterator
  -> dataset.get_step_data(trajectory_id, indices) [lerobot_sharded.py iterator around 1508-1521]
     -> get_video(...)                              [lerobot_sharded.py:583]
        -> _uniform_sample_from_language_ranges(...) [lerobot_sharded.py:1131, called at 633]
     -> get_state(...)                              [lerobot_sharded.py:690]
     -> get_action(...)                             [lerobot_sharded.py:858]
        -> _convert_to_relative_action(...)         [lerobot_sharded.py:1046 -> 1057]
  -> dataset.transforms(step_data)                  [lerobot_sharded.py:1524]

transforms includes DreamTransform
  -> DreamTransform.__call__                        [dreamzero_cotrain.py:628]
     -> apply(...)                                  [dreamzero_cotrain.py:615]
        -> apply_batch(...) / apply_single(...)     [dreamzero_cotrain.py:604 / 504]
           -> _prepare_video(...)                   [dreamzero_cotrain.py:309]
           -> _prepare_language(...)                [dreamzero_cotrain.py:383]
           -> _prepare_state(...)                   [dreamzero_cotrain.py:442]
           -> _prepare_action(...)                  [dreamzero_cotrain.py:475]

DataCollator
  -> DefaultDataCollator.__call__                   [dreamzero_cotrain.py:175]
     -> collate(...)                                [dreamzero_cotrain.py:92]
```

---

## 4) Scheduler call graph (flow matching math)

```text
WANPolicyHead.__init__
  -> FlowMatchScheduler(...)
  -> scheduler.set_timesteps(..., training=True)    [wan_flow_matching_action_tf.py:230]

WANPolicyHead.forward
  -> scheduler.add_noise(...)                       [flow_match_scheduler.py:73]
  -> scheduler.training_target(...)                 [flow_match_scheduler.py:83]
  -> scheduler.training_weight(...)                 [flow_match_scheduler.py:88]
```

---

## 5) Algorithm 1 step → exact call-chain arrows

### Steps 4–5: while loop + sample trajectory

```text
BaseExperiment.train [base.py:868]
  -> trainer.train [base.py:870]
    -> dataloader iterator
      -> dataset.get_step_data -> get_video/get_state/get_action
```

### Steps 6–7: clean latents/actions + chunked trajectory structure

```text
DreamTransform.apply_single [dreamzero_cotrain.py:504]
  -> _prepare_video / _prepare_state / _prepare_action
  -> collate
  -> WANPolicyHead.forward [wan_flow_matching_action_tf.py:603]
    -> encode_video (clean latents)
```

### Steps 8–9: clean context $\mathcal{C}_k$ (teacher forcing history)

```text
WANPolicyHead.forward
  -> self.model(..., clean_x=latents.transpose(1,2)) [wan_flow_matching_action_tf.py:763/770]
    -> CausalWanModel._forward_train
      -> if clean_x is not None: concat clean stream [wan_video_dit_action_casual_chunk.py:2097-2104]
      -> blockwise/teacher-forcing attention in blocks loop [line 2135]
```

### Steps 10–20: sample timesteps/noise, interpolate

```text
WANPolicyHead.forward
  -> sample timestep ids (uniform or decoupled mode)
  -> scheduler.add_noise(...) for video/action [line 744, 749]
  -> scheduler.training_target(...)             [line 745, 754]
```

### Steps 21–24: predict velocity/noise + MSE loss

```text
WANPolicyHead.forward
  -> self.model(...) -> (video_noise_pred, action_noise_pred)
  -> mse_loss(video_noise_pred, training_target) [line 784]
  -> mse_loss(action_noise_pred, training_target_action) [line 792]
  -> apply scheduler.training_weight(...) [line 788, 796]
  -> loss scalar [line 800]
```

### Steps 25–27: optimize and continue

```text
BaseTrainer.compute_loss [base.py:408]
  -> returns outputs["loss"]
HF Trainer
  -> backward -> optimizer.step -> next step
```

---

## 6) Why you don’t see `for k in range(M)` at top-level

Your algorithm is chunk-indexed in pseudocode, but runtime chunking is split between:
- dataset chunk sampling (`lerobot_sharded.py`), and
- token/block teacher-forcing causal attention inside `CausalWanModel` blocks.

So the equivalent of `k`-wise context is implemented by **data chunk formation + block-structured attention**, not a visible high-level Python chunk loop in trainer code.
