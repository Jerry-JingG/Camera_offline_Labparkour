# Dagger Explicit Yaw Prediction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add explicit yaw prediction to the dagger training method's depth encoder, matching the original train.py approach.

**Architecture:** Modify DepthEncoder to output both visual tokens and yaw predictions (2D). Modify Student's forward_with_mems to: (1) extract yaw from depth_encoder output, (2) scale by 1.5, (3) replace delta_yaw in proprio before encoding, (4) use replaced proprio for inference. This happens **inside the model's forward pass**, creating a closed-loop where the model sees its own yaw predictions during inference. Add yaw loss to training objective with equal weight to action loss.

**Tech Stack:** PyTorch, TransformerXL, Isaac Lab

---

## Background

Currently, the dagger method (train_student_dagger.py) uses a DepthEncoder that only extracts visual features. The original method (train.py with DistillationWithExtractor) uses a depth encoder that explicitly predicts yaw. This plan adds explicit yaw prediction to the dagger method.

**Key differences to implement:**
1. DepthEncoder outputs `[depth_tokens, yaw]` where yaw is 2D
2. Training loop uses predicted yaw (scaled by 1.5) to replace delta_yaw in observations
3. delta_yaw_ok masking: only apply predicted yaw in certain environments
4. Yaw loss: supervise yaw prediction with true delta_yaw values
5. **Zero out delta_yaw before depth encoder**: Clear delta_yaw (indices 6:8) in proprio before passing to depth encoder, forcing the model to predict yaw from visual features (matching train.py line 360)
6. **Optional yaw update frequency**: Original implementation updates yaw every 5 steps to reduce computation (train.py line 358). This is optional for dagger method.

---

## Task 1: Add Yaw Prediction Head to DepthEncoder

**Files:**
- Modify: `parkour_tasks/parkour_tasks/extreme_parkour_task/modules/tokenizers/depth_encoder.py`
- Test: Manual verification in training loop

**Step 1: Add yaw prediction head to DepthEncoder.__init__**

Add after line 43 (after self.drop):

```python
# Yaw prediction head: predicts 2D yaw from visual features
# Input: flattened visual features [B, grid_size*grid_size*token_dim]
# Output: [B, 2] yaw prediction
self.yaw_head = nn.Sequential(
    nn.Linear(grid_size * grid_size * token_dim, 256),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(256, 128),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(128, 2),  # 2D yaw output
)
```

**Step 2: Add proprio input parameter to DepthEncoder.__init__**

Add after line 32 (after self.add_2d_pos_embed):

```python
# Store num_prop for proprio input (matching RecurrentDepthBackbone design)
self.num_prop = num_prop if num_prop is not None else 0
```

Update the `__init__` signature to accept `num_prop`:

```python
def __init__(
    self,
    in_frames: int = 4,
    in_size: int = 64,
    token_dim: int = 128,
    grid_size: int = 4,
    add_2d_pos_embed: bool = True,
    dropout: float = 0.0,
    num_prop: int = 0,  # Number of proprioception features
) -> None:
```

**Step 3: Modify DepthEncoder.forward to accept proprio and output yaw**

Replace the forward method (lines 88-114) with:

```python
def forward(self, x: Tensor, proprio: Optional[Tensor] = None) -> Tensor:
    """
    Args:
        x: Tensor of shape [B, in_frames, H, W].
        proprio: Optional Tensor of shape [B, num_prop].
                 IMPORTANT: delta_yaw (indices 6:8) should be zeroed before calling this method.

    Returns:
        Tensor of shape [B, grid_size * grid_size * token_dim + 2].
        Last 2 dimensions are yaw predictions.
    """
    self._check_input(x)
    x = self._resize_if_needed(x)
    h = self.act(self.conv1(x))
    h = self.act(self.conv2(h))
    h = self.act(self.conv3(h))
    h = self.conv4(h)
    if self.pool is not None:
        h = self.pool(h)
    elif h.shape[-1] != self.grid_size or h.shape[-2] != self.grid_size:
        h = nn.functional.adaptive_avg_pool2d(h, (self.grid_size, self.grid_size))

    if self.pos_embed is not None:
        h = h + self.pos_embed.unsqueeze(0)

    tokens = h.flatten(2).transpose(1, 2).contiguous()  # [B, num_tokens, token_dim]
    tokens = self.drop(tokens)

    # Flatten visual features for yaw prediction
    visual_features_flat = tokens.flatten(1)  # [B, num_tokens * token_dim]

    # Concatenate with proprio if provided (matching RecurrentDepthBackbone design)
    if proprio is not None and self.num_prop > 0:
        yaw_input = torch.cat([visual_features_flat, proprio], dim=1)
    else:
        yaw_input = visual_features_flat

    # Predict yaw from combined features
    yaw_pred = self.yaw_head(yaw_input)  # [B, 2]

    # Concatenate tokens and yaw: [B, num_tokens, token_dim] -> [B, num_tokens * token_dim + 2]
    tokens_flat = tokens.flatten(1)  # [B, num_tokens * token_dim]
    output = torch.cat([tokens_flat, yaw_pred], dim=1)  # [B, num_tokens * token_dim + 2]

    if not torch.isfinite(output).all():
        raise ValueError("Non-finite values detected in encoder output.")
    return output
```

**Note:** If using proprio input, update the yaw_head input dimension in Step 1:
```python
# If num_prop > 0, yaw_head input should be: grid_size * grid_size * token_dim + num_prop
input_dim = grid_size * grid_size * token_dim + (num_prop if num_prop > 0 else 0)
self.yaw_head = nn.Sequential(
    nn.Linear(input_dim, 256),
    # ... rest of the layers
)
```

**Step 3: Commit**

```bash
git add parkour_tasks/parkour_tasks/extreme_parkour_task/modules/tokenizers/depth_encoder.py
git commit -m "feat: add yaw prediction head to DepthEncoder"
```

---
## Task 2: Update MultiModalStudentPolicy to Use Yaw Predictions Internally

**Files:**
- Modify: `scripts/rsl_rl/train_student_from_dataset.py:266-410`

**Goal:** Modify forward_with_mems to predict yaw from depth_encoder, then replace delta_yaw in proprio before encoding. This matches the original implementation where the model sees its own yaw predictions.

**Step 1: Add helper method to get proprio dimensions**

Add after the `__init__` method (around line 300):

```python
def _get_last_frame_indices(self, prop_hist_len: int, num_prop: int) -> Tuple[int, int]:
    """Get the indices for delta_yaw in the last frame of proprio history.

    Args:
        prop_hist_len: Number of frames in proprio history
        num_prop: Number of proprio features per frame

    Returns:
        start_idx, end_idx for delta_yaw in flattened proprio
    """
    last_frame_start = (prop_hist_len - 1) * num_prop
    return last_frame_start + 6, last_frame_start + 8
```

**Step 2: Modify forward_with_mems to replace yaw internally**

Replace the forward_with_mems method (lines 331-373) with:

```python
def forward_with_mems(
    self,
    proprio_seq: Tensor,
    depth_seq: Tensor,
    mems: Optional[List[Tensor]] = None,
    delta_yaw_ok: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, List[Tensor]]:
    """Forward pass with yaw prediction and replacement.

    Args:
        proprio_seq: Tensor[B, S, prop_hist_len * proprio_dim]
        depth_seq: Tensor[B, S, depth_hist_len, H, W]
        mems: Optional list of memory tensors from previous segment
        delta_yaw_ok: Optional[B, S] bool tensor indicating which envs should use predicted yaw

    Returns:
        actions: Predicted action means of shape [B, S, action_dim]
        yaw_pred_seq: Predicted yaw of shape [B, S, 2]
        new_mems: List of new memory tensors for next segment
    """
    batch_size, seq_len, feat_dim = proprio_seq.shape

    # 1. Prepare proprio for depth encoder (zero out delta_yaw to force visual prediction)
    # Extract last frame proprio from the flattened sequence
    proprio_flat = proprio_seq.reshape(batch_size * seq_len, feat_dim)  # [B*S, feat_dim]

    # Get the last frame's proprio (assuming prop_hist_len frames are stacked)
    # For simplicity, extract first num_prop features (or implement proper extraction)
    # IMPORTANT: Zero out delta_yaw (indices 6:8) before passing to depth_encoder
    proprio_for_encoder = proprio_flat[:, :self.num_prop].clone()  # [B*S, num_prop]
    proprio_for_encoder[:, 6:8] = 0  # Zero out delta_yaw (matching train.py line 360)

    # 2. Encode depth and extract yaw predictions
    depth_output = self.depth_encoder(
        depth_seq.reshape(batch_size * seq_len, depth_seq.size(2), depth_seq.size(3), depth_seq.size(4)),
        proprio_for_encoder  # Pass zeroed proprio
    )  # [B*S, num_tokens * token_dim + 2]

    # Split into depth tokens and yaw
    depth_tokens_flat = depth_output[:, :-2]  # [B*S, num_tokens * token_dim]
    yaw_pred = depth_output[:, -2:]  # [B*S, 2]

    # Scale yaw predictions by 1.5 (matching original implementation)
    yaw_scaled = 1.5 * yaw_pred  # [B*S, 2]

    # Reshape depth tokens back to [B*S, num_tokens, token_dim]
    num_tokens = self.depth_encoder.grid_size * self.depth_encoder.grid_size
    depth_encoded = depth_tokens_flat.reshape(batch_size * seq_len, num_tokens, self.token_dim)

    # 3. Replace delta_yaw in proprio with predicted yaw
    proprio_modified = proprio_flat.clone()

    # Get indices for delta_yaw in the last frame
    start_idx, end_idx = self._get_last_frame_indices(self.prop_hist_len, self.num_prop)

    # Replace delta_yaw with scaled yaw predictions
    if delta_yaw_ok is not None:
        # Only replace where delta_yaw_ok is True
        delta_yaw_ok_flat = delta_yaw_ok.reshape(batch_size * seq_len)  # [B*S]
        proprio_modified[delta_yaw_ok_flat, start_idx:end_idx] = yaw_scaled[delta_yaw_ok_flat]
    else:
        # Replace for all environments (default behavior)
        proprio_modified[:, start_idx:end_idx] = yaw_scaled

    # 4. Encode proprio with replaced yaw
    prop_encoded = self.proprio_encoder(proprio_modified)  # [B*S, 1, C]

    # 5. Multi-modal fusion
    fused = self.fusion_transformer(prop_encoded, depth_encoded)
    fused_seq = fused["all_pooled"].reshape(batch_size, seq_len, -1)

    # 6. Temporal modeling with memory
    temporal_out, new_mems = self.temporal_model(
        fused_seq,
        mems=mems,
        causal_mask=True,
        return_mems=True,
    )

    # 7. Action head
    actions = self.action_head.forward_sequence(temporal_out)["mean"]

    # Reshape yaw predictions back to sequence format
    yaw_pred_seq = yaw_pred.reshape(batch_size, seq_len, 2)

    return actions, yaw_pred_seq, new_mems
```

**Step 3: Update forward_step to pass through delta_yaw_ok**

Modify forward_step method (lines 375-410):

```python
def forward_step(
    self,
    proprio_step: Tensor,
    depth_step: Tensor,
    mems: Optional[List[Optional[Tensor]]] = None,
    delta_yaw_ok: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Optional[List[Tensor]]]:
    """Single-step forward for online inference / DAGGER.

    Args:
        proprio_step: Tensor[B, prop_hist_len * proprio_dim]
        depth_step: Tensor[B, depth_hist_len, H, W]
        mems: Transformer-XL memory states
        delta_yaw_ok: Optional[B] bool tensor indicating which envs should use predicted yaw

    Returns:
        actions_step: Tensor[B, action_dim]
        yaw_pred_step: Tensor[B, 2]
        new_mems: Updated memory states
    """
    if proprio_step.dim() != 2:
        raise ValueError("proprio_step must have shape [B, F].")
    if depth_step.dim() != 4:
        raise ValueError("depth_step must have shape [B, T, H, W].")

    # Add sequence dimension S=1
    proprio_seq = proprio_step.unsqueeze(1)  # [B, 1, feat_dim]
    depth_seq = depth_step.unsqueeze(1)      # [B, 1, depth_hist_len, H, W]

    # Add sequence dimension to delta_yaw_ok if provided
    delta_yaw_ok_seq = delta_yaw_ok.unsqueeze(1) if delta_yaw_ok is not None else None

    # Use forward_with_mems
    actions_seq, yaw_pred_seq, new_mems = self.forward_with_mems(
        proprio_seq, depth_seq, mems=mems, delta_yaw_ok=delta_yaw_ok_seq
    )

    # Remove sequence dimension
    actions_step = actions_seq.squeeze(1)  # [B, action_dim]
    yaw_pred_step = yaw_pred_seq.squeeze(1) if yaw_pred_seq is not None else None  # [B, 2]

    return actions_step, yaw_pred_step, new_mems
```

**Step 4: Add num_prop and prop_hist_len to __init__**

In the `__init__` method, add these attributes (around line 280):

```python
self.num_prop = num_prop  # Number of proprio features per frame
self.prop_hist_len = prop_hist_len  # Number of frames in proprio history
```

**Step 5: Commit**

```bash
git add scripts/rsl_rl/train_student_from_dataset.py
git commit -m "feat: implement internal yaw prediction and replacement in Student forward"
```

---

## Task 3: Update Dagger Training Loop to Pass delta_yaw_ok and Collect Yaw Loss

**Files:**
- Modify: `scripts/rsl_rl/train_student_dagger.py`

**Goal:** Update the training loop to pass delta_yaw_ok to Student and collect yaw errors for loss calculation. The yaw replacement now happens inside the Student model.

**Step 1: Add yaw buffer initialization and step counter**

After line 311 (inside the while loop):

```python
yaws_buffer = []  # Store yaw prediction errors for loss calculation
yaw_update_freq = 5  # Update yaw prediction every N steps (matching train.py line 358)
step_counter = 0  # Track steps for yaw update frequency
cached_yaw_pred = None  # Cache yaw predictions between updates
```

**Step 2: Extract yaw predictions with update frequency control**

Modify line 371 and add delta_yaw_ok with frequency control:

```python
# OLD:
actions_step, _, new_mems = student.forward_step(prop_step, depth_step, mems=txl_mems)

# NEW:
# Define delta_yaw_ok masking (start with all True)
delta_yaw_ok = torch.ones(args.num_envs, dtype=torch.bool, device=device)

# Update yaw prediction every N steps (matching train.py design)
if step_counter % yaw_update_freq == 0:
    # Forward pass with delta_yaw_ok to get fresh yaw predictions
    actions_step, yaw_pred_step, new_mems = student.forward_step(
        prop_step, depth_step, mems=txl_mems, delta_yaw_ok=delta_yaw_ok
    )
    cached_yaw_pred = yaw_pred_step  # Cache for next steps
else:
    # Reuse cached yaw predictions
    actions_step, _, new_mems = student.forward_step(
        prop_step, depth_step, mems=txl_mems, delta_yaw_ok=delta_yaw_ok
    )
    yaw_pred_step = cached_yaw_pred

step_counter += 1
```

**Alternative (simpler):** If you want to update yaw every step (no frequency control), use:

```python
# Define delta_yaw_ok masking (start with all True)
delta_yaw_ok = torch.ones(args.num_envs, dtype=torch.bool, device=device)

# Forward pass with delta_yaw_ok
actions_step, yaw_pred_step, new_mems = student.forward_step(
    prop_step, depth_step, mems=txl_mems, delta_yaw_ok=delta_yaw_ok
)
```

**Step 3: Calculate yaw error for loss**

Add after the forward pass:

```python
# Scale yaw predictions by 1.5 (already done inside model, but we need unscaled for loss)
# Note: yaw_pred_step is already scaled inside the model
# We need to get the true yaw from observations
true_yaw = torch.from_numpy(obs_prop_np[:, 6:8]).float().to(device)  # [B, 2]

# Calculate yaw error (true - predicted*1.5)
# Since model already scales by 1.5 internally, yaw_pred_step is the scaled version
yaw_error = true_yaw - yaw_pred_step
yaws_buffer.append(yaw_error.detach())
```

**Note:** The yaw replacement in observations now happens **inside the Student model's forward pass**, so we don't need to manually replace obs_prop_np here. The model sees its own predictions automatically.

**Step 4: Commit**

```bash
git add scripts/rsl_rl/train_student_dagger.py
git commit -m "feat: pass delta_yaw_ok to Student and collect yaw errors"
```

---

## Task 4: Add Yaw Loss to Training Objective

**Files:**
- Modify: `scripts/rsl_rl/train_student_dagger.py`

**Step 1: Extract yaw from training forward pass**

Modify line 457:

```python
# OLD:
pred, yaw_pred, new_train_mems = student.forward_with_mems(proprio_t, depth_t, mems=train_mems)

# NEW:
pred, yaw_pred_train, new_train_mems = student.forward_with_mems(proprio_t, depth_t, mems=train_mems)
```

**Step 2: Calculate yaw loss**

Add after line 476:

```python
loss_actions = nn.functional.mse_loss(pred, teacher_t)

# Calculate yaw loss
# Extract delta_yaw from batch proprio
# batch["proprio"] shape: [B, S, prop_hist_len * proprio_dim]
batch_size, seq_len, prop_feat_dim = batch["proprio"].shape
proprio_reshaped = batch["proprio"].reshape(batch_size, seq_len, args.prop_hist_len, num_prop)
proprio_last_frame = proprio_reshaped[:, :, -1, :]  # [B, S, num_prop]
true_yaw_train = proprio_last_frame[:, :, 6:8]  # [B, S, 2]

# Note: yaw_pred_train is already scaled by 1.5 inside the model
# So we compare directly with true_yaw
loss_yaw = nn.functional.mse_loss(yaw_pred_train, true_yaw_train)

# Total loss
loss = loss_actions + loss_yaw
```

**Step 3: Add yaw loss logging**

Modify console logging (around line 505):

```python
print(
    f"[iter {it}] loss={loss.item():.5f} (action={loss_actions.item():.5f}, yaw={loss_yaw.item():.5f})   "
    f"time={iter_time:.2f}s   eta={eta_str}   steps/s={steps_per_sec:.2f}   global_step={global_step}"
)
```

Add to wandb logging (around line 512):

```python
wandb_metrics = {
    "train/loss": loss.item(),
    "train/loss_actions": loss_actions.item(),
    "train/loss_yaw": loss_yaw.item(),
    # ... rest
}
```

**Step 4: Commit**

```bash
git add scripts/rsl_rl/train_student_dagger.py
git commit -m "feat: add yaw loss to training objective"
```

---

## Task 5: Testing and Validation

**Step 1: Run training test**

```bash
cd /home/jing/IsaacLab/Camera_offline_Labparkour
python scripts/rsl_rl/train_student_dagger.py \
    --task Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2-Play-v0 \
    --teacher_checkpoint logs/rsl_rl/<teacher_path>/model_*.pt \
    --num_iters 10 \
    --num_envs 4 \
    --sequence_length 32 \
    --device cuda:0 \
    --headless
```

**Step 2: Verify outputs**

Check:
- Training runs without errors
- Loss values printed correctly
- Yaw loss is reasonable (not NaN)
- Model saves successfully

**Step 3: Add debug logging**

Add temporary debug prints to verify yaw predictions:

```python
if it == 0:
    print(f"[DEBUG] yaw_pred range: [{yaw_pred_step.min():.3f}, {yaw_pred_step.max():.3f}]")
    print(f"[DEBUG] true_yaw range: [{true_yaw.min():.3f}, {true_yaw.max():.3f}]")
```

**Step 4: Commit**

```bash
git add scripts/rsl_rl/train_student_dagger.py
git commit -m "test: verify yaw prediction implementation"
```

---

## Summary

**Implementation complete! The dagger method now:**
1. ✅ DepthEncoder outputs visual tokens + yaw predictions (2D), optionally accepting proprio input
2. ✅ **Delta_yaw zeroing**: Clears delta_yaw (indices 6:8) in proprio before passing to depth encoder, forcing visual-based yaw prediction (matching train.py line 360)
3. ✅ Student's forward_with_mems internally: predicts yaw → scales by 1.5 → replaces delta_yaw in proprio → encodes
4. ✅ delta_yaw_ok masking controls which environments use predicted yaw (currently all True)
5. ✅ Yaw loss supervises predictions with true delta_yaw values (equal weight to action loss)
6. ✅ Model sees its own yaw predictions **within the same forward pass**, matching original implementation
7. ✅ **Optional**: Yaw update frequency control (every N steps) to reduce computation cost (matching train.py line 358)

**Testing checklist:**
- [ ] Training runs without errors
- [ ] Yaw loss decreases over iterations
- [ ] Action loss remains stable
- [ ] Yaw predictions in reasonable range
- [ ] Model checkpoints save correctly

**Next steps:**
- Implement delta_yaw_ok curriculum learning
- Tune yaw loss weight if needed
- Compare performance with/without yaw prediction
- Update play_student.py to use yaw predictions

**Known issues to watch:**
- Dimension mismatches in yaw extraction
- NaN values in yaw predictions
- Memory issues with large batches
