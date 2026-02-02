# Architectural Design Document: RL Fine-tuning of DAgger-Trained Student Policy

**Document Version:** 1.0
**Date:** 2026-02-02
**Author:** Architecture Review
**Status:** Draft

---

## 1. Executive Summary

This document provides a comprehensive architectural design for fine-tuning a DAgger-trained student policy using PPO reinforcement learning with domain randomization. The goal is to reduce the sim-to-real gap by making the student policy more robust to real-world conditions including camera noise, latency, lighting variations, and sensor delays.

### Key Design Principles

1. **Minimal Invasive Changes**: Leverage existing infrastructure where possible
2. **Memory Safety**: Careful Transformer-XL memory management during RL rollouts
3. **Modularity**: Clean separation between actor-critic wrapper, PPO algorithm, and domain randomization
4. **Backward Compatibility**: Maintain ability to load and use DAgger checkpoints

---

## 2. System Architecture Overview

### 2.1 High-Level Component Diagram

```
+-----------------------------------------------------------------------------------+
|                              RL Fine-tuning System                                 |
+-----------------------------------------------------------------------------------+
|                                                                                    |
|  +------------------+     +----------------------+     +------------------------+  |
|  |   Isaac Lab      |     |   StudentActorCritic |     |    PPOStudent          |  |
|  |   Environment    |<--->|   (Actor + Critic)   |<--->|    Algorithm           |  |
|  +------------------+     +----------------------+     +------------------------+  |
|         ^                          ^                            ^                  |
|         |                          |                            |                  |
|         v                          v                            v                  |
|  +------------------+     +----------------------+     +------------------------+  |
|  | Domain           |     | MultiModalStudent    |     | StudentRollout         |  |
|  | Randomization    |     | Policy (DAgger)      |     | Storage                |  |
|  +------------------+     +----------------------+     +------------------------+  |
|         |                          |                            |                  |
|         v                          v                            v                  |
|  +------------------+     +----------------------+     +------------------------+  |
|  | - Depth Noise    |     | - ProprioEncoder     |     | - Proprio Buffer       |  |
|  | - Latency Sim    |     | - DepthEncoder       |     | - Depth Buffer         |  |
|  | - Lighting Aug   |     | - FusionTransformer  |     | - Action Buffer        |  |
|  | - Camera Dropout |     | - TransformerXL      |     | - Value Buffer         |  |
|  +------------------+     | - ActionHead         |     | - TXL Memory Buffer    |  |
|                           | - ValueHead (NEW)    |     +------------------------+  |
|                           +----------------------+                                 |
|                                                                                    |
+-----------------------------------------------------------------------------------+
```

### 2.2 Data Flow Diagram

```
                                    ROLLOUT PHASE
                                    =============

    +-------------+                                              +------------------+
    | Environment |---(obs: proprio, depth)-------------------->| Domain           |
    +-------------+                                              | Randomization    |
          ^                                                      +------------------+
          |                                                              |
          |                                                              v
          |                                              +---------------------------+
          |                                              | StudentActorCritic        |
          |                                              |---------------------------|
          |                                              | 1. Encode proprio         |
          |                                              | 2. Encode depth           |
          |                                              | 3. Fuse modalities        |
          |                                              | 4. Temporal (TXL + mems)  |
          |                                              | 5. Action head -> actions |
          |                                              | 6. Value head -> values   |
          |                                              +---------------------------+
          |                                                       |         |
          +--------(actions)--------------------------------------+         |
                                                                            v
                                                              +------------------+
                                                              | StudentRollout   |
                                                              | Storage          |
                                                              | (store transition)|
                                                              +------------------+


                                    TRAINING PHASE
                                    ==============

    +------------------+                              +------------------+
    | StudentRollout   |---(mini-batches)----------->| PPOStudent       |
    | Storage          |                              | Algorithm        |
    +------------------+                              +------------------+
                                                              |
                                                              v
                                                      +------------------+
                                                      | Compute:         |
                                                      | - Surrogate loss |
                                                      | - Value loss     |
                                                      | - Entropy bonus  |
                                                      +------------------+
                                                              |
                                                              v
                                                      +------------------+
                                                      | Update:          |
                                                      | - Policy params  |
                                                      | - Value params   |
                                                      +------------------+
```

### 2.3 Integration Points with Existing Codebase

| Integration Point | Existing Component | New Component | Interface |
|-------------------|-------------------|---------------|--------------|
| Student Policy | `MultiModalStudentPolicy` | `StudentActorCritic` | Wraps existing policy |
| Environment | `ParkourRslRlVecEnvWrapper` | Training script | Standard Gym interface |
| Rewards | `parkour_isaaclab/envs/mdp/rewards.py` | PPOStudent | Reuse existing rewards |
| Terrain Curriculum | `ParkourEvent.terrain.terrain_levels` | Training script | Read terrain levels |
| Camera Dropout | `CameraDropoutManager` | Domain Randomization | Extend existing |
| TXL Memory | `TransformerXLTemporal` | StudentActorCritic | Use existing API |

---

## 3. Component Design

### 3.1 StudentActorCritic Wrapper Architecture

**File Location:** `scripts/rsl_rl/modules/student_actor_critic.py`

**Purpose:** Wrap the pre-trained `MultiModalStudentPolicy` with a value head for PPO training.

```
+------------------------------------------------------------------+
|                      StudentActorCritic                           |
+------------------------------------------------------------------+
|                                                                   |
|  +------------------------------------------------------------+  |
|  |              MultiModalStudentPolicy (frozen/trainable)     |  |
|  |  +------------------+  +------------------+                 |  |
|  |  | ProprioEncoder   |  | DepthEncoder     |                 |  |
|  |  +------------------+  +------------------+                 |  |
|  |           |                    |                            |  |
|  |           +--------+-----------+                            |  |
|  |                    v                                        |  |
|  |  +------------------------------------------+               |  |
|  |  | MultiModalFusionTransformer              |               |  |
|  |  +------------------------------------------+               |  |
|  |                    |                                        |  |
|  |                    v                                        |  |
|  |  +------------------------------------------+               |  |
|  |  | TransformerXLTemporal                    |               |  |
|  |  | (with memory management)                 |               |  |
|  |  +------------------------------------------+               |  |
|  |                    |                                        |  |
|  |         +----------+----------+                             |  |
|  |         |                     |                             |  |
|  |         v                     v                             |  |
|  |  +-------------+       +-------------+                      |  |
|  |  | ActionHead  |       | ValueHead   | <-- NEW              |  |
|  |  +-------------+       +-------------+                      |  |
|  +------------------------------------------------------------+  |
|                                                                   |
|  Methods:                                                         |
|  - act(proprio, depth, mems) -> actions, log_probs, values, mems  |
|  - evaluate(proprio, depth, mems) -> values                       |
|  - get_actions_log_prob(actions) -> log_probs                     |
|  - reset_memory(env_ids) -> None                                  |
|  - detach_memory() -> None                                        |
|                                                                   |
|  Properties:                                                      |
|  - action_mean, action_std, entropy, distribution                 |
|                                                                   |
+------------------------------------------------------------------+
```

**Key Design Decisions:**

1. **Composition over Inheritance**: Wrap `MultiModalStudentPolicy` rather than inherit from it
2. **Shared Feature Extraction**: Value head shares features from TransformerXL output
3. **Gaussian Policy**: Use Normal distribution for continuous actions (matching existing teacher)
4. **Learnable Std**: Use log-parameterized standard deviation (matching `JointPoseActionHead`)

**Interface Contract:**

```python
class StudentActorCritic(nn.Module):
    """Actor-Critic wrapper for MultiModalStudentPolicy."""

    def __init__(
        self,
        student_policy: MultiModalStudentPolicy,
        value_hidden_dims: Tuple[int, ...] = (256, 256),
        init_noise_std: float = 1.0,
        freeze_encoders: bool = False,
        freeze_fusion: bool = False,
        freeze_temporal: bool = False,
    ) -> None: ...

    def act(
        self,
        proprio: Tensor,  # [B, prop_hist_len * proprio_dim]
        depth: Tensor,    # [B, depth_hist_len, H, W]
        mems: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]:
        """Sample actions and compute values.

        Returns:
            actions: [B, action_dim]
            log_probs: [B]
            values: [B, 1]
            new_mems: List of memory tensors
        """
        ...

    def evaluate(
        self,
        proprio: Tensor,
        depth: Tensor,
        mems: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """Compute state values only."""
        ...

    def get_actions_log_prob(self, actions: Tensor) -> Tensor:
        """Compute log probability of given actions."""
        ...

    def reset_memory(self, env_ids: Tensor) -> None:
        """Reset memory for specified environments."""
        ...

    def detach_memory(self) -> None:
        """Detach memory from computation graph."""
        ...
```

### 3.2 ValueHead Module Design

**File Location:** `parkour_tasks/parkour_tasks/extreme_parkour_task/modules/actionheads/value_head.py`

**Purpose:** Estimate state values from fused temporal features.

```
+------------------------------------------+
|               ValueHead                   |
+------------------------------------------+
|                                          |
|  Input: temporal_features [B, S, d_model]|
|                    |                     |
|                    v                     |
|  +----------------------------------+    |
|  | Linear(d_model, 256) + ReLU      |    |
|  +----------------------------------+    |
|                    |                     |
|                    v                     |
|  +----------------------------------+    |
|  | Linear(256, 256) + ReLU          |    |
|  +----------------------------------+    |
|                    |                     |
|                    v                     |
|  +----------------------------------+    |
|  | Linear(256, 1)                   |    |
|  +----------------------------------+    |
|                    |                     |
|                    v                     |
|  Output: values [B, S, 1]                |
|                                          |
+------------------------------------------+
```

**Interface Contract:**

```python
class ValueHead(nn.Module):
    """Value function head for PPO."""

    def __init__(
        self,
        d_model: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ) -> None: ...

    def forward_step(self, h: Tensor) -> Tensor:
        """Single timestep value estimation.

        Args:
            h: [B, d_model]
        Returns:
            values: [B, 1]
        """
        ...

    def forward_sequence(self, h_seq: Tensor) -> Tensor:
        """Sequence value estimation.

        Args:
            h_seq: [B, S, d_model]
        Returns:
            values: [B, S, 1]
        """
        ...
```
### 3.3 PPOStudent Algorithm Design

**File Location:** `scripts/rsl_rl/modules/ppo_student.py`

**Purpose:** PPO algorithm adapted for image-based observations and Transformer-XL memory.

```
+------------------------------------------------------------------+
|                         PPOStudent                                |
+------------------------------------------------------------------+
|                                                                   |
|  Components:                                                      |
|  +------------------+  +------------------+  +------------------+ |
|  | StudentActor     |  | StudentRollout   |  | Optimizer        | |
|  | Critic           |  | Storage          |  | (AdamW)          | |
|  +------------------+  +------------------+  +------------------+ |
|                                                                   |
|  State:                                                           |
|  - current_mems: List[Tensor]  # TXL memory per layer             |
|  - transition: Transition      # Current step data                |
|                                                                   |
|  Methods:                                                         |
|  +------------------------------------------------------------+  |
|  | act(proprio, depth) -> actions                              |  |
|  |   1. Forward through actor-critic with mems                 |  |
|  |   2. Store transition data                                  |  |
|  |   3. Update mems                                            |  |
|  |   4. Return sampled actions                                 |  |
|  +------------------------------------------------------------+  |
|  | process_env_step(rewards, dones, infos)                     |  |
|  |   1. Store rewards, dones                                   |  |
|  |   2. Reset mems for done envs                               |  |
|  |   3. Handle episode termination                             |  |
|  +------------------------------------------------------------+  |
|  | compute_returns(last_values)                                |  |
|  |   1. GAE computation                                        |  |
|  |   2. Store returns and advantages                           |  |
|  +------------------------------------------------------------+  |
|  | update() -> loss_dict                                       |  |
|  |   1. Generate mini-batches (sequence-aware)                 |  |
|  |   2. Compute PPO losses                                     |  |
|  |   3. Update parameters                                      |  |
|  |   4. Return loss metrics                                    |  |
|  +------------------------------------------------------------+  |
|                                                                   |
+------------------------------------------------------------------+
```

**Key Differences from Existing PPOWithExtractor:**

| Aspect | PPOWithExtractor | PPOStudent |
|--------|------------------|------------|
| Observation | 1D vector (proprio + height scan) | Proprio + Depth images |
| Memory | None (MLP-based) | Transformer-XL memory |
| Batching | Random mini-batches | Sequence-aware batching |
| Estimator | Privileged state estimator | Not needed (uses depth) |
| Storage | Standard RolloutStorage | StudentRolloutStorage |

**Interface Contract:**

```python
class PPOStudent:
    """PPO algorithm for student policy with Transformer-XL."""

    def __init__(
        self,
        actor_critic: StudentActorCritic,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 1e-4,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        device: str = "cuda",
    ) -> None: ...

    def init_storage(
        self,
        num_envs: int,
        num_steps: int,
        proprio_dim: int,
        depth_shape: Tuple[int, ...],
        action_dim: int,
    ) -> None: ...

    def act(self, proprio: Tensor, depth: Tensor) -> Tensor: ...

    def process_env_step(
        self,
        rewards: Tensor,
        dones: Tensor,
        infos: Dict,
    ) -> None: ...

    def compute_returns(self, last_values: Tensor) -> None: ...

    def update(self) -> Dict[str, float]: ...
```


### 3.4 StudentRolloutStorage Design

**File Location:** `scripts/rsl_rl/modules/student_rollout_storage.py`

**Purpose:** Store rollout data including depth images and TXL memory states.

```
+------------------------------------------------------------------+
|                    StudentRolloutStorage                          |
+------------------------------------------------------------------+
|                                                                   |
|  Buffers (all pre-allocated):                                     |
|  +------------------------------------------------------------+  |
|  | proprio:    [num_steps, num_envs, proprio_dim]              |  |
|  | depth:      [num_steps, num_envs, depth_hist, H, W]         |  |
|  | actions:    [num_steps, num_envs, action_dim]               |  |
|  | rewards:    [num_steps, num_envs, 1]                        |  |
|  | values:     [num_steps + 1, num_envs, 1]                    |  |
|  | returns:    [num_steps, num_envs, 1]                        |  |
|  | advantages: [num_steps, num_envs, 1]                        |  |
|  | log_probs:  [num_steps, num_envs, 1]                        |  |
|  | dones:      [num_steps, num_envs, 1]                        |  |
|  | mu:         [num_steps, num_envs, action_dim]               |  |
|  | sigma:      [num_steps, num_envs, action_dim]               |  |
|  +------------------------------------------------------------+  |
|                                                                   |
|  Memory Management:                                               |
|  +------------------------------------------------------------+  |
|  | mems_at_step: List[List[Tensor]]  # [step][layer][B, M, C]  |  |
|  |   - Stores TXL memory state at each rollout step            |  |
|  |   - Used for sequence-aware mini-batch generation           |  |
|  +------------------------------------------------------------+  |
|                                                                   |
|  Methods:                                                         |
|  - add_transition(step, proprio, depth, actions, ...)             |
|  - compute_returns(last_values, gamma, lam)                       |
|  - sequence_mini_batch_generator(num_batches, num_epochs)         |
|  - clear()                                                        |
|                                                                   |
+------------------------------------------------------------------+
```

**Memory Estimation:**

For `num_envs=256`, `num_steps=64`, `depth_shape=(4, 58, 87)`:

| Buffer | Shape | Size (float32) |
|--------|-------|----------------|
| proprio | [64, 256, 53] | 3.5 MB |
| depth | [64, 256, 4, 58, 87] | 1.3 GB |
| actions | [64, 256, 12] | 0.8 MB |
| values | [65, 256, 1] | 0.07 MB |
| Total (approx) | - | ~1.5 GB |

**Sequence-Aware Mini-Batch Generation:**

Unlike standard PPO which randomly shuffles transitions, we must maintain temporal ordering for Transformer-XL:

```python
def sequence_mini_batch_generator(self, num_batches: int, num_epochs: int):
    """Generate mini-batches that preserve sequence structure.

    Strategy: Split environments into mini-batches, keep full sequences.
    Each mini-batch contains: [num_steps, num_envs // num_batches, ...]
    """
    batch_size = self.num_envs // num_batches
    env_indices = torch.randperm(self.num_envs)

    for epoch in range(num_epochs):
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            batch_env_ids = env_indices[start:end]

            yield {
                "proprio": self.proprio[:, batch_env_ids],
                "depth": self.depth[:, batch_env_ids],
                "actions": self.actions[:, batch_env_ids],
                # ... other buffers
                "initial_mems": self.mems_at_step[0][batch_env_ids],
            }
```


### 3.5 Domain Randomization Pipeline

**File Location:** `parkour_isaaclab/envs/mdp/domain_randomization.py`

**Purpose:** Apply augmentations to depth images for sim-to-real robustness.

```
+------------------------------------------------------------------+
|                    DomainRandomization                            |
+------------------------------------------------------------------+
|                                                                   |
|  +------------------------------------------------------------+  |
|  |                  DepthNoiseAugmentation                     |  |
|  |  +------------------+  +------------------+                 |  |
|  |  | Gaussian Noise   |  | Salt-Pepper      |                 |  |
|  |  | std: 0.01-0.04   |  | prob: 0.5-2%     |                 |  |
|  |  +------------------+  +------------------+                 |  |
|  |  +------------------+  +------------------+                 |  |
|  |  | Missing Pixels   |  | Depth Quant      |                 |  |
|  |  | prob: 0.5-2%     |  | levels: 256-1024 |                 |  |
|  |  +------------------+  +------------------+                 |  |
|  |  +------------------+                                       |  |
|  |  | Scale Variation  |                                       |  |
|  |  | range: 0.95-1.05 |                                       |  |
|  |  +------------------+                                       |  |
|  +------------------------------------------------------------+  |
|                                                                   |
|  +------------------------------------------------------------+  |
|  |                  LatencySimulation                          |  |
|  |  +------------------+                                       |  |
|  |  | Depth Frame      |  Buffer: deque(maxlen=delay+1)        |  |
|  |  | Delay: 1-5 frames|  Returns: buffer[0] (oldest)          |  |
|  |  +------------------+                                       |  |
|  +------------------------------------------------------------+  |
|                                                                   |
|  +------------------------------------------------------------+  |
|  |                  LightingAugmentation                       |  |
|  |  +------------------+  +------------------+                 |  |
|  |  | Brightness       |  | Contrast         |                 |  |
|  |  | range: 0.8-1.2   |  | range: 0.8-1.2   |                 |  |
|  |  +------------------+  +------------------+                 |  |
|  +------------------------------------------------------------+  |
|                                                                   |
|  +------------------------------------------------------------+  |
|  |                  CameraDropoutManager                       |  |
|  |  (Existing - extend with curriculum)                        |  |
|  |  prob: 0.1 -> 0.5 over training                             |  |
|  +------------------------------------------------------------+  |
|                                                                   |
|  +------------------------------------------------------------+  |
|  |                  DomainRandCurriculum                       |  |
|  |  - Tracks current iteration                                 |  |
|  |  - Returns augmentation params based on schedule            |  |
|  |  - Linear interpolation between stages                      |  |
|  +------------------------------------------------------------+  |
|                                                                   |
+------------------------------------------------------------------+
```

**Curriculum Schedule:**

```python
class DomainRandCurriculum:
    """Curriculum for domain randomization intensity."""

    SCHEDULE = [
        # (iteration, noise_std, salt_pepper, dropout, latency)
        (0,    0.01, 0.005, 0.1, 1),
        (1000, 0.02, 0.01,  0.2, 2),
        (3000, 0.03, 0.015, 0.3, 3),
        (5000, 0.04, 0.02,  0.5, 3),
    ]

    def get_params(self, iteration: int) -> Dict[str, float]:
        """Interpolate parameters based on current iteration."""
        ...
```

---

## 4. Memory Management Strategy

### 4.1 Transformer-XL Memory Handling During Rollouts

The Transformer-XL architecture maintains memory across time steps, which requires careful management during RL training.

**Memory Flow Diagram:**

```
Step 0:  mems = None (or zeros)
         |
         v
    +----------+
    | TXL      |---> output_0, mems_0
    +----------+
         |
         v (store mems_0)

Step 1:  mems = mems_0
         |
         v
    +----------+
    | TXL      |---> output_1, mems_1
    +----------+
         |
         v (store mems_1)

...

Step N:  mems = mems_{N-1}
         |
         v
    +----------+
    | TXL      |---> output_N, mems_N
    +----------+
         |
         v (end of rollout)
```

### 4.2 Memory Reset Logic on Episode Termination

When an episode terminates (done=True), the memory for that environment must be reset to prevent information leakage across episodes.

```python
def process_env_step(self, rewards: Tensor, dones: Tensor, infos: Dict):
    """Process environment step and handle memory reset."""

    # Store transition
    self.storage.add_transition(
        step=self.current_step,
        rewards=rewards,
        dones=dones,
        # ...
    )

    # Reset memory for done environments
    if dones.any():
        done_mask = dones.bool()
        for layer_idx in range(len(self.current_mems)):
            if self.current_mems[layer_idx] is not None:
                # Zero out memory for done environments
                self.current_mems[layer_idx][done_mask] = 0.0
```

### 4.3 Memory Detachment for Gradient Isolation

During training, we must detach memory between segments to prevent gradients from flowing through the entire rollout history (which would cause memory issues and training instability).

```python
def update(self) -> Dict[str, float]:
    """PPO update with segment recurrence."""

    for batch in self.storage.sequence_mini_batch_generator(...):
        # Get initial memory for this batch (detached)
        mems = batch["initial_mems"]
        mems = TransformerXLTemporal.detach_mems(mems)

        # Forward through full sequence
        for step in range(self.num_steps):
            # ... compute losses
            pass

        # Backpropagate through current segment only
        loss.backward()
```

### 4.4 Sequence Batching for Training

**Challenge:** Standard PPO randomly shuffles transitions, but Transformer-XL requires sequential processing.

**Solution:** Batch by environment, not by timestep.

```
Standard PPO Batching:
+---+---+---+---+---+---+---+---+
| t0| t1| t2| t3| t4| t5| t6| t7|  <- Random timesteps from random envs
+---+---+---+---+---+---+---+---+

Sequence-Aware Batching:
+---+---+---+---+---+---+---+---+
| e0| e0| e0| e0| e0| e0| e0| e0|  <- Full sequence from env 0
+---+---+---+---+---+---+---+---+
| e1| e1| e1| e1| e1| e1| e1| e1|  <- Full sequence from env 1
+---+---+---+---+---+---+---+---+
| ...                           |
```


---

## 5. Training Pipeline

### 5.1 Rollout Collection Flow

```
+------------------------------------------------------------------+
|                     Rollout Collection                            |
+------------------------------------------------------------------+
|                                                                   |
|  for step in range(num_steps_per_env):                            |
|      |                                                            |
|      v                                                            |
|  +------------------------------------------------------------+  |
|  | 1. Get observations from environment                        |  |
|  |    obs = env.get_observations()                             |  |
|  |    proprio = obs["proprio"]                                 |  |
|  |    depth = obs["depth_camera"]                              |  |
|  +------------------------------------------------------------+  |
|      |                                                            |
|      v                                                            |
|  +------------------------------------------------------------+  |
|  | 2. Apply domain randomization                               |  |
|  |    depth = domain_rand.apply(depth, iteration)              |  |
|  +------------------------------------------------------------+  |
|      |                                                            |
|      v                                                            |
|  +------------------------------------------------------------+  |
|  | 3. Sample actions from policy                               |  |
|  |    actions, log_probs, values, new_mems =                   |  |
|  |        actor_critic.act(proprio, depth, mems)               |  |
|  +------------------------------------------------------------+  |
|      |                                                            |
|      v                                                            |
|  +------------------------------------------------------------+  |
|  | 4. Step environment                                         |  |
|  |    obs_next, rewards, dones, infos = env.step(actions)      |  |
|  +------------------------------------------------------------+  |
|      |                                                            |
|      v                                                            |
|  +------------------------------------------------------------+  |
|  | 5. Store transition                                         |  |
|  |    storage.add_transition(proprio, depth, actions,          |  |
|  |                           rewards, values, log_probs, dones)|  |
|  +------------------------------------------------------------+  |
|      |                                                            |
|      v                                                            |
|  +------------------------------------------------------------+  |
|  | 6. Handle episode termination                               |  |
|  |    if dones.any(): reset_memory(done_env_ids)               |  |
|  +------------------------------------------------------------+  |
|      |                                                            |
|      v                                                            |
|  +------------------------------------------------------------+  |
|  | 7. Update memory state                                      |  |
|  |    mems = new_mems                                          |  |
|  +------------------------------------------------------------+  |
|                                                                   |
+------------------------------------------------------------------+
```

### 5.2 PPO Update Cycle

```
+------------------------------------------------------------------+
|                        PPO Update                                 |
+------------------------------------------------------------------+
|                                                                   |
|  +------------------------------------------------------------+  |
|  | 1. Compute returns and advantages                           |  |
|  |    last_values = actor_critic.evaluate(last_obs)            |  |
|  |    storage.compute_returns(last_values, gamma, lam)         |  |
|  +------------------------------------------------------------+  |
|      |                                                            |
|      v                                                            |
|  +------------------------------------------------------------+  |
|  | 2. For each learning epoch:                                 |  |
|  |    for epoch in range(num_learning_epochs):                 |  |
|  +------------------------------------------------------------+  |
|      |                                                            |
|      v                                                            |
|  +------------------------------------------------------------+  |
|  | 3. Generate sequence-aware mini-batches                     |  |
|  |    for batch in storage.sequence_mini_batch_generator():    |  |
|  +------------------------------------------------------------+  |
|      |                                                            |
|      v                                                            |
|  +------------------------------------------------------------+  |
|  | 4. Forward pass with detached memory                        |  |
|  |    mems = detach_mems(batch["initial_mems"])                |  |
|  |    for step in range(num_steps):                            |  |
|  |        actions, log_probs, values, mems = ...               |  |
|  +------------------------------------------------------------+  |
|      |                                                            |
|      v                                                            |
|  +------------------------------------------------------------+  |
|  | 5. Compute PPO losses                                       |  |
|  |    ratio = exp(log_probs - old_log_probs)                   |  |
|  |    surr1 = ratio * advantages                               |  |
|  |    surr2 = clip(ratio, 1-eps, 1+eps) * advantages           |  |
|  |    policy_loss = -min(surr1, surr2).mean()                  |  |
|  |    value_loss = (values - returns).pow(2).mean()            |  |
|  |    entropy_loss = -entropy.mean()                           |  |
|  |    total_loss = policy_loss + value_coef * value_loss       |  |
|  |                 + entropy_coef * entropy_loss               |  |
|  +------------------------------------------------------------+  |
|      |                                                            |
|      v                                                            |
|  +------------------------------------------------------------+  |
|  | 6. Backward pass and optimization                           |  |
|  |    optimizer.zero_grad()                                    |  |
|  |    total_loss.backward()                                    |  |
|  |    clip_grad_norm_(parameters, max_grad_norm)               |  |
|  |    optimizer.step()                                         |  |
|  +------------------------------------------------------------+  |
|      |                                                            |
|      v                                                            |
|  +------------------------------------------------------------+  |
|  | 7. Adaptive learning rate (optional)                        |  |
|  |    if schedule == "adaptive":                               |  |
|  |        if kl_divergence > desired_kl * 1.5:                 |  |
|  |            learning_rate *= 0.5                             |  |
|  |        elif kl_divergence < desired_kl / 1.5:               |  |
|  |            learning_rate *= 2.0                             |  |
|  +------------------------------------------------------------+  |
|                                                                   |
+------------------------------------------------------------------+
```

### 5.3 Domain Randomization Curriculum

**Progressive Difficulty Schedule:**

| Iteration Range | Gaussian Noise | Salt-Pepper | Camera Dropout | Latency |
|----------------|----------------|-------------|----------------|---------|
| 0-1000 | 0.01 | 0.005 | 0.1 | 1 frame |
| 1000-3000 | 0.02 | 0.01 | 0.2 | 2 frames |
| 3000-5000 | 0.03 | 0.015 | 0.3 | 3 frames |
| 5000+ | 0.04 | 0.02 | 0.5 | 3 frames |

**Implementation:**

```python
class DomainRandCurriculum:
    def update(self, iteration: int):
        """Update augmentation parameters based on iteration."""
        params = self.get_params(iteration)
        
        self.depth_noise.gaussian_std = params["noise_std"]
        self.depth_noise.salt_pepper_prob = params["salt_pepper"]
        self.camera_dropout.dropout_prob = params["dropout"]
        self.latency_sim.delay_frames = params["latency"]
```

### 5.4 Checkpoint Management

**Checkpoint Structure:**

```python
checkpoint = {
    "iteration": iteration,
    "actor_critic_state_dict": actor_critic.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "learning_rate": learning_rate,
    "terrain_level": current_terrain_level,
    "domain_rand_params": domain_rand.get_params(iteration),
    "training_metrics": {
        "mean_episode_return": mean_return,
        "mean_episode_length": mean_length,
        "goal_progress": goal_progress,
    },
}
```

**Save Strategy:**

- Save every 100 iterations
- Keep last 5 checkpoints
- Save best checkpoint (highest mean return)
- Save final checkpoint at end of training

---

## 6. Key Design Decisions

### 6.1 Encoder Freezing vs End-to-End Fine-tuning

**Decision: End-to-End Fine-tuning (Recommended)**

**Options Considered:**

| Approach | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| **End-to-End Fine-tuning** | Maximum adaptation to RL objectives; Better performance potential | Risk of catastrophic forgetting | **Recommended** - Use low LR (1e-4) |
| **Freeze Encoders** | Preserve DAgger representations; Safer | Limited adaptation capability | Fallback if forgetting occurs |
| **Gradual Unfreezing** | Progressive adaptation; Balanced risk | More complex training schedule | Consider if needed |

**Implementation:**

```python
class StudentActorCritic(nn.Module):
    def __init__(
        self,
        student_policy: MultiModalStudentPolicy,
        freeze_encoders: bool = False,
        freeze_fusion: bool = False,
        freeze_temporal: bool = False,
    ):
        super().__init__()
        self.student_policy = student_policy
        
        # Apply freezing if requested
        if freeze_encoders:
            for param in self.student_policy.proprio_encoder.parameters():
                param.requires_grad = False
            for param in self.student_policy.depth_encoder.parameters():
                param.requires_grad = False
        
        if freeze_fusion:
            for param in self.student_policy.fusion_transformer.parameters():
                param.requires_grad = False
        
        if freeze_temporal:
            for param in self.student_policy.temporal_transformer.parameters():
                param.requires_grad = False
```

**Monitoring Strategy:**

- Track DAgger loss during RL training
- If DAgger loss increases significantly (>20%), consider freezing encoders
- Monitor action distribution shift (KL divergence from DAgger policy)

### 6.2 Observation Space Handling

**Challenge:** Student uses depth images while teacher uses height scan.

**Design Decision: Separate Observation Pipelines**

```
Teacher Pipeline:
    proprio + height_scan -> MLP -> actions

Student Pipeline:
    proprio + depth_images -> Encoders -> Transformer -> actions
```

**Implications:**

1. **Cannot directly compare policies**: Different observation spaces
2. **Focus on task performance**: Use environment rewards as ground truth
3. **No teacher distillation during RL**: Pure RL optimization
4. **Evaluation on same terrains**: Compare success rates, not actions

### 6.3 Sequence-Based Training for Transformer-XL

**Challenge:** Transformer-XL requires sequential data, but PPO typically shuffles transitions.

**Design Decision: Environment-Level Batching**

**Standard PPO:**
```python
# Shuffle all transitions
indices = torch.randperm(num_envs * num_steps)
for batch in indices.split(batch_size):
    # Random transitions from different envs and timesteps
    train_on_batch(transitions[batch])
```

**Sequence-Aware PPO:**
```python
# Shuffle environments, keep sequences intact
env_indices = torch.randperm(num_envs)
for batch_envs in env_indices.split(batch_size):
    # Full sequences from selected environments
    sequences = transitions[:, batch_envs]  # [num_steps, batch_size, ...]
    train_on_sequences(sequences)
```

**Trade-offs:**

| Aspect | Standard PPO | Sequence-Aware PPO |
|--------|-------------|-------------------|
| Sample efficiency | Higher (more diverse batches) | Lower (correlated sequences) |
| Memory compatibility | Incompatible with TXL | Compatible with TXL |
| Batch diversity | High | Medium |
| Implementation complexity | Simple | Moderate |

**Mitigation for Lower Sample Efficiency:**

1. Use more environments (256 vs 128)
2. Longer rollouts (64 steps vs 32)
3. More learning epochs (5 vs 3)

### 6.4 Domain Randomization Strategy

**Design Decision: Curriculum-Based Progressive Augmentation**

**Rationale:**

1. **Avoid overwhelming the policy**: Start with mild augmentations
2. **Gradual difficulty increase**: Allow policy to adapt progressively
3. **Maintain performance**: Prevent catastrophic performance drop

**Alternative Approaches Considered:**

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Fixed Intensity** | Same augmentation throughout | Simple | May be too hard initially |
| **Random Intensity** | Sample intensity each episode | Diverse | Unstable training |
| **Curriculum (Chosen)** | Gradual increase | Stable, adaptive | Requires tuning schedule |
| **Adaptive** | Adjust based on performance | Optimal difficulty | Complex implementation |

**Implementation:**

```python
class DomainRandCurriculum:
    def get_params(self, iteration: int) -> Dict[str, float]:
        """Linear interpolation between curriculum stages."""
        for i in range(len(self.SCHEDULE) - 1):
            iter_start, *params_start = self.SCHEDULE[i]
            iter_end, *params_end = self.SCHEDULE[i + 1]
            
            if iter_start <= iteration < iter_end:
                # Linear interpolation
                alpha = (iteration - iter_start) / (iter_end - iter_start)
                return {
                    "noise_std": lerp(params_start[0], params_end[0], alpha),
                    "salt_pepper": lerp(params_start[1], params_end[1], alpha),
                    "dropout": lerp(params_start[2], params_end[2], alpha),
                    "latency": int(lerp(params_start[3], params_end[3], alpha)),
                }
        
        # Return final stage params
        return self.SCHEDULE[-1][1:]
```

---

## 7. Integration Points

### 7.1 Loading DAgger Checkpoint

**File:** `scripts/rsl_rl/train_student_rl_finetune.py`

```python
def load_dagger_checkpoint(checkpoint_path: str) -> MultiModalStudentPolicy:
    """Load pre-trained DAgger student policy."""
    
    # Initialize policy architecture
    policy = MultiModalStudentPolicy(
        proprio_encoder_cfg=...,
        depth_encoder_cfg=...,
        fusion_cfg=...,
        temporal_cfg=...,
        action_head_cfg=...,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    # Load weights
    policy.load_state_dict(state_dict, strict=True)
    
    print(f"Loaded DAgger checkpoint from {checkpoint_path}")
    print(f"Training iteration: {checkpoint.get('iteration', 'unknown')}")
    
    return policy
```

### 7.2 Isaac Lab Environment Integration

**Observation Format:**

```python
obs = env.get_observations()
# obs is a dict with keys:
# - "policy": dict with "proprio" and "depth_camera"
# - "critic": dict with privileged info (not used for student)

proprio = obs["policy"]["proprio"]  # [num_envs, proprio_dim]
depth = obs["policy"]["depth_camera"]  # [num_envs, H, W]
```

**Action Format:**

```python
actions = ppo.act(proprio, depth)  # [num_envs, action_dim]
obs_next, rewards, dones, infos = env.step(actions)
```

**Reward Access:**

```python
# Rewards are automatically computed by environment
# Access individual reward components via infos
for reward_name in infos["episode"]["rew_names"]:
    reward_value = infos["episode"][reward_name]
```

### 7.3 Reusing Existing Reward Functions

**Location:** `parkour_isaaclab/envs/mdp/rewards.py`

**Integration:** No changes needed - rewards are computed by environment.

**Reward Configuration:**

```python
# In environment config
@configclass
class RewardsCfg:
    # Positive rewards
    tracking_goal_vel = RewTerm(func=mdp.reward_tracking_goal_vel, weight=1.5)
    tracking_yaw = RewTerm(func=mdp.reward_tracking_yaw, weight=0.5)
    
    # Negative rewards (penalties)
    collision = RewTerm(func=mdp.reward_collision, weight=-10.0)
    feet_edge = RewTerm(func=mdp.reward_feet_edge, weight=-1.0)
    # ... (14 total reward terms)
```

### 7.4 Terrain Curriculum Integration

**Existing System:**

```python
# Terrain levels automatically adjusted based on performance
# Access current level via environment
current_level = env.unwrapped.terrain_levels  # [num_envs]
```

**Starting Point Configuration:**

```python
# In training script
env.unwrapped.terrain_levels[:] = args.start_terrain_level  # e.g., 5
```

**Automatic Adjustment:**

- Success (>80% goal progress): level += 1
- Failure (<40% goal progress): level -= 1
- Clamped to [0, max_level]

### 7.5 Camera Dropout Extension

**Existing:** `CameraDropoutManager` in DAgger training

**Extension:** Add curriculum for dropout probability

```python
class CameraDropoutManager:
    def __init__(self, initial_prob: float = 0.1):
        self.dropout_prob = initial_prob
    
    def update_curriculum(self, iteration: int):
        """Update dropout probability based on curriculum."""
        if iteration < 1000:
            self.dropout_prob = 0.1
        elif iteration < 3000:
            self.dropout_prob = 0.2
        elif iteration < 5000:
            self.dropout_prob = 0.3
        else:
            self.dropout_prob = 0.5
    
    def apply(self, depth: Tensor) -> Tensor:
        """Apply camera dropout to depth images."""
        if torch.rand(1).item() < self.dropout_prob:
            # Return last valid depth or zeros
            return self.last_valid_depth
        else:
            self.last_valid_depth = depth.clone()
            return depth
```

### 7.6 TXL Memory API

**Existing API:** `TransformerXLTemporal.forward()`

```python
# Forward pass with memory
output, new_mems = temporal_transformer(
    x=fused_features,  # [B, S, d_model]
    mems=current_mems,  # List[Tensor] or None
)

# Memory management utilities
@staticmethod
def detach_mems(mems: List[Tensor]) -> List[Tensor]:
    """Detach memory from computation graph."""
    return [mem.detach() if mem is not None else None for mem in mems]

@staticmethod
def reset_mems(mems: List[Tensor], env_ids: Tensor) -> List[Tensor]:
    """Reset memory for specified environments."""
    for mem in mems:
        if mem is not None:
            mem[env_ids] = 0.0
    return mems
```

---

## 8. Risk Mitigation Strategies

### 8.1 Catastrophic Forgetting Prevention

**Risks:**

1. RL fine-tuning destroys DAgger-learned representations
2. Policy forgets how to use depth images effectively
3. Performance degrades below DAgger baseline

**Mitigation Strategies:**

| Strategy | Implementation | When to Use |
|----------|---------------|-------------|
| **Low Learning Rate** | Use 1e-4 (vs 2e-4 for teacher) | Always |
| **Monitor DAgger Loss** | Evaluate on DAgger validation set | Every 100 iterations |
| **Freeze Encoders** | Set requires_grad=False | If DAgger loss increases >20% |
| **Early Stopping** | Stop if performance degrades | If return drops >30% |
| **Checkpoint Rollback** | Revert to best checkpoint | If training diverges |

**Monitoring Code:**

```python
def evaluate_dagger_loss(policy, dagger_val_loader):
    """Evaluate imitation loss on DAgger validation set."""
    policy.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dagger_val_loader:
            proprio, depth, teacher_actions = batch
            student_actions = policy(proprio, depth)
            loss = F.mse_loss(student_actions, teacher_actions)
            total_loss += loss.item()
    
    policy.train()
    return total_loss / len(dagger_val_loader)
```

### 8.2 Training Stability Measures

**Risks:**

1. Value function divergence
2. Policy collapse (all actions become similar)
3. Gradient explosion/vanishing

**Mitigation Strategies:**

```python
# 1. Gradient clipping
torch.nn.utils.clip_grad_norm_(
    actor_critic.parameters(),
    max_norm=1.0
)

# 2. Value loss clipping
if use_clipped_value_loss:
    value_pred_clipped = value_pred_old + torch.clamp(
        value_pred - value_pred_old,
        -clip_param,
        clip_param
    )
    value_loss = torch.max(
        (value_pred - returns).pow(2),
        (value_pred_clipped - returns).pow(2)
    ).mean()

# 3. Entropy bonus (prevent policy collapse)
entropy_loss = -entropy_coef * entropy.mean()

# 4. KL divergence monitoring
kl_div = (old_log_probs - log_probs).mean()
if kl_div > desired_kl * 2.0:
    print("WARNING: KL divergence too high, stopping epoch early")
    break
```

### 8.3 Memory Management Validation

**Risks:**

1. Memory not reset on episode termination
2. Memory shapes inconsistent
3. Gradients flow through entire history

**Validation Code:**

```python
def validate_memory_management(actor_critic, storage):
    """Validate TXL memory handling."""
    
    # Check 1: Memory reset on done
    mems = actor_critic.get_initial_mems(batch_size=4)
    dones = torch.tensor([True, False, True, False])
    actor_critic.reset_memory(dones.nonzero().squeeze())
    
    for layer_idx, mem in enumerate(mems):
        assert mem[0].abs().sum() == 0, f"Layer {layer_idx} mem[0] not reset"
        assert mem[2].abs().sum() == 0, f"Layer {layer_idx} mem[2] not reset"
        assert mem[1].abs().sum() > 0, f"Layer {layer_idx} mem[1] incorrectly reset"
    
    # Check 2: Memory detachment
    mems_detached = actor_critic.detach_memory()
    for mem in mems_detached:
        assert not mem.requires_grad, "Memory not detached"
    
    # Check 3: Memory shapes
    for layer_idx, mem in enumerate(mems):
        expected_shape = (4, actor_critic.mem_len, actor_critic.d_model)
        assert mem.shape == expected_shape, \
            f"Layer {layer_idx} memory shape {mem.shape} != {expected_shape}"
    
    print("✓ Memory management validation passed")
```


---

## 9. Implementation Roadmap

### 9.1 Phase 1: Architecture Modifications (3-4 days)

**Deliverables:**
- `value_head.py` - Value function module
- `student_actor_critic.py` - Actor-critic wrapper

**Tasks:**

1. **Create ValueHead Module**
   - Implement MLP architecture (d_model → 256 → 256 → 1)
   - Add forward_step() and forward_sequence() methods
   - Write unit tests

2. **Create StudentActorCritic Wrapper**
   - Implement composition pattern (wrap MultiModalStudentPolicy)
   - Add act(), evaluate(), get_actions_log_prob() methods
   - Implement memory management (reset_memory, detach_memory)
   - Add encoder freezing options
   - Write integration tests

3. **Validation**
   - Test forward/backward passes
   - Verify gradient flow
   - Check memory shapes
   - Validate checkpoint loading

**Success Criteria:**
- All unit tests pass
- Forward pass produces correct shapes
- Backward pass computes gradients
- Can load DAgger checkpoint successfully

### 9.2 Phase 2: PPO Algorithm Adaptation (4-5 days)

**Deliverables:**
- `student_rollout_storage.py` - Rollout buffer
- `ppo_student.py` - PPO algorithm

**Tasks:**

1. **Create StudentRolloutStorage**
   - Implement buffer allocation (proprio, depth, actions, etc.)
   - Add TXL memory storage
   - Implement sequence_mini_batch_generator()
   - Implement compute_returns() with GAE
   - Write unit tests

2. **Create PPOStudent Algorithm**
   - Implement act() for rollout collection
   - Implement process_env_step() with memory reset
   - Implement compute_returns()
   - Implement update() with sequence-aware batching
   - Add adaptive learning rate schedule
   - Write integration tests

3. **Validation**
   - Test on simple CartPole environment
   - Verify memory management during rollouts
   - Check loss computation
   - Validate gradient updates

**Success Criteria:**
- Storage correctly handles depth images
- Sequence batching preserves temporal order
- Memory reset works on episode termination
- PPO update converges on simple task

### 9.3 Phase 3: Domain Randomization (2-3 days)

**Deliverables:**
- `domain_randomization.py` - Augmentation pipeline

**Tasks:**

1. **Implement Augmentation Modules**
   - DepthNoiseAugmentation (Gaussian, salt-pepper, missing pixels)
   - LatencySimulation (frame delay buffer)
   - LightingAugmentation (brightness, contrast)
   - Extend CameraDropoutManager with curriculum

2. **Implement DomainRandCurriculum**
   - Define curriculum schedule
   - Implement parameter interpolation
   - Add update() method

3. **Integration**
   - Add augmentation pipeline to training loop
   - Test each augmentation independently
   - Test combined augmentations

**Success Criteria:**
- Each augmentation produces expected output
- Curriculum correctly interpolates parameters
- Augmentations don't break training
- Visual inspection of augmented depth images

### 9.4 Phase 4: Training Configuration (1 day)

**Deliverables:**
- `rsl_student_finetune_cfg.py` - Training config

**Tasks:**

1. **Create Configuration File**
   - Define PPO hyperparameters
   - Define domain randomization schedule
   - Define logging and checkpointing settings

2. **Validation**
   - Verify all parameters are reasonable
   - Check compatibility with existing configs

**Success Criteria:**
- Config file loads without errors
- All parameters within valid ranges

### 9.5 Phase 5: Training Script (2-3 days)

**Deliverables:**
- `train_student_rl_finetune.py` - Main training script

**Tasks:**

1. **Implement Training Loop**
   - Environment initialization
   - DAgger checkpoint loading
   - StudentActorCritic creation
   - PPOStudent initialization
   - Rollout collection loop
   - PPO update loop
   - Logging and checkpointing

2. **Add Command-Line Arguments**
   - Checkpoint paths
   - Hyperparameter overrides
   - Experiment name
   - Device selection

3. **Integration Testing**
   - Dry run with 1 environment
   - Short training run (10 iterations)
   - Verify checkpoints save/load correctly

**Success Criteria:**
- Script runs without errors
- Training progresses (loss decreases)
- Checkpoints save correctly
- Logs are informative

### 9.6 Phase 6: Evaluation (2-3 days)

**Deliverables:**
- `evaluate_student_robustness.py` - Evaluation script

**Tasks:**

1. **Implement Evaluation Metrics**
   - Episode return
   - Episode length
   - Goal progress
   - Success rate

2. **Implement Test Scenarios**
   - Clean observations
   - High noise
   - High latency
   - Camera dropout
   - Combined stress test

3. **Run Evaluation**
   - Evaluate DAgger baseline
   - Evaluate RL fine-tuned policy
   - Compare results
   - Generate plots

**Success Criteria:**
- Evaluation runs on all scenarios
- Results are reproducible
- Plots clearly show improvements

---

## 10. Testing Strategy

### 10.1 Unit Tests

**ValueHead:**
```python
def test_value_head_forward_step():
    value_head = ValueHead(d_model=256)
    h = torch.randn(32, 256)
    values = value_head.forward_step(h)
    assert values.shape == (32, 1)

def test_value_head_forward_sequence():
    value_head = ValueHead(d_model=256)
    h_seq = torch.randn(32, 10, 256)
    values = value_head.forward_sequence(h_seq)
    assert values.shape == (32, 10, 1)
```

**StudentActorCritic:**
```python
def test_student_actor_critic_act():
    policy = create_mock_student_policy()
    actor_critic = StudentActorCritic(policy)
    
    proprio = torch.randn(32, 53)
    depth = torch.randn(32, 4, 58, 87)
    
    actions, log_probs, values, mems = actor_critic.act(proprio, depth)
    
    assert actions.shape == (32, 12)
    assert log_probs.shape == (32,)
    assert values.shape == (32, 1)
    assert len(mems) == policy.num_layers
```

### 10.2 Integration Tests

**PPO Training Loop:**
```python
def test_ppo_training_loop():
    # Create simple environment
    env = gym.make("CartPole-v1")
    
    # Create actor-critic
    actor_critic = create_simple_actor_critic()
    
    # Create PPO
    ppo = PPOStudent(actor_critic, num_learning_epochs=1)
    ppo.init_storage(num_envs=4, num_steps=32, ...)
    
    # Collect rollouts
    for step in range(32):
        obs = env.get_observations()
        actions = ppo.act(obs["proprio"], obs["depth"])
        obs_next, rewards, dones, infos = env.step(actions)
        ppo.process_env_step(rewards, dones, infos)
    
    # Update
    loss_dict = ppo.update()
    
    assert "policy_loss" in loss_dict
    assert "value_loss" in loss_dict
    assert "entropy" in loss_dict
```

### 10.3 End-to-End Tests

**Full Training Run:**
```python
def test_full_training_run():
    # Run training for 10 iterations
    subprocess.run([
        "python", "scripts/rsl_rl/train_student_rl_finetune.py",
        "--dagger_checkpoint", "path/to/checkpoint.pth",
        "--max_iterations", "10",
        "--num_envs", "4",
    ], check=True)
    
    # Verify checkpoint exists
    assert os.path.exists("logs/student_rl_finetune/model_10.pth")
```

---

## 11. Monitoring and Debugging

### 11.1 Training Metrics

**Essential Metrics:**

| Metric | Description | Expected Trend |
|--------|-------------|----------------|
| `episode_return_mean` | Average episode return | Increasing |
| `episode_length_mean` | Average episode length | Increasing |
| `goal_progress` | % of goals reached | Increasing |
| `policy_loss` | PPO surrogate loss | Decreasing then stable |
| `value_loss` | Value function MSE | Decreasing then stable |
| `entropy` | Policy entropy | Slowly decreasing |
| `kl_divergence` | KL from old policy | < desired_kl |
| `learning_rate` | Current LR | Stable or adaptive |

**Robustness Metrics:**

| Metric | Description | Target |
|--------|-------------|--------|
| `return_clean` | Return without augmentation | Baseline |
| `return_noisy` | Return with noise | >80% of clean |
| `return_dropout` | Return with camera dropout | >85% of clean |
| `return_latency` | Return with latency | >80% of clean |
| `dagger_loss` | Imitation loss on val set | Stable |

### 11.2 Debugging Tools

**Memory Visualization:**
```python
def visualize_memory_states(mems: List[Tensor], step: int):
    """Visualize TXL memory states."""
    for layer_idx, mem in enumerate(mems):
        plt.figure(figsize=(10, 4))
        plt.imshow(mem[0].cpu().numpy(), aspect="auto", cmap="viridis")
        plt.title(f"Layer {layer_idx} Memory at Step {step}")
        plt.xlabel("Feature Dimension")
        plt.ylabel("Memory Length")
        plt.colorbar()
        plt.savefig(f"debug/mem_layer{layer_idx}_step{step}.png")
        plt.close()
```

**Augmentation Visualization:**
```python
def visualize_augmentations(depth: Tensor, augmented_depth: Tensor):
    """Visualize original and augmented depth images."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].imshow(depth[0, 0].cpu().numpy(), cmap="gray")
    axes[0].set_title("Original Depth")
    
    axes[1].imshow(augmented_depth[0, 0].cpu().numpy(), cmap="gray")
    axes[1].set_title("Augmented Depth")
    
    plt.savefig("debug/augmentation_comparison.png")
    plt.close()
```

**Gradient Flow Check:**
```python
def check_gradient_flow(named_parameters):
    """Check for vanishing/exploding gradients."""
    ave_grads = []
    max_grads = []
    layers = []
    
    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(ave_grads)), ave_grads, alpha=0.5, label="mean")
    plt.bar(range(len(max_grads)), max_grads, alpha=0.5, label="max")
    plt.xticks(range(len(layers)), layers, rotation=90)
    plt.xlabel("Layers")
    plt.ylabel("Gradient Magnitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("debug/gradient_flow.png")
    plt.close()
```

---

## 12. Performance Optimization

### 12.1 Memory Optimization

**Depth Image Storage:**
- Use uint8 instead of float32 for storage (4x reduction)
- Convert to float32 only during forward pass
- Use mixed precision training (fp16)

```python
class StudentRolloutStorage:
    def __init__(self, ...):
        # Store depth as uint8 (0-255)
        self.depth = torch.zeros(
            num_steps, num_envs, depth_hist, H, W,
            dtype=torch.uint8,
            device="cpu"  # Keep on CPU to save GPU memory
        )
    
    def get_batch(self, indices):
        # Convert to float32 and normalize on-the-fly
        depth_batch = self.depth[:, indices].float() / 255.0
        depth_batch = depth_batch.to(self.device)
        return depth_batch
```

### 12.2 Computation Optimization

**Encoder Caching:**
- Cache encoded features if encoders are frozen
- Recompute only when needed

```python
class StudentActorCritic(nn.Module):
    def __init__(self, ..., cache_encodings: bool = False):
        self.cache_encodings = cache_encodings
        self.encoding_cache = {}
    
    def encode(self, proprio, depth):
        if self.cache_encodings and self.encoders_frozen:
            cache_key = (proprio.data_ptr(), depth.data_ptr())
            if cache_key in self.encoding_cache:
                return self.encoding_cache[cache_key]
        
        encoded = self._encode(proprio, depth)
        
        if self.cache_encodings:
            self.encoding_cache[cache_key] = encoded
        
        return encoded
```

### 12.3 Data Loading Optimization

**Asynchronous Augmentation:**
```python
class AsyncDomainRandomization:
    def __init__(self, domain_rand, num_workers=4):
        self.domain_rand = domain_rand
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.futures = {}
    
    def apply_async(self, depth, env_id):
        """Apply augmentation asynchronously."""
        future = self.executor.submit(self.domain_rand.apply, depth)
        self.futures[env_id] = future
    
    def get_result(self, env_id):
        """Get augmentation result."""
        return self.futures[env_id].result()
```

---

## 13. Conclusion

This architectural design provides a comprehensive blueprint for fine-tuning the DAgger-trained student policy using PPO reinforcement learning with domain randomization. The design emphasizes:

1. **Modularity**: Clean separation between components
2. **Safety**: Careful memory management and catastrophic forgetting prevention
3. **Robustness**: Progressive domain randomization curriculum
4. **Maintainability**: Clear interfaces and extensive validation

### Key Innovations

1. **Sequence-Aware PPO**: Adapted PPO algorithm for Transformer-XL memory
2. **Progressive Domain Randomization**: Curriculum-based augmentation schedule
3. **Memory Management**: Robust TXL memory handling during RL training
4. **Composition Pattern**: Wrapping DAgger policy without modification

### Next Steps

1. Review and approve this architecture design
2. Create feature branch: `feature/rl-finetuning`
3. Begin Phase 1 implementation (Architecture Modifications)
4. Iterate based on testing and validation results

---

**Document Status:** Ready for Review
**Last Updated:** 2026-02-02
**Version:** 1.0

