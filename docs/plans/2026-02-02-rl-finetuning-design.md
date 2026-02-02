# Implementation Plan: RL Fine-tuning of DAgger-Trained Student Model

**Date:** 2026-02-02
**Goal:** Reduce sim-to-real gap by making the student policy more robust to real-world conditions (camera noise, latency, lighting variations, sensor delays) while maintaining performance

---

## Overview

This plan details how to fine-tune the DAgger-trained student policy (`MultiModalStudentPolicy` with Transformer-XL) using PPO reinforcement learning. The approach involves:

1. Adding a value head to the student architecture for PPO
2. Integrating with the existing PPO framework and reward system
3. Implementing comprehensive domain randomization for robustness
4. Managing Transformer-XL memory during RL rollouts

---

## Current State Analysis

### Student Model (DAgger-trained)
- **Architecture:** ProprioEncoder → DepthEncoder → MultiModalFusionTransformer → TransformerXLTemporal → JointPoseActionHead
- **Input:** Proprioception (stacked history) + Depth images (stacked frames)
- **Output:** Joint position actions (12 dims)
- **Training:** Imitation learning (DAgger) with teacher supervision
- **Location:** `scripts/rsl_rl/train_student_from_dataset.py`

### Existing RL Framework
- **Algorithm:** PPO with adaptive learning rate
- **Rewards:** 14 reward functions (tracking goal velocity, collision penalties, etc.)
- **Curriculum:** Automatic terrain difficulty adjustment
- **Teacher:** Uses height scan + privileged info
- **Location:** `scripts/rsl_rl/train.py`, `scripts/rsl_rl/modules/on_policy_runner_with_extractor.py`

### Key Challenge
Student uses **depth camera images** while teacher uses **height scan**. Need to adapt PPO framework for image-based observations and Transformer-XL temporal modeling.

---

## Architecture Modifications

### Phase 1.1: Add Value Head to Student Policy

**Complexity:** HIGH
**Risk:** MEDIUM
**Dependencies:** None

**Files to Create:**

1. `scripts/rsl_rl/modules/student_actor_critic.py` - Wrapper class combining student policy with value head
2. `parkour_tasks/parkour_tasks/extreme_parkour_task/modules/actionheads/value_head.py` - Value head module

**Implementation Steps:**

1. **Create Value Head Module** (`value_head.py`):
   - Input: Fused temporal features from TransformerXL `[B, S, d_model]`
   - Architecture: MLP with hidden dims `[256, 256, 1]`
   - Output: State value estimates `[B, S, 1]`
   - Use ReLU activation, no output activation

2. **Create StudentActorCritic Wrapper** (`student_actor_critic.py`):
   - Load pre-trained `MultiModalStudentPolicy` from DAgger checkpoint
   - Add `ValueHead` module
   - Implement PPO-required methods:
     - `act()`: Sample actions from policy
     - `evaluate()`: Compute value estimates
     - `get_actions_log_prob()`: Compute log probabilities
     - `update_distribution()`: Update action distribution parameters
   - Handle Transformer-XL memory management:
     - `reset_memory()`: Reset memory on episode done
     - `detach_memory()`: Detach memory for gradient isolation

3. **Design Decision - Encoder Freezing Strategy**:
   - **Option A (Recommended)**: Fine-tune all layers end-to-end
     - Pros: Maximum adaptation to RL objectives
     - Cons: Risk of catastrophic forgetting
   - **Option B**: Freeze encoders, train only value head and action head
     - Pros: Preserve DAgger representations
     - Cons: Limited adaptation capability
   - **Recommendation**: Start with Option A, add Option B as fallback if performance degrades

**Architecture Diagram:**
```
Input: proprio_seq [B, S, prop_hist * proprio_dim]
       depth_seq [B, S, depth_hist, H, W]
                    |
                    v
        +-------------------+
        |  ProprioEncoder   |  ← Fine-tune or freeze
        +-------------------+
                    |
                    v
        +-------------------+
        |   DepthEncoder    |  ← Fine-tune or freeze
        +-------------------+
                    |
                    v
        +-------------------+
        | MultiModalFusion  |  ← Fine-tune or freeze
        +-------------------+
                    |
                    v
        +-------------------+
        | TransformerXL     |  ← Fine-tune or freeze
        +-------------------+
                    |
          +---------+---------+
          |                   |
          v                   v
    +------------+      +------------+
    | ActionHead |      | ValueHead  |  ← NEW
    +------------+      +------------+
          |                   |
          v                   v
      actions              values
    [B, S, 12]           [B, S, 1]
```

---

## PPO Algorithm Adaptation

### Phase 2.1: Create PPO Algorithm for Student

**Complexity:** HIGH
**Risk:** HIGH
**Dependencies:** Phase 1.1 complete

**Files to Create:**

1. `scripts/rsl_rl/modules/ppo_student.py` - PPO algorithm adapted for student policy
2. `scripts/rsl_rl/modules/student_rollout_storage.py` - Rollout buffer for depth images and sequences

**Key Challenges:**

1. **Observation Space Difference**:
   - Teacher: Proprioception + Height scan (1D array)
   - Student: Proprioception + Depth images (3D tensor)
   - Solution: Custom rollout storage for image observations

2. **Transformer-XL Memory Management**:
   - Must maintain memory across rollout steps
   - Reset memory on episode termination
   - Detach memory between training segments
   - Solution: Store memory states in rollout buffer

3. **Sequence-based Training**:
   - TXL requires sequences, not single timesteps
   - Solution: Collect rollouts in sequences of length `sequence_length` (e.g., 64)

**Implementation Steps:**

1. **Create StudentRolloutStorage** (`student_rollout_storage.py`):
   ```python
   class StudentRolloutStorage:
       def __init__(self, num_envs, num_steps, proprio_dim, depth_shape, action_dim):
           # Buffers
           self.proprio = torch.zeros(num_steps, num_envs, proprio_dim)
           self.depth = torch.zeros(num_steps, num_envs, *depth_shape)
           self.actions = torch.zeros(num_steps, num_envs, action_dim)
           self.rewards = torch.zeros(num_steps, num_envs, 1)
           self.values = torch.zeros(num_steps + 1, num_envs, 1)
           self.returns = torch.zeros(num_steps, num_envs, 1)
           self.advantages = torch.zeros(num_steps, num_envs, 1)
           self.log_probs = torch.zeros(num_steps, num_envs, 1)
           self.dones = torch.zeros(num_steps, num_envs, 1)
           # TXL memory (optional)
           self.mems = None
   ```

2. **Create PPOStudent Algorithm** (`ppo_student.py`):
   - Adapt from existing `PPOWithExtractor` but handle depth images
   - Key methods:
     - `act()`: Sample actions during rollout
     - `process_env_step()`: Store transitions, handle memory reset
     - `compute_returns()`: GAE computation
     - `update()`: PPO update with mini-batches
   - Handle sequence batching for TXL

3. **Memory Management Strategy**:
   - **During Rollout**: Maintain memory across steps, reset on done
   - **During Training**: Use segment recurrence (detach memory between segments)
   - **Mini-batch Training**: Each mini-batch processes full sequences

---

## Domain Randomization for Robustness

### Phase 3.1: Implement Domain Randomization

**Complexity:** MEDIUM
**Risk:** MEDIUM
**Dependencies:** None (can be developed in parallel)

**Files to Create:**

1. `parkour_isaaclab/envs/mdp/domain_randomization.py` - Domain randomization utilities
2. `parkour_isaaclab/envs/mdp/observations.py` - MODIFY to add augmentations

**Augmentation Types:**

#### 3.1.1 Camera Noise Augmentations

**Purpose:** Simulate real-world depth camera imperfections

**Implementations:**

1. **Gaussian Noise**:
   - Add random Gaussian noise to depth values
   - Std: 0.01-0.05 (configurable)
   - Simulates sensor noise

2. **Salt-and-Pepper Noise**:
   - Randomly set pixels to min/max depth
   - Probability: 0.5-2% (configurable)
   - Simulates dead pixels

3. **Missing Pixels**:
   - Set random pixels to invalid depth (-0.5 in normalized space)
   - Probability: 0.5-2% (configurable)
   - Simulates IR reflection failures

4. **Depth Quantization**:
   - Quantize depth to simulate limited precision
   - Levels: 256-1024 (configurable)
   - Simulates sensor bit depth

5. **Depth Scale Variation**:
   - Multiply depth by random scale factor
   - Range: 0.95-1.05 (configurable)
   - Simulates calibration errors

**Implementation:**
```python
class DepthNoiseAugmentation:
    def __init__(self, cfg):
        self.gaussian_std = cfg.gaussian_std
        self.salt_pepper_prob = cfg.salt_pepper_prob
        self.missing_pixel_prob = cfg.missing_pixel_prob

    def apply(self, depth: Tensor) -> Tensor:
        # Apply augmentations
        ...
```

#### 3.1.2 Sensor Latency Simulation

**Purpose:** Simulate processing delays in real hardware

**Implementation:**

1. **Depth Frame Delay**:
   - Use depth from N frames ago
   - Delay: 1-5 frames (configurable)
   - Simulates camera processing latency

2. **Action Delay** (already exists):
   - Current system has 1-8 step action delay
   - Keep existing implementation

**Implementation:**
```python
class LatencySimulation:
    def __init__(self, depth_delay_frames=3):
        self.depth_delay = depth_delay_frames
        self.depth_buffer = deque(maxlen=depth_delay_frames + 1)

    def apply(self, depth: Tensor) -> Tensor:
        self.depth_buffer.append(depth)
        return self.depth_buffer[0]  # Return delayed depth
```

#### 3.1.3 Lighting Variations

**Purpose:** Simulate different lighting conditions affecting depth perception

**Implementation:**

1. **Brightness Adjustment**:
   - Multiply depth by brightness factor
   - Range: 0.8-1.2 (configurable)
   - Simulates ambient light changes

2. **Contrast Adjustment**:
   - Adjust depth contrast around mean
   - Range: 0.8-1.2 (configurable)
   - Simulates lighting uniformity

3. **Shadow Simulation**:
   - Add random dark regions to depth
   - Simulates shadows affecting IR pattern

**Implementation:**
```python
class LightingAugmentation:
    def __init__(self, cfg):
        self.brightness_range = cfg.brightness_range
        self.contrast_range = cfg.contrast_range

    def apply(self, depth: Tensor) -> Tensor:
        # Apply lighting augmentations
        ...
```

#### 3.1.4 Camera Dropout (Already Implemented)

**Status:** Already implemented in DAgger training via `CameraDropoutManager`

**Enhancement:** Add curriculum for dropout probability
- Start: 10% dropout probability
- End: 50% dropout probability
- Progression: Linear over training iterations

---

## Reward Function Strategy

### Phase 4.1: Reward Configuration

**Complexity:** LOW
**Risk:** LOW
**Dependencies:** None

**Decision:** Use existing 14 reward functions without modification

**Rationale:**
1. Existing rewards are well-tuned for parkour task
2. Focus on robustness through domain randomization, not reward engineering
3. Simplifies comparison with teacher baseline

**Existing Rewards (from `parkour_isaaclab/envs/mdp/rewards.py`):**

**Positive Rewards:**
- `reward_tracking_goal_vel` (weight: 1.5) - Primary objective
- `reward_tracking_yaw` (weight: 0.5) - Heading alignment

**Negative Rewards (Penalties):**
- `reward_collision` (weight: -10.0) - Strong safety penalty
- `reward_feet_edge` (weight: -1.0) - Terrain edge penalty
- `reward_lin_vel_z` (weight: -1.0) - Vertical velocity penalty
- `reward_orientation` (weight: -1.0) - Body tilt penalty
- `reward_feet_stumble` (weight: -1.0) - Foot dragging penalty
- `reward_action_rate` (weight: -0.1) - Smooth control
- `reward_ang_vel_xy` (weight: -0.05) - Stable orientation
- `reward_dof_error` (weight: -0.04) - Natural posture
- `reward_hip_pos` (weight: -0.5) - Hip configuration
- `reward_torques` (weight: -0.00001) - Energy efficiency
- `reward_dof_acc` (weight: -2.5e-7) - Smooth motion
- `reward_delta_torques` (weight: -1.0e-7) - Smooth actuation

**Optional Enhancement (Phase 4.2):**

If performance degrades, consider adding robustness-specific rewards:

1. **Depth Prediction Consistency Reward**:
   - Reward consistent actions under noisy observations
   - Compare actions with/without noise
   - Weight: 0.1-0.5

2. **Recovery Reward**:
   - Reward recovering from near-fall states
   - Detect high body tilt + successful recovery
   - Weight: 0.5-1.0

**Recommendation:** Start without additional rewards, add only if needed

---

## Training Configuration

### Phase 5.1: PPO Hyperparameters

**Complexity:** LOW
**Risk:** LOW
**Dependencies:** Phase 2 complete

**File to Create:**
- `parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/agents/rsl_student_finetune_cfg.py`

**Hyperparameter Strategy:**

**Key Principle:** Use conservative hyperparameters for fine-tuning to avoid catastrophic forgetting

**Recommended Configuration:**

```python
# Environment
num_envs = 256  # Same as DAgger
num_steps_per_env = 64  # Match TXL sequence length

# Training
max_iterations = 10000  # ~10M steps (256 envs * 64 steps * 10000)
save_interval = 100
log_interval = 10

# PPO Algorithm
num_learning_epochs = 5  # Same as teacher
num_mini_batches = 4  # Same as teacher
clip_param = 0.2  # Standard PPO
gamma = 0.99  # Standard discount
lam = 0.95  # GAE lambda
value_loss_coef = 1.0
entropy_coef = 0.01  # Encourage exploration

# Optimization
learning_rate = 1e-4  # LOWER than teacher (2e-4) for fine-tuning
schedule = "adaptive"  # Adaptive KL
desired_kl = 0.01
max_grad_norm = 1.0

# Encoder Freezing (optional)
freeze_proprio_encoder = False
freeze_depth_encoder = False
freeze_fusion_transformer = False
freeze_temporal_transformer = False

# Domain Randomization
domain_rand_enabled = True
domain_rand_curriculum = True  # Gradually increase difficulty
```

**Learning Rate Schedule:**

- **Option A (Recommended)**: Constant 1e-4 with adaptive KL
- **Option B**: Linear decay from 1e-4 to 5e-5 over training
- **Option C**: Cosine annealing with warm restarts

**Recommendation:** Start with Option A (simplest)

### Phase 5.2: Curriculum Learning Strategy

**Complexity:** LOW
**Risk:** LOW
**Dependencies:** None

**Terrain Curriculum:**

**Strategy:** Leverage existing automatic terrain difficulty adjustment

**Starting Point:**
- **Option A (Recommended)**: Start from terrain level 5 (mid-difficulty)
  - Rationale: Student already learned basics from DAgger
  - Faster convergence by skipping easy terrains
- **Option B**: Start from terrain level 0 (easy)
  - Rationale: Conservative approach, ensure stability
  - Slower but safer

**Recommendation:** Option A (start from level 5)

**Progression:**
- Use existing automatic adjustment (±1 level based on performance)
- Success threshold: >80% goal progress
- Failure threshold: <40% goal progress
- Max terrain level: 20 (same as teacher)

**Domain Randomization Curriculum:**

**Purpose:** Gradually increase augmentation intensity to avoid overwhelming the policy

**Schedule:**

| Iteration | Gaussian Noise Std | Salt-Pepper Prob | Camera Dropout | Latency Frames |
|-----------|-------------------|------------------|----------------|----------------|
| 0-1000    | 0.01              | 0.005            | 0.1            | 1              |
| 1000-3000 | 0.02              | 0.01             | 0.2            | 2              |
| 3000-5000 | 0.03              | 0.015            | 0.3            | 3              |
| 5000+     | 0.04              | 0.02             | 0.5            | 3              |

**Implementation:**
```python
class DomainRandCurriculum:
    def get_params(self, iteration):
        if iteration < 1000:
            return {"noise_std": 0.01, "dropout": 0.1, ...}
        elif iteration < 3000:
            return {"noise_std": 0.02, "dropout": 0.2, ...}
        ...
```

---

## Training Script Design

### Phase 6.1: Create Main Training Script

**Complexity:** MEDIUM
**Risk:** MEDIUM
**Dependencies:** Phases 1, 2, 3 complete

**File to Create:**
- `scripts/rsl_rl/train_student_rl_finetune.py`

**Script Structure:**

```python
# 1. Parse arguments
parser.add_argument("--dagger_checkpoint", type=str, required=True)
parser.add_argument("--freeze_encoders", action="store_true")
parser.add_argument("--domain_rand", action="store_true", default=True)
parser.add_argument("--start_terrain_level", type=int, default=5)

# 2. Initialize Isaac Lab environment
env = gym.make("Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2-v0")

# 3. Load DAgger checkpoint
student_policy = MultiModalStudentPolicy(...)
student_policy.load_state_dict(torch.load(dagger_checkpoint))

# 4. Create StudentActorCritic with value head
actor_critic = StudentActorCritic(student_policy, freeze_encoders=args.freeze_encoders)

# 5. Initialize PPO algorithm
ppo = PPOStudent(actor_critic, cfg=ppo_cfg)

# 6. Initialize domain randomization
domain_rand = DomainRandomization(cfg=domain_rand_cfg) if args.domain_rand else None

# 7. Training loop
for iteration in range(max_iterations):
    # Update domain randomization curriculum
    if domain_rand:
        domain_rand.update_curriculum(iteration)

    # Collect rollouts
    for step in range(num_steps_per_env):
        # Get observations
        obs = env.get_observations()

        # Apply domain randomization
        if domain_rand:
            obs["depth"] = domain_rand.apply(obs["depth"])

        # Sample actions
        actions = ppo.act(obs["proprio"], obs["depth"])

        # Step environment
        obs_next, rewards, dones, infos = env.step(actions)

        # Store transition
        ppo.process_env_step(rewards, dones, infos)

    # Compute returns
    last_values = ppo.get_values(obs_next["proprio"], obs_next["depth"])
    ppo.compute_returns(last_values)

    # PPO update
    train_info = ppo.update()

    # Logging
    if iteration % log_interval == 0:
        log_metrics(train_info, infos)

    # Save checkpoint
    if iteration % save_interval == 0:
        save_checkpoint(actor_critic, iteration)
```

**Key Features:**
1. Load DAgger checkpoint as initialization
2. Optional encoder freezing
3. Domain randomization with curriculum
4. Standard PPO training loop
5. Logging and checkpointing

---

## Evaluation and Validation

### Phase 7.1: Robustness Metrics

**Complexity:** LOW
**Risk:** LOW
**Dependencies:** Phase 6 complete

**Metrics to Track:**

#### Training Metrics

1. **Episode Return**:
   - Mean/std across environments
   - Track improvement over iterations

2. **Episode Length**:
   - Mean/std across environments
   - Longer episodes = better performance

3. **Goal Progress**:
   - Percentage of goals reached
   - Primary success metric

4. **Terrain Level**:
   - Current curriculum difficulty
   - Track progression

5. **Timeout Rate**:
   - Percentage of episodes ending in timeout
   - Lower = better

#### Robustness Metrics

1. **Performance Under Augmentation**:
   - Compare returns with/without domain randomization
   - Gap should decrease over training

2. **Camera Dropout Resilience**:
   - Episode return when camera offline 50% of time
   - Should remain high

3. **Noise Sensitivity**:
   - Performance degradation under high noise
   - Test with 2x training noise level

4. **Latency Tolerance**:
   - Performance with 5-frame depth delay
   - Should gracefully degrade

#### Comparison Metrics

1. **vs DAgger Baseline**:
   - Compare episode returns
   - RL fine-tuning should improve

2. **vs Teacher Policy**:
   - Compare on same terrains
   - Student may not match teacher (uses different observations)

### Phase 7.2: Test Scenarios

**Purpose:** Validate robustness improvements

**Test Scenarios:**

1. **Clean Observations** (Baseline):
   - No augmentations
   - Measure best-case performance

2. **High Noise**:
   - Gaussian noise std = 0.08 (2x training max)
   - Salt-pepper prob = 0.04 (2x training max)
   - Measure noise resilience

3. **High Latency**:
   - Depth delay = 5 frames
   - Action delay = 8 frames
   - Measure latency tolerance

4. **Camera Dropout**:
   - 70% offline probability (higher than training)
   - Measure dropout resilience

5. **Combined Stress Test**:
   - All augmentations at max intensity
   - Measure worst-case performance

**Evaluation Protocol:**

```python
def evaluate_robustness(policy, env, scenario, num_episodes=100):
    returns = []
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_return = 0

        while not done:
            # Apply scenario-specific augmentations
            obs_aug = apply_scenario(obs, scenario)

            # Get action
            action = policy.act(obs_aug)

            # Step
            obs, reward, done, info = env.step(action)
            ep_return += reward

        returns.append(ep_return)

    return {
        "mean": np.mean(returns),
        "std": np.std(returns),
        "min": np.min(returns),
        "max": np.max(returns),
    }
```

---

## Implementation Phases Summary

### Phase 1: Architecture Modifications (Week 1)
- **Complexity:** HIGH
- **Files:** 2 new files
- **Tasks:**
  - Create ValueHead module
  - Create StudentActorCritic wrapper
  - Add PPO-required methods
  - Test forward/backward passes

### Phase 2: PPO Algorithm Adaptation (Week 1-2)
- **Complexity:** HIGH
- **Files:** 2 new files
- **Tasks:**
  - Create StudentRolloutStorage
  - Create PPOStudent algorithm
  - Implement memory management
  - Test on simple environment

### Phase 3: Domain Randomization (Week 2)
- **Complexity:** MEDIUM
- **Files:** 1 new file, 1 modified
- **Tasks:**
  - Implement noise augmentations
  - Implement latency simulation
  - Implement lighting variations
  - Add curriculum scheduling

### Phase 4: Training Configuration (Week 2)
- **Complexity:** LOW
- **Files:** 1 new file
- **Tasks:**
  - Create configuration file
  - Set hyperparameters
  - Define curriculum schedule

### Phase 5: Training Script (Week 3)
- **Complexity:** MEDIUM
- **Files:** 1 new file
- **Tasks:**
  - Create main training script
  - Integrate all components
  - Add logging and checkpointing
  - Test end-to-end

### Phase 6: Evaluation (Week 3-4)
- **Complexity:** LOW
- **Files:** 1 new file (evaluation script)
- **Tasks:**
  - Implement evaluation metrics
  - Create test scenarios
  - Run robustness tests
  - Compare with baselines

---

## Risk Assessment

### High Risks

1. **Catastrophic Forgetting**:
   - **Risk:** RL fine-tuning destroys DAgger-learned representations
   - **Mitigation:**
     - Use low learning rate (1e-4)
     - Monitor DAgger loss during training
     - Option to freeze encoders
     - Early stopping if performance degrades

2. **Transformer-XL Memory Management**:
   - **Risk:** Incorrect memory handling causes training instability
   - **Mitigation:**
     - Careful testing of memory reset logic
     - Validate memory shapes at each step
     - Add assertions for memory consistency

3. **Observation Space Mismatch**:
   - **Risk:** Student uses depth, teacher uses height scan - hard to compare
   - **Mitigation:**
     - Focus on task performance, not teacher matching
     - Use environment rewards as ground truth
     - Compare on same terrains

### Medium Risks

1. **Domain Randomization Too Aggressive**:
   - **Risk:** Augmentations too strong, policy can't learn
   - **Mitigation:**
     - Use curriculum (gradual increase)
     - Monitor performance under augmentation
     - Adjust intensity if needed

2. **Hyperparameter Sensitivity**:
   - **Risk:** PPO hyperparameters not optimal for student architecture
   - **Mitigation:**
     - Start with conservative values
     - Grid search if needed
     - Monitor KL divergence

3. **Training Instability**:
   - **Risk:** Value function diverges or policy collapses
   - **Mitigation:**
     - Gradient clipping (max_grad_norm=1.0)
     - Value loss clipping
     - Monitor training metrics closely

### Low Risks

1. **Computational Cost**:
   - **Risk:** Training too slow due to image observations
   - **Mitigation:**
     - Use GPU for image encoding
     - Optimize data loading
     - Profile and optimize bottlenecks

2. **Checkpoint Compatibility**:
   - **Risk:** DAgger checkpoint format incompatible
   - **Mitigation:**
     - Test loading early
     - Add conversion script if needed

---

## Dependencies and Prerequisites

### Software Dependencies
- Isaac Lab (already installed)
- PyTorch >= 1.13
- RSL-RL library (already installed)
- WandB (optional, for logging)

### Data Dependencies
- Pre-trained DAgger student checkpoint
- Teacher policy checkpoint (for comparison)

### Hardware Requirements
- GPU with >= 16GB VRAM (for 256 environments + image observations)
- Recommended: RTX 3090 or better

---

## Success Criteria

### Minimum Success Criteria

1. **Training Stability**:
   - Training completes without crashes
   - Loss curves are smooth (no divergence)
   - Policy improves over iterations

2. **Performance Maintenance**:
   - Episode return >= 90% of DAgger baseline (clean observations)
   - Goal progress >= 70% on terrain level 10

3. **Robustness Improvement**:
   - Performance under high noise >= 80% of clean performance
   - Performance under camera dropout >= 85% of clean performance

### Stretch Goals

1. **Performance Improvement**:
   - Episode return > DAgger baseline (clean observations)
   - Reach higher terrain levels than DAgger

2. **Strong Robustness**:
   - Performance under combined stress test >= 75% of clean performance
   - Graceful degradation under extreme conditions

3. **Sim-to-Real Transfer**:
   - Policy works on real robot (requires hardware testing)

---

## Next Steps

After plan approval:

1. **Create git branch**: `feature/rl-finetuning`
2. **Implement Phase 1**: Architecture modifications
3. **Test Phase 1**: Verify forward/backward passes
4. **Implement Phase 2**: PPO algorithm
5. **Test Phase 2**: Simple environment test
6. **Implement Phase 3**: Domain randomization
7. **Implement Phase 4**: Configuration
8. **Implement Phase 5**: Training script
9. **Run training**: Monitor and adjust
10. **Evaluate**: Robustness tests
11. **Document results**: Write report

---

## Estimated Timeline

- **Phase 1 (Architecture)**: 3-4 days
- **Phase 2 (PPO)**: 4-5 days
- **Phase 3 (Domain Rand)**: 2-3 days
- **Phase 4 (Config)**: 1 day
- **Phase 5 (Training Script)**: 2-3 days
- **Phase 6 (Evaluation)**: 2-3 days
- **Buffer**: 2-3 days

**Total**: ~3-4 weeks

---

## Questions for User

Before proceeding with implementation, please confirm:

1. **Encoder Freezing**: Should we freeze encoders or fine-tune end-to-end?
   - Recommendation: Fine-tune end-to-end (Option A)

2. **Starting Terrain Level**: Start from level 5 or level 0?
   - Recommendation: Level 5 (faster convergence)

3. **Training Duration**: 10K iterations (~10M steps) sufficient?
   - Recommendation: Yes, can extend if needed

4. **Domain Randomization**: Use curriculum or fixed intensity?
   - Recommendation: Use curriculum (gradual increase)

5. **Additional Rewards**: Add robustness-specific rewards or use existing only?
   - Recommendation: Use existing only (simpler)

---

**WAITING FOR CONFIRMATION**: Proceed with this plan? (yes/no/modify)
