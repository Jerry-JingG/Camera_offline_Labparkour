"""
Keyboard command manager for injecting user commands into student policy observations.

This module provides keyboard-based control to replace waypoint navigation in DAGGER v4.0
student policy inference. It injects commands at the same observation indices that would
normally receive waypoint-derived values during training.

Observation indices (from observations.py):
- Index 6: delta_yaw (direction to waypoint) - range [-1.5, 1.5] rad
- Index 7: delta_next_yaw (direction to next waypoint) - range [-1.5, 1.5] rad
- Index 10: lin_vel_x (forward velocity command) - range [0.3, 0.8] m/s

Control Modes:
- KEYBOARD: Pure keyboard control - user has full control over direction
- HYBRID: Keyboard overrides only when keys pressed, else zeros (like autonomous)
- AUTONOMOUS: No keyboard injection, policy runs with zero privileged info
"""

from __future__ import annotations

import weakref
from enum import Enum
from typing import Optional, Tuple

import torch


class ControlMode(Enum):
    """Control mode enumeration."""
    KEYBOARD = "keyboard"      # Mode A: Pure keyboard control
    HYBRID = "hybrid"          # Mode B: Keyboard + zero fallback
    AUTONOMOUS = "autonomous"  # Mode C: No keyboard injection (zero mask)


class KeyboardCommandManager:
    """
    Manages keyboard input and generates command tensors for observation injection.

    This manager handles:
    1. Keyboard event subscription and key state tracking
    2. Command generation within training-valid ranges
    3. Smooth command transitions via exponential smoothing
    4. Multi-environment command broadcasting
    """

    # Training-time command ranges (from parkour_mdp_cfg.py)
    LIN_VEL_X_RANGE: Tuple[float, float] = (0.3, 0.8)
    DELTA_YAW_RANGE: Tuple[float, float] = (-1.5, 1.5)

    # Observation indices (from observations.py)
    IDX_DELTA_YAW: int = 6
    IDX_DELTA_NEXT_YAW: int = 7
    IDX_LIN_VEL_X: int = 10

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        control_mode: ControlMode = ControlMode.KEYBOARD,
        smoothing_factor: float = 0.3,
        default_forward_speed: float = 0.5,
        yaw_rate_scale: float = 1.0,
    ) -> None:
        """
        Initialize the keyboard command manager.

        Args:
            num_envs: Number of parallel environments
            device: Torch device for tensors
            control_mode: Initial control mode
            smoothing_factor: Exponential smoothing factor (0=no smoothing, 1=instant)
            default_forward_speed: Default forward speed when W is pressed (m/s)
            yaw_rate_scale: Scale factor for yaw rate (1.0 = full range)
        """
        self.num_envs = num_envs
        self.device = device
        self.control_mode = control_mode
        self.smoothing_factor = smoothing_factor
        self.default_forward_speed = default_forward_speed
        self.yaw_rate_scale = yaw_rate_scale

        # Command state tensors
        self.target_lin_vel_x = torch.zeros(num_envs, device=device)
        self.target_delta_yaw = torch.zeros(num_envs, device=device)
        self.current_lin_vel_x = torch.zeros(num_envs, device=device)
        self.current_delta_yaw = torch.zeros(num_envs, device=device)

        # Key state flags
        self._key_forward = False
        self._key_backward = False
        self._key_turn_left = False
        self._key_turn_right = False
        self._any_key_pressed = False

        # Control target (which env to control, None = all)
        self._controlled_env_id: Optional[int] = 0

        # Keyboard interface (initialized in setup_keyboard)
        self._input = None
        self._keyboard = None
        self._keyboard_sub = None
        self._keyboard_poll_fn = None

        # Key code cache
        self._key_codes = {}

        # Statistics
        self.step_count = 0

    def setup_keyboard(self) -> None:
        """
        Set up keyboard event subscription.

        IMPORTANT: Must be called AFTER AppLauncher is initialized,
        as carb/omni modules require the Isaac Sim runtime.
        """
        import carb
        import omni.appwindow

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()

        if self._keyboard is None:
            print("[WARN] KeyboardCommandManager: Keyboard device not found.")
            return

        # Try to find polling function for robust key detection
        for candidate in ("get_keyboard_value", "get_keyboard_button"):
            if hasattr(self._input, candidate):
                self._keyboard_poll_fn = getattr(self._input, candidate)
                break

        # Resolve key codes
        self._resolve_key_codes(carb)

        # Subscribe to keyboard events
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        print(f"[INFO] KeyboardCommandManager initialized:")
        print(f"       Control mode: {self.control_mode.value}")
        print(f"       lin_vel_x range: {self.LIN_VEL_X_RANGE}")
        print(f"       delta_yaw range: {self.DELTA_YAW_RANGE}")
        print(f"       Smoothing factor: {self.smoothing_factor}")

    def _resolve_key_codes(self, carb) -> None:
        """Resolve keyboard input enum values."""
        keyboard_enum = getattr(carb.input, "KeyboardInput", None)
        if keyboard_enum is None:
            return

        key_mappings = {
            "W": ("W", "KEY_W"),
            "S": ("S", "KEY_S"),
            "A": ("A", "KEY_A"),
            "D": ("D", "KEY_D"),
            "UP": ("UP", "UP_ARROW", "ARROW_UP"),
            "DOWN": ("DOWN", "DOWN_ARROW", "ARROW_DOWN"),
            "LEFT": ("LEFT", "LEFT_ARROW", "ARROW_LEFT"),
            "RIGHT": ("RIGHT", "RIGHT_ARROW", "ARROW_RIGHT"),
            "SPACE": ("SPACE", "KEY_SPACE"),
            "M": ("M", "KEY_M"),
            "1": ("KEY_1", "1"),
            "2": ("KEY_2", "2"),
            "3": ("KEY_3", "3"),
            "4": ("KEY_4", "4"),
            "5": ("KEY_5", "5"),
            "6": ("KEY_6", "6"),
            "7": ("KEY_7", "7"),
            "8": ("KEY_8", "8"),
            "9": ("KEY_9", "9"),
            "0": ("KEY_0", "0"),
        }

        for key_name, candidates in key_mappings.items():
            for candidate in candidates:
                if hasattr(keyboard_enum, candidate):
                    self._key_codes[key_name] = getattr(keyboard_enum, candidate)
                    break

    def _poll_key(self, key_name: str) -> bool:
        """Poll current state of a key."""
        if self._keyboard_poll_fn is None or key_name not in self._key_codes:
            return False
        try:
            return bool(self._keyboard_poll_fn(self._keyboard, self._key_codes[key_name]))
        except Exception:
            return False

    def _on_keyboard_event(self, event, *args, **kwargs) -> None:
        """Handle keyboard events."""
        import carb

        key_name = getattr(event.input, "name", str(event.input))
        event_type = getattr(event, "type", None)

        if event_type not in (
            carb.input.KeyboardEventType.KEY_PRESS,
            carb.input.KeyboardEventType.KEY_RELEASE,
        ):
            return

        is_pressed = event_type == carb.input.KeyboardEventType.KEY_PRESS

        # Movement keys
        if key_name in ("W", "UP"):
            self._key_forward = is_pressed
        elif key_name in ("S", "DOWN"):
            self._key_backward = is_pressed
        elif key_name in ("A", "LEFT", "LEFT_ARROW"):
            self._key_turn_left = is_pressed
        elif key_name in ("D", "RIGHT", "RIGHT_ARROW"):
            self._key_turn_right = is_pressed

        # Control keys (only on press)
        elif is_pressed:
            if key_name == "SPACE":
                self._emergency_stop()
            elif key_name == "M":
                self._cycle_control_mode()
            elif key_name in "0123456789":
                env_id = int(key_name)
                if env_id < self.num_envs:
                    self._controlled_env_id = env_id if env_id > 0 else None
                    ctrl_str = "all" if self._controlled_env_id is None else str(self._controlled_env_id)
                    print(f"[INFO] Now controlling env: {ctrl_str}")

        self._update_any_key_pressed()

    def _update_any_key_pressed(self) -> None:
        """Update flag indicating if any movement key is pressed."""
        self._any_key_pressed = (
            self._key_forward or self._key_backward or
            self._key_turn_left or self._key_turn_right
        )

    def _emergency_stop(self) -> None:
        """Stop all movement immediately."""
        self.target_lin_vel_x.zero_()
        self.target_delta_yaw.zero_()
        self.current_lin_vel_x.zero_()
        self.current_delta_yaw.zero_()
        print("[INFO] Emergency stop activated")

    def _cycle_control_mode(self) -> None:
        """Cycle through control modes."""
        modes = list(ControlMode)
        current_idx = modes.index(self.control_mode)
        next_idx = (current_idx + 1) % len(modes)
        self.control_mode = modes[next_idx]
        print(f"[INFO] Control mode changed to: {self.control_mode.value}")

    def update_from_polling(self) -> None:
        """Update key states from polling (more reliable than events)."""
        if self._keyboard_poll_fn is None:
            return

        self._key_forward = self._poll_key("W") or self._poll_key("UP")
        self._key_backward = self._poll_key("S") or self._poll_key("DOWN")
        self._key_turn_left = self._poll_key("A") or self._poll_key("LEFT")
        self._key_turn_right = self._poll_key("D") or self._poll_key("RIGHT")
        self._update_any_key_pressed()

    def compute_target_commands(self) -> None:
        """Compute target commands from current key states."""
        # Forward/backward velocity
        if self._key_forward and not self._key_backward:
            vel_x = self.default_forward_speed
        elif self._key_backward and not self._key_forward:
            vel_x = self.LIN_VEL_X_RANGE[0]  # Minimum forward (slow creep)
        else:
            vel_x = 0.0

        # Turn left/right (yaw)
        # Positive yaw = turn left (counter-clockwise)
        if self._key_turn_left and not self._key_turn_right:
            yaw = self.DELTA_YAW_RANGE[1] * self.yaw_rate_scale
        elif self._key_turn_right and not self._key_turn_left:
            yaw = self.DELTA_YAW_RANGE[0] * self.yaw_rate_scale
        else:
            yaw = 0.0

        # Apply to controlled environment(s)
        if self._controlled_env_id is not None:
            self.target_lin_vel_x[self._controlled_env_id] = vel_x
            self.target_delta_yaw[self._controlled_env_id] = yaw
        else:
            # Control all environments
            self.target_lin_vel_x.fill_(vel_x)
            self.target_delta_yaw.fill_(yaw)

    def apply_smoothing(self) -> None:
        """Apply exponential smoothing to commands."""
        alpha = self.smoothing_factor
        self.current_lin_vel_x = (
            alpha * self.target_lin_vel_x + (1 - alpha) * self.current_lin_vel_x
        )
        self.current_delta_yaw = (
            alpha * self.target_delta_yaw + (1 - alpha) * self.current_delta_yaw
        )

    def step(self) -> None:
        """
        Update commands for this timestep.
        Call this once per simulation step before inject_commands().
        """
        self.update_from_polling()
        self.compute_target_commands()
        self.apply_smoothing()
        self.step_count += 1

    def inject_commands(
        self,
        obs_prop: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inject keyboard commands into observation tensor.

        This replaces the waypoint-derived delta_yaw values with keyboard commands,
        ensuring consistency with the DAGGER training architecture.

        Args:
            obs_prop: Proprioceptive observation tensor [num_envs, proprio_dim]

        Returns:
            Modified observation tensor with injected commands
        """
        obs_prop = obs_prop.clone()

        if self.control_mode == ControlMode.AUTONOMOUS:
            # No injection - zero out privileged indices (same as original play_student.py)
            obs_prop[:, self.IDX_DELTA_YAW] = 0.0
            obs_prop[:, self.IDX_DELTA_NEXT_YAW] = 0.0
            return obs_prop

        if self.control_mode == ControlMode.KEYBOARD:
            # Pure keyboard control - inject commands at all times
            obs_prop[:, self.IDX_DELTA_YAW] = self.current_delta_yaw
            obs_prop[:, self.IDX_DELTA_NEXT_YAW] = self.current_delta_yaw
            obs_prop[:, self.IDX_LIN_VEL_X] = self.current_lin_vel_x

        elif self.control_mode == ControlMode.HYBRID:
            # Keyboard overrides only when keys are pressed
            if self._any_key_pressed:
                if self._controlled_env_id is not None:
                    idx = self._controlled_env_id
                    obs_prop[idx, self.IDX_DELTA_YAW] = self.current_delta_yaw[idx]
                    obs_prop[idx, self.IDX_DELTA_NEXT_YAW] = self.current_delta_yaw[idx]
                    obs_prop[idx, self.IDX_LIN_VEL_X] = self.current_lin_vel_x[idx]
                else:
                    obs_prop[:, self.IDX_DELTA_YAW] = self.current_delta_yaw
                    obs_prop[:, self.IDX_DELTA_NEXT_YAW] = self.current_delta_yaw
                    obs_prop[:, self.IDX_LIN_VEL_X] = self.current_lin_vel_x
            else:
                # No keys pressed - zero out privileged info
                obs_prop[:, self.IDX_DELTA_YAW] = 0.0
                obs_prop[:, self.IDX_DELTA_NEXT_YAW] = 0.0

        return obs_prop

    def reset_env(self, done_mask: torch.Tensor) -> None:
        """Reset command state for environments that terminated."""
        if not done_mask.any():
            return
        self.current_lin_vel_x[done_mask] = 0.0
        self.current_delta_yaw[done_mask] = 0.0
        self.target_lin_vel_x[done_mask] = 0.0
        self.target_delta_yaw[done_mask] = 0.0

    def get_status_string(self) -> str:
        """Get status string for logging."""
        env_id = self._controlled_env_id if self._controlled_env_id is not None else 0
        keys_str = "".join([
            "W" if self._key_forward else "_",
            "S" if self._key_backward else "_",
            "A" if self._key_turn_left else "_",
            "D" if self._key_turn_right else "_",
        ])
        ctrl_str = "all" if self._controlled_env_id is None else str(self._controlled_env_id)
        return (
            f"mode={self.control_mode.value} "
            f"env={ctrl_str} "
            f"vel_x={self.current_lin_vel_x[env_id]:.2f} "
            f"yaw={self.current_delta_yaw[env_id]:.2f} "
            f"keys={keys_str}"
        )

    def __del__(self):
        """Clean up keyboard subscription."""
        if self._input is not None and self._keyboard_sub is not None:
            try:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            except Exception:
                pass
