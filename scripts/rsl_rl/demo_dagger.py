"""
Demo runner for DAGGER-trained MultiModalStudentPolicy.

功能：
1) 启动 IsaacLab 跑酷环境（TeacherCam/Student Play 任务均可）。
2) 加载 DAGGER 学生策略 checkpoint（MultiModalStudentPolicy + TXL）。
3) 支持 Omniverse 视角切换/第三人称跟随，以及手柄/键盘控制命令输入。
4) 持续推理学生策略，驱动机器人在仿真中跑酷。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import weakref
from pathlib import Path

# 允许从项目根目录 import 本地包
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import cli_args  # isort: skip
from isaaclab.app import AppLauncher


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play DAGGER student policy with IsaacLab Parkour demo UI.")
    parser.add_argument("--task", type=str, required=True, help="Isaac task name (e.g., TeacherCam/Student Play).")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs to simulate.")
    parser.add_argument("--student_checkpoint", type=str, required=True, help="Path to student_epoch_*.pt checkpoint.")
    parser.add_argument("--prop_hist_len", type=int, default=3, help="History length for proprio tokens.")
    parser.add_argument("--depth_hist_len", type=int, default=4, help="History length for depth tokens.")
    parser.add_argument("--sequence_length", type=int, default=64, help="Transformer-XL mem length during play.")
    parser.add_argument("--max_steps", type=int, default=2000, help="Max simulation steps; 0 for no limit.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric (use USD I/O).")
    parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
    parser.add_argument(
        "--input_device",
        type=str,
        choices=["gamepad", "keyboard"],
        default="gamepad",
        help="Command input source: gamepad or keyboard.",
    )
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


# 先解析命令行并启动 Isaac App（omni 相关模块需要在 App 启动后 import）
parser = _make_parser()
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import carb  # noqa: E402
import omni  # noqa: E402
from omni.kit.viewport.utility import get_viewport_from_window_name  # noqa: E402
from omni.kit.viewport.utility.camera_state import ViewportCameraState  # noqa: E402
from pxr import Gf, Sdf  # noqa: E402
from play_dagger import StudentOnlineRunner, load_student_policy_for_play  # noqa: E402
from vecenv_wrapper import ParkourRslRlVecEnvWrapper  # noqa: E402
from parkour_isaaclab.envs import ParkourManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from parkour_tasks.extreme_parkour_task.config.go2.parkour_student_cfg import (  # noqa: E402
    UnitreeGo2StudentParkourEnvCfg_PLAY,
)
from parkour_tasks.extreme_parkour_task.config.go2.parkour_teacher_cfg import (  # noqa: E402
    UnitreeGo2TeacherParkourEnvCfg_PLAY,
)
from parkour_tasks.extreme_parkour_task.config.go2.parkour_teacher_cam_cfg import (  # noqa: E402
    UnitreeGo2TeacherCamParkourEnvCfg_PLAY,
)


class ParkourDemoDaggerGO2:
    """Demo wrapper that combines DAGGER student policy + IsaacLab viewport controls."""

    def __init__(self) -> None:
        # 1) 构建 env cfg（TeacherCam / Student / Teacher）
        if "Student" in args_cli.task:
            env_cfg = UnitreeGo2StudentParkourEnvCfg_PLAY()
        elif "TeacherCam" in args_cli.task:
            env_cfg = UnitreeGo2TeacherCamParkourEnvCfg_PLAY()
        else:
            env_cfg = UnitreeGo2TeacherParkourEnvCfg_PLAY()
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.episode_length_s = 1_000_000
        env_cfg.curriculum = None
        self.env_cfg = env_cfg

        # 2) 创建环境与 wrapper
        self.env = ParkourRslRlVecEnvWrapper(ParkourManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device

        # 3) 加载学生策略 checkpoint
        ckpt_path = Path(args_cli.student_checkpoint).expanduser().resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Student checkpoint not found: {ckpt_path}")

        # 先 reset 一次拿到 obs 以确定维度
        obs0, extras0 = self.env.reset()
        depth0 = extras0["observations"].get("depth_camera")
        if depth0 is None:
            raise RuntimeError("当前任务未输出 depth_camera 观测，请确认使用带相机的任务（例如 TeacherCam Play 版）。")

        student_model, meta = load_student_policy_for_play(
            checkpoint_path=ckpt_path,
            prop_hist_len=args_cli.prop_hist_len,
            depth_hist_len=args_cli.depth_hist_len,
            device=self.device,
        )
        proprio_dim = int(meta["num_prop"])
        camera_resolution = tuple(meta.get("camera_resolution", [depth0.shape[-2], depth0.shape[-1]]))

        self.runner = StudentOnlineRunner(
            model=student_model,
            num_envs=self.env.num_envs,
            proprio_dim=proprio_dim,
            prop_hist_len=args_cli.prop_hist_len,
            depth_hist_len=args_cli.depth_hist_len,
            camera_resolution=camera_resolution,  # type: ignore[arg-type]
            device=self.device,
            sequence_length=args_cli.sequence_length,
        )
        self.proprio_dim = proprio_dim
        self.command_obs_index = self._infer_command_obs_index(obs0)
        self.runner.reset()

        # 4) 设置第三人称相机 & 手柄控制
        self.create_camera()
        self.commands = torch.zeros(env_cfg.scene.num_envs, 3, device=self.device)
        self.commands[:, :] = self.env.unwrapped.command_manager.get_command("base_velocity")
        self.yaw_offsets = torch.zeros(env_cfg.scene.num_envs, device=self.device)
        self._prim_selection = omni.usd.get_context().get_selection()
        self._selected_id: int | None = None
        self._previous_selected_id: int | None = None
        # Follow camera offset in robot base frame: behind (-x), slightly above (+z).
        self._camera_local_transform = torch.tensor([-2.6, 0.0, 1.6], device=self.device)
        self._keyboard_move_forward = False
        self._keyboard_move_backward = False
        self._keyboard_turn_left = False
        self._keyboard_turn_right = False
        self._prev_key_c_down = False
        self._prev_key_escape_down = False
        self._prev_key_space_down = False
        self.keyboard_event_count = 0
        self.last_keyboard_event: str | None = None
        self._input_device = getattr(args_cli, "input_device", "gamepad")
        print(f"[INFO] Demo input_device={self._input_device}")
        self.set_up_input()

    # ------------------------------ Camera & Input ------------------------------
    def create_camera(self) -> None:
        """Create a third-person camera and keep default perspective view."""
        stage = omni.usd.get_context().get_stage()
        self.viewport = get_viewport_from_window_name("Viewport")
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))
        self.viewport.set_active_camera(self.perspective_path)

    def _infer_command_obs_index(self, obs0: torch.Tensor) -> int:
        """Heuristically infer which proprio dim corresponds to base_velocity x command."""
        with torch.no_grad():
            base_cmd = self.env.unwrapped.command_manager.get_command("base_velocity")[:, 0]
            base_cmd = base_cmd.to(obs0.device)
            obs_prop0 = obs0[:, : self.proprio_dim]
            num_dims = obs_prop0.shape[1]
            diffs = []
            for idx in range(num_dims):
                col = obs_prop0[:, idx]
                diffs.append(torch.mean(torch.abs(col - base_cmd)).item())
            best_idx = int(torch.tensor(diffs).argmin().item())
        print(f"[INFO] Inferred command obs index: {best_idx}")
        return best_idx

    def set_up_input(self) -> None:
        """Initialize input device subscription (gamepad or keyboard)."""
        if self._input_device == "gamepad":
            print("[INFO] Setting up gamepad input.")
            self.set_up_gamepad()
        elif self._input_device == "keyboard":
            print("[INFO] Setting up keyboard input.")
            self.set_up_keyboard()
        else:
            raise ValueError(f"Unsupported input_device: {self._input_device}")

    def set_up_gamepad(self) -> None:
        """Subscribe to gamepad events and map left stick to velocity commands."""
        self._input = carb.input.acquire_input_interface()
        self._gamepad = omni.appwindow.get_default_app_window().get_gamepad(0)
        self._gamepad_sub = self._input.subscribe_to_gamepad_events(
            self._gamepad,
            lambda event, *args, obj=weakref.proxy(self): obj._on_gamepad_event(event, *args),
        )
        self.dead_zone = 0.01
        self.v_x_sensitivity = 0.8
        self.v_y_sensitivity = 0.8
        self._INPUT_STICK_SPEED_MAPPING = {
            "LEFT_STICK_UP": self.env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            "LEFT_STICK_DOWN": self.env_cfg.commands.base_velocity.ranges.lin_vel_x[0],
        }
        self._INPUT_STICK_YAW_MAPPING = {
            # Positive yaw offset means "turn left" (consistent with delta_yaw sign).
            "LEFT_STICK_LEFT": self.env_cfg.commands.base_velocity.ranges.heading[1],
            "LEFT_STICK_RIGHT": self.env_cfg.commands.base_velocity.ranges.heading[0],
        }

    def _on_gamepad_event(self, event) -> None:
        input_obj = getattr(event, "input", None)
        input_name = getattr(input_obj, "name", input_obj)
        if input_name is None:
            return

        # Disconnection: stop the selected robot.
        if getattr(event, "type", None) == carb.input.GamepadConnectionEventType.DISCONNECTED:
            if self._selected_id is not None:
                self.commands[self._selected_id] = torch.zeros(3, device=self.device)
                self.yaw_offsets[self._selected_id] = 0.0
            return

        cur_val = float(getattr(event, "value", 0.0))
        if abs(cur_val) < self.dead_zone:
            cur_val = 0.0
        cur_mag = abs(cur_val)

        if input_name in self._INPUT_STICK_SPEED_MAPPING:
            if self._selected_id is not None:
                value = float(self._INPUT_STICK_SPEED_MAPPING[input_name])
                # Only inject forward speed command (matches training-time command injection position).
                self.commands[self._selected_id, 0] = value * cur_mag
        elif input_name in self._INPUT_STICK_YAW_MAPPING:
            if self._selected_id is not None:
                yaw_value = float(self._INPUT_STICK_YAW_MAPPING[input_name])
                self.yaw_offsets[self._selected_id] = yaw_value * cur_mag
        elif input_name == "LEFT_SHOULDER":
            self._prim_selection.clear_selected_prim_paths()
        elif input_name == "RIGHT_SHOULDER":
            if self._selected_id is not None:
                if self.viewport.get_active_camera() == self.camera_path:
                    self.viewport.set_active_camera(self.perspective_path)
                else:
                    self.viewport.set_active_camera(self.camera_path)

    def set_up_keyboard(self) -> None:
        """Subscribe to keyboard events and map WASD/arrow keys to velocity commands."""
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        if self._keyboard is None:
            print("[WARN] Keyboard device not found; keyboard control will be disabled.")
            return
        self._keyboard_poll_fn = None
        self._keyboard_poll_fn_name = None
        for candidate in ("get_keyboard_value", "get_keyboard_button"):
            if hasattr(self._input, candidate):
                self._keyboard_poll_fn = getattr(self._input, candidate)
                self._keyboard_poll_fn_name = candidate
                break
        if self._keyboard_poll_fn_name is not None:
            print(f"[INFO] Keyboard polling enabled via input.{self._keyboard_poll_fn_name}()")
        else:
            print("[WARN] Keyboard polling API not found; relying on event callbacks only.")

        self._key_w = self._resolve_keyboard_input("W", "KEY_W")
        self._key_s = self._resolve_keyboard_input("S", "KEY_S")
        self._key_a = self._resolve_keyboard_input("A", "KEY_A")
        self._key_d = self._resolve_keyboard_input("D", "KEY_D")
        self._key_up = self._resolve_keyboard_input("UP", "UP_ARROW", "ARROW_UP")
        self._key_down = self._resolve_keyboard_input("DOWN", "DOWN_ARROW", "ARROW_DOWN")
        self._key_left = self._resolve_keyboard_input("LEFT", "LEFT_ARROW", "ARROW_LEFT")
        self._key_right = self._resolve_keyboard_input("RIGHT", "RIGHT_ARROW", "ARROW_RIGHT")
        self._key_space = self._resolve_keyboard_input("SPACE", "KEY_SPACE")
        self._key_escape = self._resolve_keyboard_input("ESCAPE", "KEY_ESCAPE", "ESC")
        self._key_c = self._resolve_keyboard_input("C", "KEY_C")

        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        print(f"[INFO] Keyboard event subscription: {self._keyboard_sub}")

    def _resolve_keyboard_input(self, *names: str):
        """Resolve carb.input.KeyboardInput enum entry by trying multiple names."""
        keyboard_enum = getattr(carb.input, "KeyboardInput", None)
        if keyboard_enum is None:
            return None
        for name in names:
            if hasattr(keyboard_enum, name):
                return getattr(keyboard_enum, name)
        return None

    def _poll_key_down(self, key_code) -> bool:
        """Return True if key is down (polling), else False."""
        if key_code is None or self._keyboard is None or self._keyboard_poll_fn is None:
            return False
        try:
            return bool(self._keyboard_poll_fn(self._keyboard, key_code))
        except Exception:
            return False

    def _update_keyboard_state_from_polling(self) -> None:
        """Update key flags by polling current keyboard state (works even if callbacks are swallowed)."""
        if self._keyboard_poll_fn is None:
            return
        self._keyboard_move_forward = self._poll_key_down(self._key_w) or self._poll_key_down(self._key_up)
        self._keyboard_move_backward = self._poll_key_down(self._key_s) or self._poll_key_down(self._key_down)
        self._keyboard_turn_left = self._poll_key_down(self._key_a) or self._poll_key_down(self._key_left)
        self._keyboard_turn_right = self._poll_key_down(self._key_d) or self._poll_key_down(self._key_right)

        # Edge-triggered actions
        cur_c_down = self._poll_key_down(self._key_c)
        if cur_c_down and not self._prev_key_c_down:
            if self._selected_id is not None:
                if self.viewport.get_active_camera() == self.camera_path:
                    self.viewport.set_active_camera(self.perspective_path)
                else:
                    self.viewport.set_active_camera(self.camera_path)
        self._prev_key_c_down = cur_c_down

        cur_escape_down = self._poll_key_down(self._key_escape)
        if cur_escape_down and not self._prev_key_escape_down:
            self._prim_selection.clear_selected_prim_paths()
        self._prev_key_escape_down = cur_escape_down

        cur_space_down = self._poll_key_down(self._key_space)
        if cur_space_down and not self._prev_key_space_down:
            if self._selected_id is not None:
                self.commands[self._selected_id, 0] = 0.0
                self.yaw_offsets[self._selected_id] = 0.0
        self._prev_key_space_down = cur_space_down

    def _apply_keyboard_command(self) -> None:
        self._update_keyboard_state_from_polling()
        if self._selected_id is None:
            return
        lin_vel_min, lin_vel_max = self.env_cfg.commands.base_velocity.ranges.lin_vel_x
        heading_min, heading_max = self.env_cfg.commands.base_velocity.ranges.heading
        if self._keyboard_move_forward and not self._keyboard_move_backward:
            self.commands[self._selected_id, 0] = float(lin_vel_max)
        elif self._keyboard_move_backward and not self._keyboard_move_forward:
            # Some tasks only sample positive forward speeds; treat "backward" as slow/creep by default.
            self.commands[self._selected_id, 0] = float(lin_vel_min)
        else:
            # No directional key: stop (or let the policy decide).
            self.commands[self._selected_id, 0] = 0.0

        if self._keyboard_turn_left and not self._keyboard_turn_right:
            self.yaw_offsets[self._selected_id] = float(heading_max)
        elif self._keyboard_turn_right and not self._keyboard_turn_left:
            self.yaw_offsets[self._selected_id] = float(heading_min)
        else:
            self.yaw_offsets[self._selected_id] = 0.0

    def _on_keyboard_event(self, event, *args, **kwargs) -> None:
        key_name = getattr(event.input, "name", str(event.input))
        event_type = getattr(event, "type", None)
        self.keyboard_event_count += 1
        self.last_keyboard_event = f"{event_type}:{key_name}@{time.time():.3f}"

        # Only handle key press/release; other event types ignored.
        if event_type not in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_RELEASE):
            return

        is_pressed = event_type == carb.input.KeyboardEventType.KEY_PRESS

        # Camera / selection controls
        if is_pressed and key_name in ("ESCAPE", "ESC"):
            self._prim_selection.clear_selected_prim_paths()
            return
        if is_pressed and key_name in ("C",):
            if self._selected_id is not None:
                if self.viewport.get_active_camera() == self.camera_path:
                    self.viewport.set_active_camera(self.perspective_path)
                else:
                    self.viewport.set_active_camera(self.camera_path)
            return

        # Velocity command controls
        if key_name in ("W", "UP"):
            self._keyboard_move_forward = is_pressed
        elif key_name in ("S", "DOWN"):
            self._keyboard_move_backward = is_pressed
        elif key_name in ("A", "LEFT", "LEFT_ARROW"):
            self._keyboard_turn_left = is_pressed
        elif key_name in ("D", "RIGHT", "RIGHT_ARROW"):
            self._keyboard_turn_right = is_pressed
        elif is_pressed and key_name in ("SPACE",):
            if self._selected_id is not None:
                self.commands[self._selected_id, 0] = 0.0
                self.yaw_offsets[self._selected_id] = 0.0
            return
        else:
            return

        self._apply_keyboard_command()

    def update_selected_object(self) -> None:
        """Handle selection change and update follow-camera."""
        self._previous_selected_id = self._selected_id
        selected_prim_paths = self._prim_selection.get_selected_prim_paths()
        if len(selected_prim_paths) == 0:
            self._selected_id = None
            self.viewport.set_active_camera(self.perspective_path)
        elif len(selected_prim_paths) > 1:
            print("Multiple prims are selected. Please only select one!")
        else:
            prim_splitted_path = selected_prim_paths[0].split("/")
            if len(prim_splitted_path) >= 4 and prim_splitted_path[3][0:4] == "env_":
                self._selected_id = int(prim_splitted_path[3][4:])
                if self._previous_selected_id != self._selected_id:
                    self.viewport.set_active_camera(self.camera_path)
                self._update_camera()
            else:
                print("The selected prim was not a GO2 robot")

        if self._previous_selected_id is not None and self._previous_selected_id != self._selected_id:
            self.env.unwrapped.command_manager.reset([self._previous_selected_id])
            self.commands[:, :] = self.env.unwrapped.command_manager.get_command("base_velocity")
            self.yaw_offsets[self._previous_selected_id] = 0.0

    def _update_camera(self) -> None:
        """Third-person camera follows the selected robot."""
        base_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[self._selected_id, :]
        base_quat = self.env.unwrapped.scene["robot"].data.root_quat_w[self._selected_id, :]
        camera_pos = quat_apply(base_quat, self._camera_local_transform) + base_pos

        camera_state = ViewportCameraState(self.camera_path, self.viewport)
        eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
        target = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.6)
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)


def main() -> None:
    demo = ParkourDemoDaggerGO2()
    obs, extras = demo.env.reset()
    step = 0

    while simulation_app.is_running() and (args_cli.max_steps == 0 or step < args_cli.max_steps):
        demo.update_selected_object()
        if demo._input_device == "keyboard":
            demo._apply_keyboard_command()
        with torch.inference_mode():
            # 将手柄产生的命令写回 obs（保持与训练时指令注入位置一致）
            if obs.shape[1] > 7:
                # Apply user yaw bias to delta_yaw / delta_next_yaw channels.
                obs[:, 6:8] = torch.clamp(obs[:, 6:8] + demo.yaw_offsets[:, None], min=-math.pi, max=math.pi)
            if obs.shape[1] > demo.command_obs_index:
                obs[:, demo.command_obs_index] = demo.commands[:, 0]

            depth_image = extras["observations"].get("depth_camera")
            if depth_image is None:
                raise RuntimeError("当前任务未输出 depth_camera 观测，请确认使用带相机的任务。")

            obs_prop = obs[:, : demo.proprio_dim]
            actions = demo.runner.act(obs_prop.to(demo.device), depth_image.to(demo.device))

            obs, _, dones, extras = demo.env.step(actions)
            done_mask = dones.view(-1).bool() if hasattr(dones, "view") else torch.as_tensor(dones).view(-1).bool()
            if done_mask.any():
                demo.runner.reset_done(done_mask)

            # 每隔一定步数打印一次：指令前向速度 vs 当前前向速度
            if step % 30 == 0:
                env_idx = demo._selected_id if demo._selected_id is not None else 0
                try:
                    cmd_vx = float(demo.commands[env_idx, 0].item())
                except Exception:
                    cmd_vx = float("nan")
                try:
                    robot = demo.env.unwrapped.scene["robot"]
                    cur_vx = float(robot.data.root_lin_vel_b[env_idx, 0].item())
                except Exception:
                    cur_vx = float("nan")
                print(
                    f"[cmd_vs_vel] step={step} env={env_idx} cmd_vx={cmd_vx:.3f} cur_vx={cur_vx:.3f} "
                    f"kb_events={demo.keyboard_event_count} last_kb={demo.last_keyboard_event}"
                )

            step += 1

    demo.env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
