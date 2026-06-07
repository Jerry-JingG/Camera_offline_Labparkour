from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch


DEFAULT_CAMERA_OFFLINE_ROOT = "/home/jing/Camera_offline_Labparkour_new"
DEFAULT_STUDENT_CHECKPOINT = "outputs/students/train_from_dataset/xl0513/student_final.pt"


def _insert_import_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_root(camera_offline_root: str | Path) -> Path:
    root = Path(camera_offline_root).expanduser().resolve()
    txl_root = root / "scripts" / "txl_student"
    if not txl_root.is_dir():
        raise FileNotFoundError(f"Camera-offline TXL source not found: {txl_root}")
    _insert_import_path(root)
    _insert_import_path(txl_root)
    return root


def _load_student_utils(camera_offline_root: str | Path):
    root = _resolve_root(camera_offline_root)
    utils_path = root / "scripts" / "txl_student" / "utils" / "student_utils.py"
    if not utils_path.is_file():
        raise FileNotFoundError(f"TXL student utils not found: {utils_path}")
    return _load_module_from_path("_isaac_camera_offline_txl_student_utils", utils_path)


def _load_dropout_module(camera_offline_root: str | Path):
    root = _resolve_root(camera_offline_root)
    dropout_path = root / "scripts" / "txl_student" / "utils" / "dropout_manager.py"
    if not dropout_path.is_file():
        raise FileNotFoundError(f"TXL dropout manager not found: {dropout_path}")
    return _load_module_from_path("_isaac_camera_offline_txl_dropout_manager", dropout_path)


def _resolve_checkpoint(checkpoint_path: str | Path, camera_offline_root: str | Path) -> Path:
    path = Path(checkpoint_path).expanduser()
    if not path.is_absolute():
        path = Path(camera_offline_root).expanduser().resolve() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Camera-offline student checkpoint not found: {path}")
    return path


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)
    except Exception:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected checkpoint payload type: {type(payload)!r}")
    return payload


def _get_state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model_state_dict", "student_state_dict", "actor_state_dict", "state_dict"):
        state_dict = payload.get(key)
        if state_dict is not None:
            return state_dict
    raise KeyError("Student checkpoint missing model_state_dict")


def _meta_int(meta: dict[str, Any], key: str, default: int) -> int:
    return int(meta.get(key, default))


class CameraOfflineStudentRunner:
    """Online wrapper for the latest Camera_offline TXL student checkpoint."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        camera_offline_root: str | Path = DEFAULT_CAMERA_OFFLINE_ROOT,
        device: str | torch.device = "cuda:0",
        num_envs: int = 1,
        mem_len: int | None = None,
    ) -> None:
        self.camera_offline_root = _resolve_root(camera_offline_root)
        self.checkpoint_path = _resolve_checkpoint(checkpoint_path, self.camera_offline_root)
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")

        student_utils = _load_student_utils(self.camera_offline_root)
        payload = _load_checkpoint(self.checkpoint_path, self.device)
        self.meta: dict[str, Any] = dict(payload.get("meta", {}))

        self.num_prop = _meta_int(self.meta, "num_prop", 53)
        self.action_dim = _meta_int(self.meta, "action_dim", 12)
        camera_resolution = self.meta.get("camera_resolution", (58, 87))
        self.camera_resolution = tuple(int(v) for v in camera_resolution)
        self.prop_hist_len = _meta_int(self.meta, "prop_hist_len", 1)
        self.depth_hist_len = _meta_int(self.meta, "depth_hist_len", 1)
        checkpoint_mem_len = int(self.meta.get("sequence_length", self.meta.get("mem_len", 64)))
        self.sequence_length = int(mem_len) if mem_len is not None and int(mem_len) > 0 else checkpoint_mem_len
        if self.sequence_length <= 0:
            raise ValueError(f"mem_len must be positive, got {self.sequence_length}")

        if self.num_prop != 53:
            raise ValueError(f"Expected camera-offline num_prop=53, got {self.num_prop}")
        if self.action_dim != 12:
            raise ValueError(f"Expected camera-offline action_dim=12, got {self.action_dim}")
        if self.camera_resolution != (58, 87):
            raise ValueError(
                f"Expected camera-offline depth resolution (58, 87), got {self.camera_resolution}"
            )

        self.model = student_utils.build_student_model(
            proprio_dim=self.num_prop,
            action_dim=self.action_dim,
            camera_resolution=self.camera_resolution,
            prop_hist_len=self.prop_hist_len,
            depth_hist_len=self.depth_hist_len,
            mem_len=self.sequence_length,
            token_dim=128,
        )
        self.model.load_state_dict(_get_state_dict(payload))
        self.model.to(self.device)
        self.model.eval()

        self.prop_hist: torch.Tensor
        self.depth_hist: torch.Tensor
        self.dones_hist: torch.Tensor
        self.mems: list[torch.Tensor] | None
        self.current_mem_len = 0
        self.last_yaw_pred = torch.zeros(self.num_envs, 2, device=self.device)
        self.reset()

    def reset(self) -> None:
        self.prop_hist = torch.zeros(
            self.num_envs,
            self.prop_hist_len,
            self.num_prop,
            dtype=torch.float32,
            device=self.device,
        )
        self.depth_hist = torch.zeros(
            self.num_envs,
            self.depth_hist_len,
            *self.camera_resolution,
            dtype=torch.float32,
            device=self.device,
        )
        self.dones_hist = torch.zeros(
            self.num_envs,
            self.sequence_length,
            dtype=torch.bool,
            device=self.device,
        )
        self.mems = None
        self.current_mem_len = 0
        self.last_yaw_pred = torch.zeros(self.num_envs, 2, device=self.device)

    def reset_done(self, done_mask: torch.Tensor) -> None:
        mask = torch.as_tensor(done_mask, device=self.device).flatten().to(dtype=torch.bool)
        if mask.numel() != self.num_envs:
            raise ValueError(f"done_mask must have shape [{self.num_envs}]")
        if not mask.any():
            return
        self.prop_hist[mask] = 0.0
        self.depth_hist[mask] = 0.0
        self.dones_hist[mask] = False
        self.last_yaw_pred[mask] = 0.0
        if self.mems is not None:
            for mem in self.mems:
                mem[mask] = 0.0
        if mask.all():
            self.mems = None
            self.current_mem_len = 0

    def _normalize_inputs(
        self,
        proprio: torch.Tensor,
        depth_image: torch.Tensor,
        prev_done: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        proprio = torch.as_tensor(proprio, device=self.device, dtype=torch.float32)
        depth_image = torch.as_tensor(depth_image, device=self.device, dtype=torch.float32)
        if proprio.dim() != 2 or proprio.shape != (self.num_envs, self.num_prop):
            raise ValueError(f"proprio must have shape [{self.num_envs}, {self.num_prop}]")

        if depth_image.dim() == 2 and self.num_envs == 1:
            depth_image = depth_image.unsqueeze(0)
        if depth_image.dim() == 4 and depth_image.shape[1] == 1:
            depth_image = depth_image.squeeze(1)
        expected_depth_shape = (self.num_envs, *self.camera_resolution)
        if depth_image.dim() != 3 or tuple(depth_image.shape) != expected_depth_shape:
            raise ValueError(f"depth_image must have shape {expected_depth_shape}")

        if prev_done is None:
            prev_done_tensor = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            prev_done_tensor = torch.as_tensor(prev_done, device=self.device).flatten().to(torch.bool)
            if prev_done_tensor.numel() != self.num_envs:
                raise ValueError(f"prev_done must have shape [{self.num_envs}]")
        return proprio, depth_image, prev_done_tensor

    def act(
        self,
        proprio: torch.Tensor,
        depth_image: torch.Tensor,
        prev_done: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        proprio, depth_image, prev_done_tensor = self._normalize_inputs(
            proprio,
            depth_image,
            prev_done,
        )

        self.prop_hist = torch.roll(self.prop_hist, -1, dims=1)
        self.depth_hist = torch.roll(self.depth_hist, -1, dims=1)
        self.dones_hist = torch.roll(self.dones_hist, -1, dims=1)
        self.prop_hist[:, -1, :] = proprio
        self.depth_hist[:, -1, :, :] = depth_image
        self.dones_hist[:, -1] = prev_done_tensor

        prop_input = self.prop_hist.reshape(self.num_envs, -1).unsqueeze(1)
        depth_input = self.depth_hist.unsqueeze(1)
        current_done = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        if self.current_mem_len > 0:
            actual_dones_hist = self.dones_hist[:, -self.current_mem_len :]
        else:
            actual_dones_hist = torch.empty(
                self.num_envs,
                0,
                dtype=torch.bool,
                device=self.device,
            )
        full_dones = torch.cat([actual_dones_hist, current_done], dim=1)

        with torch.inference_mode():
            actions, yaw_pred, new_mems = self.model.forward_with_mems(
                prop_input,
                depth_input,
                mems=self.mems,
                full_dones=full_dones,
            )
        if new_mems is None:
            raise RuntimeError("Camera-offline TXL student did not return mems")

        self.mems = [mem.detach() for mem in new_mems]
        self.current_mem_len = min(
            self.current_mem_len + 1,
            int(self.model.temporal_model.mem_len),
        )
        actions = actions.squeeze(1)
        yaw_pred = yaw_pred.squeeze(1)
        self.last_yaw_pred = yaw_pred.detach()
        return actions, yaw_pred


def build_camera_dropout_manager(
    *,
    camera_offline_root: str | Path | None,
    num_envs: int,
    device: str | torch.device,
    dt: float,
):
    root = camera_offline_root or DEFAULT_CAMERA_OFFLINE_ROOT
    dropout_module = _load_dropout_module(root)
    return dropout_module.CameraDropoutManager(
        num_envs=int(num_envs),
        device=torch.device(device),
        dt=float(dt),
        prob_start_offline=0.0,
        prob_cam_offline=1.0,
        online_duration_range=(5.0, 5.0),
        offline_duration_range=(2.0, 2.0),
    )


def build_camera_offline_student_runner(
    *,
    checkpoint_path: str | Path | None,
    camera_offline_root: str | Path | None,
    device: str | torch.device,
    num_envs: int = 1,
    mem_len: int | None = None,
) -> CameraOfflineStudentRunner:
    root = camera_offline_root or DEFAULT_CAMERA_OFFLINE_ROOT
    checkpoint = checkpoint_path or DEFAULT_STUDENT_CHECKPOINT
    return CameraOfflineStudentRunner(
        checkpoint_path=checkpoint,
        camera_offline_root=root,
        device=device,
        num_envs=num_envs,
        mem_len=mem_len,
    )
