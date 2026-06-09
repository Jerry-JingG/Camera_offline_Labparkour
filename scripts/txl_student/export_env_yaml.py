from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "parkour_tasks", PROJECT_ROOT / "scripts" / "rsl_rl"):
    path_str = os.fspath(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


DEFAULT_TASK = "Isaac-Extreme-Parkour-TeacherCam-Unitree-Go2-Collect-v0"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "students" / "train_from_dagger" / "xl0505_pro" / "params" / "env.yaml"


def parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Export the env.yaml used by student DAgger training.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Isaac task name.")
    parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel environments.")
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        help="Match train_student_dagger.py behavior when Fabric is disabled.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output env.yaml path, or a directory where env.yaml should be written.",
    )
    parser.add_argument(
        "--no_mirror_run_dir",
        action="store_true",
        help="Do not also write env.yaml next to the checkpoint when output is under a params directory.",
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser.parse_args()


def resolve_output_path(output: Path) -> Path:
    if output.suffix.lower() in {".yaml", ".yml"}:
        return output
    return output / "env.yaml"


def main() -> None:
    args = parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import parkour_tasks  # noqa: F401
        from isaaclab.utils.io import dump_yaml
        from isaaclab_tasks.utils import parse_env_cfg

        env_cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=args.num_envs,
            use_fabric=not args.disable_fabric,
        )

        output_path = resolve_output_path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dump_yaml(os.fspath(output_path), env_cfg)

        print(f"[INFO] Exported env config: {output_path}")

        print(
            f"[INFO] task={args.task}, num_envs={args.num_envs}, "
            f"device={args.device}, use_fabric={not args.disable_fabric}"
        )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
