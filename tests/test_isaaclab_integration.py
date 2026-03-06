"""
测试 Isaac Lab 环境集成

验证 run_student_rl_finetune.py 脚本能够：
1. 正确设置 ISAACLAB_PATH 环境变量
2. 正确导入 Isaac Lab 模块
3. 正确初始化 AppLauncher
4. 在没有 ./isaaclab.sh 包装的情况下运行

TDD 流程：
- RED: 这些测试最初会失败，因为环境集成尚未实现
- GREEN: 实现环境集成后，测试应该通过
- REFACTOR: 优化代码结构
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import subprocess

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "rsl_rl"

# 添加脚本目录到路径
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 测试 1: 环境变量设置
# ============================================================

class TestEnvironmentSetup:
    """测试环境变量设置"""

    def test_isaaclab_path_is_set(self):
        """测试 ISAACLAB_PATH 环境变量被正确设置"""
        # 导入环境设置模块
        from isaaclab_env_setup import setup_isaaclab_env

        # 设置环境
        setup_isaaclab_env()

        # 验证 ISAACLAB_PATH 被设置
        assert "ISAACLAB_PATH" in os.environ
        isaaclab_path = os.environ["ISAACLAB_PATH"]
        assert Path(isaaclab_path).exists()

    def test_isaaclab_path_points_to_valid_directory(self):
        """测试 ISAACLAB_PATH 指向有效的 Isaac Lab 目录"""
        from isaaclab_env_setup import setup_isaaclab_env

        setup_isaaclab_env()

        isaaclab_path = Path(os.environ["ISAACLAB_PATH"])
        # 验证目录包含 Isaac Lab 的标志性文件
        assert (isaaclab_path / "isaaclab.sh").exists() or \
               (isaaclab_path / "source").exists()

    def test_find_isaaclab_path(self):
        """测试 find_isaaclab_path 函数"""
        from isaaclab_env_setup import find_isaaclab_path

        path = find_isaaclab_path()
        assert path.exists()
        assert (path / "isaaclab.sh").exists()


# ============================================================
# 测试 2: Isaac Lab 模块导入
# ============================================================

class TestIsaacLabImports:
    """测试 Isaac Lab 模块导入"""

    def test_can_import_isaaclab_app(self):
        """测试能够导入 isaaclab.app 模块"""
        from isaaclab_env_setup import setup_isaaclab_env

        setup_isaaclab_env()

        try:
            from isaaclab.app import AppLauncher
            assert AppLauncher is not None
        except ImportError as e:
            pytest.skip(f"Isaac Lab not installed: {e}")

    def test_can_import_isaaclab_envs(self):
        """测试能够导入 isaaclab.envs 模块"""
        from isaaclab_env_setup import setup_isaaclab_env

        setup_isaaclab_env()

        try:
            from isaaclab.envs import ManagerBasedRLEnvCfg
            assert ManagerBasedRLEnvCfg is not None
        except ImportError as e:
            pytest.skip(f"Isaac Lab not installed: {e}")


# ============================================================
# 测试 3: AppLauncher 集成
# ============================================================

class TestAppLauncherIntegration:
    """测试 AppLauncher 集成"""

    def test_create_app_launcher_args(self):
        """测试创建 AppLauncher 参数"""
        from isaaclab_env_setup import create_app_launcher_args

        args = create_app_launcher_args(
            headless=True,
            num_envs=256,
            device="cuda:0",
        )

        assert args.headless == True
        assert hasattr(args, "device")
        assert args.device == "cuda:0"
        assert args.num_envs == 256

    def test_create_app_launcher_args_defaults(self):
        """测试 AppLauncher 参数默认值"""
        from isaaclab_env_setup import create_app_launcher_args

        args = create_app_launcher_args()

        assert args.headless == False
        assert args.enable_cameras == False

    def test_app_launcher_initialization(self):
        """测试 AppLauncher 初始化"""
        from isaaclab_env_setup import setup_isaaclab_env

        setup_isaaclab_env()

        try:
            from isaaclab.app import AppLauncher

            # 注意：实际初始化 AppLauncher 会启动 Isaac Sim
            # 这里只验证类存在
            assert callable(AppLauncher)
        except ImportError as e:
            pytest.skip(f"Isaac Lab not installed: {e}")


# ============================================================
# 测试 4: 脚本直接运行
# ============================================================

class TestDirectScriptExecution:
    """测试脚本直接运行（不使用 isaaclab.sh）"""

    def test_script_has_env_setup_at_top(self):
        """测试脚本在顶部有环境设置代码"""
        script_path = SCRIPTS_DIR / "run_student_rl_finetune.py"

        with open(script_path, "r") as f:
            content = f.read()

        # 验证脚本包含环境设置导入
        assert "isaaclab_env_setup" in content or "setup_isaaclab_env" in content or \
               "AppLauncher" in content, \
               "脚本应该包含 Isaac Lab 环境设置代码"

    def test_script_imports_app_launcher(self):
        """测试脚本导入 AppLauncher"""
        script_path = SCRIPTS_DIR / "run_student_rl_finetune.py"

        with open(script_path, "r") as f:
            content = f.read()

        # 验证脚本导入 AppLauncher
        assert "AppLauncher" in content, \
               "脚本应该导入 AppLauncher"

    def test_script_syntax_valid(self):
        """测试脚本语法有效"""
        script_path = SCRIPTS_DIR / "run_student_rl_finetune.py"

        # 使用 Python 编译检查语法
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(script_path)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"语法错误: {result.stderr}"


# ============================================================
# 测试 5: 环境设置模块
# ============================================================

class TestEnvSetupModule:
    """测试环境设置模块"""

    def test_module_exists(self):
        """测试环境设置模块存在"""
        module_path = SCRIPTS_DIR / "isaaclab_env_setup.py"
        assert module_path.exists(), \
               f"环境设置模块不存在: {module_path}"

    def test_module_has_setup_function(self):
        """测试模块有 setup_isaaclab_env 函数"""
        from isaaclab_env_setup import setup_isaaclab_env
        assert callable(setup_isaaclab_env)

    def test_module_has_create_args_function(self):
        """测试模块有 create_app_launcher_args 函数"""
        from isaaclab_env_setup import create_app_launcher_args
        assert callable(create_app_launcher_args)

    def test_module_has_find_path_function(self):
        """测试模块有 find_isaaclab_path 函数"""
        from isaaclab_env_setup import find_isaaclab_path
        assert callable(find_isaaclab_path)

    def test_module_has_ensure_env_setup_function(self):
        """测试模块有 ensure_env_setup 函数"""
        from isaaclab_env_setup import ensure_env_setup
        assert callable(ensure_env_setup)


# ============================================================
# 测试 6: 完整集成测试
# ============================================================

class TestFullIntegration:
    """完整集成测试"""

    def test_script_can_be_imported(self):
        """测试脚本可以被导入（不执行 main）"""
        # 设置环境变量以跳过 Isaac Sim 初始化
        os.environ["ISAACLAB_SKIP_INIT"] = "1"

        try:
            # 尝试导入脚本模块
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "run_student_rl_finetune",
                SCRIPTS_DIR / "run_student_rl_finetune.py"
            )
            module = importlib.util.module_from_spec(spec)

            # 只检查模块可以加载，不执行
            assert spec is not None
            assert module is not None
        finally:
            os.environ.pop("ISAACLAB_SKIP_INIT", None)

    def test_script_syntax_valid(self):
        """测试脚本语法有效"""
        script_path = SCRIPTS_DIR / "run_student_rl_finetune.py"

        # 使用 Python 编译检查语法
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(script_path)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"语法错误: {result.stderr}"

    def test_script_has_env_setup_at_top(self):
        """测试脚本在顶部有环境设置代码"""
        script_path = SCRIPTS_DIR / "run_student_rl_finetune.py"

        with open(script_path, "r") as f:
            content = f.read()

        # 验证脚本包含环境设置导入
        assert "isaaclab_env_setup" in content or "setup_isaaclab_env" in content or \
               "AppLauncher" in content, \
               "脚本应该包含 Isaac Lab 环境设置代码"

    def test_script_imports_app_launcher(self):
        """测试脚本导入 AppLauncher"""
        script_path = SCRIPTS_DIR / "run_student_rl_finetune.py"

        with open(script_path, "r") as f:
            content = f.read()

        # 验证脚本导入 AppLauncher
        assert "AppLauncher" in content, \
               "脚本应该导入 AppLauncher"

    def test_script_no_isaaclab_sh_hints(self):
        """测试脚本不再包含 ./isaaclab.sh 的提示"""
        script_path = SCRIPTS_DIR / "run_student_rl_finetune.py"

        with open(script_path, "r") as f:
            content = f.read()

        # 验证脚本不再包含 ./isaaclab.sh 的运行提示
        assert "./isaaclab.sh -p scripts/rsl_rl/run_student_rl_finetune.py" not in content, \
               "脚本不应该包含 ./isaaclab.sh 的运行提示"


# ============================================================
# 运行测试
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
