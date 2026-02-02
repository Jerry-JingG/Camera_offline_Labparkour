"""简单的测试运行器，避免 pytest 依赖问题"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入测试模块
from test_domain_randomization import (
    TestDepthNoiseAugmentation,
    TestLatencySimulation,
    TestLightingAugmentation,
    TestDomainRandCurriculum,
)

def run_tests():
    """运行所有测试"""
    test_classes = [
        TestDepthNoiseAugmentation,
        TestLatencySimulation,
        TestLightingAugmentation,
        TestDomainRandCurriculum,
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"运行测试类: {test_class.__name__}")
        print(f"{'='*60}")

        test_instance = test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]

        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"✓ {method_name} - PASSED")
                passed_tests += 1
            except Exception as e:
                print(f"✗ {method_name} - FAILED")
                print(f"  Error: {str(e)}")
                failed_tests += 1

    print(f"\n{'='*60}")
    print(f"测试总结")
    print(f"{'='*60}")
    print(f"总计: {total_tests} 个测试")
    print(f"通过: {passed_tests} 个测试")
    print(f"失败: {failed_tests} 个测试")

    return failed_tests == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
