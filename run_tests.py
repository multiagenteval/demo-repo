import pytest
import sys

def run_tests():
    """Run all tests in specific order"""
    test_order = [
        "tests/test_config.py",        # Test config first
        "tests/test_dependencies.py",  # Then test dependencies
        "tests/test_evaluator.py",     # Then test components
    ]
    
    failed = []
    for test_file in test_order:
        print(f"\nRunning {test_file}...")
        result = pytest.main(["-v", test_file])
        if result != 0:
            failed.append(test_file)
    
    if failed:
        print("\nThe following tests failed:")
        for test in failed:
            print(f"  - {test}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")

if __name__ == "__main__":
    run_tests() 