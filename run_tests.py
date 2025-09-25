#!/usr/bin/env python3
"""
Test runner for DOE Pipeline

Simple script to run all tests with proper configuration.
Can be executed directly or used as a module.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_tests(verbose=True, coverage=False):
    """
    Run all tests for the DOE pipeline.

    Parameters
    ----------
    verbose : bool
        Enable verbose output
    coverage : bool
        Run with coverage reporting (requires pytest-cov)
    """

    # Ensure we're in the project root
    project_root = Path(__file__).parent
    os.chdir(project_root)

    # Build pytest command
    cmd = ['python3', '-m', 'pytest', 'tests/']

    if verbose:
        cmd.append('-v')

    if coverage:
        cmd.extend(['--cov=src', '--cov-report=term-missing'])

    # Add useful pytest options
    cmd.extend([
        '--tb=short',  # Shorter traceback format
        '--strict-markers',  # Strict marker usage
        '-x',  # Stop on first failure (for debugging)
    ])

    print(f"Running tests with command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        # Run the tests
        result = subprocess.run(cmd, capture_output=False)
        return result.returncode == 0

    except FileNotFoundError:
        print("❌ Error: pytest not found. Please install it:")
        print("   pip install pytest pytest-cov")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def check_test_requirements():
    """Check if required test dependencies are available"""

    try:
        import pytest
        print("✅ pytest found")
    except ImportError:
        print("❌ pytest not found - install with: pip install pytest")
        return False

    # Check if source modules can be imported
    sys.path.append('src')

    try:
        from processing.doe.yates_algorithm import YatesAlgorithm
        print("✅ YatesAlgorithm module accessible")
    except ImportError as e:
        print(f"⚠️ Warning: Cannot import YatesAlgorithm - {e}")

    try:
        from processing.doe.costs import CostAnalyzer
        print("✅ CostAnalyzer module accessible")
    except ImportError as e:
        print(f"⚠️ Warning: Cannot import CostAnalyzer - {e}")

    try:
        from processing.doe.integrated_pipeline import DOEIntegratedPipeline
        print("✅ DOEIntegratedPipeline module accessible")
    except ImportError as e:
        print(f"⚠️ Warning: Cannot import DOEIntegratedPipeline - {e}")

    return True

def main():
    """Main test runner function"""

    print("🧪 DOE Pipeline Test Runner")
    print("=" * 40)

    # Check requirements
    if not check_test_requirements():
        print("\n❌ Test requirements check failed")
        return 1

    print("\n📋 Available test files:")
    test_files = list(Path('tests').glob('test_*.py'))
    for test_file in test_files:
        print(f"   - {test_file.name}")

    if not test_files:
        print("⚠️ No test files found in tests/ directory")
        return 1

    print(f"\n🚀 Running {len(test_files)} test modules...")

    # Run tests
    success = run_tests(verbose=True, coverage=False)

    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())