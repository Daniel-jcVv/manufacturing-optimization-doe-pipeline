"""
Pytest configuration and fixtures for DOE analysis testing

This module provides common fixtures and test data for all test modules.
It includes synthetic data generation for testing DOE algorithms and
cost analysis functions without depending on external data files.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os


@pytest.fixture
def sample_experiment_data():
    """
    Generate sample experiment data for DOE testing.

    Returns
    -------
    pd.DataFrame
        Experiment data with required columns for Yates analysis
    """
    np.random.seed(42)  # Reproducible results

    # Generate experiment results for 2^(3-1) fractional factorial
    # Treatments: c, a, b, abc (Yates order)
    treatments = ['c', 'a', 'b', 'abc']

    data = []
    experiment_id = 1

    # Generate 3 replicates per treatment (as in real case study)
    for treatment in treatments:
        for replicate in range(1, 4):
            # Simulate tool life based on treatment effect
            base_life = 4000  # Base tool life pieces

            # Add treatment effects (simplified simulation)
            if treatment == 'c':  # Factor C high
                tool_life = base_life + 1200 + np.random.normal(0, 200)
            elif treatment == 'a':  # Factor A high
                tool_life = base_life + 800 + np.random.normal(0, 200)
            elif treatment == 'b':  # Factor B high
                tool_life = base_life + 600 + np.random.normal(0, 200)
            elif treatment == 'abc':  # All factors high
                tool_life = base_life + 2000 + np.random.normal(0, 200)

            # Ensure positive values
            tool_life = max(tool_life, 1000)

            data.append({
                'experiment_id': f'EXP_{experiment_id:03d}',
                'yates_order': treatment,
                'factor_A': 1050.0 if 'a' in treatment else 750.0,
                'factor_B': 6.0 if 'b' in treatment else 4.0,
                'factor_C': 3700.0 if 'c' in treatment else 3183.0,
                'tool_life_pieces': int(tool_life),
                'replicate': replicate
            })
            experiment_id += 1

    return pd.DataFrame(data)


@pytest.fixture
def sample_production_data():
    """
    Generate sample production data for cost analysis testing.

    Returns
    -------
    pd.DataFrame
        Production data with current and optimized configurations
    """
    np.random.seed(42)

    data = []
    record_id = 1

    tools = ['ZC1668', 'ZC1445']
    configurations = ['current', 'optimized']

    # Generate production records for each tool and configuration
    for tool in tools:
        for config in configurations:
            # Generate 15 records per tool-config combination
            for i in range(15):
                # Simulate production metrics
                daily_production = int(np.random.normal(500, 50))  # Average 500 pieces/day

                if config == 'current':
                    # Current scenario - higher costs
                    if tool == 'ZC1668':
                        cpu = np.random.normal(0.041, 0.003)  # $0.041 +/- variation
                        total_cost = cpu * daily_production
                    else:  # ZC1445
                        cpu = np.random.normal(0.058, 0.004)  # $0.058 +/- variation
                        total_cost = cpu * daily_production
                else:  # optimized
                    # Optimized scenario - lower costs (60% reduction)
                    if tool == 'ZC1668':
                        cpu = np.random.normal(0.0164, 0.001)  # 60% reduction
                        total_cost = cpu * daily_production
                    else:  # ZC1445
                        cpu = np.random.normal(0.0221, 0.002)  # 61% reduction
                        total_cost = cpu * daily_production

                data.append({
                    'record_id': record_id,
                    'tool_id': tool,
                    'configuration': config,
                    'daily_production': daily_production,
                    'cpu': cpu,
                    'total_cost': total_cost,
                    'analysis_date': '2025-09-24'
                })
                record_id += 1

    return pd.DataFrame(data)


@pytest.fixture
def temp_data_files(sample_experiment_data, sample_production_data):
    """
    Create temporary CSV files for testing file I/O operations.

    Parameters
    ----------
    sample_experiment_data : pd.DataFrame
        Experiment data fixture
    sample_production_data : pd.DataFrame
        Production data fixture

    Returns
    -------
    dict
        Dictionary with paths to temporary files
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    # Save data to temporary CSV files
    exp_path = os.path.join(temp_dir, 'test_experiment_results.csv')
    prod_path = os.path.join(temp_dir, 'test_production_data.csv')

    sample_experiment_data.to_csv(exp_path, index=False)
    sample_production_data.to_csv(prod_path, index=False)

    yield {
        'experiment_data': exp_path,
        'production_data': prod_path,
        'temp_dir': temp_dir
    }

    # Cleanup temporary files
    import shutil
    shutil.rmtree(temp_dir)


@pytest.fixture
def expected_yates_effects():
    """
    Expected Yates algorithm effects for validation.

    Based on the synthetic data generation logic, we can predict
    approximate effect values for testing purposes.

    Returns
    -------
    dict
        Expected effect values with tolerance ranges
    """
    return {
        'Overall_Mean': {'value': 5200, 'tolerance': 300},  # Approximate mean
        'A': {'value': 800, 'tolerance': 200},   # Factor A effect
        'B': {'value': 600, 'tolerance': 200},   # Factor B effect
        'AB': {'value': 400, 'tolerance': 300},  # AB interaction
    }


@pytest.fixture
def expected_cost_metrics():
    """
    Expected cost analysis metrics for validation.

    Returns
    -------
    dict
        Expected cost metrics by tool
    """
    return {
        'ZC1668': {
            'cpu_reduction_pct': {'value': 60.0, 'tolerance': 5.0},
            'annual_savings': {'min': 2500, 'max': 4000},
            'roi_pct': {'min': 5.0, 'max': 15.0}
        },
        'ZC1445': {
            'cpu_reduction_pct': {'value': 61.0, 'tolerance': 5.0},
            'annual_savings': {'min': 3500, 'max': 6000},
            'roi_pct': {'min': 7.0, 'max': 18.0}
        }
    }


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Session-wide setup for test environment.

    Ensures that all tests run with consistent settings and
    that any required directories exist.
    """
    # Ensure test output directory exists
    test_output_dir = Path("tests/output")
    test_output_dir.mkdir(exist_ok=True)

    # Set numpy random seed for reproducibility
    np.random.seed(42)

    yield

    # Cleanup test outputs (optional)
    # You might want to keep outputs for debugging
    pass


# Constants for testing
TEST_TOOL_COSTS = {
    'ZC1668': {
        'cost_new': 148.94,
        'cost_regrind': 30.67,
        'max_regrinds': 3
    },
    'ZC1445': {
        'cost_new': 206.76,
        'cost_regrind': 47.14,
        'max_regrinds': 3
    }
}

TEST_PRODUCTION_BASELINE = {
    'daily_production_pieces': 500,
    'current_tool_life_pieces': 4000,
    'optimized_tool_life_pieces': 15000,
    'working_days_per_year': 250
}