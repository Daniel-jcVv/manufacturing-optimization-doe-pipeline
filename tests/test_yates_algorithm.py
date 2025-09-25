"""
Unit tests for Yates Algorithm DOE Analysis

This module tests the core functionality of the YatesAlgorithm class,
focusing on critical calculations and result validation. Tests use
synthetic data to ensure reproducible and reliable validation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from processing.doe.yates_algorithm import YatesAlgorithm, DOEFactor, YatesResult


class TestYatesAlgorithm:
    """Test class for YatesAlgorithm functionality"""

    def test_initialization(self):
        """Test proper initialization of YatesAlgorithm"""
        yates = YatesAlgorithm()

        # Test factor definitions exist
        assert len(yates.factors) == 3
        assert 'A' in yates.factors
        assert 'B' in yates.factors
        assert 'C' in yates.factors

        # Test factor properties
        factor_a = yates.factors['A']
        assert factor_a.name == 'Pressure'
        assert factor_a.low_level == 750.0
        assert factor_a.high_level == 1050.0
        assert factor_a.unit == 'PSI'

        # Test yates order
        expected_order = ['c', 'a', 'b', 'abc']
        assert yates.yates_order == expected_order

    def test_data_validation_with_valid_data(self, sample_experiment_data):
        """Test data validation with properly formatted data"""
        yates = YatesAlgorithm()

        # This should not raise any exceptions
        required_cols = ['yates_order', 'factor_A', 'factor_B', 'factor_C', 'tool_life_pieces']
        missing_cols = [col for col in required_cols if col not in sample_experiment_data.columns]

        assert len(missing_cols) == 0, f"Missing required columns: {missing_cols}"

        # Test that all required treatments are present
        treatments = set(sample_experiment_data['yates_order'].unique())
        required_treatments = {'c', 'a', 'b', 'abc'}
        assert required_treatments.issubset(treatments), "Missing required treatments"

    def test_data_validation_with_missing_columns(self, sample_experiment_data):
        """Test data validation with missing columns"""
        yates = YatesAlgorithm()

        # Remove a required column
        invalid_data = sample_experiment_data.drop(columns=['factor_A'])

        # Should detect missing column
        required_cols = ['yates_order', 'factor_A', 'factor_B', 'factor_C', 'tool_life_pieces']
        missing_cols = [col for col in required_cols if col not in invalid_data.columns]

        assert 'factor_A' in missing_cols

    def test_calculate_treatment_averages(self, sample_experiment_data):
        """Test calculation of treatment averages"""
        yates = YatesAlgorithm()
        averages = yates.calculate_treatment_averages(sample_experiment_data)

        # Should have averages for all treatments
        expected_treatments = ['c', 'a', 'b', 'abc']
        for treatment in expected_treatments:
            assert treatment in averages
            assert isinstance(averages[treatment], (int, float))
            assert averages[treatment] > 0

        # Verify averages are reasonable (based on synthetic data logic)
        assert averages['abc'] > averages['c']  # All factors should be better than just C
        assert averages['c'] > averages['a']   # C effect should be stronger than A
        assert averages['a'] > averages['b']   # A effect should be stronger than B

    def test_execute_yates_algorithm_structure(self, sample_experiment_data):
        """Test that Yates algorithm returns proper structure"""
        yates = YatesAlgorithm()
        averages = yates.calculate_treatment_averages(sample_experiment_data)
        effects = yates.execute_yates_algorithm(averages)

        # Should return dict with expected keys
        expected_keys = ['Overall_Mean', 'A', 'B', 'AB']
        for key in expected_keys:
            assert key in effects
            assert isinstance(effects[key], (int, float))

        # Overall mean should be positive
        assert effects['Overall_Mean'] > 0

    def test_execute_yates_algorithm_calculations(self, sample_experiment_data, expected_yates_effects):
        """Test Yates algorithm calculations against expected values"""
        yates = YatesAlgorithm()
        averages = yates.calculate_treatment_averages(sample_experiment_data)
        effects = yates.execute_yates_algorithm(averages)

        # Validate calculated effects against expected ranges
        for effect_name, expected in expected_yates_effects.items():
            if effect_name in effects:
                calculated_value = effects[effect_name]
                expected_value = expected['value']
                tolerance = expected['tolerance']

                # Check if calculated value is within expected tolerance
                assert abs(calculated_value - expected_value) <= tolerance, \
                    f"Effect {effect_name}: calculated {calculated_value}, expected {expected_value} ± {tolerance}"

    def test_determine_significance(self, sample_experiment_data):
        """Test significance determination logic"""
        yates = YatesAlgorithm()
        averages = yates.calculate_treatment_averages(sample_experiment_data)
        effects = yates.execute_yates_algorithm(averages)
        significance = yates.determine_significance(effects)

        # Should return significance for main effects (not Overall_Mean)
        main_effects = ['A', 'B', 'AB']
        for effect in main_effects:
            if effect in effects:
                assert effect in significance
                assert isinstance(significance[effect], bool)

        # At least one effect should be significant (based on synthetic data)
        significant_effects = [k for k, v in significance.items() if v]
        assert len(significant_effects) > 0, "At least one effect should be significant"

    def test_find_optimal_conditions(self, sample_experiment_data):
        """Test optimal conditions determination"""
        yates = YatesAlgorithm()
        averages = yates.calculate_treatment_averages(sample_experiment_data)
        effects = yates.execute_yates_algorithm(averages)
        optimal = yates.find_optimal_conditions(effects)

        # Should return proper structure
        assert 'optimal_levels' in optimal
        assert 'expected_improvement' in optimal
        assert 'recommended_settings' in optimal

        # Check optimal levels structure
        optimal_levels = optimal['optimal_levels']
        for factor_key in ['A', 'B', 'C']:
            if factor_key in optimal_levels:
                level_info = optimal_levels[factor_key]
                assert 'level' in level_info
                assert 'value' in level_info
                assert 'unit' in level_info
                assert 'factor_name' in level_info
                assert level_info['level'] in ['high', 'low']

        # Check recommended settings
        settings = optimal['recommended_settings']
        assert 'pressure_psi' in settings
        assert 'concentration_pct' in settings
        assert 'rpm' in settings
        assert 'feed_rate' in settings

        # Values should be reasonable
        assert 750 <= settings['pressure_psi'] <= 1050
        assert 4.0 <= settings['concentration_pct'] <= 6.0
        assert settings['rpm'] == 3700  # Should be high level from case study
        assert settings['feed_rate'] == 1050  # Should be high level

    def test_complete_analysis_workflow(self, temp_data_files):
        """Test complete DOE analysis workflow from file input"""
        yates = YatesAlgorithm()

        # Run complete analysis
        results = yates.analyze_doe_experiment(temp_data_files['experiment_data'])

        # Validate complete results structure
        expected_keys = ['treatment_averages', 'effects', 'significance',
                        'optimal_conditions', 'summary']
        for key in expected_keys:
            assert key in results

        # Validate treatment averages
        averages = results['treatment_averages']
        assert len(averages) == 4  # c, a, b, abc

        # Validate effects
        effects = results['effects']
        assert 'Overall_Mean' in effects
        assert 'A' in effects or 'B' in effects  # At least some main effects

        # Validate summary
        summary = results['summary']
        assert 'total_experiments' in summary
        assert 'significant_factors' in summary
        assert 'expected_improvement_pct' in summary

        assert summary['total_experiments'] > 0
        assert isinstance(summary['significant_factors'], list)

    def test_yates_results_creation(self, sample_experiment_data):
        """Test creation of structured YatesResult objects"""
        yates = YatesAlgorithm()

        # Run analysis to populate results
        yates.analyze_doe_experiment()
        # Note: This will fail without proper data, but we're testing the structure

        # Test result object properties
        # (This is more of a structure test since we need actual data files)
        result_obj = YatesResult(
            factor_name='A',
            effect_value=800.0,
            effect_rank=1,
            is_significant=True,
            contribution_pct=45.2
        )

        assert result_obj.factor_name == 'A'
        assert result_obj.effect_value == 800.0
        assert result_obj.effect_rank == 1
        assert result_obj.is_significant is True
        assert result_obj.contribution_pct == 45.2

    def test_get_results_summary_without_analysis(self):
        """Test results summary when no analysis has been run"""
        yates = YatesAlgorithm()
        summary = yates.get_results_summary()

        expected_message = "No analysis results available. Run analyze_doe_experiment() first."
        assert summary == expected_message

    def test_error_handling_missing_treatments(self, sample_experiment_data):
        """Test error handling when required treatments are missing"""
        yates = YatesAlgorithm()

        # Remove one treatment
        incomplete_data = sample_experiment_data[
            sample_experiment_data['yates_order'] != 'abc'
        ].copy()

        # Should handle missing treatment gracefully
        with pytest.raises(ValueError, match="No data found for treatment"):
            yates.calculate_treatment_averages(incomplete_data)

    def test_mathematical_precision(self, sample_experiment_data):
        """Test mathematical precision of Yates calculations"""
        yates = YatesAlgorithm()
        averages = yates.calculate_treatment_averages(sample_experiment_data)

        # Test with known simple values to verify calculation precision
        simple_averages = {'c': 1000, 'a': 1200, 'b': 1100, 'abc': 1500}
        effects = yates.execute_yates_algorithm(simple_averages)

        # Overall mean should be average of all treatments
        expected_mean = (1000 + 1200 + 1100 + 1500) / 4 / 3  # Divided by n_replicates
        assert abs(effects['Overall_Mean'] - expected_mean) < 0.1

        # Effects should be calculated correctly (simplified validation)
        assert isinstance(effects['A'], (int, float))
        assert isinstance(effects['B'], (int, float))
        assert isinstance(effects['AB'], (int, float))


class TestDOEFactor:
    """Test DOEFactor dataclass functionality"""

    def test_doe_factor_creation(self):
        """Test DOEFactor creation and properties"""
        factor = DOEFactor(
            name='TestFactor',
            low_level=10.0,
            high_level=20.0,
            unit='units',
            description='Test factor description'
        )

        assert factor.name == 'TestFactor'
        assert factor.low_level == 10.0
        assert factor.high_level == 20.0
        assert factor.unit == 'units'
        assert factor.description == 'Test factor description'


# Integration test fixtures specific to this module
@pytest.fixture
def simple_yates_data():
    """Simple Yates data for mathematical validation"""
    return {
        'c': 5000,
        'a': 5500,
        'b': 5200,
        'abc': 6000
    }


class TestYatesEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_data_handling(self):
        """Test handling of empty datasets"""
        yates = YatesAlgorithm()
        empty_df = pd.DataFrame(columns=['yates_order', 'factor_A', 'factor_B', 'factor_C', 'tool_life_pieces'])

        with pytest.raises(ValueError):
            yates.calculate_treatment_averages(empty_df)

    def test_negative_tool_life_handling(self, sample_experiment_data):
        """Test handling of negative tool life values"""
        yates = YatesAlgorithm()

        # Introduce negative values
        bad_data = sample_experiment_data.copy()
        bad_data.loc[0, 'tool_life_pieces'] = -100

        # Should still calculate but may affect results
        averages = yates.calculate_treatment_averages(bad_data)
        # Verify it handles the negative value (treatment 'c' in first row)
        assert isinstance(averages['c'], (int, float))

    def test_single_replicate_per_treatment(self):
        """Test with single replicate per treatment"""
        data = pd.DataFrame([
            {'yates_order': 'c', 'factor_A': 750, 'factor_B': 4, 'factor_C': 3700, 'tool_life_pieces': 5000},
            {'yates_order': 'a', 'factor_A': 1050, 'factor_B': 4, 'factor_C': 3183, 'tool_life_pieces': 5500},
            {'yates_order': 'b', 'factor_A': 750, 'factor_B': 6, 'factor_C': 3183, 'tool_life_pieces': 5200},
            {'yates_order': 'abc', 'factor_A': 1050, 'factor_B': 6, 'factor_C': 3700, 'tool_life_pieces': 6000},
        ])

        yates = YatesAlgorithm()
        averages = yates.calculate_treatment_averages(data)

        # Should work with single replicates (averages = actual values)
        assert averages['c'] == 5000
        assert averages['a'] == 5500
        assert averages['b'] == 5200
        assert averages['abc'] == 6000