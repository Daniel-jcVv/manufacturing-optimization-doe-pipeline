"""
Unit tests for Cost Analysis Module

This module tests the core functionality of the CostAnalyzer class,
focusing on financial calculations, ROI metrics, and business case
generation. Tests validate mathematical accuracy and business logic.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from processing.doe.costs import (
    CostAnalyzer, ToolCostData, ProductionMetrics,
    CostSavingsResult
)


class TestCostAnalyzer:
    """Test class for CostAnalyzer functionality"""

    def test_initialization(self):
        """Test proper initialization of CostAnalyzer"""
        analyzer = CostAnalyzer()

        # Test tool costs are loaded
        assert len(analyzer.tool_costs) == 2
        assert 'ZC1668' in analyzer.tool_costs
        assert 'ZC1445' in analyzer.tool_costs

        # Test tool cost data structure
        zc1668 = analyzer.tool_costs['ZC1668']
        assert zc1668.tool_id == 'ZC1668'
        assert zc1668.cost_new == 148.94
        assert zc1668.cost_regrind == 30.67
        assert zc1668.max_regrinds == 3

        # Test baseline metrics
        baseline = analyzer.baseline_metrics
        assert baseline.daily_production_pieces == 500
        assert baseline.current_tool_life_pieces == 4000
        assert baseline.optimized_tool_life_pieces == 15000
        assert baseline.working_days_per_year == 250

    def test_calculate_tool_lifecycle_cost(self):
        """Test tool lifecycle cost calculation"""
        analyzer = CostAnalyzer()

        # Test ZC1668 lifecycle cost
        lifecycle_cost_1668 = analyzer.calculate_tool_lifecycle_cost('ZC1668', 4000)
        expected_cost_1668 = 148.94 + (30.67 * 3)  # New cost + regrind costs
        assert abs(lifecycle_cost_1668 - expected_cost_1668) < 0.01

        # Test ZC1445 lifecycle cost
        lifecycle_cost_1445 = analyzer.calculate_tool_lifecycle_cost('ZC1445', 4000)
        expected_cost_1445 = 206.76 + (47.14 * 3)
        assert abs(lifecycle_cost_1445 - expected_cost_1445) < 0.01

        # Test unknown tool error
        with pytest.raises(ValueError, match="Unknown tool_id"):
            analyzer.calculate_tool_lifecycle_cost('UNKNOWN_TOOL', 4000)

    def test_calculate_cpu_metrics(self):
        """Test Cost Per Unit (CPU) metrics calculation"""
        analyzer = CostAnalyzer()

        # Test CPU metrics for ZC1668
        metrics = analyzer.calculate_cpu_metrics(
            tool_id='ZC1668',
            current_life=4000,
            optimized_life=15000,
            daily_production=500
        )

        # Validate structure
        expected_keys = ['current_cpu', 'optimized_cpu', 'cpu_reduction',
                        'cpu_reduction_pct', 'life_improvement_pct']
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))

        # Validate calculations
        lifecycle_cost = 148.94 + (30.67 * 3)  # 241.95
        expected_current_cpu = lifecycle_cost / 4000
        expected_optimized_cpu = lifecycle_cost / 15000

        assert abs(metrics['current_cpu'] - expected_current_cpu) < 0.0001
        assert abs(metrics['optimized_cpu'] - expected_optimized_cpu) < 0.0001

        # CPU reduction should be positive
        assert metrics['cpu_reduction'] > 0
        assert metrics['cpu_reduction_pct'] > 0

        # Life improvement should be significant
        assert metrics['life_improvement_pct'] > 200  # 275% improvement expected

    def test_calculate_annual_savings(self):
        """Test annual savings calculation"""
        analyzer = CostAnalyzer()

        # Test annual savings calculation
        cpu_reduction = 0.025  # $0.025 per piece reduction
        daily_production = 500
        working_days = 250

        annual_savings = analyzer.calculate_annual_savings(
            tool_id='ZC1668',
            cpu_reduction=cpu_reduction,
            daily_production=daily_production,
            working_days=working_days
        )

        expected_savings = cpu_reduction * daily_production * working_days
        assert abs(annual_savings - expected_savings) < 0.01

        # Test with different parameters
        annual_savings_custom = analyzer.calculate_annual_savings(
            tool_id='ZC1445',
            cpu_reduction=0.030,
            daily_production=600,
            working_days=300
        )

        expected_custom = 0.030 * 600 * 300
        assert abs(annual_savings_custom - expected_custom) < 0.01

    def test_calculate_roi_metrics(self):
        """Test ROI and payback period calculations"""
        analyzer = CostAnalyzer()

        # Test with positive annual savings
        roi_metrics = analyzer.calculate_roi_metrics(
            annual_savings=10000,  # $10,000 annual savings
            implementation_cost=50000  # $50,000 implementation cost
        )

        expected_keys = ['roi_pct', 'payback_period_months', 'net_present_value_3yr']
        for key in expected_keys:
            assert key in roi_metrics
            assert isinstance(roi_metrics[key], (int, float))

        # Validate calculations
        expected_roi = (10000 / 50000) * 100  # 20%
        assert abs(roi_metrics['roi_pct'] - expected_roi) < 0.01

        expected_payback = (50000 / 10000) * 12  # 60 months
        assert abs(roi_metrics['payback_period_months'] - expected_payback) < 0.01

        expected_npv = (10000 * 3) - 50000  # -$20,000 (break-even in 5 years)
        assert abs(roi_metrics['net_present_value_3yr'] - expected_npv) < 0.01

        # Test with zero savings
        zero_metrics = analyzer.calculate_roi_metrics(annual_savings=0)
        assert zero_metrics['roi_pct'] == 0
        assert zero_metrics['payback_period_months'] == float('inf')

    def test_analyze_cost_savings_with_data(self, sample_production_data, temp_data_files):
        """Test complete cost savings analysis with sample data"""
        analyzer = CostAnalyzer()

        # Save sample data to temporary file
        prod_path = temp_data_files['temp_dir'] + '/test_production.csv'
        sample_production_data.to_csv(prod_path, index=False)

        # Run analysis
        results = analyzer.analyze_cost_savings(data_source=prod_path)

        # Validate results structure
        assert isinstance(results, dict)
        assert len(results) > 0

        # Check that we have results for expected tools
        for tool_id in ['ZC1668', 'ZC1445']:
            if tool_id in results:
                result = results[tool_id]
                assert isinstance(result, CostSavingsResult)

                # Validate result attributes
                assert result.tool_id == tool_id
                assert result.current_cpu > 0
                assert result.optimized_cpu > 0
                assert result.current_cpu > result.optimized_cpu  # Should be improvement
                assert result.cpu_reduction_pct > 0
                assert result.annual_savings_usd > 0

    def test_generate_business_case_report(self, expected_cost_metrics):
        """Test business case report generation"""
        analyzer = CostAnalyzer()

        # Create sample cost results
        cost_results = {
            'ZC1668': CostSavingsResult(
                tool_id='ZC1668',
                current_cpu=0.0411,
                optimized_cpu=0.0164,
                cpu_reduction_pct=60.1,
                tool_life_improvement_pct=150.4,
                annual_savings_usd=3084.52,
                roi_pct=6.17,
                payback_period_months=194.5
            ),
            'ZC1445': CostSavingsResult(
                tool_id='ZC1445',
                current_cpu=0.0580,
                optimized_cpu=0.0221,
                cpu_reduction_pct=61.8,
                tool_life_improvement_pct=154.6,
                annual_savings_usd=4544.72,
                roi_pct=9.09,
                payback_period_months=132.0
            )
        }

        # Generate report
        report = analyzer.generate_business_case_report(cost_results)

        # Validate report structure
        assert isinstance(report, str)
        assert len(report) > 0

        # Check for key sections
        assert "DOE OPTIMIZATION - BUSINESS CASE ANALYSIS" in report
        assert "EXECUTIVE SUMMARY" in report
        assert "DETAILED ANALYSIS BY TOOL" in report
        assert "IMPLEMENT DOE OPTIMIZATION IMMEDIATELY" in report

        # Check for key metrics
        assert "7,629" in report  # Total annual savings
        assert "ZC1668" in report
        assert "ZC1445" in report
        assert "60.1%" in report or "61%" in report  # CPU reductions

    def test_generate_business_case_report_empty(self):
        """Test business case report with no results"""
        analyzer = CostAnalyzer()
        report = analyzer.generate_business_case_report({})

        assert report == "No cost analysis results available."

    def test_save_results_to_database(self, expected_cost_metrics):
        """Test saving results to CSV (database simulation)"""
        analyzer = CostAnalyzer()

        # Create sample results
        cost_results = {
            'ZC1668': CostSavingsResult(
                tool_id='ZC1668',
                current_cpu=0.0411,
                optimized_cpu=0.0164,
                cpu_reduction_pct=60.1,
                tool_life_improvement_pct=150.4,
                annual_savings_usd=3084.52,
                roi_pct=6.17,
                payback_period_months=194.5
            )
        }

        # Test save operation
        success = analyzer.save_results_to_database(cost_results)
        assert success is True

        # Verify file was created
        output_path = Path("data/results/cost_savings_analysis.csv")
        if output_path.exists():
            # Read and validate saved data
            saved_df = pd.read_csv(output_path)
            assert len(saved_df) > 0
            assert 'tool_id' in saved_df.columns
            assert 'current_cpu' in saved_df.columns
            assert 'annual_savings_usd' in saved_df.columns


class TestToolCostData:
    """Test ToolCostData dataclass"""

    def test_tool_cost_data_creation(self):
        """Test ToolCostData creation and attributes"""
        tool_data = ToolCostData(
            tool_id='TEST_TOOL',
            cost_new=100.0,
            cost_regrind=25.0,
            max_regrinds=2
        )

        assert tool_data.tool_id == 'TEST_TOOL'
        assert tool_data.cost_new == 100.0
        assert tool_data.cost_regrind == 25.0
        assert tool_data.max_regrinds == 2


class TestProductionMetrics:
    """Test ProductionMetrics dataclass"""

    def test_production_metrics_creation(self):
        """Test ProductionMetrics creation and defaults"""
        metrics = ProductionMetrics(
            daily_production_pieces=600,
            current_tool_life_pieces=5000,
            optimized_tool_life_pieces=18000
        )

        assert metrics.daily_production_pieces == 600
        assert metrics.current_tool_life_pieces == 5000
        assert metrics.optimized_tool_life_pieces == 18000
        assert metrics.working_days_per_year == 250  # Default value

        # Test with custom working days
        custom_metrics = ProductionMetrics(
            daily_production_pieces=400,
            current_tool_life_pieces=3000,
            optimized_tool_life_pieces=12000,
            working_days_per_year=260
        )

        assert custom_metrics.working_days_per_year == 260


class TestCostSavingsResult:
    """Test CostSavingsResult dataclass"""

    def test_cost_savings_result_creation(self):
        """Test CostSavingsResult creation and attributes"""
        result = CostSavingsResult(
            tool_id='TEST_TOOL',
            current_cpu=0.050,
            optimized_cpu=0.020,
            cpu_reduction_pct=60.0,
            tool_life_improvement_pct=150.0,
            annual_savings_usd=5000.0,
            roi_pct=10.0,
            payback_period_months=120.0
        )

        assert result.tool_id == 'TEST_TOOL'
        assert result.current_cpu == 0.050
        assert result.optimized_cpu == 0.020
        assert result.cpu_reduction_pct == 60.0
        assert result.tool_life_improvement_pct == 150.0
        assert result.annual_savings_usd == 5000.0
        assert result.roi_pct == 10.0
        assert result.payback_period_months == 120.0


class TestCostAnalysisEdgeCases:
    """Test edge cases and error conditions"""

    def test_missing_production_data_file(self):
        """Test handling of missing production data file"""
        analyzer = CostAnalyzer()

        with pytest.raises(FileNotFoundError):
            analyzer.load_production_data("nonexistent_file.csv")

    def test_invalid_production_data_columns(self, temp_data_files):
        """Test handling of invalid production data structure"""
        analyzer = CostAnalyzer()

        # Create invalid data (missing required columns)
        invalid_data = pd.DataFrame([
            {'tool_id': 'ZC1668', 'some_other_column': 123}
        ])

        invalid_path = temp_data_files['temp_dir'] + '/invalid_production.csv'
        invalid_data.to_csv(invalid_path, index=False)

        with pytest.raises(ValueError, match="Missing required columns"):
            analyzer.load_production_data(invalid_path)

    def test_zero_tool_life_scenarios(self):
        """Test handling of zero or very small tool life values"""
        analyzer = CostAnalyzer()

        # Test with very small tool life (should not cause division by zero)
        metrics = analyzer.calculate_cpu_metrics(
            tool_id='ZC1668',
            current_life=1,  # Very small value
            optimized_life=10,
            daily_production=500
        )

        assert metrics['current_cpu'] > 0
        assert metrics['optimized_cpu'] > 0
        assert not np.isnan(metrics['cpu_reduction_pct'])
        assert not np.isinf(metrics['cpu_reduction_pct'])

    def test_negative_annual_savings_roi(self):
        """Test ROI calculation with negative scenario"""
        analyzer = CostAnalyzer()

        # Test with negative savings (cost increase)
        roi_metrics = analyzer.calculate_roi_metrics(
            annual_savings=-1000,  # Cost increase
            implementation_cost=50000
        )

        assert roi_metrics['roi_pct'] < 0
        assert roi_metrics['payback_period_months'] == float('inf')
        assert roi_metrics['net_present_value_3yr'] < -50000

    def test_very_high_implementation_cost(self):
        """Test with unrealistically high implementation costs"""
        analyzer = CostAnalyzer()

        roi_metrics = analyzer.calculate_roi_metrics(
            annual_savings=5000,
            implementation_cost=1000000  # Very high cost
        )

        assert roi_metrics['roi_pct'] < 1  # Very low ROI
        assert roi_metrics['payback_period_months'] > 1000  # Very long payback

    def test_mathematical_precision_edge_cases(self):
        """Test mathematical precision with edge case values"""
        analyzer = CostAnalyzer()

        # Test with very small CPU values
        small_metrics = analyzer.calculate_cpu_metrics(
            tool_id='ZC1668',
            current_life=1000000,  # Very high tool life
            optimized_life=2000000,
            daily_production=1
        )

        assert small_metrics['current_cpu'] > 0
        assert small_metrics['optimized_cpu'] > 0
        assert small_metrics['cpu_reduction_pct'] > 0
        assert small_metrics['cpu_reduction_pct'] < 100

    def test_data_consistency_validation(self, sample_production_data):
        """Test data consistency validation"""
        analyzer = CostAnalyzer()

        # Verify current configuration has higher CPU than optimized
        current_data = sample_production_data[
            sample_production_data['configuration'] == 'current'
        ]
        optimized_data = sample_production_data[
            sample_production_data['configuration'] == 'optimized'
        ]

        for tool_id in ['ZC1668', 'ZC1445']:
            current_tool = current_data[current_data['tool_id'] == tool_id]
            optimized_tool = optimized_data[optimized_data['tool_id'] == tool_id]

            if len(current_tool) > 0 and len(optimized_tool) > 0:
                avg_current_cpu = current_tool['cpu'].mean()
                avg_optimized_cpu = optimized_tool['cpu'].mean()

                # Current should be higher than optimized (worse performance)
                assert avg_current_cpu > avg_optimized_cpu, \
                    f"Tool {tool_id}: current CPU {avg_current_cpu} should be > optimized {avg_optimized_cpu}"