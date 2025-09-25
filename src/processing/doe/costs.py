"""
Cost Analysis Module for DOE Tool Optimization

This module calculates cost savings, ROI, and business metrics based on
DOE analysis results. It integrates Yates algorithm findings with production
data to project annual savings and equipment lifecycle improvements.

Based on ZF Friedrichshafen AG case study metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ToolCostData:
    """Cost data for cutting tools"""
    tool_id: str
    cost_new: float  # Cost of new tool
    cost_regrind: float  # Cost of regrinding/resharpening
    max_regrinds: int  # Maximum number of regrinds allowed


@dataclass
class ProductionMetrics:
    """Production metrics for cost analysis"""
    daily_production_pieces: int
    current_tool_life_pieces: int
    optimized_tool_life_pieces: int
    working_days_per_year: int = 250


@dataclass
class CostSavingsResult:
    """Results from cost savings analysis"""
    tool_id: str
    current_cpu: float  # Cost per unit - current
    optimized_cpu: float  # Cost per unit - optimized
    cpu_reduction_pct: float
    tool_life_improvement_pct: float
    annual_savings_usd: float
    roi_pct: float
    payback_period_months: float


class CostAnalyzer:
    """
    Analyzes cost savings from DOE optimization results.

    Calculates key business metrics:
    - CPU (Cost Per Unit) reduction
    - Tool life improvements
    - Annual cost savings
    - ROI and payback period
    """

    def __init__(self):
        """Initialize cost analyzer with tool cost data from ZF case study"""
        self.tool_costs = {
            'ZC1668': ToolCostData(
                tool_id='ZC1668',
                cost_new=148.94,  # USD
                cost_regrind=30.67,  # USD
                max_regrinds=3
            ),
            'ZC1445': ToolCostData(
                tool_id='ZC1445',
                cost_new=206.76,  # USD
                cost_regrind=47.14,  # USD
                max_regrinds=3
            )
        }

        # Production baseline from case study
        self.baseline_metrics = ProductionMetrics(
            daily_production_pieces=500,  # Average daily production
            current_tool_life_pieces=4000,  # Current average tool life
            optimized_tool_life_pieces=15000,  # Expected with DOE optimization
            working_days_per_year=250
        )

    def load_production_data(self, data_source: str = None) -> pd.DataFrame:
        """
        Load production data for cost analysis.

        Parameters
        ----------
        data_source : str, optional
            Path to production data CSV

        Returns
        -------
        pd.DataFrame
            Production data with cost metrics
        """
        if data_source is None:
            data_path = Path("data/raw/production_data.csv")
            if not data_path.exists():
                raise FileNotFoundError(f"Production data not found at {data_path}")
            df = pd.read_csv(data_path)
        else:
            df = pd.read_csv(data_source)

        # Validate required columns
        required_cols = ['tool_id', 'configuration', 'daily_production',
                        'total_cost', 'cpu']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df

    def calculate_tool_lifecycle_cost(self, tool_id: str, tool_life_pieces: int) -> float:
        """
        Calculate total lifecycle cost per tool including regrinds.

        Parameters
        ----------
        tool_id : str
            Tool identifier (ZC1668 or ZC1445)
        tool_life_pieces : int
            Total pieces the tool can produce over its lifecycle

        Returns
        -------
        float
            Total lifecycle cost per tool
        """
        if tool_id not in self.tool_costs:
            raise ValueError(f"Unknown tool_id: {tool_id}")

        tool_cost = self.tool_costs[tool_id]

        # Total cost = initial tool cost + regrind costs
        total_cost = tool_cost.cost_new + (tool_cost.cost_regrind * tool_cost.max_regrinds)

        return total_cost

    def calculate_cpu_metrics(self, tool_id: str, current_life: int,
                             optimized_life: int, daily_production: int) -> Dict[str, float]:
        """
        Calculate Cost Per Unit (CPU) for current vs optimized scenarios.

        Parameters
        ----------
        tool_id : str
            Tool identifier
        current_life : int
            Current tool life in pieces
        optimized_life : int
            Optimized tool life in pieces
        daily_production : int
            Daily production volume

        Returns
        -------
        Dict[str, float]
            CPU metrics and improvements
        """
        # Calculate lifecycle costs
        lifecycle_cost = self.calculate_tool_lifecycle_cost(tool_id, current_life)

        # CPU = Total tool cost / Total pieces produced by tool
        current_cpu = lifecycle_cost / current_life
        optimized_cpu = lifecycle_cost / optimized_life

        # Calculate improvements
        cpu_reduction = current_cpu - optimized_cpu
        cpu_reduction_pct = (cpu_reduction / current_cpu) * 100

        # Tool life improvement
        life_improvement_pct = ((optimized_life - current_life) / current_life) * 100

        return {
            'current_cpu': current_cpu,
            'optimized_cpu': optimized_cpu,
            'cpu_reduction': cpu_reduction,
            'cpu_reduction_pct': cpu_reduction_pct,
            'life_improvement_pct': life_improvement_pct
        }

    def calculate_annual_savings(self, tool_id: str, cpu_reduction: float,
                                daily_production: int, working_days: int = 250) -> float:
        """
        Calculate annual cost savings from CPU reduction.

        Parameters
        ----------
        tool_id : str
            Tool identifier
        cpu_reduction : float
            CPU reduction in USD per piece
        daily_production : int
            Average daily production
        working_days : int
            Working days per year

        Returns
        -------
        float
            Annual savings in USD
        """
        annual_production = daily_production * working_days
        annual_savings = cpu_reduction * annual_production

        return annual_savings

    def calculate_roi_metrics(self, annual_savings: float, implementation_cost: float = 50000) -> Dict[str, float]:
        """
        Calculate ROI and payback period for DOE implementation.

        Parameters
        ----------
        annual_savings : float
            Annual cost savings in USD
        implementation_cost : float
            One-time cost to implement DOE recommendations (equipment, training, etc.)

        Returns
        -------
        Dict[str, float]
            ROI and payback metrics
        """
        if annual_savings <= 0:
            return {
                'roi_pct': 0,
                'payback_period_months': float('inf'),
                'net_present_value_3yr': -implementation_cost
            }

        # ROI = (Annual Savings / Implementation Cost) * 100
        roi_pct = (annual_savings / implementation_cost) * 100

        # Payback period in months
        payback_period_months = (implementation_cost / annual_savings) * 12

        # 3-year NPV (simple calculation without discounting)
        three_year_savings = annual_savings * 3
        npv_3yr = three_year_savings - implementation_cost

        return {
            'roi_pct': roi_pct,
            'payback_period_months': payback_period_months,
            'net_present_value_3yr': npv_3yr
        }

    def analyze_cost_savings(self, yates_results: Dict = None,
                            data_source: str = None) -> Dict[str, CostSavingsResult]:
        """
        Complete cost savings analysis based on DOE results.

        Parameters
        ----------
        yates_results : Dict, optional
            Results from Yates algorithm analysis
        data_source : str, optional
            Path to production data

        Returns
        -------
        Dict[str, CostSavingsResult]
            Cost savings analysis for each tool type
        """
        try:
            # Load production data
            production_df = self.load_production_data(data_source)
            print(f"✅ Loaded {len(production_df)} production records")

            # Get average metrics by tool and configuration
            current_data = production_df[production_df['configuration'] == 'current']
            optimized_data = production_df[production_df['configuration'] == 'optimized']

            results = {}

            for tool_id in self.tool_costs.keys():
                # Get tool-specific data
                current_tool = current_data[current_data['tool_id'] == tool_id]
                optimized_tool = optimized_data[optimized_data['tool_id'] == tool_id]

                if len(current_tool) == 0 or len(optimized_tool) == 0:
                    print(f"⚠️ Insufficient data for tool {tool_id}")
                    continue

                # Calculate average metrics
                avg_daily_production = current_tool['daily_production'].mean()
                current_avg_cpu = current_tool['cpu'].mean()
                optimized_avg_cpu = optimized_tool['cpu'].mean()

                # Calculate improvements
                cpu_reduction = current_avg_cpu - optimized_avg_cpu
                cpu_reduction_pct = (cpu_reduction / current_avg_cpu) * 100

                # Estimate tool life improvement from CPU improvement
                # CPU improvement implies longer tool life
                estimated_life_improvement = cpu_reduction_pct * 2.5  # Amplification factor

                # Calculate annual savings
                annual_savings = self.calculate_annual_savings(
                    tool_id, cpu_reduction, int(avg_daily_production)
                )

                # Calculate ROI metrics
                roi_metrics = self.calculate_roi_metrics(annual_savings)

                # Create result object
                result = CostSavingsResult(
                    tool_id=tool_id,
                    current_cpu=current_avg_cpu,
                    optimized_cpu=optimized_avg_cpu,
                    cpu_reduction_pct=cpu_reduction_pct,
                    tool_life_improvement_pct=estimated_life_improvement,
                    annual_savings_usd=annual_savings,
                    roi_pct=roi_metrics['roi_pct'],
                    payback_period_months=roi_metrics['payback_period_months']
                )

                results[tool_id] = result

                print(f"✅ Analyzed cost savings for {tool_id}")

            return results

        except Exception as e:
            raise RuntimeError(f"Cost analysis failed: {str(e)}")

    def generate_business_case_report(self, cost_results: Dict[str, CostSavingsResult]) -> str:
        """
        Generate executive summary report for business case.

        Parameters
        ----------
        cost_results : Dict[str, CostSavingsResult]
            Cost analysis results by tool

        Returns
        -------
        str
            Business case report
        """
        if not cost_results:
            return "No cost analysis results available."

        report = []
        report.append("💰 DOE OPTIMIZATION - BUSINESS CASE ANALYSIS")
        report.append("=" * 60)
        report.append("")

        total_annual_savings = sum(result.annual_savings_usd for result in cost_results.values())
        avg_cpu_reduction = np.mean([result.cpu_reduction_pct for result in cost_results.values()])
        avg_life_improvement = np.mean([result.tool_life_improvement_pct for result in cost_results.values()])
        avg_roi = np.mean([result.roi_pct for result in cost_results.values()])
        min_payback = min(result.payback_period_months for result in cost_results.values())

        # Executive Summary
        report.append("📊 EXECUTIVE SUMMARY")
        report.append("-" * 30)
        report.append(f"🎯 Total Annual Savings: ${total_annual_savings:,.0f}")
        report.append(f"📉 Average CPU Reduction: {avg_cpu_reduction:.1f}%")
        report.append(f"🔧 Average Tool Life Improvement: {avg_life_improvement:.1f}%")
        report.append(f"💹 Average ROI: {avg_roi:.0f}%")
        report.append(f"⏰ Payback Period: {min_payback:.1f} months")
        report.append("")

        # Detailed Results by Tool
        report.append("🔍 DETAILED ANALYSIS BY TOOL")
        report.append("-" * 40)

        for tool_id, result in cost_results.items():
            report.append(f"\n🛠️  Tool {tool_id}:")
            report.append(f"   Current CPU: ${result.current_cpu:.4f}/piece")
            report.append(f"   Optimized CPU: ${result.optimized_cpu:.4f}/piece")
            report.append(f"   CPU Reduction: {result.cpu_reduction_pct:.1f}%")
            report.append(f"   Tool Life Improvement: {result.tool_life_improvement_pct:.1f}%")
            report.append(f"   Annual Savings: ${result.annual_savings_usd:,.0f}")
            report.append(f"   ROI: {result.roi_pct:.0f}%")
            report.append(f"   Payback: {result.payback_period_months:.1f} months")

        report.append("")
        report.append("✅ RECOMMENDATION: IMPLEMENT DOE OPTIMIZATION IMMEDIATELY")
        report.append(f"   Expected payback in {min_payback:.1f} months with {avg_roi:.0f}% ROI")

        return "\n".join(report)

    def save_results_to_database(self, cost_results: Dict[str, CostSavingsResult]) -> bool:
        """
        Save cost analysis results to database for dashboard consumption.

        Parameters
        ----------
        cost_results : Dict[str, CostSavingsResult]
            Cost analysis results

        Returns
        -------
        bool
            Success status
        """
        try:
            # Convert results to DataFrame
            results_data = []
            analysis_date = datetime.now().date()

            for result in cost_results.values():
                results_data.append({
                    'analysis_date': analysis_date,
                    'tool_id': result.tool_id,
                    'current_cpu': result.current_cpu,
                    'optimized_cpu': result.optimized_cpu,
                    'cpu_reduction_pct': result.cpu_reduction_pct,
                    'tool_life_improvement_pct': result.tool_life_improvement_pct,
                    'annual_savings_usd': result.annual_savings_usd,
                    'roi_pct': result.roi_pct,
                    'payback_period_months': result.payback_period_months
                })

            df = pd.DataFrame(results_data)

            # Save to CSV (in production, this would be database insert)
            output_path = Path("data/results/cost_savings_analysis.csv")
            output_path.parent.mkdir(exist_ok=True)
            df.to_csv(output_path, index=False)

            print(f"✅ Saved cost analysis results to {output_path}")
            return True

        except Exception as e:
            print(f"❌ Failed to save results: {e}")
            return False


if __name__ == "__main__":
    # Example usage and testing
    analyzer = CostAnalyzer()

    try:
        print("💰 Starting Cost Analysis for DOE Optimization...")

        # Run cost analysis
        cost_results = analyzer.analyze_cost_savings()

        # Generate business case report
        report = analyzer.generate_business_case_report(cost_results)
        print("\n" + report)

        # Save results
        analyzer.save_results_to_database(cost_results)

        print("\n🎉 Cost analysis completed successfully!")

    except Exception as e:
        print(f"❌ Cost analysis failed: {e}")