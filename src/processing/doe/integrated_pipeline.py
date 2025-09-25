"""
Integrated DOE Analysis Pipeline

This module provides a complete end-to-end DOE analysis pipeline that combines:
1. Yates algorithm for factor effect analysis
2. Cost savings analysis and ROI calculations
3. Business case report generation
4. Results persistence for dashboard consumption

Based on ZF Friedrichshafen AG case study for cutting tool optimization.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .yates_algorithm import YatesAlgorithm
from .costs import CostAnalyzer, CostSavingsResult


class DOEIntegratedPipeline:
    """
    Complete DOE analysis pipeline integrating Yates algorithm with cost analysis.

    This pipeline orchestrates the entire DOE analysis workflow:
    1. Load and validate experiment data
    2. Execute Yates algorithm analysis
    3. Calculate cost savings and ROI metrics
    4. Generate comprehensive business case report
    5. Save results for dashboard and reporting
    """

    def __init__(self):
        """Initialize integrated pipeline with Yates and Cost analyzers"""
        self.yates_analyzer = YatesAlgorithm()
        self.cost_analyzer = CostAnalyzer()
        self.pipeline_results = {}
        self.execution_metadata = {}

    def validate_input_data(self, experiment_data_path: str = None,
                           production_data_path: str = None) -> Dict[str, bool]:
        """
        Validate input data files for completeness and consistency.

        Parameters
        ----------
        experiment_data_path : str, optional
            Path to experiment results CSV
        production_data_path : str, optional
            Path to production data CSV

        Returns
        -------
        Dict[str, bool]
            Validation results for each data source
        """
        validation_results = {}

        # Validate experiment data
        try:
            exp_df = self.yates_analyzer.load_experiment_data(experiment_data_path)
            required_treatments = {'c', 'a', 'b', 'abc'}
            available_treatments = set(exp_df['yates_order'].unique())

            validation_results['experiment_data'] = required_treatments.issubset(available_treatments)
            validation_results['experiment_records'] = len(exp_df)

            if not validation_results['experiment_data']:
                missing = required_treatments - available_treatments
                print(f"⚠️ Missing treatments in experiment data: {missing}")

        except Exception as e:
            validation_results['experiment_data'] = False
            print(f"❌ Experiment data validation failed: {e}")

        # Validate production data
        try:
            prod_df = self.cost_analyzer.load_production_data(production_data_path)
            required_configs = {'current', 'optimized'}
            available_configs = set(prod_df['configuration'].unique())

            validation_results['production_data'] = required_configs.issubset(available_configs)
            validation_results['production_records'] = len(prod_df)

            if not validation_results['production_data']:
                missing = required_configs - available_configs
                print(f"⚠️ Missing configurations in production data: {missing}")

        except Exception as e:
            validation_results['production_data'] = False
            print(f"❌ Production data validation failed: {e}")

        return validation_results

    def execute_doe_analysis(self, experiment_data_path: str = None) -> Dict[str, any]:
        """
        Execute complete Yates algorithm analysis.

        Parameters
        ----------
        experiment_data_path : str, optional
            Path to experiment data file

        Returns
        -------
        Dict[str, any]
            Complete DOE analysis results including effects and optimal conditions
        """
        print("🧪 Starting Yates Algorithm Analysis...")

        try:
            yates_results = self.yates_analyzer.analyze_doe_experiment(experiment_data_path)

            # Extract key metrics for integration
            significant_factors = yates_results['summary']['significant_factors']
            expected_improvement = yates_results['summary']['expected_improvement_pct']
            optimal_settings = yates_results['optimal_conditions']['recommended_settings']

            print(f"✅ DOE Analysis Complete - {len(significant_factors)} significant factors found")
            print(f"📈 Expected improvement: {expected_improvement:.1f}%")

            return yates_results

        except Exception as e:
            raise RuntimeError(f"DOE analysis failed: {str(e)}")

    def execute_cost_analysis(self, production_data_path: str = None,
                             yates_results: Dict = None) -> Dict[str, CostSavingsResult]:
        """
        Execute complete cost savings analysis.

        Parameters
        ----------
        production_data_path : str, optional
            Path to production data file
        yates_results : Dict, optional
            Results from Yates analysis for integration

        Returns
        -------
        Dict[str, CostSavingsResult]
            Cost savings analysis results by tool type
        """
        print("💰 Starting Cost Savings Analysis...")

        try:
            cost_results = self.cost_analyzer.analyze_cost_savings(
                yates_results=yates_results,
                data_source=production_data_path
            )

            # Calculate summary metrics
            total_savings = sum(result.annual_savings_usd for result in cost_results.values())
            avg_cpu_reduction = np.mean([result.cpu_reduction_pct for result in cost_results.values()])

            print(f"✅ Cost Analysis Complete - ${total_savings:,.0f} total annual savings")
            print(f"📉 Average CPU reduction: {avg_cpu_reduction:.1f}%")

            return cost_results

        except Exception as e:
            raise RuntimeError(f"Cost analysis failed: {str(e)}")

    def generate_integrated_report(self, yates_results: Dict,
                                  cost_results: Dict[str, CostSavingsResult]) -> str:
        """
        Generate comprehensive integrated business case report.

        Parameters
        ----------
        yates_results : Dict
            Results from Yates algorithm analysis
        cost_results : Dict[str, CostSavingsResult]
            Results from cost analysis

        Returns
        -------
        str
            Integrated business case report
        """
        report = []

        # Header
        report.append("🏭 INTEGRATED DOE ANALYSIS - COMPLETE BUSINESS CASE")
        report.append("=" * 70)
        report.append(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"🔬 Analysis Method: Yates Algorithm (2^(3-1) Fractional Factorial)")
        report.append("")

        # DOE Results Summary
        report.append("🧪 DESIGN OF EXPERIMENTS RESULTS")
        report.append("-" * 40)

        effects = yates_results.get('effects', {})
        optimal = yates_results.get('optimal_conditions', {})

        # Main effects
        main_effects = {k: v for k, v in effects.items() if k in ['A', 'B', 'AB']}
        for factor, effect in main_effects.items():
            significance = "✅ Significant" if abs(effect) > 100 else "❌ Not significant"
            report.append(f"   Factor {factor}: {effect:+.1f} pieces ({significance})")

        # Optimal conditions
        if 'recommended_settings' in optimal:
            settings = optimal['recommended_settings']
            report.append("")
            report.append("🎯 OPTIMAL PARAMETER SETTINGS:")
            report.append(f"   • Pressure: {settings.get('pressure_psi', 'N/A')} PSI")
            report.append(f"   • Concentration: {settings.get('concentration_pct', 'N/A')}%")
            report.append(f"   • RPM: {settings.get('rpm', 'N/A')}")
            report.append(f"   • Feed Rate: {settings.get('feed_rate', 'N/A')}")

        report.append("")

        # Cost Analysis Results
        report.append("💰 COST SAVINGS ANALYSIS")
        report.append("-" * 30)

        total_savings = sum(result.annual_savings_usd for result in cost_results.values())
        avg_cpu_reduction = np.mean([result.cpu_reduction_pct for result in cost_results.values()])
        avg_life_improvement = np.mean([result.tool_life_improvement_pct for result in cost_results.values()])
        avg_roi = np.mean([result.roi_pct for result in cost_results.values()])
        min_payback = min(result.payback_period_months for result in cost_results.values())

        report.append(f"💵 Total Annual Savings: ${total_savings:,.0f}")
        report.append(f"📉 Average CPU Reduction: {avg_cpu_reduction:.1f}%")
        report.append(f"🔧 Tool Life Improvement: {avg_life_improvement:.1f}%")
        report.append(f"💹 Average ROI: {avg_roi:.0f}%")
        report.append(f"⏰ Payback Period: {min_payback:.1f} months")
        report.append("")

        # Detailed tool analysis
        report.append("🔍 DETAILED ANALYSIS BY TOOL")
        report.append("-" * 35)

        for tool_id, result in cost_results.items():
            report.append(f"")
            report.append(f"🛠️  Tool {tool_id}:")
            report.append(f"   Current CPU: ${result.current_cpu:.5f}/piece")
            report.append(f"   Optimized CPU: ${result.optimized_cpu:.5f}/piece")
            report.append(f"   Improvement: {result.cpu_reduction_pct:.1f}% CPU reduction")
            report.append(f"   Tool Life: +{result.tool_life_improvement_pct:.1f}%")
            report.append(f"   Annual Value: ${result.annual_savings_usd:,.0f}")
            report.append(f"   ROI: {result.roi_pct:.0f}% (Payback: {result.payback_period_months:.1f} months)")

        report.append("")

        # Business recommendation
        report.append("📊 EXECUTIVE RECOMMENDATION")
        report.append("-" * 35)
        report.append("✅ IMMEDIATE IMPLEMENTATION RECOMMENDED")
        report.append(f"   • Expected ROI: {avg_roi:.0f}% with {min_payback:.1f} month payback")
        report.append(f"   • Annual value creation: ${total_savings:,.0f}")
        report.append(f"   • Production efficiency gain: {avg_cpu_reduction:.1f}%")
        report.append(f"   • Equipment lifecycle extension: {avg_life_improvement:.1f}%")

        report.append("")
        report.append("🎯 NEXT STEPS:")
        report.append("   1. Implement optimal parameter settings on production lines")
        report.append("   2. Monitor tool performance with new parameters for 30 days")
        report.append("   3. Scale implementation across all compatible machining centers")
        report.append("   4. Establish continuous monitoring dashboard")

        return "\n".join(report)

    def save_integrated_results(self, yates_results: Dict, cost_results: Dict[str, CostSavingsResult],
                               integrated_report: str) -> Dict[str, str]:
        """
        Save all analysis results to files for dashboard and reporting.

        Parameters
        ----------
        yates_results : Dict
            DOE analysis results
        cost_results : Dict[str, CostSavingsResult]
            Cost analysis results
        integrated_report : str
            Complete business case report

        Returns
        -------
        Dict[str, str]
            Paths to saved files
        """
        results_dir = Path("data/results")
        results_dir.mkdir(exist_ok=True)

        saved_files = {}

        try:
            # Save cost results (already implemented in cost_analyzer)
            cost_saved = self.cost_analyzer.save_results_to_database(cost_results)
            if cost_saved:
                saved_files['cost_analysis'] = str(results_dir / "cost_savings_analysis.csv")

            # Save DOE effects results
            effects_data = []
            analysis_date = datetime.now().date()

            for factor_name, effect_value in yates_results.get('effects', {}).items():
                if factor_name != 'Overall_Mean':
                    effects_data.append({
                        'analysis_date': analysis_date,
                        'factor_name': factor_name,
                        'effect_value': effect_value,
                        'is_significant': factor_name in yates_results.get('summary', {}).get('significant_factors', [])
                    })

            if effects_data:
                effects_df = pd.DataFrame(effects_data)
                effects_path = results_dir / "doe_effects_analysis.csv"
                effects_df.to_csv(effects_path, index=False)
                saved_files['doe_effects'] = str(effects_path)
                print(f"✅ Saved DOE effects to {effects_path}")

            # Save integrated business report
            report_path = results_dir / f"business_case_report_{analysis_date}.txt"
            with open(report_path, 'w') as f:
                f.write(integrated_report)
            saved_files['business_report'] = str(report_path)
            print(f"✅ Saved business report to {report_path}")

            # Save optimal settings as JSON-like CSV
            optimal_settings = yates_results.get('optimal_conditions', {}).get('recommended_settings', {})
            if optimal_settings:
                settings_data = [{
                    'analysis_date': analysis_date,
                    'parameter': 'pressure_psi',
                    'optimal_value': optimal_settings.get('pressure_psi', 0),
                    'unit': 'PSI'
                }, {
                    'analysis_date': analysis_date,
                    'parameter': 'concentration_pct',
                    'optimal_value': optimal_settings.get('concentration_pct', 0),
                    'unit': '%'
                }, {
                    'analysis_date': analysis_date,
                    'parameter': 'rpm',
                    'optimal_value': optimal_settings.get('rpm', 0),
                    'unit': 'RPM'
                }, {
                    'analysis_date': analysis_date,
                    'parameter': 'feed_rate',
                    'optimal_value': optimal_settings.get('feed_rate', 0),
                    'unit': 'units/min'
                }]

                settings_df = pd.DataFrame(settings_data)
                settings_path = results_dir / "optimal_parameters.csv"
                settings_df.to_csv(settings_path, index=False)
                saved_files['optimal_parameters'] = str(settings_path)
                print(f"✅ Saved optimal parameters to {settings_path}")

            return saved_files

        except Exception as e:
            print(f"❌ Failed to save some results: {e}")
            return saved_files

    def run_complete_pipeline(self, experiment_data_path: str = None,
                             production_data_path: str = None) -> Dict[str, any]:
        """
        Execute the complete integrated DOE analysis pipeline.

        This is the main orchestration method that runs the entire analysis:
        1. Validate input data
        2. Execute DOE analysis (Yates algorithm)
        3. Execute cost savings analysis
        4. Generate integrated business case report
        5. Save all results

        Parameters
        ----------
        experiment_data_path : str, optional
            Path to experiment results CSV
        production_data_path : str, optional
            Path to production data CSV

        Returns
        -------
        Dict[str, any]
            Complete pipeline results including all analyses and file paths
        """
        pipeline_start = datetime.now()
        print("🚀 STARTING INTEGRATED DOE ANALYSIS PIPELINE")
        print("=" * 60)

        try:
            # Step 1: Validate input data
            print("\n📋 Step 1: Validating Input Data...")
            validation = self.validate_input_data(experiment_data_path, production_data_path)

            if not all([validation.get('experiment_data'), validation.get('production_data')]):
                raise RuntimeError("Data validation failed - cannot proceed with analysis")

            print(f"✅ Validation complete: {validation['experiment_records']} experiment records, "
                  f"{validation['production_records']} production records")

            # Step 2: Execute DOE Analysis
            print("\n🧪 Step 2: Executing DOE Analysis (Yates Algorithm)...")
            yates_results = self.execute_doe_analysis(experiment_data_path)

            # Step 3: Execute Cost Analysis
            print("\n💰 Step 3: Executing Cost Savings Analysis...")
            cost_results = self.execute_cost_analysis(production_data_path, yates_results)

            # Step 4: Generate Integrated Report
            print("\n📊 Step 4: Generating Integrated Business Case Report...")
            integrated_report = self.generate_integrated_report(yates_results, cost_results)

            # Step 5: Save Results
            print("\n💾 Step 5: Saving Results...")
            saved_files = self.save_integrated_results(yates_results, cost_results, integrated_report)

            # Calculate pipeline execution time
            pipeline_end = datetime.now()
            execution_time = (pipeline_end - pipeline_start).total_seconds()

            # Prepare final results
            pipeline_results = {
                'validation': validation,
                'yates_analysis': yates_results,
                'cost_analysis': cost_results,
                'integrated_report': integrated_report,
                'saved_files': saved_files,
                'execution_metadata': {
                    'start_time': pipeline_start.isoformat(),
                    'end_time': pipeline_end.isoformat(),
                    'execution_time_seconds': execution_time,
                    'status': 'success'
                }
            }

            self.pipeline_results = pipeline_results

            print("\n🎉 PIPELINE EXECUTION COMPLETE!")
            print(f"⏱️ Total execution time: {execution_time:.2f} seconds")
            print(f"📁 Results saved to {len(saved_files)} files")

            # Print executive summary
            print("\n" + "="*60)
            print("📈 EXECUTIVE SUMMARY")
            print("-" * 20)

            total_savings = sum(result.annual_savings_usd for result in cost_results.values())
            avg_cpu_reduction = np.mean([result.cpu_reduction_pct for result in cost_results.values()])

            print(f"💵 Annual Savings Potential: ${total_savings:,.0f}")
            print(f"📉 CPU Reduction: {avg_cpu_reduction:.1f}%")
            print(f"🎯 Recommendation: IMPLEMENT IMMEDIATELY")

            return pipeline_results

        except Exception as e:
            error_time = datetime.now()
            execution_time = (error_time - pipeline_start).total_seconds()

            error_results = {
                'status': 'failed',
                'error': str(e),
                'execution_metadata': {
                    'start_time': pipeline_start.isoformat(),
                    'error_time': error_time.isoformat(),
                    'execution_time_seconds': execution_time,
                    'status': 'failed'
                }
            }

            print(f"\n❌ PIPELINE EXECUTION FAILED")
            print(f"Error: {e}")
            print(f"Execution time before failure: {execution_time:.2f} seconds")

            raise RuntimeError(f"Integrated pipeline failed: {str(e)}")


if __name__ == "__main__":
    # Example usage and testing
    pipeline = DOEIntegratedPipeline()

    try:
        print("🏭 Starting Integrated DOE Analysis Pipeline...")
        results = pipeline.run_complete_pipeline()

        print("\n📋 PIPELINE EXECUTION SUMMARY:")
        print(f"Status: {results['execution_metadata']['status']}")
        print(f"Execution time: {results['execution_metadata']['execution_time_seconds']:.2f}s")
        print(f"Files generated: {len(results.get('saved_files', {}))}")

        # Display business case summary
        if 'integrated_report' in results:
            report_lines = results['integrated_report'].split('\n')
            # Show first 20 lines of report as preview
            print("\n📊 BUSINESS CASE PREVIEW:")
            for line in report_lines[:20]:
                print(line)
            print("... (full report saved to file)")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")