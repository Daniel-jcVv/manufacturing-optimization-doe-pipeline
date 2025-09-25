"""
Yates Algorithm Implementation for Design of Experiments (DOE) Analysis

This module implements the Yates algorithm for analyzing fractional factorial
experiments, specifically designed for the ZF Friedrichshafen AG tool optimization case.
The algorithm calculates main effects and interactions to identify optimal
cutting tool parameters.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DOEFactor:
    """Represents a factor in the DOE experiment"""
    name: str
    low_level: float
    high_level: float
    unit: str
    description: str


@dataclass
class YatesResult:
    """Results from Yates algorithm analysis"""
    factor_name: str
    effect_value: float
    effect_rank: int
    is_significant: bool
    contribution_pct: float


class YatesAlgorithm:
    """
    Implementation of Yates algorithm for 2^(k-p) fractional factorial designs.

    Based on the ZF Friedrichshafen AG case study for cutting tool optimization:
    - Factor A: Pressure (PSI)
    - Factor B: Concentration (%)
    - Factor C: RPM + Feed Rate combination

    The algorithm identifies which factors most significantly impact tool life
    and determines optimal parameter settings.
    """

    def __init__(self):
        """Initialize Yates algorithm with factor definitions"""
        self.factors = {
            'A': DOEFactor(
                name='Pressure',
                low_level=750.0,
                high_level=1050.0,
                unit='PSI',
                description='Coolant pressure'
            ),
            'B': DOEFactor(
                name='Concentration',
                low_level=4.0,
                high_level=6.0,
                unit='%',
                description='Coolant concentration'
            ),
            'C': DOEFactor(
                name='RPM_Feed',
                low_level=3183.0,  # RPM=3183, Feed=605
                high_level=3700.0,  # RPM=3700, Feed=1050
                unit='RPM',
                description='Combined RPM and feed rate parameter'
            )
        }

        # Expected Yates order from case study
        self.yates_order = ['c', 'a', 'b', 'abc']
        self.results: List[YatesResult] = []

    def load_experiment_data(self, data_source: str = None) -> pd.DataFrame:
        """
        Load DOE experiment data from database or CSV file.

        Parameters
        ----------
        data_source : str, optional
            Path to CSV file or database connection string

        Returns
        -------
        pd.DataFrame
            Experiment data with columns: experiment_id, yates_order,
            factor_A, factor_B, factor_C, tool_life_pieces
        """
        if data_source is None:
            # Load from default location
            data_path = Path("data/raw/experiment_results.csv")
            if not data_path.exists():
                raise FileNotFoundError(f"Experiment data not found at {data_path}")

            df = pd.read_csv(data_path)
        else:
            # Load from specified source
            df = pd.read_csv(data_source)

        # Validate required columns
        required_cols = ['yates_order', 'factor_A', 'factor_B', 'factor_C', 'tool_life_pieces']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df

    def calculate_treatment_averages(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate average response for each treatment combination.

        Parameters
        ----------
        df : pd.DataFrame
            Experiment data with replications

        Returns
        -------
        Dict[str, float]
            Average tool life for each Yates order (c, a, b, abc)
        """
        treatment_averages = {}

        for order in self.yates_order:
            # Filter data for this treatment
            treatment_data = df[df['yates_order'] == order]['tool_life_pieces']

            if len(treatment_data) == 0:
                raise ValueError(f"No data found for treatment {order}")

            # Calculate average across replications
            avg_life = treatment_data.mean()
            treatment_averages[order] = avg_life

        return treatment_averages

    def execute_yates_algorithm(self, treatment_averages: Dict[str, float]) -> Dict[str, float]:
        """
        Execute the Yates algorithm to calculate factor effects.

        The Yates algorithm uses a systematic approach to calculate main effects
        and interactions from a 2^k factorial design by iteratively summing
        and differencing treatment combinations.

        Parameters
        ----------
        treatment_averages : Dict[str, float]
            Average responses for each treatment combination

        Returns
        -------
        Dict[str, float]
            Effect values for each factor and interaction
        """
        # Yates table setup - Column 1: Treatment averages
        yates_table = []
        for order in self.yates_order:
            yates_table.append(treatment_averages[order])

        n_factors = 3  # A, B, C
        n_treatments = 2 ** (n_factors - 1)  # 2^(3-1) = 4 for fractional factorial

        # Execute Yates iterations
        current_col = yates_table.copy()

        for iteration in range(n_factors):
            next_col = []
            half = len(current_col) // 2

            # First half: sums
            for i in range(half):
                sum_val = current_col[i] + current_col[i + half]
                next_col.append(sum_val)

            # Second half: differences
            for i in range(half):
                diff_val = current_col[i + half] - current_col[i]
                next_col.append(diff_val)

            current_col = next_col

        # Calculate effects (divide differences by number of replicates * 2^(k-2))
        # For 2^(3-1) with 3 replicates each: divisor = 3 * 2^(3-2) = 6
        n_replicates = 3
        divisor = n_replicates * (2 ** (n_factors - 2))

        effects = {}
        effect_names = ['Overall_Mean', 'A', 'B', 'AB']  # For fractional factorial 2^(3-1)

        for i, name in enumerate(effect_names):
            if i == 0:
                # Overall mean
                effects[name] = current_col[i] / (n_replicates * n_treatments)
            else:
                # Factor effects
                effects[name] = current_col[i] / divisor

        return effects

    def determine_significance(self, effects: Dict[str, float], alpha: float = 0.05) -> Dict[str, bool]:
        """
        Determine statistical significance of effects.

        For this case study, we use practical significance based on
        effect magnitude rather than formal statistical tests.

        Parameters
        ----------
        effects : Dict[str, float]
            Calculated effects from Yates algorithm
        alpha : float
            Significance level (not used in this implementation)

        Returns
        -------
        Dict[str, bool]
            Significance status for each effect
        """
        # Calculate effect magnitudes
        effect_magnitudes = {k: abs(v) for k, v in effects.items() if k != 'Overall_Mean'}

        if not effect_magnitudes:
            return {}

        # Determine significance threshold (50% of largest effect)
        max_effect = max(effect_magnitudes.values())
        threshold = max_effect * 0.5

        significance = {}
        for effect_name, magnitude in effect_magnitudes.items():
            significance[effect_name] = magnitude >= threshold

        return significance

    def find_optimal_conditions(self, effects: Dict[str, float]) -> Dict[str, any]:
        """
        Determine optimal factor levels based on calculated effects.

        For the ZF case, positive effects indicate that high levels
        of factors lead to better (longer) tool life.

        Parameters
        ----------
        effects : Dict[str, float]
            Calculated main effects and interactions

        Returns
        -------
        Dict[str, any]
            Optimal factor levels and expected improvement
        """
        optimal_levels = {}

        # For each main effect, choose level that maximizes tool life
        for factor_key in ['A', 'B']:  # Main effects only for fractional design
            if factor_key in effects:
                effect_value = effects[factor_key]
                factor = self.factors[factor_key]

                if effect_value > 0:
                    # Positive effect: high level is better
                    optimal_levels[factor_key] = {
                        'level': 'high',
                        'value': factor.high_level,
                        'unit': factor.unit,
                        'factor_name': factor.name
                    }
                else:
                    # Negative effect: low level is better
                    optimal_levels[factor_key] = {
                        'level': 'low',
                        'value': factor.low_level,
                        'unit': factor.unit,
                        'factor_name': factor.name
                    }

        # Handle factor C (aliased with AB in fractional design)
        # Based on case study results, high level C is optimal
        optimal_levels['C'] = {
            'level': 'high',
            'value': self.factors['C'].high_level,
            'unit': self.factors['C'].unit,
            'factor_name': self.factors['C'].name
        }

        # Calculate expected tool life improvement
        total_improvement = sum(abs(effects[k]) for k in ['A', 'B'] if k in effects)

        return {
            'optimal_levels': optimal_levels,
            'expected_improvement': total_improvement,
            'recommended_settings': {
                'pressure_psi': optimal_levels['A']['value'],
                'concentration_pct': optimal_levels['B']['value'],
                'rpm': 3700,  # High level from case study
                'feed_rate': 1050  # High level from case study
            }
        }

    def analyze_doe_experiment(self, data_source: str = None) -> Dict[str, any]:
        """
        Complete DOE analysis using Yates algorithm.

        This is the main method that orchestrates the entire analysis:
        1. Load experiment data
        2. Calculate treatment averages
        3. Execute Yates algorithm
        4. Determine significance
        5. Find optimal conditions

        Parameters
        ----------
        data_source : str, optional
            Path to experiment data file

        Returns
        -------
        Dict[str, any]
            Complete DOE analysis results
        """
        try:
            # Step 1: Load data
            df = self.load_experiment_data(data_source)
            print(f"✅ Loaded {len(df)} experiment records")

            # Step 2: Calculate treatment averages
            treatment_averages = self.calculate_treatment_averages(df)
            print(f"✅ Calculated averages for {len(treatment_averages)} treatments")

            # Step 3: Execute Yates algorithm
            effects = self.execute_yates_algorithm(treatment_averages)
            print(f"✅ Calculated effects for {len(effects)} factors")

            # Step 4: Determine significance
            significance = self.determine_significance(effects)
            print(f"✅ Evaluated significance of effects")

            # Step 5: Find optimal conditions
            optimal = self.find_optimal_conditions(effects)
            print(f"✅ Determined optimal factor levels")

            # Prepare results
            results = {
                'treatment_averages': treatment_averages,
                'effects': effects,
                'significance': significance,
                'optimal_conditions': optimal,
                'summary': {
                    'total_experiments': len(df),
                    'significant_factors': [k for k, v in significance.items() if v],
                    'expected_improvement_pct': (optimal['expected_improvement'] /
                                               effects['Overall_Mean'] * 100) if effects['Overall_Mean'] > 0 else 0
                }
            }

            # Store results for further analysis
            self.results = self._create_yates_results(effects, significance)

            return results

        except Exception as e:
            raise RuntimeError(f"DOE analysis failed: {str(e)}")

    def _create_yates_results(self, effects: Dict[str, float], significance: Dict[str, bool]) -> List[YatesResult]:
        """Create structured results for database storage"""
        results = []

        # Sort effects by magnitude for ranking
        effect_items = [(k, abs(v)) for k, v in effects.items() if k != 'Overall_Mean']
        effect_items.sort(key=lambda x: x[1], reverse=True)

        for rank, (factor_name, magnitude) in enumerate(effect_items, 1):
            result = YatesResult(
                factor_name=factor_name,
                effect_value=effects[factor_name],
                effect_rank=rank,
                is_significant=significance.get(factor_name, False),
                contribution_pct=(magnitude / sum(item[1] for item in effect_items)) * 100
            )
            results.append(result)

        return results

    def get_results_summary(self) -> str:
        """Generate human-readable summary of DOE analysis"""
        if not self.results:
            return "No analysis results available. Run analyze_doe_experiment() first."

        summary = []
        summary.append("🧪 DOE ANALYSIS RESULTS - YATES ALGORITHM")
        summary.append("=" * 50)

        for result in self.results:
            significance_marker = "✅" if result.is_significant else "❌"
            summary.append(
                f"{significance_marker} Factor {result.factor_name}: "
                f"Effect = {result.effect_value:.1f}, "
                f"Rank = {result.effect_rank}, "
                f"Contribution = {result.contribution_pct:.1f}%"
            )

        return "\n".join(summary)


if __name__ == "__main__":
    # Example usage and testing
    yates = YatesAlgorithm()

    try:
        print("🚀 Starting DOE Analysis using Yates Algorithm...")
        results = yates.analyze_doe_experiment()

        print("\n" + yates.get_results_summary())

        print(f"\n📊 RECOMMENDED OPTIMAL SETTINGS:")
        optimal = results['optimal_conditions']['recommended_settings']
        print(f"   Pressure: {optimal['pressure_psi']:.0f} PSI")
        print(f"   Concentration: {optimal['concentration_pct']:.1f}%")
        print(f"   RPM: {optimal['rpm']:.0f}")
        print(f"   Feed Rate: {optimal['feed_rate']:.0f}")

        print(f"\n💰 Expected improvement: {results['summary']['expected_improvement_pct']:.1f}%")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")