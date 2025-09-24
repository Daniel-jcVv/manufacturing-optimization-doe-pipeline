import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import random


class DOEDataSimulator:
	"""Generate synthetic DOE experiment and production datasets.

	This simulator creates two datasets:
	- experiment_results.csv: fractional factorial 2^(3-1) with 3
	replicas per condition
	- production_data.csv: daily production metrics for current vs
	optimized configuration
	"""

	def __init__(self, out_dir: str):
		self.out_dir = Path(out_dir)
		(self.out_dir / "raw").mkdir(parents=True, exist_ok=True)

	def simulate_experiments(self, noise_factor: float = 0.1) -> pd.DataFrame:
		"""Simulate fractional factorial 2^(3-1) experiment results.

		Parameters
		----------
		noise_factor: float
			Relative noise to add to base life values to look more realistic.
		"""
		design = [
			{"run": 1, "A": -1, "B": -1, "C": 1, "order": "c"},
			{"run": 2, "A": 1, "B": -1, "C": -1, "order": "a"},
			{"run": 3, "A": -1, "B": 1, "C": -1, "order": "b"},
			{"run": 4, "A": 1, "B": 1, "C": 1, "order": "abc"},
		]

		# Base tool life per Yates order from case study
		base = {
			"c": 4000,
			"a": 5500,
			"b": 15000,
			"abc": 9000,
		}

		factors = {
			"A": {"levels": [750, 1050]},
			"B": {"levels": [4, 6]},
			"C": {"levels": [(3183, 605), (3700, 1050)]},
		}
		tools = ["ZC1668", "ZC1445"]
		rows = []

		for cond in design:
			for tool in tools:
				for replica in range(1, 4):
					life_mu = base[cond["order"]]
					life_sigma = life_mu * noise_factor
					life = max(100, int(np.random.normal(life_mu, life_sigma)))

					pressure = (
						factors["A"]["levels"][0]
						if cond["A"] == -1
						else factors["A"]["levels"][1]
					)
					concentration = (
						factors["B"]["levels"][0]
						if cond["B"] == -1
						else factors["B"]["levels"][1]
					)
					rpm, feed = (
						factors["C"]["levels"][0]
						if cond["C"] == -1
						else factors["C"]["levels"][1]
					)

					rows.append(
						{
							"experiment_id": f"EXP_{cond['run']:03d}_{tool}_{replica}",
							"timestamp": (
								datetime.now() - timedelta(days=random.randint(1, 7))
							),
							"run_number": cond["run"],
							"yates_order": cond["order"],
							"tool_id": tool,
							"replica": replica,
							"factor_A": cond["A"],
							"factor_B": cond["B"],
							"factor_C": cond["C"],
							"pressure_psi": pressure,
							"concentration_pct": concentration,
							"rpm": rpm,
							"feed_rate": feed,
							"tool_life_pieces": life,
						}
					)

		df = pd.DataFrame(rows)
		df.to_csv(self.out_dir / "raw" / "experiment_results.csv", index=False)
		return df

	def simulate_production(self, days: int = 45) -> pd.DataFrame:
		"""Simulate daily production and costs for current vs optimized config.

		Parameters
		----------
		days: int
			Number of days to simulate.
		"""
		rng = np.random.default_rng(42)
		tools = ["ZC1668", "ZC1445"]
		rows = []

		for d in range(days):
			date = (datetime.now().date() - timedelta(days=days - d))
			optimized = d > days / 2
			for tool in tools:
				daily = int(rng.poisson(500))
				tool_life = (
					rng.normal(10000, 1000) if optimized else rng.normal(4000, 500)
				)
				failures = float(rng.poisson(0.5 if optimized else 2.0))
				tools_used = max(1e-6, daily / max(100, tool_life))
				cost_new = tools_used * (148.94 if tool == "ZC1668" else 206.76)
				cost_regrind = (
					tools_used * (30.67 if tool == "ZC1668" else 47.14) * 0.3
				)
				rows.append(
					{
						"date": date,
						"tool_id": tool,
						"configuration": "optimized" if optimized else "current",
						"daily_production": daily,
						"tool_changes": int(tools_used),
						"failures": failures,
						"cost_new_tools": float(cost_new),
						"cost_resharpening": float(cost_regrind),
						"total_cost": float(cost_new + cost_regrind),
						"cpu": float((cost_new + cost_regrind) / max(1, daily)),
					}
				)

		df = pd.DataFrame(rows)
		(self.out_dir / "raw").mkdir(parents=True, exist_ok=True)
		df.to_csv(self.out_dir / "raw" / "production_data.csv", index=False)
		return df


if __name__ == "__main__":
	sim = DOEDataSimulator(out_dir="./data")
	sim.simulate_experiments()
	sim.simulate_production()