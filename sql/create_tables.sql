-- DOE Tool Optimization - Database Schema
-- Creates tables for experiment results, production data, and cost savings

-- Experiment results from DOE fractional factorial design
CREATE TABLE IF NOT EXISTS experiment_results (
    experiment_id VARCHAR(50) PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    run_number INTEGER NOT NULL,
    yates_order VARCHAR(10) NOT NULL,
    tool_id VARCHAR(20) NOT NULL,
    replica INTEGER NOT NULL,
    factor_A INTEGER NOT NULL CHECK (factor_A IN (-1, 1)),
    factor_B INTEGER NOT NULL CHECK (factor_B IN (-1, 1)),
    factor_C INTEGER NOT NULL CHECK (factor_C IN (-1, 1)),
    pressure_psi FLOAT NOT NULL,
    concentration_pct FLOAT NOT NULL,
    rpm INTEGER NOT NULL,
    feed_rate INTEGER NOT NULL,
    tool_life_pieces INTEGER NOT NULL CHECK (tool_life_pieces > 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Production data for current vs optimized configurations
CREATE TABLE IF NOT EXISTS production_data (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    tool_id VARCHAR(20) NOT NULL,
    configuration VARCHAR(20) NOT NULL CHECK (configuration IN ('current', 'optimized')),
    daily_production INTEGER NOT NULL CHECK (daily_production >= 0),
    tool_changes INTEGER NOT NULL CHECK (tool_changes >= 0),
    failures FLOAT NOT NULL CHECK (failures >= 0),
    cost_new_tools FLOAT NOT NULL CHECK (cost_new_tools >= 0),
    cost_resharpening FLOAT NOT NULL CHECK (cost_resharpening >= 0),
    total_cost FLOAT NOT NULL CHECK (total_cost >= 0),
    cpu FLOAT NOT NULL CHECK (cpu >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, tool_id, configuration)
);

-- Cost savings analysis results
CREATE TABLE IF NOT EXISTS cost_savings (
    id SERIAL PRIMARY KEY,
    analysis_date DATE NOT NULL,
    tool_id VARCHAR(20) NOT NULL,
    current_cpu FLOAT NOT NULL,
    optimized_cpu FLOAT NOT NULL,
    cpu_reduction_pct FLOAT NOT NULL,
    tool_life_improvement_pct FLOAT NOT NULL,
    annual_savings_usd FLOAT NOT NULL,
    roi_pct FLOAT NOT NULL,
    payback_period_months FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(analysis_date, tool_id)
);

-- DOE analysis results from Yates algorithm
CREATE TABLE IF NOT EXISTS doe_analysis (
    id SERIAL PRIMARY KEY,
    analysis_date DATE NOT NULL,
    tool_id VARCHAR(20) NOT NULL,
    factor_name VARCHAR(10) NOT NULL,
    effect_value FLOAT NOT NULL,
    effect_rank INTEGER NOT NULL,
    is_significant BOOLEAN NOT NULL,
    optimal_level VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_experiment_results_tool_id ON experiment_results(tool_id);
CREATE INDEX IF NOT EXISTS idx_experiment_results_timestamp ON experiment_results(timestamp);
CREATE INDEX IF NOT EXISTS idx_production_data_date ON production_data(date);
CREATE INDEX IF NOT EXISTS idx_production_data_tool_config ON production_data(tool_id, configuration);
CREATE INDEX IF NOT EXISTS idx_cost_savings_analysis_date ON cost_savings(analysis_date);
CREATE INDEX IF NOT EXISTS idx_doe_analysis_tool_date ON doe_analysis(tool_id, analysis_date);