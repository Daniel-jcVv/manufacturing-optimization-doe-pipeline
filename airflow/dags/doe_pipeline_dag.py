"""
DOE Analysis Pipeline DAG

This DAG orchestrates the complete Design of Experiments analysis pipeline:
1. Data validation and preprocessing
2. DOE analysis using Yates algorithm
3. Cost savings analysis and ROI calculations
4. Business case report generation
5. Results persistence for dashboard consumption

Based on ZF Friedrichshafen AG case study for cutting tool optimization.
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable

# Add src to Python path for imports
sys.path.append('/opt/airflow/src')

# Import our pipeline modules
from processing.doe.integrated_pipeline import DOEIntegratedPipeline
from processing.doe.yates_algorithm import YatesAlgorithm
from processing.doe.costs import CostAnalyzer


def validate_input_data(**context):
    """
    Validate experiment and production data files exist and are valid.

    Returns
    -------
    dict
        Validation results with file paths and record counts
    """
    print("🔍 Starting data validation...")

    pipeline = DOEIntegratedPipeline()

    # Get data paths from Airflow variables or use defaults
    experiment_data_path = Variable.get("experiment_data_path",
                                       default_var="/opt/airflow/data/raw/experiment_results.csv")
    production_data_path = Variable.get("production_data_path",
                                       default_var="/opt/airflow/data/raw/production_data.csv")

    print(f"📁 Experiment data: {experiment_data_path}")
    print(f"📁 Production data: {production_data_path}")

    # Validate data
    validation_results = pipeline.validate_input_data(
        experiment_data_path=experiment_data_path,
        production_data_path=production_data_path
    )

    # Push results to XCom for downstream tasks
    context['task_instance'].xcom_push(key='validation_results', value=validation_results)
    context['task_instance'].xcom_push(key='experiment_data_path', value=experiment_data_path)
    context['task_instance'].xcom_push(key='production_data_path', value=production_data_path)

    # Check if validation passed
    if not all([validation_results.get('experiment_data'), validation_results.get('production_data')]):
        failed_validations = [k for k, v in validation_results.items() if v is False]
        raise ValueError(f"Data validation failed for: {failed_validations}")

    print(f"✅ Data validation successful!")
    print(f"   Experiment records: {validation_results.get('experiment_records', 'N/A')}")
    print(f"   Production records: {validation_results.get('production_records', 'N/A')}")

    return validation_results


def execute_doe_analysis(**context):
    """
    Execute DOE analysis using Yates algorithm.

    Returns
    -------
    dict
        DOE analysis results including effects and optimal conditions
    """
    print("🧪 Starting DOE analysis (Yates algorithm)...")

    # Get data path from upstream task
    experiment_data_path = context['task_instance'].xcom_pull(
        task_ids='validate_input_data', key='experiment_data_path'
    )

    pipeline = DOEIntegratedPipeline()

    # Execute DOE analysis
    yates_results = pipeline.execute_doe_analysis(experiment_data_path)

    # Log key results
    effects = yates_results.get('effects', {})
    significant_factors = yates_results.get('summary', {}).get('significant_factors', [])

    print(f"✅ DOE Analysis completed!")
    print(f"   Calculated effects: {list(effects.keys())}")
    print(f"   Significant factors: {significant_factors}")
    print(f"   Overall mean: {effects.get('Overall_Mean', 'N/A'):.1f}")

    # Push results to XCom
    context['task_instance'].xcom_push(key='yates_results', value=yates_results)

    return yates_results


def execute_cost_analysis(**context):
    """
    Execute cost savings analysis and ROI calculations.

    Returns
    -------
    dict
        Cost analysis results by tool type
    """
    print("💰 Starting cost savings analysis...")

    # Get data from upstream tasks
    production_data_path = context['task_instance'].xcom_pull(
        task_ids='validate_input_data', key='production_data_path'
    )
    yates_results = context['task_instance'].xcom_pull(
        task_ids='execute_doe_analysis', key='yates_results'
    )

    pipeline = DOEIntegratedPipeline()

    # Execute cost analysis
    cost_results = pipeline.execute_cost_analysis(
        production_data_path=production_data_path,
        yates_results=yates_results
    )

    # Log key results
    total_savings = sum(result.annual_savings_usd for result in cost_results.values())
    avg_cpu_reduction = sum(result.cpu_reduction_pct for result in cost_results.values()) / len(cost_results)

    print(f"✅ Cost analysis completed!")
    print(f"   Tools analyzed: {list(cost_results.keys())}")
    print(f"   Total annual savings: ${total_savings:,.0f}")
    print(f"   Average CPU reduction: {avg_cpu_reduction:.1f}%")

    # Push results to XCom
    context['task_instance'].xcom_push(key='cost_results', value=cost_results)

    return cost_results


def generate_integrated_report(**context):
    """
    Generate comprehensive business case report.

    Returns
    -------
    str
        Complete business case report
    """
    print("📊 Generating integrated business case report...")

    # Get results from upstream tasks
    yates_results = context['task_instance'].xcom_pull(
        task_ids='execute_doe_analysis', key='yates_results'
    )
    cost_results = context['task_instance'].xcom_pull(
        task_ids='execute_cost_analysis', key='cost_results'
    )

    pipeline = DOEIntegratedPipeline()

    # Generate integrated report
    integrated_report = pipeline.generate_integrated_report(yates_results, cost_results)

    # Log report metrics
    lines = integrated_report.split('\n')
    report_sections = [line for line in lines if line.startswith('##') or line.startswith('🏭')]

    print(f"✅ Integrated report generated!")
    print(f"   Report length: {len(lines)} lines")
    print(f"   Main sections: {len(report_sections)}")

    # Push report to XCom
    context['task_instance'].xcom_push(key='integrated_report', value=integrated_report)

    return integrated_report


def save_results_to_storage(**context):
    """
    Save all analysis results to persistent storage.

    Returns
    -------
    dict
        Dictionary with paths to saved files
    """
    print("💾 Saving results to persistent storage...")

    # Get results from upstream tasks
    yates_results = context['task_instance'].xcom_pull(
        task_ids='execute_doe_analysis', key='yates_results'
    )
    cost_results = context['task_instance'].xcom_pull(
        task_ids='execute_cost_analysis', key='cost_results'
    )
    integrated_report = context['task_instance'].xcom_pull(
        task_ids='generate_integrated_report', key='integrated_report'
    )

    pipeline = DOEIntegratedPipeline()

    # Save all results
    saved_files = pipeline.save_integrated_results(
        yates_results=yates_results,
        cost_results=cost_results,
        integrated_report=integrated_report
    )

    print(f"✅ Results saved successfully!")
    for file_type, file_path in saved_files.items():
        print(f"   {file_type}: {file_path}")

    # Push file paths to XCom
    context['task_instance'].xcom_push(key='saved_files', value=saved_files)

    return saved_files


def send_completion_notification(**context):
    """
    Send notification about pipeline completion with key metrics.
    """
    print("📧 Sending pipeline completion notification...")

    # Get final results
    cost_results = context['task_instance'].xcom_pull(
        task_ids='execute_cost_analysis', key='cost_results'
    )
    saved_files = context['task_instance'].xcom_pull(
        task_ids='save_results_to_storage', key='saved_files'
    )

    if cost_results:
        total_savings = sum(result.annual_savings_usd for result in cost_results.values())
        avg_cpu_reduction = sum(result.cpu_reduction_pct for result in cost_results.values()) / len(cost_results)

        notification_summary = f"""
        🎉 DOE Analysis Pipeline Completed Successfully!

        📊 Key Results:
        - Total Annual Savings: ${total_savings:,.0f}
        - Average CPU Reduction: {avg_cpu_reduction:.1f}%
        - Tools Analyzed: {len(cost_results)}
        - Files Generated: {len(saved_files)}

        📁 Generated Files:
        {chr(10).join(f"  - {file_type}: {file_path}" for file_type, file_path in saved_files.items())}

        ✅ Business case report ready for executive review.
        """

        print(notification_summary)

        # In production, this would send email/Slack notification
        # For now, we just log the summary

    return "notification_sent"


# DAG configuration
default_args = {
    'owner': 'doe-pipeline',
    'depends_on_past': False,
    'start_date': datetime(2025, 9, 25),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=1),
}

# Create DAG
dag = DAG(
    'doe_analysis_pipeline',
    default_args=default_args,
    description='Complete DOE analysis pipeline with Yates algorithm and cost analysis',
    schedule_interval='0 6 * * 1',  # Weekly on Monday at 6 AM
    start_date=datetime(2025, 9, 25),
    catchup=False,
    max_active_runs=1,
    tags=['doe', 'analysis', 'production', 'yates', 'cost-optimization'],
)

# Task definitions
start_pipeline = DummyOperator(
    task_id='start_pipeline',
    dag=dag,
)

# Data validation task
validate_data = PythonOperator(
    task_id='validate_input_data',
    python_callable=validate_input_data,
    dag=dag,
    doc_md="""
    ### Data Validation Task

    Validates that required experiment and production data files exist and contain
    the expected structure for DOE analysis:

    - **Experiment data**: Must contain treatments {'c', 'a', 'b', 'abc'}
    - **Production data**: Must contain configurations {'current', 'optimized'}

    Fails pipeline early if data quality issues are detected.
    """,
)

# Analysis task group
with TaskGroup('analysis_tasks', dag=dag) as analysis_group:

    # DOE analysis task
    doe_analysis = PythonOperator(
        task_id='execute_doe_analysis',
        python_callable=execute_doe_analysis,
        dag=dag,
        doc_md="""
        ### DOE Analysis Task

        Executes the Yates algorithm for 2^(3-1) fractional factorial design:

        - Calculates main effects for factors A, B, C
        - Determines statistical significance
        - Identifies optimal parameter settings
        - Generates factor effect rankings
        """,
    )

    # Cost analysis task (can run in parallel with reporting)
    cost_analysis = PythonOperator(
        task_id='execute_cost_analysis',
        python_callable=execute_cost_analysis,
        dag=dag,
        doc_md="""
        ### Cost Analysis Task

        Calculates business impact metrics:

        - Cost Per Unit (CPU) reduction percentages
        - Annual savings projections
        - ROI and payback period calculations
        - Tool lifecycle improvements
        """,
    )

# Reporting task group
with TaskGroup('reporting_tasks', dag=dag) as reporting_group:

    # Generate business case report
    generate_report = PythonOperator(
        task_id='generate_integrated_report',
        python_callable=generate_integrated_report,
        dag=dag,
    )

    # Save results to storage
    save_results = PythonOperator(
        task_id='save_results_to_storage',
        python_callable=save_results_to_storage,
        dag=dag,
    )

# Notification task
send_notification = PythonOperator(
    task_id='send_completion_notification',
    python_callable=send_completion_notification,
    dag=dag,
    trigger_rule='all_success',
)

# Pipeline completion marker
end_pipeline = DummyOperator(
    task_id='end_pipeline',
    dag=dag,
    trigger_rule='all_done',  # Runs regardless of upstream success/failure
)

# Define task dependencies
start_pipeline >> validate_data >> analysis_group
analysis_group >> reporting_group >> send_notification >> end_pipeline

# Within analysis group: DOE analysis must complete before cost analysis
doe_analysis >> cost_analysis

# Within reporting group: both tasks can run in parallel, but save_results needs both analyses
generate_report >> save_results