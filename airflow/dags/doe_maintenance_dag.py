"""
DOE Pipeline Maintenance and Monitoring DAG

This DAG handles maintenance tasks for the DOE analysis pipeline:
1. Data quality monitoring
2. System health checks
3. Performance metrics collection
4. Results validation and archival
5. Database cleanup and optimization

Runs more frequently than the main analysis pipeline to ensure
system reliability and data quality.
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.email import EmailOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
from airflow.sensors.filesystem import FileSensor

# Add src to Python path for imports
sys.path.append('/opt/airflow/src')

# Import utility modules
try:
    from utils.database import test_connecton
    from processing.doe.integrated_pipeline import DOEIntegratedPipeline
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")


def check_data_freshness(**context):
    """
    Check if input data files are recent and valid.

    Returns
    -------
    dict
        Data freshness report with file ages and sizes
    """
    print("📅 Checking data freshness...")

    data_paths = [
        "/opt/airflow/data/raw/experiment_results.csv",
        "/opt/airflow/data/raw/production_data.csv"
    ]

    freshness_report = {}

    for data_path in data_paths:
        file_path = Path(data_path)
        if file_path.exists():
            stat = file_path.stat()
            age_hours = (datetime.now().timestamp() - stat.st_mtime) / 3600
            size_mb = stat.st_size / (1024 * 1024)

            freshness_report[data_path] = {
                'exists': True,
                'age_hours': age_hours,
                'size_mb': round(size_mb, 2),
                'is_fresh': age_hours < 168,  # Less than 7 days old
            }

            print(f"📁 {file_path.name}: {age_hours:.1f}h old, {size_mb:.2f}MB")
        else:
            freshness_report[data_path] = {
                'exists': False,
                'age_hours': None,
                'size_mb': None,
                'is_fresh': False,
            }
            print(f"❌ {file_path.name}: File not found")

    # Check if any files are stale
    stale_files = [path for path, info in freshness_report.items()
                   if not info['is_fresh']]

    if stale_files:
        print(f"⚠️ Warning: {len(stale_files)} files are stale")
        # In production, this might trigger data refresh
    else:
        print("✅ All data files are fresh")

    context['task_instance'].xcom_push(key='freshness_report', value=freshness_report)
    return freshness_report


def validate_database_connection(**context):
    """
    Test database connectivity and basic queries.

    Returns
    -------
    dict
        Database health status
    """
    print("🔗 Testing database connection...")

    try:
        # Test basic connection
        connection_result = test_connecton()

        db_health = {
            'connection_success': bool(connection_result),
            'connection_time': datetime.now().isoformat(),
            'status': 'healthy' if connection_result else 'failed'
        }

        if connection_result:
            print("✅ Database connection successful")
        else:
            print("❌ Database connection failed")
            raise Exception("Database connection test failed")

    except Exception as e:
        print(f"❌ Database error: {str(e)}")
        db_health = {
            'connection_success': False,
            'connection_time': datetime.now().isoformat(),
            'status': 'error',
            'error': str(e)
        }

    context['task_instance'].xcom_push(key='db_health', value=db_health)
    return db_health


def check_results_quality(**context):
    """
    Validate quality of recent analysis results.

    Returns
    -------
    dict
        Results quality assessment
    """
    print("🔍 Checking results quality...")

    results_dir = Path("/opt/airflow/data/results")
    quality_report = {
        'results_dir_exists': results_dir.exists(),
        'files_found': [],
        'quality_issues': [],
        'overall_status': 'unknown'
    }

    if not results_dir.exists():
        quality_report['quality_issues'].append("Results directory does not exist")
        quality_report['overall_status'] = 'failed'
        return quality_report

    # Check for expected result files
    expected_files = [
        'cost_savings_analysis.csv',
        'doe_effects_analysis.csv',
        'optimal_parameters.csv'
    ]

    for filename in expected_files:
        file_path = results_dir / filename
        if file_path.exists():
            try:
                # Basic file validation
                if filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    file_info = {
                        'filename': filename,
                        'exists': True,
                        'size_mb': round(file_path.stat().st_size / (1024 * 1024), 3),
                        'rows': len(df),
                        'columns': len(df.columns) if not df.empty else 0
                    }

                    # Basic data quality checks
                    if df.empty:
                        quality_report['quality_issues'].append(f"{filename} is empty")
                    elif filename == 'cost_savings_analysis.csv':
                        # Specific validation for cost analysis
                        required_cols = ['tool_id', 'annual_savings_usd', 'roi_pct']
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        if missing_cols:
                            quality_report['quality_issues'].append(
                                f"{filename} missing columns: {missing_cols}"
                            )

                    quality_report['files_found'].append(file_info)

                else:
                    # Non-CSV file (like business reports)
                    file_info = {
                        'filename': filename,
                        'exists': True,
                        'size_mb': round(file_path.stat().st_size / (1024 * 1024), 3)
                    }
                    quality_report['files_found'].append(file_info)

            except Exception as e:
                quality_report['quality_issues'].append(f"Error reading {filename}: {str(e)}")

        else:
            quality_report['quality_issues'].append(f"Missing expected file: {filename}")

    # Determine overall status
    if not quality_report['quality_issues']:
        quality_report['overall_status'] = 'healthy'
    elif len(quality_report['quality_issues']) <= 2:
        quality_report['overall_status'] = 'warning'
    else:
        quality_report['overall_status'] = 'failed'

    print(f"📊 Results quality: {quality_report['overall_status']}")
    print(f"   Files found: {len(quality_report['files_found'])}")
    print(f"   Issues detected: {len(quality_report['quality_issues'])}")

    context['task_instance'].xcom_push(key='quality_report', value=quality_report)
    return quality_report


def collect_performance_metrics(**context):
    """
    Collect system performance metrics.

    Returns
    -------
    dict
        Performance metrics report
    """
    print("📈 Collecting performance metrics...")

    import psutil
    import os

    # System metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/opt/airflow')

    # Airflow-specific metrics
    airflow_home = os.environ.get('AIRFLOW__CORE__DAGS_FOLDER', '/opt/airflow/dags')
    dags_folder = Path(airflow_home)

    performance_metrics = {
        'timestamp': datetime.now().isoformat(),
        'system': {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': round(memory.available / (1024**3), 2),
            'disk_free_gb': round(disk.free / (1024**3), 2),
            'disk_percent': (disk.used / disk.total) * 100
        },
        'airflow': {
            'dags_folder_exists': dags_folder.exists(),
            'dag_files_count': len(list(dags_folder.glob('*.py'))) if dags_folder.exists() else 0,
        },
        'data_directories': {}
    }

    # Check data directories sizes
    data_dirs = [
        '/opt/airflow/data/raw',
        '/opt/airflow/data/results'
    ]

    for data_dir in data_dirs:
        dir_path = Path(data_dir)
        if dir_path.exists():
            total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
            performance_metrics['data_directories'][data_dir] = {
                'exists': True,
                'size_mb': round(total_size / (1024**2), 2),
                'file_count': len(list(dir_path.rglob('*')))
            }
        else:
            performance_metrics['data_directories'][data_dir] = {'exists': False}

    print(f"💻 System metrics: CPU {cpu_percent}%, Memory {memory.percent}%, Disk {disk.percent:.1f}%")

    context['task_instance'].xcom_push(key='performance_metrics', value=performance_metrics)
    return performance_metrics


def cleanup_old_results(**context):
    """
    Clean up old result files to manage disk space.

    Returns
    -------
    dict
        Cleanup summary
    """
    print("🧹 Cleaning up old result files...")

    results_dir = Path("/opt/airflow/data/results")
    cleanup_summary = {
        'files_deleted': 0,
        'space_freed_mb': 0,
        'errors': []
    }

    if not results_dir.exists():
        print("📁 Results directory does not exist, skipping cleanup")
        return cleanup_summary

    # Define retention policy: keep files newer than 30 days
    retention_days = 30
    cutoff_timestamp = (datetime.now() - timedelta(days=retention_days)).timestamp()

    try:
        for file_path in results_dir.rglob('*'):
            if file_path.is_file():
                file_age = file_path.stat().st_mtime
                if file_age < cutoff_timestamp:
                    # File is older than retention period
                    file_size = file_path.stat().st_size
                    try:
                        file_path.unlink()  # Delete the file
                        cleanup_summary['files_deleted'] += 1
                        cleanup_summary['space_freed_mb'] += file_size / (1024**2)
                        print(f"🗑️ Deleted old file: {file_path.name}")
                    except Exception as e:
                        error_msg = f"Failed to delete {file_path.name}: {str(e)}"
                        cleanup_summary['errors'].append(error_msg)
                        print(f"❌ {error_msg}")

    except Exception as e:
        cleanup_summary['errors'].append(f"Cleanup error: {str(e)}")
        print(f"❌ Cleanup failed: {str(e)}")

    cleanup_summary['space_freed_mb'] = round(cleanup_summary['space_freed_mb'], 2)

    print(f"✅ Cleanup completed:")
    print(f"   Files deleted: {cleanup_summary['files_deleted']}")
    print(f"   Space freed: {cleanup_summary['space_freed_mb']} MB")

    return cleanup_summary


def generate_health_report(**context):
    """
    Generate comprehensive system health report.

    Returns
    -------
    str
        Health report summary
    """
    print("📋 Generating system health report...")

    # Gather data from all upstream tasks
    freshness_report = context['task_instance'].xcom_pull(
        task_ids='check_data_freshness', key='freshness_report'
    ) or {}

    db_health = context['task_instance'].xcom_pull(
        task_ids='validate_database_connection', key='db_health'
    ) or {}

    quality_report = context['task_instance'].xcom_pull(
        task_ids='check_results_quality', key='quality_report'
    ) or {}

    performance_metrics = context['task_instance'].xcom_pull(
        task_ids='collect_performance_metrics', key='performance_metrics'
    ) or {}

    # Generate health report
    report_lines = []
    report_lines.append("🏥 DOE PIPELINE HEALTH REPORT")
    report_lines.append("=" * 50)
    report_lines.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Database health
    db_status = db_health.get('status', 'unknown')
    status_emoji = "✅" if db_status == 'healthy' else "❌"
    report_lines.append(f"🔗 Database Status: {status_emoji} {db_status}")

    # Data freshness
    fresh_files = sum(1 for info in freshness_report.values() if info.get('is_fresh', False))
    total_files = len(freshness_report)
    report_lines.append(f"📅 Data Freshness: {fresh_files}/{total_files} files fresh")

    # Results quality
    quality_status = quality_report.get('overall_status', 'unknown')
    quality_emoji = {"healthy": "✅", "warning": "⚠️", "failed": "❌", "unknown": "❓"}
    report_lines.append(f"🔍 Results Quality: {quality_emoji.get(quality_status, '❓')} {quality_status}")

    # System performance
    if performance_metrics.get('system'):
        sys_metrics = performance_metrics['system']
        report_lines.append(f"💻 System Performance:")
        report_lines.append(f"   CPU: {sys_metrics.get('cpu_percent', 'N/A')}%")
        report_lines.append(f"   Memory: {sys_metrics.get('memory_percent', 'N/A')}%")
        report_lines.append(f"   Disk: {sys_metrics.get('disk_percent', 'N/A'):.1f}%")

    # Issues summary
    issues = quality_report.get('quality_issues', [])
    if issues:
        report_lines.append("")
        report_lines.append("⚠️ ISSUES DETECTED:")
        for issue in issues[:5]:  # Show first 5 issues
            report_lines.append(f"   - {issue}")
        if len(issues) > 5:
            report_lines.append(f"   ... and {len(issues) - 5} more issues")

    # Overall status
    report_lines.append("")
    if db_status == 'healthy' and quality_status == 'healthy':
        report_lines.append("🎉 OVERALL STATUS: HEALTHY")
    elif any(status in ['warning', 'failed'] for status in [db_status, quality_status]):
        report_lines.append("⚠️ OVERALL STATUS: NEEDS ATTENTION")
    else:
        report_lines.append("❌ OVERALL STATUS: UNHEALTHY")

    health_report = "\n".join(report_lines)
    print(health_report)

    return health_report


# DAG configuration
default_args = {
    'owner': 'doe-pipeline',
    'depends_on_past': False,
    'start_date': datetime(2025, 9, 25),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'execution_timeout': timedelta(minutes=30),
}

# Create maintenance DAG
maintenance_dag = DAG(
    'doe_pipeline_maintenance',
    default_args=default_args,
    description='Maintenance and monitoring for DOE analysis pipeline',
    schedule_interval='0 */6 * * *',  # Every 6 hours
    start_date=datetime(2025, 9, 25),
    catchup=False,
    max_active_runs=1,
    tags=['maintenance', 'monitoring', 'doe', 'health-check'],
)

# Task definitions
start_maintenance = DummyOperator(
    task_id='start_maintenance',
    dag=maintenance_dag,
)

# Health check task group
with TaskGroup('health_checks', dag=maintenance_dag) as health_group:

    check_data_freshness_task = PythonOperator(
        task_id='check_data_freshness',
        python_callable=check_data_freshness,
        dag=maintenance_dag,
    )

    validate_db_task = PythonOperator(
        task_id='validate_database_connection',
        python_callable=validate_database_connection,
        dag=maintenance_dag,
    )

    check_results_quality_task = PythonOperator(
        task_id='check_results_quality',
        python_callable=check_results_quality,
        dag=maintenance_dag,
    )

    collect_performance_task = PythonOperator(
        task_id='collect_performance_metrics',
        python_callable=collect_performance_metrics,
        dag=maintenance_dag,
    )

# Maintenance task group
with TaskGroup('maintenance_tasks', dag=maintenance_dag) as maintenance_group:

    cleanup_task = PythonOperator(
        task_id='cleanup_old_results',
        python_callable=cleanup_old_results,
        dag=maintenance_dag,
    )

# Reporting task
generate_health_report_task = PythonOperator(
    task_id='generate_health_report',
    python_callable=generate_health_report,
    dag=maintenance_dag,
)

end_maintenance = DummyOperator(
    task_id='end_maintenance',
    dag=maintenance_dag,
)

# Define task dependencies
start_maintenance >> health_group >> maintenance_group >> generate_health_report_task >> end_maintenance