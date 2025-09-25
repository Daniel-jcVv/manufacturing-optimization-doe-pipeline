"""
Test DAG to verify Airflow functionality
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

def test_python_task():
    """Simple Python task for testing"""
    print("✅ Python task executed successfully!")
    return "success"

def test_database_connection():
    """Test database connectivity from Airflow"""
    import sys
    sys.path.append('/opt/airflow/src')
    from utils.database import test_connecton
    result = test_connecton()
    print(f"✅ Database connection test: {result}")
    return result

default_args = {
    'owner': 'doe-pipeline',
    'depends_on_past': False,
    'start_date': datetime(2025, 9, 24),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'test_infrastructure',
    default_args=default_args,
    description='Test basic Airflow functionality',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['test', 'infrastructure'],
)

# Task 1: Simple bash test
test_bash = BashOperator(
    task_id='test_bash_task',
    bash_command='echo "✅ Bash task executed successfully!"',
    dag=dag,
)

# Task 2: Python test
test_python = PythonOperator(
    task_id='test_python_task',
    python_callable=test_python_task,
    dag=dag,
)

# Task 3: Database connection test
test_db = PythonOperator(
    task_id='test_database_connection',
    python_callable=test_database_connection,
    dag=dag,
)

# Define task dependencies
test_bash >> test_python >> test_db