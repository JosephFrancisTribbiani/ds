from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator


with DAG(
    'titanic',
    description='pipeline to predict the survival of Titanic passengers',
    start_date=days_ago(0, 0, 0, 0, 0)
) as dag:
    task_1 = BashOperator(
        task_id='print-date',
        bash_command='date'
    )

    task_2 = BashOperator(
        task_id='greeting',
        bash_command='echo "hello"'
    )

    task_1 >> task_2
