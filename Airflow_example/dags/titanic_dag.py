from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator


with DAG(
    'titanic',
    description='pipeline to predict the survival of Titanic passengers',
    start_date=days_ago(0, 0, 0, 0, 0)
) as dag:
    pass
