from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator


with DAG(
    'titanic',
    description='pipeline to predict the survival of Titanic passengers',
    start_date=days_ago(0, 0, 0, 0, 0)
) as dag:

    def load_features():
        from titanic.db import load_data
        return load_data(table="features")

    def load_targets():
        from titanic.db import load_data
        return load_data(table="target")

    def prepare_sets():
        return

    task_load_features = PythonOperator(
        dag=dag,
        task_id="load_features",
        python_callable=load_features,
    )

    task_load_targets = PythonOperator(
        dag=dag,
        task_id="load_targets",
        python_callable=load_targets,
    )

    task_prepare_sets = PythonOperator(
        dag=dag,
        task_id="prepare_sets",
        python_callable=prepare_sets,
    )

    task_load_features >> task_prepare_sets
    task_load_targets >> task_prepare_sets
