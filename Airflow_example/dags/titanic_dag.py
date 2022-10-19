from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator


with DAG(
    'titanic',
    description='pipeline to predict the survival of Titanic passengers',
    start_date=days_ago(0, 0, 0, 0, 0)
) as dag:

    def callable_load_data(table: str):
        import logging
        from titanic.db import load_data

        LOGGER = logging.getLogger("airflow.load_data")
        LOGGER.info("airflow.load_data >>> airflow.prepare_sets - INFO loading data from {}".format(table))

        loaded_data = load_data(table=table)
        return

    def prepare_sets():
        print("Hello")
        return

    task_load_features = PythonOperator(
        dag=dag,
        task_id="load_features",
        python_callable=callable_load_data,
        op_kwargs={"table": "features"}
    )

    task_load_targets = PythonOperator(
        dag=dag,
        task_id="load_targets",
        python_callable=callable_load_data,
        op_kwargs={"table": "targets"}
    )

    task_prepare_sets = PythonOperator(
        dag=dag,
        task_id="prepare_sets",
        python_callable=prepare_sets,
    )

    task_load_features >> task_prepare_sets
    task_load_targets >> task_prepare_sets
