import datetime as dt
from airflow import DAG
from airflow.operators.bash_operator import BashOperator


# Following are defaults wich can be overridden later on
args = {
    "owner": "Andrey",
    "start_date": dt.datetime(2018, 11, 1),
    "provide_context": True,
}

with DAG(
    "Hello-world",  # название дага
    description="Hello-world",  # описание дага
    schedule_interval="*/1 * * * *",  # интервал времени, через который даг будет повторяться (в данном случае одна минута)
    catchup=False,  # если бы не было, то AirFlow пфтался бы "нагнать" и запускал бы все даги (которые не были выполнены), начиная с start_date в args по текущий момент
                    # как бы мы говорим, что не надо глядень в прошлое, а реализую последующие
    default_args=args) as dag:  # 0 * * * *    */1 * * * *

    t1 = BashOperator(
        task_id="task_1",
        bash_command="echo 'Hello World from Task 1'"
    )

    t2 = BashOperator(
        task_id="task_2",
        bash_command="echo 'Hello World from Task 2'"
    )

    t3 = BashOperator(
        task_id="task_3",
        bash_command="echo 'Hello World from Task 3'"
    )

    t4 = BashOperator(
        task_id="task_4",
        bash_command="echo 'Hello World from Task 4'"
    )

    t1 >> t2
    t1 >> t3
    t2 >> t4
    t3 >> t4
