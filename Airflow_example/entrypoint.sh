#!/usr/bin/env bash

# Initiliase the metastore
airflow db init

# create admin user
airflow users create --role Admin --username admin --email admin --firstname admin --lastname admin --password admin

# Run the scheduler in background
airflow scheduler &> /dev/null &

# Run the web server in foreground (for docker logs)
exec airflow webserver
