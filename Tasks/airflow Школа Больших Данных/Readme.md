Выполнялось на базе серии видео
1. [Начало работы с apache airflow](https://www.youtube.com/watch?v=G6ipydgZRnE)
2. [ETL на airflow](https://www.youtube.com/watch?v=XFQ0KPaDIT8&t)
3. [ETL на airflow c postgresql](https://www.youtube.com/watch?v=55D9Eu7mUW0)

# Установка и запуск AirFlow

Запускать AirFlow будем в локальном режиме (LocalExecutor). [Ссылка](https://github.com/puckel/docker-airflow) на стабильный образ.

Из корня проекта клонируем репозиторий:

```bash
git clone https://github.com/puckel/docker-airflow.git
```

Появится папка `docker-airflow`. В папке `./docker-airflow/dags/` находятся все наши DAGs.

Запускаем AirFlow в режиме LocalExecuter из папки `docker-airflow` с помощью команды:

```bash
docker-compose -f docker-compose-LocalExecutor.yml up -d
```

Появятся два контейнера (посмотреть можно через `docker ps`) - `postgres` и `websever`.

Для запуска в режиме CeleryExecutor:

```bash
docker-compose -f docker-compose-CeleryExecutor.yml up -d
```

В браузере на `localhost:8080` (указано в конфигурационном файле `.yml`) смотрим, что же у нас поднялось.

Остановить все контейнеры можно с помощью команды:

```bash
docker stop $(docker ps -a -q)
```

# Виды executers

1. `SequentialExecutor` - самый простой, используется по умолчанию. Одновременно работает только с одной задачей. На практике не используется, а только для тестировки.
2. `LocalExecutor` - может выполнять задачи параллельно. Но все равно на практике не используется, т.к. нет страховки от сбоев. К примеру, если будет запущено много задач, и какая-то задача упала, то он ничего не вернет. Можно использовать, если кол-во задач небольшое и не хочется достраивать сервисы.
3. `CeleryExecutor` - более продуктовый вариант на базе библиотеки Celery, написанной на языке python для управления распределенной очередью заданием. Удобно масштабировать. Можно создавать worker-ы и celery распределяет их на кластере. Если задачка на каком-то worker-е упадет, то она будет выполнена на другом worker-е (отказоустойчивость).
4. `DaskExecutor`
5. `KubernetesExecutor`

# Xcom

Позволяет передавать данные между тасками в даге. Допустим, Task 1 через какую-то API выгрузил какие-то данные, и далее, чтобы они были доступны остальным таскам, эти данные пушатся в базу серез manager Xcom.

# Использование переменных окружения

В AirFlow можно хранить переменные окружения. Добавить можно в GUI AirFlow через Admin > Variables. Получить доступ к ним возможно с помощью следующей команды:

```python
from airflow.models import Variable


var_name = Variable.get("Имя переменной в Gui AirFlow")
```

# Посмотреть IP сервиса, развернутого в докере

Сначала ищем ID контейнера

```bash
docker ps
```

Далее для выбранного контейнера смотрим IP, выделенного в подсети

```bash
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' id_контейнера
```

Например

```bash
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' c98b8c89b4dd
```

# Прописать настройки подключения к PostgreSQL

Через Gui AirFlow admin > Connections указывает подключение к PostgreSQL базе. В нашем примере мы использовали название подключения database_PG.