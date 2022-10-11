Airflow развернут в Docker контейнере по <a href="https://medium.com/@technologIT/apache-airflow-%D1%87%D0%B0%D1%81%D1%82%D1%8C-4-%D1%83%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%BA%D0%B0-9fa5cb9d0e06">инструкции</a> (с учетом установки обновленной версии airflow 2.4.1)

Для запуска контейнера необходимо выполнить последовательно следующие команды:
- Создание образа с помощью Docker файла
```bash
docker build -t airflow-basic .
```
Наш созданный образ будет называться **airflow-basic**
- Запуск Docker контейнера
```bash
docker run --rm --mount src="C:\Users\alpex\HSE_study\Git\ds\Airflow_example\dags",dst="//usr/local/airflow/dags",type=bind -d -p 8080:8080 airflow-basic
```
```--rm``` означает, что как только контейнер будет остановлен, он будет автоматически удален
```--mount``` означает, что мы привязываем папку в контейнере к папке на локальном компьютере и все изменения в одной из этих папок автоматически будут отображены в другой
- Для доступа к терминалу контейнера необходимо ввести
```bash
docker exec -it CONTAINER_ID /bin/bash
```
CONTAINER_ID можно посмотреть с помощью команды
```bash
docker ps
```
Данная команда выдаст нечто подобное:
```bash
CONTAINER ID   IMAGE           COMMAND            CREATED          STATUS          PORTS                    NAMES
598b12a402be   airflow-basic   "/entrypoint.sh"   21 seconds ago   Up 20 seconds   0.0.0.0:8080->8080/tcp   determined_beaver
```
где CONTAINER_ID - 598b12a402be
- Теперь можно перейти в браузере по адресу http://localhost:8080

Для выхода из терминала необходимо нажать ```CTRL+D```.
Основные команды из туториала представлены в папке docs
