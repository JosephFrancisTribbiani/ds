import psycopg2
import pandas as pd
import numpy as np
import psycopg2.extensions as pe
from functools import wraps
from pathlib import Path
from . import read_yaml


def _connect(func):

    @wraps(func)
    def inner(*args, **kwargs):

        # считываем конфигурационный файл
        params = read_yaml(file_path=Path(__file__).parent.resolve() / "config.yaml").get("db")

        # подключаемся к базе
        with psycopg2.connect(**params) as conn, conn.cursor() as cur:
            res = func(*args, conn=conn, cur=cur, **kwargs)
        return res

    return inner


@_connect
def init_db(conn, cur, force: bool = False) -> None:
    if force:
        cur.execute("DROP TABLE IF EXISTS features, target;")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS features (
        PassengerId INTEGER PRIMARY KEY,
        Pclass INTEGER,
        Name TEXT,
        Sex TEXT,
        Age INTEGER,
        SibSp INTEGER,
        Parch INTEGER,
        Ticket TEXT,
        Fare DECIMAL,
        Cabin TEXT,
        Embarked CHAR);""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS target (
        PassengerId INTEGER PRIMARY KEY,
        Survived INTEGER NOT NULL
    );""")

    conn.commit()
    return


def data_to_postgres(loc: str = "../input_data", force: bool = True) -> None:
    abs_loc = Path(__file__).parent.resolve() / loc

    # create table
    init_db(force=force)

    # adapt numpy dtypes
    pe.register_adapter(np.int64, _adapt_int64)
    pe.register_adapter(float, _nan_to_null)

    # train data to postgres
    train_df = pd.read_csv(abs_loc / "./train.csv", sep=',')
    train_x, train_y = train_df.drop(["Survived"], axis="columns"), train_df[["PassengerId", "Survived"]]
    append(data=train_x, table="features")
    append(data=train_y, table="target")

    # test data to postgres
    test_x = pd.read_csv(abs_loc / "./test.csv", sep=',')
    append(data=test_x, table="features")
    return


def append(data: pd.DataFrame, table: str, bin_size: int = 16) -> None:
    # разбиваем на бины
    n_bins = data.shape[0] // bin_size + (1 if data.shape[0] % bin_size else 0)
    bins = np.array_split(data, n_bins)

    # итерируемся по бинам и загружаем из в postgres
    for single_bin_data in bins:
        _append(data=single_bin_data, table=table)
    return


@_connect
def _append(data: pd.DataFrame, table: str, conn, cur) -> None:
    cols = ", ".join(data.columns)
    tuples = [tuple(row) for row in data.to_numpy()]
    args_str = ", ".join(["%s"]*data.shape[1])

    query = """
    INSERT INTO {} ({})
    VALUES ({}) ON CONFLICT DO NOTHING;
    """.format(table, cols, args_str)

    try:
        cur.executemany(query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: {}".format(error))
        conn.rollback()
    return


def _adapt_int64(val: np.int64):
    return pe.AsIs(val)


def _nan_to_null(f, _null=pe.AsIs('NULL'), _float=pe.Float):
    if f != f:
        return _null
    return _float(f)


def get_data(table: str) -> pd.DataFrame:
    """
    Функция загрузки данных из базы Postgres.
    :param table: название таблицы в Postgres.
    :return: датафрейм с данными.
    """
    # считываем названия колонок
    cols = get_columns_names(table=table)

    # формируем текст запроса
    query = """
    SELECT {} FROM {}
    LIMIT 10;
    """.format(", ".join(cols), table)

    # загружаем данные из Postgres
    data = read_sql(query=query)
    return pd.DataFrame(columns=cols, data=data)


def get_columns_names(table: str) -> list:
    query = """
    SELECT column_name FROM information_schema.columns
    WHERE table_name = '{}'
    ORDER BY ordinal_position ASC;
    """.format(table)

    cols = read_sql(query=query)
    cols = [i for i, in cols]
    return cols


@_connect
def read_sql(query: str, conn, cur):
    try:
        cur.execute(query)
        data = cur.fetchall()
        return data
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: {}".format(error))
    return
