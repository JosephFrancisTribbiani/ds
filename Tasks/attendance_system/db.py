import psycopg2
import argparse
import numpy as np
from functools import wraps
from config import DATABASE_URL
from typing import Tuple, List


def get_connection(func):
    """
    Декоратор для создания подключения к базе и гарантированного его закрытия
    :param func: функция, которой требуется подключиться к базе
    :return:
    """

    @wraps(func)
    def inner(*args, **kwargs):
        with psycopg2.connect(DATABASE_URL) as conn, conn.cursor() as cur:
            res = func(*args, conn=conn, cur=cur, **kwargs)
        return res

    return inner


@get_connection
def init_db(conn, cur, force: bool = False, hidden_size: int = 128) -> None:
    """
    Функция формирует систему таблиц базы данных
    :param conn: connection к базе
    :param cur: курсор подключения к базе
    :param force: True если необходимо пересоздать базу данных (все данные будут удалены)
    :return: nothing
    """

    if force:
        cur.execute('DROP TABLE IF EXISTS encodings, students CASCADE;')

    query_students = """
    CREATE TABLE IF NOT EXISTS students
    (
        id INTEGER PRIMARY KEY,
        firstname TEXT NOT NULL,
        secondname TEXT NOT NULL,
        standing CHAR NOT NULL,
        major TEXT NOT NULL,
        starting_year INTEGER NOT NULL,
        writetime TIMESTAMP(0) WITH TIME ZONE NOT NULL DEFAULT now()
    );
    
    COMMENT ON TABLE students
    IS 'Таблица с метаданными студентов';

    COMMENT ON COLUMN students.id
    IS 'Уникальный ID студента';

    COMMENT ON COLUMN students.firstname
    IS 'Имя';

    COMMENT ON COLUMN students.secondname
    IS 'Фамилия';

    COMMENT ON COLUMN students.standing
    IS 'Статус';

    COMMENT ON COLUMN students.major
    IS 'Специализация';

    COMMENT ON COLUMN students.starting_year
    IS 'Год начала обучения';
    
    COMMENT ON COLUMN students.writetime
    IS 'Дата и время записи';"""
    cur.execute(query_students)

    sub_query = ",\n\t".join(map(lambda idx: '"%s" DOUBLE PRECISION NOT NULL' % idx, range(1, 1 + hidden_size)))
    query_embeddings = """
    CREATE TABLE IF NOT EXISTS embeddings
    (   
        emb_id SERIAL,
        student_id INTEGER NOT NULL REFERENCES students (id),
        writetime TIMESTAMP(0) WITH TIME ZONE NOT NULL DEFAULT now(),
        %s
    );

    COMMENT ON TABLE embeddings
    IS 'Таблица с эмбеддингами лиц студентов';
    
    COMMENT ON COLUMN embeddings.emb_id
    IS 'Уникальный ID эмбеддинга';

    COMMENT ON COLUMN embeddings.student_id
    IS 'Уникальный ID студента';

    COMMENT ON COLUMN embeddings.writetime
    IS 'Дата и время записи эмбеддинга (с timezone)';
    """ % sub_query
    cur.execute(query_embeddings)

    query_attendance = """
    CREATE TABLE IF NOT EXISTS attendace
    (
        student_id INTEGER REFERENCES students (id),
        attendance_datetime TIMESTAMP(0) WITH TIME ZONE NOT NULL DEFAULT now()
    );

    COMMENT ON TABLE attendace
    IS 'Таблица с информацией о посещениях';

    COMMENT ON COLUMN attendace.student_id
    IS 'Уникальный ID студента';

    COMMENT ON COLUMN attendace.attendance_datetime
    IS 'Дата и время посещения (с timezone)';
    """
    cur.execute(query_attendance)
    
    conn.commit()
    return


@get_connection
def check_student(id: int, conn, cur) -> bool:
    """
    Функция проверки существования уникального ID студента в DB
    :param id: ID студента, который необходимо проверить
    """
    query = """
    SELECT EXISTS (SELECT 1 FROM students WHERE id = %s);
    """ % id
    cur.execute(query)
    return cur.fetchone()[0]


@get_connection
def get_next_id(conn, cur) -> int:
    """
    Функция для получения следующего свободного ID студента
    """
    query = """
    SELECT max(id) FROM students;
    """
    cur.execute(query)
    max_id = cur.fetchone()[0]
    if max_id is None:
        return 0
    return max_id + 1


@get_connection
def save_student(id: int, firstname: str, secondname: str, standing: str, 
                 major: str, starting_year: int, conn, cur) -> None:
    query = """
    INSERT INTO students (id, firstname, secondname, standing, major, starting_year)
    VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING;
    """
    cur.execute(query, (id, firstname, secondname, standing, major, starting_year))
    conn.commit()
    return


@get_connection
def save_embedding(id: int, emb: np.ndarray, conn, cur) -> None:
    """
    Функция сохранения эмбеддинга в БД
    :param id: уникальный ID студента
    :param emb: эмбеддинг лица
    """
    query = """
    INSERT INTO embeddings (student_id, "{}")
    VALUES (%s, {})
    """.format('", "'.join(map(str, range(1, len(emb) + 1))), ", ".join(["%s"]*len(emb)))
    cur.execute(query, (id, *emb))
    conn.commit()
    return


@get_connection
def read_embeddings(conn, cur, hidden_size: int = 128) -> Tuple[dict, np.ndarray]:
    metacols = ["id", "firstname", "secondname", "standing", "major", "starting_year"]

    query = """
    SELECT 
        %s,
        %s
    FROM students AS st
    LEFT JOIN
    (
        SELECT DISTINCT ON (student_id) * FROM embeddings
        ORDER BY student_id, writetime DESC
    ) AS emb
    ON st.id = emb.student_id
    """ % (
        ",\n    ".join(map(lambda c: 'st.%s' % c, metacols)),
        ",\n    ".join(map(lambda c: 'emb."%s"' % (c + 1),range(hidden_size)))
        )
    
    cur.execute(query)
    data = cur.fetchall()
    metadata = [dict(zip(metacols, d[:len(metacols)])) for d in data]
    embeddings = [np.array(list(d[len(metacols):]), dtype="float64") for d in data]
    return metadata, embeddings


def main(args) -> None:
    init_db(force=args.force, hidden_size=args.hidden)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Создание таблиц в БД.")
    parser.add_argument("--hidden", "-H", type=int, default=128, help="Размер эмбеддинга лица.")
    parser.add_argument("-force", "-F", action="store_true", help="Если флаг стоит, то таблицы пересоздаются")
    args = parser.parse_args()

    main(args)
